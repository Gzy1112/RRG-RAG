"""Inspector GRPO v2: multi-action recovery Inspector trained on the val split.

Differences from trainer_vidorag_controller_grpo.py (Inspector v1):

  * Data. Records come from the official SlideVQA val split (deck-disjoint
    from train and test) via build_inspector_branch_cache.py, which ran the
    DEPLOYED pipeline (GMM pool, TFR seeker/synthesizer, ColQwen2.5 ranking,
    rank-1 rescue) once per recovery action and judged every branch. Only the
    'train' deck split is used here; the 'holdout' decks are kept for
    checkpoint selection (eval_inspector_v2_offline.py).
  * Actions. INSPECTOR_V2_ACTIONS (default STOP,DECK,EXPAND), each mapped to a
    cached branch (INSPECTOR_V2_BRANCH_MAP).
  * Sampling. Questions are drawn from their natural distribution
    (INSPECTOR_V2_INFORMATIVE_WEIGHT optionally up-weights questions whose
    branch scores differ). No forced flips: every rollout is on-policy, and a
    group whose rollouts all pick the same action carries no gradient.
  * Prompt. inspector_v2_prompt.py (project root) is the single source of
    truth for training, offline selection and closed-loop inference.

Reward per rollout:
    parse failure   -> CONTROLLER_GRPO_FORMAT_PENALTY (default -0.5)
    action a        -> (score[branch(a)] - 1) / 4 - INSPECTOR_V2_COST * [a != STOP]
    KL anchor       -> GRPO_KL_BETA * KL(pi || pi_ref) subtracted before the
                       group advantage (computed once per distinct response).

Activated via VIDORAG_MAPPO_TRAINER=inspector_grpo_v2 in
workflow_restore_grpo.py. Checkpoints hold only the adapter, under
checkpoint-N/controller/ (same layout as v1).
"""

import collections
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from .trainer_vidorag_controller_grpo import ViDoRAGControllerGRPOTrainer, _PROJECT_ROOT
from .trainer_vidorag_restore_grpo import _grpo_group_advantages, _GRPO_KL_BETA

logger = logging.getLogger(__name__)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import inspector_v2_prompt as iv2  # noqa: E402


def _parse_branch_map(spec: str) -> Dict[str, str]:
    out = {}
    for item in spec.split(","):
        if item.strip():
            action, branch = item.split(":")
            out[action.strip().upper()] = branch.strip()
    return out


class ViDoRAGInspectorGRPOV2Trainer(ViDoRAGControllerGRPOTrainer):
    """GRPO for the multi-action Inspector against cached branch outcomes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v2_actions = iv2.parse_actions(
            os.environ.get("INSPECTOR_V2_ACTIONS", "STOP,DECK,EXPAND"))
        self._v2_branch_map = _parse_branch_map(os.environ.get(
            "INSPECTOR_V2_BRANCH_MAP", "STOP:stop,DECK:deck,EXPAND:expand"))
        missing = [a for a in self._v2_actions if a not in self._v2_branch_map]
        if missing:
            raise ValueError(f"INSPECTOR_V2_BRANCH_MAP has no branch for {missing}")
        self._v2_cost = float(os.environ.get("INSPECTOR_V2_COST", "0.02"))
        self._v2_informative_weight = float(
            os.environ.get("INSPECTOR_V2_INFORMATIVE_WEIGHT", "1.0"))
        self._v2_split = os.environ.get("INSPECTOR_V2_SPLIT", "train")
        self._v2_img_dir = os.path.join(_PROJECT_ROOT, "data/SlideVQA/img")
        logger.info(
            f"InspectorGRPOv2 init: actions={self._v2_actions} branches={self._v2_branch_map} "
            f"cost={self._v2_cost} informative_w={self._v2_informative_weight} "
            f"split={self._v2_split}")

    # ------------------------------------------------------------------
    # Training data: val branch cache (+ ColQwen2.5 ranking), train decks only
    # ------------------------------------------------------------------

    def _load_v2_records(self) -> None:
        rank_path = os.environ.get(
            "INSPECTOR_V2_RANK_CACHE",
            os.path.join(_PROJECT_ROOT, "outputs/rank_cache/SlideVQA_val.jsonl"))
        rankings: Dict[str, List[str]] = {}
        with open(rank_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("ranking"):
                        rankings.setdefault(rec["uid"], rec["ranking"])

        cache_paths = [p for p in os.environ.get("INSPECTOR_V2_BRANCH_CACHE", "").split(",") if p]
        if not cache_paths:
            raise FileNotFoundError("INSPECTOR_V2_BRANCH_CACHE is not set")
        records, weights, seen = [], [], set()
        skipped = collections.Counter()
        for path in cache_paths:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    uid = rec["uid"]
                    if uid in seen:
                        continue
                    seen.add(uid)
                    if iv2.val_split_of(rec["meta_info"]["file_name"]) != self._v2_split:
                        skipped["other_split"] += 1
                        continue
                    if uid not in rankings:
                        skipped["no_ranking"] += 1
                        continue
                    scores = {}
                    for a in self._v2_actions:
                        br = rec["branches"].get(self._v2_branch_map[a]) or {}
                        if br.get("score") is not None:
                            scores[a] = float(br["score"])
                    if len(scores) != len(self._v2_actions):
                        skipped["missing_branch"] += 1
                        continue
                    if not rec.get("e0"):
                        skipped["empty_e0"] += 1
                        continue
                    informative = len(set(scores.values())) > 1
                    records.append({
                        "uid": uid,
                        "query": rec["query"],
                        "selected": rec["e0"],
                        "ranking": rankings[uid],
                        "scores": scores,
                        "best": [a for a in self._v2_actions if scores[a] == max(scores.values())],
                        "informative": informative,
                    })
                    weights.append(self._v2_informative_weight if informative else 1.0)
        if not records:
            raise RuntimeError(f"InspectorGRPOv2: no training records (skipped={dict(skipped)})")
        self._ctrl_records, self._ctrl_weights = records, weights
        n_inf = sum(r["informative"] for r in records)
        mean = {a: sum(r["scores"][a] for r in records) / len(records) for a in self._v2_actions}
        oracle = sum(max(r["scores"].values()) for r in records) / len(records)
        logger.info(
            f"InspectorGRPOv2: {len(records)} records ({n_inf} informative), skipped={dict(skipped)}; "
            f"mean score per action={ {a: round(v, 4) for a, v in mean.items()} } oracle={oracle:.4f}")

    # ------------------------------------------------------------------
    # Prompt / rollouts / reward
    # ------------------------------------------------------------------

    def _build_controller_prompt(self, record: Dict[str, Any]
                                 ) -> Tuple[List[Dict[str, str]], List[str]]:
        ids, paths = iv2.shown_pages(record["selected"], self._v2_img_dir)
        instruction = iv2.build_instruction(record["query"], ids, record["ranking"],
                                            self._v2_actions)
        vision_tok = "<|vision_start|><|image_pad|><|vision_end|>"
        messages = [
            {"role": "system", "content": iv2.SYSTEM},
            {"role": "user", "content": vision_tok.join(instruction.split("<image>"))},
        ]
        return messages, paths

    def _rollout_group(self, record: Dict[str, Any], step: int = 0) -> List[Dict[str, Any]]:
        """G on-policy samples for one prompt (no forced flips)."""
        G = self._ctrl_group_size
        messages, image_paths = self._build_controller_prompt(record)
        batch = self.trans_vlm_messages_to_inputs([messages] * G, [image_paths] * G)
        self.generation_config.do_sample = True
        self.generation_config.temperature = self._ctrl_temperature
        self.generation_config.top_p = float(os.environ.get("CONTROLLER_GRPO_TOP_P", "1.0"))
        _queries, responses = self.get_vlm_inputs(
            batch, logits_processor=None, max_new_tokens=self._ctrl_max_new_tokens)
        out = []
        for r in responses:
            text = self.tokenizer.decode(r, skip_special_tokens=True)
            action = iv2.parse_decision(text, self._v2_actions)
            out.append({"response": r, "action": action, "parse_ok": action is not None,
                        "text": text})
        if self._ctrl_debug_rollouts < int(os.environ.get("CONTROLLER_GRPO_DEBUG_ROLLOUTS", "0")):
            self._ctrl_debug_rollouts += 1
            logger.info(f"[InspectorGRPOv2 rollout step={step}] uid={record['uid']} "
                        f"scores={record['scores']} texts="
                        + " || ".join(repr(o["text"][:60]) for o in out))
        return out

    def _v2_reward(self, record: Dict[str, Any], action: Optional[str]) -> float:
        if action is None:
            return self._ctrl_format_penalty
        r = (record["scores"][action] - 1.0) / 4.0
        if action != "STOP":
            r -= self._v2_cost
        return r

    def _v2_kl(self, samples: List[Dict[str, Any]], group_ids: List[int]) -> List[float]:
        """KL penalty per rollout, one forward pair per distinct (prompt, response)."""
        if _GRPO_KL_BETA <= 0:
            return [0.0] * len(samples)
        memo: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        out = []
        for s, g in zip(samples, group_ids):
            key = (g, tuple(s["response"].tolist()))
            if key not in memo:
                memo[key] = self._compute_kl_penalties([s])[0]
            out.append(memo[key])
        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        self._load_controller_adapter()
        self._load_v2_records()

        max_steps = self.args.max_steps if self.args.max_steps > 0 else 100
        max_steps = int(os.environ.get("CONTROLLER_GRPO_MAX_STEPS", str(max_steps)))
        logger.info(f"InspectorGRPOv2: starting {max_steps} steps, "
                    f"{self._ctrl_batch_questions} questions/step, G={self._ctrl_group_size}, "
                    f"T={self._ctrl_temperature}, kl_beta={_GRPO_KL_BETA}")

        window = collections.deque(maxlen=self._ctrl_log_every)
        for step in range(1, max_steps + 1):
            self.state.global_step = step
            step_records = self._ctrl_rng.choices(
                self._ctrl_records, weights=self._ctrl_weights, k=self._ctrl_batch_questions)

            samples, rewards, group_ids, step_log = [], [], [], []
            self.model.eval()
            self._ctrl_debug_rollouts = 0
            for g_idx, record in enumerate(step_records):
                messages, image_paths = self._build_controller_prompt(record)
                prompt_batch = self.trans_vlm_messages_to_inputs([messages], [image_paths])
                q_ids = prompt_batch["input_ids"][0].cpu()
                q_ids = q_ids[(q_ids != self.tokenizer.pad_token_id).nonzero()[0].item():]
                for o in self._rollout_group(record, step=step):
                    samples.append({"query": q_ids, "response": o["response"],
                                    "batch_dict": prompt_batch})
                    rewards.append(self._v2_reward(record, o["action"]))
                    group_ids.append(g_idx)
                    best = max(record["scores"].values())
                    step_log.append({
                        "action": o["action"],
                        "score": record["scores"][o["action"]] if o["action"] else None,
                        "regret": (best - record["scores"][o["action"]]) if o["action"] else None,
                        "informative": record["informative"],
                    })

            kls = self._v2_kl(samples, group_ids)
            rw = torch.tensor([r - _GRPO_KL_BETA * k for r, k in zip(rewards, kls)],
                              dtype=torch.float32)
            adv = _grpo_group_advantages(rw, group_ids)
            n_active = int((adv != 0).sum().item())
            stats = self._grpo_step(self._ctrl_collect_records(samples), adv.tolist())
            window.append((step_log, rewards, kls, n_active, len(samples), stats))

            if step % self._ctrl_log_every == 0 or step == 1 or step == max_steps:
                logs = [x for w in window for x in w[0]]
                n = max(len(logs), 1)
                dist = collections.Counter(x["action"] or "PARSE_FAIL" for x in logs)
                parsed = [x for x in logs if x["action"]]
                mean_score = sum(x["score"] for x in parsed) / max(len(parsed), 1)
                mean_regret = sum(x["regret"] for x in parsed) / max(len(parsed), 1)
                inf = [x for x in parsed if x["informative"]]
                inf_regret = sum(x["regret"] for x in inf) / max(len(inf), 1)
                all_r = [r for w in window for r in w[1]]
                all_k = [k for w in window for k in w[2]]
                active = sum(w[3] for w in window) / max(sum(w[4] for w in window), 1)
                msg = (f"[InspectorGRPOv2 step {step}/{max_steps}] "
                       f"actions={ {a: round(c / n, 3) for a, c in sorted(dist.items())} } "
                       f"mean_score={mean_score:.4f} regret={mean_regret:.4f} "
                       f"inf_regret={inf_regret:.4f} mean_r={sum(all_r) / max(len(all_r), 1):.4f} "
                       f"kl={sum(all_k) / max(len(all_k), 1):.4f} active_frac={active:.3f} "
                       f"loss={stats.get('grpo/loss', 0.0):.4f} "
                       f"grad_norm={stats.get('grpo/grad_norm', 0.0):.4f}")
                logger.info(msg)
                print(msg, flush=True)

            if self._ctrl_save_every > 0 and step % self._ctrl_save_every == 0:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
                self.save_model(ckpt_dir)
                logger.info(f"InspectorGRPOv2: saved checkpoint {ckpt_dir}")

        logger.info("[InspectorGRPOv2] training loop finished")
