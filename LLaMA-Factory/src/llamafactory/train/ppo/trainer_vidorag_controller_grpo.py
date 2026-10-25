"""Task 70: GRPO trainer for the STOP/RETRIEVE_MORE Inspector controller.

Trains a binary decision controller (Qwen2.5-VL-3B + one LoRA adapter) with
GRPO against an OFFLINE reward cache:

    Key insight: all frozen downstream agents (TFR seeker / synthesizer) decode
    greedily, so the judge score of each controller branch is a deterministic
    function of the decision. ``build_controller_reward_cache.py`` precomputed
    both branch scores per training question; training therefore only needs short
    controller rollouts (~15 tokens) plus a cache lookup -- no online
    closed-loop inference.

Reward shape (per rollout):
    parse failure       -> CONTROLLER_GRPO_FORMAT_PENALTY (default -0.5)
    r(action)           -> (cached_score[action] - 1) / 4
                           - lambda * [action == RETRIEVE_MORE]
    lambda              -> CONTROLLER_GRPO_LAMBDA (default 0.03), a small cost
                           on retrieval so ties break towards STOP.
    KL anchor           -> GRPO_KL_BETA * KL(pi || pi_ref) subtracted when
                           GRPO_KL_BETA > 0 (pi_ref = kl_ref_0 adapter).

Prompt: byte-identical to build_controller_sft_data_v3.CONTROLLER_TEMPLATE_V3
(question + round-1 selected page images, <=8, + [Retrieval Confidence]
fusion-rank block). The template and confidence block are imported from the
project-root builder module (single source of truth); a local fallback exists
only so this module stays importable outside the project tree, and the launch
script asserts byte-identity either way.

Sampling: each step draws BATCH questions (default 8, weighted: oracle
RETRIEVE_MORE questions x CONTROLLER_GRPO_POS_WEIGHT) and rolls out G
responses per prompt (MAPPO_GRPO_GROUP_SIZE). Stratified fix-up: if all G
rollouts decode to the SAME action, the last rollout is replaced by the
tokenized opposite decision so every group carries gradient signal.

Activated via VIDORAG_MAPPO_TRAINER=controller_grpo in
workflow_restore_grpo.py. Checkpoints save ONLY the controller adapter under
checkpoint-N/controller/.
"""

import json
import os
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .trainer_vidorag_restore_grpo import (
    ViDoRAGVLMPPOTrainer,
    _grpo_group_advantages,
    _GRPO_KL_BETA,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template: single source of truth is build_controller_sft_data_v3.py
# in the project root. Import tolerantly, fall back to a local copy.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.environ.get("VIDORAG_PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))

CONTROLLER_SYSTEM_FALLBACK = (
    "You are a strict evidence auditor for retrieval-augmented question answering "
    "over slide decks. You decide whether the retrieved pages are sufficient to "
    "answer a question, or whether more pages must be retrieved."
)

CONTROLLER_TEMPLATE_FALLBACK = (
    "Question: {question}\n\n"
    "A first-stage retriever selected the following {n} document page images as "
    "evidence for this question:\n"
    "{image_tokens}\n\n"
    "{confidence}\n\n"
    "Judge STRICTLY whether these pages alone contain ALL evidence needed to answer "
    "the question completely and accurately (specific numbers, names, dates, table "
    "cells, figure details; for multi-part questions every part must be covered). "
    "Trust only what is visible in these pages - never your own background knowledge.\n\n"
    'Return ONLY a JSON object, either:\n'
    '{{"decision": "STOP"}}\n'
    "if the pages are sufficient, or\n"
    '{{"decision": "RETRIEVE_MORE"}}\n'
    "if any required evidence is absent from these pages."
)


def _import_controller_template():
    """Import template + confidence builder from the project root (SSOT).

    Returns (system, template, build_confidence_block, source_tag). Falls back
    to the local copies above when the project root is unreachable (tests).
    """
    import importlib.util
    path = os.path.join(_PROJECT_ROOT, "build_controller_sft_data_v3.py")
    try:
        spec = importlib.util.spec_from_file_location("build_controller_sft_data_v3", path)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return (mod.CONTROLLER_SYSTEM, mod.CONTROLLER_TEMPLATE_V3,
                    mod.build_confidence_block, "project_v3")
    except Exception as e:  # keep the trainer importable anywhere
        logger.warning(f"controller template import failed for {path}: {e}")
    logger.warning("Using FALLBACK controller template (not the project SSOT!)")
    return (CONTROLLER_SYSTEM_FALLBACK, CONTROLLER_TEMPLATE_FALLBACK, None, "fallback")


def _parse_decision(text: str) -> Optional[str]:
    """Parse {"decision": ...} JSON; tolerate noise around the JSON object."""
    STOP, RETR = "STOP", "RETRIEVE_MORE"
    try:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            obj = json.loads(text[start:end + 1])
            d = str(obj.get("decision", "")).strip().upper().replace(" ", "_")
            if d in (STOP, RETR):
                return d
    except Exception:
        pass
    return None


class ControllerRewardCache:
    """uid -> precomputed branch scores from build_controller_reward_cache.py."""

    def __init__(self, path: str):
        self.by_uid: Dict[str, Dict[str, Any]] = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "stop_score" in rec and "retrieve_score" in rec:
                    self.by_uid.setdefault(rec["uid"], rec)
        logger.info(f"ControllerRewardCache: {len(self.by_uid)} uids from {path}")

    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.by_uid.get(uid)

    def __len__(self):
        return len(self.by_uid)


class ViDoRAGControllerGRPOTrainer(ViDoRAGVLMPPOTrainer):
    """GRPO trainer for the STOP/RETRIEVE_MORE controller (cached rewards)."""

    STOP = "STOP"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    ADAPTER_NAME = "controller"
    KL_REF_NAME = "kl_ref_0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctrl_group_size = int(os.environ.get("MAPPO_GRPO_GROUP_SIZE", "4"))
        self._ctrl_batch_questions = int(os.environ.get("CONTROLLER_GRPO_BATCH", "8"))
        self._ctrl_lambda = float(os.environ.get("CONTROLLER_GRPO_LAMBDA", "0.03"))
        self._ctrl_format_penalty = float(os.environ.get("CONTROLLER_GRPO_FORMAT_PENALTY", "-0.5"))
        self._ctrl_pos_weight = float(os.environ.get("CONTROLLER_GRPO_POS_WEIGHT", "4"))
        self._ctrl_temperature = float(os.environ.get("CONTROLLER_GRPO_TEMPERATURE", "0.9"))
        self._ctrl_max_new_tokens = int(os.environ.get("CONTROLLER_GRPO_MAX_NEW_TOKENS", "24"))
        self._ctrl_max_images = int(os.environ.get("CONTROLLER_GRPO_MAX_IMAGES", "8"))
        self._ctrl_log_every = int(os.environ.get("CONTROLLER_GRPO_LOG_EVERY", "20"))
        self._ctrl_save_every = int(os.environ.get("CONTROLLER_GRPO_SAVE_EVERY", "20"))
        self._ctrl_rng = random.Random(int(os.environ.get("CONTROLLER_GRPO_SEED", "42")))
        self._ctrl_debug_rollouts = 0

        (self._ctrl_system, self._ctrl_template,
         self._ctrl_confidence_fn, self._ctrl_tpl_src) = _import_controller_template()

        # Tokenized forced decisions for the stratified fix-up.
        self._ctrl_forced_ids = {
            self.STOP: self.tokenizer.encode(
                json.dumps({"decision": self.STOP}), add_special_tokens=False),
            self.RETRIEVE_MORE: self.tokenizer.encode(
                json.dumps({"decision": self.RETRIEVE_MORE}), add_special_tokens=False),
        }
        self._ctrl_records: List[Dict[str, Any]] = []
        self._ctrl_weights: List[float] = []
        self._ctrl_cache: Optional[ControllerRewardCache] = None
        logger.info(
            f"ControllerGRPO init: G={self._ctrl_group_size} batch={self._ctrl_batch_questions} "
            f"lambda={self._ctrl_lambda} fmt_pen={self._ctrl_format_penalty} "
            f"pos_w={self._ctrl_pos_weight} temp={self._ctrl_temperature} "
            f"kl_beta={_GRPO_KL_BETA} template={self._ctrl_tpl_src}"
        )

    # ------------------------------------------------------------------
    # Adapter setup
    # ------------------------------------------------------------------

    def _new_controller_lora_config(self, peft_model):
        from peft import LoraConfig
        # Explicit module list (NOT the string "all-linear"): the vision
        # tower's Conv3d patch embed matches all-linear and PEFT rejects it.
        # Also avoid find_all_linear_modules here: the model already carries a
        # PEFT wrapper, whose artifact names (base_layer/default/Identity)
        # would pollute the target list. Collect real Linear leaf names only,
        # excluding the vision tower ('visual') and the mm projector ('merger').
        import torch.nn as nn
        # Preferred: reuse the 'default' adapter's target_modules (computed by
        # LLaMA-Factory at load time on the UNWRAPPED model, honouring
        # freeze_vision_tower=true). The model already carries that adapter, so
        # walking named_modules here would only see PEFT-wrapped artifacts
        # (base_layer / lora_A / dropout Identity).
        default_cfg = getattr(peft_model, "peft_config", {}).get("default", None)
        targets = getattr(default_cfg, "target_modules", None)
        if isinstance(targets, (list, set, tuple)) and targets:
            module_names = sorted(set(targets))
        else:
            forbidden = ("lm_head", "visual", "merger", "lora_",
                         "modules_to_save")
            names = set()
            for name, module in peft_model.named_modules():
                if any(f in name for f in forbidden):
                    continue
                if isinstance(module, nn.Linear):
                    parts = name.split(".")
                    leaf = parts[-2] if parts[-1] == "base_layer" else parts[-1]
                    if leaf not in ("default", "base_layer"):
                        names.add(leaf)
            module_names = sorted(names)
        logger.info(f"ControllerGRPO: LoRA targets: {module_names}")
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=int(os.environ.get("CONTROLLER_GRPO_LORA_RANK", "16")),
            lora_alpha=int(os.environ.get("CONTROLLER_GRPO_LORA_ALPHA", "32")),
            lora_dropout=0.0,
            bias="none",
            target_modules=module_names,
        )

    def _load_controller_adapter(self) -> None:
        """Load (or cold-create) the controller LoRA + frozen kl_ref_0, then
        rebuild the optimizer so it covers the new adapter params.

        Mirrors the base trainer's _load_mappo_adapters optimizer-scope fix:
        the optimizer created in __init__ predates these adapters and would
        silently no-op on them otherwise.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        peft_model = getattr(unwrapped, "pretrained_model", unwrapped)

        adapter_path = os.environ.get("VIDORAG_CONTROLLER_ADAPTER", "")
        kl_ref_path = os.environ.get("VIDORAG_CONTROLLER_KL_REF_ADAPTER", "") or adapter_path

        if adapter_path and os.path.isdir(adapter_path):
            peft_model.load_adapter(adapter_path, adapter_name=self.ADAPTER_NAME, is_trainable=True)
            logger.info(f"ControllerGRPO: warm-started controller adapter from {adapter_path}")
        else:
            peft_model.add_adapter(self.ADAPTER_NAME, self._new_controller_lora_config(peft_model))
            logger.info("ControllerGRPO: cold-started a fresh controller LoRA (r16/a32)")

        # Frozen KL reference (same weights as the start point; identity LoRA
        # when cold-starting). _enable_all_adapter_grads keeps 'kl_ref_' frozen.
        if _GRPO_KL_BETA > 0:
            if kl_ref_path and os.path.isdir(kl_ref_path):
                peft_model.load_adapter(kl_ref_path, adapter_name=self.KL_REF_NAME, is_trainable=False)
            else:
                peft_model.add_adapter(self.KL_REF_NAME, self._new_controller_lora_config(peft_model))
            logger.info(f"ControllerGRPO: KL reference '{self.KL_REF_NAME}' attached "
                        f"(beta={_GRPO_KL_BETA}, src={kl_ref_path or 'fresh'})")

        self._adapter_names = [self.ADAPTER_NAME]
        self._mappo_independent = False  # saving path is custom (single adapter)

        self._enable_all_adapter_grads(peft_model)
        # Optimizer-scope fix (identical root cause as MAPPO independent).
        optimizer = self.create_optimizer(self.model, self.args, self.finetuning_args)
        scheduler = self.create_scheduler(self.args, self._num_train_steps, optimizer)
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        opt_params = sum(1 for g in self.optimizer.param_groups for _ in g["params"])
        logger.info(f"ControllerGRPO: optimizer rebuilt over {opt_params} params")

        peft_model.set_adapter(self.ADAPTER_NAME)
        self._enable_all_adapter_grads(peft_model)

    # ------------------------------------------------------------------
    # Training data (records must exist in the reward cache)
    # ------------------------------------------------------------------

    def _load_controller_records(self) -> None:
        """Load (question, round-1 pages, fusion ranking, oracle label) per uid
        from the CONTROLLER_SFT Stage-1 caches, restricted to cached uids."""
        data_dir = os.environ.get(
            "CONTROLLER_GRPO_DATA_DIR",
            os.path.join(_PROJECT_ROOT, "outputs/CONTROLLER_SFT/data"))

        def _jsonl_keyed(path):
            out = {}
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            out.setdefault(rec["uid"], rec)
            return out

        sampled = _jsonl_keyed(os.path.join(data_dir, "sampled.jsonl"))
        labels: Dict[str, Any] = {}
        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("labels") and fname.endswith(".jsonl"):
                labels.update(_jsonl_keyed(os.path.join(data_dir, fname)))
        rankings: Dict[str, List[str]] = {}
        for fname in ("retrieval.jsonl", "retrieval_shard1.jsonl"):
            for uid, rec in _jsonl_keyed(os.path.join(data_dir, fname)).items():
                if rec.get("ok") and rec.get("fusion_ranking"):
                    rankings.setdefault(uid, rec["fusion_ranking"])

        records, weights = [], []
        missing = {"query": 0, "label": 0, "ranking": 0}
        for uid in self._ctrl_cache.by_uid:
            q = sampled.get(uid)
            lab = labels.get(uid)
            rk = rankings.get(uid)
            if not q:
                missing["query"] += 1
                continue
            if not (lab and lab.get("ok") and lab.get("label")):
                missing["label"] += 1
                continue
            if not rk:
                missing["ranking"] += 1
                continue
            records.append({
                "uid": uid,
                "query": q["query"],
                "selected": lab.get("selected") or [],
                "label": lab["label"],
                "ranking": rk[:20],
            })
            weights.append(self._ctrl_pos_weight
                           if lab["label"] == self.RETRIEVE_MORE else 1.0)
        if not records:
            raise RuntimeError(
                "ControllerGRPO: no training records matched the reward cache "
                f"(cache={len(self._ctrl_cache)}, missing={missing})")
        self._ctrl_records, self._ctrl_weights = records, weights
        n_pos = sum(1 for r in records if r["label"] == self.RETRIEVE_MORE)
        logger.info(f"ControllerGRPO: {len(records)} decision records "
                    f"({n_pos} RETRIEVE_MORE, weighted x{self._ctrl_pos_weight}); "
                    f"missing={missing}")

    # ------------------------------------------------------------------
    # Prompt construction (byte-identical to the SFT v3 training data)
    # ------------------------------------------------------------------

    def _build_controller_prompt(self, record: Dict[str, Any]
                                 ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Returns (messages, image_paths) for trans_vlm_messages_to_inputs."""
        img_dir = os.path.join(_PROJECT_ROOT, "data/SlideVQA/img")
        kept_ids, kept_paths = [], []
        for pid in record["selected"]:
            p = os.path.join(img_dir, pid + ".jpg")
            if os.path.exists(p):
                kept_ids.append(pid)
                kept_paths.append(p)
            if len(kept_paths) >= self._ctrl_max_images:
                break

        image_tokens = " ".join(["<image>"] * len(kept_paths))
        if self._ctrl_confidence_fn is not None:
            confidence = self._ctrl_confidence_fn(kept_ids, record["ranking"])
        else:
            confidence = "[Retrieval Confidence]\n(fallback: ranks unavailable)"
        instruction = self._ctrl_template.format(
            question=record["query"], n=len(kept_paths),
            image_tokens=image_tokens, confidence=confidence)

        # Inline image placement at the exact <image> positions (byte-consistent
        # with the SFT data; Qwen vision placeholder tokens per base trainer).
        vision_tok = "<|vision_start|><|image_pad|><|vision_end|>"
        user_content = vision_tok.join(instruction.split("<image>"))
        messages = [
            {"role": "system", "content": self._ctrl_system},
            {"role": "user", "content": user_content},
        ]
        return messages, kept_paths

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def _rollout_group(self, record: Dict[str, Any], step: int = 0) -> List[Dict[str, Any]]:
        """G sampled decision rollouts for one prompt, with stratified fix-up.

        Returns list of dicts: {response (1-D LongTensor), action, parse_ok,
        forced, text}. If all parseable rollouts decode to the SAME action,
        the last rollout is replaced by the tokenized opposite decision so the
        group is never constant (constant groups yield zero advantage).
        """
        G = self._ctrl_group_size
        messages, image_paths = self._build_controller_prompt(record)
        batch = self.trans_vlm_messages_to_inputs([messages] * G, [image_paths] * G)

        self.generation_config.do_sample = True
        self.generation_config.temperature = self._ctrl_temperature
        self.generation_config.top_p = float(os.environ.get("CONTROLLER_GRPO_TOP_P", "0.95"))

        _queries, responses = self.get_vlm_inputs(
            batch, logits_processor=None, max_new_tokens=self._ctrl_max_new_tokens)

        out = []
        for r in responses:
            text = self.tokenizer.decode(r, skip_special_tokens=True)
            action = _parse_decision(text)
            out.append({"response": r, "action": action,
                        "parse_ok": action is not None, "forced": False, "text": text})

        parsed = [o["action"] for o in out if o["action"] is not None]
        if self._ctrl_debug_rollouts < int(os.environ.get("CONTROLLER_GRPO_DEBUG_ROLLOUTS", "0")):
            self._ctrl_debug_rollouts += 1
            logger.info(f"[ControllerGRPO rollout sample step={step}] uid={record['uid']} "
                        f"oracle={record['label']} texts="
                        + " || ".join(repr(o["text"][:80]) for o in out))
        if parsed and len(set(parsed)) == 1:
            # All parseable rollouts agree -> force the opposite on the last.
            opposite = self.STOP if parsed[0] == self.RETRIEVE_MORE else self.RETRIEVE_MORE
            forced_ids = torch.tensor(self._ctrl_forced_ids[opposite], dtype=torch.long)
            out[-1] = {"response": forced_ids, "action": opposite,
                       "parse_ok": True, "forced": True,
                       "text": json.dumps({"decision": opposite})}
        return out

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_cached_reward(self, cached: Dict[str, Any], action: Optional[str],
                               parse_ok: bool) -> float:
        """r = (score-1)/4 - lambda*[RETRIEVE_MORE]; parse failure -> penalty."""
        if not parse_ok or action is None:
            return self._ctrl_format_penalty
        score = (cached["retrieve_score"] if action == self.RETRIEVE_MORE
                 else cached["stop_score"])
        r = (float(score) - 1.0) / 4.0
        if action == self.RETRIEVE_MORE:
            r -= self._ctrl_lambda
        return r

    @torch.no_grad()
    def _response_mean_logp(self, adapter_name: str, batch_dict, q, r) -> float:
        """Mean response-token log-prob under the named adapter (KL anchor)."""
        import torch.nn.functional as F
        device = self.current_device
        unwrapped = self.accelerator.unwrap_model(self.model)
        peft_model = getattr(unwrapped, "pretrained_model", unwrapped)
        prev = list(getattr(peft_model, "active_adapters", [self.ADAPTER_NAME]))
        try:
            peft_model.set_adapter(adapter_name)
            input_ids = torch.cat([q, r], dim=0).unsqueeze(0).to(device)
            fwd = dict(input_ids=input_ids,
                       attention_mask=torch.ones_like(input_ids),
                       return_dict=True, use_cache=False)
            pv = batch_dict.get("pixel_values")
            thw = batch_dict.get("image_grid_thw")
            if pv is not None and thw is not None:
                fwd["pixel_values"] = pv.to(device)
                fwd["image_grid_thw"] = thw.to(device)
            out = self.model(**fwd)
            logits = out.logits if hasattr(out, "logits") else out[0]
            logp = F.log_softmax(logits.float()[:, :-1, :], dim=-1)
            tok = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
            return float(tok[-r.shape[0]:].mean().item())
        finally:
            try:
                peft_model.set_adapter(prev)
            except Exception:
                peft_model.set_adapter(self.ADAPTER_NAME)

    def _compute_kl_penalties(self, samples: List[Dict[str, Any]]) -> List[float]:
        """Per-rollout KL(pi || pi_ref) ~ mean_resp(logp_pi - logp_ref)."""
        if _GRPO_KL_BETA <= 0:
            return [0.0] * len(samples)
        kls = []
        for s in samples:
            try:
                lp_pi = self._response_mean_logp(self.ADAPTER_NAME, s["batch_dict"],
                                                 s["query"], s["response"])
                lp_ref = self._response_mean_logp(self.KL_REF_NAME, s["batch_dict"],
                                                  s["query"], s["response"])
                kls.append(max(0.0, lp_pi - lp_ref))
            except Exception as e:
                logger.warning(f"ControllerGRPO KL failed ({e}); using 0")
                kls.append(0.0)
        return kls

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Controller GRPO loop: weighted question sampling -> G rollouts ->
        cached-reward lookup -> group advantages -> _grpo_step."""
        cache_path = os.environ.get(
            "CONTROLLER_REWARD_CACHE",
            os.path.join(_PROJECT_ROOT, "outputs/CONTROLLER_SFT/controller_reward_cache.jsonl"))
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"CONTROLLER_REWARD_CACHE not found: {cache_path} "
                "(run build_controller_reward_cache.py first)")
        self._ctrl_cache = ControllerRewardCache(cache_path)
        self._load_controller_adapter()
        self._load_controller_records()

        max_steps = self.args.max_steps if self.args.max_steps > 0 else 100
        # Smoke-test override: CONTROLLER_GRPO_MAX_STEPS=5 for a 5-step run.
        max_steps = int(os.environ.get("CONTROLLER_GRPO_MAX_STEPS", str(max_steps)))
        logger.info(f"ControllerGRPO: starting {max_steps} steps, "
                    f"{self._ctrl_batch_questions} questions/step, G={self._ctrl_group_size}")

        for step in range(1, max_steps + 1):
            self.state.global_step = step
            step_records = self._ctrl_rng.choices(
                self._ctrl_records, weights=self._ctrl_weights,
                k=self._ctrl_batch_questions)

            samples: List[Dict[str, Any]] = []
            rewards: List[float] = []
            group_ids: List[int] = []
            step_actions: List[Tuple[Optional[str], str, bool]] = []  # (action, label, parse_ok)

            # Rollouts + KL in eval mode (the previous _grpo_step left the model
            # in train() with gradient checkpointing, which corrupts generate()).
            self.model.eval()
            self._ctrl_debug_rollouts = 0  # budget applies per step

            for g_idx, record in enumerate(step_records):
                cached = self._ctrl_cache.get(record["uid"])
                if cached is None:  # cannot happen after _load_controller_records
                    continue
                messages, image_paths = self._build_controller_prompt(record)
                prompt_batch = self.trans_vlm_messages_to_inputs([messages], [image_paths])
                rollouts = self._rollout_group(record, step=step)

                # Prompt ids (pads stripped) shared by all G rollouts of this
                # question; needed by the KL forward and _grpo_step.
                q_ids = prompt_batch["input_ids"][0].cpu()
                q_ids = q_ids[(q_ids != self.tokenizer.pad_token_id).nonzero()[0].item():]

                for o in rollouts:
                    r = self._compute_cached_reward(cached, o["action"], o["parse_ok"])
                    samples.append({
                        "query": q_ids,
                        "response": o["response"],
                        "batch_dict": prompt_batch,
                    })
                    rewards.append(r)
                    group_ids.append(g_idx)
                    step_actions.append((o["action"], record["label"], o["parse_ok"]))

            # ---- KL anchor (subtract before advantage) ----
            kls = self._compute_kl_penalties(samples)
            rw = torch.tensor(
                [r - _GRPO_KL_BETA * k for r, k in zip(rewards, kls)],
                dtype=torch.float32)
            adv = _grpo_group_advantages(rw, group_ids)

            # ---- policy update on the ACTIVE controller adapter ----
            records_for_step = self._ctrl_collect_records(samples)
            if step == 1:
                for _bi, (_bd, _ql, _rl) in enumerate(records_for_step):
                    _pad_id = self.tokenizer.pad_token_id
                    try:
                        _pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    except Exception:
                        pass
                    _q0 = _ql[0]
                    _npad = int((_q0 == _pad_id).sum().item())
                    _pv = _bd.get("pixel_values")
                    _thw = _bd.get("image_grid_thw")
                    logger.info(
                        f"[CTRLDBG] rec{_bi} rows={len(_ql)} qlen={_q0.shape[0]} "
                        f"imgpad={_npad} pv={tuple(_pv.shape) if _pv is not None else None} "
                        f"thw={_thw.tolist() if _thw is not None else None} "
                        f"thw_counts={_bd.get('thw_counts')}")
            stats = self._grpo_step(records_for_step, adv.tolist())

            # ---- logging ----
            n = len(step_actions)
            trigger = sum(1 for a, _l, _p in step_actions if a == self.RETRIEVE_MORE) / max(n, 1)
            parse_fail = sum(1 for _a, _l, p in step_actions if not p) / max(n, 1)
            agree = sum(1 for a, l, p in step_actions if p and a == l) / max(n, 1)
            mean_r = sum(rewards) / max(len(rewards), 1)
            if step % self._ctrl_log_every == 1 or step == max_steps:
                logger.info(
                    f"[ControllerGRPO step {step}/{max_steps}] "
                    f"trigger_rate={trigger:.4f} parse_fail_rate={parse_fail:.4f} "
                    f"oracle_agreement={agree:.4f} mean_reward={mean_r:.4f} "
                    f"kl_mean={sum(kls) / max(len(kls), 1):.4f} "
                    f"grpo/loss={stats.get('grpo/loss', 0.0):.4f} "
                    f"grad_norm={stats.get('grpo/grad_norm', 0.0):.4f}")
                print(f"[ControllerGRPO step {step}] trigger={trigger:.3f} "
                      f"parse_fail={parse_fail:.3f} oracle_agree={agree:.3f} "
                      f"mean_r={mean_r:.4f}", flush=True)

            # ---- checkpointing ----
            if self._ctrl_save_every > 0 and step % self._ctrl_save_every == 0:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
                self.save_model(ckpt_dir)
                logger.info(f"ControllerGRPO: saved checkpoint {ckpt_dir}")

        logger.info("[ControllerGRPO] training loop finished")

    def _ctrl_collect_records(self, samples: List[Dict[str, Any]]):
        """One (batch_dict, [query], [response]) record PER rollout sample.

        _grpo_step slices pixel_values by ROW index using thw_counts, which is
        per-prompt-row (len==1 here). Grouping G rollouts under one batch_dict
        would send rows 1..G-1 into the 1-image-per-row fallback and mismatch
        image tokens vs features. One record per sample keeps every forward on
        row 0, which slices ALL of that prompt's images correctly. Sample order
        (and thus advantage alignment) is preserved.
        """
        return [(s["batch_dict"], [s["query"]], [s["response"]]) for s in samples]

    # ------------------------------------------------------------------
    # Saving: controller adapter only, under checkpoint-N/controller/
    # ------------------------------------------------------------------

    def save_model(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        backbone = getattr(unwrapped, "pretrained_model", None)
        if backbone is not None and hasattr(backbone, "save_pretrained") \
                and hasattr(backbone, "peft_config") and self.ADAPTER_NAME in backbone.peft_config:
            # PEFT writes the selected adapter into <output_dir>/controller/
            # itself, so pass output_dir directly (spec: checkpoint-N/controller/).
            os.makedirs(output_dir, exist_ok=True)
            try:
                backbone.set_adapter(self.ADAPTER_NAME)
                try:
                    backbone.save_pretrained(output_dir, selected_adapters=[self.ADAPTER_NAME])
                except TypeError:
                    # Older PEFT: save everything then copy the adapter files.
                    import shutil
                    import tempfile
                    adapter_dir = os.path.join(output_dir, self.ADAPTER_NAME)
                    os.makedirs(adapter_dir, exist_ok=True)
                    with tempfile.TemporaryDirectory() as tmp:
                        backbone.save_pretrained(tmp)
                        for fname in os.listdir(tmp):
                            shutil.copy2(os.path.join(tmp, fname),
                                         os.path.join(adapter_dir, fname))
                logger.info(f"ControllerGRPO: saved controller adapter under {output_dir}")
            except Exception as e:
                logger.warning(f"ControllerGRPO: controller adapter save failed: {e}")
        else:
            logger.warning("ControllerGRPO: no peft backbone found; skipping adapter save")

        if self.processor is not None:
            try:
                self.processor.save_pretrained(output_dir)
            except Exception:
                pass
