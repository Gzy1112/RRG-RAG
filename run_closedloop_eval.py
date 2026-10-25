#!/usr/bin/env python3
"""STOP/RETRIEVE_MORE closed-loop inference eval (Phase 1, inference-only).

Validates the "second-retrieval closed loop" mechanism at inference time, with
NO training and NO changes to any existing asset:

    S first-round selection (GMM pool ~10 pages) -> Controller: STOP / RETRIEVE_MORE
      STOP          -> standard downstream pipeline (Inspector -> Synthesizer)
      RETRIEVE_MORE -> extended pool (original pool UNION fusion ranks 11-40,
                       sorted by fusion rank, truncated to CLOSED_LOOP_RESELECT_POOL)
                       -> S re-selects -> standard downstream pipeline

Everything downstream of the controller is byte-for-byte the same code path as
the baseline TFR v4fix + Qwen3-hybrid run, so a STOP decision reproduces the
baseline behaviour and any score movement is attributable to the loop itself.

RETRIEVAL EXTENSION (no engine, no server change)
    The extended pool is read from recovery_ceiling_ranks.jsonl, which caches
    the EXACT full post-fusion (pre-GMM) ranking of this same 300-question
    protocol, produced by measure_recovery_ceiling.py's fusion reproduction
    (per-engine min-max norm, 0.5/0.5 sum). Using the cache is strictly more
    stable than rebuilding an in-process engine (no GPU memory next to the 3
    agents, no drift vs. the measurement that defined class B) and requires no
    modification of the port-8000 server.

CONTROLLER MODES (env CONTROLLER_MODE)
    oracle : RETRIEVE_MORE iff gold not in first-round seeker selection AND
             best gold fusion rank <= CLOSED_LOOP_TRIGGER_RANK (40). Measures
             the mechanism's efficiency upper bound.
    prompt : zero-shot sufficiency judgement with the BASE Qwen2.5-VL-3B
             (QWEN dir, no adapter): question + first-round selected page
             images -> {"decision": "STOP"|"RETRIEVE_MORE", "missing_evidence": ...}
    sft    : the SFT-trained controller adapter (Phase 2, task #64):
             Qwen2.5-VL-3B + LoRA trained on frozen-TFR-seeker first-round
             labels. Adapter path via CONTROLLER_SFT_ADAPTER (default
             outputs/CONTROLLER_SFT/v1). Prompt/system match the
             training data byte-for-byte; ANY parse/model failure defaults to
             STOP (preserves baseline behaviour).
    sft_thresh : same adapter as 'sft', but instead of greedy text decoding
             it computes P(RETRIEVE_MORE) = p(STOP-tok vs RETRIEVE-MORE-tok)
             after the shared response prefix '{"decision": "' in ONE forward
             pass, and triggers iff p >= CONTROLLER_TAU (env, default 0.5 =
             greedy-equivalent). Task #66 v2a. Failure defaults to STOP.
    sft_thresh_v3 : sft_thresh with the V3 prompt (fusion-rank confidence
             block), byte-identical to the SFT-v3/GRPO training format -
             required when thresholding a v3-trained (e.g. GRPO) adapter.
             Task #73. Failure defaults to STOP.
    inspector_v2 : val-trained multi-action GRPO Inspector. Scores the valid
             responses STOP / DECK / EXPAND under INSPECTOR_V2_ADAPTER, with
             optional val-calibrated INSPECTOR_V2_ACTION_BIASES. DECK gates
             same-deck HOP2; EXPAND gates top-N union reselection. Use
             HOP2_MODE=deck, HOP2_MIN_SEL=1, CLOSED_LOOP_FIRE_MODE=union and
             RANK1_RESCUE=append to reproduce its cached training branches.

RESELECTION POOL SIZE (env CLOSED_LOOP_RESELECT_POOL, default 20)
    Chosen 20 (upper end of the advised 15-20 range): the seeker v4 adapter
    was trained on <=10 candidates and 20 images already stress its context,
    so 20 is the widest defensible pool. It covers fusion ranks 1-20
    (recall@20 = 0.7483 vs 0.7267 of the current GMM pool). Consequence: of
    the 19 class-B samples only those with best gold rank <=20 are physically
    rescuable; the oracle still TRIGGERS for rank<=40 per spec, so triggers at
    ranks 21-40 document the truncation cost instead of hiding it.

RESELECTION POOL COMPOSITION (env RETRIEVE_MODE, default 'merged')
    Plan-B two-stage ablation (task #65): the merged 20-page pool stresses the
    seeker (trained on <=10 candidates) and its gold hit-rate drops as the pool
    widens (39% @20 -> 27% @40). Alternative compositions for the reselection
    round, all keeping the SAME oracle trigger set:
      merged        : original behaviour - fusion ranks 1..N (default 20)
      new_only      : ONLY fusion ranks 11-20 (10 fresh pages, the seeker's
                      native scale)
      keep_plus_new : first-round seeker selection (context kept) UNION fusion
                      ranks 11-20 (~13-15 pages, deduped)
    Default 'merged' reproduces every previous run byte-for-byte.

FIRE MODE (env CLOSED_LOOP_FIRE_MODE, default 'replace')
    What a RETRIEVE_MORE decision does to the evidence the inspector sees.
      replace : original behaviour - the reselection result REPLACES the
                round-1 seeker selection (byte-identical to every previous
                run).
      union   : the round-1 selection is KEPT, the reselected pages are
                appended (deduped, round-1 first).
    Motivated by the measured failure mode of 'replace': when the round-1
    pick already contained the gold, the reselection could drop it and
    destroy a correct answer. On the deployed GRPO full run, 489 of 746
    fires were on questions whose round-1 selection already had the gold,
    and 45 of those dropped it - their scores are 26x1, 8x2, 2x3, 2x4, 7x5
    (mean 2.02). Union is additive, so a fire cannot lose evidence; the
    union set stays tiny (mean 2.0 pages, max 5, vs 1.5 for replace).
    Both modes run the SAME seeker on the SAME reselection pool - the only
    difference is which pages reach the inspector/synthesizer.

USAGE
    The paper configuration (TFR agents + Inspector GRPO + rank-1 rescue) is
    launched by scripts/eval_closedloop.sh. Minimal manual call:
    CONTROLLER_MODE=always_stop EVAL_MODE_NAME=tfr_baseline \
        EVAL_SEEKER_PATH=... EVAL_SYNTH_PATH=... EVAL_INSPECTOR_PATH=... \
        EVAL_BASE_MODEL=... EVAL_RETRIEVAL_SERVER=http://localhost:8000 \
        python run_closedloop_eval.py --max_samples 300

Per-query JSONL rows carry a 'closed_loop' block: controller decision, whether
reselection fired, pre/post seeker selections, extended pool, and the final
score (in eval_result). The STOP path writes identical metrics to baseline.
"""
import os
import sys
import json
import argparse

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from eval import MMRAG  # noqa: E402
from llms.llm import LLM  # noqa: E402
from llms.evaluator import JudgeUnavailableError  # noqa: E402
from llama_index.core.schema import NodeWithScore, ImageNode  # noqa: E402
from utils.parse_tool import extract_json  # noqa: E402

DATASET = os.environ.get('EVAL_DATASET', 'SlideVQA')
QUERY_FILE = os.environ.get('EVAL_QUERY_FILE', 'rag_dataset_test.json' if DATASET == 'SlideVQA' else 'rag_dataset.json')
EVAL_DIR = os.environ.get('EVAL_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs/eval', DATASET))
# Only used when no retrieval server is given (EVAL_RETRIEVAL_SERVER unset).
COLQWEN_MODEL = os.environ.get('EVAL_COLQWEN_MODEL', 'vidore/colqwen2-v1.0')
EMBED_MODEL = os.environ.get('EVAL_EMBED_MODEL', 'BAAI/bge-m3')
BASE_MODEL_3B = os.environ.get('EVAL_PROMPT_CONTROLLER_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')  # 'prompt' controller only
CONTROLLER_SFT_ADAPTER = os.environ.get(
    'CONTROLLER_SFT_ADAPTER',
    os.path.join(PROJECT_ROOT, 'outputs/CONTROLLER_SFT/v1'))

RANK_CACHE_PATH = os.environ.get(
    'RANK_CACHE_PATH',
    os.path.join(PROJECT_ROOT, 'outputs/rank_cache',
                 'SlideVQA_test.jsonl' if DATASET == 'SlideVQA' else f'{DATASET}.jsonl'))

# Env-configurable knobs
CONTROLLER_MODE = os.environ.get('CONTROLLER_MODE', 'oracle').lower()
CONTROLLER_TAU = float(os.environ.get('CONTROLLER_TAU', '0.5'))
TRIGGER_RANK_LIMIT = int(os.environ.get('CLOSED_LOOP_TRIGGER_RANK', '40'))
RESELECT_POOL_N = int(os.environ.get('CLOSED_LOOP_RESELECT_POOL', '20'))
PROMPT_MAX_IMAGES = int(os.environ.get('CLOSED_LOOP_PROMPT_MAX_IMAGES', '8'))

# What a fire does to the evidence set - see FIRE MODE in the header.
FIRE_MODE = os.environ.get('CLOSED_LOOP_FIRE_MODE', 'replace').lower()
if FIRE_MODE not in ('replace', 'union'):
    raise SystemExit(f'unknown CLOSED_LOOP_FIRE_MODE={FIRE_MODE!r} '
                     f'(expected replace | union)')

# Plan-B two-stage reselection: which pages the seeker re-selects from.
RETRIEVE_MODE = os.environ.get('RETRIEVE_MODE', 'merged').lower()
NEW_CAND_START = int(os.environ.get('CLOSED_LOOP_NEW_CAND_START', '11'))   # 1-based
NEW_CAND_END = int(os.environ.get('CLOSED_LOOP_NEW_CAND_END', '20'))       # inclusive
if RETRIEVE_MODE not in ('merged', 'new_only', 'keep_plus_new'):
    raise SystemExit(f'unknown RETRIEVE_MODE={RETRIEVE_MODE!r} '
                     f'(expected merged | new_only | keep_plus_new)')

# Same-deck second hop (multi-hop evidence completion). When the round-1 seeker
# picks >= HOP2_MIN_SEL pages (a strong multi-hop signal), offer it the
# best-ranked pages from the same deck(s) as its picks and let it re-select.
# Off by default.
HOP2_MODE = os.environ.get('HOP2_MODE', 'off').lower()
HOP2_K = int(os.environ.get('HOP2_K', '5'))
HOP2_MIN_SEL = int(os.environ.get('HOP2_MIN_SEL', '2'))
HOP2_FIRE = os.environ.get('HOP2_FIRE', 'union').lower()
if HOP2_MODE not in ('off', 'deck'):
    raise SystemExit(f'unknown HOP2_MODE={HOP2_MODE!r} (expected off | deck)')
if HOP2_FIRE not in ('replace', 'union'):
    raise SystemExit(f'unknown HOP2_FIRE={HOP2_FIRE!r} (expected replace | union)')

# Rank-1 rescue. The first page of the GMM pool is gold for ~80% of questions,
# yet the seeker sometimes drops it (105 of the 151 seeker-miss failures after
# HOP2). When on, that page is added to the evidence if missing: 'append' puts
# it after the seeker's picks, 'prepend' before them. Off by default.
RANK1_RESCUE = os.environ.get('RANK1_RESCUE', 'off').lower()
if RANK1_RESCUE not in ('off', 'append', 'prepend'):
    raise SystemExit(f'unknown RANK1_RESCUE={RANK1_RESCUE!r} (expected off | append | prepend)')

# The legacy ViDoRAG inspector never parses under the current adapters: it fails
# three times and hands every selected page to the synthesizer with no candidate
# answer (6648/6648 parse failures on the full run). With this flag the hand-off
# is made directly, which gives the same synthesizer input and saves three
# generations per question. Off by default.
SKIP_PASSTHROUGH_INSPECTOR = os.environ.get('SKIP_PASSTHROUGH_INSPECTOR', '0') == '1'

# Latency measurement. CLOSED_LOOP_TIMING=1 records per-stage wall-clock seconds
# (CUDA-synchronised) in closed_loop.timing; CLOSED_LOOP_SKIP_JUDGE=1 skips the
# LLM judge (score/passing recorded as 0) for timing-only runs. Both off by default.
TIMING = os.environ.get('CLOSED_LOOP_TIMING', '0') == '1'
SKIP_JUDGE = os.environ.get('CLOSED_LOOP_SKIP_JUDGE', '0') == '1'

STOP = 'STOP'
RETRIEVE_MORE = 'RETRIEVE_MORE'


def _page_id_of(image_path):
    """'.../doc_95_7.jpg' -> 'doc_95_7'."""
    return os.path.splitext(os.path.basename(image_path))[0]


def _now():
    """perf_counter after draining queued CUDA work (only called when TIMING)."""
    import time
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _deck_of(page_id):
    """'doc_95_7' -> 'doc_95'."""
    return page_id.rsplit('_', 1)[0]


def hop2_deck_candidates(ranking, selected_ids, k):
    """Top-k pages of the selected pages' deck(s), in retrieval-rank order,
    excluding the pages already selected. Uses only the retrieval ranking."""
    decks = {_deck_of(p) for p in selected_ids}
    chosen = set(selected_ids)
    out = []
    for pid in ranking or []:
        if len(out) >= k:
            break
        if pid not in chosen and _deck_of(pid) in decks:
            out.append(pid)
    return out


# ---------------------------------------------------------------------------
# Fusion-rank cache -> extended pool
# ---------------------------------------------------------------------------

class RankCache:
    """Loads recovery_ceiling_ranks.jsonl (full pre-GMM fusion ranking per uid)."""

    def __init__(self, path=RANK_CACHE_PATH):
        self.by_uid = {}
        if not os.path.exists(path):
            print(f'[closed-loop] rank cache not found (skipping): {path}')
            return
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.by_uid.setdefault(rec['uid'], rec)
        print(f'[closed-loop] rank cache loaded: {len(self.by_uid)} uids from {path}')

    def get(self, uid):
        return self.by_uid.get(uid)

    def gold_ids(self, uid):
        """Gold page-id set for this uid (empty if uid missing)."""
        rec = self.by_uid.get(uid)
        return set((rec or {}).get('gold') or [])

    def new_candidate_ids(self, uid, start=NEW_CAND_START, end=NEW_CAND_END):
        """Fusion ranks start..end (1-based, inclusive) in fusion-rank order.

        Used by the Plan-B two-stage modes: these are the pages the seeker has
        NOT seen in the first round (which drew from the GMM-filtered top-10).
        """
        rec = self.by_uid.get(uid)
        if rec is None:
            return []
        return list(rec['ranking'][start - 1:end])

    def keep_plus_new_pool_ids(self, uid, kept_ids, start=NEW_CAND_START, end=NEW_CAND_END):
        """First-round seeker selection (order preserved) UNION fusion ranks
        start..end, deduped. Kept pages carry the round-1 context forward."""
        pool = list(dict.fromkeys(kept_ids))
        for pid in self.new_candidate_ids(uid, start, end):
            if pid not in pool:
                pool.append(pid)
        return pool

    def extended_pool_ids(self, uid, first_pool_ids, extend_to_rank=TRIGGER_RANK_LIMIT):
        """Original pool UNION fusion ranks 1..extend_to_rank, fusion-rank order.

        Returns (ordered page ids, gold set for this uid). Pool pages missing
        from the cached ranking (should not happen) sort AFTER ranked pages.
        """
        rec = self.by_uid.get(uid)
        if rec is None:
            return list(first_pool_ids), set()
        ranking = rec['ranking']
        pos = {pid: i for i, pid in enumerate(ranking)}
        pool = list(first_pool_ids)
        for pid in ranking[:extend_to_rank]:
            if pid not in pool:
                pool.append(pid)
        pool.sort(key=lambda p: pos.get(p, len(ranking)))
        return pool, set(rec.get('gold') or [])


# ---------------------------------------------------------------------------
# Controllers: unified interface  decide(question, selected_pages, uid) -> dict
# ---------------------------------------------------------------------------

class AlwaysStopController:
    """Always returns STOP — no re-retrieval. Used as the no-controller baseline."""
    name = 'always_stop'

    def decide(self, question, images, uid=None):
        return {'decision': STOP, 'source': 'always_stop', 'reason': 'no-loop baseline'}


class InspectorV2Controller:
    """Inspector GRPO v2 (multi-action recovery, trained on the val split).

    Picks STOP / DECK / EXPAND by the log-probability of each valid response
    under the adapter (inspector_v2_prompt.ActionScorer, same inputs as GRPO
    training). DECK runs the same-deck second hop (needs HOP2_MODE=deck);
    EXPAND is a RETRIEVE_MORE re-selection from the ranking top
    CLOSED_LOOP_RESELECT_POOL. The adapter is loaded into the agents' shared
    base model.
    """

    name = 'inspector_v2'
    ADAPTER_NAME = 'inspector_v2'

    def __init__(self, adapter_path, actions, rank_cache, peft_model, processor, img_dir):
        import inspector_v2_prompt as iv2
        self.rank_cache = rank_cache
        if self.ADAPTER_NAME not in getattr(peft_model, 'peft_config', {}):
            peft_model.load_adapter(adapter_path, adapter_name=self.ADAPTER_NAME)
        self.scorer = iv2.ActionScorer(peft_model, processor, iv2.parse_actions(actions),
                                       adapter_name=self.ADAPTER_NAME, img_dir=img_dir)
        self.biases = {a: 0.0 for a in self.scorer.actions}
        for item in os.environ.get('INSPECTOR_V2_ACTION_BIASES', '').split(','):
            if item.strip():
                action, value = item.split(':')
                action = action.strip().upper()
                if action not in self.biases:
                    raise ValueError(f'Bias specified for unavailable Inspector v2 action {action}')
                self.biases[action] = float(value)
        self.adapter_path = adapter_path

    def decide(self, question, selected_pages, uid=None):
        page_ids = [_page_id_of(p) for p in selected_pages]
        if not page_ids:
            return {'decision': STOP, 'action': 'STOP', 'source': self.name,
                    'reason': 'no pages selected'}
        ranking = (self.rank_cache.get(uid) or {}).get('ranking') or []
        logp = self.scorer.score(question, page_ids, ranking)
        action = max(self.scorer.actions,
                     key=lambda a: (logp[a] + self.biases[a], a == 'STOP'))
        return {'decision': RETRIEVE_MORE if action == 'EXPAND' else STOP,
                'action': action, 'logp': logp, 'action_biases': self.biases,
                'source': self.name}


class OracleController:
    """Mechanism upper bound: triggers exactly when the loop COULD bring gold
    into reach: gold not among the seeker's first-round picks and best gold
    fusion rank <= TRIGGER_RANK_LIMIT.

    Test-only escape hatch: CLOSED_LOOP_FORCE_RETRIEVE=1 makes every decision
    RETRIEVE_MORE (for exercising the reselection path in smoke tests)."""

    name = 'oracle'

    def __init__(self, rank_cache):
        self.rank_cache = rank_cache

    def decide(self, question, selected_pages, uid=None):
        if os.environ.get('CLOSED_LOOP_FORCE_RETRIEVE') == '1':
            return {'decision': RETRIEVE_MORE, 'source': 'forced',
                    'reason': 'CLOSED_LOOP_FORCE_RETRIEVE=1'}
        rec = self.rank_cache.get(uid) if uid else None
        if rec is None:
            return {'decision': STOP, 'source': 'fallback_no_rank', 'reason': 'uid missing in rank cache'}
        gold = set(rec.get('gold') or [])
        selected = {_page_id_of(p) for p in selected_pages}
        best_rank = rec.get('best_gold_rank')
        gold_selected = bool(gold & selected)
        trigger = (not gold_selected) and (best_rank is not None) and (best_rank <= TRIGGER_RANK_LIMIT)
        return {
            'decision': RETRIEVE_MORE if trigger else STOP,
            'source': 'oracle',
            'gold_in_first_selected': gold_selected,
            'best_gold_rank': best_rank,
            'reason': (f'gold not selected, best_rank={best_rank}<={TRIGGER_RANK_LIMIT}'
                       if trigger else
                       ('gold already selected' if gold_selected
                        else f'best_gold_rank={best_rank} beyond {TRIGGER_RANK_LIMIT} or absent')),
        }


PROMPT_SYSTEM = (
    "You are a strict evidence auditor for retrieval-augmented question answering "
    "over slide decks. You decide whether the retrieved pages are sufficient to "
    "answer a question, or whether more pages must be retrieved."
)

PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "A first-stage retriever selected the following {n} document page images as "
    "evidence for this question:\n"
    "{image_tokens}\n\n"
    "Judge STRICTLY whether these pages alone contain ALL evidence needed to answer "
    "the question completely and accurately (specific numbers, names, dates, table "
    "cells, figure details; for multi-part questions every part must be covered). "
    "Trust only what is visible in these pages - never your own background knowledge.\n\n"
    'Return ONLY a JSON object, either:\n'
    '{{"decision": "STOP", "missing_evidence": ""}}\n'
    "if the pages are sufficient, or\n"
    '{{"decision": "RETRIEVE_MORE", "missing_evidence": "<one short sentence stating '
    'exactly what evidence is missing>"}}\n'
    "if any required evidence is absent from these pages."
)


class PromptController:
    """Zero-shot sufficiency judgement with the base Qwen2.5-VL-3B (no adapter).

    Verifies whether an untrained 3B can make sensible STOP/RETRIEVE_MORE calls.
    Any parse/model failure degrades to STOP (preserves baseline behaviour).
    """

    name = 'prompt'

    def __init__(self, model_path=BASE_MODEL_3B):
        self._llm = None
        self.model_path = model_path

    def _ensure_model(self):
        if self._llm is None:
            print(f'[closed-loop] PROMPT controller loading base VLM: {self.model_path} (no adapter)')
            self._llm = LLM(model_name=self.model_path)
        return self._llm

    def decide(self, question, selected_pages, uid=None):
        images = list(selected_pages)[:PROMPT_MAX_IMAGES]
        if not images:
            return {'decision': RETRIEVE_MORE, 'source': 'prompt',
                    'reason': 'no pages selected', 'missing_evidence': 'no evidence pages were selected',
                    'parse_ok': True, 'raw': ''}
        llm = self._ensure_model()
        image_tokens = ' '.join(['<image>'] * len(images))
        prompt = PROMPT_TEMPLATE.format(question=question, n=len(images), image_tokens=image_tokens)
        last_raw = ''
        for attempt in range(2):
            try:
                raw = llm.generate(query=prompt, image=images, system_prompt=PROMPT_SYSTEM)
            except Exception as e:
                print(f'[closed-loop] prompt controller generate failed: {e}')
                return {'decision': STOP, 'source': 'fallback_generate_error',
                        'reason': str(e), 'parse_ok': False, 'raw': ''}
            last_raw = raw
            try:
                obj = extract_json(raw)
                decision = str(obj.get('decision', '')).strip().upper().replace(' ', '_')
                if decision in (STOP, RETRIEVE_MORE):
                    return {'decision': decision, 'source': 'prompt',
                            'missing_evidence': obj.get('missing_evidence', ''),
                            'parse_ok': True, 'raw': raw}
            except Exception as e:
                print(f'[closed-loop] prompt controller parse failed: {e}')
        return {'decision': STOP, 'source': 'fallback_parse_error',
                'reason': 'could not parse decision JSON', 'parse_ok': False, 'raw': last_raw}


# SFT controller prompt: MUST stay byte-identical to CONTROLLER_SYSTEM /
# CONTROLLER_TEMPLATE in build_controller_sft_data.py (the training data).
# Deliberately NOT shared with PROMPT_TEMPLATE above, which asks for a
# 'missing_evidence' field the SFT model was never trained to emit.
SFT_SYSTEM = (
    "You are a strict evidence auditor for retrieval-augmented question answering "
    "over slide decks. You decide whether the retrieved pages are sufficient to "
    "answer a question, or whether more pages must be retrieved."
)

SFT_TEMPLATE = (
    "Question: {question}\n\n"
    "A first-stage retriever selected the following {n} document page images as "
    "evidence for this question:\n"
    "{image_tokens}\n\n"
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


class SFTController:
    """SFT-trained STOP/RETRIEVE_MORE controller (Phase 2, task #64).

    Qwen2.5-VL-3B base + LoRA adapter trained on oracle labels derived from
    frozen-TFR-seeker first-round selections. Same decide() contract as
    PromptController; any parse/model failure degrades to STOP so a broken
    controller can never score below the no-loop baseline.
    """

    name = 'sft'

    def __init__(self, adapter_path=CONTROLLER_SFT_ADAPTER):
        self._llm = None
        self.adapter_path = adapter_path

    def _ensure_model(self):
        if self._llm is None:
            _dev = os.environ.get('CONTROLLER_DEVICE', None)
            print(f'[closed-loop] SFT controller loading adapter: {self.adapter_path} (device={_dev})')
            self._llm = LLM(model_name=self.adapter_path, device=_dev)
        return self._llm

    def decide(self, question, selected_pages, uid=None):
        images = list(selected_pages)[:PROMPT_MAX_IMAGES]
        if not images:
            return {'decision': RETRIEVE_MORE, 'source': 'sft',
                    'reason': 'no pages selected', 'parse_ok': True, 'raw': ''}
        llm = self._ensure_model()
        image_tokens = ' '.join(['<image>'] * len(images))
        prompt = SFT_TEMPLATE.format(question=question, n=len(images), image_tokens=image_tokens)
        last_raw = ''
        for attempt in range(2):
            try:
                raw = llm.generate(query=prompt, image=images, system_prompt=SFT_SYSTEM)
            except Exception as e:
                print(f'[closed-loop] sft controller generate failed: {e}')
                return {'decision': STOP, 'source': 'fallback_generate_error',
                        'reason': str(e), 'parse_ok': False, 'raw': ''}
            last_raw = raw
            try:
                obj = extract_json(raw)
                decision = str(obj.get('decision', '')).strip().upper().replace(' ', '_')
                if decision in (STOP, RETRIEVE_MORE):
                    return {'decision': decision, 'source': 'sft',
                            'parse_ok': True, 'raw': raw}
            except Exception as e:
                print(f'[closed-loop] sft controller parse failed: {e}')
        return {'decision': STOP, 'source': 'fallback_parse_error',
                'reason': 'could not parse decision JSON', 'parse_ok': False, 'raw': last_raw}


class SFTControllerV3:
    """v3 SFT controller (task #68): v1 recipe + a [Retrieval Confidence]
    fusion-rank feature block injected into the prompt.

    The prompt template and confidence-block formatting are imported from
    build_controller_sft_data_v3.py (single source of truth, byte-identical
    to the training data). Fusion ranks of the shown pages come from the
    RankCache (recovery_ceiling_ranks.jsonl), capped at top-20 to match the
    training cache. Any parse/model failure degrades to STOP.
    """

    name = 'sft_v3'

    def __init__(self, adapter_path=CONTROLLER_SFT_ADAPTER, rank_cache=None, shared_llm=None):
        self._llm = shared_llm  # may be None → lazy load in _ensure_model()
        self.adapter_path = adapter_path
        self.rank_cache = rank_cache

    def _ensure_model(self):
        if self._llm is None:
            _dev = os.environ.get('CONTROLLER_DEVICE', None)
            print(f'[closed-loop] SFT-v3 controller loading adapter: {self.adapter_path} (device={_dev})')
            self._llm = LLM(model_name=self.adapter_path, device=_dev)
        return self._llm

    def decide(self, question, selected_pages, uid=None):
        images = list(selected_pages)[:PROMPT_MAX_IMAGES]
        if not images:
            return {'decision': RETRIEVE_MORE, 'source': 'sft_v3',
                    'reason': 'no pages selected', 'parse_ok': True, 'raw': ''}
        from build_controller_sft_data_v3 import (  # lazy: keep v1/v2 paths import-free
            CONTROLLER_TEMPLATE_V3, build_confidence_block)
        page_ids = [_page_id_of(p) for p in images]
        ranking = []
        if self.rank_cache is not None and uid is not None:
            rec = self.rank_cache.get(uid)
            ranking = (rec or {}).get('ranking') or []
        confidence = build_confidence_block(page_ids, ranking)
        llm = self._ensure_model()
        image_tokens = ' '.join(['<image>'] * len(images))
        prompt = CONTROLLER_TEMPLATE_V3.format(
            question=question, n=len(images), image_tokens=image_tokens,
            confidence=confidence)
        last_raw = ''
        for attempt in range(2):
            try:
                raw = llm.generate(query=prompt, image=images, system_prompt=SFT_SYSTEM)
            except Exception as e:
                print(f'[closed-loop] sft_v3 controller generate failed: {e}')
                return {'decision': STOP, 'source': 'fallback_generate_error',
                        'reason': str(e), 'parse_ok': False, 'raw': ''}
            last_raw = raw
            try:
                obj = extract_json(raw)
                decision = str(obj.get('decision', '')).strip().upper().replace(' ', '_')
                if decision in (STOP, RETRIEVE_MORE):
                    return {'decision': decision, 'source': 'sft_v3',
                            'parse_ok': True, 'raw': raw}
            except Exception as e:
                print(f'[closed-loop] sft_v3 controller parse failed: {e}')
        return {'decision': STOP, 'source': 'fallback_parse_error',
                'reason': 'could not parse decision JSON', 'parse_ok': False, 'raw': last_raw}


class SFTThreshController:
    """Thresholded SFT controller (task #66 v2a).

    Same adapter/prompt as SFTController, but the decision comes from the
    calibrated probability P(RETRIEVE_MORE): one forward pass over the exact
    training-format input plus the shared response prefix '{"decision": "',
    then softmax mass of the first STOP token vs the first RETRIEVE_MORE
    token. Trigger iff p >= tau (env CONTROLLER_TAU, default 0.5 which
    reproduces greedy-argmax behaviour). Any failure degrades to STOP.
    """

    name = 'sft_thresh'
    RESP_PREFIX = '{"decision": "'

    def __init__(self, adapter_path=CONTROLLER_SFT_ADAPTER, tau=CONTROLLER_TAU):
        self._llm = None
        self.adapter_path = adapter_path
        self.tau = tau

    def _ensure_model(self):
        if self._llm is None:
            import torch  # noqa: F401
            print(f'[closed-loop] sft_thresh controller loading adapter: '
                  f'{self.adapter_path} (tau={self.tau})')
            self._llm = LLM(model_name=self.adapter_path)
            tvm = self._llm.model
            self._model = tvm.model
            self._processor = tvm.processor
            self._model.eval()
            tok = self._processor.tokenizer
            self._tok_stop = tok.encode('STOP', add_special_tokens=False)[0]
            self._tok_retr = tok.encode('RETRIEVE_MORE', add_special_tokens=False)[0]
            self._prefix_ids = tok.encode(self.RESP_PREFIX, add_special_tokens=False)
        return self._llm

    def score(self, question, images, uid=None):
        """P(RETRIEVE_MORE) in [0,1] for this (question, page images) pair."""
        image_tokens = ' '.join(['<image>'] * len(images))
        instruction = SFT_TEMPLATE.format(
            question=question, n=len(images), image_tokens=image_tokens)
        return self._score_instruction(instruction, images)

    def _score_instruction(self, instruction, images):
        """Calibrated p over one forward pass of the given user instruction."""
        import torch
        from qwen_vl_utils import process_vision_info
        self._ensure_model()
        content, img_idx = [], 0
        parts = instruction.split('<image>')
        for k, part in enumerate(parts):
            if part:
                content.append({'type': 'text', 'text': part})
            if k < len(parts) - 1 and img_idx < len(images):
                content.append({'type': 'image', 'image': images[img_idx]})
                img_idx += 1
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SFT_SYSTEM}]},
            {'role': 'user', 'content': content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors='pt')
        input_ids = torch.cat([
            inputs['input_ids'],
            torch.tensor([self._prefix_ids], dtype=inputs['input_ids'].dtype)], dim=1)
        attention_mask = torch.cat([
            inputs['attention_mask'],
            torch.ones((1, len(self._prefix_ids)), dtype=inputs['attention_mask'].dtype)], dim=1)
        fwd = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        fwd['input_ids'] = input_ids
        fwd['attention_mask'] = attention_mask
        if 'mm_token_type_ids' in fwd:
            fwd['mm_token_type_ids'] = torch.cat([
                fwd['mm_token_type_ids'],
                torch.zeros((1, len(self._prefix_ids)), dtype=fwd['mm_token_type_ids'].dtype)], dim=1)
        fwd = {k: (v.to(self._model.device) if torch.is_tensor(v) else v)
               for k, v in fwd.items()}
        with torch.no_grad():
            logits = self._model(**fwd).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_stop = float(probs[self._tok_stop])
        p_retr = float(probs[self._tok_retr])
        denom = p_stop + p_retr
        return p_retr / denom if denom > 0 else 0.0

    def decide(self, question, selected_pages, uid=None):
        images = list(selected_pages)[:PROMPT_MAX_IMAGES]
        if not images:
            return {'decision': RETRIEVE_MORE, 'source': 'sft_thresh',
                    'reason': 'no pages selected', 'parse_ok': True, 'raw': '',
                    'score': 1.0, 'tau': self.tau}
        try:
            s = self.score(question, images, uid=uid)
        except Exception as e:
            print(f'[closed-loop] sft_thresh scoring failed: {e}')
            return {'decision': STOP, 'source': 'fallback_generate_error',
                    'reason': str(e), 'parse_ok': False, 'raw': '', 'tau': self.tau}
        decision = RETRIEVE_MORE if s >= self.tau else STOP
        return {'decision': decision, 'source': self.name, 'parse_ok': True,
                'raw': '', 'score': round(s, 6), 'tau': self.tau}


class SFTThreshV3Controller(SFTThreshController):
    """Thresholded V3 controller (task #73).

    Same calibrated-probability mechanism as SFTThreshController, but the
    forward pass uses CONTROLLER_TEMPLATE_V3 + the fusion-rank confidence
    block imported from build_controller_sft_data_v3.py - byte-identical to
    the SFT-v3/GRPO training format. Thresholding a v3-trained adapter with
    the v1 prompt ('sft_thresh') would score it out-of-distribution.

    Task 73 runs this with CONTROLLER_TAU=0.0 (fire on every sample) so ONE
    eval records the per-sample probability AND the loop outcome for every
    potential trigger; any tau > 0 is then reconstructed offline against the
    no-loop baseline (samples are independent of each other's decisions).
    """

    name = 'sft_thresh_v3'

    def __init__(self, adapter_path=CONTROLLER_SFT_ADAPTER, tau=CONTROLLER_TAU,
                 rank_cache=None):
        super().__init__(adapter_path, tau)
        self.rank_cache = rank_cache

    def score(self, question, images, uid=None):
        from build_controller_sft_data_v3 import (  # lazy: keep v1/v2 paths import-free
            CONTROLLER_TEMPLATE_V3, build_confidence_block)
        page_ids = [_page_id_of(p) for p in images]
        ranking = []
        if self.rank_cache is not None and uid is not None:
            rec = self.rank_cache.get(uid)
            ranking = (rec or {}).get('ranking') or []
        confidence = build_confidence_block(page_ids, ranking)
        image_tokens = ' '.join(['<image>'] * len(images))
        instruction = CONTROLLER_TEMPLATE_V3.format(
            question=question, n=len(images), image_tokens=image_tokens,
            confidence=confidence)
        return self._score_instruction(instruction, images)


# ---------------------------------------------------------------------------
# Closed-loop orchestration (mirrors ViDoRAG_Agents.run_agent downstream)
# ---------------------------------------------------------------------------

def run_closed_loop(agents, controller, rank_cache, uid, query, first_pool_images, img_dir):
    """Full closed-loop pipeline for one query.

    Returns (answer, trace). Everything after the controller decision is the
    same code path as ViDoRAG_Agents.run_agent (inspector loop -> synthesizer).
    """
    agents.seeker.buffer_images = None
    agents.inspector.buffer_images = []

    trace = {
        'candidate_images': list(first_pool_images),   # first-round pool (baseline semantics)
        'seeker_selected_images': [],                  # FINAL selection (set below)
        'final_evidence_images': [],
        'iterations': 0,
        'termination': None,
    }
    cl = {
        'mode': CONTROLLER_MODE,
        'reselect_mode': RETRIEVE_MODE,
        'fire_mode': FIRE_MODE,
        'new_cand_ranks': [NEW_CAND_START, NEW_CAND_END],
        'trigger_rank_limit': TRIGGER_RANK_LIMIT,
        'reselect_pool_n': RESELECT_POOL_N,
        'decision': None,
        'decision_detail': None,
        'triggered': False,
        'first_round_selected': [],
        'reselect_pool': [],
        'reselect_selected': [],
        'oracle_decision': None,
        'oracle_detail': None,
    }

    # ---- round 1: seeker selects from the GMM pool (baseline pool size) ----
    timing, clock = {}, [_now() if TIMING else None]

    def lap(stage):
        if TIMING:
            now = _now()
            timing[stage] = now - clock[0]
            clock[0] = now

    selected_images, summary, reason = agents.seeker.run(query=query, images_path=list(first_pool_images))
    first_selected = list(selected_images)
    cl['first_round_selected'] = [_page_id_of(p) for p in first_selected]
    lap('seeker')

    # ---- controller decision ----
    detail = controller.decide(query, first_selected, uid=uid)
    cl['decision'] = detail['decision']
    cl['decision_detail'] = detail
    lap('controller')

    # Oracle decision is ALWAYS recorded (cheap, no model) so PROMPT runs can
    # be scored against the oracle trigger rule without re-deriving anything.
    oracle_detail = (detail if controller.name == 'oracle'
                     else OracleController(rank_cache).decide(query, first_selected, uid=uid))
    cl['oracle_decision'] = oracle_detail['decision']
    cl['oracle_detail'] = oracle_detail

    # ---- optional reselection from the extended pool ----
    if detail['decision'] == RETRIEVE_MORE:
        first_pool_page_ids = [_page_id_of(p) for p in first_pool_images]
        first_selected_ids = [_page_id_of(p) for p in first_selected]
        if RETRIEVE_MODE == 'new_only':
            # Only the 10 fresh pages at fusion ranks 11-20 (seeker's native scale).
            pool_ids = rank_cache.new_candidate_ids(uid)
            truncated_from = len(pool_ids)
            gold = rank_cache.gold_ids(uid)
        elif RETRIEVE_MODE == 'keep_plus_new':
            # Round-1 seeker picks (context) + fusion ranks 11-20, deduped.
            pool_ids = rank_cache.keep_plus_new_pool_ids(uid, first_selected_ids)
            truncated_from = len(pool_ids)
            gold = rank_cache.gold_ids(uid)
        else:
            # merged: rank-cache top-N (fusion-ranked, no noisy rank 11-40).
            # Previous approach (pool U ranks 1..40, cut to N) introduced noise
            # that confused the Seeker into selecting worse pages.
            rec = rank_cache.by_uid.get(uid)
            if rec is not None:
                pool_ids = rec['ranking'][:RESELECT_POOL_N]
                gold = rank_cache.gold_ids(uid)
                truncated_from = len(rec['ranking'])
            else:
                # Fallback: original merged behaviour when no cache entry
                ext_ids, gold = rank_cache.extended_pool_ids(uid, first_pool_page_ids)
                pool_ids = ext_ids[:RESELECT_POOL_N]
                truncated_from = len(ext_ids)
        reselect_pool = []
        missing_files = []
        for pid in pool_ids:
            path = os.path.join(img_dir, pid + '.jpg')
            if os.path.exists(path):
                reselect_pool.append(path)
            else:
                missing_files.append(pid)
        cl['reselect_pool'] = [_page_id_of(p) for p in reselect_pool]
        cl['reselect_pool_truncated_from'] = truncated_from
        cl['reselect_pool_missing_files'] = missing_files
        cl['gold'] = sorted(gold)
        cl['gold_in_reselect_pool'] = sorted(gold & set(cl['reselect_pool']))

        if len(reselect_pool) <= 1:
            print('[closed-loop] extended pool degenerate, keeping first-round selection')
        else:
            # Fresh reselection: seeker sees the extended pool with the same
            # generation settings; temporarily raise the candidate cap to the
            # pool size (vidorag_agents reads SEEKER_V4_MAX_CANDIDATES at call
            # time; module default 10 would silently cut the pool back).
            prev = os.environ.get('SEEKER_V4_MAX_CANDIDATES')
            os.environ['SEEKER_V4_MAX_CANDIDATES'] = str(len(reselect_pool))
            try:
                selected_images, summary, reason = agents.seeker.run(query=query, images_path=list(reselect_pool))
            finally:
                if prev is None:
                    os.environ.pop('SEEKER_V4_MAX_CANDIDATES', None)
                else:
                    os.environ['SEEKER_V4_MAX_CANDIDATES'] = prev
            cl['triggered'] = True
            cl['reselect_selected'] = [_page_id_of(p) for p in selected_images]
            if FIRE_MODE == 'union':
                # Keep what round 1 found: an additive fire can never
                # discard evidence the seeker had already selected.
                seen = set(first_selected)
                selected_images = list(first_selected) + [
                    p for p in selected_images if p not in seen]
            cl['evidence_after_fire'] = [_page_id_of(p) for p in selected_images]

    # ---- optional: same-deck second hop ----
    # Skipped when the controller already re-selected, so the two mechanisms
    # never stack on one question. A controller that reports an 'action'
    # (Inspector v2) runs it only when that action is DECK.
    lap('other')   # oracle bookkeeping + RETRIEVE_MORE path
    if HOP2_MODE == 'deck' and not cl['triggered'] and len(first_selected) >= HOP2_MIN_SEL \
            and detail.get('action', 'DECK') == 'DECK':
        first_selected_ids = [_page_id_of(p) for p in first_selected]
        rec = rank_cache.by_uid.get(uid)
        cand_ids = hop2_deck_candidates((rec or {}).get('ranking'), first_selected_ids, HOP2_K)
        cand_paths = [os.path.join(img_dir, pid + '.jpg') for pid in cand_ids]
        cand_paths = [p for p in cand_paths if os.path.exists(p)]
        hop2 = {'k': HOP2_K, 'fire': HOP2_FIRE, 'candidates': [_page_id_of(p) for p in cand_paths]}
        if cand_paths:
            pool2 = list(first_selected) + cand_paths
            prev = os.environ.get('SEEKER_V4_MAX_CANDIDATES')
            os.environ['SEEKER_V4_MAX_CANDIDATES'] = str(max(len(pool2), 10))
            try:
                hop2_sel, hop2_summary, _reason = agents.seeker.run(query=query, images_path=pool2)
            finally:
                if prev is None:
                    os.environ.pop('SEEKER_V4_MAX_CANDIDATES', None)
                else:
                    os.environ['SEEKER_V4_MAX_CANDIDATES'] = prev
            hop2['selected'] = [_page_id_of(p) for p in hop2_sel]
            if HOP2_FIRE == 'union':
                seen = set(first_selected)
                new_sel = list(first_selected) + [p for p in hop2_sel if p not in seen]
            else:
                new_sel = list(hop2_sel) or list(first_selected)
            if new_sel != list(first_selected):
                selected_images = new_sel
                summary = hop2_summary
            hop2['evidence'] = [_page_id_of(p) for p in selected_images]
            print(f'[hop2] {len(first_selected)} sel + {len(cand_paths)} same-deck cand '
                  f'-> {len(hop2_sel)} picked -> {len(selected_images)} page(s)')
        cl['hop2'] = hop2

    # ---- optional: rank-1 rescue ----
    if RANK1_RESCUE != 'off' and first_pool_images:
        top = first_pool_images[0]
        if _page_id_of(top) not in {_page_id_of(p) for p in selected_images}:
            if RANK1_RESCUE == 'prepend':
                selected_images = [top] + list(selected_images)
            else:
                selected_images = list(selected_images) + [top]
            cl['rank1_rescue'] = _page_id_of(top)

    trace['seeker_selected_images'] = list(selected_images)
    lap('hop2')    # DECK candidates + Seeker re-selection + rank-1 rescue

    # ---- optional: cap the evidence handed to the downstream agents ----
    # Measured on the fixed-pool run: when the top-ranked page already holds the
    # answer, handing the synthesizer a second page costs ~1.0 judge point
    # (gold@rank1: 1 page 4.157 vs 2 pages 3.154). This flag tests that dilution
    # directly by truncating to the seeker's first K picks. Off by default.
    _evidence_topk = int(os.environ.get('CLOSED_LOOP_EVIDENCE_TOPK', '0') or 0)
    if _evidence_topk > 0 and len(selected_images) > _evidence_topk:
        cl['evidence_topk_truncated_from'] = len(selected_images)
        selected_images = list(selected_images[:_evidence_topk])
        cl['evidence_after_topk'] = [_page_id_of(p) for p in selected_images]

    # ---- downstream pipeline: identical to ViDoRAG_Agents.run_agent ----
    answer = None
    max_iterations = 5
    iteration = 0
    if SKIP_PASSTHROUGH_INSPECTOR:
        trace['iterations'] = 1
        _reason, answer = agents.synthesizer.run(query, None, list(selected_images))
        trace['final_evidence_images'] = list(selected_images)
        trace['termination'] = 'synthesizer'
    while not SKIP_PASSTHROUGH_INSPECTOR:
        iteration += 1
        trace['iterations'] = iteration
        if iteration > max_iterations:
            print(f'[closed-loop] Max iterations ({max_iterations}) reached, forcing synthesizer')
            evidence = agents.inspector.buffer_images if agents.inspector.buffer_images else selected_images
            _reason, answer = agents.synthesizer.run(query, summary, evidence)
            trace['final_evidence_images'] = list(evidence)
            trace['termination'] = 'max_iterations'
            break

        status, information, images = agents.inspector.run(query, selected_images)
        if status == 'answer':
            trace['final_evidence_images'] = list(agents.inspector.buffer_images)
            trace['termination'] = 'inspector_answer'
            answer = information
            break
        elif status == 'synthesizer':
            _reason, answer = agents.synthesizer.run(query, information, images)
            trace['final_evidence_images'] = list(images or [])
            trace['termination'] = 'synthesizer'
            break
        elif status == 'seeker':
            selected_images, summary, reason = agents.seeker.run(feedback=information)
            trace['seeker_selected_images'].extend(
                image for image in selected_images
                if image not in trace['seeker_selected_images']
            )
            continue
        else:
            print('No related information')
            trace['termination'] = 'no_information'
            answer = None
            break

    # bookkeeping for the analysis script
    lap('synth')
    if TIMING:
        timing['total'] = sum(timing.values())
        cl['timing'] = timing
    cl['gold_in_first_selected'] = oracle_detail.get('gold_in_first_selected')
    cl['best_gold_rank'] = oracle_detail.get('best_gold_rank')
    return answer, trace, cl


# ---------------------------------------------------------------------------
# MMRAG subclass: closed-loop variant of the 'vidorag' eval function
# ---------------------------------------------------------------------------

class ClosedLoopMMRAG(MMRAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rank_cache = RankCache()
        # CLOSED_LOOP_POOL_FILE (timing runs): take each question's round-1 pool from a
        # previous run's JSONL (agent_trace.candidate_images) instead of the retrieval
        # server, so agent latency is measured without retrieval-server contention.
        self.pool_cache = None
        if os.environ.get('CLOSED_LOOP_POOL_FILE'):
            with open(os.environ['CLOSED_LOOP_POOL_FILE']) as f:
                self.pool_cache = {r['uid']: [os.path.basename(p) for p in r['agent_trace']['candidate_images']]
                                   for r in map(json.loads, f) if r.get('agent_trace')}
            print(f'[closed-loop] round-1 pools from {os.environ["CLOSED_LOOP_POOL_FILE"]} '
                  f'({len(self.pool_cache)} questions; retrieval server not queried)')

        # Memory-efficient controller: share the base model with agents.
        # Skip sharing when CONTROLLER_DEVICE=cpu (separate model on CPU avoids
        # adapter-switching CUDA errors and leaves more GPU memory for agents).
        shared_controller_llm = None
        _ctrl_device = os.environ.get('CONTROLLER_DEVICE', None)
        if _ctrl_device == 'cpu':
            print(f'[closed-loop] Controller will run on CPU (separate model)')
        elif CONTROLLER_MODE == 'sft_v3':   # the only mode that uses shared_controller_llm
            try:
                seeker_vlm = getattr(self.agents, 'seeker', None)
                if seeker_vlm and hasattr(seeker_vlm.vlm, '_group'):
                    shared_group = seeker_vlm.vlm._group
                    peft_model = shared_group._peft_model
                    adapter_name = 'controller'
                    if adapter_name not in shared_group._adapter_names:
                        print(f'[closed-loop] Loading controller adapter into shared model: {CONTROLLER_SFT_ADAPTER}')
                        peft_model.load_adapter(CONTROLLER_SFT_ADAPTER, adapter_name=adapter_name)
                        shared_group._adapter_names.append(adapter_name)
                        shared_group._adapter_paths[adapter_name] = CONTROLLER_SFT_ADAPTER
                    shared_controller_llm = shared_group.get_agent(adapter_name)
                    print(f'[closed-loop] Controller shares base model with agents (memory-efficient)')
            except Exception as e:
                print(f'[closed-loop] Shared controller setup failed ({e}), falling back to separate model')
                shared_controller_llm = None

        if CONTROLLER_MODE == 'oracle':
            self.controller = OracleController(self.rank_cache)
        elif CONTROLLER_MODE == 'prompt':
            self.controller = PromptController(BASE_MODEL_3B)
        elif CONTROLLER_MODE == 'sft':
            self.controller = SFTController(CONTROLLER_SFT_ADAPTER)
        elif CONTROLLER_MODE == 'sft_v3':
            self.controller = SFTControllerV3(CONTROLLER_SFT_ADAPTER, self.rank_cache,
                                              shared_llm=shared_controller_llm)
        elif CONTROLLER_MODE == 'sft_thresh':
            self.controller = SFTThreshController(CONTROLLER_SFT_ADAPTER, CONTROLLER_TAU)
        elif CONTROLLER_MODE == 'sft_thresh_v3':
            self.controller = SFTThreshV3Controller(
                CONTROLLER_SFT_ADAPTER, CONTROLLER_TAU, self.rank_cache)
        elif CONTROLLER_MODE == 'always_stop':
            self.controller = AlwaysStopController()
        elif CONTROLLER_MODE == 'inspector_v2':
            group = self.agents.seeker.vlm._group   # shared base model (required)
            self.controller = InspectorV2Controller(
                os.environ['INSPECTOR_V2_ADAPTER'],
                os.environ.get('INSPECTOR_V2_ACTIONS', 'STOP,DECK,EXPAND'),
                self.rank_cache, group._peft_model, group.processor,
                os.path.normpath(os.path.join(PROJECT_ROOT, self.img_dir)))
            print(f'[closed-loop] Inspector v2 adapter: {self.controller.adapter_path} '
                  f'actions={self.controller.scorer.actions} biases={self.controller.biases} '
                  f'hop2={HOP2_MODE} min_sel={HOP2_MIN_SEL} '
                  f'fire={FIRE_MODE} r1={RANK1_RESCUE}')
        else:
            raise ValueError(f'unknown CONTROLLER_MODE: {CONTROLLER_MODE}')
        print(f'[closed-loop] controller={self.controller.name}  '
              f'trigger_rank<={TRIGGER_RANK_LIMIT}  reselect_pool_n={RESELECT_POOL_N}  '
              f'retrieve_mode={RETRIEVE_MODE}'
              + (f'  new_cand_ranks={NEW_CAND_START}-{NEW_CAND_END}'
                 if RETRIEVE_MODE != 'merged' else ''))

    def vidorag(self, sample):
        """Closed-loop variant of MMRAG.vidorag. First-round pool comes from the
        SAME retrieval call as the baseline (HTTP server, GMM on); the loop only
        intervenes between seeker selection and the downstream pipeline."""
        query = sample['query']
        print(query)
        t_retrieval = _now() if TIMING else None
        if self.pool_cache is not None:
            candidate_image = [os.path.join(self.img_dir, b) for b in self.pool_cache[sample['uid']]]
        else:
            recall_results = self.search_engine.search(query)
            candidate_image = [
                os.path.join(self.img_dir,
                             os.path.basename(node['node']['metadata']
                                              .get('file_name', node['node']['metadata'].get('filename'))
                                              .replace('.txt', '.jpg')))
                for node in recall_results['source_nodes']
            ]
        if TIMING:
            t_retrieval = _now() - t_retrieval
        if 'gmm' not in self.experiment_type:
            candidate_image = candidate_image[:self.top_k]
        try:
            answer, agent_trace, cl_trace = run_closed_loop(
                self.agents, self.controller, self.rank_cache,
                sample['uid'], query, candidate_image, self.img_dir)
        except RuntimeError as e:
            # Re-raise CUDA errors so eval.py retry logic can handle them
            if 'CUDA' in str(e) or 'cuda' in str(e) or 'CUDACachingAllocator' in str(e):
                raise
            print(f'[Agent Error] {e}')
            return None
        except Exception as e:
            print(f'[Agent Error] {e}')
            return None

        if answer is None:
            print('[Agent] No answer produced')
            return None

        try:
            if SKIP_JUDGE:
                sample['eval_result'] = dict(score=0, passing=0, judge='skipped (CLOSED_LOOP_SKIP_JUDGE=1)')
            else:
                sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        except JudgeUnavailableError:
            raise
        except Exception as e:
            print(f'[Evaluator Error] {e}')
            sample['eval_result'] = dict(score=0, passing=0, judge=f'Evaluator failed: {e}')

        if TIMING:
            cl_trace['timing']['retrieval'] = t_retrieval
            cl_trace['timing']['e2e'] = t_retrieval + cl_trace['timing']['total']

        sample['response'] = str(answer)
        sample['agent_trace'] = agent_trace
        sample['closed_loop'] = cl_trace
        sample['closed_loop']['final_score'] = sample['eval_result'].get('score')
        sample['closed_loop']['final_passing'] = sample['eval_result'].get('passing')
        sample['recall_results'] = dict(
            source_nodes=[NodeWithScore(node=ImageNode(image_path=image, metadata=dict(file_name=image)),
                                        score=None).to_dict() for image in candidate_image],
            response=None,
            metadata=None)
        selected_images = agent_trace.get('seeker_selected_images', [])
        sample['seeker_results'] = dict(
            source_nodes=[NodeWithScore(node=ImageNode(image_path=image, metadata=dict(file_name=image)),
                                        score=None).to_dict() for image in selected_images],
            response=None,
            metadata=None,
        )
        return sample


def main():
    ap = argparse.ArgumentParser(description='Closed-loop (STOP/RETRIEVE_MORE) inference eval')
    ap.add_argument('--mode', default=os.environ.get('EVAL_MODE_NAME', 'tfr_v4fix_closedloop_300'),
                    help='output mode name (controller suffix should be included)')
    ap.add_argument('--max_samples', type=int, default=300)
    ap.add_argument('--shard_id', type=int, default=None, help='shard index for parallel eval')
    ap.add_argument('--num_shards', type=int, default=None, help='total number of shards')
    args = ap.parse_args()

    seeker_path = os.environ.get('EVAL_SEEKER_PATH') or None
    synth_path = os.environ.get('EVAL_SYNTH_PATH') or None
    base_model = os.environ.get('EVAL_BASE_MODEL') or BASE_MODEL_3B
    inspector_path = os.environ.get('EVAL_INSPECTOR_PATH') or None
    if inspector_path is None and SKIP_PASSTHROUGH_INSPECTOR:
        # The pass-through Inspector is never called in this mode; reuse the
        # Synthesizer adapter to satisfy the three-adapter shared-base loader.
        inspector_path = synth_path
    if not (seeker_path and synth_path and inspector_path):
        raise SystemExit('EVAL_SEEKER_PATH / EVAL_SYNTH_PATH / EVAL_INSPECTOR_PATH must all be set')

    if not os.environ.get('EVAL_RETRIEVAL_SERVER'):
        raise SystemExit('EVAL_RETRIEVAL_SERVER must be set (closed loop reuses the baseline '
                         'first-round pool from the resident hybrid server)')

    _ds_prefix = DATASET.lower() if DATASET != 'SlideVQA' else 'slidevqa'
    output_name = f'{_ds_prefix}_{args.mode}_hybrid_{_ds_prefix}'
    os.makedirs(EVAL_DIR, exist_ok=True)
    if os.environ.get('EVAL_NO_CLEAN') != '1':
        for suffix in ['.jsonl', '_eval.json', '_eval_type_wise.json']:
            path = os.path.join(EVAL_DIR, f'{output_name}{suffix}')
            if os.path.exists(path):
                os.remove(path)
                print(f'  [Clean] Removed existing {path}')
    else:
        print('  [Clean] EVAL_NO_CLEAN=1 - resume mode')

    kwargs = dict(
        dataset=DATASET,
        query_file=QUERY_FILE,
        experiment_type='vidorag',
        generate_vlm=base_model,
        max_samples=args.max_samples,
        embed_model_name=EMBED_MODEL,
        embed_model_name_vl=COLQWEN_MODEL,
        embed_model_name_text=os.environ.get('EVAL_EMBED_MODEL', EMBED_MODEL),
        topk=20,
        vl_rerank=False,
        seeker_model_path=seeker_path,
        synthesizer_model_path=synth_path,
        inspector_model_path=inspector_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    mmrag = ClosedLoopMMRAG(**kwargs)
    _shard_suffix = f'_shard{args.shard_id}' if args.shard_id is not None else ''
    mmrag.output_file_path = os.path.join(EVAL_DIR, f'{output_name}{_shard_suffix}.jsonl')

    mmrag.eval_dataset()
    mmrag.eval_overall()
    mmrag.eval_overall_type_wise()
    print(f'\n[Done] closed-loop results:')
    print(f'  Samples: {EVAL_DIR}/{output_name}.jsonl')
    print(f'  Eval:    {EVAL_DIR}/{output_name}_eval.json')


if __name__ == '__main__':
    main()
