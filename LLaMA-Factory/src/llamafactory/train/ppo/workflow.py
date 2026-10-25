# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from trl.core import PPODecorators
from trl.models.utils import unwrap_model_for_generation

from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer_qr_s_g import CustomPPOTrainer_QSG


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reward Method Switch (original ViDoRAG vs V-MMOA paper)
# ---------------------------------------------------------------------------
# Selects the reward formulation used by per-agent PPO.  Mirrors the switch in
# ``vidorag_rewards.py`` so that per-agent and MAPPO training stay consistent.
#
#   "original" : ViDoRAG per-agent rewards (default).
#                  seeker       -> page-selection F1 + format penalty
#                  inspector    -> INSPECTOR_REWARD_MODE (contain/f1/exact/combined)
#                  synthesizer  -> token F1
#
#   "vmmoa"    : V-MMOA paper (Anonymous EMNLP Submission, Sec. 4.2) agent-local
#                shaping signals, matching the dense terms of Eq. 1:
#                  seeker       -> R_ret = recall@k (pure recall vs gold pages)
#                  inspector    -> R_ins = precision of accepted evidence vs gold
#                                  pages (falls back to answer F1 when the
#                                  inspector emits an answer rather than pages)
#                  synthesizer  -> R_ans = answer-level F1 (global terminal term)
#
# Per-agent training has no retrieval loop, so the global answer reward R_ans and
# the iteration-cost term C(τ) are only applied in the joint MAPPO trainer; here
# each agent is trained on its own local signal.
#
# Toggle at runtime via the environment variable:
#     export VIDORAG_REWARD_METHOD=vmmoa   # or "original" (default)
REWARD_METHOD = os.environ.get("VIDORAG_REWARD_METHOD", "original").lower()


# ---------------------------------------------------------------------------
# Per-Agent PPO Reward Functions
# ---------------------------------------------------------------------------

def compute_seeker_evidence_reward(
    predicted_text: str, gold_pages: List[int], num_candidates: int, k_budget: int = 6
) -> float:
    """Evidence-level reward for Seeker PPO (diagnosis Sec. 5, Priority 2).

    R_seeker = 1.0 * Recall(gold, pred)
             + 0.3 * Precision(gold, pred)
             - 0.05 * max(0, selected_count - k_budget)
             - 0.3 * missing_any_gold
             - 0.2 * invalid_json

    Recall is weighted 3x more than precision because missing a gold page
    (low recall) is costlier than selecting an extra non-gold page (lower
    precision) — the Inspector/Synthesizer can filter non-gold pages, but
    cannot recover gold pages the Seeker missed.

    The ``missing_any_gold`` penalty (-0.3) is a heavy penalty applied when
    the Seeker fails to select ANY gold page.  This directly targets the
    multi-page/multi-hop failure mode identified in the diagnosis: 66/300
    test queries are multi-page gold, and the Seeker frequently selects only
    one of multiple required pages.  Without this penalty, partial recall
    (hitting 1 of 2 gold pages) yields reward=0.5, which is not punitive
    enough for incomplete evidence.

    The over-selection penalty discourages selecting all candidates to
    trivially maximize recall (reward hacking prevention).
    """
    pred_pages: List[int] = []
    valid_json = True

    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict):
            pred_pages = parsed.get("choice", []) or []
            if not isinstance(pred_pages, list):
                pred_pages = [pred_pages]
            pred_pages = [int(p) for p in pred_pages if isinstance(p, (int, float, str)) and str(p).strip().isdigit()]
        elif isinstance(parsed, list):
            pred_pages = [int(x) for x in parsed if isinstance(x, (int, float))]
        else:
            valid_json = False
    except (json.JSONDecodeError, TypeError, ValueError):
        valid_json = False
        numbers = re.findall(r"\d+", predicted_text)
        pred_pages = [int(n) for n in numbers if int(n) < num_candidates]

    # Filter to valid range
    pred_pages = [p for p in pred_pages if 0 <= p < num_candidates]

    if not gold_pages:
        return 0.0

    pred_set = set(pred_pages)
    gold_set = set(gold_pages)
    hits = len(pred_set & gold_set)

    recall = hits / max(len(gold_set), 1)
    precision = hits / max(len(pred_set), 1)

    # Over-selection penalty
    excess = max(0, len(pred_pages) - k_budget)
    over_selection_penalty = -0.05 * excess

    # Missing-any-gold penalty: heavy penalty if the Seeker fails to select
    # ALL gold pages.  This is critical for multi-page/multi-hop queries where
    # incomplete evidence leads to wrong answers (diagnosis Sec. 4.4).
    missing_any_gold = 1.0 if hits < len(gold_set) else 0.0
    missing_gold_penalty = -0.3 * missing_any_gold

    # Invalid JSON penalty
    invalid_json_penalty = -0.2 if not valid_json else 0.0

    reward = (
        1.0 * recall
        + 0.3 * precision
        + over_selection_penalty
        + missing_gold_penalty
        + invalid_json_penalty
    )
    return max(-1.0, min(1.0, reward))


# ---------------------------------------------------------------------------
# Strategy A: End-to-end answer-level reward (additive, no existing code changes)
# ---------------------------------------------------------------------------
# Loads a Synthesizer model and runs it on the Seeker's selected pages to
# produce an answer, then computes F1 against the gold answer.  This directly
# aligns the reward signal with the evaluation metric.
#
# Activated via:  export VIDORAG_REWARD_METHOD=e2e
# ---------------------------------------------------------------------------

_SYNTHESIZER_MODEL = None  # lazy-loaded global


def _get_synthesizer_model():
    """Lazy-load the Synthesizer model for e2e reward computation.

    Uses the base Qwen2.5-VL-3B model (no SFT adapter) by default, since the
    SFT v4 Synthesizer had poor results.  The base 3B model provides a more
    neutral evaluator that doesn't bias the reward toward SFT-specific output
    formats.

    Set VIDORAG_E2E_SYNTH_ADAPTER to use a specific SFT adapter instead.
    """
    global _SYNTHESIZER_MODEL
    if _SYNTHESIZER_MODEL is not None:
        return _SYNTHESIZER_MODEL

    import sys as _sys
    _vidorag_root = os.environ.get("VIDORAG_PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
    if _vidorag_root not in _sys.path:
        _sys.path.insert(0, _vidorag_root)

    from llms.llm import TrainedVLM, LocalVLM

    synth_adapter = os.environ.get("VIDORAG_E2E_SYNTH_ADAPTER", "")
    if synth_adapter:
        # Use a specific SFT adapter if explicitly provided
        logger.info(f"[E2E Reward] Loading Synthesizer SFT from: {synth_adapter}")
        _SYNTHESIZER_MODEL = TrainedVLM(adapter_path=synth_adapter)
    else:
        # Default: use base Qwen2.5-VL-3B model (no SFT) for neutral evaluation
        base_model_path = os.environ.get("VIDORAG_BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        logger.info(f"[E2E Reward] Loading base 3B model (no SFT) from: {base_model_path}")
        _SYNTHESIZER_MODEL = LocalVLM(model_path=base_model_path)

    logger.info("[E2E Reward] Synthesizer loaded successfully.")
    return _SYNTHESIZER_MODEL


def compute_seeker_e2e_reward(
    predicted_text: str,
    gold_pages: List[int],
    num_candidates: int,
    image_paths: List[str],
    question: str,
    gold_answer: str,
) -> float:
    """End-to-end reward for Seeker PPO (hybrid evidence + answer F1).

    R_e2e = w_evidence * full_evidence_reward + w_answer * answer_f1

    The evidence component uses the full compute_seeker_evidence_reward
    function, which includes 1.0*Recall + 0.3*Precision - 0.05*over_selection
    - 0.3*missing_any_gold - 0.2*invalid_json (clamped to [-1, 1]).
    The answer F1 is computed by running the Synthesizer on the Seeker's
    selected pages and comparing its answer to the gold.

    The invalid_json penalty is already included in evidence_reward and is
    not double-counted here.

    The weights w_evidence and w_answer are configurable via environment
    variables VIDORAG_E2E_W_EVIDENCE (default 0.4) and VIDORAG_E2E_W_ANSWER
    (default 0.6), allowing ablation studies (e.g., set w_evidence=0.0 to
    test pure answer-F1 reward).

    Args:
        predicted_text: Seeker's raw output text (JSON with choice array).
        gold_pages: Gold page indices (0-based relative to candidates).
        num_candidates: Total number of candidate pages.
        image_paths: List of all candidate image file paths.
        question: The question text.
        gold_answer: Gold answer string for F1 computation.
    """
    # --- Parse Seeker output ---
    pred_pages: List[int] = []
    valid_json = True
    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict):
            pred_pages = parsed.get("choice", []) or []
            if not isinstance(pred_pages, list):
                pred_pages = [pred_pages]
            pred_pages = [int(p) for p in pred_pages if isinstance(p, (int, float, str)) and str(p).strip().isdigit()]
        elif isinstance(parsed, list):
            pred_pages = [int(x) for x in parsed if isinstance(x, (int, float))]
        else:
            valid_json = False
    except (json.JSONDecodeError, TypeError, ValueError):
        valid_json = False
        numbers = re.findall(r"\d+", predicted_text)
        pred_pages = [int(n) for n in numbers if int(n) < num_candidates]

    # Filter to valid range
    pred_pages = [p for p in pred_pages if 0 <= p < num_candidates]

    if not gold_pages:
        return 0.0

    # --- Compute full evidence reward (with diagnosis-based penalties) ---
    # Uses the complete evidence-only reward function, which includes:
    # 1.0*Recall + 0.3*Precision - 0.05*over_selection - 0.3*missing_any_gold - 0.2*invalid_json
    # This replaces the simplified 1.0*recall + 0.3*precision that was previously used.
    evidence_reward = compute_seeker_evidence_reward(
        predicted_text, gold_pages, num_candidates
    )

    # --- Compute end-to-end answer F1 ---
    answer_f1 = 0.0
    if pred_pages and image_paths and gold_answer:
        # Select images corresponding to the Seeker's page choices
        selected_images = [image_paths[i] for i in pred_pages if i < len(image_paths)]

        if selected_images:
            try:
                synth = _get_synthesizer_model()
                synth_query = f"Question: {question}\n\nBased on the provided slide images, answer the question concisely."
                synth_response = synth.generate(synth_query, selected_images)

                # Extract answer from Synthesizer response
                synth_answer = synth_response.strip()
                try:
                    synth_parsed = json.loads(synth_response)
                    if isinstance(synth_parsed, dict) and "answer" in synth_parsed:
                        synth_answer = str(synth_parsed["answer"]).strip()
                except (json.JSONDecodeError, TypeError):
                    pass  # use raw response

                answer_f1 = compute_synthesizer_f1(synth_answer, gold_answer)
            except Exception as e:
                logger.warning(f"[E2E Reward] Synthesizer inference failed: {e}")
                answer_f1 = 0.0

    # --- Combine rewards ---
    # Uses the full evidence-only reward (with missing_any_gold and over_selection
    # penalties from diagnosis) as the evidence component, combined with answer F1.
    # The invalid_json penalty is already included in evidence_reward, so it is
    # not double-counted here.
    # Weights are configurable via env vars for ablation studies.
    # Default: w_evidence=0.4, w_answer=0.6
    # Ablation: set VIDORAG_E2E_W_EVIDENCE=0.0 for pure answer-F1 reward
    w_evidence = float(os.environ.get("VIDORAG_E2E_W_EVIDENCE", "0.4"))
    w_answer = float(os.environ.get("VIDORAG_E2E_W_ANSWER", "0.6"))
    reward = w_evidence * evidence_reward + w_answer * answer_f1
    return max(-1.0, min(1.0, reward))


def compute_seeker_page_f1(
    predicted_text: str, gold_pages: List[int], num_candidates: int
) -> float:
    """Compute F1 reward for seeker page selection."""
    # Extract page indices from predicted JSON or text
    pred_pages = []
    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict):
            pred_pages = parsed.get("choice", [])
        elif isinstance(parsed, list):
            # Model output a bare list like [1, 2, 3]
            pred_pages = [int(x) for x in parsed if isinstance(x, (int, float))]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: extract numbers from text
        numbers = re.findall(r"\d+", predicted_text)
        pred_pages = [int(n) for n in numbers if int(n) < num_candidates]

    if not gold_pages:
        return 0.0

    pred_set = set(pred_pages)
    gold_set = set(gold_pages)
    hits = len(pred_set & gold_set)
    precision = hits / max(len(pred_set), 1)
    recall = hits / max(len(gold_set), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


# Inspector reward mode switch (original method only): "contain" | "f1" | "exact" | "combined"
# Used only when VIDORAG_REWARD_METHOD=original.  Set at runtime without code
# edits via the environment variable, e.g.  export INSPECTOR_REWARD_MODE=f1
INSPECTOR_REWARD_MODE = os.environ.get("INSPECTOR_REWARD_MODE", "combined").lower()

def compute_inspector_reward(predicted_text: str, gold_answer: str, mode: str = None) -> float:
    """Compute reward for inspector based on answer quality.
    
    Supports multiple reward modes via the `mode` parameter:
      - "contain": Substring containment (default, best for short gold answers)
      - "f1": Token-level F1 score (good for longer answers)
      - "exact": Exact match (strictest)
      - "combined": contain(0.6) + f1(0.4) blend
    
    Args:
        predicted_text: Raw model output text
        gold_answer: Gold standard answer string
        mode: Reward computation strategy ("contain", "f1", "exact", "combined")
    """
    if mode is None:
        mode = INSPECTOR_REWARD_MODE

    # Try to extract answer from JSON output
    try:
        parsed = json.loads(predicted_text)
        pred_answer = str(parsed.get("answer", predicted_text)).strip()
    except (json.JSONDecodeError, TypeError):
        pred_answer = predicted_text.strip()
    
    if not gold_answer:
        # No gold answer available - give small format-based reward
        try:
            json.loads(predicted_text)
            return 0.3  # Valid JSON format
        except:
            return 0.0
    
    gold_str = str(gold_answer).strip().lower()
    pred_str = pred_answer.lower()
    
    if mode == "contain":
        # Substring containment: checks if gold answer appears in prediction
        # Best for short answers like "4", "yes", "2481"
        if gold_str in pred_str:
            return 1.0
        # Partial credit: check if any gold token is in prediction
        gold_tokens = gold_str.split()
        hits = sum(1 for t in gold_tokens if t in pred_str)
        return hits / max(len(gold_tokens), 1) * 0.5
    
    elif mode == "exact":
        # Exact match (after normalization)
        return 1.0 if gold_str == pred_str else 0.0
    
    elif mode == "f1":
        # Token-level F1
        pred_tokens = pred_str.split()
        gold_tokens = gold_str.split()
        if not gold_tokens:
            return 0.5
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        precision = num_same / max(len(pred_tokens), 1)
        recall = num_same / max(len(gold_tokens), 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    elif mode == "combined":
        # Blend: contain (0.6) + f1 (0.4)
        contain_score = 1.0 if gold_str in pred_str else 0.0
        if not contain_score:
            gold_tokens = gold_str.split()
            hits = sum(1 for t in gold_tokens if t in pred_str)
            contain_score = hits / max(len(gold_tokens), 1) * 0.5
        
        pred_tokens = pred_str.split()
        gold_tokens = gold_str.split()
        if gold_tokens:
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            precision = num_same / max(len(pred_tokens), 1)
            recall = num_same / max(len(gold_tokens), 1)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.5
        
        return 0.6 * contain_score + 0.4 * f1
    
    else:
        # Default fallback to contain
        return 1.0 if gold_str in pred_str else 0.0


def compute_synthesizer_f1(predicted_text: str, gold_answer: str) -> float:
    """Compute token F1 reward for synthesizer answer generation."""
    pred_tokens = predicted_text.lower().split()
    gold_tokens = gold_answer.lower().split()
    if not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gold_tokens), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def compute_format_penalty(predicted_text: str) -> float:
    """Format signal disabled to prevent mode collapse on small models.
    
    Returns 0.0 regardless of format quality, allowing the model to focus
    purely on task reward (page selection F1) without format-related penalties
    that cause mode collapse in 0.8B models.
    """
    return 0.0


# ---------------------------------------------------------------------------
# V-MMOA paper per-agent reward signals (Eq. 1 dense terms)
# ---------------------------------------------------------------------------
# These mirror the agent-local shaping terms of the V-MMOA return for use in
# per-agent PPO (where each agent is trained in isolation, so the global R_ans
# and the iteration-cost C(τ) are not available).
# ---------------------------------------------------------------------------


def compute_seeker_recall(
    predicted_text: str, gold_pages: List[int], num_candidates: int, k: int = 8
) -> float:
    """R_ret: recall@k of the Seeker's selected pages against gold evidence.

    Paper Sec. 4.2: pure recall, no precision bonus / format penalty.  This is
    the low-variance shaping signal the paper recommends for the Seeker.
    """
    pred_pages: List[int] = []
    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict):
            pred_pages = parsed.get("choice", []) or []
        elif isinstance(parsed, list):
            pred_pages = [int(x) for x in parsed if isinstance(x, (int, float))]
    except (json.JSONDecodeError, TypeError, ValueError):
        numbers = re.findall(r"\d+", predicted_text)
        pred_pages = [int(n) for n in numbers if int(n) < num_candidates]

    pred_pages = [p for p in pred_pages if 0 <= p < num_candidates][:k]

    if not gold_pages:
        return 0.0
    gold_set = set(gold_pages)
    if not gold_set:
        return 0.0
    return len(set(pred_pages) & gold_set) / len(gold_set)


def compute_inspector_precision(
    predicted_text: str, gold_pages: List[int], gold_answer: str
) -> float:
    """R_ins: precision of the Inspector's accepted evidence over gold evidence.

    Paper Sec. 4.2: rewards keeping only gold-aligned pages.  When the Inspector
    emits an answer rather than a page selection (common in per-agent training),
    falls back to answer-level F1 so the signal remains informative.
    """
    parsed = None
    try:
        parsed = json.loads(predicted_text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    accepted_pages: List[int] = []
    if isinstance(parsed, dict):
        for key in ("choice", "information", "evidence", "pages"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, list):
                    accepted_pages = [
                        int(x) for x in val if isinstance(x, (int, float))
                    ]
                elif isinstance(val, str):
                    accepted_pages = [
                        int(n) for n in re.findall(r"\d+", val)
                    ]
                break

    if accepted_pages and gold_pages:
        accepted_set = set(accepted_pages)
        gold_set = set(gold_pages)
        if not accepted_set:
            return 0.0
        return len(accepted_set & gold_set) / len(accepted_set)

    # Fallback: Inspector produced an answer.
    if isinstance(parsed, dict) and "answer" in parsed:
        model_answer = str(parsed.get("answer", "")).strip()
        return compute_synthesizer_f1(model_answer, gold_answer)
    return 0.0


def compute_answer_f1(predicted_text: str, gold_answer: str) -> float:
    """R_ans: answer-level F1 (global terminal term of Eq. 1).

    Used for the Synthesizer in V-MMOA per-agent mode: pure answer F1 with no
    EM bonus or length penalty, matching the paper's R_ans definition.
    """
    model_answer = predicted_text.strip()
    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict) and "answer" in parsed:
            model_answer = str(parsed.get("answer", "")).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return compute_synthesizer_f1(model_answer, gold_answer)


def compute_reference_f1(pred_refs: List[int], gold_refs: List[int]) -> float:
    """F1 between predicted and gold reference page indices.

    Since the Synthesizer receives gold pages as input, the gold reference
    is all provided images (indices 0..N-1).  The model should cite the
    correct subset that supports its answer.
    """
    if not gold_refs and not pred_refs:
        return 1.0
    if not gold_refs or not pred_refs:
        return 0.0
    pred_set = set(pred_refs)
    gold_set = set(gold_refs)
    hits = len(pred_set & gold_set)
    if hits == 0:
        return 0.0
    precision = hits / len(pred_set)
    recall = hits / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def compute_groundedness_score(answer: str, reason: str) -> float:
    """Heuristic groundedness score: does the answer appear in the reason?

    A grounded answer should be derived from (or at least mentioned in) the
    reasoning text that references the page content.  If the answer string
    appears in the reason, it's likely grounded; otherwise it may be
    hallucinated.
    """
    if not answer or not reason:
        return 0.0
    answer_lower = answer.lower().strip()
    reason_lower = reason.lower()
    if answer_lower in reason_lower:
        return 1.0
    # Partial match: check if answer tokens appear in reason
    answer_tokens = answer_lower.split()
    reason_tokens = reason_lower.split()
    if not answer_tokens:
        return 0.0
    matches = sum(1 for t in answer_tokens if t in reason_tokens)
    return matches / len(answer_tokens)


def compute_synthesizer_grounding_reward(
    predicted_text: str, gold_answer: str, gold_refs: List[int], num_pages: int
) -> float:
    """Grounding-aware reward for independent Synthesizer PPO.

    R_synth = 1.0 * AnswerF1
           + 0.3 * ReferenceF1
           + 0.2 * groundedness_score
           - 0.2 * hallucinated_reference
           - 0.2 * invalid_json

    Where:
      - AnswerF1: token-level F1 between predicted and gold answer
      - ReferenceF1: F1 between predicted and gold reference indices
      - groundedness_score: whether the answer appears in the reason text
      - hallucinated_reference: 1.0 if any cited index is out of range
      - invalid_json: 1.0 if the output is not valid JSON

    This reward prevents the model from guessing common answers without
    grounding them in the reference pages (experiment: Synthesizer-only training).
    """
    # Parse model output
    model_answer = ""
    model_refs: List[int] = []
    model_reason = ""
    valid_json = True

    try:
        parsed = json.loads(predicted_text)
        if isinstance(parsed, dict):
            model_answer = str(parsed.get("answer", "")).strip()
            model_reason = str(parsed.get("reason", "")).strip()
            refs = parsed.get("reference", [])
            if isinstance(refs, list):
                model_refs = [int(r) for r in refs if isinstance(r, (int, float))]
            elif isinstance(refs, (int, float)):
                model_refs = [int(refs)]
        else:
            valid_json = False
            model_answer = predicted_text.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        valid_json = False
        model_answer = predicted_text.strip()

    # Answer F1
    answer_f1 = compute_synthesizer_f1(model_answer, gold_answer)

    # Reference F1 (gold_refs = all provided page indices)
    reference_f1 = compute_reference_f1(model_refs, gold_refs)

    # Groundedness score
    groundedness = compute_groundedness_score(model_answer, model_reason)

    # Hallucinated reference: any cited index out of range
    hallucinated = 1.0 if any(r < 0 or r >= num_pages for r in model_refs) else 0.0

    # Invalid JSON penalty
    invalid_json = 1.0 if not valid_json else 0.0

    reward = (
        1.0 * answer_f1
        + 0.3 * reference_f1
        + 0.2 * groundedness
        - 0.2 * hallucinated
        - 0.2 * invalid_json
    )
    return max(-1.0, min(1.0, reward))


# OCR text cache for grounding reward (loaded lazily)
_OCR_CACHE: Dict[str, str] = {}


def _load_ocr_for_image(image_path: str) -> str:
    """Load OCR text for a SlideVQA image from ppocr directory.
    
    Image: /path/to/SlideVQA/img/docname_page.jpg
    OCR:   /path/to/SlideVQA/ppocr/docname_page.txt
    """
    global _OCR_CACHE
    if image_path in _OCR_CACHE:
        return _OCR_CACHE[image_path]
    
    # Convert image path to OCR path
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    ocr_path = os.path.join(
        os.path.dirname(image_path).replace('/img', '/ppocr'),
        name_without_ext + '.txt'
    )
    
    ocr_text = ""
    if os.path.exists(ocr_path):
        try:
            with open(ocr_path, 'r', encoding='utf-8', errors='ignore') as f:
                ocr_text = f.read().lower().strip()
        except Exception:
            ocr_text = ""
    
    _OCR_CACHE[image_path] = ocr_text
    return ocr_text


def compute_synthesizer_grounding_reward_v7(
    predicted_text: str, gold_answer: str, image_paths: List[str]
) -> float:
    """Grounding-aware reward for Synthesizer PPO v7 (plain-text output).
    
    Adapted from compute_synthesizer_grounding_reward for SFT v7 models
    that output plain answers (not JSON). Uses OCR text from reference
    images to verify grounding.
    
    R_synth = 1.0 * AnswerF1
           + 0.3 * ExactMatch
           + 0.2 * ocr_groundedness
           - 0.2 * unsupported_answer
    
    Where:
      - AnswerF1: token-level F1 between predicted and gold answer
      - ExactMatch: binary 1/0 for exact string match with gold answer
      - ocr_groundedness: binary 1/0 — 1 if answer tokens appear in OCR text
        of the provided reference images (prevents guessing without evidence)
      - unsupported_answer: binary penalty — 1 if answer is non-trivial (length > 2)
        but has zero overlap with OCR text AND low F1 with gold answer
    
    This reward prevents the model from guessing common answers without
    grounding them in the reference pages.
    """
    model_answer = predicted_text.strip()
    
    # Answer F1
    answer_f1 = compute_synthesizer_f1(model_answer, gold_answer)
    
    # Exact match (normalized)
    exact_match = 1.0 if model_answer.lower() == gold_answer.lower() else 0.0
    
    # OCR-based groundedness: check if answer tokens appear in OCR text
    ocr_text = ""
    for img_path in image_paths:
        ocr_text += _load_ocr_for_image(img_path) + " "
    ocr_text = ocr_text.strip()
    
    # Check if key answer tokens appear in OCR text
    answer_tokens = [t for t in model_answer.lower().split() if len(t) > 1]
    if not answer_tokens:
        # Very short answer (e.g., single number) — check substring
        ocr_groundedness = 1.0 if (model_answer.lower() in ocr_text or not ocr_text) else 0.0
    else:
        # Check what fraction of answer tokens appear in OCR text
        ocr_tokens = set(ocr_text.split())
        matched = sum(1 for t in answer_tokens if t in ocr_tokens)
        ocr_groundedness = matched / len(answer_tokens) if answer_tokens else 0.0
    
    # Unsupported answer penalty: answer is non-trivial but has no OCR evidence
    # and poor F1 — likely a guess
    unsupported = 0.0
    if len(answer_tokens) > 0 and ocr_groundedness < 0.1 and answer_f1 < 0.2:
        unsupported = 1.0
    
    reward = (
        1.0 * answer_f1
        + 0.3 * exact_match
        + 0.2 * ocr_groundedness
        - 0.2 * unsupported
    )
    return max(-0.5, min(1.5, reward))


# ---------------------------------------------------------------------------
# Per-Agent PPO Training Loop
# ---------------------------------------------------------------------------

def shuffle_candidates_with_gold(image_paths: List[str], gold_pages: List[int], seed: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Shuffle candidate order and remap gold page indices (diagnosis Sec. 5.4).

    Returns the shuffled image_paths and the corresponding gold page indices
    in the new ordering.  This utility is used when the data pipeline supports
    on-the-fly candidate shuffling to remove rank/position bias.

    Note: True per-step shuffling requires re-tokenizing the instruction text
    with the new candidate order.  The per-agent PPO workflow currently uses
    pre-tokenized data, so shuffling is handled at the data preparation level
    (regenerate PPO data with shuffled candidates).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    n = len(image_paths)
    perm = list(range(n))
    rng.shuffle(perm)

    shuffled_images = [image_paths[i] for i in perm]

    # Build old-index → new-index mapping
    old_to_new = {old: new for new, old in enumerate(perm)}
    shuffled_gold = [old_to_new[g] for g in gold_pages if g in old_to_new]

    return shuffled_images, sorted(shuffled_gold)


def per_agent_ppo_train(trainer: "CustomPPOTrainer_QSG", gold_data: Dict, agent_type: str) -> None:
    """
    Simple per-agent PPO training loop for ViDoRAG agents.

    Instead of orchestrating 3 agents simultaneously (as in MAPPO),
    this trains a single agent with reward based on its individual task:
      - seeker: page selection F1
      - inspector: routing accuracy
      - synthesizer: answer generation F1
    """
    total_train_batch_size = (
        trainer.args.per_device_train_batch_size
        * trainer.args.gradient_accumulation_steps
        * trainer.finetuning_args.ppo_buffer_size
        * trainer.args.world_size
    )
    if trainer.args.max_steps > 0:
        max_steps = trainer.args.max_steps
    else:
        len_dataloader = len(trainer.dataloader)
        num_train_epochs = trainer.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

    trainer.state.max_steps = max_steps
    trainer.state.num_train_epochs = trainer.args.num_train_epochs
    trainer.state.is_local_process_zero = trainer.is_local_process_zero()
    trainer.state.is_world_process_zero = trainer.is_world_process_zero()

    if trainer.is_world_process_zero():
        logger.info(f"***** Running Per-Agent PPO Training ({agent_type}) *****")
        logger.info(f"  Reward method = {REWARD_METHOD}")
        logger.info(f"  Num examples = {len(trainer.dataset):,}")
        logger.info(f"  Num Epochs = {trainer.args.num_train_epochs}")
        logger.info(f"  Batch size per device = {trainer.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {trainer.args.gradient_accumulation_steps:,}")
        logger.info(f"  PPO epochs per batch = {trainer.finetuning_args.ppo_epochs}")
        logger.info(f"  Total training steps = {max_steps:,}")
        logger.info(f"  Trainable parameters = {count_parameters(trainer.model)[0]:,}")
        logger.info(f"  Gold data entries = {len(gold_data):,}")

    dataiter = iter(trainer.dataloader)
    loss_meter = AverageMeter()
    reward_meter = AverageMeter()
    # Enhanced metrics (diagnosis Sec. 7): track per-step seeker diagnostics
    missing_gold_meter = AverageMeter()   # missing_any_gold_rate
    avg_selected_meter = AverageMeter()    # average_selected_pages
    invalid_json_meter = AverageMeter()    # invalid_json_rate
    seeker_recall_meter = AverageMeter()   # seeker recall per step
    entropy_meter = AverageMeter()         # policy entropy (from stats)
    # Synthesizer grounding diagnostics (for synth_grounding_v7)
    answer_f1_meter = AverageMeter()       # avg AnswerF1
    exact_match_meter = AverageMeter()     # exact match rate
    ocr_grounded_meter = AverageMeter()    # OCR groundedness
    unsupported_meter = AverageMeter()     # unsupported answer rate
    trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)

    os.makedirs(trainer.args.output_dir, exist_ok=True)
    log_path = os.path.join(trainer.args.output_dir, "per_agent_ppo_log.txt")

    for step in tqdm(range(max_steps), disable=not trainer.is_local_process_zero()):
        try:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(trainer.dataloader)
                batch = next(dataiter)

            # --- Generate responses ---
            trainer.model.eval()
            trainer.tokenizer.padding_side = "right"

            # Use the trainer's get_inputs method for generation
            queries, responses = trainer.get_inputs(batch)

            # --- Compute rewards ---
            rewards = []
            batch_size = len(queries)
            for i in range(batch_size):
                # Decode the query to extract the question
                query_text = trainer.tokenizer.decode(queries[i], skip_special_tokens=True)
                response_text = trainer.tokenizer.decode(responses[i], skip_special_tokens=True)

                # Try to find gold data for this question
                reward_val = 0.0
                matched_gold = None

                # Try matching by question substring
                for gold_key, gold_info in gold_data.items():
                    if gold_key in query_text or query_text[:100] in gold_key:
                        matched_gold = gold_info
                        break

                if matched_gold is None:
                    # Fallback: use simple format-based reward
                    reward_val = compute_format_penalty(response_text) + 0.1
                elif REWARD_METHOD == "vmmoa":
                    # V-MMOA paper (Sec. 4.2) agent-local shaping signals.
                    # Per-agent training has no retrieval loop, so only the
                    # agent-local dense term of Eq. 1 is used here (no global
                    # R_ans / iteration cost C(τ); those live in the MAPPO trainer).
                    if agent_type == "seeker":
                        gold_pages = matched_gold.get("gold_pages", [])
                        num_candidates = matched_gold.get("num_candidates", 21)
                        reward_val = compute_seeker_recall(
                            response_text, gold_pages, num_candidates
                        )
                    elif agent_type == "inspector":
                        gold_answer = matched_gold.get("answer", "")
                        gold_pages = matched_gold.get("gold_pages", [])
                        reward_val = compute_inspector_precision(
                            response_text, gold_pages, gold_answer
                        )
                    elif agent_type == "synthesizer":
                        gold_answer = matched_gold.get("answer", "")
                        reward_val = compute_answer_f1(response_text, gold_answer)
                    else:
                        reward_val = compute_format_penalty(response_text) + 0.1
                elif agent_type == "seeker":
                    gold_pages = matched_gold.get("gold_pages", [])
                    num_candidates = matched_gold.get("num_candidates", 21)
                    if REWARD_METHOD == "e2e":
                        # Hybrid e2e reward: full evidence + answer F1
                        # Evidence component uses compute_seeker_evidence_reward
                        # (includes missing_any_gold, over_selection, invalid_json penalties)
                        # R = 0.4 * full_evidence_reward + 0.6 * answer_f1
                        image_paths = matched_gold.get("image_paths", [])
                        question = matched_gold.get("question", "")
                        gold_answer = matched_gold.get("answer", "")
                        reward_val = compute_seeker_e2e_reward(
                            response_text, gold_pages, num_candidates,
                            image_paths, question, gold_answer
                        )
                    elif REWARD_METHOD == "evidence":
                        # Evidence-level reward (plan Sec. III.3):
                        # 1.0*Recall + 0.3*Precision - over_selection - invalid_json
                        reward_val = compute_seeker_evidence_reward(
                            response_text, gold_pages, num_candidates
                        )
                    else:
                        f1_reward = compute_seeker_page_f1(response_text, gold_pages, num_candidates)
                        format_reward = compute_format_penalty(response_text)
                        reward_val = f1_reward + format_reward
                elif agent_type == "inspector":
                    gold_answer = matched_gold.get("answer", "")
                    reward_val = compute_inspector_reward(response_text, gold_answer)
                elif agent_type == "synthesizer":
                    gold_answer = matched_gold.get("answer", "")
                    if REWARD_METHOD == "synth_grounding_v7":
                        # Grounding-aware reward for SFT v7 (plain-text output)
                        # R = 1.0*AnswerF1 + 0.3*ExactMatch + 0.2*ocr_groundedness
                        #     - 0.2*unsupported_answer
                        image_paths = matched_gold.get("image_paths", [])
                        reward_val = compute_synthesizer_grounding_reward_v7(
                            response_text, gold_answer, image_paths
                        )
                    elif REWARD_METHOD == "synth_grounding":
                        # Grounding-aware reward (experiment: Synthesizer-only training)
                        # R = 1.0*AnswerF1 + 0.3*RefF1 + 0.2*groundedness
                        #     - 0.2*hallucinated - 0.2*invalid_json
                        num_pages = matched_gold.get("num_candidates", len(matched_gold.get("image_paths", [])))
                        gold_refs = list(range(num_pages))
                        reward_val = compute_synthesizer_grounding_reward(
                            response_text, gold_answer, gold_refs, num_pages
                        )
                    else:
                        reward_val = compute_synthesizer_f1(response_text, gold_answer)
                else:
                    reward_val = compute_format_penalty(response_text) + 0.1

                rewards.append(torch.tensor([reward_val]))

                # --- Enhanced metrics for Seeker (diagnosis Sec. 7) ---
                if agent_type == "seeker" and matched_gold is not None:
                    gold_pages_diag = matched_gold.get("gold_pages", [])
                    num_cands_diag = matched_gold.get("num_candidates", 21)
                    # Parse predicted pages for diagnostics
                    pred_diag: List[int] = []
                    valid_json_diag = True
                    try:
                        parsed_diag = json.loads(response_text)
                        if isinstance(parsed_diag, dict):
                            pred_diag = parsed_diag.get("choice", []) or []
                            if not isinstance(pred_diag, list):
                                pred_diag = [pred_diag]
                            pred_diag = [int(p) for p in pred_diag if isinstance(p, (int, float, str)) and str(p).strip().isdigit()]
                        elif isinstance(parsed_diag, list):
                            pred_diag = [int(x) for x in parsed_diag if isinstance(x, (int, float))]
                        else:
                            valid_json_diag = False
                    except (json.JSONDecodeError, TypeError, ValueError):
                        valid_json_diag = False
                        numbers_diag = re.findall(r"\d+", response_text)
                        pred_diag = [int(n) for n in numbers_diag if int(n) < num_cands_diag]
                    pred_diag = [p for p in pred_diag if 0 <= p < num_cands_diag]

                    pred_set_diag = set(pred_diag)
                    gold_set_diag = set(gold_pages_diag)
                    hits_diag = len(pred_set_diag & gold_set_diag)

                    # Track metrics
                    missing_gold_meter.update(1.0 if hits_diag < len(gold_set_diag) else 0.0)
                    avg_selected_meter.update(float(len(pred_diag)))
                    invalid_json_meter.update(0.0 if valid_json_diag else 1.0)
                    seeker_recall_meter.update(hits_diag / max(len(gold_set_diag), 1))

                # --- Enhanced metrics for Synthesizer (grounding v7) ---
                if agent_type == "synthesizer" and matched_gold is not None and REWARD_METHOD == "synth_grounding_v7":
                    gold_ans = matched_gold.get("answer", "")
                    img_paths = matched_gold.get("image_paths", [])
                    model_ans = response_text.strip()
                    # AnswerF1
                    af1 = compute_synthesizer_f1(model_ans, gold_ans)
                    answer_f1_meter.update(af1)
                    # Exact match
                    exact_match_meter.update(1.0 if model_ans.lower() == gold_ans.lower() else 0.0)
                    # OCR groundedness
                    ocr_txt = ""
                    for ip in img_paths:
                        ocr_txt += _load_ocr_for_image(ip) + " "
                    ans_tokens = [t for t in model_ans.lower().split() if len(t) > 1]
                    if ans_tokens:
                        ocr_tokens_set = set(ocr_txt.split())
                        matched_t = sum(1 for t in ans_tokens if t in ocr_tokens_set)
                        ocr_grounded_meter.update(matched_t / len(ans_tokens))
                    else:
                        ocr_grounded_meter.update(1.0 if model_ans.lower() in ocr_txt or not ocr_txt else 0.0)
                    # Unsupported
                    ocr_g = ocr_grounded_meter.avg
                    unsupported_meter.update(1.0 if (len(ans_tokens) > 0 and ocr_g < 0.1 and af1 < 0.2) else 0.0)

            # --- PPO Step ---
            trainer.model.train()
            trainer.tokenizer.padding_side = "left"
            stats = trainer.step(queries, responses, rewards)

            # --- Logging ---
            if stats:
                loss = stats.get("ppo/loss/total", stats.get("objective/kl", 0.0))
                mean_reward = torch.stack(rewards).mean().item()
                kl_val = float(stats.get('objective/kl', 0.0))
                entropy_val = float(stats.get('ppo/loss/entropy', stats.get('objective/entropy', 0.0)))
                loss_meter.update(float(loss) if isinstance(loss, (int, float)) else 0.0)
                reward_meter.update(mean_reward)
                entropy_meter.update(entropy_val)

                trainer.state.global_step = step + 1

                if (step + 1) % trainer.args.logging_steps == 0 and trainer.is_world_process_zero():
                    log_msg = (
                        f"Step {step+1}/{max_steps} | "
                        f"Loss: {loss_meter.avg:.4f} | "
                        f"Reward: {reward_meter.avg:.4f} | "
                        f"KL: {kl_val:.4f} | "
                        f"Entropy: {entropy_meter.avg:.4f}"
                    )
                    # Add Seeker-specific diagnostics (diagnosis Sec. 7)
                    if agent_type == "seeker":
                        log_msg += (
                            f" | MissGold: {missing_gold_meter.avg:.3f}"
                            f" | AvgSel: {avg_selected_meter.avg:.2f}"
                            f" | BadJSON: {invalid_json_meter.avg:.3f}"
                            f" | S_recall: {seeker_recall_meter.avg:.3f}"
                        )
                    # Add Synthesizer grounding diagnostics (synth_grounding_v7)
                    if agent_type == "synthesizer" and REWARD_METHOD == "synth_grounding_v7":
                        log_msg += (
                            f" | AnsF1: {answer_f1_meter.avg:.3f}"
                            f" | EM: {exact_match_meter.avg:.3f}"
                            f" | OCRGround: {ocr_grounded_meter.avg:.3f}"
                            f" | Unsupported: {unsupported_meter.avg:.3f}"
                        )
                    logger.info(log_msg)
                    with open(log_path, "a") as f:
                        f.write(log_msg + "\n")

                    loss_meter.reset()
                    reward_meter.reset()
                    entropy_meter.reset()
                    missing_gold_meter.reset()
                    avg_selected_meter.reset()
                    invalid_json_meter.reset()
                    seeker_recall_meter.reset()
                    answer_f1_meter.reset()
                    exact_match_meter.reset()
                    ocr_grounded_meter.reset()
                    unsupported_meter.reset()

            # --- Save checkpoint ---
            if (step + 1) % trainer.args.save_steps == 0:
                trainer.save_model(
                    os.path.join(trainer.args.output_dir, f"checkpoint-{step+1}")
                )

        except Exception as e:
            if trainer.is_world_process_zero():
                error_msg = f"Step {step+1}/{max_steps} | ERROR: {type(e).__name__}: {str(e)[:200]}"
                logger.warning(error_msg)
                with open(log_path, "a") as f:
                    f.write(error_msg + "\n")
            continue  # Skip this batch and continue training

    # --- Final save (always save at end of training) ---
    if trainer.is_world_process_zero():
        logger.info(f"Training complete. Saving final model to {trainer.args.output_dir}")
        with open(log_path, "a") as f:
            f.write(f"Training complete at step {max_steps}. Saving final model.\n")
    trainer.save_model(trainer.args.output_dir)
    trainer.callback_handler.on_train_end(trainer.args, trainer.state, trainer.control)


# ---------------------------------------------------------------------------
# Gold data loading
# ---------------------------------------------------------------------------

def load_gold_data_from_training_file(data_args, dataset_name: str) -> Dict:
    """
    Load gold answers from the training data file for reward computation.
    Builds a dict keyed by question text for O(1) lookup during training.
    """
    # Find the dataset file from dataset_info.json
    data_dir = getattr(data_args, "dataset_dir", None) or "data"
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")

    # Try standard LLaMA-Factory data location
    if not os.path.exists(dataset_info_path):
        dataset_info_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "data", "dataset_info.json"
        )

    if not os.path.exists(dataset_info_path):
        logger.warning(f"dataset_info.json not found at {dataset_info_path}")
        return {}

    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    if dataset_name not in dataset_info:
        logger.warning(f"Dataset '{dataset_name}' not found in dataset_info.json")
        return {}

    file_name = dataset_info[dataset_name]["file_name"]
    # Resolve relative path
    if not os.path.isabs(file_name):
        file_name = os.path.join(os.path.dirname(dataset_info_path), file_name)
    file_name = os.path.normpath(file_name)

    if not os.path.exists(file_name):
        logger.warning(f"Training data file not found: {file_name}")
        return {}

    logger.info(f"Loading gold data from: {file_name}")
    with open(file_name, "r") as f:
        data = json.load(f)

    gold_dict = {}
    for item in data:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        # Extract question from instruction
        question_match = re.search(r"Question:\s*(.+?)(?:\n|You are given)", instruction, re.DOTALL)
        if question_match:
            question_key = question_match.group(1).strip()[:200]  # first 200 chars as key
        else:
            question_key = instruction[:200]

        # Parse gold output
        gold_info = {"raw_output": output}
        try:
            parsed = json.loads(output)
            if "choice" in parsed:
                gold_info["gold_pages"] = parsed["choice"]
            if "answer" in parsed:
                gold_info["answer"] = parsed["answer"]
            if "routing" in parsed:
                gold_info["routing"] = parsed["routing"]
            if "decision" in parsed:
                gold_info["routing"] = parsed["decision"]
        except (json.JSONDecodeError, TypeError):
            gold_info["answer"] = output

        # Count candidates from instruction
        page_mentions = re.findall(r"Page \d+", instruction)
        # v4 format: "You are given N candidate page images indexed 0-(N-1)"
        v4_match = re.search(r"You are given (\d+) candidate page images", instruction)
        if v4_match:
            gold_info["num_candidates"] = int(v4_match.group(1))
        else:
            gold_info["num_candidates"] = max(len(page_mentions), 21)
        # Also count images field if available
        if "images" in item:
            gold_info["num_candidates"] = max(gold_info["num_candidates"], len(item["images"]))
            gold_info["num_candidates"] = len(item["images"]) if len(item["images"]) <= 10 else gold_info["num_candidates"]
            # Store image paths for e2e reward (Strategy A)
            gold_info["image_paths"] = item["images"]

        # Store the full question text for e2e reward (Strategy A)
        if question_match:
            gold_info["question"] = question_match.group(1).strip()

        # Enrich with gold answer from vidorag_gold_answers.json if available (Strategy A)
        # This is loaded lazily to avoid overhead when e2e reward is not used
        gold_dict[question_key] = gold_info

    logger.info(f"Loaded {len(gold_dict)} gold data entries.")
    return gold_dict


# ---------------------------------------------------------------------------
# Strategy A: Enrich gold_data with gold answers for e2e reward (additive)
# ---------------------------------------------------------------------------

_GOLD_ANSWERS_CACHE = None


def _load_gold_answers_cache() -> Dict:
    """Lazy-load vidorag_gold_answers.json for e2e reward enrichment."""
    global _GOLD_ANSWERS_CACHE
    if _GOLD_ANSWERS_CACHE is not None:
        return _GOLD_ANSWERS_CACHE

    gold_path = os.environ.get("VIDORAG_GOLD_ANSWERS") or os.path.join(
        os.environ.get("VIDORAG_PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5)),
        "data", "SlideVQA", "vidorag_gold_answers.json")
    if not os.path.exists(gold_path):
        logger.warning(f"[E2E Reward] Gold answers file not found: {gold_path}")
        _GOLD_ANSWERS_CACHE = {}
        return _GOLD_ANSWERS_CACHE

    with open(gold_path, "r") as f:
        _GOLD_ANSWERS_CACHE = json.load(f)
    logger.info(f"[E2E Reward] Loaded {len(_GOLD_ANSWERS_CACHE)} gold answers.")
    return _GOLD_ANSWERS_CACHE


def enrich_gold_data_with_answers(gold_data: Dict) -> Dict:
    """Enrich gold_data with gold answers from vidorag_gold_answers.json.

    Called only when VIDORAG_REWARD_METHOD=e2e.
    """
    gold_answers = _load_gold_answers_cache()
    if not gold_answers:
        return gold_data

    enriched = 0
    for question_key, gold_info in gold_data.items():
        # Try exact match first
        if question_key in gold_answers:
            gold_info["answer"] = gold_answers[question_key].get("answer", "")
            enriched += 1
            continue

        # Try substring matching (question_key is first 200 chars)
        for ans_key, ans_info in gold_answers.items():
            if question_key in ans_key or ans_key[:200] in question_key:
                gold_info["answer"] = ans_info.get("answer", "")
                enriched += 1
                break

    logger.info(f"[E2E Reward] Enriched {enriched}/{len(gold_data)} entries with gold answers.")
    return gold_data


def detect_agent_type(dataset_name: str) -> str:
    """Detect which agent is being trained from the dataset name."""
    name_lower = dataset_name.lower()
    if "seeker" in name_lower:
        return "seeker"
    elif "inspector" in name_lower:
        return "inspector"
    elif "synthesizer" in name_lower:
        return "synthesizer"
    return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, **tokenizer_module)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Select the PPO trainer.  The default is the QR-S-G multi-agent trainer
    # (CustomPPOTrainer_QSG).  When VIDORAG_MAPPO_TRAINER=vidorag is set, use the
    # ViDoRAG VLM MAPPO trainer (ViDoRAGVLMPPOTrainer) which implements the
    # Seeker/Inspector/Synthesizer loop and the V-MMOA hierarchical reward
    # decomposition (vidorag_rewards.compute_batch_rewards).  This is required to
    # run ViDoRAG MAPPO in V-MMOA mode; the QR-S-G trainer does not use the
    # V-MMOA reward signals.
    use_vidorag_trainer = os.environ.get("VIDORAG_MAPPO_TRAINER", "").lower() == "vidorag"
    if use_vidorag_trainer:
        from .trainer_vidorag import ViDoRAGVLMPPOTrainer as _PPOTrainerClass
        logger.info("Using ViDoRAG VLM MAPPO trainer (VIDORAG_MAPPO_TRAINER=vidorag)")
    else:
        _PPOTrainerClass = CustomPPOTrainer_QSG

    # Initialize our Trainer
    ppo_trainer = _PPOTrainerClass(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        # Detect if this is a ViDoRAG per-agent training (no external reward model)
        dataset_name = data_args.dataset
        if isinstance(dataset_name, list):
            dataset_name = dataset_name[0] if dataset_name else ""
        agent_type = detect_agent_type(dataset_name)

        if agent_type != "unknown" and reward_model is None and not use_vidorag_trainer:
            # Per-agent PPO mode for ViDoRAG
            logger.info(f"Detected ViDoRAG per-agent PPO mode for: {agent_type}")
            gold_data = load_gold_data_from_training_file(data_args, dataset_name)
            # Strategy A: Enrich gold_data with gold answers when e2e reward is active
            if REWARD_METHOD == "e2e":
                logger.info("[E2E Reward] Enriching gold_data with gold answers...")
                gold_data = enrich_gold_data_with_answers(gold_data)
            per_agent_ppo_train(ppo_trainer, gold_data, agent_type)
        else:
            # Joint MAPPO mode (QR-S-G by default, or ViDoRAG VLM MAPPO when
            # VIDORAG_MAPPO_TRAINER=vidorag).
            ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, getattr(training_args, "save_safetensors", True))

        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
