# Copyright 2024 ViDoRAG Team.
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

"""
ViDoRAG-specific reward functions for multi-agent PPO training.

This module provides reward signals for the three agents in the ViDoRAG pipeline:
- Seeker: Selects relevant document pages
- Inspector: Routes between answering and requesting more information
- Synthesizer: Produces the final answer

All reward functions are pure (no side effects, no API calls) for fast batch computation.
All rewards are clamped to [-1.0, 1.0] for PPO stability.
"""

import json
import os
import re
import string
from collections import Counter
from typing import Dict, List, Optional


# =============================================================================
# Reward Method Switch (original ViDoRAG vs V-MMOA paper)
# =============================================================================
# Controls which reward formulation is used for PPO / MAPPO training.
#
#   "original" : ViDoRAG per-agent blended rewards.
#                Each agent receives  (1 - w) * individual + w * shared_answer_F1
#                with different individual signals per agent (see compute_batch_rewards).
#
#   "vmmoa"    : V-MMOA paper (Anonymous EMNLP Submission, Sec. 4.2, Eq. 1)
#                hierarchical reward decomposition:
#
#                    R(τ) = R_ans  + α·R_ret  + β·R_ins  - λ·C(τ)
#                             global  Seeker    Inspector   cost
#
#                where  R_ans  = answer-level F1 of the final answer (global, terminal),
#                       R_ret  = recall@k of retrieved pages vs gold evidence (Seeker-local),
#                       R_ins  = precision of Inspector's accepted set vs gold evidence,
#                       C(τ)   = Σ_t 1[a_insp_t ≠ stop]  (non-terminal iteration count).
#
#                Under CTDE MAPPO all three agents share the SAME return R(τ) and a single
#                centralised critic provides the baseline (shared advantage, paper Sec. 4.3).
#
# Toggle at runtime without code changes via the environment variable:
#     export VIDORAG_REWARD_METHOD=vmmoa   # or "original" (default)
REWARD_METHOD = os.environ.get("VIDORAG_REWARD_METHOD", "original").lower()

# V-MMOA reward coefficients (paper Sec. 4.4 / Appendix B).
VMMOA_ALPHA = float(os.environ.get("VMMOA_ALPHA", "0.5"))    # Seeker recall weight (α)
VMMOA_BETA = float(os.environ.get("VMMOA_BETA", "0.3"))     # Inspector precision weight (β)
VMMOA_LAMBDA = float(os.environ.get("VMMOA_LAMBDA", "0.05"))  # iteration-cost penalty (λ)

# MAPPO independent-adapter cooperative bonus coefficient.
# When the final answer F1 > 0.5, a small bonus is added to ALL agents
# to encourage cooperation without destroying per-agent credit assignment.
MAPPO_COOP_BONUS = float(os.environ.get("MAPPO_COOP_BONUS", "0.2"))
MAPPO_COOP_THRESHOLD = float(os.environ.get("MAPPO_COOP_THRESHOLD", "0.5"))


# =============================================================================
# Utility Functions
# =============================================================================


def parse_json_output(text: str) -> Optional[dict]:
    """Safely parse JSON from model output, handling common formatting issues.

    Attempts multiple strategies:
    1. Direct JSON parsing of the full text
    2. Extract JSON from markdown code blocks (```json ... ```)
    3. Find the first { ... } substring and parse it

    Args:
        text: Raw model output text.

    Returns:
        Parsed dictionary if successful, None otherwise.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_page_indices(choice_field) -> List[int]:
    """Extract valid page indices from various formats.

    Handles:
    - List of integers: [1, 3, 5]
    - List of strings: ["1", "3", "5"]
    - Comma-separated string: "1, 3, 5"
    - Single integer: 3
    - Mixed formats with "page" prefix: ["page1", "page3"]

    Args:
        choice_field: The raw value from the "choice" field in JSON output.

    Returns:
        List of integer page indices (deduplicated, order preserved).
    """
    indices: List[int] = []

    if choice_field is None:
        return indices

    if isinstance(choice_field, int):
        return [choice_field]

    if isinstance(choice_field, str):
        # Try comma-separated
        parts = re.split(r"[,;\s]+", choice_field.strip())
        for part in parts:
            # Remove common prefixes like "page", "p", "Page"
            cleaned = re.sub(r"(?i)^(?:page|p)\s*", "", part.strip())
            try:
                indices.append(int(cleaned))
            except (ValueError, TypeError):
                continue
        return _deduplicate(indices)

    if isinstance(choice_field, (list, tuple)):
        for item in choice_field:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, float) and item == int(item):
                indices.append(int(item))
            elif isinstance(item, str):
                cleaned = re.sub(r"(?i)^(?:page|p)\s*", "", item.strip())
                try:
                    indices.append(int(cleaned))
                except (ValueError, TypeError):
                    continue
        return _deduplicate(indices)

    return indices


def _deduplicate(items: List[int]) -> List[int]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =============================================================================
# Answer Normalization
# =============================================================================


def normalize_answer(s: str) -> str:
    """Normalize answer text: lowercase, remove articles, punctuation, extra whitespace.

    Adapted from SQuAD evaluation script.

    Args:
        s: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if not s:
        return ""

    def _remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def _remove_punc(text: str) -> str:
        for punct in string.punctuation:
            text = text.replace(punct, " ")
        return text

    def _lower(text: str) -> str:
        return text.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))


# =============================================================================
# Base Metrics
# =============================================================================


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth.

    Both inputs are normalized before comparison. Returns 0.0 if either
    normalized string is empty.

    Args:
        prediction: Model's predicted answer text.
        ground_truth: Reference answer text.

    Returns:
        F1 score in range [0.0, 1.0].
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 1.0 if normalized_prediction == normalized_ground_truth else 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)

    return f1


def compute_em(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization.

    Args:
        prediction: Model's predicted answer text.
        ground_truth: Reference answer text.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_accuracy(prediction: str, ground_truth: str) -> float:
    """Check if normalized ground truth is contained in normalized prediction.

    Args:
        prediction: Model's predicted answer text.
        ground_truth: Reference answer text.

    Returns:
        1.0 if ground truth is contained in prediction, 0.0 otherwise.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    if not norm_gt:
        return 1.0 if not norm_pred else 0.0

    return 1.0 if norm_gt in norm_pred else 0.0


# =============================================================================
# Seeker Reward Function
# =============================================================================


def compute_seeker_reward(
    prediction: str,
    gold_pages: List[int],
    total_pages: int,
    format_weight: float = -1.0,
    max_pages: int = 10,
) -> float:
    """Compute reward for the Seeker agent (page selection).

    The Seeker must output JSON with a "choice" field containing selected page indices.

    Reward components:
    - Base reward: Page recall = |predicted ∩ gold| / |gold|
    - Precision bonus: +0.3 * (|predicted ∩ gold| / |predicted|)
    - Format penalty: -1.0 if JSON parsing fails or "choice" field missing
    - Over-selection penalty: -0.2 * max(0, |predicted| - max_pages) / max_pages
    - Empty penalty: -0.5 if no pages selected

    Args:
        prediction: Model's raw output text (expected JSON with "choice" field).
        gold_pages: Ground truth evidence page indices.
        total_pages: Total candidate pages available.
        format_weight: Penalty for format violations (default: -1.0).
        max_pages: Maximum allowed page selections before penalty.

    Returns:
        Reward value clamped to [-1.0, 1.0].
    """
    # Parse JSON output
    parsed = parse_json_output(prediction)
    if not isinstance(parsed, dict):
        return max(-1.0, format_weight)

    # Extract choice field
    if "choice" not in parsed:
        return max(-1.0, format_weight)

    # Extract page indices
    predicted_pages = extract_page_indices(parsed["choice"])

    # Filter to valid range [0, total_pages)
    predicted_pages = [p for p in predicted_pages if 0 <= p < total_pages]

    # Empty penalty
    if len(predicted_pages) == 0:
        return max(-1.0, min(1.0, -0.5))

    # Compute sets
    predicted_set = set(predicted_pages)
    gold_set = set(gold_pages)

    # Base reward: recall
    if len(gold_set) == 0:
        # If no gold pages, reward based on selecting few pages
        base_reward = max(0.0, 1.0 - len(predicted_set) / max(total_pages, 1))
    else:
        base_reward = len(predicted_set & gold_set) / len(gold_set)

    # Precision bonus
    precision_bonus = 0.0
    if len(predicted_set) > 0 and len(gold_set) > 0:
        precision_bonus = 0.3 * (len(predicted_set & gold_set) / len(predicted_set))

    # Over-selection penalty
    over_selection_penalty = 0.0
    excess = len(predicted_pages) - max_pages
    if excess > 0:
        over_selection_penalty = -0.2 * excess / max_pages

    # Total reward
    reward = base_reward + precision_bonus + over_selection_penalty

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, reward))


# =============================================================================
# Inspector Reward Function
# =============================================================================


def compute_inspector_reward(
    prediction: str,
    gold_answer: str,
    has_sufficient_evidence: bool,
    format_weight: float = -1.0,
) -> float:
    """Compute reward for the Inspector agent (routing + answer quality).

    The Inspector must output JSON with either:
    - "answer" key: model decided evidence is sufficient and provides answer
    - "information" or "choice" key: model decided more information is needed

    Reward components:
    - Routing reward: +0.5 for correct routing, -0.5 for incorrect
    - Answer quality: +0.5 * F1(model_answer, gold_answer) when model provides answer
    - Format penalty: -1.0 if JSON parsing fails

    Args:
        prediction: Model's raw output text (expected JSON).
        gold_answer: Ground truth answer.
        has_sufficient_evidence: Whether the provided pages contain the answer.
        format_weight: Penalty for format violations (default: -1.0).

    Returns:
        Reward value clamped to [-1.0, 1.0].
    """
    # Parse JSON output
    parsed = parse_json_output(prediction)
    if not isinstance(parsed, dict):
        return max(-1.0, format_weight)

    # Determine routing decision
    model_decides_to_answer = "answer" in parsed
    model_decides_needs_more = "information" in parsed or "choice" in parsed

    # If neither key is present, format penalty
    if not model_decides_to_answer and not model_decides_needs_more:
        return max(-1.0, format_weight)

    reward = 0.0

    # Routing reward
    if has_sufficient_evidence and model_decides_to_answer:
        # Correct: evidence is sufficient and model answers
        reward += 0.5
    elif not has_sufficient_evidence and model_decides_needs_more:
        # Correct: evidence is insufficient and model requests more
        reward += 0.5
    else:
        # Incorrect routing
        reward -= 0.5

    # Answer quality (only when model provides answer)
    if model_decides_to_answer:
        model_answer = parsed.get("answer", "")
        if isinstance(model_answer, str):
            answer_f1 = compute_f1(model_answer, gold_answer)
            reward += 0.5 * answer_f1

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, reward))


# =============================================================================
# Synthesizer Reward Function
# =============================================================================


def compute_synthesizer_reward(
    prediction: str,
    gold_answer: str,
    max_words: int = 50,
    format_weight: float = -1.0,
) -> float:
    """Compute reward for the Synthesizer agent (final answer generation).

    The Synthesizer must output JSON with an "answer" field.

    Reward components:
    - Base reward: F1 score between predicted and gold answer
    - EM bonus: +0.3 if exact match
    - Length penalty: -0.3 if answer exceeds max_words
    - Empty penalty: -0.5 if answer is empty or only whitespace
    - Format penalty: -1.0 if JSON parsing fails or "answer" field missing

    Args:
        prediction: Model's raw output text (expected JSON with "answer" field).
        gold_answer: Ground truth answer.
        max_words: Maximum allowed word count before penalty.
        format_weight: Penalty for format violations (default: -1.0).

    Returns:
        Reward value clamped to [-1.0, 1.0].
    """
    # Parse JSON output
    parsed = parse_json_output(prediction)
    if not isinstance(parsed, dict):
        return max(-1.0, format_weight)

    # Extract answer field
    if "answer" not in parsed:
        return max(-1.0, format_weight)

    model_answer = parsed["answer"]
    if not isinstance(model_answer, str):
        model_answer = str(model_answer) if model_answer is not None else ""

    # Empty penalty
    if not model_answer.strip():
        return max(-1.0, min(1.0, -0.5))

    # Base reward: F1 score
    f1 = compute_f1(model_answer, gold_answer)
    reward = f1

    # EM bonus
    if compute_em(model_answer, gold_answer) == 1.0:
        reward += 0.3

    # Length penalty
    word_count = len(model_answer.split())
    if word_count > max_words:
        reward -= 0.3

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, reward))


# =============================================================================
# Shared Reward Computation
# =============================================================================


def compute_shared_reward(final_answer: str, gold_answer: str) -> float:
    """Shared reward given to ALL agents based on final pipeline answer quality.

    This encourages cooperation between agents: all agents benefit when
    the final answer is good, regardless of their individual performance.

    Args:
        final_answer: The final answer produced by the full pipeline.
        gold_answer: Ground truth answer.

    Returns:
        F1 score in range [0.0, 1.0].
    """
    return compute_f1(final_answer, gold_answer)


# =============================================================================
# V-MMOA Paper Reward Components (hierarchical decomposition, Eq. 1)
# =============================================================================
# These implement the three terms of the V-MMOA return
#     R(τ) = R_ans + α·R_ret + β·R_ins − λ·C(τ)
# R_ans is the global terminal answer-F1, R_ret / R_ins are dense agent-local
# shaping signals (recall@k for the Seeker, precision for the Inspector), and
# C(τ) penalises non-terminal iterations to encourage early stopping.
# =============================================================================


def compute_vmmoa_seeker_reward(
    prediction: str,
    gold_pages: List[int],
    total_pages: int,
    k: int = 8,
) -> float:
    """R_ret: recall@k of the Seeker's retrieved pages against gold evidence.

    Paper Sec. 4.2: "R_ret is the recall@k of R_t against gold evidence pages".
    Pure recall (no precision bonus, no format penalty) — the V-MMOA shaping
    signal is intentionally a single, low-variance term.

    Args:
        prediction: Seeker raw output (JSON with a ``choice`` field, or text).
        gold_pages: Ground-truth evidence page indices.
        total_pages: Total candidate pages available (for range filtering).
        k: retrieval depth (paper Appendix B: k = 8).

    Returns:
        Recall in [0.0, 1.0].  Returns 0.0 when no gold pages are available.
    """
    parsed = parse_json_output(prediction)
    predicted_pages: List[int] = []
    if isinstance(parsed, dict) and "choice" in parsed:
        predicted_pages = extract_page_indices(parsed["choice"])
    else:
        # Fallback: extract integers from raw text.
        predicted_pages = [
            int(n) for n in re.findall(r"\d+", prediction) if int(n) < total_pages
        ]

    predicted_pages = [p for p in predicted_pages if 0 <= p < total_pages][:k]

    if not gold_pages:
        return 0.0

    gold_set = set(gold_pages)
    if not gold_set:
        return 0.0

    predicted_set = set(predicted_pages)
    return len(predicted_set & gold_set) / len(gold_set)


def compute_vmmoa_inspector_reward(
    prediction: str,
    gold_pages: List[int],
    gold_answer: str,
) -> float:
    """R_ins: precision of the Inspector's accepted evidence set over gold evidence.

    Paper Sec. 4.2: "R_ins is the precision of the Inspector's accepted set E_t
    over the same gold evidence".  The Inspector filters the retrieved set; this
    term rewards keeping only gold-aligned pages.

    When the Inspector emits an answer instead of a page selection (as happens in
    the per-agent PPO setup where there is no retrieval loop), we fall back to the
    answer-level F1 so the signal is still informative.

    Args:
        prediction: Inspector raw output (JSON with ``choice``/``information``/
            ``evidence``/``pages`` for filtering, or ``answer`` for answering).
        gold_pages: Ground-truth evidence page indices.
        gold_answer: Ground-truth answer (used by the fallback path).

    Returns:
        Precision in [0.0, 1.0] (or answer F1 in the fallback path).
    """
    try:
        parsed = parse_json_output(prediction)
    except Exception:
        return 0.0

    # parse_json_output only returns dict or None, but guard defensively.
    if not isinstance(parsed, dict):
        return 0.0

    accepted_pages: List[int] = []
    for key in ("choice", "information", "evidence", "pages"):
        if key in parsed:
            try:
                accepted_pages = extract_page_indices(parsed[key])
            except Exception:
                accepted_pages = []
            break

    if accepted_pages and gold_pages:
        accepted_set = set(accepted_pages)
        gold_set = set(gold_pages)
        if not accepted_set:
            return 0.0
        return len(accepted_set & gold_set) / len(accepted_set)  # precision

    # Fallback: Inspector produced an answer rather than a filtered evidence set.
    if "answer" in parsed:
        model_answer = parsed["answer"]
        if not isinstance(model_answer, str):
            model_answer = str(model_answer) if model_answer is not None else ""
        return compute_f1(model_answer, gold_answer)

    return 0.0


def compute_vmmoa_answer_reward(prediction: str, gold_answer: str) -> float:
    """R_ans: answer-level F1 of the final answer (global terminal reward).

    Paper Sec. 4.2: "R_ans is the answer-level F1 of ŷ against the reference".
    This is the single global signal shared by all agents under the centralised
    critic (shared advantage, paper Sec. 4.3).

    Args:
        prediction: Synthesizer raw output (JSON with an ``answer`` field, or text).
        gold_answer: Ground-truth answer.

    Returns:
        F1 score in [0.0, 1.0].
    """
    parsed = parse_json_output(prediction)
    model_answer = prediction
    if isinstance(parsed, dict) and "answer" in parsed:
        model_answer = parsed["answer"]
        if not isinstance(model_answer, str):
            model_answer = str(model_answer) if model_answer is not None else ""
    return compute_f1(model_answer, gold_answer)


def compute_vmmoa_iteration_cost(inspector_output: str) -> float:
    """C(τ) contribution for a single trajectory: 1[a_insp ≠ stop].

    Paper Sec. 4.2: "C(τ) = Σ_t 1[a_insp_t ≠ stop] counts non-terminal iterations".
    In the single-pass MAPPO loop each question has one retrieval round; the
    Inspector ``stop`` action corresponds to emitting an ``answer`` (terminal),
    while requesting more information (``information``/``choice``) is a non-stop
    action that would trigger another round in a truly iterative rollout.

    Args:
        inspector_output: Inspector raw output for one question.

    Returns:
        0.0 if the Inspector stopped (answered), 1.0 otherwise.
    """
    parsed = parse_json_output(inspector_output)
    if not isinstance(parsed, dict):
        return 1.0
    # ``stop`` ⟺ Inspector commits to an answer.
    return 0.0 if "answer" in parsed else 1.0


# =============================================================================
# MAPPO Independent-Adapter Reward Functions
# =============================================================================
# These functions implement the zip's compose_mappo_rewards design: each agent
# gets its own DISTINCT reward computed from its own action. No broadcasting.
#
# This replaces the failed shared-return approach (V-MMOA mode gave all agents
# the same r_total, leading to score 2.06). With independent adapters, each
# agent's LoRA receives gradients only from its own reward signal.
# =============================================================================


# ---------------------------------------------------------------------------
# Seeker reward shape selector.
#
# LEGACY (default, MAPPO_SEEKER_REWARD_V7 unset):
#     recall + 0.3*precision - 0.2*max(0, n-10)/10
# GRPO V6 showed two defects in this form:
#   1. The over-selection penalty is DEAD CODE. It needs n > 10, but the seeker
#      picks from at most 10 candidates, so `excess` is always 0.
#   2. With the penalty dead, the only pressure on selection size is
#      +0.3*precision, which is maximised at n=1. Because the v2 training set is
#      filtered to episodes where gold IS among the candidates (100% by
#      construction), shrinking to a single page is nearly free during training
#      -- but at eval gold is present only ~55% of the time. The seeker
#      collapsed 3.51 -> 1.21 pages and lost 12.5pp of Seeker Recall@all.
#
# V7 (MAPPO_SEEKER_REWARD_V7=1): drop the precision term and replace it with a
# budget penalty that actually fires, so selection size is pushed toward a
# target K instead of toward 1:
#     recall - over_w * max(0, n - K) / n_candidates
#
# Calibration of the defaults, from measured candidate recall on the 300-question
# harness (Recall@1 0.3143, @5 0.5286, @10 0.5554): marginal recall per extra
# page is ~5.4pp over ranks 1-5 but only ~0.54pp over ranks 5-10. With
# over_w=0.1 and n_candidates=10 each extra page costs a flat 0.01, so expanding
# the selection pays until marginal recall drops below 0.01 -- an equilibrium
# around 5-6 pages, close to the warm-start operating point of 3.51 that scored
# best end-to-end. K is the free budget below which no penalty applies.
_SEEKER_REWARD_V7 = os.environ.get("MAPPO_SEEKER_REWARD_V7", "0") == "1"
_SEEKER_PREC_W = float(os.environ.get("MAPPO_SEEKER_PREC_W", "0.0" if _SEEKER_REWARD_V7 else "0.3"))
_SEEKER_BUDGET_K = int(os.environ.get("MAPPO_SEEKER_BUDGET_K", "4"))
_SEEKER_OVER_W = float(os.environ.get("MAPPO_SEEKER_OVER_W", "0.1"))


def seeker_reward(
    gold_pages: List[int],
    selected_pages: List[int],
    invalid_json: bool = False,
    num_candidates: int = 10,
) -> float:
    """R_ret: Seeker reward from page selection quality.

    Legacy components (default):
      +recall   |predicted ∩ gold| / |gold|
      +0.3*precision   |predicted ∩ gold| / |predicted|
      -0.5      empty selection
      -1.0      invalid JSON
      -0.2*over  over-selection penalty (excess pages beyond 10) -- never fires,
                 see the note above ``_SEEKER_REWARD_V7``.

    With ``MAPPO_SEEKER_REWARD_V7=1`` the precision term is dropped and the
    over-selection penalty is measured against a budget of
    ``MAPPO_SEEKER_BUDGET_K`` pages rather than against the (unreachable)
    candidate-count ceiling.

    ``num_candidates`` only scales the V7 budget penalty; it is ignored by the
    legacy shape so existing callers keep their exact previous rewards.
    """
    if invalid_json:
        return -1.0

    if not selected_pages:
        return -0.5

    pred_set = set(selected_pages)
    gold_set = set(gold_pages) if gold_pages else set()

    if not gold_set:
        # No gold pages: return NEUTRAL reward (0.0) instead of spurious
        # "select few pages" bonus. The old formula (1.0 - len/20) gave
        # reward 0.95 for selecting 1 page, inflating R_ret to ~0.9 on
        # 39% of training episodes that have gold_pages=[]. This made
        # R_ret dominate R_total and masked R_ans (answer quality)
        # degradation during GRPO training.
        return 0.0

    recall = len(pred_set & gold_set) / len(gold_set)
    precision = len(pred_set & gold_set) / len(pred_set) if pred_set else 0.0

    if _SEEKER_REWARD_V7:
        denom = max(1, num_candidates)
        over_penalty = -_SEEKER_OVER_W * max(0, len(pred_set) - _SEEKER_BUDGET_K) / denom
    else:
        # Over-selection penalty
        excess = max(0, len(selected_pages) - 10)
        over_penalty = -0.2 * excess / 10.0

    reward = recall + _SEEKER_PREC_W * precision + over_penalty
    return max(-1.0, min(1.5, reward))


def inspector_reward(
    pred_sufficient: bool,
    gold_sufficient: bool,
    answer_correct: Optional[bool] = None,
    wrong_reference: bool = False,
) -> float:
    """R_ins: Inspector reward from sufficiency classification.

    Components:
      +1.0   correct classification
      -0.7   false-sufficient (says sufficient when gold says insufficient)
      -0.3   false-insufficient (says insufficient when gold says sufficient)
      +0.5   answer correct (when correctly sufficient)
      -0.2   wrong reference indices
    """
    if pred_sufficient == gold_sufficient:
        r = 1.0
        if pred_sufficient and answer_correct:
            r += 0.5 if answer_correct else 0.0
        if wrong_reference:
            r -= 0.2
    elif pred_sufficient and not gold_sufficient:
        # False-sufficient: unrecoverable error
        r = -0.7
    else:
        # False-insufficient: recoverable (extra round needed)
        r = -0.3

    return max(-1.0, min(1.5, r))


def synthesizer_reward(
    answer_f1: float,
    reference_f1: float = 0.0,
    groundedness: float = 0.0,
    hallucinated_reference: bool = False,
    invalid_json: bool = False,
    overlap_bonus: float = 0.0,
) -> float:
    """R_ans: Synthesizer reward from final answer quality.

    Components:
      +answer_f1      token-level F1 of final answer
      +0.3*reference_f1   precision of cited references
      +0.2*groundedness  answer grounded in evidence
      +overlap_bonus  continuous partial credit based on word overlap
                     with gold answer (breaks bimodal F1 distribution
                     without biasing toward shorter answers)
      -0.3             hallucinated reference
      -1.0             invalid JSON
    """
    if invalid_json:
        return -1.0

    reward = answer_f1
    reward += 0.3 * reference_f1
    reward += 0.2 * groundedness
    reward += overlap_bonus
    if hallucinated_reference:
        reward -= 0.3

    return max(-1.0, min(1.5, reward))


def compose_mappo_rewards(
    *,
    # Seeker signal
    gold_pages: Optional[List[int]],
    selected_pages: Optional[List[int]],
    seeker_invalid_json: bool = False,
    # Inspector signal
    inspector_pred_sufficient: bool,
    inspector_gold_sufficient: bool,
    inspector_answer_correct: Optional[bool] = None,
    inspector_wrong_reference: bool = False,
    # Synthesizer signal
    answer_f1: float,
    reference_f1: float = 0.0,
    groundedness: float = 0.0,
    hallucinated_reference: bool = False,
    synth_invalid_json: bool = False,
    overlap_bonus: float = 0.0,
    # optional per-turn cost applied to the total
    turn_cost: float = 0.0,
) -> Dict[str, float]:
    """Return a DISTINCT reward per agent for one MAPPO episode.

    This is the corrected replacement for the bugged code that made
    r_ans == r_ret == r_ins. Each agent's reward is computed from its own
    action; ``r_total`` is a bookkeeping aggregate only and must NOT be used
    to update the individual agents.
    """
    r_ret = seeker_reward(
        gold_pages or [],
        selected_pages or [],
        invalid_json=seeker_invalid_json,
    )
    r_ins = inspector_reward(
        inspector_pred_sufficient,
        inspector_gold_sufficient,
        answer_correct=inspector_answer_correct,
        wrong_reference=inspector_wrong_reference,
    )
    r_ans = synthesizer_reward(
        answer_f1,
        reference_f1=reference_f1,
        groundedness=groundedness,
        hallucinated_reference=hallucinated_reference,
        invalid_json=synth_invalid_json,
        overlap_bonus=overlap_bonus,
    )

    # Optional cooperative bonus: when the pipeline succeeds (F1 > threshold),
    # all agents get a small bonus. This is ADDED to individual rewards,
    # not used as a replacement.
    coop_bonus = 0.0
    if answer_f1 > MAPPO_COOP_THRESHOLD:
        coop_bonus = MAPPO_COOP_BONUS

    r_ret += coop_bonus
    r_ins += coop_bonus
    r_ans += coop_bonus

    # Clamp each agent reward to [-1, 1.5] (allow positive overshoot from bonus)
    r_ret = max(-1.0, min(1.5, r_ret))
    r_ins = max(-1.0, min(1.5, r_ins))
    r_ans = max(-1.0, min(1.5, r_ans))

    r_total = r_ret + r_ins + r_ans - turn_cost
    return {
        "r_ret": r_ret,   # -> update Seeker only
        "r_ins": r_ins,   # -> update Inspector only
        "r_ans": r_ans,   # -> update Synthesizer only
        "r_total": r_total,  # bookkeeping only, do NOT use for updates
        "coop_bonus": coop_bonus,
    }


def assert_rewards_distinct(rewards: Dict[str, float], tol: float = 1e-9) -> None:
    """Guardrail used in training/eval loops and tests to catch a regression
    of the original bug (all agent rewards collapsing to one value).

    Raises AssertionError if r_ret, r_ins, r_ans are all mutually equal, which
    is the signature of the broadcast bug. (It is statistically possible but
    vanishingly unlikely for three independently-computed rewards to coincide
    exactly; if they do in real data, inspect the pipeline.)

    Exception: when all three rewards are at the clamp boundary (max=1.5 or
    min=-1.0), this is a legitimate "all perfect" or "all worst" case, not a
    broadcast bug. We skip the assertion in that case.
    """
    a, b, c = rewards["r_ret"], rewards["r_ins"], rewards["r_ans"]
    if abs(a - b) <= tol and abs(b - c) <= tol:
        # Allow boundary cases (all at max or all at min)
        is_at_boundary = (abs(a - 1.5) <= tol) or (abs(a - (-1.0)) <= tol)
        if not is_at_boundary:
            raise AssertionError(
                "MAPPO reward bug detected: r_ret == r_ins == r_ans "
                f"({a}, {b}, {c}). Each agent must be scored on its own action."
            )


# =============================================================================
# Batch Reward Computation
# =============================================================================


def compute_batch_rewards(
    seeker_outputs: List[str],
    inspector_outputs: List[str],
    synthesizer_outputs: List[str],
    gold_answers: List[str],
    gold_pages: List[List[int]],
    total_pages: List[int],
    has_sufficient_evidence: List[bool],
    shared_reward_weight: float = 0.3,
    reward_method: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Compute rewards for a batch of samples across all three agents.

    Reward formulation is selected by ``reward_method`` (falls back to the module
    level ``REWARD_METHOD`` switch, configurable via the ``VIDORAG_REWARD_METHOD``
    environment variable):

      * ``"original"`` (default) — ViDoRAG blended rewards.  For each sample the
        final reward for each agent is::

            reward = (1 - shared_weight) * individual_reward + shared_weight * shared_reward

        where ``shared_reward`` is the F1 of the synthesizer answer, encouraging
        all agents to cooperate for a good final answer.  Each agent receives a
        DIFFERENT reward (its own individual signal blended with the shared term).

      * ``"vmmoa"`` — V-MMOA paper hierarchical decomposition (Eq. 1)::

            R(τ) = R_ans + α·R_ret + β·R_ins − λ·C(τ)

        where ``R_ans`` is the global answer F1, ``R_ret`` the Seeker recall@k,
        ``R_ins`` the Inspector evidence precision, and ``C(τ)`` the non-terminal
        iteration count.  Under CTDE MAPPO all three agents share the SAME return
        (centralised critic, shared advantage), so the identical ``R(τ)`` is
        returned for every agent.

    Args:
        seeker_outputs: List of raw Seeker model outputs.
        inspector_outputs: List of raw Inspector model outputs.
        synthesizer_outputs: List of raw Synthesizer model outputs.
        gold_answers: List of ground truth answers.
        gold_pages: List of ground truth evidence page indices per sample.
        total_pages: List of total candidate pages per sample.
        has_sufficient_evidence: List of booleans per sample (original mode only).
        shared_reward_weight: Weight for shared reward (original mode, default 0.3).
        reward_method: ``"original"`` / ``"vmmoa"`` / ``None`` (use global switch).

    Returns:
        Dictionary with keys 'seeker_rewards', 'inspector_rewards',
        'synthesizer_rewards'. Each value is a list of float rewards.
    """
    if reward_method is None:
        reward_method = REWARD_METHOD

    if reward_method == "vmmoa":
        return _compute_batch_rewards_vmmoa(
            seeker_outputs=seeker_outputs,
            inspector_outputs=inspector_outputs,
            synthesizer_outputs=synthesizer_outputs,
            gold_answers=gold_answers,
            gold_pages=gold_pages,
            total_pages=total_pages,
        )

    if reward_method == "mappo_independent":
        return _compute_batch_rewards_mappo_independent(
            seeker_outputs=seeker_outputs,
            inspector_outputs=inspector_outputs,
            synthesizer_outputs=synthesizer_outputs,
            gold_answers=gold_answers,
            gold_pages=gold_pages,
            total_pages=total_pages,
            has_sufficient_evidence=has_sufficient_evidence,
        )

    # ----------------------- original ViDoRAG blended rewards -----------------------
    batch_size = len(seeker_outputs)
    assert len(inspector_outputs) == batch_size
    assert len(synthesizer_outputs) == batch_size
    assert len(gold_answers) == batch_size
    assert len(gold_pages) == batch_size
    assert len(total_pages) == batch_size
    assert len(has_sufficient_evidence) == batch_size

    seeker_rewards: List[float] = []
    inspector_rewards: List[float] = []
    synthesizer_rewards: List[float] = []

    individual_weight = 1.0 - shared_reward_weight

    for i in range(batch_size):
        # Compute individual rewards
        seeker_ind = compute_seeker_reward(
            prediction=seeker_outputs[i],
            gold_pages=gold_pages[i],
            total_pages=total_pages[i],
        )
        inspector_ind = compute_inspector_reward(
            prediction=inspector_outputs[i],
            gold_answer=gold_answers[i],
            has_sufficient_evidence=has_sufficient_evidence[i],
        )
        synthesizer_ind = compute_synthesizer_reward(
            prediction=synthesizer_outputs[i],
            gold_answer=gold_answers[i],
        )

        # Compute shared reward from synthesizer output
        synth_parsed = parse_json_output(synthesizer_outputs[i])
        if synth_parsed and "answer" in synth_parsed:
            final_answer = synth_parsed["answer"]
            if not isinstance(final_answer, str):
                final_answer = str(final_answer) if final_answer is not None else ""
        else:
            final_answer = ""

        shared = compute_shared_reward(final_answer, gold_answers[i])

        # Combine individual and shared rewards
        seeker_r = individual_weight * seeker_ind + shared_reward_weight * shared
        inspector_r = individual_weight * inspector_ind + shared_reward_weight * shared
        synthesizer_r = individual_weight * synthesizer_ind + shared_reward_weight * shared

        # Clamp final combined rewards
        seeker_rewards.append(max(-1.0, min(1.0, seeker_r)))
        inspector_rewards.append(max(-1.0, min(1.0, inspector_r)))
        synthesizer_rewards.append(max(-1.0, min(1.0, synthesizer_r)))

    return {
        "seeker_rewards": seeker_rewards,
        "inspector_rewards": inspector_rewards,
        "synthesizer_rewards": synthesizer_rewards,
    }


def _compute_batch_rewards_vmmoa(
    seeker_outputs: List[str],
    inspector_outputs: List[str],
    synthesizer_outputs: List[str],
    gold_answers: List[str],
    gold_pages: List[List[int]],
    total_pages: List[int],
) -> Dict[str, List[float]]:
    """V-MMOA hierarchical reward decomposition (paper Eq. 1).

        R(τ) = R_ans + α·R_ret + β·R_ins − λ·C(τ)

    All three agents share the SAME return R(τ): under CTDE MAPPO a single
    centralised critic provides the baseline for every agent's policy gradient
    (shared advantage, paper Sec. 4.3), so the identical shaped return is fed to
    the Seeker, Inspector and Synthesizer rollouts.

    Coefficients (α, β, λ) = (0.5, 0.3, 0.05) follow paper Sec. 4.4 and can be
    overridden via the ``VMMOA_ALPHA`` / ``VMMOA_BETA`` / ``VMMOA_LAMBDA``
    environment variables.
    """
    batch_size = len(seeker_outputs)
    assert len(inspector_outputs) == batch_size
    assert len(synthesizer_outputs) == batch_size
    assert len(gold_answers) == batch_size
    assert len(gold_pages) == batch_size
    assert len(total_pages) == batch_size

    seeker_rewards: List[float] = []
    inspector_rewards: List[float] = []
    synthesizer_rewards: List[float] = []

    # Accumulators for logging
    _log_r_ans: List[float] = []
    _log_r_ret: List[float] = []
    _log_r_ins: List[float] = []
    _log_c_tau: List[float] = []

    for i in range(batch_size):
        # Global terminal answer reward (R_ans).
        r_ans = compute_vmmoa_answer_reward(synthesizer_outputs[i], gold_answers[i])
        # Dense Seeker-local shaping (R_ret = recall@k).
        r_ret = compute_vmmoa_seeker_reward(
            seeker_outputs[i], gold_pages[i], total_pages[i]
        )

        # ---- Inspector reward and iteration cost ----
        # In MAPPO the Inspector outputs plain text ("SUFFICIENT"/"NEED_MORE"),
        # not JSON.  We handle both formats: if the output is JSON with page
        # selections we use the original precision logic; otherwise we treat
        # the Inspector as a gatekeeper that accepts the Seeker's selected
        # pages when it says SUFFICIENT.
        _insp_parsed = parse_json_output(inspector_outputs[i])
        _is_plain_text = not isinstance(_insp_parsed, dict)

        if _is_plain_text:
            # MAPPO plain-text routing: SUFFICIENT ⟹ accept seeker's pages.
            _insp_lower = inspector_outputs[i].strip().lower()
            _is_sufficient = "suff" in _insp_lower

            # C(τ): 0 if terminal (SUFFICIENT), 1 if non-terminal.
            c_tau = 0.0 if _is_sufficient else 1.0

            # R_ins: Inspector evidence precision.
            # When SUFFICIENT: precision = |accepted intersect gold| / |accepted|.
            # When NEED_MORE: no accepted set, but reward correct rejection
            #   of non-gold pages with a small bonus (0.1) so the R_ins channel
            #   is not dead and provides gradient signal to the inspector.
            r_ins = 0.0
            if gold_pages[i]:
                _seeker_parsed = parse_json_output(seeker_outputs[i])
                if isinstance(_seeker_parsed, dict) and "choice" in _seeker_parsed:
                    _accepted = extract_page_indices(_seeker_parsed["choice"])
                else:
                    _accepted = [
                        int(n) for n in re.findall(r"\d+", seeker_outputs[i])
                        if int(n) < total_pages[i]
                    ]
                _accepted = [p for p in _accepted if 0 <= p < total_pages[i]]
                if _accepted:
                    _acc_set = set(_accepted)
                    _gold_set = set(gold_pages[i])
                    _overlap = len(_acc_set & _gold_set)
                    if _is_sufficient:
                        # Precision of accepted evidence over gold.
                        r_ins = _overlap / len(_acc_set)
                    elif _overlap == 0:
                        # NEED_MORE with no gold overlap: correct rejection.
                        r_ins = 0.1
                    # else: NEED_MORE but seeker found gold -> false rejection,
                    #   r_ins stays 0.0 (no reward for incorrect rejection).
        else:
            # JSON inspector output (per-agent or future iterative MAPPO).
            r_ins = compute_vmmoa_inspector_reward(
                inspector_outputs[i], gold_pages[i], gold_answers[i]
            )
            c_tau = compute_vmmoa_iteration_cost(inspector_outputs[i])

        # Aggregate the shaped return (paper Eq. 1).
        r_total = r_ans + VMMOA_ALPHA * r_ret + VMMOA_BETA * r_ins - VMMOA_LAMBDA * c_tau

        # Clamp for PPO stability. The V-MMOA return naturally lives in roughly
        # [-λ, 1 + α + β] = [-0.05, 1.8]; we clamp to [-1, 2] to guard outliers
        # without crushing the shaped signal (unlike the original [-1, 1] clamp).
        r_total = max(-1.0, min(2.0, r_total))

        # Shared return across all agents (centralised critic, shared advantage).
        seeker_rewards.append(r_total)
        inspector_rewards.append(r_total)
        synthesizer_rewards.append(r_total)

        _log_r_ans.append(r_ans)
        _log_r_ret.append(r_ret)
        _log_r_ins.append(r_ins)
        _log_c_tau.append(c_tau)

    # Log V-MMOA reward components for monitoring
    _n = max(batch_size, 1)
    _avg_r_ans = sum(_log_r_ans) / _n
    _avg_r_ret = sum(_log_r_ret) / _n
    _avg_r_ins = sum(_log_r_ins) / _n
    _avg_c_tau = sum(_log_c_tau) / _n
    _avg_total = sum(seeker_rewards) / _n
    print(
        f"[V-MMOA] R_ans={_avg_r_ans:.4f}  R_ret={_avg_r_ret:.4f}  "
        f"R_ins={_avg_r_ins:.4f}  C(tau)={_avg_c_tau:.4f}  "
        f"R_total={_avg_total:.4f}  "
        f"gold_pages_nonempty={sum(1 for gp in gold_pages if len(gp) > 0)}/{_n}",
        flush=True,
    )

    return {
        "seeker_rewards": seeker_rewards,
        "inspector_rewards": inspector_rewards,
        "synthesizer_rewards": synthesizer_rewards,
    }


def _compute_batch_rewards_mappo_independent(
    seeker_outputs: List[str],
    inspector_outputs: List[str],
    synthesizer_outputs: List[str],
    gold_answers: List[str],
    gold_pages: List[List[int]],
    total_pages: List[int],
    has_sufficient_evidence: List[bool],
) -> Dict[str, List[float]]:
    """MAPPO independent-adapter reward: each agent gets its OWN distinct reward.

    Uses compose_mappo_rewards() to compute per-agent rewards from each agent's
    own action. No broadcasting — the reward broadcast bug that caused the old
    shared-adapter MAPPO to fail (score 2.06) is caught by assert_rewards_distinct.

    Each agent's reward:
      - Seeker (r_ret):   page recall + precision - over-selection penalty
      - Inspector (r_ins): classification accuracy - false-sufficient penalty
      - Synth (r_ans):    answer F1 + reference F1 + groundedness
      + cooperative bonus when answer F1 > 0.5
    """
    batch_size = len(seeker_outputs)
    assert len(inspector_outputs) == batch_size
    assert len(synthesizer_outputs) == batch_size
    assert len(gold_answers) == batch_size
    assert len(gold_pages) == batch_size
    assert len(total_pages) == batch_size
    assert len(has_sufficient_evidence) == batch_size

    seeker_rewards: List[float] = []
    inspector_rewards: List[float] = []
    synthesizer_rewards: List[float] = []

    _log_r_ret: List[float] = []
    _log_r_ins: List[float] = []
    _log_r_ans: List[float] = []
    _log_coop: List[float] = []

    for i in range(batch_size):
        # ---- Parse Seeker output ----
        # Seeker PPO E2E outputs plain text ("Page 0, Page 1, ..."), while
        # older adapters used JSON {"choice": [0, 1, ...]}. Handle both.
        seeker_parsed = parse_json_output(seeker_outputs[i])
        if seeker_parsed and isinstance(seeker_parsed, dict) and "choice" in seeker_parsed:
            selected_pages = extract_page_indices(seeker_parsed["choice"])
            selected_pages = [p for p in selected_pages if 0 <= p < total_pages[i]]
            seeker_invalid = False
        else:
            # Plain text format: "Page 0, Page 3, Page 5" or "0, 3, 5"
            import re as _re
            numbers = _re.findall(r"\d+", seeker_outputs[i])
            selected_pages = []
            for n in numbers:
                idx = int(n)
                if 0 <= idx < total_pages[i] and idx not in selected_pages:
                    selected_pages.append(idx)
            selected_pages = selected_pages[:total_pages[i]]
            seeker_invalid = len(selected_pages) == 0 and len(seeker_outputs[i].strip()) > 0

        # ---- Parse Inspector output (v6 JSON format) ----
        insp_parsed = parse_json_output(inspector_outputs[i])
        if isinstance(insp_parsed, dict) and "sufficient" in insp_parsed:
            pred_sufficient = bool(insp_parsed["sufficient"])
            insp_answer = insp_parsed.get("answer")
            insp_ref = insp_parsed.get("reference", [])
        elif "sufficient" in inspector_outputs[i].lower():
            # Plain text fallback: "SUFFICIENT" or "NEED_MORE"
            pred_sufficient = "sufficient" in inspector_outputs[i].lower() and "need" not in inspector_outputs[i].lower()
            insp_answer = ""
            insp_ref = []
        else:
            # Fallback: treat as sufficient if answer key exists
            pred_sufficient = isinstance(insp_parsed, dict) and "answer" in insp_parsed
            insp_answer = insp_parsed.get("answer", "") if isinstance(insp_parsed, dict) else ""
            insp_ref = []

        gold_sufficient = has_sufficient_evidence[i]

        # Check if inspector's answer is correct
        answer_correct = None
        if insp_answer and isinstance(insp_answer, str) and gold_answers[i]:
            answer_f1_insp = compute_f1(insp_answer, gold_answers[i])
            answer_correct = answer_f1_insp > 0.5

        # Check wrong reference
        wrong_reference = False
        if isinstance(insp_ref, list) and gold_pages[i]:
            try:
                ref_set = set(int(x) for x in insp_ref if str(x).strip().isdigit())
                gold_set = set(gold_pages[i])
                if ref_set and not (ref_set & gold_set):
                    wrong_reference = True
            except (ValueError, TypeError):
                pass

        # ---- Parse Synthesizer output ----
        # Synth PPO v11 outputs plain text (not JSON), so we handle both formats.
        # If JSON parsing fails, use the raw text as the answer (no format penalty).
        synth_parsed = parse_json_output(synthesizer_outputs[i])
        if synth_parsed and isinstance(synth_parsed, dict) and "answer" in synth_parsed:
            synth_answer = synth_parsed["answer"]
            if not isinstance(synth_answer, str):
                synth_answer = str(synth_answer) if synth_answer is not None else ""
            synth_ref = synth_parsed.get("reference", [])
            synth_invalid = False
        else:
            # Plain text output (Synth SFT v7 / PPO v11 format)
            synth_answer = synthesizer_outputs[i].strip()
            synth_ref = []
            synth_invalid = False  # Don't penalize plain-text format

        # Compute answer F1
        answer_f1 = compute_f1(synth_answer, gold_answers[i]) if synth_answer else 0.0

        # Compute reference F1
        reference_f1 = 0.0
        hallucinated_reference = False
        if isinstance(synth_ref, list) and gold_pages[i]:
            try:
                ref_set = set(int(x) for x in synth_ref if str(x).strip().isdigit())
                gold_set = set(gold_pages[i])
                if ref_set and gold_set:
                    overlap = ref_set & gold_set
                    precision = len(overlap) / len(ref_set)
                    recall = len(overlap) / len(gold_set)
                    if precision + recall > 0:
                        reference_f1 = 2 * precision * recall / (precision + recall)
                if ref_set and not (ref_set & gold_set):
                    hallucinated_reference = True
            except (ValueError, TypeError):
                pass

        # Groundedness: answer F1 > 0 means some grounding
        groundedness = min(1.0, answer_f1)

        # Compose per-agent rewards (DISTINCT, no broadcasting)
        rewards = compose_mappo_rewards(
            gold_pages=gold_pages[i],
            selected_pages=selected_pages,
            seeker_invalid_json=seeker_invalid,
            inspector_pred_sufficient=pred_sufficient,
            inspector_gold_sufficient=gold_sufficient,
            inspector_answer_correct=answer_correct,
            inspector_wrong_reference=wrong_reference,
            answer_f1=answer_f1,
            reference_f1=reference_f1,
            groundedness=groundedness,
            hallucinated_reference=hallucinated_reference,
            synth_invalid_json=synth_invalid,
        )

        # Guardrail: catch reward broadcast bug
        assert_rewards_distinct(rewards)

        seeker_rewards.append(rewards["r_ret"])
        inspector_rewards.append(rewards["r_ins"])
        synthesizer_rewards.append(rewards["r_ans"])

        _log_r_ret.append(rewards["r_ret"])
        _log_r_ins.append(rewards["r_ins"])
        _log_r_ans.append(rewards["r_ans"])
        _log_coop.append(rewards["coop_bonus"])

    # Log MAPPO reward components for monitoring
    _n = max(batch_size, 1)
    _avg_r_ret = sum(_log_r_ret) / _n
    _avg_r_ins = sum(_log_r_ins) / _n
    _avg_r_ans = sum(_log_r_ans) / _n
    _avg_coop = sum(_log_coop) / _n
    print(
        f"[MAPPO-Indep] R_ret={_avg_r_ret:.4f}  R_ins={_avg_r_ins:.4f}  "
        f"R_ans={_avg_r_ans:.4f}  CoopBonus={_avg_coop:.4f}  "
        f"gold_pages_nonempty={sum(1 for gp in gold_pages if len(gp) > 0)}/{_n}",
        flush=True,
    )

    return {
        "seeker_rewards": seeker_rewards,
        "inspector_rewards": inspector_rewards,
        "synthesizer_rewards": synthesizer_rewards,
    }
