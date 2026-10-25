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
# Reward Method Switch (original ViDoRAG vs HRD reward)
# =============================================================================
# Controls which reward formulation is used for PPO / MAPPO training.
#
#   "original" : ViDoRAG per-agent blended rewards.
#                Each agent receives  (1 - w) * individual + w * shared_answer_F1
#                with different individual signals per agent (see compute_batch_rewards).
#
#   "hrd"      : HRD (paper Sec. 4.2, Eq. 1)
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
#     export VIDORAG_REWARD_METHOD=hrd   # or "original" (default)
REWARD_METHOD = os.environ.get("VIDORAG_REWARD_METHOD", "original").lower()

# HRD reward coefficients (paper Sec. 4.4 / Appendix B).
HRD_ALPHA = float(os.environ.get("HRD_ALPHA", "0.5"))    # Seeker recall weight (α)
HRD_BETA = float(os.environ.get("HRD_BETA", "0.3"))     # Inspector precision weight (β)
HRD_LAMBDA = float(os.environ.get("HRD_LAMBDA", "0.05"))  # iteration-cost penalty (λ)


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

    The Seeker outputs either JSON with a "choice" field, or plain text in
    "Page X, Page Y" format (enforced by the constrained logits processor).

    Reward components:
    - Base reward: Page recall = |predicted ∩ gold| / |gold|
    - Precision bonus: +0.3 * (|predicted ∩ gold| / |predicted|)
    - Format penalty: -1.0 if no valid page indices can be extracted
    - Over-selection penalty: -0.2 * max(0, |predicted| - max_pages) / max_pages
    - Empty penalty: -0.5 if no pages selected

    Args:
        prediction: Model's raw output text (JSON with "choice" or "Page X, Page Y").
        gold_pages: Ground truth evidence page indices.
        total_pages: Total candidate pages available.
        format_weight: Penalty for format violations (default: -1.0).
        max_pages: Maximum allowed page selections before penalty.

    Returns:
        Reward value clamped to [-1.0, 1.0].
    """
    predicted_pages: List[int] = []

    # Try JSON parsing first
    parsed = parse_json_output(prediction)
    if isinstance(parsed, dict) and "choice" in parsed:
        predicted_pages = extract_page_indices(parsed["choice"])
    else:
        # Fallback: extract integers from plain text (e.g. "Page 0, Page 1")
        predicted_pages = [
            int(n) for n in re.findall(r"\d+", prediction)
            if int(n) < total_pages
        ]
        predicted_pages = _deduplicate(predicted_pages)

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

    The Synthesizer outputs either JSON with an "answer" field, or plain text
    containing the answer directly (e.g. "8x").

    Reward components:
    - Base reward: F1 score between predicted and gold answer
    - EM bonus: +0.3 if exact match
    - Length penalty: -0.3 if answer exceeds max_words
    - Empty penalty: -0.5 if answer is empty or only whitespace

    Args:
        prediction: Model's raw output text (JSON with "answer" or plain text).
        gold_answer: Ground truth answer.
        max_words: Maximum allowed word count before penalty.
        format_weight: Penalty for format violations (default: -1.0).

    Returns:
        Reward value clamped to [-1.0, 1.0].
    """
    model_answer = ""

    # Try JSON parsing first
    parsed = parse_json_output(prediction)
    if isinstance(parsed, dict) and "answer" in parsed:
        model_answer = parsed["answer"]
        if not isinstance(model_answer, str):
            model_answer = str(model_answer) if model_answer is not None else ""
    else:
        # Fallback: treat the raw text as the answer
        model_answer = prediction.strip()

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
# HRD Reward Components (hierarchical decomposition, Eq. 1)
# =============================================================================
# These implement the three terms of the HRD return
#     R(τ) = R_ans + α·R_ret + β·R_ins − λ·C(τ)
# R_ans is the global terminal answer-F1, R_ret / R_ins are dense agent-local
# shaping signals (recall@k for the Seeker, precision for the Inspector), and
# C(τ) penalises non-terminal iterations to encourage early stopping.
# =============================================================================


def compute_hrd_seeker_reward(
    prediction: str,
    gold_pages: List[int],
    total_pages: int,
    k: int = 8,
) -> float:
    """R_ret: recall@k of the Seeker's retrieved pages against gold evidence.

    Paper Sec. 4.2: "R_ret is the recall@k of R_t against gold evidence pages".
    Pure recall (no precision bonus, no format penalty) — the HRD shaping
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


def compute_hrd_inspector_reward(
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


def compute_hrd_answer_reward(prediction: str, gold_answer: str) -> float:
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


def compute_hrd_iteration_cost(inspector_output: str) -> float:
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

      * ``"hrd"`` — HRD hierarchical decomposition (Eq. 1)::

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
        reward_method: ``"original"`` / ``"hrd"`` / ``None`` (use global switch).

    Returns:
        Dictionary with keys 'seeker_rewards', 'inspector_rewards',
        'synthesizer_rewards'. Each value is a list of float rewards.
    """
    if reward_method is None:
        reward_method = REWARD_METHOD

    if reward_method == "hrd":
        return _compute_batch_rewards_hrd(
            seeker_outputs=seeker_outputs,
            inspector_outputs=inspector_outputs,
            synthesizer_outputs=synthesizer_outputs,
            gold_answers=gold_answers,
            gold_pages=gold_pages,
            total_pages=total_pages,
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

        # Compute shared reward from synthesizer output.
        # Handles both JSON (with "answer" field) and plain text outputs.
        synth_parsed = parse_json_output(synthesizer_outputs[i])
        if isinstance(synth_parsed, dict) and "answer" in synth_parsed:
            final_answer = synth_parsed["answer"]
            if not isinstance(final_answer, str):
                final_answer = str(final_answer) if final_answer is not None else ""
        else:
            # Fallback: use the raw synthesizer text as the final answer
            final_answer = synthesizer_outputs[i].strip() if synthesizer_outputs[i] else ""

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


def _compute_batch_rewards_hrd(
    seeker_outputs: List[str],
    inspector_outputs: List[str],
    synthesizer_outputs: List[str],
    gold_answers: List[str],
    gold_pages: List[List[int]],
    total_pages: List[int],
) -> Dict[str, List[float]]:
    """HRD hierarchical reward decomposition (paper Eq. 1).

        R(τ) = R_ans + α·R_ret + β·R_ins − λ·C(τ)

    All three agents share the SAME return R(τ): under CTDE MAPPO a single
    centralised critic provides the baseline for every agent's policy gradient
    (shared advantage, paper Sec. 4.3), so the identical shaped return is fed to
    the Seeker, Inspector and Synthesizer rollouts.

    Coefficients (α, β, λ) = (0.5, 0.3, 0.05) follow paper Sec. 4.4 and can be
    overridden via the ``HRD_ALPHA`` / ``HRD_BETA`` / ``HRD_LAMBDA``
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
        r_ans = compute_hrd_answer_reward(synthesizer_outputs[i], gold_answers[i])
        # Dense Seeker-local shaping (R_ret = recall@k).
        r_ret = compute_hrd_seeker_reward(
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
            r_ins = compute_hrd_inspector_reward(
                inspector_outputs[i], gold_pages[i], gold_answers[i]
            )
            c_tau = compute_hrd_iteration_cost(inspector_outputs[i])

        # Aggregate the shaped return (paper Eq. 1).
        r_total = r_ans + HRD_ALPHA * r_ret + HRD_BETA * r_ins - HRD_LAMBDA * c_tau

        # Clamp for PPO stability. The HRD return naturally lives in roughly
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

    # Log HRD reward components for monitoring
    _n = max(batch_size, 1)
    _avg_r_ans = sum(_log_r_ans) / _n
    _avg_r_ret = sum(_log_r_ret) / _n
    _avg_r_ins = sum(_log_r_ins) / _n
    _avg_c_tau = sum(_log_c_tau) / _n
    _avg_total = sum(seeker_rewards) / _n
    print(
        f"[HRD] R_ans={_avg_r_ans:.4f}  R_ret={_avg_r_ret:.4f}  "
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
