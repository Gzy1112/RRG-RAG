from PIL import Image
import os
import re
import json
import sys

from agent.agent_prompt import seeker_prompt,inspector_prompt,answer_prompt
from agent.map_dict import arrangement_map_dict,page_map_dict_normal,page_map_dict

from utils.parse_tool import extract_json, extract_answer_from_text
from utils.image_preprosser import concat_images_with_bbox


# ============================================================
# System Prompts (must match training exactly)
# ============================================================
SEEKER_SYSTEM = (
    "You are an AI assistant with strong ability to find references through images. "
    "Given a question, analyze the provided document page images and select the most relevant pages."
)

# v4: Choice-only, short format (no page numbers, no reasoning, no OCR)
SEEKER_SYSTEM_V4 = (
    "You are an evidence page selector. "
    "Return candidate indices only, not original page numbers."
)

# v5: Maximum candidate images at inference (must match training)
# Increased from 6 to 10 to reduce gold page miss rate (27% → ~15%)
SEEKER_V4_MAX_CANDIDATES = 10

INSPECTOR_SYSTEM = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question and selected page images, determine if there is sufficient information to answer."
)

SYNTHESIZER_SYSTEM = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question, candidate answer, and reference page images, synthesize the final answer."
)

# ============================================================
# Feature Flags (non-destructive, opt-in via env var)
# ============================================================
# When DISABLE_INSPECTOR_SHORTCIRCUIT=1, the Inspector will always route to
# 'synthesizer' instead of 'answer' when it has a valid answer+reference.
# This allows the specialized Synthesizer model to refine the final answer.
# Default: 0 (preserve original short-circuit behavior)
def _is_shortcircuit_disabled():
    """Check at runtime (not import time) so different eval runs can toggle."""
    return os.environ.get("DISABLE_INSPECTOR_SHORTCIRCUIT", "0") == "1"


# SEEKER_V4_MAX_CANDIDATES can be overridden per run via the env var of the same
# name, so an eval can match the candidate count the adapter was trained on
# (v4 SFT used 6) without editing the module default.
def _seeker_max_candidates():
    """Check at runtime (not import time) so different eval runs can toggle."""
    raw = os.environ.get("SEEKER_V4_MAX_CANDIDATES")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return SEEKER_V4_MAX_CANDIDATES


# ---- Inspector v6 contract -------------------------------------------------
# The v6 inspector adapter was trained on SlideVQA/training_data_v6 with the
# prompt/schema pair below (exact strings taken from that dataset; every sample
# used 4 candidate pages). The legacy SFT prompt in Inspector.run asks for prose
# while the legacy parser requires a 'reason' field the v6 schema does not even
# contain, so the adapter never produced parseable JSON: three generations were
# burned per query and the component degraded into a pass-through. Enabling
# INSPECTOR_V6_PROMPT=1 restores the trained contract.
INSPECTOR_SYSTEM_V6 = (
    "You are a sufficiency judge. Decide if the given pages are enough to answer the "
    "question. Use candidate indices only. Do not repeat page text."
)


def _use_inspector_v6_prompt():
    """Check at runtime (not import time) so different eval runs can toggle."""
    return os.environ.get("INSPECTOR_V6_PROMPT", "0") == "1"


# Off by default: every recorded run (and the cycle-2 synthesizer PPO-trained on
# this pipeline) saw candidate_answer=None, so forwarding the judge's answer would
# move the synthesizer off its training distribution.
def _pass_inspector_v6_answer():
    return os.environ.get("INSPECTOR_V6_PASS_ANSWER", "0") == "1"


def _extract_page_num(image_path):
    """Extract page number from image filename like 'docname_42.jpg' -> 42."""
    basename = os.path.basename(image_path)
    match = re.search(r'_(\d+)\.jpg$', basename)
    if match:
        return int(match.group(1))
    return None


def _build_page_descriptions(images):
    """Build page description string and image tokens for seeker (training-format prompt).
    Format: 'Page X (index 0), Page Y (index 1), ...'
    """
    page_nums = [_extract_page_num(img) for img in images]
    descriptions = []
    for idx, (img, pnum) in enumerate(zip(images, page_nums)):
        if pnum is not None:
            descriptions.append(f"Page {pnum} (index {idx})")
        else:
            descriptions.append(f"Image {idx}")
    page_desc_str = ", ".join(descriptions)
    image_tokens = " ".join(["<image>" for _ in images])
    return page_desc_str, image_tokens


def _build_inspector_page_list(images):
    """Build page list string for inspector (training-format prompt).
    Format: '7, 12, 16, 30, 31, 33, 35' (just page numbers, matching training data)
    """
    page_nums = [_extract_page_num(img) for img in images]
    page_strs = []
    for idx, pnum in enumerate(page_nums):
        if pnum is not None:
            page_strs.append(str(pnum))
        else:
            page_strs.append(str(idx))
    page_list_str = ", ".join(page_strs)
    image_tokens = " ".join(["<image>" for _ in images])
    return page_list_str, image_tokens


def _page_nums_to_indices(page_nums_output, images):
    """Convert page numbers from model output to buffer indices.
    
    The SFT model may output:
      - 0-based indices (e.g., [0, 1])        — use directly
      - 1-based page numbers (e.g., [5, 12]) — map via filename
      - String values (e.g., ["2", "5"])     — convert to int first
      - Out-of-bounds values                  — filter out
    
    This function handles all cases robustly.
    """
    # Build mapping from page number -> buffer index
    page_to_idx = {}
    for idx, img in enumerate(images):
        pnum = _extract_page_num(img)
        if pnum is not None:
            page_to_idx[pnum] = idx

    indices = []
    for p in page_nums_output:
        # Convert string to int: "2" -> 2
        if isinstance(p, str) and p.strip().isdigit():
            p = int(p.strip())
        if not isinstance(p, int):
            continue
        # Primary: treat as page number and map to index
        if p in page_to_idx:
            indices.append(page_to_idx[p])
        # Fallback: treat as 0-based index if in range
        elif 0 <= p < len(images):
            indices.append(p)
    return indices


def _candidate_indices(values, candidate_count):
    """Parse model output strictly as 0-based candidate indices.

    Unlike _page_nums_to_indices, values are never reinterpreted as original
    page numbers, so an output of [2] always means candidate #2. Out-of-range
    and non-integer values are dropped.
    """
    indices = []
    for value in values:
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < candidate_count:
            indices.append(value)
    return indices


class Seeker:
    def __init__(self, vlm):
        self.vlm = vlm
        # Use multi-image mode for SFT-trained models (matches training format)
        self._use_sft_format = getattr(vlm, '_is_trained_vlm', False)
        self.seeker_multi_image = self._use_sft_format  # SFT models need separate images

        # v4/v5 format detection: check if adapter path contains 'v4' or 'v5'
        # Both use the same choice-only, short-instruction format.
        # TFR round adapters are warm-started from seeker_sft_v4 and trained on
        # the slidevqa_seeker_ppo_v4 dataset, but their paths carry the cycle name
        # instead of a version tag, so match 'tfr' too — otherwise they are fed
        # the v3 page-number prompt they were never trained on.
        adapter_path = getattr(vlm, '_adapter_path', '') or ''
        path_lower = adapter_path.lower()
        self._use_v4_format = 'v4' in path_lower or 'v5' in path_lower or 'tfr' in path_lower
        if self._use_sft_format:
            fmt = 'v4/v5 choice-only' if self._use_v4_format else 'v3 page-number'
            print(f"[Seeker] prompt format: {fmt} (adapter: {adapter_path})")

        if self.seeker_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query=None, images_path=None, feedback=None):

        if query is not None and images_path is not None:
            self.buffer_images = images_path
            self.query = query
        elif feedback is not None:
            # If buffer is empty, return empty selection to signal no more pages
            if not self.buffer_images:
                return [], "No more pages available", "Buffer exhausted"

        # Edge case: if only 1 image left, auto-select it without model call
        if len(self.buffer_images) == 1:
            selected = self.buffer_images[:]
            self.buffer_images = []
            return selected, "Auto-selected last remaining page", "Only one page in buffer"

        if self._use_v4_format:
            # v4/v5 format: choice-only, short instruction, no page numbers
            # Limit candidates to match training (10 images in v5, was 6 in v4)
            candidates = self.buffer_images[:_seeker_max_candidates()]
                    
            effective_query = self.query
            if feedback is not None:
                effective_query = self.query + '\n\n## Additional Information\n' + feedback
                    
            num_cands = len(candidates)
            image_tokens = " ".join(["<image>"] * num_cands)
            prompt = (
                f"Question: {effective_query}\n\n"
                f"You are given {num_cands} candidate page images indexed 0-{num_cands - 1}. "
                f"Select all candidate indices that contain evidence needed to answer the question.\n"
                f"{image_tokens}\n\n"
                f'Return only JSON: {{"choice": [candidate_indices]}}.\n'
                f"Use candidate indices only, not original page numbers."
            )
            input_images = candidates
            system_prompt = SEEKER_SYSTEM_V4
        elif self._use_sft_format:
            # v3 format: training-compatible prompt with page numbers and reasoning
            effective_query = self.query
            if feedback is not None:
                effective_query = self.query + '\n\n## Additional Information\n' + feedback
                    
            page_desc_str, image_tokens = _build_page_descriptions(self.buffer_images)
            prompt = (
                f"Question: {effective_query}\n\n"
                f"You are given {len(self.buffer_images)} document page images. "
                f"The pages are: {page_desc_str}.\n"
                f"{image_tokens}\n\n"
                f"Analyze these pages and select the ones most relevant to answering the question. "
                f"Output your reasoning and selected page indices in JSON format: "
                f'{{"reason": "...", "summary": "...", "choice": [...]}}'
            )
            input_images = self.buffer_images
            system_prompt = SEEKER_SYSTEM
        else:
            # Original format for API-based models
            if query is not None and images_path is not None:
                prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])
            elif feedback is not None:
                additional_information = self.query + '\n\n## Additional Information\n' + feedback
                prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])

            if self.seeker_multi_image:
                input_images = self.buffer_images
            else:
                input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]
            system_prompt = None

        times = 0
        while True:
            if times > 2:
                # After 3 failed attempts, auto-select first few images as fallback
                fallback_count = min(3, len(self.buffer_images))
                selected_images = self.buffer_images[:fallback_count]
                self.buffer_images = self.buffer_images[fallback_count:]
                return selected_images, "Fallback selection after parse failure", "Could not parse model output"
            times += 1
            select_response = self.vlm.generate(query=prompt, image=input_images, system_prompt=system_prompt)
            print(select_response)
            try:
                select_response_json = extract_json(select_response)

                if self._use_v4_format:
                    # v4 format: choice-only output, indices relative to candidates
                    select_page_num = select_response_json.get('choice', None)
                    if select_page_num is None:
                        raise Exception(f'v4 select json format error: missing choice, length: {len(candidates)}')
                    if not isinstance(select_page_num, list):
                        select_page_num = [select_page_num]
                    # v4 indices are 0-based relative to candidates (no page number mapping)
                    select_page_num = [int(p) for p in select_page_num if isinstance(p, (int, str)) and str(p).strip().isdigit()]
                    select_page_num = [p for p in select_page_num if 0 <= p < len(candidates)]
                    # Deduplicate
                    seen = set()
                    select_page_num = [p for p in select_page_num if not (p in seen or seen.add(p))]
                    if not select_page_num:
                        raise Exception(f'v4 select json format error: no valid choices, length: {len(candidates)}')
                    selected_images = [candidates[page] for page in select_page_num]
                    self.buffer_images = [image for image in self.buffer_images if image not in selected_images]
                    summary = f'Selected {len(selected_images)} page(s) from {len(candidates)} candidates'
                    reason = 'v4 choice-only'
                else:
                    # v3 format: requires reason, summary, and choice
                    reason = select_response_json.get('reason', None)
                    summary = select_response_json.get('summary', None)
                    select_page_num = select_response_json.get('choice', None)
                    if reason is None or summary is None or select_page_num is None:
                        raise Exception(f'select json format error: missing fields, length: {len(self.buffer_images)}')
                    if not isinstance(select_page_num, list):
                        select_page_num = [select_page_num]
                    # The v3 prompt asks for candidate indices, so parse them
                    # strictly as indices: mapping them through page numbers
                    # (_page_nums_to_indices) silently rewrote valid choices.
                    select_page_num = _candidate_indices(select_page_num, len(self.buffer_images))
                    # Deduplicate while preserving order (prevents double-selection)
                    seen = set()
                    select_page_num = [p for p in select_page_num if not (p in seen or seen.add(p))]
                    if not select_page_num:
                        raise Exception(f'select json format error: no valid choices, length: {len(self.buffer_images)}')

                    selected_images = [self.buffer_images[page] for page in select_page_num]
                    self.buffer_images = [image for image in self.buffer_images if image not in selected_images]

            except Exception as e:
                print(e)
                print(select_response)
                print('seeker')
                continue
            break
        
        return selected_images, summary, reason

class Inspector:
    def __init__(self, vlm):
        self.vlm = vlm
        self._use_sft_format = getattr(vlm, '_is_trained_vlm', False)
        self.inspector_multi_image = self._use_sft_format

        if self.inspector_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

        self.buffer_images = []

    def run(self, query, images_path):
        # answer or not, (images, candidate)/feedback
        if len(self.buffer_images) == 0 and len(images_path) == 0:
            return None, None, None
        elif len(images_path) == 0:
            return 'synthesizer', None, self.buffer_images
        elif len(images_path) != 0:
            self.buffer_images.extend(images_path)

        _v6 = bool(self._use_sft_format) and _use_inspector_v6_prompt()

        if _v6:
            # Exact v6 training contract. Page numbers are deliberately absent:
            # the adapter reasons over candidate indices 0..n-1 only.
            _n = len(self.buffer_images)
            image_tokens = " ".join(["<image>"] * _n)
            prompt = (
                f"Question: {query}\n\n"
                f"You are given {_n} candidate page images indexed 0-{_n - 1}.\n"
                f"{image_tokens}\n\n"
                f"Decide whether these pages are sufficient to answer the question. "
                f'Return only JSON: {{"sufficient": true/false, "answer": <answer or null>, '
                f'"reference": [indices], "missing": [hints]}}.'
            )
            input_images = self.buffer_images
            system_prompt = INSPECTOR_SYSTEM_V6
        elif self._use_sft_format:
            # Training-compatible prompt format (matches prepare_vidorag_sft_data.py)
            page_list_str, image_tokens = _build_inspector_page_list(self.buffer_images)
            prompt = (
                f"Question: {query}\n\n"
                f"The following {len(self.buffer_images)} page images have been selected as potentially relevant "
                f"(pages: {page_list_str}).\n"
                f"{image_tokens}\n\n"
                f"Determine if these pages contain sufficient information to answer the question. "
                f"If sufficient, provide the answer. If not, indicate what additional pages should be checked."
            )
            input_images = self.buffer_images
            system_prompt = INSPECTOR_SYSTEM
        else:
            if self.inspector_multi_image:
                input_images = self.buffer_images
            else:
                input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]
            prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])
            system_prompt = None

        times = 0
        while True:
            if times >2:
                # After 3 failed attempts, force-pass to synthesizer with all buffer images
                return 'synthesizer', None, self.buffer_images
            times +=1

            response = self.vlm.generate(query=prompt,image=input_images, system_prompt=system_prompt)
            print(response)
            try:
                response_json = extract_json(response)
                if _v6 and isinstance(response_json, dict) and 'sufficient' in response_json:
                    # v6 schema: {"sufficient": bool, "answer": str|null,
                    #             "reference": [candidate indices], "missing": [hints]}
                    sufficient = response_json.get('sufficient')
                    if isinstance(sufficient, str):
                        sufficient = sufficient.strip().lower() in ('true', 'yes', '1')
                    answer = response_json.get('answer')
                    ref = response_json.get('reference') or []
                    if not isinstance(ref, list):
                        ref = [ref]
                    ref_indices = _candidate_indices(ref, len(self.buffer_images))
                    if not sufficient:
                        # Insufficient: keep the legacy fall-through (all pages, no
                        # candidate answer). Routing to the seeker instead would
                        # activate a recovery loop no recorded run has ever used.
                        print('[inspector-v6] insufficient -> pass all pages')
                        return 'synthesizer', None, self.buffer_images
                    ref_images = ([self.buffer_images[i] for i in ref_indices]
                                  if ref_indices else self.buffer_images)
                    cand_answer = answer if isinstance(answer, str) and answer.strip() else None
                    if not _pass_inspector_v6_answer():
                        cand_answer = None
                    print(f'[inspector-v6] sufficient ref={ref_indices} '
                          f'-> {len(ref_images)}/{len(self.buffer_images)} page(s)')
                    return 'synthesizer', cand_answer, ref_images
                # thought
                reason = response_json.get('reason',None)
                # if feedback
                info = response_json.get('information',None)
                choice = response_json.get('choice',None)
                # can answer
                answer = response_json.get('answer',None)
                ref = response_json.get('reference',None)

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    if not isinstance(ref, list):
                        ref = [ref]
                    # Convert page numbers to indices
                    ref_indices = _page_nums_to_indices(ref, self.buffer_images)
                    if not ref_indices:
                        # Fallback: try treating as raw indices
                        ref_indices = [p for p in ref if isinstance(p, int) and 0 <= p < len(self.buffer_images)]
                    if len(ref_indices) == 0:
                        # If still no valid refs, use all buffer images
                        return 'synthesizer', answer, self.buffer_images
                    if _is_shortcircuit_disabled():
                        # Experiment: always route to Synthesizer (never short-circuit)
                        ref_images = [self.buffer_images[page] for page in ref_indices]
                        return 'synthesizer', answer, ref_images
                    if len(ref_indices) == len(self.buffer_images):
                        return 'answer', answer, self.buffer_images
                    else:
                        ref_images = [self.buffer_images[page] for page in ref_indices]
                        return 'synthesizer', answer, ref_images
                elif info is not None and choice is not None:
                    if not isinstance(choice, list):
                        choice = [choice]
                    # Inspector outputs page numbers in choice (pages to check next).
                    # Convert to buffer indices.
                    valid_indices = _page_nums_to_indices(choice, self.buffer_images)
                    if not valid_indices:
                        # Fallback: try treating as raw indices
                        valid_indices = [p for p in choice if isinstance(p, int) and 0 <= p < len(self.buffer_images)]
                    if valid_indices:
                        self.buffer_images = [self.buffer_images[page] for page in valid_indices]
                    # Return to seeker for more info regardless
                    return 'seeker', info, self.buffer_images
                else:
                    # Model output has reason but neither answer+ref nor info+choice
                    # Treat as "insufficient info, need more pages"
                    return 'seeker', reason, self.buffer_images
            
            except Exception as e:
                print(e)
                print("inspector")
                continue

class Synthesizer:
    def __init__(self, vlm):
        self.vlm = vlm
        self._use_sft_format = getattr(vlm, '_is_trained_vlm', False)
        self.synthesizer_multi_image = self._use_sft_format
        if self.synthesizer_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query, candidate_answer, ref_images):
        if self._use_sft_format:
            # Training-compatible prompt format (matches prepare_vidorag_sft_data.py)
            page_list_str, image_tokens = _build_inspector_page_list(ref_images)
            if candidate_answer is not None:
                effective_query = f"Question: {query}\n\nCandidate answer: {candidate_answer}\n\n"
            else:
                effective_query = f"Question: {query}\n\n"
            prompt = (
                f"{effective_query}"
                f"Reference page images (pages: {page_list_str}):\n"
                f"{image_tokens}\n\n"
                f"Based on the question, candidate answer, and reference page images, "
                f"synthesize the final accurate answer."
            )
            input_images = ref_images
            system_prompt = SYNTHESIZER_SYSTEM
        else:
            if candidate_answer is not None:
                query = query + '\n\n## Related Information\n' + candidate_answer
            prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

            if self.synthesizer_multi_image:
                input_images = ref_images
            else:
                input_images = [concat_images_with_bbox(ref_images, arrangement=arrangement_map_dict[len(ref_images)], scale=1, line_width=40)]
            system_prompt = None
        
        times = 0
        while True:
            if times > 2:
                # Last-resort: try to salvage an answer from verbose text via
                # regex. Small SFT models often produce valid natural-language
                # answers but fail to wrap them in the expected JSON envelope.
                salvaged = extract_answer_from_text(final_answer_response)
                if salvaged:
                    return "regex_fallback", salvaged
                # Return the raw response as answer if regex also fails
                return "parse_failed", final_answer_response if final_answer_response else "No answer"
            times += 1
            final_answer_response = self.vlm.generate(query=prompt,image=input_images, system_prompt=system_prompt)
            print(final_answer_response)
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None :
                    raise Exception('Synthesizer time out')
                return reason, answer
            except Exception as e:
                print(e)
                print(final_answer_response)
                print("answer")
                # Solution B: try regex extraction for verbose responses.
                salvaged = extract_answer_from_text(final_answer_response)
                if salvaged:
                    return "regex_fallback", salvaged
                # Solution A (v4): model outputs just the answer string
                # (no JSON envelope). If the response is short and clean,
                # use it directly as the answer without retrying.
                if (final_answer_response
                        and len(final_answer_response) < 200
                        and '\n' not in final_answer_response):
                    return "direct_answer", final_answer_response.strip()
                continue

class ViDoRAG_Agents:
    def __init__(self, vlm=None, seeker_vlm=None, inspector_vlm=None, synthesizer_vlm=None):
        """Initialize ViDoRAG agents.
        
        Args:
            vlm: Shared VLM for all agents (backward compatible).
            seeker_vlm: Dedicated VLM for the Seeker agent.
            inspector_vlm: Dedicated VLM for the Inspector agent.
            synthesizer_vlm: Dedicated VLM for the Synthesizer agent.
        
        If per-agent VLMs are provided, they take priority over the shared vlm.
        """
        self.seeker = Seeker(seeker_vlm if seeker_vlm is not None else vlm)
        self.inspector = Inspector(inspector_vlm if inspector_vlm is not None else vlm)
        self.synthesizer = Synthesizer(synthesizer_vlm if synthesizer_vlm is not None else vlm)

    def run_agent(self, query, images_path, return_trace=False):
        # initial
        self.seeker.buffer_images = None
        self.inspector.buffer_images = []

        # Track what the seeker actually selected (vs the candidate pool) so
        # evaluation can separate candidate recall from seeker recall.
        trace = {
            "candidate_images": list(images_path),
            "seeker_selected_images": [],
            "final_evidence_images": [],
            "iterations": 0,
            "termination": None,
        }

        selected_images, summary, reason = self.seeker.run(query=query, images_path=images_path)
        trace["seeker_selected_images"].extend(selected_images)
        # iter with max iteration limit to prevent infinite loops
        max_iterations = 5
        iteration = 0
        while True:
            iteration += 1
            trace["iterations"] = iteration
            if iteration > max_iterations:
                # Force synthesizer with whatever info we have
                print(f"[Agent] Max iterations ({max_iterations}) reached, forcing synthesizer")
                evidence = self.inspector.buffer_images if self.inspector.buffer_images else selected_images
                reason, answer = self.synthesizer.run(query, summary, evidence)
                trace["final_evidence_images"] = list(evidence)
                trace["termination"] = "max_iterations"
                return {"answer": answer, "trace": trace} if return_trace else answer

            status, information, images = self.inspector.run(query, selected_images)
            if status == 'answer':
                trace["final_evidence_images"] = list(self.inspector.buffer_images)
                trace["termination"] = "inspector_answer"
                return {"answer": information, "trace": trace} if return_trace else information
            elif status == 'synthesizer':
                reason, answer = self.synthesizer.run(query, information, images)
                trace["final_evidence_images"] = list(images or [])
                trace["termination"] = "synthesizer"
                return {"answer": answer, "trace": trace} if return_trace else answer
            elif status == 'seeker':
                selected_images, summary, reason = self.seeker.run(feedback=information)
                trace["seeker_selected_images"].extend(
                    image for image in selected_images
                    if image not in trace["seeker_selected_images"]
                )
                continue
            else:
                print('No related information')
                trace["termination"] = "no_information"
                return {"answer": None, "trace": trace} if return_trace else None

if __name__ == '__main__':
    from llms.llm import LLM
    vlm = LLM('qwen-vl-max')
    agent = ViDoRAG_Agents(vlm)
    re=agent.run_agent(query='Who is Tim?', images_path=['./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_2.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_3.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_4.jpg'])
    print(re)
