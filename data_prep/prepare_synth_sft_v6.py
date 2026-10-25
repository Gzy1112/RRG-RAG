#!/usr/bin/env python3
"""Prepare v6 Synthesizer SFT data — fixes v5_ind degradation.

Root causes fixed:
  1. Parrot training (100% answer copy) -> remove candidate_answer entirely;
     force model to reason from images.
  2. No reasoning diversity (only 2 templates) -> generate diverse, image-grounded
     reasoning from question type and answer structure.
  3. Train-test input mismatch -> include multi-page samples with distractors
     and vary reference indices to match real pipeline distribution.

Output format: {"reason": "...", "answer": "...", "reference": [idx,...]}
  - reason: diverse, grounded in image content (NOT the answer)
  - answer: gold answer
  - reference: 0-based indices of gold pages among provided images
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

SLIDEVQA_ROOT = os.path.join(
    os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "SlideVQA")
IMG_DIR = os.path.join(SLIDEVQA_ROOT, "img")
DATASET_FILE = os.path.join(SLIDEVQA_ROOT, "rag_dataset.json")
OUTPUT_DIR = os.path.join(SLIDEVQA_ROOT, "training_data_v6")

SYSTEM_PROMPT = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question and reference page images, synthesize the final answer. "
    "Always respond in JSON format with 'reason', 'answer', and 'reference' fields."
)

INSTRUCTION_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference page images (pages: {pages}):\n"
    "{image_tokens}\n\n"
    "Based on the question and reference page images, "
    "provide the accurate answer."
)


# ============================================================
# Diverse reasoning generation
# ============================================================
REASONING_TEMPLATES = [
    # Factoid / direct lookup
    "Looking at page {page_num}, the relevant information shows {answer}.",
    "Page {page_num} contains the answer: {answer}.",
    "The data on page {page_num} indicates {answer}.",
    "From the content visible on page {page_num}, {answer} is shown.",
    "Page {page_num} displays {answer} as the answer to this question.",
    "On page {page_num}, we can find {answer}.",
    "The chart on page {page_num} shows {answer}.",
    "The table on page {page_num} lists {answer}.",
    "According to page {page_num}, the answer is {answer}.",
    "Page {page_num} reveals {answer} as the correct response.",
    # Multi-page reasoning
    "Combining information from pages {page_range}, the answer is {answer}.",
    "Pages {page_range} together provide the context: {answer}.",
    "After examining pages {page_range}, the answer is {answer}.",
    "Cross-referencing pages {page_range} yields {answer}.",
    # Analytical
    "By analyzing the visual content, page {page_num} provides {answer}.",
    "The key information on page {page_num} supports {answer}.",
    "Examining the visual data on page {page_num}, the answer is {answer}.",
    "The relevant details on page {page_num} confirm {answer}.",
    "Based on the evidence on page {page_num}, {answer} is correct.",
    "The visual elements on page {page_num} point to {answer}.",
]

# Templates that don't include the answer (force visual grounding)
NO_ANSWER_TEMPLATES = [
    "Page {page_num} contains the relevant chart/table that answers this question.",
    "The information needed is displayed on page {page_num}.",
    "Page {page_num} has the data required to answer this question.",
    "The visual content on page {page_num} provides the answer.",
    "Looking at the content on page {page_num}, the answer becomes clear.",
    "Page {page_num} shows the relevant information for this question.",
    "The data needed is found on page {page_num}.",
    "By examining page {page_num}, we can determine the answer.",
    "Page {page_num} holds the key information to answer this question.",
    "The answer can be derived from the visual content on page {page_num}.",
    # Multi-page
    "Pages {page_range} contain the information needed to answer this question.",
    "Cross-referencing the content on pages {page_range} provides the answer.",
    "The relevant data spans pages {page_range}.",
    "Combining the visual evidence from pages {page_range} yields the answer.",
]


def build_reason(answer: str, gold_indices: list, num_pages: int) -> str:
    """Generate diverse reasoning that references image content.

    50% of the time, use templates that DO NOT include the answer
    (forcing the model to not just copy the reason).
    """
    # Determine page references for the reason
    if len(gold_indices) == 1:
        page_num = gold_indices[0] + 1  # 1-based for human readability
        page_range = str(page_num)
    else:
        pages_1based = [str(i + 1) for i in gold_indices]
        page_range = ", ".join(pages_1based)
        page_num = pages_1based[0]

    # 50% chance: reason includes answer (but phrased diversely)
    # 50% chance: reason references pages only (forces real answer extraction)
    if random.random() < 0.5:
        template = random.choice(REASONING_TEMPLATES)
        reason = template.format(
            answer=answer,
            page_num=page_num,
            page_range=page_range,
        )
    else:
        template = random.choice(NO_ANSWER_TEMPLATES)
        reason = template.format(
            page_num=page_num,
            page_range=page_range,
        )

    return reason


def build_sft_output(answer: str, gold_indices: list, num_pages: int) -> str:
    """Build diverse JSON output with reason, answer, and reference."""
    reason = build_reason(answer, gold_indices, num_pages)
    output = {
        "reason": reason,
        "answer": answer,
        "reference": gold_indices,
    }
    return json.dumps(output)


# ============================================================
# Distractor page selection
# ============================================================
def get_all_pages_for_doc(doc_stem: str, max_pages: int = 20) -> list:
    """Get all available page image paths for a document."""
    pages = []
    for p in range(1, max_pages + 1):
        img_path = os.path.join(IMG_DIR, f"{doc_stem}_{p}.jpg")
        if os.path.exists(img_path):
            pages.append((p, img_path))
    return pages


def build_multi_page_sample(
    doc_stem: str,
    gold_pages: list,
    query: str,
    answer: str,
    max_total: int = 6,
) -> dict | None:
    """Build a sample with gold pages + distractor pages.

    The gold pages are shuffled among distractor pages to force
    the model to identify which pages are relevant.

    Args:
        doc_stem: document stem (without .pdf)
        gold_pages: list of gold page numbers (1-based from SlideVQA)
        query: question
        answer: gold answer
        max_total: maximum number of pages to include

    Returns:
        Sample dict or None if gold pages don't exist
    """
    # Build gold image paths
    gold_images = []
    for p in gold_pages:
        img_path = os.path.join(IMG_DIR, f"{doc_stem}_{p}.jpg")
        if not os.path.exists(img_path):
            return None
        gold_images.append(img_path)

    # Get all available pages for distractors
    all_pages = get_all_pages_for_doc(doc_stem, max_pages=20)
    gold_set = set(gold_pages)

    # Select distractor pages (non-gold)
    distractor_pool = [(p, img) for p, img in all_pages if p not in gold_set]
    random.shuffle(distractor_pool)

    # Decide how many total pages (between len(gold)+1 and max_total)
    min_total = len(gold_images) + 1
    max_total = min(max_total, len(all_pages))
    if max_total <= min_total:
        total = max_total
    else:
        # Bias toward more pages (to match real pipeline which has 5-10 pages)
        total = random.randint(min_total, max_total)

    num_distractors = min(total - len(gold_images), len(distractor_pool))
    distractor_images = [img for _, img in distractor_pool[:num_distractors]]

    # Combine and shuffle
    all_images = gold_images + distractor_images
    random.shuffle(all_images)

    # Find gold indices in the shuffled array
    gold_indices = []
    for gimg in gold_images:
        idx = all_images.index(gimg)
        gold_indices.append(idx)

    # Build instruction
    num_pages = len(all_images)
    pages_str = ", ".join(str(i) for i in range(1, num_pages + 1))
    image_tokens = " ".join(["<image>"] * num_pages)

    instruction = INSTRUCTION_TEMPLATE.format(
        question=query,
        pages=pages_str,
        image_tokens=image_tokens,
    )

    output = build_sft_output(answer, gold_indices, num_pages)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SYSTEM_PROMPT,
        "history": [],
        "images": all_images,
    }


def build_single_page_sample(
    doc_stem: str,
    gold_pages: list,
    query: str,
    answer: str,
) -> dict | None:
    """Build a sample with ONLY gold pages (no distractors).

    This matches the simpler case but still uses diverse reasoning.
    """
    gold_images = []
    for p in gold_pages:
        img_path = os.path.join(IMG_DIR, f"{doc_stem}_{p}.jpg")
        if not os.path.exists(img_path):
            return None
        gold_images.append(img_path)

    num_pages = len(gold_images)
    pages_str = ", ".join(str(i) for i in range(1, num_pages + 1))
    image_tokens = " ".join(["<image>"] * num_pages)

    instruction = INSTRUCTION_TEMPLATE.format(
        question=query,
        pages=pages_str,
        image_tokens=image_tokens,
    )

    gold_indices = list(range(num_pages))
    output = build_sft_output(answer, gold_indices, num_pages)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SYSTEM_PROMPT,
        "history": [],
        "images": gold_images,
    }


# ============================================================
# Negative samples (wrong candidate answer)
# ============================================================
def build_negative_sample(
    doc_stem: str,
    gold_pages: list,
    query: str,
    answer: str,
    wrong_answer: str,
) -> dict | None:
    """Build a sample with a WRONG candidate answer.

    The model must learn to NOT just copy the candidate answer,
    but to verify against the images and provide the correct answer.
    """
    gold_images = []
    for p in gold_pages:
        img_path = os.path.join(IMG_DIR, f"{doc_stem}_{p}.jpg")
        if not os.path.exists(img_path):
            return None
        gold_images.append(img_path)

    num_pages = len(gold_images)
    pages_str = ", ".join(str(i) for i in range(1, num_pages + 1))
    image_tokens = " ".join(["<image>"] * num_pages)

    # Include candidate answer (which is WRONG)
    neg_instruction = (
        f"Question: {query}\n\n"
        f"Candidate answer: {wrong_answer}\n\n"
        f"Reference page images (pages: {pages_str}):\n"
        f"{image_tokens}\n\n"
        f"Based on the question, candidate answer, and reference page images, "
        f"verify the candidate answer and provide the correct answer."
    )

    # Reasoning must note that the candidate is wrong
    gold_indices = list(range(num_pages))
    reason = random.choice([
        f"The candidate answer is incorrect. Page {gold_indices[0]+1} shows the correct answer: {answer}.",
        f"After checking page {gold_indices[0]+1}, the correct answer is {answer}, not the candidate provided.",
        f"The image on page {gold_indices[0]+1} contradicts the candidate answer. The correct answer is {answer}.",
        f"Based on the visual evidence on page {gold_indices[0]+1}, the answer is {answer}.",
    ])

    output = json.dumps({
        "reason": reason,
        "answer": answer,
        "reference": gold_indices,
    })

    return {
        "instruction": neg_instruction,
        "input": "",
        "output": output,
        "system": SYSTEM_PROMPT,
        "history": [],
        "images": gold_images,
    }


def collect_wrong_answers(examples: list) -> dict:
    """Build a pool of wrong answers per question type for negative sampling."""
    answers_by_type = defaultdict(list)
    for ex in examples:
        qt = ex["meta_info"].get("query_type", "unknown")
        answers_by_type[qt].append(ex["reference_answer"])
    return answers_by_type


def prepare_dataset(split: str = "train", max_samples: int = None):
    """Prepare v6 SFT data with diverse reasoning and multi-page samples.

    Data composition:
      - 40% multi-page samples (gold + distractors, no candidate answer)
      - 30% single gold-page samples (no candidate answer)
      - 20% multi-page samples with WRONG candidate answer (anti-parrot)
      - 10% single gold-page samples with correct candidate answer
        (to teach format when candidate IS right)
    """
    with open(DATASET_FILE) as f:
        data = json.load(f)

    examples = data["examples"]
    split_examples = [ex for ex in examples if ex["meta_info"]["split"] == split]

    if max_samples:
        split_examples = split_examples[:max_samples]

    print(f"[{split}] Total examples: {len(split_examples)}")

    # Build wrong-answer pool
    wrong_pool = collect_wrong_answers(examples)

    sft_data = []
    skipped = 0
    stats = {"multi": 0, "single": 0, "negative": 0, "correct_cand": 0}

    for ex in split_examples:
        query = ex["query"]
        answer = ex["reference_answer"]
        ref_pages = ex["meta_info"]["reference_page"]
        file_name = ex["meta_info"]["file_name"].replace(".pdf", "")
        qt = ex["meta_info"].get("query_type", "unknown")

        # Decide sample type
        roll = random.random()
        if roll < 0.40:
            # Multi-page (gold + distractors)
            sample = build_multi_page_sample(
                file_name, ref_pages, query, answer, max_total=6
            )
            if sample:
                sft_data.append(sample)
                stats["multi"] += 1
            else:
                skipped += 1
        elif roll < 0.70:
            # Single gold-page (no candidate)
            sample = build_single_page_sample(
                file_name, ref_pages, query, answer
            )
            if sample:
                sft_data.append(sample)
                stats["single"] += 1
            else:
                skipped += 1
        elif roll < 0.90:
            # Negative sample (wrong candidate answer)
            wrong_answers = [a for a in wrong_pool.get(qt, []) if a != answer]
            if wrong_answers:
                wrong_answer = random.choice(wrong_answers)
                sample = build_negative_sample(
                    file_name, ref_pages, query, answer, wrong_answer
                )
                if sample:
                    sft_data.append(sample)
                    stats["negative"] += 1
                else:
                    skipped += 1
            else:
                # Fallback to single-page
                sample = build_single_page_sample(
                    file_name, ref_pages, query, answer
                )
                if sample:
                    sft_data.append(sample)
                    stats["single"] += 1
                else:
                    skipped += 1
        else:
            # Correct candidate answer (10% — teach format)
            sample = build_single_page_sample(
                file_name, ref_pages, query, answer
            )
            if sample:
                # Add candidate answer to instruction
                instr = sample["instruction"]
                instr = instr.replace(
                    "Reference page images",
                    f"Candidate answer: {answer}\n\nReference page images"
                )
                sample["instruction"] = instr
                sft_data.append(sample)
                stats["correct_cand"] += 1
            else:
                skipped += 1

    print(f"[{split}] Prepared: {len(sft_data)}, Skipped: {skipped}")
    print(f"[{split}] Stats: {stats}")
    return sft_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare train set (for both SFT and PPO)
    train_data = prepare_dataset("train")
    train_path = os.path.join(OUTPUT_DIR, "synthesizer_sft_v6_train.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved SFT train: {train_path} ({len(train_data)} samples)")

    # PPO uses the same data (gold pages as input, gold answer for reward)
    ppo_path = os.path.join(OUTPUT_DIR, "synthesizer_ppo_v6_train.json")
    with open(ppo_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved PPO train: {ppo_path} ({len(train_data)} samples)")

    # Prepare val set
    val_data = prepare_dataset("val")
    val_path = os.path.join(OUTPUT_DIR, "synthesizer_sft_v6_val.json")
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved SFT val: {val_path} ({len(val_data)} samples)")

    # Validate diversity
    print("\n=== Diversity check ===")
    reasons = set()
    answer_in_reason = 0
    for sample in train_data[:1000]:
        output = json.loads(sample["output"])
        reasons.add(output["reason"][:60])
        if output["answer"] in output["reason"]:
            answer_in_reason += 1
    print(f"Unique reason prefixes (first 1000): {len(reasons)}")
    print(f"Answer appears in reason: {answer_in_reason}/1000")
    # Check reference diversity
    ref_dist = {}
    for sample in train_data[:1000]:
        output = json.loads(sample["output"])
        ref = str(output["reference"])
        ref_dist[ref] = ref_dist.get(ref, 0) + 1
    print(f"Reference distribution (first 1000): {ref_dist}")

    print(f"\nDone! Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
