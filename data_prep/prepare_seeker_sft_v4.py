#!/usr/bin/env python3
"""
Prepare ViDoRAG Seeker SFT training data v4 — Choice-Only, Short Format.

Based on the degradation analysis (seeker_sft_degradation_analysis.txt), v4 implements:

1. CHOICE-ONLY OUTPUT:
   - v3 output: {"reason": "300-800 tokens of OCR + per-page analysis...", "summary": "...", "choice": [0, 5]}
   - v4 output: {"choice": [0, 5]}
   - Rationale: Token-level loss was dominated by long reasoning (300-800 tokens) vs choice (5-20 tokens).
     SFT trained the model to be a "long explanation generator", not a "stable evidence selector".

2. REDUCED IMAGE COUNT (6 instead of 10-20):
   - Rationale: Too many images cause input truncation, vision token compression, and position bias.
     With 20 images, later pages (index 6-9, 10-19) may not be effectively encoded.

3. SHORT INSTRUCTION:
   - v3: Long instruction with "Page 18 (index 0), Page 8 (index 1)..." page number listing
   - v4: Short instruction with format constraint, no page number listing
   - Rationale: Page number listing confuses the model (original page number vs candidate index).

4. CANDIDATE ORDER RANDOMIZATION:
   - Gold pages are randomly placed at different positions to prevent position bias.
   - Rationale: If gold pages are always at specific positions, model learns position bias.

5. CANDIDATE INDEX ONLY:
   - No original page numbers in instruction or output.
   - Rationale: Page number/index confusion causes off-by-one errors in choice.

6. NO OCR IN OUTPUT:
   - OCR noise is not used as supervision signal.
   - Rationale: OCR text has truncation, errors, and noise that pollutes training signal.

Usage:
  python prepare_sft_data_v4.py
"""

import json
import os
import random
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ============================================================
# Configuration
# ============================================================
DATASET = 'SlideVQA'
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'SlideVQA')
IMAGE_DIR = os.path.join(DATASET_DIR, 'img')
TRAIN_FILE = os.path.join(DATASET_DIR, 'rag_dataset_train.json')

# v4 output directory
OUTPUT_DIR = os.path.join(DATASET_DIR, 'training_data_v4')

# Reuse v2 retrieval cache (same BGE-M3 retrieval, no need to re-run)
RETRIEVAL_CACHE = os.path.join(os.path.join(DATASET_DIR, 'training_data_v2'), 'retrieval_cache.json')

# v4 key parameters
MAX_CANDIDATES = 6          # Reduced from 20 → 6 (analysis: at most 6-8 input images)
GOLD_INJECTION = True       # Always inject gold pages into candidates
RANDOMIZE_ORDER = True      # Shuffle candidate order to prevent position bias
CHOICE_ONLY = True           # Output only {"choice": [...]}, no reason/summary

# System prompt: focused evidence selector
SEEKER_SYSTEM_V4 = (
    "You are an evidence page selector. "
    "Return candidate indices only, not original page numbers."
)

# Short instruction template (no page number listing, explicit format constraint)
INSTRUCTION_TEMPLATE = (
    "Question: {query}\n\n"
    "You are given {num_candidates} candidate page images indexed 0-{max_idx}. "
    "Select all candidate indices that contain evidence needed to answer the question.\n"
    "{image_tokens}\n\n"
    'Return only JSON: {{"choice": [candidate_indices]}}.\n'
    "Use candidate indices only, not original page numbers."
)


# ============================================================
# Utility Functions
# ============================================================

def get_doc_stem(file_name: str) -> str:
    """Strip .pdf extension from filename."""
    if file_name.lower().endswith(".pdf"):
        return file_name[:-4]
    return file_name


def get_page_image_path(doc_stem: str, page_num: int) -> str:
    """Build image path from doc_stem and page_num."""
    return os.path.join(IMAGE_DIR, f"{doc_stem}_{page_num}.jpg")


def extract_doc_and_page(image_path: str) -> Tuple[str, int]:
    """Extract doc_stem and page_num from image path like '/path/to/doc_stem_5.jpg'."""
    basename = os.path.basename(image_path)
    name = os.path.splitext(basename)[0]
    match = re.match(r'^(.+?)_(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def load_dataset() -> List[Dict]:
    """Load SlideVQA training dataset."""
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = data.get("examples", data if isinstance(data, list) else [])
    return examples


def load_retrieval_cache() -> Dict[str, List[str]]:
    """Load pre-computed BGE-M3 retrieval cache from v2."""
    if not os.path.isfile(RETRIEVAL_CACHE):
        raise FileNotFoundError(
            f"Retrieval cache not found: {RETRIEVAL_CACHE}\n"
            "Run prepare_sft_data_v2.py --phase retrieve first."
        )
    with open(RETRIEVAL_CACHE, 'r') as f:
        return json.load(f)


# ============================================================
# v4 Seeker SFT Generation
# ============================================================

def build_candidate_list(
    retrieved_paths: List[str],
    evidence_image_paths: List[str],
    max_candidates: int = MAX_CANDIDATES
) -> Tuple[List[str], List[int]]:
    """Build candidate list with gold page injection and randomization.

    Returns:
        candidate_paths: List of image paths (length <= max_candidates)
        gold_indices: 0-based indices of gold pages within candidate_paths
    """
    seen = set()
    candidates = []

    # Step 1: Add gold/evidence pages first (ensure they're in the candidate set)
    for path in evidence_image_paths:
        if path not in seen and os.path.isfile(path):
            candidates.append(path)
            seen.add(path)

    # Step 2: Fill remaining slots with retrieved pages
    for path in retrieved_paths:
        if len(candidates) >= max_candidates:
            break
        if path not in seen and os.path.isfile(path):
            candidates.append(path)
            seen.add(path)

    if not candidates:
        return [], []

    # Trim to max_candidates
    candidates = candidates[:max_candidates]

    # Step 3: Randomize order to prevent position bias
    # (gold pages should not always be at index 0)
    if RANDOMIZE_ORDER:
        random.shuffle(candidates)

    # Step 4: Find gold indices in the (shuffled) candidate list
    gold_set = set(evidence_image_paths)
    gold_indices = [i for i, p in enumerate(candidates) if p in gold_set]

    return candidates, gold_indices


def generate_seeker_sft_v4(
    entry: Dict,
    retrieved_paths: List[str]
) -> Optional[Dict]:
    """Generate Seeker SFT v4 sample with choice-only output.

    Key differences from v3:
    - Output: {"choice": [0, 5]} (no reason, no summary, no OCR)
    - Instruction: Short, no page number listing, explicit JSON format
    - Candidates: 6 images (not 20), with gold page injection
    - Order: Randomized to prevent position bias
    """
    query = entry["query"]
    evidence_pages = entry["meta_info"]["reference_page"]
    doc_stem = get_doc_stem(entry["meta_info"]["file_name"])

    # Build evidence image paths
    evidence_image_paths = []
    for ep in evidence_pages:
        path = get_page_image_path(doc_stem, ep)
        if os.path.isfile(path):
            evidence_image_paths.append(path)

    if not evidence_image_paths:
        return None

    # Build candidate list with gold injection + randomization
    candidate_paths, gold_indices = build_candidate_list(
        retrieved_paths, evidence_image_paths, MAX_CANDIDATES
    )

    if not candidate_paths or not gold_indices:
        return None

    num_candidates = len(candidate_paths)

    # Build short instruction (no page number listing)
    image_tokens = " ".join(["<image>"] * num_candidates)
    instruction = INSTRUCTION_TEMPLATE.format(
        query=query,
        num_candidates=num_candidates,
        max_idx=num_candidates - 1,
        image_tokens=image_tokens,
    )

    # Build choice-only output
    if CHOICE_ONLY:
        output = json.dumps({"choice": gold_indices})
    else:
        # Optional: choice-first with one-sentence reason
        output = json.dumps({
            "choice": gold_indices,
            "reason": f"The selected pages contain evidence needed to answer the question."
        })

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SEEKER_SYSTEM_V4,
        "history": [],
        "images": candidate_paths,
    }


# ============================================================
# Statistics and Quality Analysis
# ============================================================

def print_statistics(data: List[Dict]):
    """Print detailed statistics about the generated data."""
    print("\n" + "=" * 60)
    print("v4 Data Statistics")
    print("=" * 60)
    print(f"  Total samples: {len(data)}")

    # Candidate count distribution
    candidate_counts = [len(s["images"]) for s in data]
    count_dist = Counter(candidate_counts)
    print(f"  Candidate count distribution: {dict(sorted(count_dist.items()))}")

    # Choice size distribution
    choice_sizes = [len(json.loads(s["output"])["choice"]) for s in data]
    size_dist = Counter(choice_sizes)
    print(f"  Choice size distribution: {dict(sorted(size_dist.items()))}")

    # Gold page position distribution (check for position bias)
    gold_positions = []
    for s in data:
        choice = json.loads(s["output"])["choice"]
        gold_positions.extend(choice)
    pos_dist = Counter(gold_positions)
    print(f"  Gold index distribution (should be uniform):")
    for idx in sorted(pos_dist.keys()):
        pct = pos_dist[idx] / len(data) * 100
        print(f"    Index {idx}: {pos_dist[idx]} ({pct:.1f}%)")

    # Output length statistics
    output_lengths = [len(s["output"]) for s in data]
    avg_len = sum(output_lengths) / len(output_lengths)
    max_len = max(output_lengths)
    min_len = min(output_lengths)
    print(f"  Output length: avg={avg_len:.0f}, min={min_len}, max={max_len} chars")

    # Instruction length statistics
    inst_lengths = [len(s["instruction"]) for s in data]
    avg_inst = sum(inst_lengths) / len(inst_lengths)
    print(f"  Instruction length: avg={avg_inst:.0f} chars")

    # Verify all outputs are valid JSON with "choice" key
    valid = 0
    for s in data:
        try:
            out = json.loads(s["output"])
            if "choice" in out and isinstance(out["choice"], list):
                valid += 1
        except Exception:
            pass
    print(f"  Valid JSON with choice: {valid}/{len(data)} ({valid/len(data)*100:.1f}%)")

    # Check for out-of-range indices
    oob_count = 0
    for s in data:
        out = json.loads(s["output"])
        num_imgs = len(s["images"])
        for idx in out["choice"]:
            if idx < 0 or idx >= num_imgs:
                oob_count += 1
                break
    print(f"  Out-of-range indices: {oob_count}/{len(data)} ({oob_count/len(data)*100:.1f}%)")

    # Compare with v3
    v3_path = os.path.join(os.path.join(DATASET_DIR, 'training_data_v3'), 'seeker_sft_v3_train.json')
    if os.path.isfile(v3_path):
        with open(v3_path) as f:
            v3_data = json.load(f)
        v3_output_lens = [len(s["output"]) for s in v3_data]
        v3_avg = sum(v3_output_lens) / len(v3_output_lens)
        v3_img_counts = [len(s["images"]) for s in v3_data]
        v3_avg_imgs = sum(v3_img_counts) / len(v3_img_counts)
        print(f"\n  Comparison with v3:")
        print(f"    v3 avg output length: {v3_avg:.0f} chars")
        print(f"    v4 avg output length: {avg_len:.0f} chars")
        print(f"    v3 avg image count: {v3_avg_imgs:.1f}")
        print(f"    v4 avg image count: {sum(candidate_counts)/len(data):.1f}")
        print(f"    Output reduction: {(1 - avg_len/v3_avg)*100:.1f}%")
        print(f"    Image reduction: {(1 - sum(candidate_counts)/len(data)/v3_avg_imgs)*100:.1f}%")


# ============================================================
# Main Pipeline
# ============================================================

def split_and_save(data: List[Dict], name: str, split_ratio: float = 0.9):
    """Split data into train/val and save."""
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data, val_data = data[:split_idx], data[split_idx:]

    for suffix, d in [("_train", train_data), ("_val", val_data), ("", data)]:
        path = os.path.join(OUTPUT_DIR, f"{name}{suffix}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(d)} samples to {path}")


def main():
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Seeker SFT Data v4 — Choice-Only, Short Format")
    print("=" * 60)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Max candidates: {MAX_CANDIDATES}")
    print(f"  Choice only: {CHOICE_ONLY}")
    print(f"  Randomize order: {RANDOMIZE_ORDER}")
    print(f"  Gold injection: {GOLD_INJECTION}")

    # Load data
    examples = load_dataset()
    print(f"\nLoaded {len(examples)} training examples.")

    retrieval_cache = load_retrieval_cache()
    print(f"Loaded retrieval cache with {len(retrieval_cache)} entries.")

    # Generate v4 data
    seeker_data = []
    skipped = 0
    no_gold_in_candidates = 0

    for i, entry in enumerate(examples):
        uid = entry.get("uid", str(i))
        retrieved_paths = retrieval_cache.get(uid, [])

        sample = generate_seeker_sft_v4(entry, retrieved_paths)
        if sample:
            seeker_data.append(sample)
        else:
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(examples)} examples, "
                  f"generated {len(seeker_data)} samples")

    print(f"\nGeneration complete:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Generated samples: {len(seeker_data)}")
    print(f"  Skipped: {skipped}")

    # Print statistics
    print_statistics(seeker_data)

    # Save
    print("\n" + "=" * 60)
    print("Saving v4 training data...")
    split_and_save(seeker_data, "seeker_sft_v4")

    # Copy retrieval cache for reproducibility
    import shutil
    cache_dst = os.path.join(OUTPUT_DIR, 'retrieval_cache.json')
    shutil.copy2(RETRIEVAL_CACHE, cache_dst)
    print(f"  Copied retrieval_cache.json to v4 dir")

    print(f"\n{'=' * 60}")
    print(f"Done! All v4 data saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Register dataset 'slidevqa_seeker_sft_v4' in dataset_info.json")
    print(f"  2. Train with qwen3_5_0_8b_seeker_sft_slidevqa_v4.yaml")
    print(f"  3. Evaluate with --mode seeker_sft_v4 --max_samples 100")


if __name__ == "__main__":
    main()
