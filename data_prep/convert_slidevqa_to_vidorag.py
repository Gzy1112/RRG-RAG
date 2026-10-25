"""
Convert SlideVQA Parquet dataset into ViDoRAG-compatible format.

This script performs two key transformations:
  Step 1 — Extract page images from Parquet struct<bytes, path> columns
           into individual JPG files on disk.
  Step 2 — Build rag_dataset.json in ViDoRAG's format
           (uid / query / reference_answer / meta_info).

Output structure:
    data/SlideVQA/
    ├── img/                     # Extracted page images ({deck_name}_{page}.jpg)
    ├── rag_dataset.json         # ViDoRAG-format dataset (all splits)
    ├── rag_dataset_train.json   # Train split only
    ├── rag_dataset_val.json     # Val split only
    ├── rag_dataset_test.json    # Test split only
    └── data/                    # Original Parquet files (untouched)

Usage:
    python convert_slidevqa_to_vidorag.py
    python convert_slidevqa_to_vidorag.py --skip-images   # skip Step 1
"""

import io
import os
import json
import argparse
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

SLIDEVQA_DIR = os.path.join(
    os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "SlideVQA")
PARQUET_DATA_DIR = os.path.join(SLIDEVQA_DIR, "data")
IMAGE_OUTPUT_DIR = os.path.join(SLIDEVQA_DIR, "img")
RAG_DATASET_PATH = os.path.join(SLIDEVQA_DIR, "rag_dataset.json")

MAX_PAGES = 20  # SlideVQA has page_1 through page_20


# ============================================================
# Helpers
# ============================================================

def _split_from_filename(fname: str) -> str:
    """Determine split ('train', 'val', 'test') from parquet filename."""
    if fname.startswith("train"):
        return "train"
    if fname.startswith("val"):
        return "val"
    if fname.startswith("test"):
        return "test"
    return "unknown"


def _find_parquet_files(parquet_dir: str) -> List[str]:
    """Return sorted list of parquet file paths."""
    return sorted(
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    )


# ============================================================
# Step 1: Extract Images
# ============================================================

def extract_images(parquet_dir: str, image_dir: str) -> Dict[str, List[int]]:
    """
    Extract all page images from SlideVQA Parquet files.

    Each parquet row contains page_1..page_20 as struct<bytes, path>.
    We decode the raw JPEG bytes and save as {deck_name}_{page_num}.jpg.

    Decks are deduplicated — if the same deck appears in multiple QA pairs
    or parquet shards, its pages are extracted only once.

    Returns:
        Dict mapping deck_name -> sorted list of available page numbers (1-indexed)
    """
    os.makedirs(image_dir, exist_ok=True)

    parquet_files = _find_parquet_files(parquet_dir)
    print(f"\n{'='*70}")
    print(f"Step 1: Extracting images from {len(parquet_files)} Parquet files")
    print(f"Output: {image_dir}")
    print(f"{'='*70}\n")

    # Track extracted decks/pages to avoid re-saving
    deck_pages: Dict[str, Set[int]] = {}
    total_saved = 0
    total_skipped = 0

    for pq_file in tqdm(parquet_files, desc="Parquet files"):
        table = pq.read_table(pq_file)
        rows = table.to_pylist()

        for row in tqdm(rows, desc=f"  {os.path.basename(pq_file)}", leave=False):
            deck_name = row["deck_name"]

            if deck_name not in deck_pages:
                deck_pages[deck_name] = set()

            for page_num in range(1, MAX_PAGES + 1):
                page_key = f"page_{page_num}"
                page_data = row.get(page_key)

                if page_data is None:
                    continue

                img_bytes = page_data.get("bytes") if isinstance(page_data, dict) else None
                if img_bytes is None:
                    continue

                # Skip if already extracted
                if page_num in deck_pages[deck_name]:
                    continue

                img_path = os.path.join(image_dir, f"{deck_name}_{page_num}.jpg")

                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(img_path, "JPEG", quality=95)
                    deck_pages[deck_name].add(page_num)
                    total_saved += 1
                except Exception as e:
                    total_skipped += 1
                    if total_skipped <= 5:
                        print(f"  WARNING: Failed {deck_name}_{page_num}: {e}")

    print(f"\n  Extracted {total_saved:,} images from {len(deck_pages):,} unique decks")
    if total_skipped:
        print(f"  Skipped {total_skipped} images due to errors")

    return {deck: sorted(pages) for deck, pages in deck_pages.items()}


def rebuild_deck_pages_from_disk(image_dir: str) -> Dict[str, List[int]]:
    """Reconstruct deck_pages mapping from existing image files on disk."""
    deck_pages: Dict[str, Set[int]] = {}
    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg"):
            continue
        # Filename: {deck_name}_{page_num}.jpg
        # Use rsplit to handle deck names containing underscores
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        deck_name, page_part = parts
        page_num_str = page_part.replace(".jpg", "")
        try:
            page_num = int(page_num_str)
        except ValueError:
            continue
        deck_pages.setdefault(deck_name, set()).add(page_num)

    return {deck: sorted(pages) for deck, pages in deck_pages.items()}


# ============================================================
# Step 2: Build rag_dataset.json
# ============================================================

def build_rag_dataset(
    parquet_dir: str,
    output_path: str,
    deck_pages: Dict[str, List[int]],
) -> Dict:
    """
    Build rag_dataset.json in ViDoRAG format from SlideVQA data.

    Each QA pair becomes one entry:
    {
        "uid": "{deck_name}_{qa_id}",
        "query": "<question>",
        "reference_answer": "<answer>",
        "meta_info": {
            "file_name": "{deck_name}.pdf",
            "reference_page": [4, 5],          # 1-indexed, filtered to existing images
            "source_type": "slide",
            "query_type": "slide_qa",
            "split": "train",                  # train / val / test
            "deck_url": "...",
            "arithmetic_expression": "...",
            "original_evidence_pages": [4, 5]  # unfiltered, for reference
        }
    }

    Also writes per-split files: rag_dataset_{split}.json
    """
    print(f"\n{'='*70}")
    print(f"Step 2: Building rag_dataset.json")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    parquet_files = _find_parquet_files(parquet_dir)
    examples: List[Dict] = []
    seen_uids: Set[str] = set()
    split_examples: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for pq_file in tqdm(parquet_files, desc="Building dataset"):
        split = _split_from_filename(os.path.basename(pq_file))
        table = pq.read_table(pq_file)
        rows = table.to_pylist()

        for row in rows:
            deck_name = row["deck_name"]
            qa_id = row["qa_id"]
            uid = f"{deck_name}_{qa_id}"

            if uid in seen_uids:
                continue
            seen_uids.add(uid)

            question = row.get("question", "")
            answer = row.get("answer", "")
            evidence_pages = row.get("evidence_pages", [])

            # Filter evidence pages to only those with extracted images
            available = set(deck_pages.get(deck_name, []))
            valid_evidence = [p for p in evidence_pages if p in available]

            if not valid_evidence:
                continue

            entry = {
                "uid": uid,
                "query": question,
                "reference_answer": answer,
                "meta_info": {
                    "file_name": f"{deck_name}.pdf",
                    "reference_page": valid_evidence,
                    "source_type": "slide",
                    "query_type": "slide_qa",
                    "split": split,
                    "deck_url": row.get("deck_url", ""),
                    "arithmetic_expression": row.get("arithmetic_expression"),
                    "original_evidence_pages": evidence_pages,
                },
            }
            examples.append(entry)
            if split in split_examples:
                split_examples[split].append(entry)

    # Write combined dataset
    dataset = {"examples": examples}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Write per-split datasets
    base_dir = os.path.dirname(output_path)
    for split, split_exs in split_examples.items():
        split_path = os.path.join(base_dir, f"rag_dataset_{split}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({"examples": split_exs}, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"\n  Total examples: {len(examples):,}")
    for split, split_exs in split_examples.items():
        print(f"    {split}: {len(split_exs):,} examples")
    print(f"  Unique decks with images: {len(deck_pages):,}")

    if examples:
        ref_counts = [len(e["meta_info"]["reference_page"]) for e in examples]
        print(
            f"  Evidence pages/question: min={min(ref_counts)}, "
            f"max={max(ref_counts)}, avg={sum(ref_counts)/len(ref_counts):.1f}"
        )

    print(f"\n  Files written:")
    print(f"    {output_path}")
    for split in split_examples:
        print(f"    {os.path.join(base_dir, f'rag_dataset_{split}.json')}")

    return dataset


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert SlideVQA Parquet → ViDoRAG format (images + rag_dataset.json)"
    )
    parser.add_argument(
        "--parquet-dir",
        default=PARQUET_DATA_DIR,
        help="Directory containing SlideVQA Parquet files",
    )
    parser.add_argument(
        "--image-dir",
        default=IMAGE_OUTPUT_DIR,
        help="Output directory for extracted page images",
    )
    parser.add_argument(
        "--output",
        default=RAG_DATASET_PATH,
        help="Output path for rag_dataset.json",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image extraction (rebuild deck_pages from existing files)",
    )
    args = parser.parse_args()

    print("SlideVQA → ViDoRAG Format Conversion")
    print(f"  Parquet dir: {args.parquet_dir}")
    print(f"  Image dir:   {args.image_dir}")
    print(f"  Output:      {args.output}")

    # Step 1: Extract images
    if args.skip_images:
        print("\nSkipping image extraction (--skip-images)")
        if os.path.isdir(args.image_dir):
            deck_pages = rebuild_deck_pages_from_disk(args.image_dir)
            total_imgs = sum(len(p) for p in deck_pages.values())
            print(f"  Found {total_imgs:,} existing images across {len(deck_pages):,} decks")
        else:
            print(f"  WARNING: Image dir {args.image_dir} does not exist — no images available")
            deck_pages = {}
    else:
        deck_pages = extract_images(args.parquet_dir, args.image_dir)

    # Step 2: Build rag_dataset.json
    build_rag_dataset(args.parquet_dir, args.output, deck_pages)

    print(f"\n{'='*70}")
    print("Conversion complete!")
    print(f"  Images:  {args.image_dir}/")
    print(f"  Dataset: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
