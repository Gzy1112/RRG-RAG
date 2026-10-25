#!/usr/bin/env python3
"""
Prepare ViDoRAG SFT training data v2 with retrieval-based candidates and rich reasoning.

Improvements over v1:
1. Retrieval-based candidates from BGE-M3 (matches evaluation distribution)
2. Rich reasoning outputs using OCR text (instead of templated strings)
3. Consistent 0-based indices in both reason and choice fields
4. Variable candidate counts (10-20, mostly 20 to match evaluation)
5. Cross-deck hard negatives (not just same-deck random negatives)

Usage:
  Phase 1 — Run retrieval and cache results (~30-60 min):
    CUDA_VISIBLE_DEVICES=0 python prepare_sft_data_v2.py --phase retrieve

  Phase 2 — Generate SFT data from cached retrieval (~1 min):
    python prepare_sft_data_v2.py --phase generate

  Or run both phases at once:
    CUDA_VISIBLE_DEVICES=0 python prepare_sft_data_v2.py --phase all
"""

import argparse
import json
import os
import random
import glob
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
DATASET_DIR = os.path.join('./data', DATASET)
IMAGE_DIR = os.path.join(DATASET_DIR, 'img')
OCR_DIR = os.path.join(DATASET_DIR, 'ppocr')
TRAIN_FILE = os.path.join(DATASET_DIR, 'rag_dataset_train.json')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'training_data_v2')
RETRIEVAL_CACHE = os.path.join(OUTPUT_DIR, 'retrieval_cache.json')
EMBED_MODEL = os.environ.get('SFT_RETRIEVAL_EMBED_MODEL', 'BAAI/bge-m3')

MAX_CANDIDATES = 20
CANDIDATE_COUNTS = [10, 15, 20]
CANDIDATE_WEIGHTS = [0.15, 0.25, 0.60]

# ============================================================
# System Prompts (must match training/inference exactly)
# ============================================================
SEEKER_SYSTEM = (
    "You are an AI assistant with strong ability to find references through images. "
    "Given a question, analyze the provided document page images and select the most relevant pages."
)
INSPECTOR_SYSTEM = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question and selected page images, determine if there is sufficient information to answer."
)
SYNTHESIZER_SYSTEM = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question, candidate answer, and reference page images, synthesize the final answer."
)


# ============================================================
# Utility Functions
# ============================================================

def get_doc_stem(file_name: str) -> str:
    if file_name.lower().endswith(".pdf"):
        return file_name[:-4]
    return file_name


def get_page_image_path(doc_stem: str, page_num: int) -> str:
    return os.path.join(IMAGE_DIR, f"{doc_stem}_{page_num}.jpg")


def extract_doc_and_page(image_path: str) -> Tuple[str, int]:
    """Extract doc_stem and page_num from image path like '/path/to/doc_stem_5.jpg'."""
    basename = os.path.basename(image_path)
    name = os.path.splitext(basename)[0]
    match = re.match(r'^(.+?)_(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def read_ocr_text(doc_stem: str, page_num: int) -> str:
    """Read OCR text for a given page, truncated and cleaned."""
    ocr_path = os.path.join(OCR_DIR, f"{doc_stem}_{page_num}.txt")
    if not os.path.isfile(ocr_path):
        return ""
    try:
        with open(ocr_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        text = re.sub(r'\s+', ' ', text)
        return text[:120]
    except Exception:
        return ""


# ============================================================
# Rich Reasoning Generation
# ============================================================

def generate_seeker_reason(candidate_paths: List[str], evidence_indices: List[int],
                           query: str) -> str:
    """Generate rich per-page reasoning using OCR text."""
    lines = [f"Analysis of {len(candidate_paths)} candidate pages:\n"]

    for idx, img_path in enumerate(candidate_paths):
        doc_stem, page_num = extract_doc_and_page(img_path)
        ocr_text = read_ocr_text(doc_stem, page_num)
        is_evidence = idx in evidence_indices

        if ocr_text:
            snippet = ocr_text[:100] + ("..." if len(ocr_text) > 100 else "")
            desc = f'OCR: "{snippet}"'
        else:
            desc = "Visual content (no text extracted)"

        marker = "RELEVANT" if is_evidence else "Not relevant"
        lines.append(f"Index {idx} (Page {page_num}): {desc} -> {marker}")

    lines.append(
        f"\nSelected indices {evidence_indices} as they contain information "
        f"needed to answer the question."
    )
    return "\n".join(lines)


def generate_inspector_reason_sufficient(query: str, answer: str,
                                         evidence_paths: List[str]) -> str:
    """Generate rich reasoning for sufficient information case."""
    lines = [f"After analyzing {len(evidence_paths)} page(s):"]
    for idx, img_path in enumerate(evidence_paths):
        doc_stem, page_num = extract_doc_and_page(img_path)
        ocr_text = read_ocr_text(doc_stem, page_num)
        if ocr_text:
            snippet = ocr_text[:80] + ("..." if len(ocr_text) > 80 else "")
            lines.append(f'  Index {idx} (Page {page_num}): "{snippet}"')
        else:
            lines.append(f'  Index {idx} (Page {page_num}): Visual content')
    lines.append(f"\nThe pages contain sufficient information to answer: {answer[:100]}")
    return "\n".join(lines)


def generate_inspector_reason_insufficient(query: str, provided_paths: List[str],
                                            missing_pages: List[int]) -> str:
    """Generate rich reasoning for insufficient information case."""
    lines = [f"After analyzing {len(provided_paths)} page(s):"]
    for idx, img_path in enumerate(provided_paths):
        doc_stem, page_num = extract_doc_and_page(img_path)
        ocr_text = read_ocr_text(doc_stem, page_num)
        if ocr_text:
            snippet = ocr_text[:80] + ("..." if len(ocr_text) > 80 else "")
            lines.append(f'  Index {idx} (Page {page_num}): "{snippet}"')
        else:
            lines.append(f'  Index {idx} (Page {page_num}): Visual content')
    lines.append(f"\nThe current pages provide partial information but more context "
                 f"is needed from pages {missing_pages}.")
    return "\n".join(lines)


# ============================================================
# Seeker SFT v2 Generation
# ============================================================

def generate_seeker_sft_v2(entry: Dict, retrieved_paths: List[str]) -> Optional[Dict]:
    """Generate Seeker SFT v2 sample with retrieval-based candidates and rich reasoning."""
    query = entry["query"]
    evidence_pages = entry["meta_info"]["reference_page"]
    doc_stem = get_doc_stem(entry["meta_info"]["file_name"])

    # Build candidate list from retrieved pages + ensure evidence pages are included
    all_paths = []
    seen_paths = set()

    for path in retrieved_paths[:MAX_CANDIDATES]:
        if path not in seen_paths and os.path.isfile(path):
            all_paths.append(path)
            seen_paths.add(path)

    # Add evidence pages if not already in candidates
    for ep in evidence_pages:
        ev_path = get_page_image_path(doc_stem, ep)
        if ev_path not in seen_paths and os.path.isfile(ev_path):
            all_paths.append(ev_path)
            seen_paths.add(ev_path)

    if not all_paths:
        return None

    # Sample variable candidate count (mostly 20, matching evaluation)
    target_count = random.choices(CANDIDATE_COUNTS, weights=CANDIDATE_WEIGHTS, k=1)[0]
    target_count = min(target_count, len(all_paths))
    all_paths = all_paths[:target_count]

    # Extract page numbers for index mapping
    candidate_pages = []
    for path in all_paths:
        _, page_num = extract_doc_and_page(path)
        candidate_pages.append(page_num)

    # Find evidence indices (0-based) in candidate list
    page_to_idx = {}
    for idx, p in enumerate(candidate_pages):
        if p not in page_to_idx:
            page_to_idx[p] = idx

    evidence_indices = sorted([page_to_idx[p] for p in evidence_pages if p in page_to_idx])

    if not evidence_indices:
        return None

    # Build instruction (same format as evaluation)
    page_descriptions = ", ".join(
        [f"Page {p} (index {idx})" for idx, p in enumerate(candidate_pages)]
    )
    image_tokens = " ".join(["<image>" for _ in candidate_pages])

    instruction = (
        f"Question: {query}\n\n"
        f"You are given {len(candidate_pages)} document page images. "
        f"The pages are: {page_descriptions}.\n"
        f"{image_tokens}\n\n"
        f"Analyze these pages and select the ones most relevant to answering the question. "
        f"Output your reasoning and selected page indices."
    )

    # Generate rich reasoning using OCR text
    reason = generate_seeker_reason(all_paths, evidence_indices, query)
    summary = (
        f"Found {len(evidence_indices)} relevant page(s) at indices "
        f"{evidence_indices} out of {len(candidate_pages)} candidates."
    )

    output = json.dumps({
        "reason": reason,
        "summary": summary,
        "choice": evidence_indices
    })

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SEEKER_SYSTEM,
        "history": [],
        "images": all_paths
    }


# ============================================================
# Inspector SFT v2 Generation
# ============================================================

def generate_inspector_sft_sufficient_v2(entry: Dict) -> Optional[Dict]:
    """Generate Inspector SFT v2 sample for sufficient information case."""
    query = entry["query"]
    answer = entry["reference_answer"]
    evidence_pages = entry["meta_info"]["reference_page"]
    doc_stem = get_doc_stem(entry["meta_info"]["file_name"])

    image_paths = []
    for ep in evidence_pages:
        path = get_page_image_path(doc_stem, ep)
        if os.path.isfile(path):
            image_paths.append(path)

    if not image_paths:
        return None

    page_nums = [extract_doc_and_page(p)[1] for p in image_paths]
    page_list_str = ", ".join([str(p) for p in page_nums])
    image_tokens = " ".join(["<image>" for _ in image_paths])

    instruction = (
        f"Question: {query}\n\n"
        f"The following {len(image_paths)} page images have been selected as potentially relevant "
        f"(pages: {page_list_str}).\n"
        f"{image_tokens}\n\n"
        f"Determine if these pages contain sufficient information to answer the question. "
        f"If sufficient, provide the answer. If not, indicate what additional pages should be checked."
    )

    reference_indices = list(range(len(image_paths)))
    reason = generate_inspector_reason_sufficient(query, answer, image_paths)

    output = json.dumps({
        "reason": reason,
        "answer": answer,
        "reference": reference_indices
    })

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": INSPECTOR_SYSTEM,
        "history": [],
        "images": image_paths
    }


def generate_inspector_sft_insufficient_v2(entry: Dict, all_pages: List[int]) -> Optional[Dict]:
    """Generate Inspector SFT v2 sample for insufficient information case."""
    query = entry["query"]
    evidence_pages = entry["meta_info"]["reference_page"]
    doc_stem = get_doc_stem(entry["meta_info"]["file_name"])

    if len(evidence_pages) < 2:
        return None

    num_partial = random.randint(1, len(evidence_pages) - 1)
    partial_evidence = sorted(random.sample(evidence_pages, num_partial))
    remaining_evidence = sorted([p for p in evidence_pages if p not in partial_evidence])

    non_evidence = [p for p in all_pages if p not in evidence_pages]
    num_neg = min(3, len(non_evidence))
    negative_pages = random.sample(non_evidence, num_neg) if num_neg > 0 else []
    provided_pages = sorted(partial_evidence + negative_pages)

    image_paths = []
    for p in provided_pages:
        path = get_page_image_path(doc_stem, p)
        if os.path.isfile(path):
            image_paths.append(path)

    if not image_paths:
        return None

    remaining_existing = []
    for p in remaining_evidence:
        path = get_page_image_path(doc_stem, p)
        if os.path.isfile(path):
            remaining_existing.append(p)

    if not remaining_existing:
        return None

    page_nums = [extract_doc_and_page(p)[1] for p in image_paths]
    page_list_str = ", ".join([str(p) for p in page_nums])
    image_tokens = " ".join(["<image>" for _ in image_paths])

    instruction = (
        f"Question: {query}\n\n"
        f"The following {len(image_paths)} page images have been selected as potentially relevant "
        f"(pages: {page_list_str}).\n"
        f"{image_tokens}\n\n"
        f"Determine if these pages contain sufficient information to answer the question. "
        f"If sufficient, provide the answer. If not, indicate what additional pages should be checked."
    )

    reason = generate_inspector_reason_insufficient(query, image_paths, remaining_evidence)

    output = json.dumps({
        "reason": reason,
        "information": f"Need additional pages: {remaining_evidence}",
        "choice": remaining_evidence
    })

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": INSPECTOR_SYSTEM,
        "history": [],
        "images": image_paths
    }


# ============================================================
# Synthesizer SFT v2 Generation
# ============================================================

def generate_synthesizer_sft_v2(entry: Dict) -> Optional[Dict]:
    """Generate Synthesizer SFT v2 sample."""
    query = entry["query"]
    answer = entry["reference_answer"]
    evidence_pages = entry["meta_info"]["reference_page"]
    doc_stem = get_doc_stem(entry["meta_info"]["file_name"])

    image_paths = []
    for ep in evidence_pages:
        path = get_page_image_path(doc_stem, ep)
        if os.path.isfile(path):
            image_paths.append(path)

    if not image_paths:
        return None

    page_nums = [extract_doc_and_page(p)[1] for p in image_paths]
    page_list_str = ", ".join([str(p) for p in page_nums])
    image_tokens = " ".join(["<image>" for _ in image_paths])

    candidate_answer = answer

    instruction = (
        f"Question: {query}\n\n"
        f"Candidate answer: {candidate_answer}\n\n"
        f"Reference page images (pages: {page_list_str}):\n"
        f"{image_tokens}\n\n"
        f"Based on the question, candidate answer, and reference page images, "
        f"synthesize the final accurate answer."
    )

    reference_indices = list(range(len(image_paths)))

    reason_lines = [f"After reviewing {len(image_paths)} reference page(s):"]
    for idx, img_path in enumerate(image_paths):
        ds, pn = extract_doc_and_page(img_path)
        ocr_text = read_ocr_text(ds, pn)
        if ocr_text:
            snippet = ocr_text[:80] + ("..." if len(ocr_text) > 80 else "")
            reason_lines.append(f'  Index {idx} (Page {pn}): "{snippet}"')
        else:
            reason_lines.append(f'  Index {idx} (Page {pn}): Visual content')
    reason_lines.append(f"\nThe information supports the answer: {answer[:100]}")
    reason = "\n".join(reason_lines)

    output = json.dumps({
        "reason": reason,
        "answer": answer,
        "reference": reference_indices
    })

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SYNTHESIZER_SYSTEM,
        "history": [],
        "images": image_paths
    }


# ============================================================
# Retrieval Cache
# ============================================================

def run_retrieval(examples: List[Dict]) -> Dict[str, List[str]]:
    """Run BGE-M3 retrieval using numpy-based similarity search.

    Bypasses LlamaIndex entirely — loads .node files as raw JSON and extracts
    pre-computed embeddings directly into a numpy array. This is ~100x faster
    than using LlamaIndex's VectorStoreIndex or TextNode objects.
    """
    import numpy as np
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Limit threads to prevent CPU explosion
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'

    node_dir = os.path.join(DATASET_DIR, 'bge_ingestion')

    # Step 1: Load BGE-M3 embedding model (for query embedding only)
    print("=" * 60)
    print("Loading BGE-M3 embedding model (for query encoding)...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=10,
        max_length=512,
        trust_remote_code=True,
        device='cuda'
    )

    # Step 2: Load all .node files as raw JSON and extract embeddings directly
    # (bypasses LlamaIndex TextNode which is extremely slow for 52K files)
    print(f"Loading embeddings from {node_dir}...")
    node_files = sorted([f for f in os.listdir(node_dir) if f.endswith('.node')])
    print(f"Found {len(node_files)} .node files")

    all_embeddings = []
    all_filenames = []

    for idx, nf in enumerate(node_files):
        path = os.path.join(node_dir, nf)
        with open(path, 'r') as f:
            data = json.load(f)
        for doc in data:
            emb = doc.get('embedding')
            if emb is not None:
                all_embeddings.append(emb)
                meta = doc.get('metadata', {})
                filename = meta.get('file_name', meta.get('filename', ''))
                all_filenames.append(filename)
        if (idx + 1) % 10000 == 0:
            print(f"  Loaded {idx + 1}/{len(node_files)} files, {len(all_embeddings)} embeddings so far")

    print(f"Loaded {len(all_embeddings)} total embeddings")

    # Step 3: Convert to numpy array and normalize
    print("Converting to numpy array...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / np.maximum(norms, 1e-8)

    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Sample filename: {all_filenames[0] if all_filenames else 'NONE'}")
    del all_embeddings  # Free memory

    # Step 4: Run retrieval for each query
    print(f"\nRunning retrieval for {len(examples)} training questions...")
    cache = {}
    TOP_K = 20

    for i, entry in enumerate(examples):
        query = entry["query"]
        uid = entry.get("uid", str(i))

        try:
            query_embedding = embed_model.get_query_embedding(query)
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_vec = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-8)

            scores = embeddings_array @ query_vec
            top_k = min(TOP_K, len(scores))
            top_indices = np.argsort(-scores)[:top_k]

            image_paths = []
            for idx in top_indices:
                filename = all_filenames[idx]
                if filename:
                    img_name = os.path.basename(filename).replace('.txt', '.jpg')
                    img_path = os.path.join(IMAGE_DIR, img_name)
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)

            cache[uid] = image_paths
        except Exception as e:
            print(f"  [Error] uid={uid}: {e}")
            cache[uid] = []

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(examples)} questions...")

    print(f"\nRetrieval complete. Cached {len(cache)} results.")
    return cache


def load_retrieval_cache() -> Optional[Dict[str, List[str]]]:
    if os.path.isfile(RETRIEVAL_CACHE):
        print(f"Loading retrieval cache from {RETRIEVAL_CACHE}...")
        with open(RETRIEVAL_CACHE, 'r') as f:
            return json.load(f)
    return None


def save_retrieval_cache(cache: Dict[str, List[str]]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RETRIEVAL_CACHE, 'w') as f:
        json.dump(cache, f)
    print(f"Saved retrieval cache to {RETRIEVAL_CACHE}")


# ============================================================
# Main Pipeline
# ============================================================

def load_dataset() -> List[Dict]:
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = data.get("examples", data if isinstance(data, list) else [])
    return examples


def get_all_pages_for_doc(doc_stem: str) -> List[int]:
    pattern = os.path.join(IMAGE_DIR, f"{doc_stem}_*.jpg")
    files = glob.glob(pattern)
    pages = []
    for f in files:
        basename = os.path.basename(f)
        suffix = basename[len(doc_stem) + 1:]
        page_str = suffix.replace(".jpg", "")
        try:
            pages.append(int(page_str))
        except ValueError:
            continue
    return sorted(pages)


def split_and_save(data: List[Dict], name: str, split_ratio: float = 0.9):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data, val_data = data[:split_idx], data[split_idx:]

    for suffix, d in [("_train", train_data), ("_val", val_data), ("", data)]:
        path = os.path.join(OUTPUT_DIR, f"{name}{suffix}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(d)} samples to {path}")


def phase_retrieve():
    """Phase 1: Run retrieval and cache results."""
    examples = load_dataset()
    print(f"Loaded {len(examples)} training examples.")
    cache = run_retrieval(examples)
    save_retrieval_cache(cache)


def phase_generate():
    """Phase 2: Generate SFT data from cached retrieval results."""
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = load_dataset()
    print(f"Loaded {len(examples)} training examples.")

    retrieval_cache = load_retrieval_cache()
    if retrieval_cache is None:
        print("ERROR: No retrieval cache found. Run --phase retrieve first.")
        return

    print(f"Loaded retrieval cache with {len(retrieval_cache)} entries.")

    # Build doc pages cache for Inspector insufficient case
    doc_stems = set()
    for entry in examples:
        doc_stems.add(get_doc_stem(entry["meta_info"]["file_name"]))
    doc_pages_cache = {}
    for stem in sorted(doc_stems):
        pages = get_all_pages_for_doc(stem)
        if pages:
            doc_pages_cache[stem] = pages
    print(f"Found images for {len(doc_pages_cache)} documents.")

    seeker_data = []
    inspector_data = []
    synthesizer_data = []

    skipped = 0
    for i, entry in enumerate(examples):
        uid = entry.get("uid", str(i))
        doc_stem = get_doc_stem(entry["meta_info"]["file_name"])
        evidence_pages = entry["meta_info"]["reference_page"]

        if doc_stem not in doc_pages_cache:
            skipped += 1
            continue

        all_pages = doc_pages_cache[doc_stem]
        if not any(p in all_pages for p in evidence_pages):
            skipped += 1
            continue

        retrieved_paths = retrieval_cache.get(uid, [])

        # --- Seeker SFT v2 ---
        sample = generate_seeker_sft_v2(entry, retrieved_paths)
        if sample:
            seeker_data.append(sample)

        # --- Inspector SFT v2 (sufficient) ---
        sample = generate_inspector_sft_sufficient_v2(entry)
        if sample:
            inspector_data.append(sample)

        # --- Inspector SFT v2 (insufficient) ---
        sample = generate_inspector_sft_insufficient_v2(entry, all_pages)
        if sample:
            inspector_data.append(sample)

        # --- Synthesizer SFT v2 ---
        sample = generate_synthesizer_sft_v2(entry)
        if sample:
            synthesizer_data.append(sample)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(examples)} examples...")

    # Print statistics
    print("=" * 60)
    print("Generation Statistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Skipped (missing images): {skipped}")
    print(f"  Seeker SFT v2: {len(seeker_data)}")
    print(f"  Inspector SFT v2: {len(inspector_data)}")
    print(f"  Synthesizer SFT v2: {len(synthesizer_data)}")

    # Analyze quality
    if seeker_data:
        reasons = [json.loads(s["output"])["reason"] for s in seeker_data]
        unique_reasons = len(set(reasons))
        print(f"\n  Seeker unique reasons: {unique_reasons}/{len(reasons)} "
              f"({unique_reasons/len(reasons)*100:.1f}%)")

        choice_sizes = [len(json.loads(s["output"])["choice"]) for s in seeker_data]
        size_dist = Counter(choice_sizes)
        print(f"  Choice size distribution: {dict(sorted(size_dist.items()))}")

        candidate_counts = [len(s["images"]) for s in seeker_data]
        count_dist = Counter(candidate_counts)
        print(f"  Candidate count distribution: {dict(sorted(count_dist.items()))}")

    # Save all data
    print("=" * 60)
    print("Saving training data v2...")

    split_and_save(seeker_data, "seeker_sft_v2")
    split_and_save(inspector_data, "inspector_sft_v2")
    split_and_save(synthesizer_data, "synthesizer_sft_v2")

    print("=" * 60)
    print(f"Done! All v2 training data saved to: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ViDoRAG SFT training data v2 (retrieval-based + rich reasoning)"
    )
    parser.add_argument(
        "--phase", choices=["retrieve", "generate", "all"], default="all",
        help="Phase to run: 'retrieve' (BGE-M3 retrieval), 'generate' (SFT data), or 'all'"
    )
    args = parser.parse_args()

    if args.phase in ("retrieve", "all"):
        phase_retrieve()

    if args.phase in ("generate", "all"):
        phase_generate()


if __name__ == "__main__":
    main()
