#!/usr/bin/env python3
"""Task 68: build STOP/RETRIEVE_MORE controller SFT data v3 (retrieval-confidence
features injected into the prompt).

v3 = v1 labels + v1 question set (3,500 Stage-1 questions, NO re-fetch) + a new
"[Retrieval Confidence]" text block appended right after the page images.

Design note (deviation from the task suggestion): the Stage-1 cache
(data/retrieval.jsonl + retrieval_shard1.jsonl) stores only doc-id lists
(`fusion_ranking`, top-20), NOT numeric fusion scores - scores were never
cached. Fusion rank is a strictly monotone transform of the fusion score, so
all proposed score features have exact rank-based equivalents:
    fusion scores of selected pages -> fusion ranks of selected pages
    mean pool score (top-10)        -> mean selected rank
    score gap (max - mean)          -> best rank / ranked-count statistics
Ranks are also exactly what the closed-loop eval can reproduce at inference
time from recovery_ceiling_ranks.jsonl (RankCache), keeping train/eval
features byte-consistent.

Feature semantics (identical at train and eval time):
  - rank = position in the pre-GMM fusion ranking, 1-based; rank > 20 or
    absent -> ">20"
  - features are computed over the SAME subset of pages shown as images
    (existing files, first MAX_CONTROLLER_IMAGES)
  - eval-time ranks are capped at 20 to match the training cache (the rank
    cache keeps up to 200 entries, the training cache only top-20)

Label rule / upsampling: v1's labels are reused as-is; positives upsampled
exactly x4 (v1 used x4 because its rule fired only below 400 positives; v3
pins x4 per task spec, final ratio ~1:2 like v1's actual data).
"""
import argparse
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from build_controller_sft_data import (  # noqa: E402  (single source of truth)
    CONTROLLER_SYSTEM, IMG_DIR, SEED, MAX_CONTROLLER_IMAGES, RANK_LIMIT,
    STOP, RETRIEVE_MORE, load_jsonl, log)

SRC_DIR = os.path.join(PROJECT_ROOT, 'outputs/CONTROLLER_SFT/data')
OUT_DIR = os.environ.get(
    'CONTROLLER_V3_DATA_DIR',
    os.path.join(PROJECT_ROOT, 'outputs/CONTROLLER_SFT/data_v3'))
SAMPLED = os.path.join(SRC_DIR, 'sampled.jsonl')
RETRIEVAL_FILES = ['retrieval.jsonl', 'retrieval_shard1.jsonl']
POS_UPSAMPLE_MULT = 4      # v1's actual multiplier (pinned per task 68)
ABSENT_RANK_STR = '>20'

# ---------------------------------------------------------------------------
# v3 prompt: v1 template + a [Retrieval Confidence] block after the images.
# KEEP IN SYNC with SFTControllerV3 in run_closedloop_eval.py (byte-identical).
# ---------------------------------------------------------------------------
CONTROLLER_TEMPLATE_V3 = (
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


def ranks_of(page_ids, ranking, cap=RANK_LIMIT):
    """Fusion ranks (1-based) of page_ids within `ranking`, capped: rank > cap
    or not present -> None. `ranking` may be the 20-entry training cache or the
    full eval rank cache (capped to keep train/eval identical)."""
    pos = {pid: i + 1 for i, pid in enumerate(ranking[:cap])}
    return [pos.get(pid) for pid in page_ids]


def build_confidence_block(page_ids, ranking):
    """The [Retrieval Confidence] text block for the given shown pages."""
    rks = ranks_of(page_ids, ranking)
    parts = [ABSENT_RANK_STR if r is None else str(r) for r in rks]
    ranked = [r for r in rks if r is not None]
    n_in_top10 = sum(1 for r in ranked if r <= 10)
    lines = ['[Retrieval Confidence]']
    lines.append("Fusion ranks of the selected pages (1 = most relevant): "
                 + ', '.join(parts))
    if ranked:
        lines.append(f"Best fusion rank among the selected pages: {min(ranked)}")
        lines.append(f"Mean fusion rank of the ranked selected pages: "
                     f"{sum(ranked) / len(ranked):.2f}")
    else:
        lines.append("Best fusion rank among the selected pages: none ranked")
        lines.append("Mean fusion rank of the ranked selected pages: none ranked")
    lines.append(f"Selected pages within the fusion top-10: {n_in_top10} of "
                 f"{len(page_ids)}")
    return '\n'.join(lines)


def select_image_paths(selected_ids):
    """v1-identical image selection: existing files, first MAX_CONTROLLER_IMAGES.

    Returns (kept_page_ids, kept_paths). Confidence features MUST be computed
    over kept_page_ids so train and eval see the same page subset."""
    kept_ids, kept_paths = [], []
    for pid in selected_ids:
        p = os.path.join(IMG_DIR, pid + '.jpg')
        if os.path.exists(p):
            kept_ids.append(pid)
            kept_paths.append(p)
        if len(kept_paths) >= MAX_CONTROLLER_IMAGES:
            break
    return kept_ids, kept_paths


def build_controller_sample_v3(query, selected_ids, label, ranking):
    kept_ids, kept_paths = select_image_paths(selected_ids)
    if not kept_paths:
        return None
    image_tokens = ' '.join(['<image>'] * len(kept_paths))
    confidence = build_confidence_block(kept_ids, ranking)
    instruction = CONTROLLER_TEMPLATE_V3.format(
        question=query, n=len(kept_paths), image_tokens=image_tokens,
        confidence=confidence)
    return {
        'instruction': instruction,
        'input': '',
        'output': json.dumps({'decision': label}, ensure_ascii=False),
        'system': CONTROLLER_SYSTEM,
        'history': [],
        'images': kept_paths,
    }


def load_retrieval():
    by_uid = {}
    for fname in RETRIEVAL_FILES:
        path = os.path.join(SRC_DIR, fname)
        n0 = len(by_uid)
        for rec in load_jsonl(path).values():
            if rec.get('ok') and rec.get('fusion_ranking'):
                by_uid.setdefault(rec['uid'], rec['fusion_ranking'])
        log(f'loaded {path}: +{len(by_uid) - n0} uids (total {len(by_uid)})')
    return by_uid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    sampled = load_jsonl(SAMPLED)
    log(f'sampled questions: {len(sampled)}')
    rankings = load_retrieval()

    labels = {}
    for fname in sorted(os.listdir(SRC_DIR)):
        if fname.startswith('labels') and fname.endswith('.jsonl'):
            labels.update(load_jsonl(os.path.join(SRC_DIR, fname)))
    ok = [r for r in labels.values() if r.get('ok') and r.get('label')]
    pos = [r for r in ok if r['label'] == RETRIEVE_MORE]
    neg = [r for r in ok if r['label'] == STOP]
    log(f'labels ok={len(ok)}  RETRIEVE_MORE={len(pos)} '
        f'({len(pos) / max(len(ok), 1):.1%})  STOP={len(neg)}')

    # v1's positive upsampling strategy, pinned at x4 (v1's effective mult):
    # final ~1:2 pos:neg, NOT v2b's 1:1.
    upsampled = 0
    if pos:
        upsampled = len(pos) * (POS_UPSAMPLE_MULT - 1)
        pos = pos * POS_UPSAMPLE_MULT
        log(f'positives upsampled x{POS_UPSAMPLE_MULT} (+{upsampled} duplicates, '
            f'now {len(pos)} pos : {len(neg)} neg)')

    samples, skipped_noq, skipped_noretr, skipped_noimg = [], 0, 0, 0
    for rec in pos + neg:
        q = sampled.get(rec['uid'])
        if not q:
            skipped_noq += 1
            continue
        ranking = rankings.get(rec['uid'])
        if not ranking:
            skipped_noretr += 1
            continue
        s = build_controller_sample_v3(q['query'], rec['selected'], rec['label'],
                                       ranking)
        if s is None:
            skipped_noimg += 1
            continue
        samples.append(s)

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    out_json = os.path.join(OUT_DIR, 'controller_sft_v3.json')
    with open(out_json, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    final_pos = sum(1 for s in samples
                    if s['output'] == json.dumps({'decision': RETRIEVE_MORE}))
    meta = {
        'version': 'v3',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'task 68: retrieval-confidence features for the STOP/RETRIEVE_MORE controller',
        'source': {
            'questions': SAMPLED,
            'retrieval_cache': [os.path.join(SRC_DIR, f) for f in RETRIEVAL_FILES],
            'label_files': 'data/labels_shard*of2.jsonl (v1 Stage-1 labels, reused as-is)',
            'n_sampled_questions': len(sampled),
            'n_labelled_ok': len(ok),
            'skipped_missing_question': skipped_noq,
            'skipped_missing_retrieval': skipped_noretr,
            'skipped_missing_images': skipped_noimg,
            'seed': args.seed,
        },
        'feature_note': ('numeric fusion scores were never cached in Stage 1 '
                         '(retrieval.jsonl stores doc-id lists only); v3 uses '
                         'fusion RANKS, a monotone transform of fusion scores, '
                         'capped at top-20 to match the training cache at eval time'),
        'confidence_features': [
            'fusion ranks of the shown selected pages (1-based; >20/absent -> ">20")',
            'best fusion rank among the shown selected pages',
            'mean fusion rank of the ranked shown selected pages',
            'count of shown selected pages within fusion top-10',
        ],
        'label_rule': (f'RETRIEVE_MORE iff gold NOT in frozen-seeker first-round '
                       f'selection AND best gold fusion rank <= {RANK_LIMIT} '
                       f'(v1 labels, unchanged)'),
        'label_distribution_raw': {
            'RETRIEVE_MORE': len([r for r in ok if r['label'] == RETRIEVE_MORE]),
            'STOP': len([r for r in ok if r['label'] == STOP]),
        },
        'upsampling': {'multiplier': POS_UPSAMPLE_MULT,
                       'added_duplicates': upsampled,
                       'policy': 'v1 strategy pinned at x4 (NOT v2b 1:1)'},
        'final_dataset': {
            'path': out_json,
            'n_samples': len(samples),
            'RETRIEVE_MORE': final_pos,
            'STOP': len(samples) - final_pos,
            'positive_rate': final_pos / max(len(samples), 1),
            'max_controller_images': MAX_CONTROLLER_IMAGES,
            'output_format': '{"decision": "STOP"} or {"decision": "RETRIEVE_MORE"}',
        },
    }
    meta_path = os.path.join(OUT_DIR, 'controller_sft_v3_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    log(f'wrote {out_json} ({len(samples)} samples) and {meta_path}')
    log(f'final: {final_pos} RETRIEVE_MORE : {len(samples) - final_pos} STOP '
        f'(1:{(len(samples) - final_pos) / max(final_pos, 1):.2f})')


if __name__ == '__main__':
    main()
