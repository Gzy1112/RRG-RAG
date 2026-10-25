#!/usr/bin/env python3
"""Phase 2 Stage 1: build STOP/RETRIEVE_MORE controller SFT data (v1).

Generates oracle-labelled controller training samples from the SlideVQA TRAIN
split, using the EXACT production first-round pipeline:

    per question:
      1. fusion ranking : port-8000 hybrid server (Qwen3-Embedding-0.6B +
         ColQwen2-v1.0), use_gmm=false, N=20 -> fusion ranks 1..20
         (same engine, same fusion math as measure_recovery_ceiling.py,
         zero extra GPU memory: the resident server does the work)
      2. first-round pool: same server, use_gmm=true, N=20 (production GMM
         pool, byte-identical to the baseline eval first round)
      3. frozen TFR seeker (round_B_seeker, READ-ONLY) selects from the pool
         via the real vidorag_agents.Seeker code path (v4 format, greedy)
      4. label: RETRIEVE_MORE iff gold NOT in seeker selection AND best gold
         fusion rank <= 20 (deployment pool=20); else STOP

Output: LLaMA-Factory alpaca+images dataset (same convention as the inspector
SFT data) with a single JSON decision output (no free-form reason in v1):
    {"decision": "STOP"}  /  {"decision": "RETRIEVE_MORE"}

Subcommands (resume-friendly, streaming jsonl caches):
    sample   : sample train questions                       -> sampled.jsonl
    fetch    : query the retrieval server (2 calls/q)       -> retrieval.jsonl
    label    : frozen-seeker first-round selection + labels -> labels_shard*.jsonl
    finalize : assemble SFT dataset + meta (upsample rule included)

Iron rules honoured: TFR adapters read-only; retrieval server untouched
(only standard /search requests with per-request overrides); all outputs in
brand-new outputs/CONTROLLER_SFT/ directories.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import requests  # noqa: E402

SERVER_URL = os.environ.get('CONTROLLER_LABEL_SERVER', 'http://127.0.0.1:8000')
TRAIN_FILE = os.path.join(PROJECT_ROOT, 'data/SlideVQA/rag_dataset_train.json')
IMG_DIR = os.path.join(PROJECT_ROOT, 'data/SlideVQA/img')
SEEKER_ADAPTER = os.environ.get(
    'CONTROLLER_LABEL_SEEKER',
    os.path.join(PROJECT_ROOT, 'outputs/TFR_7B/cycle2/round_b_seeker'))  # READ-ONLY

OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs/CONTROLLER_SFT/data')
SAMPLED = os.path.join(OUT_DIR, 'sampled.jsonl')
RETRIEVAL = os.path.join(OUT_DIR, 'retrieval.jsonl')
FETCH_DONE = os.path.join(OUT_DIR, 'fetch.done')

RANK_LIMIT = 20            # deployment pool=20 -> label condition gold rank <= 20
SERVER_TOPK = 20
MAX_CONTROLLER_IMAGES = 8  # == CLOSED_LOOP_PROMPT_MAX_IMAGES default at eval
SEED = 42

# ---------------------------------------------------------------------------
# Controller prompt (v1). KEEP IN SYNC with SFTController in run_closedloop_eval.py
# ---------------------------------------------------------------------------
CONTROLLER_SYSTEM = (
    "You are a strict evidence auditor for retrieval-augmented question answering "
    "over slide decks. You decide whether the retrieved pages are sufficient to "
    "answer a question, or whether more pages must be retrieved."
)

CONTROLLER_TEMPLATE = (
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

STOP = 'STOP'
RETRIEVE_MORE = 'RETRIEVE_MORE'


def log(msg):
    print(f'[controller-data {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def page_id_of(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def gold_ids_of(example):
    doc = os.path.splitext(example['meta_info']['file_name'])[0]
    return {f"{doc}_{p}" for p in example['meta_info']['reference_page']}


def load_jsonl(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out.setdefault(rec['uid'], rec)
    return out


def gpu_free_mib(gpu_index):
    try:
        txt = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
             '-i', str(gpu_index)], text=True)
        return int(txt.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def cmd_sample(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    examples = json.load(open(TRAIN_FILE))['examples']
    examples = [e for e in examples if e['meta_info'].get('reference_page')]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    picked = examples[:args.n]
    with open(SAMPLED, 'w') as f:
        for e in picked:
            f.write(json.dumps({
                'uid': e['uid'], 'query': e['query'],
                'gold': sorted(gold_ids_of(e)),
            }, ensure_ascii=False) + '\n')
    log(f'sampled {len(picked)} questions from {len(examples)} train examples -> {SAMPLED}')


# ---------------------------------------------------------------------------
# fetch (retrieval server: fusion ranking + production GMM pool)
# ---------------------------------------------------------------------------

def server_search(question, n, use_gmm, timeout=300, retries=3, server_url=None):
    server_url = server_url or SERVER_URL
    payload = {'questions': [question], 'N': n, 'use_gmm': use_gmm}
    for attempt in range(retries):
        try:
            r = requests.post(f'{SERVER_URL}/search', json=payload, timeout=timeout)
            r.raise_for_status()
            res = r.json()['results'][0]
            if res.get('error'):
                raise RuntimeError(res['error'])
            return res['documents']
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f'server call failed ({e}); retry {attempt + 1}')
            time.sleep(3 * (attempt + 1))


def server_search_batch(questions, n, use_gmm, timeout=900, retries=3,
                        server_url=None):
    """Batch /search: ~2.9x faster than per-question calls (measured: 7.0 vs
    20.0 s/q at batch=8). Returns a list of document lists aligned by position
    with the input questions; entries whose result carries an error are None.
    ALWAYS sends an explicit use_gmm: the server mutates a global gmm flag per
    request when overridden, which is only safe because each server instance
    is driven by exactly one strictly-sequential client (one fetch shard)."""
    server_url = server_url or SERVER_URL
    payload = {'questions': questions, 'N': n, 'use_gmm': use_gmm}
    for attempt in range(retries):
        try:
            r = requests.post(f'{server_url}/search', json=payload, timeout=timeout)
            r.raise_for_status()
            results = r.json()['results']
            if len(results) != len(questions):
                raise RuntimeError(
                    f'result count {len(results)} != question count {len(questions)}')
            return [None if res.get('error') else res['documents'] for res in results]
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f'batch server call failed ({e}); retry {attempt + 1}')
            time.sleep(3 * (attempt + 1))


def cmd_fetch(args):
    server_url = args.server or SERVER_URL
    sampled_path = args.sampled_file or SAMPLED
    out_path = args.out or RETRIEVAL
    sampled = load_jsonl(sampled_path)
    if not sampled:
        raise SystemExit('no sampled questions - run `sample` first')
    records = sorted(sampled.values(), key=lambda r: r['uid'])
    records = [r for i, r in enumerate(records) if i % args.nshards == args.shard]
    done = load_jsonl(out_path)
    todo = [rec for rec in records if rec['uid'] not in done]
    log(f'fetch shard {args.shard}/{args.nshards}: {len(todo)} to do '
        f'({len(done)} cached) against {server_url} (batch={args.batch})')

    health = requests.get(f'{server_url}/health', timeout=10).json()
    if not health.get('engine_loaded'):
        raise SystemExit('retrieval server has no engine loaded')

    fout = open(out_path, 'a')
    t0 = time.time()
    i = 0
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        queries = [rec['query'] for rec in batch]
        try:
            fusion_all = server_search_batch(
                queries, SERVER_TOPK, use_gmm=False, server_url=server_url)
            pool_all = server_search_batch(
                queries, SERVER_TOPK, use_gmm=True, server_url=server_url)
        except Exception as e:
            # Whole batch failed after retries: leave records unfetched so a
            # later resume retries them, instead of caching them as not-ok.
            log(f'fetch batch FAILED entirely ({e}); batch left for resume')
            continue
        for rec, fusion_docs, pool_docs in zip(batch, fusion_all, pool_all):
            i += 1
            if fusion_docs is None or pool_docs is None:
                # Item-level server error: retry this one question directly
                # before giving up, so transient errors do not drop questions.
                try:
                    fusion_docs = server_search(
                        rec['query'], SERVER_TOPK, use_gmm=False, server_url=server_url)
                    pool_docs = server_search(
                        rec['query'], SERVER_TOPK, use_gmm=True, server_url=server_url)
                except Exception as e2:
                    log(f'fetch FAILED for {rec["uid"]} even after retry: {e2}')
            if fusion_docs is None or pool_docs is None:
                out = {'uid': rec['uid'], 'ok': False, 'error': 'batch_item_failed',
                       'fusion_ranking': [], 'gmm_pool': []}
            else:
                out = {
                    'uid': rec['uid'],
                    'fusion_ranking': [d['doc_id'] for d in fusion_docs],
                    'gmm_pool': [d['doc_id'] for d in pool_docs],
                    'ok': True,
                }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
        fout.flush()
        if i % 25 < args.batch or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            log(f'fetch shard{args.shard} {i}/{len(todo)}  {el / 60:.1f}min elapsed  '
                f'{el / i:.2f}s/q  ETA {eta / 60:.1f}min')
    fout.close()
    open(FETCH_DONE, 'a').write(f'shard{args.shard}:{len(sampled)}\n')
    log(f'fetch shard {args.shard} complete')


# ---------------------------------------------------------------------------
# label (frozen TFR seeker first-round selection, exact eval code path)
# ---------------------------------------------------------------------------

def cmd_label(args):
    retrieval = {}
    for path in (args.input or RETRIEVAL).split(','):
        retrieval.update(load_jsonl(path.strip()))
    records = [r for r in retrieval.values() if r.get('ok') and r['gmm_pool']]
    records.sort(key=lambda r: r['uid'])
    shard = [r for i, r in enumerate(records) if i % args.nshards == args.shard]
    out_path = os.path.join(OUT_DIR, f'labels_shard{args.shard}of{args.nshards}.jsonl')
    done = load_jsonl(out_path)
    todo = [r for r in shard if r['uid'] not in done]
    log(f'label shard {args.shard}/{args.nshards}: {len(todo)} to do ({len(done)} cached)')
    if not todo:
        return

    free = gpu_free_mib(int(os.environ.get('CONTROLLER_LABEL_GPU', '1')))
    log(f'GPU free memory: {free} MiB')
    if free is not None and free < 12000:
        raise SystemExit('not enough free GPU memory for the frozen seeker')

    # Frozen seeker: identical loading/inference path as eval (TrainedVLM,
    # greedy). Adapter stays read-only (no merge, no save).
    from llms.llm import LLM  # noqa: E402
    from vidorag_agents import Seeker  # noqa: E402
    os.environ.pop('SEEKER_V4_MAX_CANDIDATES', None)  # eval default = module default 10
    t0 = time.time()
    llm = LLM(SEEKER_ADAPTER)
    seeker = Seeker(llm)
    log(f'seeker loaded in {time.time() - t0:.0f}s '
        f'(v4_format={seeker._use_v4_format}, adapter read-only)')

    sampled = load_jsonl(SAMPLED)
    fout = open(out_path, 'a')
    t0 = time.time()
    for i, rec in enumerate(todo, 1):
        uid = rec['uid']
        gold = set(sampled[uid]['gold']) if uid in sampled else set()
        query = sampled[uid]['query'] if uid in sampled else ''
        pool = [os.path.join(IMG_DIR, pid + '.jpg') for pid in rec['gmm_pool']]
        pool = [p for p in pool if os.path.exists(p)]
        entry = {'uid': uid, 'label': None, 'selected': [], 'ok': False}
        try:
            if pool:
                selected, _summary, _reason = seeker.run(query=query, images_path=list(pool))
                selected_ids = [page_id_of(p) for p in selected]
            else:
                selected_ids = []
            ranking = rec['fusion_ranking']
            pos = {pid: r + 1 for r, pid in enumerate(ranking)}
            best_rank = min((pos[g] for g in gold if g in pos), default=None)
            gold_selected = bool(gold & set(selected_ids))
            trigger = (not gold_selected) and (best_rank is not None) and (best_rank <= RANK_LIMIT)
            entry.update({
                'ok': True,
                'selected': selected_ids,
                'best_gold_rank': best_rank,
                'gold_selected': gold_selected,
                'label': RETRIEVE_MORE if trigger else STOP,
                'reason_stop': ('gold_selected' if gold_selected
                                else f'best_gold_rank={best_rank} beyond {RANK_LIMIT} or absent'),
            })
        except Exception as e:
            log(f'label FAILED for {uid}: {e}')
            entry['error'] = str(e)
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
        fout.flush()
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            log(f'label shard{args.shard} {i}/{len(todo)}  {el / 60:.1f}min  '
                f'{el / i:.2f}s/q  ETA {eta / 60:.1f}min')
    fout.close()
    log(f'label shard {args.shard} complete')


# ---------------------------------------------------------------------------
# finalize (SFT dataset + meta + optional positive upsampling)
# ---------------------------------------------------------------------------

def build_controller_sample(query, selected_paths, label):
    images = selected_paths[:MAX_CONTROLLER_IMAGES]
    image_tokens = ' '.join(['<image>'] * len(images))
    instruction = CONTROLLER_TEMPLATE.format(
        question=query, n=len(images), image_tokens=image_tokens)
    return {
        'instruction': instruction,
        'input': '',
        'output': json.dumps({'decision': label}, ensure_ascii=False),
        'system': CONTROLLER_SYSTEM,
        'history': [],
        'images': images,
    }


def cmd_finalize(args):
    sampled = load_jsonl(SAMPLED)
    labels = {}
    for fname in os.listdir(OUT_DIR):
        if fname.startswith('labels_shard') and fname.endswith('.jsonl'):
            labels.update(load_jsonl(os.path.join(OUT_DIR, fname)))
    ok = [r for r in labels.values() if r.get('ok') and r['label']]
    pos = [r for r in ok if r['label'] == RETRIEVE_MORE]
    neg = [r for r in ok if r['label'] == STOP]
    log(f'labels ok={len(ok)}  RETRIEVE_MORE={len(pos)} ({len(pos) / max(len(ok), 1):.1%})'
        f'  STOP={len(neg)}')

    # Upsample rule from the task: if positives < 400, repeat them up to ~1:2
    upsampled = 0
    if len(pos) < 400 and pos:
        target = len(neg) // 2
        mult = max(1, target // len(pos))
        upsampled = len(pos) * (mult - 1)
        pos = pos * mult
        log(f'positives < 400 -> upsampled x{mult} (+{upsampled} duplicates, '
            f'now {len(pos)} pos : {len(neg)} neg)')

    samples, skipped_noimg = [], 0
    for rec in pos + neg:
        q = sampled.get(rec['uid'])
        if not q:
            continue
        paths = [os.path.join(IMG_DIR, pid + '.jpg') for pid in rec['selected']]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            skipped_noimg += 1
            continue
        samples.append(build_controller_sample(q['query'], paths, rec['label']))

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    out_json = os.path.join(OUT_DIR, 'controller_sft_v1.json')
    with open(out_json, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    final_pos = sum(1 for s in samples if s['output'] == json.dumps({'decision': RETRIEVE_MORE}))
    meta = {
        'version': 'v1',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': {
            'train_file': TRAIN_FILE,
            'n_sampled_questions': len(sampled),
            'n_labelled_ok': len(ok),
            'skipped_missing_images': skipped_noimg,
            'seed': args.seed,
        },
        'label_rule': (f'RETRIEVE_MORE iff gold NOT in frozen-seeker first-round '
                       f'selection AND best gold fusion rank <= {RANK_LIMIT}'),
        'pipeline': {
            'retrieval_server': SERVER_URL,
            'fusion': 'hybrid Qwen3-Embedding-0.6B + ColQwen2-v1.0, per-engine '
                      'min-max norm, 0.5/0.5 sum, GMM off, N=20',
            'first_round_pool': 'server GMM on, topk=20 (production config)',
            'seeker': f'{SEEKER_ADAPTER} (frozen, read-only, greedy, v4 format, '
                      'max_candidates=10)',
        },
        'label_distribution_raw': {
            'RETRIEVE_MORE': len([r for r in ok if r['label'] == RETRIEVE_MORE]),
            'STOP': len([r for r in ok if r['label'] == STOP]),
            'positive_rate': len([r for r in ok if r['label'] == RETRIEVE_MORE]) / max(len(ok), 1),
        },
        'upsampling': {'applied': upsampled > 0, 'added_duplicates': upsampled},
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
    meta_path = os.path.join(OUT_DIR, 'controller_sft_v1_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    log(f'wrote {out_json} ({len(samples)} samples) and {meta_path}')
    log(json.dumps(meta['label_distribution_raw'], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_sample = sub.add_parser('sample')
    p_sample.add_argument('--n', type=int, default=3000)
    p_sample.add_argument('--seed', type=int, default=SEED)
    p_fetch = sub.add_parser('fetch')
    p_fetch.add_argument('--batch', type=int, default=8)
    p_fetch.add_argument('--shard', type=int, default=0)
    p_fetch.add_argument('--nshards', type=int, default=1)
    p_fetch.add_argument('--server', type=str, default=None)
    p_fetch.add_argument('--sampled-file', type=str, default=None)
    p_fetch.add_argument('--out', type=str, default=None)
    p_label = sub.add_parser('label')
    p_label.add_argument('--shard', type=int, default=0)
    p_label.add_argument('--nshards', type=int, default=1)
    p_label.add_argument('--input', type=str, default=None,
                         help='comma-separated retrieval jsonl files (default: retrieval.jsonl)')
    p_fin = sub.add_parser('finalize')
    p_fin.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    {'sample': cmd_sample, 'fetch': cmd_fetch,
     'label': cmd_label, 'finalize': cmd_finalize}[args.cmd](args)


if __name__ == '__main__':
    main()
