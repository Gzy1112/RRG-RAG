#!/usr/bin/env python3
"""Rebuild the closed-loop rank cache using ColQwen2.5 standalone retrieval.

Background
----------
The original `recovery_ceiling_ranks.jsonl` was built by
`measure_recovery_ceiling.py` using the OLD hybrid retrieval engine
(Qwen3-Embedding-0.6B + ColQwen2-v1.0).  The current closed-loop eval uses
ColQwen2.5 standalone, so the cached ranking describes a *different*
retrieval system: measured gold coverage in the cached top-20 was only
62.7% vs 88.4% for the live ColQwen2.5 pool.  RETRIEVE_MORE therefore
handed the Seeker a pool that was missing ~342 gold pages, which is why
round-2 re-selection scored worse than round-1.

This script re-records the per-question ColQwen2.5 MaxSim ordering
(pre-GMM, top-100) so the round-2 pool is consistent with round-1.

Output format matches the original cache:
    {"uid", "gold", "gold_ranks": {page: rank}, "best_gold_rank", "ranking": [...]}

Usage
-----
    python rebuild_rank_cache_colqwen25.py                 # full 2215 (HTTP)
    python rebuild_rank_cache_colqwen25.py --limit 50      # smoke test
    python rebuild_rank_cache_colqwen25.py --resume        # continue partial run

Two backends are supported and can run in parallel on different GPUs:
    http   : POST to the running Flask retrieval server (GPU 1)
    inproc : instantiate ColQwen25Engine locally (GPU 0)

Shard the 2215 questions across both to roughly halve the wall time:
    python rebuild_rank_cache_colqwen25.py --shard-idx 0 --num-shards 2 \
        --backend inproc --device cuda:0 --out rank_cache_shard0.jsonl
    python rebuild_rank_cache_colqwen25.py --shard-idx 1 --num-shards 2 \
        --backend http   --out rank_cache_shard1.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.environ.get(
    'RANK_CACHE_DATASET', os.path.join(PROJECT_ROOT, 'data/SlideVQA/rag_dataset_test.json'))
# Corpus searched by --backend inproc (data/<name>/colqwen25_ingestion_v3.pt).
ENGINE_DATASET = os.environ.get('RANK_CACHE_ENGINE_DATASET', 'SlideVQA')
OUT_PATH = os.path.join(PROJECT_ROOT, 'outputs/rank_cache/SlideVQA_test.jsonl')
SERVER_URL = os.environ.get('RETRIEVAL_SERVER_URL', 'http://localhost:8000')
RANKING_DEPTH = int(os.environ.get('RANK_CACHE_DEPTH', '100'))
BATCH_SIZE = int(os.environ.get('RANK_CACHE_BATCH', '4'))


def load_questions():
    """Return ordered list of (uid, query) from the SlideVQA test split."""
    with open(DATASET_PATH) as f:
        examples = json.load(f)['examples']
    return [(ex['uid'], ex['query']) for ex in examples]


def load_questions_filtered(limit=None, offset=0, shard_idx=0, num_shards=1):
    """Questions after applying offset/limit then modulo sharding (stable order)."""
    questions = load_questions()
    if offset:
        questions = questions[offset:]
    if limit:
        questions = questions[:limit]
    if num_shards > 1:
        questions = [q for i, q in enumerate(questions) if i % num_shards == shard_idx]
    return questions


def load_gold():
    """uid -> sorted list of gold page ids, derived from dataset ground truth."""
    with open(DATASET_PATH) as f:
        examples = json.load(f)['examples']
    gold = {}
    for ex in examples:
        meta = ex.get('meta_info', {}) or {}
        stem = os.path.splitext(os.path.basename(meta.get('file_name', '')))[0]
        pages = meta.get('reference_page') or meta.get('original_evidence_pages') or []
        gold[ex['uid']] = sorted({f'{stem}_{p}' for p in pages})
    return gold


def server_health():
    try:
        with urllib.request.urlopen(f'{SERVER_URL}/health', timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {'status': 'error', 'error': str(exc)}


def query_server(questions):
    """POST a batch of questions; return list of ordered page-id rankings."""
    payload = json.dumps({
        'questions': questions,
        'N': RANKING_DEPTH,
        'use_gmm': False,          # keep the complete pre-GMM MaxSim ordering
    }).encode()
    req = urllib.request.Request(
        f'{SERVER_URL}/search', data=payload,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode())

    rankings = []
    for result in body.get('results', []):
        if result.get('error'):
            raise RuntimeError(f"server error: {result['error']}")
        rankings.append([doc['doc_id'] for doc in result.get('documents', [])])
    return rankings


class InprocBackend:
    """ColQwen25Engine held in this process (uses a local GPU, no HTTP).

    Mirrors the Flask server's _parse_node: source_nodes entries are
    NodeWithScore dicts shaped {'node': {...}, 'score': ...}, and the page id
    is the stem of node['image_path'].  Order is preserved (descending score).
    """

    def __init__(self, dataset=ENGINE_DATASET, device='cuda:0'):
        from search_engine import ColQwen25Engine
        self.engine = ColQwen25Engine(dataset=dataset, device=device)
        self.engine.gmm = False
        self.engine.return_raw = False

    def rankings(self, questions):
        out = []
        for q in questions:
            raw = self.engine.search(q)
            ids = []
            for sn in raw.get('source_nodes', []):
                inner = sn.get('node', {}) if isinstance(sn, dict) else {}
                image_path = inner.get('image_path') or inner.get('image', '')
                ids.append(os.path.splitext(os.path.basename(image_path))[0]
                           if image_path else '')
            out.append(ids[:RANKING_DEPTH])
        return out


def load_done(path):
    """uids already written (for --resume)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)['uid'])
    return done


def main():
    ap = argparse.ArgumentParser(description='Rebuild rank cache with ColQwen2.5')
    ap.add_argument('--limit', type=int, default=None, help='only process first N questions')
    ap.add_argument('--offset', type=int, default=0, help='skip the first N questions')
    ap.add_argument('--num-shards', type=int, default=1, help='split questions into N shards')
    ap.add_argument('--shard-idx', type=int, default=0, help='which shard this process handles')
    ap.add_argument('--backend', choices=['http', 'inproc'], default='http',
                    help='http = Flask server; inproc = local ColQwen25Engine')
    ap.add_argument('--device', default='cuda:0', help='device for --backend inproc')
    ap.add_argument('--resume', action='store_true', help='skip uids already in output')
    ap.add_argument('--out', default=OUT_PATH, help='output jsonl path')
    args = ap.parse_args()

    if args.backend == 'http':
        health = server_health()
        print(f'[rebuild] server health: {health}')
        if not health.get('engine_loaded'):
            raise SystemExit('retrieval server not ready (engine_loaded != true)')
        backend = None
    else:
        print(f'[rebuild] loading in-process ColQwen25Engine ({ENGINE_DATASET}) on {args.device} ...')
        backend = InprocBackend(device=args.device)
        print('[rebuild] in-process engine ready')

    gold_map = load_gold()
    questions = load_questions_filtered(
        limit=args.limit, offset=args.offset,
        shard_idx=args.shard_idx, num_shards=args.num_shards)

    done = load_done(args.out) if args.resume else set()
    if done:
        print(f'[rebuild] resuming: {len(done)} uids already done')
    pending = [(u, q) for u, q in questions if u not in done]
    print(f'[rebuild] {len(pending)} questions to process '
          f'(depth={RANKING_DEPTH}, batch={BATCH_SIZE})')

    mode = 'a' if args.resume else 'w'
    t0 = time.time()
    processed = 0
    with open(args.out, mode) as out:
        for start in range(0, len(pending), BATCH_SIZE):
            chunk = pending[start:start + BATCH_SIZE]
            uids = [u for u, _ in chunk]
            queries = [q for _, q in chunk]
            if backend is None:
                rankings = query_server(queries)
            else:
                rankings = backend.rankings(queries)

            for uid, ranking in zip(uids, rankings):
                gold = gold_map.get(uid, [])
                pos = {pid: i + 1 for i, pid in enumerate(ranking)}
                gold_ranks = {g: pos[g] for g in gold if g in pos}
                best = min(gold_ranks.values()) if gold_ranks else None
                out.write(json.dumps({
                    'uid': uid,
                    'gold': gold,
                    'gold_ranks': gold_ranks,
                    'best_gold_rank': best,
                    'ranking': ranking,
                }, ensure_ascii=False) + '\n')
            out.flush()

            processed += len(chunk)
            if processed % 20 == 0 or processed == len(pending):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-9)
                eta = (len(pending) - processed) / max(rate, 1e-9)
                print(f'[rebuild] {processed}/{len(pending)}  '
                      f'{rate:.2f} q/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m',
                      flush=True)

    print(f'[rebuild] done -> {args.out}')


if __name__ == '__main__':
    main()
