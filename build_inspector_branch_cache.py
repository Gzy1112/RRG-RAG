#!/usr/bin/env python3
"""Counterfactual branch outcomes for Inspector GRPO (SlideVQA val split).

Every question is run through the deployed pipeline (run_closed_loop, the code
path used for test evaluation) once per Inspector action, with everything else
fixed:

    stop        round-1 evidence E0, + rank-1 rescue
    deck        same-deck second hop on E0 (any |E0| >= 1), union, + rank-1 rescue
    expand      re-selection from the ColQwen2.5 top-20, union, + rank-1 rescue
    stop_nor1   / deck_nor1: the same evidence without rank-1 rescue, so the
                fixed rules can be validated on questions none of the agents saw

Seeker, synthesizer and judge calls are memoized per question. All three are
deterministic (greedy decoding), so round 1 runs once per question and identical
evidence is answered and judged once.

The val split shares no deck with train or test, and none of the deployed
agents (Seeker SFT v4 / TFR, Synthesizer SFT v7 / TFR, Inspector SFT v6) was
trained on it.

Usage (env from scripts/env.sh, see scripts/build_inspector_branch_cache.sh):
    python build_inspector_branch_cache.py --out PATH [--shard_id i --num_shards n]
                                           [--limit N] [--uid_file F]
"""
import argparse
import json
import os
import random
import sys
import time

# Knobs read by run_closedloop_eval at import time.
os.environ.setdefault('EVAL_QUERY_FILE', 'rag_dataset_val.json')
os.environ.setdefault('RANK_CACHE_PATH',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'outputs/rank_cache/SlideVQA_val.jsonl'))
os.environ['CONTROLLER_MODE'] = 'always_stop'
os.environ['RETRIEVE_MODE'] = 'merged'
os.environ['CLOSED_LOOP_FIRE_MODE'] = 'union'
os.environ['CLOSED_LOOP_RESELECT_POOL'] = '20'
os.environ['SKIP_PASSTHROUGH_INSPECTOR'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closedloop_eval as rc  # noqa: E402
from llms.evaluator import JudgeUnavailableError  # noqa: E402

HOP2_K = 5

# (branch name, HOP2_MODE, RANK1_RESCUE, controller)
STOP_CTRL = rc.AlwaysStopController()


class ForcedRetrieveController:
    name = 'forced_retrieve'

    def decide(self, question, selected_pages, uid=None):
        return {'decision': rc.RETRIEVE_MORE, 'source': 'forced'}


BRANCHES = [
    ('stop_nor1', 'off', 'off', STOP_CTRL),
    ('stop', 'off', 'append', STOP_CTRL),
    ('deck_nor1', 'deck', 'off', STOP_CTRL),
    ('deck', 'deck', 'append', STOP_CTRL),
    ('expand', 'off', 'append', ForcedRetrieveController()),
]


def log(msg):
    print(f'[branch-cache {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def memoize_agents(mm):
    """Wrap seeker / synthesizer / judge with per-question caches."""
    caches = {'seeker': {}, 'synth': {}, 'judge': {}}
    seeker, synth, judge = mm.agents.seeker, mm.agents.synthesizer, mm.evaluator
    seeker_run, synth_run, judge_eval = seeker.run, synth.run, judge.evaluate

    def seeker_cached(query=None, images_path=None, feedback=None):
        if feedback is not None or images_path is None:
            return seeker_run(query=query, images_path=images_path, feedback=feedback)
        key = (query, tuple(images_path), os.environ.get('SEEKER_V4_MAX_CANDIDATES'))
        if key not in caches['seeker']:
            caches['seeker'][key] = seeker_run(query=query, images_path=list(images_path))
        sel, summary, reason = caches['seeker'][key]
        return list(sel), summary, reason

    def synth_cached(query, candidate_answer, ref_images):
        key = (query, candidate_answer, tuple(ref_images))
        if key not in caches['synth']:
            caches['synth'][key] = synth_run(query, candidate_answer, ref_images)
        return caches['synth'][key]

    def judge_cached(query, reference, answer):
        key = (query, reference, answer)
        if key not in caches['judge']:
            caches['judge'][key] = judge_eval(query, reference, answer)
        return caches['judge'][key]

    seeker.run, synth.run, judge.evaluate = seeker_cached, synth_cached, judge_cached
    return caches


def first_pool(mm, query):
    """Identical to ClosedLoopMMRAG.vidorag: the GMM pool from the resident server."""
    recall_results = mm.search_engine.search(query)
    images = [
        os.path.join(mm.img_dir,
                     os.path.basename(node['node']['metadata']
                                      .get('file_name', node['node']['metadata'].get('filename'))
                                      .replace('.txt', '.jpg')))
        for node in recall_results['source_nodes']
    ]
    if 'gmm' not in mm.experiment_type:
        images = images[:mm.top_k]
    return images


def run_question(mm, ex):
    uid, query, ref = ex['uid'], ex['query'], ex['reference_answer']
    pool = first_pool(mm, query)
    rec = {
        'uid': uid, 'query': query, 'reference_answer': ref,
        'meta_info': ex.get('meta_info', {}),
        'gold': sorted(mm.rank_cache.gold_ids(uid)),
        'pool': [rc._page_id_of(p) for p in pool],
        'e0': None,
        'branches': {},
        'policy_metadata': {
            'base_model': os.environ.get('EVAL_BASE_MODEL') or rc.BASE_MODEL_3B,
            'seeker_adapter': os.environ['EVAL_SEEKER_PATH'],
            'synthesizer_adapter': os.environ['EVAL_SYNTH_PATH'],
            'rank_cache': os.environ['RANK_CACHE_PATH'],
            'image_max_pixels': os.environ.get('VIDORAG_IMAGE_MAX_PIXELS'),
        },
    }
    rc.HOP2_K, rc.HOP2_MIN_SEL, rc.HOP2_FIRE = HOP2_K, 1, 'union'
    for name, hop2_mode, rank1, ctrl in BRANCHES:
        rc.HOP2_MODE, rc.RANK1_RESCUE = hop2_mode, rank1
        answer, trace, cl = rc.run_closed_loop(mm.agents, ctrl, mm.rank_cache,
                                               uid, query, pool, mm.img_dir)
        if answer is None:
            raise RuntimeError(f'branch {name}: no answer')
        verdict = mm.evaluator.evaluate(query, ref, str(answer))
        if rec['e0'] is None:
            rec['e0'] = cl['first_round_selected']
        elif rec['e0'] != cl['first_round_selected']:
            raise RuntimeError(f'branch {name}: round-1 selection differs')
        rec['branches'][name] = {
            'evidence': [rc._page_id_of(p) for p in trace['final_evidence_images']],
            'answer': str(answer),
            'score': int(verdict['score']),
            'passing': int(verdict['passing']),
            'judge_model': verdict.get('judge_model'),
            'hop2': cl.get('hop2'),
            'reselect_pool': cl.get('reselect_pool') or None,
            'reselect_selected': cl.get('reselect_selected') or None,
            'rank1_rescue': cl.get('rank1_rescue'),
        }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--uid_file', default=None)
    ap.add_argument('--branches', default=','.join(b[0] for b in BRANCHES),
                    help='comma-separated cached branches; use stop,deck for the binary policy')
    ap.add_argument('--shuffle_seed', type=int, default=0,
                    help='process questions in a seeded random order so a partial '
                         'run is still a random sample (the split is sorted by deck)')
    args = ap.parse_args()
    requested = args.branches.split(',')
    available = {b[0] for b in BRANCHES}
    if not requested or len(set(requested)) != len(requested) or set(requested) - available:
        raise SystemExit(f'invalid branches {requested}; available: {sorted(available)}')
    BRANCHES[:] = [b for b in BRANCHES if b[0] in requested]

    for path in (os.environ['RANK_CACHE_PATH'],):
        if not os.path.exists(path):
            raise SystemExit(f'missing {path}')
    data_path = os.path.join(rc.PROJECT_ROOT, 'data', rc.DATASET, rc.QUERY_FILE)
    with open(data_path) as f:
        examples = json.load(f)['examples']
    if args.uid_file:
        with open(args.uid_file) as f:
            keep = {line.strip() for line in f if line.strip()}
        examples = [ex for ex in examples if ex['uid'] in keep]
    if args.limit:
        examples = examples[:args.limit]
    random.Random(args.shuffle_seed).shuffle(examples)
    examples = [ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_id]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)['uid'])
    todo = [ex for ex in examples if ex['uid'] not in done]
    log(f'{data_path}: shard {args.shard_id}/{args.num_shards} -> {len(examples)} questions, '
        f'{len(done)} done, {len(todo)} to do')

    kwargs = dict(
        dataset=rc.DATASET,
        query_file=rc.QUERY_FILE,
        experiment_type='vidorag',
        generate_vlm=os.environ.get('EVAL_BASE_MODEL') or rc.BASE_MODEL_3B,
        max_samples=len(todo),
        embed_model_name=rc.EMBED_MODEL,
        embed_model_name_vl=rc.COLQWEN_MODEL,
        embed_model_name_text=os.environ.get('EVAL_EMBED_MODEL', rc.EMBED_MODEL),
        topk=20,
        vl_rerank=False,
        seeker_model_path=os.environ['EVAL_SEEKER_PATH'],
        synthesizer_model_path=os.environ['EVAL_SYNTH_PATH'],
        # Pass-through Inspector is skipped (SKIP_PASSTHROUGH_INSPECTOR=1).
        inspector_model_path=os.environ.get('EVAL_INSPECTOR_PATH') or os.environ['EVAL_SYNTH_PATH'],
        shard_id=None,
        num_shards=None,
    )
    mm = rc.ClosedLoopMMRAG(**kwargs)
    missing = [ex['uid'] for ex in todo if mm.rank_cache.get(ex['uid']) is None]
    if missing:
        raise SystemExit(f'{len(missing)} uids missing from the rank cache, e.g. {missing[:3]}')
    caches = memoize_agents(mm)

    t0 = time.time()
    n_ok = n_fail = 0
    with open(args.out, 'a') as fout:
        for i, ex in enumerate(todo):
            for c in caches.values():
                c.clear()
            try:
                rec = run_question(mm, ex)
            except JudgeUnavailableError:
                raise
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'cuda' in str(e):
                    raise
                n_fail += 1
                log(f'[{i + 1}/{len(todo)}] uid={ex["uid"]} FAILED ({e})')
                continue
            except Exception as e:
                n_fail += 1
                log(f'[{i + 1}/{len(todo)}] uid={ex["uid"]} FAILED ({e})')
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            n_ok += 1
            b = rec['branches']
            log(f'[{i + 1}/{len(todo)}] ok={n_ok} fail={n_fail} |E0|={len(rec["e0"])} '
                + ' '.join(f'{k}={v["score"]}' for k, v in b.items())
                + f'  ({(time.time() - t0) / n_ok:.1f}s/q)')
    log(f'DONE ok={n_ok} fail={n_fail} -> {args.out}')


if __name__ == '__main__':
    main()
