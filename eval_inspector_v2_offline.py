#!/usr/bin/env python3
"""Offline policy evaluation of Inspector v2 checkpoints on the val holdout decks.

Every recovery branch of every val question was run through the deployed
pipeline and judged once (build_inspector_branch_cache.py), so the value of a
deterministic Inspector policy on those questions is EXACT: the mean judge
score / pass rate of the branch it picks. Used only for checkpoint selection;
the test set is never touched here.

Reported per policy (adapter checkpoints + reference rules):
    score, pass, action distribution, paired delta vs always-STOP and vs the
    deck rule (DECK iff |E0| >= 2), with sigma = delta / SE(paired diff).

Usage:
    python eval_inspector_v2_offline.py \
        --branch_cache a.jsonl,b.jsonl --split holdout \
        --adapters run/checkpoint-20/controller,run/checkpoint-40/controller \
        [--base] [--actions STOP,DECK,EXPAND] [--out decisions.jsonl]
"""
import argparse
import json
import math
import os
import sys

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import inspector_v2_prompt as iv2  # noqa: E402

BRANCH_OF = {'STOP': 'stop', 'DECK': 'deck', 'EXPAND': 'expand'}


def load_records(paths, split, rank_path, actions):
    rankings = {}
    for line in open(rank_path):
        if line.strip():
            r = json.loads(line)
            if r.get('ranking'):
                rankings.setdefault(r['uid'], r['ranking'])
    out, seen = [], set()
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['uid'] in seen or not r.get('e0') or r['uid'] not in rankings:
                continue
            seen.add(r['uid'])
            if split != 'all' and iv2.val_split_of(r['meta_info']['file_name']) != split:
                continue
            br = {a: r['branches'].get(BRANCH_OF[a]) for a in actions}
            if any(b is None or b.get('score') is None for b in br.values()):
                continue
            out.append({'uid': r['uid'], 'query': r['query'], 'e0': r['e0'],
                        'ranking': rankings[r['uid']],
                        'score': {a: float(br[a]['score']) for a in actions},
                        'pass': {a: bool(br[a]['passing']) for a in actions}})
    return out


def summarize(name, recs, decisions, ref=None):
    n = len(recs)
    sc = [r['score'][d] for r, d in zip(recs, decisions)]
    ps = [r['pass'][d] for r, d in zip(recs, decisions)]
    dist = {}
    for d in decisions:
        dist[d] = dist.get(d, 0) + 1
    line = (f'{name:<34} score={sum(sc) / n:.4f} pass={sum(ps) / n:.4f} '
            f'acts={ {a: round(c / n, 3) for a, c in sorted(dist.items())} }')
    for ref_name, ref_dec in (ref or {}).items():
        diff = [r['score'][d] - r['score'][e] for r, d, e in zip(recs, decisions, ref_dec)]
        m = sum(diff) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in diff) / max(n - 1, 1))
        sig = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
        line += f' | vs {ref_name} {m:+.4f} ({sig:+.2f}σ)'
    print(line, flush=True)
    return sum(sc) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch_cache', required=True)
    ap.add_argument('--split', default='holdout', choices=['holdout', 'train', 'all'])
    ap.add_argument('--rank_cache', default=os.path.join(
        PROJECT_ROOT, 'outputs/rank_cache/SlideVQA_val.jsonl'))
    ap.add_argument('--actions', default='STOP,DECK,EXPAND')
    ap.add_argument('--adapters', default='')
    ap.add_argument('--base', action='store_true', help='also score the base model (no adapter)')
    ap.add_argument('--out', default='')
    ap.add_argument('--base_model', default=os.environ.get('VIDORAG_BASE_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct'))
    args = ap.parse_args()

    actions = iv2.parse_actions(args.actions)
    recs = load_records(args.branch_cache.split(','), args.split, args.rank_cache, actions)
    print(f'[offline] {len(recs)} {args.split} records, actions={actions}', flush=True)

    stop = ['STOP'] * len(recs)
    rule = [('DECK' if 'DECK' in actions and len(r['e0']) >= 2 else 'STOP') for r in recs]
    oracle = [max(actions, key=lambda a: (r['score'][a], a == 'STOP')) for r in recs]
    refs = {'stop': stop, 'rule': rule}
    summarize('always STOP', recs, stop, {})
    summarize('rule (DECK iff |E0|>=2)', recs, rule, {'stop': stop})
    for a in actions:
        if a != 'STOP':
            summarize(f'always {a}', recs, [a] * len(recs), refs)
    summarize('oracle', recs, oracle, refs)

    adapters = [p for p in args.adapters.split(',') if p]
    if not adapters and not args.base:
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base_path = args.base_model
    for path in adapters:
        with open(os.path.join(path, 'adapter_config.json')) as f:
            declared_base = json.load(f)['base_model_name_or_path']
        if os.path.realpath(declared_base) != os.path.realpath(base_path):
            raise ValueError(f'Adapter base mismatch: {path} uses {declared_base}, not {base_path}')
    proc = AutoProcessor.from_pretrained(base_path)
    model = AutoModelForImageTextToText.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map='cuda')
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[0], adapter_name='ckpt0')
        for i, p in enumerate(adapters[1:], 1):
            model.load_adapter(p, adapter_name=f'ckpt{i}')
    model.eval()
    img_dir = os.path.join(PROJECT_ROOT, 'data/SlideVQA/img')

    policies = ([('base', None)] if args.base else []) + \
        [(p, f'ckpt{i}') for i, p in enumerate(adapters)]
    fout = open(args.out, 'w') if args.out else None
    for name, adapter in policies:
        scorer = iv2.ActionScorer(model, proc, actions, adapter_name=adapter, img_dir=img_dir)
        decisions, expected = [], 0.0
        for r in recs:
            if adapter is None and adapters:
                with model.disable_adapter():
                    d, s = scorer.decide(r['query'], r['e0'], r['ranking'])
            else:
                d, s = scorer.decide(r['query'], r['e0'], r['ranking'])
            m = max(s.values())
            z = sum(math.exp(v - m) for v in s.values())
            expected += sum(math.exp(s[a] - m) / z * r['score'][a] for a in actions) / len(recs)
            decisions.append(d)
            if fout:
                fout.write(json.dumps({'policy': name, 'uid': r['uid'], 'decision': d,
                                       'logp': s, 'score': r['score']}) + '\n')
        short = '/'.join(name.rstrip('/').split('/')[-2:]) if adapter else name
        summarize(short, recs, decisions, refs)
        print(f'{"":<34} expected score under sampling={expected:.4f}', flush=True)
    if fout:
        fout.close()


if __name__ == '__main__':
    main()
