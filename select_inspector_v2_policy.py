#!/usr/bin/env python3
"""Select the Inspector v2 checkpoint and action biases on val-holdout.

The offline decision files contain each checkpoint's mean-token log-probability
for every valid action and the cached branch score for each action. This script
uses a small, fixed bias grid (STOP bias=0) to calibrate the decision boundary,
then picks by mean judge score, pass rate, lower recovery rate and smaller bias
(in that order). No test records are read.
"""
import argparse
import collections
import itertools
import json
import os
import re

P = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.abspath(__file__))


def metric(rows, biases):
    scores, passing, actions = [], [], []
    for r in rows:
        acts = tuple(r['score'])
        a = max(acts, key=lambda x: (r['logp'][x] + biases.get(x, 0.0), x == 'STOP'))
        v = float(r['score'][a])
        actions.append(a)
        scores.append(v)
        passing.append(v >= 4)
    n = len(rows)
    return {
        'score': sum(scores) / n,
        'pass': sum(passing) / n,
        'recovery_rate': sum(a != 'STOP' for a in actions) / n,
        'actions': dict(collections.Counter(actions)),
    }


def grids(actions):
    non_stop = [a for a in actions if a != 'STOP']
    # Mean-token log-probability differences are small; 0.1 is intentionally
    # coarse for the three-action policy to limit calibration overfit.
    step = 0.05 if len(non_stop) == 1 else 0.1
    vals = [round(-1.0 + step * i, 6) for i in range(round(2.0 / step) + 1)]
    for combo in itertools.product(vals, repeat=len(non_stop)):
        yield {'STOP': 0.0, **dict(zip(non_stop, combo))}


def key(result):
    # round the primary metric before tie-breaking; score differences below
    # 1e-9 are numerical noise, not evidence for a more complex calibration.
    return (round(result['score'], 9), round(result['pass'], 9),
            -result['recovery_rate'], -sum(abs(v) for v in result['biases'].values()),
            -result['step'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('decision_files', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_records', type=int, default=300)
    args = ap.parse_args()

    grouped = collections.defaultdict(dict)
    for path in args.decision_files:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['policy'] == 'base':
                continue
            grouped[r['policy']].setdefault(r['uid'], r)

    results = []
    for policy, by_uid in grouped.items():
        rows = list(by_uid.values())
        if len(rows) < args.min_records:
            print(f'skip incomplete {policy}: {len(rows)} < {args.min_records}')
            continue
        actions = tuple(rows[0]['score'])
        if any(tuple(r['score']) != actions for r in rows):
            raise RuntimeError(f'inconsistent action set in {policy}')
        uncal = metric(rows, {a: 0.0 for a in actions})
        best = None
        m = re.search(r'checkpoint-(\d+)', policy)
        step = int(m.group(1)) if m else 10**9
        for bias in grids(actions):
            cur = metric(rows, bias)
            cur.update(policy=policy, actions_set=list(actions), biases=bias,
                       step=step, n=len(rows), uncalibrated=uncal)
            if best is None or key(cur) > key(best):
                best = cur
        results.append(best)
        print(f'{policy:<80} raw={uncal["score"]:.4f}/{uncal["pass"]:.4f} '
              f'cal={best["score"]:.4f}/{best["pass"]:.4f} '
              f'recovery={best["recovery_rate"]:.3f} bias={best["biases"]}')

    if not results:
        raise SystemExit('no complete checkpoint policies')
    best = max(results, key=key)
    adapter = best['policy']
    if not os.path.isabs(adapter):
        adapter = os.path.join(P, adapter)
    selected = {
        'adapter': adapter,
        'actions': ','.join(best['actions_set']),
        'biases': ','.join(f'{a}:{best["biases"][a]}' for a in best['actions_set']),
        'holdout_score': best['score'],
        'holdout_pass': best['pass'],
        'holdout_recovery_rate': best['recovery_rate'],
        'holdout_action_counts': best['actions'],
        'holdout_n': best['n'],
        'checkpoint_step': best['step'],
        'uncalibrated': best['uncalibrated'],
        'selection_key': 'score, pass, lower recovery, smaller bias, earlier checkpoint',
    }
    json.dump(selected, open(args.out, 'w'), indent=2)
    print('\nSELECTED')
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
