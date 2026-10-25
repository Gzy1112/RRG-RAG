#!/usr/bin/env python3
"""Merge sharded closed-loop result files and recompute the metrics.

Usage:
    python scripts/merge_eval_shards.py outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa.jsonl \
        outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa_shard0.jsonl \
        outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa_shard1.jsonl

Writes <out>.jsonl (one row per uid, first occurrence wins), <out>_eval.json
and <out>_eval_type_wise.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.overall_evaluator import eval_search, eval_search_type_wise  # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, shards = sys.argv[1], sys.argv[2:]
    rows, seen = [], set()
    for path in shards:
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row['uid'] not in seen:
                        seen.add(row['uid'])
                        rows.append(row)
    with open(out, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    results = eval_search(rows)
    with open(out.replace('.jsonl', '_eval.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out.replace('.jsonl', '_eval_type_wise.json'), 'w') as f:
        json.dump(eval_search_type_wise(rows), f, indent=2, ensure_ascii=False)
    print(f'{len(rows)} questions  score={results.get("score")}  passing={results.get("passing")}')


if __name__ == '__main__':
    main()
