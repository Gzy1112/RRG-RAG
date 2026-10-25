#!/usr/bin/env python3
"""Build the question -> gold answer table used by the Seeker e2e reward (TFR).

During Seeker PPO the frozen Synthesizer answers from the selected pages and the
answer is scored by F1 against the gold answer. The PPO workflows look answers
up by question text in data/SlideVQA/vidorag_gold_answers.json (override with
VIDORAG_GOLD_ANSWERS):

    {"<question>": {"answer": "<reference answer>"}, ...}

For repeated questions the first occurrence in the train split wins.

Usage:
    python data_prep/build_gold_answers.py
"""
import argparse
import json
import os

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDEVQA_DIR = os.path.join(PROJECT_ROOT, 'data', 'SlideVQA')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', default=os.path.join(SLIDEVQA_DIR, 'rag_dataset_train.json'))
    ap.add_argument('--out', default=os.path.join(SLIDEVQA_DIR, 'vidorag_gold_answers.json'))
    args = ap.parse_args()

    with open(args.train_file) as f:
        examples = json.load(f)['examples']
    gold = {}
    for ex in examples:
        gold.setdefault(ex['query'], {'answer': str(ex['reference_answer'])})
    with open(args.out, 'w') as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)
    print(f'{len(gold)} questions -> {args.out}')


if __name__ == '__main__':
    main()
