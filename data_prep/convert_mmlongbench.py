#!/usr/bin/env python3
"""Convert MMLongBench-Doc samples.json to the rag_dataset.json format used here.

Usage:
    python data_prep/convert_mmlongbench.py --input data/MMLongBench/samples.json
"""
import argparse
import ast
import json
import os

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert_to_vidorag_format(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    vidorag_format = {"examples": []}
    for i, item in enumerate(data):
        uid = f"{item['doc_id'].replace('.pdf', '')}_{i+1}"

        # Parse evidence_pages from string "[5]" to list [5]
        pages_str = item.get('evidence_pages', '[]')
        try:
            reference_pages = ast.literal_eval(pages_str) if isinstance(pages_str, str) else pages_str
        except (ValueError, SyntaxError):
            reference_pages = []

        # Determine source_type from evidence_sources
        evidence_sources = item.get('evidence_sources', '')
        if 'Chart' in evidence_sources or 'Table' in evidence_sources:
            source_type = '2d_layout'
        elif 'Layout' in evidence_sources:
            source_type = '2d_layout'
        else:
            source_type = 'text'

        # Determine query_type
        if 'Count' in item.get('question', ''):
            query_type = 'counting'
        else:
            query_type = 'content_extraction'

        vidorag_format["examples"].append({
            "uid": uid,
            "query": item['question'],
            "reference_answer": item['answer'],
            "meta_info": {
                "file_name": item['doc_id'],
                "reference_page": reference_pages,
                "source_type": source_type,
                "query_type": query_type,
            },
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vidorag_format, f, indent=4, ensure_ascii=False)
    print(f"Converted {len(vidorag_format['examples'])} samples to {output_path}")
    return vidorag_format


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(PROJECT_ROOT, 'data/MMLongBench/samples.json'))
    ap.add_argument('--output', default=os.path.join(PROJECT_ROOT, 'data/MMLongBench/rag_dataset.json'))
    args = ap.parse_args()
    convert_to_vidorag_format(args.input, args.output)
