import os
import json
# read
import argparse
import math
from tqdm import tqdm
def recall_at_k(relevant_docs, retrieved_docs, k):
    recall_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            ret_set = set(ret)
        else:
            ret_set = set(ret[:k]) # 取前 k 个检索结果
        rel_set = set(rel)
        if len(rel_set) == 0:
            continue  # Skip queries with no relevant docs
        recall = len(rel_set & ret_set) / len(rel_set)
        recall_values.append(recall)
    return sum(recall_values) / len(recall_values) if recall_values else 0.0

def hit_rate_at_k(relevant_docs, retrieved_docs, k):
    hit_count = 0
    if not relevant_docs:
        return 0.0
    for rel, ret in zip(relevant_docs, retrieved_docs):
        # Use a per-query cutoff: assigning to k in-place leaked the first
        # query's length into every later query when k == -1.
        cutoff = len(ret) if k == -1 else k
        if any(doc in set(ret[:cutoff]) for doc in rel):
            hit_count += 1
    return hit_count / len(relevant_docs)

def ndcg_at_k(relevant_docs, retrieved_docs, k):
    def dcg(scores):
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

    ndcg_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        cutoff = len(ret) if k == -1 else k
        rel_set = set(rel)
        deduped_ret = list(dict.fromkeys(ret))
        #  DCG@k
        scores = [1 if doc in rel_set else 0 for doc in deduped_ret[:cutoff]]
        dcg_value = dcg(scores)
        #  IDCG@k
        ideal_scores = sorted([1] * len(rel_set) + [0] * max(0, cutoff - len(rel_set)), reverse=True)[:cutoff]
        idcg_value = dcg(ideal_scores)
        ndcg = dcg_value / idcg_value if idcg_value > 0 else 0
        ndcg_values.append(ndcg)
    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

def mrr_at_k(relevant_docs, retrieved_docs, k):
    rr_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        cutoff = len(ret) if k == -1 else k
        for rank, doc in enumerate(dict.fromkeys(ret[:cutoff]), start=1):
            if doc in rel:
                rr_values.append(1 / rank)
                break
        else:
            rr_values.append(0)
    return sum(rr_values) / len(rr_values) if rr_values else 0.0

def eval_sample(example):
    retrieved_docs_list = []
    relevant_docs_list = []
    
    file_name = example['meta_info']['file_name']

def _extract_retrieved_docs(example, result_key):
    result = example.get(result_key)
    if result is None:
        return []
    nodes = result.get('source_nodes', []) if isinstance(result, dict) else result
    retrieved_docs = []
    for node in nodes:
        if 'node' in node:
            metadata = node['node'].get('metadata', {})
            filename = metadata.get('filename') or metadata.get('file_name') or node['node'].get('image_path')
        else:
            metadata = node.get('metadata', {})
            filename = metadata.get('file_path') or metadata.get('filename') or metadata.get('file_name')
        if filename:
            retrieved_docs.append(os.path.splitext(os.path.basename(filename))[0])
    return retrieved_docs


def _retrieval_metrics(relevant_docs_list, retrieved_docs_list, prefix=''):
    results = {}
    for k in [-1, 1, 5, 10]:
        label = 'all' if k == -1 else k
        results[f'{prefix}Recall@{label}'] = recall_at_k(relevant_docs_list, retrieved_docs_list, k)
        results[f'{prefix}Hit rate@{label}'] = hit_rate_at_k(relevant_docs_list, retrieved_docs_list, k)
        results[f'{prefix}nDCG@{label}'] = ndcg_at_k(relevant_docs_list, retrieved_docs_list, k)
        results[f'{prefix}MRR@{label}'] = mrr_at_k(relevant_docs_list, retrieved_docs_list, k)
    return results


def eval_search(examples):
    relevant_docs_list = []
    retrieved_docs_list = []
    results = {}
    if not examples:
        return results
    if examples[0].get('recall_results',None) is not None:
        for example in tqdm(examples):
            file_name = example['meta_info']['file_name']
            file_name = os.path.splitext(file_name)[0]
            reference_page = example['meta_info']['reference_page']
            reference_page = [str(page) for page in reference_page]
            relevant_docs_list.append([file_name + '_' + page for page in reference_page])
            retrieved_docs_list.append(_extract_retrieved_docs(example, 'recall_results'))

        results.update(_retrieval_metrics(relevant_docs_list, retrieved_docs_list, prefix='Candidate'))
        # Preserve historical names for compatibility; they are candidate-pool metrics.
        results.update(_retrieval_metrics(relevant_docs_list, retrieved_docs_list))
        if any(example.get('seeker_results') is not None for example in examples):
            seeker_docs = [_extract_retrieved_docs(example, 'seeker_results') for example in examples]
            results.update(_retrieval_metrics(relevant_docs_list, seeker_docs, prefix='Seeker'))

    if examples[0].get('eval_result',None) is not None:
        score = 0
        passing = 0
        for example in examples:
            score += example['eval_result']['score']
            passing += example['eval_result']['passing']
        results.update(dict(
            score=score / len(examples),
            passing=passing / len(examples)
        ))
    return results

def eval_search_type_wise(examples):
    examples_by_type = {}
    for example in examples:
        query_type = example['meta_info'].get('query_type', '')
        source_type = example['meta_info'].get('source_type', '')
        if '-Hop' in query_type:
            source_type = query_type.split('_')[-1]
            query_type = query_type.split('_')[0]
        if query_type != '':
            if query_type not in examples_by_type:
                examples_by_type[query_type] = []
            examples_by_type[query_type].append(example)
        if source_type != '':
            if source_type not in examples_by_type:
                examples_by_type[source_type] = []
            examples_by_type[source_type].append(example)
    results = {}
    for query_type, examples in examples_by_type.items():
        results[query_type] = eval_search(examples)
    return results
