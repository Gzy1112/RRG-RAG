from typing import Optional, List, Mapping, Any, Dict
from pathlib import Path
import json
from tqdm import tqdm
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.mixture import GaussianMixture

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode, IndexNode, ImageNode, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llms.vl_embedding import VL_Embedding
from llms.qwen3_embedding import DEFAULT_QWEN3_EMBED_PATH, Qwen3Embedding, is_qwen3_embedding
from utils.format_converter import nodefile2node,nodes2dict


PROJECT_ROOT = Path(__file__).resolve().parent


def _dataset_dir(dataset: str) -> str:
    data_root = Path(os.environ.get("VIDORAG_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
    return str((data_root / dataset).resolve())


class HTTPSearchEngine:
    """Thin retrieval client that delegates to a running flask_server_vidorag.

    Lets an eval process reuse a resident HybridSearchEngine over HTTP instead of
    building its own in-process ColQwen2 + text embedders. That keeps ~5GB of
    embedder weights out of the eval process so a 3-agent VL composite fits next
    to the retrieval server on a single GPU. The retrieval itself (models, fusion,
    GMM) is exactly whatever the server was launched with, so results match an
    in-process engine using the same configuration.

    search() returns the same shape as SearchEngine/HybridSearchEngine
    (nodes2dict): {'source_nodes': [{'node': {...}, 'score': ...}, ...]}, with
    metadata.file_name preserved so downstream candidate_image derivation and
    recall metrics are identical to the in-process path.
    """

    def __init__(self, server_url, topk=20, use_gmm=True, timeout=120):
        self.server_url = server_url.rstrip('/')
        self.topk = topk
        self.use_gmm = use_gmm
        self.timeout = timeout
        # Attributes the eval harness may set for retrieval/hybrid experiment
        # types; harmless no-ops here since filtering happens server-side.
        self.gmm = use_gmm
        self.gmm_candidate_length = False
        self.return_raw = False
        # --- Task 96: env-flag retrieval variants forwarded per-request ---
        # Read at construction so an eval process launched with the flags in its
        # environment propagates them to the resident server. When all are
        # default/inactive the POST body below is byte-identical to production.
        self._fused_order = os.environ.get('VIDORAG_FUSED_ORDER', '0') == '1'
        self._trunc_mode = (os.environ.get('VIDORAG_TRUNC_MODE', '') or '').strip().lower()
        self._gbm_target_pool_env = os.environ.get('VIDORAG_GBM_TARGET_POOL', '') or ''
        import requests as _requests  # lazy import; only needed in HTTP mode
        self._requests = _requests
        # Fail fast if the server is unreachable rather than mid-eval.
        r = self._requests.get(f'{self.server_url}/health', timeout=10)
        r.raise_for_status()
        info = r.json()
        if not info.get('engine_loaded'):
            raise RuntimeError(f'Retrieval server at {self.server_url} has no engine loaded: {info}')
        print(f'[HTTPSearchEngine] Using retrieval server {self.server_url} '
              f'(dataset={info.get("dataset")}, topk={self.topk}, use_gmm={self.use_gmm})')

    def search(self, query):
        payload = {'questions': [query], 'N': self.topk, 'use_gmm': self.use_gmm}
        # Only attach Task-96 fields when a flag is active, so the default
        # request body (and therefore the server response) is unchanged.
        if self._fused_order:
            payload['fused_order'] = True
        if self._trunc_mode:
            payload['trunc_mode'] = self._trunc_mode
        if self._gbm_target_pool_env:
            payload['gbm_target_pool'] = self._gbm_target_pool_env
        resp = self._requests.post(
            f'{self.server_url}/search',
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json().get('results', [])
        documents = results[0].get('documents', []) if results else []
        source_nodes = []
        for doc in documents:
            metadata = dict(doc.get('metadata', {}) or {})
            # Guarantee a file_name key so eval's candidate_image derivation
            # (metadata.file_name -> basename -> .jpg) never KeyErrors.
            if 'file_name' not in metadata and 'filename' not in metadata:
                metadata['file_name'] = doc.get('image_path', '')
            source_nodes.append({
                'node': {
                    'metadata': metadata,
                    'text': doc.get('ocr_text', ''),
                    'image_path': doc.get('image_path', ''),
                },
                'score': doc.get('score'),
            })
        return {'source_nodes': source_nodes}


def rrf_fuse(ranked_lists: List[List[NodeWithScore]], k: int = 60,
             max_candidates: int = 40) -> List[NodeWithScore]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    For each document appearing in any input list:
        RRF_score(d) = Σ 1 / (k + rank_i(d))
    where rank is 1-based position in list i.

    Returns a single ranked list sorted by RRF_score descending, with
    NodeWithScore.score set to the RRF_score.
    """
    rrf_scores: Dict[str, float] = {}
    node_map: Dict[str, NodeWithScore] = {}

    for ranked_list in ranked_lists:
        for rank_idx, nws in enumerate(ranked_list):
            # Build a unique key from the node's identity
            node = nws.node
            image_path = getattr(node, 'image_path', '') or ''
            if image_path:
                key = '.'.join(os.path.basename(image_path).split('.')[:-1])
            else:
                metadata = getattr(node, 'metadata', {}) or {}
                filename = metadata.get('filename', '') or metadata.get('file_name', '')
                key = '.'.join(filename.split('.')[:-1]) if filename else f'node_{id(node)}'
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank_idx + 1)  # 1-based rank
            if key not in node_map:
                node_map[key] = nws

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    result = []
    for key in sorted_keys[:max_candidates]:
        nws = node_map[key]
        nws.score = rrf_scores[key]
        result.append(nws)
    return result


def gmm(recall_result: list[NodeWithScore], input_length: int=20, max_valid_length: int=10, min_valid_length: int=5) -> List[NodeWithScore]:
    candidates = list(recall_result[:max(0, input_length)])
    max_valid_length = max(0, max_valid_length)
    min_valid_length = max(0, min(min_valid_length, max_valid_length))
    if len(candidates) <= 1 or max_valid_length == 0:
        return candidates[:max_valid_length]

    scores = [node.score for node in candidates]
    if any(score is None or not np.isfinite(float(score)) for score in scores):
        return candidates[:max_valid_length]

    scores = np.array(scores)
    scores = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=1,random_state=0)
    gmm.fit(scores)
    labels = gmm.predict(scores)

    scores = scores.flatten()
    scores = [scores[labels == label] for label in np.unique(labels)]
    clusters = [np.array(candidates, dtype=object)[labels == label].tolist() for label in np.unique(labels)]

    max_values = np.array([np.max(p) for p in scores])
    sorted_indices = np.argsort(-max_values)
    
    if len(sorted_indices) == 1:
        return clusters[0][:max_valid_length]
    
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]
    
    valid_recall_result = list(clusters[max_index])

    if len(valid_recall_result) > max_valid_length:
        valid_recall_result = valid_recall_result[:max_valid_length]
    elif len(valid_recall_result) < min_valid_length:
        second_valid_recall_result_len = min_valid_length - len(valid_recall_result)
        valid_recall_result.extend(clusters[second_max_index][:second_valid_recall_result_len])

    return valid_recall_result

class SearchEngine:
    def __init__(self,dataset, node_dir_prefix=None,embed_model_name='BAAI/bge-m3'):# nvidia/NV-Embed-v2 "vidore/colqwen2-v0.1"
        Settings.llm = None
        self.gmm=False
        self.gmm_candidate_length = False
        self.return_raw = False
        self.input_gmm = 20
        self.max_output_gmm = 10
        self.min_output_gmm = 5
        self.dataset = dataset
        self.dataset_dir = _dataset_dir(dataset)
        if node_dir_prefix is None:
            if 'bge' in embed_model_name:
                node_dir_prefix = 'bge_ingestion'
            elif 'NV-Embed' in embed_model_name:
                node_dir_prefix = 'nv_ingestion'
            elif 'colqwen' in embed_model_name:
                node_dir_prefix = 'colqwen_ingestion'
            elif 'openbmb' in embed_model_name:
                node_dir_prefix = 'visrag_ingestion'
            elif 'colpali' in embed_model_name:
                node_dir_prefix = 'colpali_ingestion'
            elif is_qwen3_embedding(embed_model_name):
                # Checked after colqwen/colpali so VL model ids keep their own dirs.
                node_dir_prefix = 'qwen3_ingestion'
            else:
                raise ValueError('Please specify the node_dir_prefix')
        
        if node_dir_prefix in ['colqwen_ingestion','visrag_ingestion','colpali_ingestion']:
            self.vl_ret = True
        else:
            self.vl_ret = False
        
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.rag_dataset_path = os.path.join(self.dataset_dir, 'rag_dataset.json')
        self.workers = 1
        self.embed_model_name = embed_model_name
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name:
            if self.vl_ret:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
            else:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='text')
        elif is_qwen3_embedding(embed_model_name):
            # Qwen3-Embedding is asymmetric: the instruction prefix is applied to
            # queries only, inside Qwen3Embedding._get_query_embedding. Stored node
            # embeddings (qwen3_ingestion) were produced without any prefix.
            model_path = embed_model_name if os.path.isdir(embed_model_name) else DEFAULT_QWEN3_EMBED_PATH
            # Node embeddings are precomputed on disk, so this model only encodes
            # the query string at search time. When VRAM is tight next to a full
            # 3-agent VL eval, VIDORAG_QWEN3_DEVICE=cpu offloads that (~1.2GB)
            # query encoding to CPU (float32 — half-precision matmul is not
            # supported on CPU) with negligible latency (~1 short query each).
            _q3_device = os.environ.get('VIDORAG_QWEN3_DEVICE', 'cuda')
            _q3_dtype = 'float32' if _q3_device.startswith('cpu') else 'float16'
            self.vector_embed_model = Qwen3Embedding(
                model_name=model_path, embed_batch_size=16, max_length=1024,
                device=_q3_device, torch_dtype=_q3_dtype)
        else:
            self.vector_embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, embed_batch_size=10, max_length=512, trust_remote_code=True, device='cuda')
        self.recall_num = 100
        self.query_engine = self.load_query_engine()
        self.output_dir = os.path.join(self.dataset_dir, 'search_output')
        # os.makedirs(self.output_dir, exist_ok=True)
    
    def online_search(self,query,node_list,topk=9):
        nodes = [TextNode.from_dict(node['node']) for node in node_list]
        vector_index = VectorStoreIndex(nodes, embed_model= self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=topk)
        node_postprocessors = self.load_node_postprocessors()
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=node_postprocessors
        )
        query_bundle = QueryBundle(query_str=query)
        recall_results = query_engine.retrieve(query_bundle)
        return nodes2dict(recall_results)
        
    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 10
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                try:
                    nodes = nodefile2node(input_file)
                    parsed_files.extend(nodes)
                except Exception as e:
                    print(f"[SearchEngine] Skipping corrupted node file {file}: {e}")
                    continue
        else:
            def parse_file(file,node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                try:
                    return nodefile2node(input_file)
                except Exception as e:
                    print(f"[SearchEngine] Skipping corrupted node file {file}: {e}")
                    return []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # results = list(tqdm(executor.map(parse_file, files, self.node_dir), total=len(files)))
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            for result in results:
                parsed_files.extend(result)
        return parsed_files
        
    def _load_vl_nodes_streaming(self):
        """Load ColQwen2 VL nodes — fast path via pre-converted .pt file.

        Tries to load a pre-converted .pt file first (created by
        precompute_vl_embeddings.py, ~2 min to load). Falls back to
        streaming JSON parsing if .pt file doesn't exist.

        The .pt file contains bfloat16 tensors (~9.7 GB for 52K nodes) and
        lightweight node dicts. Loading is I/O-bound, not CPU-bound.

        The streaming fallback uses executor.map to parse each .node file,
        immediately convert the embedding to bfloat16, and null the raw list.
        Peak RAM: ~10-15 GB (vs ~167 GB for naive bulk load).
        """

        # --- Fast path: load pre-converted .pt file ---
        pt_path = os.path.join(self.dataset_dir, 'colqwen_ingestion.pt')
        if os.path.exists(pt_path):
            print(f"[SearchEngine] Loading pre-converted VL embeddings from {pt_path}")
            start_time = __import__('time').time()
            data = torch.load(pt_path, weights_only=False)
            self.embedding_img = data['embeddings']
            # Reconstruct ImageNode objects from saved dicts
            self.nodes = [ImageNode.from_dict(doc) for doc in data['node_docs']]
            self._vl_embeddings_on_cpu = True
            elapsed = __import__('time').time() - start_time
            peak_gb = len(self.embedding_img) * 0.00019
            print(f"[SearchEngine] Loaded {len(self.embedding_img)} VL embeddings from .pt "
                  f"in {elapsed:.1f}s (peak ~{peak_gb:.1f} GB)")
            return
        max_nodes = os.environ.get('VIDORAG_VL_MAX_NODES')
        max_nodes = int(max_nodes) if max_nodes else None

        files = [f for f in os.listdir(self.node_dir)
                 if f.split('.')[-1] == 'node']
        if max_nodes is not None:
            files = files[:max_nodes]
            print(f"[SearchEngine] VIDORAG_VL_MAX_NODES={max_nodes}: subsampling node files (recall will degrade)")

        self.embedding_img = []
        nodes_light = []  # node objects with embedding=None (metadata only)
        bad = 0
        max_workers = 10

        def parse_file(file):
            input_file = os.path.join(self.node_dir, file)
            try:
                return nodefile2node(input_file)
            except Exception as e:
                print(f"[SearchEngine] Skipping corrupted node file {file}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(files), desc='Loading VL nodes (streaming)')
            # Use executor.map (NOT submit+as_completed) to avoid backlog:
            # submit creates all futures at once; if workers parse faster than
            # the main thread converts tensors, completed results pile up holding
            # ~3.2 MB raw float-lists each → OOM at ~100 GB for 7% of nodes.
            # executor.map yields results in submission order, processed inline.
            for result in executor.map(parse_file, files):
                if max_nodes is not None and len(self.embedding_img) >= max_nodes:
                    pbar.update(1)
                    continue
                for node in result:
                    try:
                        if node.embedding is None:
                            bad += 1
                            continue
                        tensor = torch.tensor(node.embedding).view(-1, 128).bfloat16()
                        self.embedding_img.append(tensor)
                        node.embedding = None  # free ~3.2 MB raw float-list NOW
                        nodes_light.append(node)
                    except Exception:
                        bad += 1
                pbar.update(1)
            pbar.close()

        self.nodes = nodes_light
        self._vl_embeddings_on_cpu = True
        peak_gb = len(self.embedding_img) * 0.00019  # ~199KB per bfloat16 tensor
        print(f"[SearchEngine] Streamed {len(self.embedding_img)} VL embeddings to "
              f"CPU bfloat16 tensors (peak ~{peak_gb:.1f} GB), skipped {bad} bad nodes, "
              f"cap={max_nodes}")

    def load_query_engine(self):
        if self.vl_ret and 'vidore' in self.embed_model_name:
            # ColQwen2 path: stream-load + immediate bfloat16 conversion to
            # avoid the CPU-RAM OOM from holding 52K raw float-list embeddings
            # (~167 GB). Embeddings stay on CPU; search() batches GPU transfers.
            self._load_vl_nodes_streaming()
            return None
        # Text (BGE-M3) path: raw embeddings are small (~1.3 GB for 52K nodes),
        # safe to load all at once and build a VectorStoreIndex.
        print('Loading nodes...')
        self.nodes = self.load_nodes()
        retriever = self.load_retriever_embed(self.nodes)
        node_postprocessors = self.load_node_postprocessors()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors
        )
        return query_engine
        
    def load_node_postprocessors(self):
        return []
    
    def load_retriever_embed(self, nodes):
        vector_index = VectorStoreIndex(nodes, embed_model= self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=self.recall_num)
        return vector_retriever
    
    def search(self, query):
        if self.vl_ret and 'vidore' in self.embed_model_name:
            query_embedding = self.vector_embed_model.embed_text(query)
            # Batched scoring: move image embeddings to GPU in chunks to prevent OOM.
            # Each ColQwen2 embedding is ~[1000, 128] bfloat16 (~256KB), so 1000
            # embeddings per batch is ~256MB — well within GPU memory limits.
            device = self.vector_embed_model.embed_model.device
            if getattr(self, '_vl_embeddings_on_cpu', False):
                batch_size = 1000
                all_scores = []
                for i in range(0, len(self.embedding_img), batch_size):
                    batch = self.embedding_img[i:i+batch_size]
                    batch_gpu = [t.to(device) for t in batch]
                    batch_scores = self.vector_embed_model.processor.score(query_embedding, batch_gpu)
                    all_scores.append(batch_scores[0])
                    del batch_gpu
                    torch.cuda.empty_cache()
                scores = [torch.cat(all_scores)]
            else:
                scores = self.vector_embed_model.processor.score(query_embedding,self.embedding_img)
            k = min(100, scores[0].numel())
            values, indices = torch.topk(scores[0], k=k)
            recall_results = [self.nodes[i] for i in indices]
            for node in recall_results:
                node.embedding = None
            recall_results = [NodeWithScore(node=node, score=score) for node, score in zip(recall_results, values)]
            recall_results_output = recall_results
        else:
            query_bundle = QueryBundle(query_str=query)
            recall_results = self.query_engine.retrieve(query_bundle)
            recall_results_output = recall_results
        if self.gmm:
            recall_results_output = gmm(recall_results,self.input_gmm,self.max_output_gmm,self.min_output_gmm)
        if self.return_raw:
            return recall_results_output
        if self.gmm_candidate_length:
            candidate_length = [1,2,4,6,9,12,16,20]
            current_length = len(recall_results_output)
            # Pad up to the next fixed candidate length; if already >= the
            # largest bucket, keep the current length (min() on an empty list
            # would raise). Truncate the post-GMM list, not the raw recall.
            larger = [num for num in candidate_length if num >= current_length]
            target_length = larger[0] if larger else current_length
            recall_results_output = recall_results_output[:target_length]
        return nodes2dict(recall_results_output)
    
    def search_example(self,example):
        query = example['query']
        recall_result =  self.search(query)
        example['recall_result'] = recall_result
        return example
    
    def search_multi_session(self,output_file='search_result.json'):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.rag_dataset_path, 'r') as f:
            dataset = json.load(f)
        data = dataset['examples']
        results = []
        if self.workers == 1:
            for example in tqdm(data):
                results.append(self.search_example(example))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.search_example, example): example for example in data}
                for future in tqdm(as_completed(future_to_file), total=len(data), desc='Processing files'):
                    results.append(future.result())
        with open(os.path.join(self.output_dir, output_file), 'w') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ColQwen2.5 VL search engine (full-corpus MaxSim, drop-in for HybridSearchEngine)
# ---------------------------------------------------------------------------
class ColQwen25Engine:
    """ColQwen2.5 vision-language retrieval engine.

    Loads pre-computed page embeddings from ``colqwen25_ingestion_v3.pt`` and
    scores queries against ALL pages via batched MaxSim (late interaction).
    Returns results in the same NodeWithScore format as the ColQwen2 path,
    so it can be used as a drop-in replacement for ``SearchEngine`` in
    ``HybridSearchEngine``.

    Memory footprint: ~10 GB for 52K page embeddings (bf16).
    """

    # Paths (override via class attributes or env vars before instantiation).
    # BASE_MODEL_PATH: local snapshot of vidore/colqwen2.5-base.
    # ADAPTER_PATH: vidore/colqwen2.5-v0.2 with keys renamed for the installed
    # transformers (data_prep/fix_colqwen25_adapter.py).
    BASE_MODEL_PATH = os.environ.get(
        'VIDORAG_COLQWEN25_BASE', str(PROJECT_ROOT / 'models' / 'colqwen2.5-base'))
    ADAPTER_PATH = os.environ.get(
        'VIDORAG_COLQWEN25_ADAPTER', str(PROJECT_ROOT / 'models' / 'colqwen2.5-v0.2-fixed'))
    CACHE_PATH = None  # auto-resolved from dataset_dir

    def __init__(self, dataset, cache_path=None, device='cuda:0'):
        self.dataset = dataset
        self.dataset_dir = _dataset_dir(dataset)
        self.device = device
        self.gmm = False
        self.return_raw = False
        self.input_gmm = 20
        self.max_output_gmm = 10
        self.min_output_gmm = 5

        # Resolve cache path
        if cache_path:
            self.CACHE_PATH = cache_path
        elif not self.CACHE_PATH:
            self.CACHE_PATH = os.path.join(self.dataset_dir, 'colqwen25_ingestion_v3.pt')

        print(f"[ColQwen25Engine] Loading model: base={self.BASE_MODEL_PATH}")
        self._load_model()

        print(f"[ColQwen25Engine] Loading page embeddings: {self.CACHE_PATH}")
        self._load_cache()

        print(f"[ColQwen25Engine] Ready. {len(self.page_ids)} pages loaded.")

    def _load_model(self):
        """Load ColQwen2.5 base model + fixed LoRA adapter."""
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from peft import PeftModel
        from safetensors.torch import load_file
        from collections import OrderedDict

        # 1. Create model structure from base config
        model = ColQwen2_5.from_pretrained(
            self.BASE_MODEL_PATH, torch_dtype='auto', device_map='cpu'
        )

        # 2. Remap checkpoint keys: model.* -> language_model.*
        #    (modelscope checkpoint saved with older transformers version)
        idx_path = os.path.join(self.BASE_MODEL_PATH, 'model.safetensors.index.json')
        with open(idx_path) as f:
            weight_map = json.load(f)['weight_map']
        files = {}
        for k, v in weight_map.items():
            files.setdefault(v, []).append(k)
        state_dict = OrderedDict()
        for fname, keys in files.items():
            fpath = os.path.join(self.BASE_MODEL_PATH, fname)
            sd = load_file(fpath)
            for k in keys:
                new_k = k.replace('model.', 'language_model.', 1) if k.startswith('model.') else k
                state_dict[new_k] = sd[k]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[ColQwen25Engine] Base model remap: missing={len(missing)}, unexpected={len(unexpected)}")
        del state_dict

        # 3. Move to GPU
        model = model.to(self.device, dtype=torch.bfloat16)

        # 4. Load fixed LoRA adapter
        print(f"[ColQwen25Engine] Loading adapter: {self.ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, self.ADAPTER_PATH)
        model.eval()

        # 5. Load processor
        processor = ColQwen2_5_Processor.from_pretrained(self.BASE_MODEL_PATH)

        self.model = model
        self.processor = processor

    def _load_cache(self):
        """Load pre-computed page embeddings from .pt cache.
        
        Cache format: dict {page_id: Tensor[seq_len, 128] bf16}
        """
        data = torch.load(self.CACHE_PATH, weights_only=False)
        
        # Handle both formats: dict {page_id: tensor} or {page_ids: [], page_embs: []}
        if isinstance(data, dict) and 'page_ids' not in data:
            # Dict format: {page_id: tensor}
            self.page_ids = list(data.keys())
            self.page_embs = [data[pid] for pid in self.page_ids]
        else:
            # List format: {page_ids: [...], page_embs: [...]}
            self.page_ids = data['page_ids']
            self.page_embs = data['page_embs']
        
        print(f"[ColQwen25Engine] Loaded {len(self.page_ids)} page embeddings "
              f"(avg seq_len={sum(e.shape[0] for e in self.page_embs[:100])/100:.0f})")

        # Build ImageNode objects for each page (metadata only, no embeddings)
        self.nodes = []
        img_dir = os.path.join(self.dataset_dir, 'img')
        for pid in self.page_ids:
            # pid format: "doc_pageidx" (e.g., "10_101_3")
            parts = pid.rsplit('_', 1)
            if len(parts) == 2:
                doc, page_str = parts
                try:
                    page = int(page_str)
                except ValueError:
                    page = 0
            else:
                doc, page = pid, 0
            file_path = os.path.join(img_dir, f'{doc}_{page}.jpg')
            node = ImageNode(
                image_path=file_path,
                metadata={'filename': f'{doc}_{page}.pdf', 'file_name': file_path}
            )
            self.nodes.append(node)

    def search(self, query):
        """Encode query, compute MaxSim scores for ALL pages, return top results."""
        # 1. Encode query using model (not just processor)
        batch = self.processor.process_queries([query])
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        with torch.no_grad():
            query_emb = self.model(**batch)  # [1, q_len, 128]
        query_emb = query_emb[0].to(torch.bfloat16)  # [q_len, 128]

        # 2. Batched MaxSim scoring over all pages
        batch_size = 500  # 500 pages per batch (~500*770*128*2 bytes ≈ 100MB)
        all_scores = []
        for i in range(0, len(self.page_embs), batch_size):
            batch_embs = self.page_embs[i:i + batch_size]
            batch_gpu = [t.to(self.device) for t in batch_embs]
            # Compute MaxSim manually: einsum qd,pd->qp then sum(max per query token)
            # query_emb: [q_len, 128], page_emb: [p_len, 128]
            scores_list = []
            for pe in batch_gpu:
                sim = torch.einsum('qd,pd->qp', query_emb.float(), pe.float())  # [q_len, p_len]
                max_sim = sim.max(dim=1)[0]  # [q_len]
                score = max_sim.sum()  # scalar
                scores_list.append(score)
            all_scores.append(torch.stack(scores_list))
            del batch_gpu
            torch.cuda.empty_cache()

        scores = torch.cat(all_scores)  # [N_pages]

        # 3. Top-K selection
        k = min(100, scores.numel())
        values, indices = torch.topk(scores, k=k)

        # 4. Build NodeWithScore results
        results = []
        for score, idx in zip(values, indices):
            node = self.nodes[idx.item()]
            results.append(NodeWithScore(node=node, score=score.item()))

        # 5. Apply GMM if enabled (same interface as SearchEngine)
        if self.gmm and len(results) >= 2:
            results = gmm(results, self.input_gmm, self.max_output_gmm, self.min_output_gmm)

        if self.return_raw:
            return results

        return nodes2dict(results)


class HybridSearchEngine:
    
    def __init__(self,
                 dataset,
                 node_dir_prefix_vl = None,
                 node_dir_prefix_text = None,
                 embed_model_name_vl = 'vidore/colqwen2-v1.0',
                 embed_model_name_text = 'BAAI/bge-m3',
                 topk=10,
                 gmm=False,
                 vl_rerank=False):
        self.dataset = dataset
        self.dataset_dir = _dataset_dir(dataset)
        self.img_dir = os.path.join(self.dataset_dir, 'img')
        self.ppocr_dir = os.path.join(self.dataset_dir, 'bge_ingestion')
        self.topk = topk
        self.gmm = gmm
        self.vl_rerank = vl_rerank
        if embed_model_name_vl:
            if vl_rerank:
                # Re-ranking mode: load ColQwen2 model only (no bulk embedding load).
                # Embeddings are loaded on-demand for each query's text-search candidates.
                # This avoids the ~167 GB RAM needed to hold all 52K ColQwen2 embeddings.
                print("[HybridSearchEngine] VL re-ranking mode: loading ColQwen2 model only (no bulk embedding load)")
                self.vl_embed_model = VL_Embedding(model=embed_model_name_vl, mode='text')
                self.vl_node_dir = os.path.join(self.dataset_dir, 'colqwen_ingestion')
                self._vl_cache = {}
                self.engine_vl = None
            else:
                try:
                    # --- ColQwen2.5 detection: use dedicated engine ---
                    # If the VL model name contains 'colqwen2.5' or 'colqwen2_5',
                    # use the ColQwen25Engine which does full-corpus MaxSim scoring.
                    # This is a drop-in replacement for the ColQwen2 SearchEngine.
                    _vl_name_lower = (embed_model_name_vl or '').lower()
                    if 'colqwen2.5' in _vl_name_lower or 'colqwen2_5' in _vl_name_lower:
                        print(f"[HybridSearchEngine] Detected ColQwen2.5 VL model: {embed_model_name_vl}")
                        print("[HybridSearchEngine] Using ColQwen25Engine for full-corpus MaxSim retrieval")
                        self.engine_vl = ColQwen25Engine(dataset=dataset)
                        # Propagate GMM settings from HybridSearchEngine to the VL engine
                        if gmm:
                            self.engine_vl.gmm = True
                            self.engine_vl.input_gmm = topk * 2
                            self.engine_vl.max_output_gmm = topk
                            self.engine_vl.min_output_gmm = max(topk // 2, 5)
                    else:
                        # Original ColQwen2 / other VL model path (unchanged)
                        self.engine_vl = SearchEngine(dataset,node_dir_prefix=node_dir_prefix_vl,embed_model_name=embed_model_name_vl)
                except Exception as e:
                    import traceback
                    # Silent degradation to text-only retrieval would skew every
                    # downstream reward/metric without any client-visible signal,
                    # so fail fast unless the fallback is explicitly allowed.
                    if os.environ.get('VIDORAG_ALLOW_VL_FALLBACK', '0') != '1':
                        print(f"[HybridSearchEngine] VL engine initialization failed: {e}")
                        print("[HybridSearchEngine] Set VIDORAG_ALLOW_VL_FALLBACK=1 to run text-only instead.")
                        raise
                    print(f"[HybridSearchEngine] VL engine initialization failed: {e}")
                    traceback.print_exc()
                    print(f"[HybridSearchEngine] Falling back to text-only mode (BGE-M3 only).")
                    self.engine_vl = None
        else:
            self.engine_vl = None
        self.engine_text = SearchEngine(dataset,node_dir_prefix=node_dir_prefix_text,embed_model_name=embed_model_name_text)
        # Text content for the final nodes is read back from the text engine's own
        # node dir (bge_ingestion by default, qwen3_ingestion when swapped) — the
        # stored text is identical across text embedders, only vectors differ.
        self.ppocr_dir = self.engine_text.node_dir
        # --- Task 96: env-flag gated retrieval-return variants (default OFF) ---
        # These flags change ONLY the return path (which candidates survive and
        # in what presentation order). When all are inactive, ``search`` falls
        # through to the untouched production path (GMM + doc-group/page-ascending
        # reorder), so the default deployment is byte-identical to production.
        # Values may be overridden per-request by the flask server, which sets
        # these three attrs under its engine lock and restores them afterwards.
        self._fused_order = os.environ.get('VIDORAG_FUSED_ORDER', '0') == '1'
        self._trunc_mode = (os.environ.get('VIDORAG_TRUNC_MODE', '') or '').strip().lower()
        self._gbm_target_pool_env = os.environ.get('VIDORAG_GBM_TARGET_POOL', '') or ''
        self._gbm_truncator = None  # lazy-loaded only when a gbm trunc mode is used
        # --- P0 retrieval improvements: fusion method + GMM params ---
        self._fusion_method = os.environ.get('VIDORAG_FUSION', 'score').strip().lower()
        self._rrf_k = int(os.environ.get('VIDORAG_RRF_K', '60'))
        self._gmm_max_output = int(os.environ.get('VIDORAG_GMM_MAX', str(topk)))
        self._gmm_min_output = int(os.environ.get('VIDORAG_GMM_MIN', '5'))
        # --- Asymmetric fusion weights ---
        self._text_weight = float(os.environ.get('VIDORAG_TEXT_WEIGHT', '0.5'))
        self._vl_weight = float(os.environ.get('VIDORAG_VL_WEIGHT', '0.5'))
        if self._fusion_method == 'rrf':
            print(f"[HybridSearchEngine] Using RRF fusion (k={self._rrf_k}) instead of score fusion")
        if self._gmm_max_output != topk or self._gmm_min_output != 5:
            print(f"[HybridSearchEngine] GMM params overridden: max_output={self._gmm_max_output}, min_output={self._gmm_min_output}")

    def _task96_flag_active(self):
        """True iff any Task-96 flag / per-request override is in effect."""
        return bool(self._fused_order) or bool(self._trunc_mode)

    @staticmethod
    def _prod_reorder_pairs(pairs):
        """Reproduce the production doc-group + page-ascending presentation order.

        Input: ordered list of (doc, page) pairs in fused-rank order. Output: the
        same pairs re-grouped by document (documents in first-appearance order)
        with pages ascending within each document — exactly the rebuild the
        production ``search`` performs on ``result_gmm``.
        """
        result_docs = {}
        for doc, page in pairs:
            if doc not in result_docs:
                result_docs[doc] = [page]
            elif page not in result_docs[doc]:
                result_docs[doc].append(page)
        ordered = []
        for doc, pages in result_docs.items():
            for page in sorted(pages):
                ordered.append((doc, page))
        return ordered

    def _gbm_select(self, merged_nodes):
        """Learned adaptive truncation via the Task-95 GBM (lazy-loaded once)."""
        from task95_fusion_hook import GBMAdaptiveTruncator, parse_gbm_target_pool
        if self._gbm_truncator is None:
            self._gbm_truncator = GBMAdaptiveTruncator.get()
        scores = [nws.score if nws.score is not None else 0.0 for nws in merged_nodes]
        target_pool = parse_gbm_target_pool(self._trunc_mode, self._gbm_target_pool_env)
        keep_idx = self._gbm_truncator.select_indices(scores, target_pool)
        return [merged_nodes[i] for i in keep_idx]

    def _search_flagged(self, merged_nodes):
        """Task-96 flagged return path (only reached when a flag is active).

        ``merged_nodes`` is the fused-descending NodeWithScore list (scores set).
        Truncation is chosen by ``self._trunc_mode``:
          * ``''``      -> production GMM (same call as the default path),
          * ``topk40``  -> fused top-40 (no GMM, no topk=20 cap),
          * ``gbm<K>``  -> learned adaptive cut targeting avg pool ~K.
        Presentation order is chosen by ``self._fused_order``:
          * True  -> keep fused-rank order (recovers Recall@1 0.280 -> 0.407),
          * False -> production doc-group/page-ascending rebuild.
        """
        trunc_mode = self._trunc_mode
        # --- 1. decide WHICH candidates survive ---
        if trunc_mode in ('topk40', 'topk_40'):
            from task95_fusion_hook import TOPK40_K
            selected = merged_nodes[:TOPK40_K]
        elif trunc_mode.startswith('gbm'):
            selected = self._gbm_select(merged_nodes)
        else:
            if self.gmm and len(merged_nodes) >= 2:
                selected = gmm(merged_nodes, self.topk * 2, self.topk, 5)
            else:
                selected = merged_nodes[:self.topk]
        # --- 2. convert to ordered (doc, page) pairs, then choose ORDER ---
        pairs = []
        for nws in selected:
            image_path = nws.node.image_path or ''
            file = '.'.join(os.path.basename(image_path).split('.')[:-1])
            parts = file.rsplit('_', 1)
            if len(parts) != 2:
                continue
            doc, page_str = parts
            try:
                page = int(page_str)
            except ValueError:
                continue
            pairs.append((doc, page))
        if not self._fused_order:
            pairs = self._prod_reorder_pairs(pairs)
        # --- 3. build final nodes (read OCR text back, score=None) in order ---
        recall_result = []
        seen = set()
        for doc, page in pairs:
            if (doc, page) in seen:
                continue
            seen.add((doc, page))
            with open(os.path.join(self.ppocr_dir, f'{doc}_{page}.node'), 'r') as f:
                text = json.load(f)
            text = ' '.join([item['text'] for item in text])
            file_path = os.path.join(self.img_dir, f'{doc}_{page}.jpg')
            node = ImageNode(image_path=file_path, text=text, metadata=dict(file_name=file_path))
            recall_result.append(NodeWithScore(node=node, score=None))
        return nodes2dict(recall_result)

    def _load_vl_embedding_for_page(self, page_id):
        """Load ColQwen2 embedding for a single page on-demand (with caching)."""
        if page_id in self._vl_cache:
            return self._vl_cache[page_id]
        node_file = os.path.join(self.vl_node_dir, f'{page_id}.node')
        if not os.path.exists(node_file):
            return None
        try:
            nodes = nodefile2node(node_file)
            for node in nodes:
                if node.embedding is not None:
                    tensor = torch.tensor(node.embedding).view(-1, 128).bfloat16()
                    self._vl_cache[page_id] = tensor
                    return tensor
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_page_id_from_node(node):
        """Extract page identifier from a LlamaIndex node object."""
        image_path = getattr(node, 'image_path', None)
        if image_path:
            return '.'.join(os.path.basename(image_path).split('.')[:-1])
        metadata = getattr(node, 'metadata', {}) or {}
        filename = metadata.get('filename', '')
        if filename:
            return '.'.join(os.path.basename(filename).split('.')[:-1])
        return ''

    def _vl_rerank(self, query, text_results_raw):
        """Re-rank text search results using ColQwen2 visual scoring (on-demand).

        Combines normalized BGE-M3 and ColQwen2 scores (0.5/0.5) so that GMM
        clustering works on a score distribution that includes both text and
        visual relevance. ColQwen2 MaxSim scores (~50-200) are min-max
        normalized to [0,1] before combining with BGE-M3 cosine scores.
        """
        query_embedding = self.vl_embed_model.embed_text(query)
        device = self.vl_embed_model.embed_model.device

        # Save original BGE-M3 scores for normalization
        bge_scores = [node_ws.score if node_ws.score is not None else 0.0
                      for node_ws in text_results_raw]

        # Load ColQwen2 embeddings for candidates on-demand
        embeddings = []
        valid_indices = []
        for i, node_ws in enumerate(text_results_raw):
            page_id = self._extract_page_id_from_node(node_ws.node)
            if not page_id:
                continue
            emb = self._load_vl_embedding_for_page(page_id)
            if emb is not None:
                embeddings.append(emb.to(device))
                valid_indices.append(i)

        if not embeddings:
            return text_results_raw

        vl_scores_raw = self.vl_embed_model.processor.score(query_embedding, embeddings)
        vl_scores_raw = vl_scores_raw[0]  # 1D tensor [num_valid]

        # Normalize BGE-M3 scores to [0, 1]
        bge_min = min(bge_scores) if bge_scores else 0.0
        bge_max = max(bge_scores) if bge_scores else 1.0
        bge_range = max(bge_max - bge_min, 1e-8)

        # Normalize ColQwen2 scores to [0, 1]
        vl_vals = [vl_scores_raw[j].item() for j in range(len(valid_indices))]
        vl_min = min(vl_vals) if vl_vals else 0.0
        vl_max = max(vl_vals) if vl_vals else 1.0
        vl_range = max(vl_max - vl_min, 1e-8)

        # Combined score = 0.5 * norm_bge + 0.5 * norm_vl
        valid_set = set(valid_indices)
        for j, idx in enumerate(valid_indices):
            bge_norm = (bge_scores[idx] - bge_min) / bge_range
            vl_norm = (vl_vals[j] - vl_min) / vl_range
            text_results_raw[idx].score = 0.5 * bge_norm + 0.5 * vl_norm
        # Pages without ColQwen2 embedding: keep normalized BGE score only
        for i in range(len(text_results_raw)):
            if i not in valid_set:
                text_results_raw[i].score = (bge_scores[i] - bge_min) / bge_range

        text_results_raw.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        return text_results_raw

    def _search_rerank(self, query):
        """Hybrid search with VL re-ranking (memory-efficient: no bulk embedding load)."""
        self.engine_text.return_raw = True
        result_text = self.engine_text.search(query)

        result_reranked = self._vl_rerank(query, result_text)

        if self.gmm:
            result_gmm = gmm(result_reranked, self.topk * 2, self.topk, 5)
        else:
            result_gmm = result_reranked[:self.topk]
            for node in result_gmm:
                node.score = None

        result_gmm_dict = nodes2dict(result_gmm)

        result_docs = dict()
        for node_dict in result_gmm_dict['source_nodes']:
            node = node_dict.get('node', {})
            image_path = node.get('image_path', '')
            if image_path:
                file = '.'.join(os.path.basename(image_path).split('.')[:-1])
            else:
                metadata = node.get('metadata', {})
                filename = metadata.get('filename', '')
                file = '.'.join(os.path.basename(filename).split('.')[:-1])
            if not file:
                continue
            doc = '_'.join(file.split('_')[:-1])
            page = file.split('_')[-1]
            if doc not in result_docs:
                result_docs[doc] = [int(page)]
            else:
                if int(page) not in result_docs[doc]:
                    result_docs[doc].append(int(page))

        recall_result = []
        for key, pages in result_docs.items():
            pages = sorted(pages)
            for page in pages:
                with open(os.path.join(self.ppocr_dir, f'{key}_{page}.node'), 'r') as f:
                    text = json.load(f)
                text = ' '.join([item['text'] for item in text])
                file_path = os.path.join(self.img_dir, f'{key}_{page}.jpg')
                node = ImageNode(image_path=file_path, text=text, metadata=dict(file_name=file_path))
                recall_result.append(NodeWithScore(node=node, score=None))
        return nodes2dict(recall_result)

    def search(self,query):
        if getattr(self, 'vl_rerank', False):
            return self._search_rerank(query)
        union_result = False
        if union_result:
            if self.gmm:
                self.engine_vl.gmm = True
                self.engine_text.gmm = True
                self.engine_vl.input_gmm = self.topk *2
                self.engine_text.input_gmm = self.topk *2
                self.engine_vl.max_output_gmm = self.topk
                self.engine_text.max_output_gmm = self.topk
                self.engine_vl.min_output_gmm = self.topk//2
                self.engine_text.min_output_gmm = self.topk//2
            result_vl = self.engine_vl.search(query)
            result_text = self.engine_text.search(query)
            result_vl['source_nodes'] = result_vl['source_nodes'][:self.topk]
            result_text['source_nodes'] = result_text['source_nodes'][:self.topk]

            result_docs = dict()
            for node in result_vl['source_nodes']:
                file = os.path.basename(node['node']['image_path']).split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
                
            for node in result_text['source_nodes']:
                file = node['node']['metadata']['filename'].split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
            
            recall_result = []
            
            for key,pages in result_docs.items():
                # from small to big
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir,f'{key}_{page}.node'),'r') as f:
                        text = json.load(f)
                    text = ' '.join([item['text'] for item in text])
                        
                    file_path = os.path.join(self.img_dir,f'{key}_{page}.jpg')
                    
                    node = ImageNode(image_path=file_path,text=text,metadata=dict(file_name=file_path))
                    
                    recall_result.append(NodeWithScore(node=node,score=None))
            
            return nodes2dict(recall_result)
        
        else:
            # Hybrid fusion: get raw results from both engines, then apply
            # either score-based fusion (0.5*text + 0.5*VL with min-max norm)
            # or RRF (Reciprocal Rank Fusion), followed by GMM truncation.

            self.engine_text.return_raw = True
            result_text = self.engine_text.search(query)

            # Get VL results if engine available
            result_vl = None
            if self.engine_vl is not None:
                self.engine_vl.return_raw = True
                result_vl = self.engine_vl.search(query)

            # --- Build merged_nodes via chosen fusion method ---
            if self._fusion_method == 'rrf':
                # RRF fusion: rank-based, no score normalization needed
                ranked_lists = []
                if result_text:
                    ranked_lists.append(result_text)
                if result_vl:
                    ranked_lists.append(result_vl)
                rrf_nodes = rrf_fuse(ranked_lists, k=self._rrf_k,
                                     max_candidates=self.topk * 2)
                # Normalize nodes: ensure all have image_path (TextNode → ImageNode)
                merged_nodes = []
                for nws in rrf_nodes:
                    node = nws.node
                    image_path = getattr(node, 'image_path', '') or ''
                    if not image_path:
                        metadata = getattr(node, 'metadata', {}) or {}
                        filename = metadata.get('filename', '') or metadata.get('file_name', '')
                        if filename:
                            file = '.'.join(filename.split('.')[:-1])
                            image_path = os.path.join(self.img_dir, f'{file}.jpg')
                    if image_path:
                        new_node = ImageNode(
                            image_path=image_path,
                            metadata={'filename': os.path.basename(image_path), 'file_name': image_path}
                        )
                        merged_nodes.append(NodeWithScore(node=new_node, score=nws.score))
            else:
                # Score-based fusion (production default)
                merged = {}  # (doc, page) → score

                # --- Process text results ---
                if result_text:
                    text_scores = [n.score if n.score is not None else 0.0 for n in result_text]
                    t_min = min(text_scores)
                    t_max = max(text_scores)
                    t_range = max(t_max - t_min, 1e-8)

                    for n in result_text:
                        filename = n.node.metadata.get('filename', '')
                        file = '.'.join(filename.split('.')[:-1])
                        parts = file.rsplit('_', 1)
                        if len(parts) != 2:
                            continue
                        doc, page_str = parts
                        try:
                            page = int(page_str)
                        except ValueError:
                            continue
                        norm_score = ((n.score or 0.0) - t_min) / t_range
                        key = (doc, page)
                        if key not in merged:
                            merged[key] = self._text_weight * norm_score
                        else:
                            merged[key] += self._text_weight * norm_score

                # --- Process VL (ColQwen2) results ---
                if result_vl:
                    vl_scores = [n.score if n.score is not None else 0.0 for n in result_vl]
                    v_min = min(vl_scores)
                    v_max = max(vl_scores)
                    v_range = max(v_max - v_min, 1e-8)

                    for n in result_vl:
                        image_path = n.node.image_path or ''
                        file = '.'.join(os.path.basename(image_path).split('.')[:-1])
                        parts = file.rsplit('_', 1)
                        if len(parts) != 2:
                            continue
                        doc, page_str = parts
                        try:
                            page = int(page_str)
                        except ValueError:
                            continue
                        norm_score = ((n.score or 0.0) - v_min) / v_range
                        key = (doc, page)
                        if key not in merged:
                            merged[key] = self._vl_weight * norm_score
                        else:
                            merged[key] += self._vl_weight * norm_score

                # --- Sort by combined score (descending) ---
                sorted_pages = sorted(merged.items(), key=lambda x: x[1], reverse=True)

                # --- Create NodeWithScore objects ---
                merged_nodes = []
                for (doc, page), score in sorted_pages:
                    file_path = os.path.join(self.img_dir, f'{doc}_{page}.jpg')
                    node = ImageNode(
                        image_path=file_path,
                        metadata={'filename': f'{doc}_{page}.pdf', 'file_name': file_path}
                    )
                    merged_nodes.append(NodeWithScore(node=node, score=score))

            # --- Task 96: env-flag gated return variants (default: inactive) ---
            if self._task96_flag_active():
                return self._search_flagged(merged_nodes)

            # --- Apply GMM on merged results (with configurable params) ---
            if self.gmm and len(merged_nodes) >= 2:
                result_gmm = gmm(merged_nodes, self.topk * 2,
                                 self._gmm_max_output, self._gmm_min_output)
            else:
                result_gmm = merged_nodes[:self.topk]
                for node in result_gmm:
                    node.score = None

            # --- Build result_docs from GMM-filtered results ---
            result_docs = dict()
            for nws in result_gmm:
                image_path = nws.node.image_path or ''
                file = '.'.join(os.path.basename(image_path).split('.')[:-1])
                parts = file.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                doc, page_str = parts
                try:
                    page = int(page_str)
                except ValueError:
                    continue
                if doc not in result_docs:
                    result_docs[doc] = [page]
                elif page not in result_docs[doc]:
                    result_docs[doc].append(page)

            # --- Build final recall result ---
            recall_result = []
            for key,pages in result_docs.items():
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir,f'{key}_{page}.node'),'r') as f:
                        text = json.load(f)
                    text = ' '.join([item['text'] for item in text])
                    file_path = os.path.join(self.img_dir,f'{key}_{page}.jpg')
                    node = ImageNode(image_path=file_path,text=text,metadata=dict(file_name=file_path))
                    recall_result.append(NodeWithScore(node=node,score=None))
            # Hard limit: never return more than topk candidates.
            return nodes2dict(recall_result[:self.topk])
            
    
    

if __name__ == '__main__':
    datasets = ['ExampleDataset']
    for dataset in datasets:
        # search_engine = SearchEngine(dataset,node_dir_prefix='visrag_ingestion',embed_model_name='openbmb/VisRAG-Ret')
        # search_engine = SearchEngine(dataset,node_dir_prefix='nv_ingestion',embed_model_name='nvidia/NV-Embed-v2')
        search_engine = HybridSearchEngine(dataset,gmm=True)
        print(search_engine.search('ok'))
    

    