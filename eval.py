import os
import sys
import json
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core.schema import NodeWithScore, ImageNode

from search_engine import SearchEngine, HybridSearchEngine, ColQwen25Engine

from llms.llm import LLM
from llms.evaluator import Evaluator, BinaryEvaluator, JudgeUnavailableError

from utils.overall_evaluator import eval_search,eval_search_type_wise
from vidorag_agents import ViDoRAG_Agents

class MMRAG:
    def __init__(self,
                dataset='ExampleDataset',
                query_file='rag_dataset.json',
                experiment_type = 'retrieval_infer',
                generate_vlm='qwen-vl-max',
                merge_lora=False,
                embed_model_name='BAAI/bge-m3',
                embed_model_name_vl=None, # openbmb/VisRAG-Ret vidore/colqwen2-v1.0
                embed_model_name_text=None, # nvidia/NV-Embed-v2 BAAI/bge-m3
                workers_num = 1,
                topk=10,
                max_samples=None,
                seeker_model_path=None,
                inspector_model_path=None,
                synthesizer_model_path=None,
                binary_judge=False,
                binary_judge_model='qwen2.5-7b-instruct',
                shard_id=None,
                num_shards=None,
                vl_rerank=False,
                output_suffix=None):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.max_samples = max_samples
        self.dataset = dataset
        self.query_file = query_file
        self.binary_judge = binary_judge
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.output_suffix = output_suffix  # Optional suffix for output filename
        self.dataset_dir = os.path.join('./data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # ViDoRAG_Agents uses this one only as a FALLBACK for an agent that was
        # given no adapter of its own (vidorag_agents.py:
        # `seeker_vlm if seeker_vlm is not None else vlm`). When all three
        # per-agent adapters are supplied it never generates anything, yet it is
        # still a full model copy — ~7.6GB for the 3B. EVAL_SKIP_MAIN_VLM=1
        # drops it so a 3-agent composite eval fits on a GPU that is already
        # holding the retrieval server (4 copies + the server OOMs a 40GB card).
        # Off by default: every published number was produced with it loaded,
        # and any mode that omits an agent's adapter still needs the fallback.
        _skip_main_vlm = (
            os.environ.get('EVAL_SKIP_MAIN_VLM') == '1'
            and seeker_model_path and inspector_model_path and synthesizer_model_path
        )
        # Per-agent ablation (EVAL_SHARED_RAW_BASE=1): agents without an adapter are
        # served by the adapters' shared base with LoRA disabled instead of a second
        # full LocalVLM copy, so one-adapter mixes fit next to the retrieval server.
        _agent_paths = [p for p in (seeker_model_path, inspector_model_path, synthesizer_model_path) if p]
        _shared_raw_base = os.environ.get('EVAL_SHARED_RAW_BASE') == '1' and 0 < len(_agent_paths) < 3
        if _shared_raw_base:
            self.vlm = None  # set below from the shared base
        elif _skip_main_vlm:
            print("[Eval] EVAL_SKIP_MAIN_VLM=1 and all three agent adapters given — "
                  "skipping the redundant fallback VLM copy")
            self.vlm = None
        else:
            self.vlm = LLM(model_name=generate_vlm, merge_lora=merge_lora)
        if binary_judge:
            print(f"[Eval] Binary judge mode ON: using '{binary_judge_model}' with VRAG-RL Figure 12 prompt")
            self.evaluator = BinaryEvaluator(model_name=binary_judge_model)
        else:
            self.evaluator = Evaluator()

        # Per-agent LoRA adapters — use shared base model to save GPU memory.
        # Instead of loading 3 separate copies of the base VLM (~7 GB each),
        # load ONE base model and attach all adapters to it (~14 GB saved).
        self._seeker_vlm = None
        self._inspector_vlm = None
        self._synthesizer_vlm = None
        _agent_adapters = {}
        if seeker_model_path:
            _agent_adapters['seeker'] = seeker_model_path
        if inspector_model_path:
            _agent_adapters['inspector'] = inspector_model_path
        if synthesizer_model_path:
            _agent_adapters['synthesizer'] = synthesizer_model_path

        if len(_agent_adapters) >= 2 or _shared_raw_base:
            from llms.llm import SharedMultiAdapterVLM
            print(f"[SFT] Shared base model for {len(_agent_adapters)} agents: {list(_agent_adapters.keys())}")
            _shared = SharedMultiAdapterVLM(adapter_paths=_agent_adapters)
            if _shared_raw_base:
                if os.path.realpath(_shared._base_model_name) != os.path.realpath(generate_vlm):
                    raise ValueError(f"adapter base {_shared._base_model_name} != generate_vlm {generate_vlm}")
                print(f"[SFT] Agents without an adapter use the shared base with LoRA disabled ({generate_vlm})")
                self.vlm = _shared.get_raw_agent()
            if 'seeker' in _agent_adapters:
                print(f"[SFT] Seeker adapter: {_agent_adapters['seeker']}")
                self._seeker_vlm = _shared.get_agent('seeker')
            if 'inspector' in _agent_adapters:
                print(f"[SFT] Inspector adapter: {_agent_adapters['inspector']}")
                self._inspector_vlm = _shared.get_agent('inspector')
            if 'synthesizer' in _agent_adapters:
                print(f"[SFT] Synthesizer adapter: {_agent_adapters['synthesizer']}")
                self._synthesizer_vlm = _shared.get_agent('synthesizer')
        else:
            # Fallback: load each agent independently (uses more GPU memory)
            if seeker_model_path:
                print(f"[SFT] Loading Seeker LoRA adapter from: {seeker_model_path}")
                self._seeker_vlm = LLM(model_name=seeker_model_path, merge_lora=merge_lora)
            if inspector_model_path:
                print(f"[SFT] Loading Inspector LoRA adapter from: {inspector_model_path}")
                self._inspector_vlm = LLM(model_name=inspector_model_path, merge_lora=merge_lora)
            if synthesizer_model_path:
                print(f"[SFT] Loading Synthesizer LoRA adapter from: {synthesizer_model_path}")
                self._synthesizer_vlm = LLM(model_name=synthesizer_model_path, merge_lora=merge_lora)
        
        # load search_engine
        # When EVAL_RETRIEVAL_SERVER is set, delegate retrieval to a resident
        # flask_server_vidorag over HTTP instead of building an in-process engine.
        # This keeps the ColQwen2 + text embedders (~5GB) out of the eval process
        # so a 3-agent VL composite fits next to the retrieval server on one GPU.
        _retrieval_server = os.environ.get('EVAL_RETRIEVAL_SERVER')
        if _retrieval_server:
            from search_engine import HTTPSearchEngine
            self.search_engine = HTTPSearchEngine(
                _retrieval_server,
                topk=topk,
                use_gmm=os.environ.get('EVAL_RETRIEVAL_SERVER_GMM', '1') in ('1', 'true', 'True'),
            )
        elif embed_model_name_vl is not None and embed_model_name_text is not None:
            self.search_engine = HybridSearchEngine(self.dataset,
                                                    embed_model_name_vl=embed_model_name_vl,
                                                    embed_model_name_text=embed_model_name_text,
                                                    topk=topk,
                                                    vl_rerank=vl_rerank)
        elif embed_model_name_vl is not None and 'colqwen2.5' in embed_model_name_vl.lower():
            # ColQwen2.5 standalone mode (no text branch)
            print(f"[MMRAG] Using ColQwen2.5 standalone retrieval: {embed_model_name_vl}")
            # Put ColQwen2.5 on a separate GPU to avoid conflicts with agents
            colqwen_device = os.environ.get('VIDORAG_COLQWEN_DEVICE', 'cuda:0')
            self.search_engine = ColQwen25Engine(dataset=self.dataset, device=colqwen_device)
            self.search_engine.gmm = False  # will be enabled later if experiment_type requires it
        else:
            self.search_engine = SearchEngine(self.dataset, embed_model_name=embed_model_name)

        # retrieval only
        if experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'base_retrieval_{embed_model_name}.jsonl'
        # hybrid retrieval
        elif experiment_type == 'dynamic_hybird_retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.search_engine.gmm = True
            self.output_file_name = f'dynamic_hybird_retrieval_{embed_model_name_vl}_{embed_model_name_text}.jsonl'
        # vidorag
        elif experiment_type == 'vidorag':
            self.agents = ViDoRAG_Agents(
                vlm=self.vlm,
                seeker_vlm=self._seeker_vlm,
                inspector_vlm=self._inspector_vlm,
                synthesizer_vlm=self._synthesizer_vlm,
            )
            self.eval_func = self.vidorag
            self.search_engine.gmm = True
            self.search_engine.gmm_candidate_length = True
            # Use "sft" in filename when per-agent adapters are used
            if seeker_model_path or inspector_model_path or synthesizer_model_path:
                self.output_file_name = 'vidorag_sft_per_agent.jsonl'
            else:
                self.output_file_name = f'vidorag_{generate_vlm}.jsonl'
        
        # Append _binary suffix to output filename when binary judge mode is active
        if binary_judge:
            base, ext = os.path.splitext(self.output_file_name)
            self.output_file_name = f'{base}_binary{ext}'

        # Append optional output suffix for distinguishing runs
        if self.output_suffix:
            base, ext = os.path.splitext(self.output_file_name)
            self.output_file_name = f'{base}_{self.output_suffix}{ext}'

        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))

    def retrieval_infer(self,sample):
        query = sample['query']
        recall_results = self.search_engine.search(query)
        recall_results['source_nodes'] = recall_results['source_nodes'][:100]
        sample['recall_results'] = recall_results
        return sample
    def vidorag(self,sample):
        query = sample['query']
        print(query)
        try:
            recall_results = self.search_engine.search(query)
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cuda' in str(e):
                print(f"[Search CUDA Error] {e}")
                return None
            raise
        candidate_image = [os.path.join(self.img_dir, os.path.basename(node['node']['metadata'].get('file_name',node['node']['metadata'].get('filename')).replace('.txt','.jpg'))) for node in recall_results['source_nodes']]
        if 'gmm' not in self.experiment_type:
            candidate_image = candidate_image[:self.top_k]
        try:
            agent_result = self.agents.run_agent(query, candidate_image, return_trace=True)
            answer = agent_result.get("answer") if isinstance(agent_result, dict) else agent_result
            agent_trace = agent_result.get("trace", {}) if isinstance(agent_result, dict) else {}
        except Exception as e:
            print(f"[Agent Error] {e}")
            return None
        
        if answer is None:
            print("[Agent] No answer produced")
            return None

        try:
            sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        except JudgeUnavailableError:
            # The judge backend is down (billing/auth/quota). Recording score=0 here
            # would be written into the results file and then averaged, producing a
            # plausible-looking near-zero score that imitates a training collapse.
            # Abort instead so the run yields no number rather than a wrong one.
            raise
        except Exception as e:
            print(f"[Evaluator Error] {e}")
            sample['eval_result'] = dict(score=0, passing=0, judge=f"Evaluator failed: {e}")

        sample['response'] = str(answer)
        sample['agent_trace'] = agent_trace
        sample['recall_results'] = dict(
            source_nodes=[NodeWithScore(node=ImageNode(image_path=image,metadata=dict(file_name=image)), score=None).to_dict() for image in candidate_image],
            response=None,
            metadata=None)
        # Record the seeker's own selections separately from the candidate
        # pool so retrieval metrics can attribute recall to the right stage.
        selected_images = agent_trace.get("seeker_selected_images", [])
        sample['seeker_results'] = dict(
            source_nodes=[NodeWithScore(
                node=ImageNode(image_path=image, metadata=dict(file_name=image)), score=None
            ).to_dict() for image in selected_images],
            response=None,
            metadata=None,
        )
        return sample


    def eval_dataset(self):
        eval_func = self.eval_func
        
        rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data['examples']

        # Optional uid subset (one uid per line), applied before resume/max_samples.
        # Used to re-run only the questions a pipeline change can affect.
        _uid_file = os.environ.get('EVAL_UID_FILE')
        if _uid_file:
            with open(_uid_file) as f:
                _keep = {line.strip() for line in f if line.strip()}
            data = [item for item in data if item['uid'] in _keep]
            print(f"[UID filter] kept {len(data)} samples from {_uid_file}")
        
        # Collect UIDs already completed — check both the shard-specific file
        # AND the main output file (so sharded runs skip UIDs from prior
        # non-sharded or single-GPU runs that wrote to the main file).
        uid_already_set = set()
        if os.path.exists(self.output_file_path):
            with open(self.output_file_path, "r") as f:
                for line in f:
                    if line.strip():
                        uid_already_set.add(json.loads(line.strip())['uid'])
        # In sharded mode, also check the main (non-shard) output file.
        # Handles both naming conventions: _shard{N}.jsonl and _shard{N}of{M}.jsonl
        if self.shard_id is not None and self.num_shards is not None and self.num_shards > 1:
            import re
            main_path = re.sub(r'_shard\d+(?:of\d+)?\.jsonl$', '.jsonl', self.output_file_path)
            if main_path != self.output_file_path and os.path.exists(main_path):
                with open(main_path, "r") as f:
                    for line in f:
                        if line.strip():
                            uid_already_set.add(json.loads(line.strip())['uid'])
        if uid_already_set:
            data = [item for item in data if item['uid'] not in uid_already_set]
            print(f"[Resume] Skipped {len(uid_already_set)} already-done UIDs, {len(data)} remaining")

        # Limit number of samples for quick testing
        if self.max_samples is not None and len(data) > self.max_samples:
            data = data[:self.max_samples]
        
        # Shard data for parallel execution
        if self.shard_id is not None and self.num_shards is not None and self.num_shards > 1:
            shard_size = len(data) // self.num_shards
            start = self.shard_id * shard_size
            end = start + shard_size if self.shard_id < self.num_shards - 1 else len(data)
            data = data[start:end]
            print(f"[Shard {self.shard_id}/{self.num_shards}] Processing samples {start}-{end-1} ({len(data)} samples)")
            
        cuda_err_streak = 0
        if self.workers_num == 1:
            for item in tqdm(data):
                # Clear CUDA cache before each question to prevent fragmentation
                # that causes OOM on long sequences (SharedMultiAdapterVLM issue)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    result = eval_func(item)
                    cuda_err_streak = 0          # reset on success
                except JudgeUnavailableError as e:
                    self._abort_on_judge_outage(e)
                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'cuda' in str(e):
                        cuda_err_streak += 1
                        print(f"[CUDA Error] {e}")
                        # Retry once after clearing cache
                        try:
                            import torch
                            torch.cuda.empty_cache()
                            print(f"[CUDA Retry] Attempting retry after empty_cache...")
                            result = eval_func(item)
                            cuda_err_streak = 0
                            print(f"[CUDA Retry] Success!")
                        except Exception as retry_e:
                            print(f"[CUDA Retry] Failed: {retry_e}")
                            if cuda_err_streak >= 3:
                                print(f"[CUDA Error] {cuda_err_streak} consecutive CUDA errors, aborting")
                                raise
                            result = None
                    else:
                        raise
                except Exception as e:
                    print(f"[Eval Error] {type(e).__name__}: {e}")
                    result = None
                if result is None:
                    continue
                with open(self.output_file_path, "a") as f:
                    json.dump(result, f,ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    try:
                        result = future.result()
                    except JudgeUnavailableError as e:
                        for pending in futures:
                            pending.cancel()
                        if results:
                            with open(self.output_file_path, "a") as f:
                                for res in results:
                                    if res is None:
                                        continue
                                    f.write(json.dumps(res,ensure_ascii=False) + "\n")
                        self._abort_on_judge_outage(e)
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res,ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res,ensure_ascii=False) + "\n")

        return self.output_file_path

    def _abort_on_judge_outage(self, exc):
        """Stop the run with a non-zero exit and no score, so nothing downstream
        mistakes a judge outage for a model regression."""
        print(f"\n{'='*60}", file=sys.stderr)
        print("[FATAL] LLM judge backend unavailable -- eval ABORTED.", file=sys.stderr)
        print(f"  reason : {exc}", file=sys.stderr)
        print("  No score was written. Partial results kept for resume:", file=sys.stderr)
        print(f"  {self.output_file_path}", file=sys.stderr)
        print("  Restore judge access (DashScope credit / API key), then re-run the", file=sys.stderr)
        print("  same command: already-scored uids are skipped automatically.", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        sys.exit(3)

    def eval_overall(self):
        if not os.path.exists(self.output_file_path):
            print(f"[Warning] No results file found at {self.output_file_path}, skipping eval_overall")
            return
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search(data)
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
        if self.binary_judge:
            accuracy = results.get('passing', results.get('score', 0))
            print(f"\n{'='*60}")
            print(f"[Binary Judge] VRAG-RL-style Accuracy = {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  Judge: Qwen2.5-7B-Instruct (Figure 12 prompt)")
            print(f"  N = {len(data)} samples")
            print(f"  Results: {self.output_file_path.replace('.jsonl', '_eval.json')}")
            print(f"{'='*60}\n")
    
    def eval_overall_type_wise(self):
        if not os.path.exists(self.output_file_path):
            print(f"[Warning] No results file found at {self.output_file_path}, skipping eval_overall_type_wise")
            return
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search_type_wise(data)
        with open(self.output_file_path.replace(".jsonl", "_eval_type_wise.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
        
def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ExampleDataset', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='retrieval_infer', help="The type of experiment")
    parser.add_argument("--embed_model_name", type=str, default='BAAI/bge-m3', help="The name of embedding model")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--embed_model_name_vl", type=str, default=None, help="The name of embedding model for vl")
    parser.add_argument("--embed_model_name_text", type=str, default=None, help="The name of embedding model for text")
    parser.add_argument("--generate_vlm", type=str, default='qwen-vl-max', help="The name of VLM model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to LoRA adapter checkpoint directory (overrides --generate_vlm)")
    parser.add_argument("--merge_lora", action='store_true', default=False, help="Merge LoRA weights into base model for faster inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--seeker_model_path", type=str, default=None, help="Path to Seeker LoRA adapter directory")
    parser.add_argument("--inspector_model_path", type=str, default=None, help="Path to Inspector LoRA adapter directory")
    parser.add_argument("--synthesizer_model_path", type=str, default=None, help="Path to Synthesizer LoRA adapter directory")
    parser.add_argument("--binary_judge", action='store_true', default=False, help="Use binary True/False judge (VRAG-RL Figure 12 protocol with Qwen2.5-7B-Instruct) instead of the default 1-5 scale evaluator")
    parser.add_argument("--binary_judge_model", type=str, default='qwen2.5-7b-instruct', help="Judge model for binary mode (default: qwen2.5-7b-instruct via DashScope API)")
    parser.add_argument("--output_suffix", type=str, default=None, help="Optional suffix to append to output filename (for distinguishing runs)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    # Use model_path (LoRA adapter) if provided, otherwise use generate_vlm name
    vlm_model = args.model_path if args.model_path else args.generate_vlm
    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        embed_model_name=args.embed_model_name,
        workers_num=args.workers_num,
        topk=args.topk,
        embed_model_name_vl=args.embed_model_name_vl,
        embed_model_name_text=args.embed_model_name_text,
        generate_vlm=vlm_model,
        merge_lora=args.merge_lora,
        max_samples=args.max_samples,
        seeker_model_path=args.seeker_model_path,
        inspector_model_path=args.inspector_model_path,
        synthesizer_model_path=args.synthesizer_model_path,
        binary_judge=args.binary_judge,
        binary_judge_model=args.binary_judge_model,
        output_suffix=args.output_suffix,
    )
    mmrag.eval_dataset()
    mmrag.eval_overall()
    mmrag.eval_overall_type_wise()