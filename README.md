# RRG-RAG

Anonymous code release accompanying the paper submission. Author information is removed for double-blind review.

RRG-RAG answers questions over large collections of visually rich document pages (slide decks, reports) with three role-specialized agents on a single **Qwen2.5-VL-7B** backbone. Each role is its own LoRA adapter on the shared base model:

| Agent | Input | Output |
|---|---|---|
| **Seeker** | question + the ColQwen2.5 candidate pages (GMM triage, 5-10 pages) | indices of the evidence pages `E0` |
| **Inspector** | question + `E0` + retrieval-confidence features | `STOP` (keep `E0`) or `RETRIEVE-MORE` (look for more evidence in the same document) |
| **Synthesizer** | question + final evidence pages | short answer |

The repository covers the full pipeline:

1. **Supervised initialization** of the Seeker and the Synthesizer (LLaMA-Factory SFT).
2. **Role-Specialized Sequential Optimization**: sequential PPO of the Synthesizer and the Seeker. For 2 cycles, the Synthesizer is optimized with PPO first, then the Seeker is optimized with PPO. The Seeker reward is end-to-end: `0.4 * evidence + 0.6 * answer F1`, where the answer comes from the Synthesizer of the previous round, kept frozen.
3. **Selective Visual Evidence Recovery** at inference:
   - **Rank-1 rescue**: the top-ranked retrieved page is appended to the selected evidence.
   - **Inspector-gated `RETRIEVE-MORE` (same-deck second hop)**: the Seeker re-selects from `E0` plus the best-ranked pages of the same deck(s), and the result is unioned with `E0`.
4. **GRPO-Based Inspector Controller Optimization**:
   - The Inspector is a fresh LoRA trained with GRPO.
   - Its rewards are counterfactual branch outcomes (the judge score of `STOP` vs `RETRIEVE-MORE`), cached on the SlideVQA val split.
   - The checkpoint and the action bias are selected on held-out val decks only.

Evaluation covers **SlideVQA** (test, 2,215 questions), **ViDoSeek** (1,142) and **MMLongBench-Doc** (1,082). Retrieval always runs over the whole page collection of each benchmark, never within the gold document.

**Paper terms and code identifiers.** Some identifiers in the code predate the paper's terminology:

| Paper | Code |
|---|---|
| Role-Specialized Sequential Optimization | `TFR` (`scripts/train_tfr.sh`, `outputs/TFR_7B/`, `LLaMA-Factory/src/train_restore.py` / `train_restore_7b.py`, eval variants `tfr` / `tfr_r1`) |
| Selective Visual Evidence Recovery | `run_closedloop_eval.py` (`CONTROLLER_MODE=inspector_v2`, `HOP2_MODE=deck`, `RANK1_RESCUE=append`) |
| GRPO-Based Inspector Controller Optimization | `inspector_grpo_v2` (`scripts/train_inspector_grpo.sh`, `LLaMA-Factory/src/train_restore_grpo.py`, `outputs/INSPECTOR_GRPO/`) |
| `RETRIEVE-MORE` action | `DECK` |
| Accuracy | `passing` (share of questions with judge score ≥ 4) |

---

## Repository layout

```
.
├── run_closedloop_eval.py            # inference + evaluation (all variants)
├── eval.py, vidorag_agents.py        # agent pipeline, LLM-judge scoring
├── search_engine.py                  # ColQwen2.5 retrieval engine + GMM triage
├── flask_server_vidorag.py           # resident retrieval server (HTTP)
├── rebuild_rank_cache_colqwen25.py   # per-question top-100 ranking cache
├── inspector_v2_prompt.py            # Inspector prompt / action scoring
├── build_controller_sft_data*.py     # prompt template + confidence features (shared by Inspector code)
├── build_inspector_branch_cache.py   # counterfactual STOP / RETRIEVE-MORE branch outcomes (val split)
├── eval_inspector_v2_offline.py      # offline scoring of Inspector checkpoints on val holdout
├── select_inspector_v2_policy.py     # checkpoint + action-bias selection
├── llms/  agent/  utils/             # model wrappers, prompts, judge, metrics
├── data_prep/                        # dataset conversion, page indexing, SFT data construction
├── scripts/                          # end-to-end launchers (see below)
└── LLaMA-Factory/                    # modified LLaMA-Factory (SFT, sequential PPO, Inspector controller GRPO)
    ├── src/train.py                  # SFT entry
    ├── src/train_restore.py          # Role-Specialized Sequential Optimization, Synthesizer round
    ├── src/train_restore_7b.py       # Role-Specialized Sequential Optimization, Seeker round (frozen Synthesizer on a 2nd GPU)
    ├── src/train_restore_grpo.py     # GRPO-Based Inspector Controller Optimization
    ├── src/llamafactory/train/ppo/   # trainers, rewards, workflows
    ├── examples/train_lora/*.yaml    # training configs
    └── data/dataset_info.json
```

---

## 1. Environment

The code was developed and tested with:
- Python 3.10, CUDA 12, 2 × A100 40GB.
- torch 2.6.0, transformers 5.9.0, peft 0.12.0, trl 0.9.6.

```bash
conda create -n rrg python=3.10 -y && conda activate rrg
pip install -r requirements.txt
```

Notes:
- LLaMA-Factory runs from source; no installation is needed, because `scripts/env.sh` puts `LLaMA-Factory/src` on `PYTHONPATH`.
- `DISABLE_VERSION_CHECK=1` is set because the tested library versions are newer than the upstream LLaMA-Factory pins.
- The LLM judge (`qwen3-vl-plus`) is called through the DashScope OpenAI-compatible endpoint:

```bash
export DASHSCOPE_API_KEY=...            # required for evaluation and the Inspector branch cache
# optional: export DASHSCOPE_BASE_URL=...  JUDGE_MODEL=qwen3-vl-plus
```

### Models

Put local snapshots under `models/`. Paths can be overridden in `scripts/env.sh`.

| Path | Source |
|---|---|
| `models/Qwen2.5-VL-7B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `models/colqwen2.5-base` | `vidore/colqwen2.5-base` |
| `models/colqwen2.5-v0.2` | `vidore/colqwen2.5-v0.2` |
| `BAAI/bge-m3` *(SFT data only; HF id or local path via `SFT_RETRIEVAL_EMBED_MODEL` / `--embed_model`)* | `BAAI/bge-m3` |

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
huggingface-cli download vidore/colqwen2.5-base      --local-dir models/colqwen2.5-base
huggingface-cli download vidore/colqwen2.5-v0.2      --local-dir models/colqwen2.5-v0.2
# The base checkpoint is loaded with its decoder remapped model.* -> language_model.*;
# rename the adapter keys accordingly (tensors are unchanged):
python data_prep/fix_colqwen25_adapter.py --src models/colqwen2.5-v0.2 --dst models/colqwen2.5-v0.2-fixed
```

All scripts read `scripts/env.sh`. Its main variables are:
- `BASE_MODEL`, `VIDORAG_COLQWEN25_BASE`, `VIDORAG_COLQWEN25_ADAPTER`;
- `VIDORAG_DATA_ROOT` (default `data/`) and `RRG_OUTPUT_ROOT` (default `outputs/`).

---

## 2. Data

Every benchmark lives in `data/<name>/`:
- `rag_dataset*.json`: questions in the format `{"examples": [{uid, query, reference_answer, meta_info: {file_name, reference_page, ...}}]}`;
- `img/`: one image per page, named `{doc}_{page}.jpg` with 1-indexed pages.

**SlideVQA.** Use the parquet release (train / val / test, 20 page images per deck) from [`NTT-hil-insight/SlideVQA`](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) (original release: [nttmdlab-nlp/SlideVQA](https://github.com/nttmdlab-nlp/SlideVQA)).

```bash
# parquet shards -> data/SlideVQA/data/*.parquet
python data_prep/convert_slidevqa_to_vidorag.py      # img/ + rag_dataset{,_train,_val,_test}.json
```

**ViDoSeek.** Download the dataset from [`Qiuchen-Wang/ViDoSeek`](https://huggingface.co/datasets/Qiuchen-Wang/ViDoSeek) (from ViDoRAG).
1. Save `vidoseek.json` (1,142 questions) as `data/ViDoSeek/rag_dataset.json`.
2. Unzip `vidoseek_pdf_document.zip` into `data/ViDoSeek/`, which creates `pdf/`.
3. Render the pages:

```bash
python data_prep/pdf2images.py --dataset ViDoSeek --backend pymupdf --dpi 150 --workers 8
```

**MMLongBench-Doc.** Put `samples.json` and the PDFs (`pdf/`) from the MMLongBench-Doc release under `data/MMLongBench/`, then run:

```bash
python data_prep/convert_mmlongbench.py              # -> data/MMLongBench/rag_dataset.json
python data_prep/pdf2images.py --dataset MMLongBench --backend pdf2image
```

### Page index and retrieval server

```bash
for DS in SlideVQA ViDoSeek MMLongBench; do
  CUDA_VISIBLE_DEVICES=0 python data_prep/ingest_colqwen25.py --dataset $DS   # data/$DS/colqwen25_ingestion_v3.pt
done

scripts/start_retrieval_server.sh SlideVQA    8000 0 &    # one resident server per benchmark
scripts/start_retrieval_server.sh ViDoSeek    8001 0 &
scripts/start_retrieval_server.sh MMLongBench 8002 0 &
```

The server serves the ColQwen2.5 MaxSim ranking with the GMM candidate triage (top-20 → 5-10 pages).

The closed loop also needs a per-question top-100 **rank cache**, which is used by rank-1 rescue, the `RETRIEVE-MORE` candidate expansion and the Inspector features:

```bash
scripts/build_rank_cache.sh SlideVQA    rag_dataset_val.json  outputs/rank_cache/SlideVQA_val.jsonl
scripts/build_rank_cache.sh SlideVQA    rag_dataset_test.json outputs/rank_cache/SlideVQA_test.jsonl
scripts/build_rank_cache.sh ViDoSeek    rag_dataset.json      outputs/rank_cache/ViDoSeek.jsonl    8001
scripts/build_rank_cache.sh MMLongBench rag_dataset.json      outputs/rank_cache/MMLongBench.jsonl 8002
```

### SFT data (SlideVQA train split)

**Seeker candidate pools** are built from BGE-M3 text retrieval over PP-OCRv3 page text:
- The OCR step needs `fastdeploy` and `opencv-python`, plus the PP-OCRv3 English det/rec models, the cls model and `en_dict.txt` under `models/ppocr/`.
- The agents never see OCR text.

```bash
python data_prep/ocr_slidevqa.py --workers 8                      # data/SlideVQA/ppocr/*.txt
python data_prep/ingest_text.py --dataset SlideVQA                # data/SlideVQA/bge_ingestion/
python data_prep/prepare_sft_data_v2.py --phase retrieve          # BGE-M3 retrieval cache
python data_prep/prepare_seeker_sft_v4.py                         # training_data_v4/seeker_sft_v4_train.json
```

```bash
# Synthesizer: gold pages + distractors, plain-answer targets
python data_prep/prepare_synth_sft_v6.py
python data_prep/prepare_synth_sft_v7.py                          # training_data_v7/synthesizer_sft_v7_train.json

# question -> gold answer table for the end-to-end Seeker reward
python data_prep/build_gold_answers.py                            # data/SlideVQA/vidorag_gold_answers.json
```

---

## 3. Training

All outputs go to `outputs/`. The GPU ids below are examples.

### 3.1 Supervised initialization

One GPU per role; LoRA r32/α64 on all LM linear layers, vision tower frozen.

```bash
scripts/train_sft.sh seeker      0
scripts/train_sft.sh synthesizer 0
```

### 3.2 Role-Specialized Sequential Optimization

Sequential PPO: 2 cycles × (Synthesizer PPO → Seeker PPO), 30 steps per round. Round A (Synthesizer) runs on GPU 0. Round B (Seeker) runs on GPUs 0,1: the policy on the first, the frozen Synthesizer on the second.

```bash
scripts/train_tfr.sh 0 0,1
# -> outputs/TFR_7B/cycle2/round_b_seeker, outputs/TFR_7B/cycle2/round_a_synth
```

### 3.3 GRPO-Based Inspector Controller Optimization

```bash
# (a) counterfactual branch outcomes on the SlideVQA val split (needs the SlideVQA server + judge)
scripts/build_inspector_branch_cache.sh 0 0 2 &
scripts/build_inspector_branch_cache.sh 1 1 2 &
wait
# (b) GRPO: actions {STOP, RETRIEVE-MORE} (STOP,DECK in the code), G=8, 8 questions/step, KL β=0.05, 120 steps, fresh LoRA r16/α32
scripts/train_inspector_grpo.sh 0 run1
# (c) checkpoint + bias selection on held-out val decks -> outputs/INSPECTOR_GRPO/selected_policy.json
scripts/select_inspector_policy.sh 0 run1
```

Hyperparameters are in `LLaMA-Factory/examples/train_lora/*.yaml` and in the launchers.

---

## 4. Evaluation

```bash
# usage: scripts/eval_closedloop.sh <gpu> <dataset> [variant] [port] [shard_id num_shards]
scripts/eval_closedloop.sh 0 SlideVQA    full   8000
scripts/eval_closedloop.sh 0 ViDoSeek    full   8001
scripts/eval_closedloop.sh 0 MMLongBench full   8002
```

| variant | configuration |
|---|---|
| `full` | full RRG-RAG: sequentially optimized agents + Selective Visual Evidence Recovery (rank-1 rescue + GRPO-trained Inspector choosing `STOP` / `RETRIEVE-MORE`, selected checkpoint and bias) |
| `rule` | sequentially optimized agents + rank-1 rescue + fixed rule (`RETRIEVE-MORE` iff the Seeker kept ≥ 2 pages) |
| `tfr_r1` | sequentially optimized agents + rank-1 rescue |
| `tfr` | sequentially optimized agents only, no evidence recovery |

*Sequentially optimized agents* are the Seeker and the Synthesizer after Role-Specialized Sequential Optimization.

**Output.** Results are written to `outputs/eval/<dataset>/<ds>_rrg_<variant>_hybrid_<ds>.jsonl`, one row per question with the full agent trace. Metrics are in `*_eval.json`:
- `score`: mean judge score on a 1-5 scale;
- `passing`: **Accuracy**, the share of questions with score ≥ 4.

**Resuming.** Runs resume automatically (`EVAL_NO_CLEAN=1`).

**Sharding.** Pass `shard_id num_shards`, then combine the shards:

```bash
python scripts/merge_eval_shards.py outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa.jsonl \
    outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa_shard{0,1}.jsonl
```

**Page cap.** Page images are capped at 1.6 MP (2,048 visual tokens). Every SlideVQA page is below the cap; native ViDoSeek / MMLongBench pages are downscaled.

### Reference results (judge: qwen3-vl-plus)

| Benchmark | Configuration | Score (1-5) | Accuracy (%) |
|---|---|---|---|
| SlideVQA test (2,215) | Role-Specialized Sequential Optimization only (`tfr`) | 3.986 | 70.1 |
| | + rank-1 rescue (`tfr_r1`) | 4.034 | 71.5 |
| | full RRG-RAG (`full`) | **4.084** | **72.8** |
| ViDoSeek (1,142) | Role-Specialized Sequential Optimization only (`tfr`) | 4.099 | 73.1 |
| | full RRG-RAG (`full`) | **4.178** | **75.5** |
| MMLongBench-Doc (1,082) | Role-Specialized Sequential Optimization only (`tfr`) | 2.375 | 26.9 |
| | full RRG-RAG (`full`) | **2.421** | **28.0** |

Accuracy is the share of questions with judge score ≥ 4 (`passing` in `*_eval.json`). LLM-judge scores can shift slightly across judge API versions. See the paper for the full tables and ablations.

---

## Acknowledgements

This code builds on [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache-2.0, see `LLaMA-Factory/LICENSE`), [MMOA-RAG](https://github.com/chenyiqun/mmoa-rag), [ColPali / ColQwen](https://github.com/illuin-tech/colpali) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). The datasets keep their original licenses: SlideVQA, ViDoSeek and MMLongBench-Doc.
