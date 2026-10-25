# Retrieve, Recover, and Generate: Collaborative Role-Specialized Reinforcement Learning with Evidence Retrieval and Recovery for Multi-modal Retrieval-Augmented Generation

Anonymous code release accompanying the paper submission of WWW 2027. Author information is removed for double-blind review.

RRG-RAG answers questions over large collections of visually rich document pages with three role-specialized agents on a single **Qwen2.5-VL-7B or Qwen2.5-VL-3B** backbone. Each role is its own LoRA adapter on the shared base model:

| Agent | Input | Output |
|---|---|---|
| **Seeker** | question + candidate pages (GMM triage, 5-10 pages) | indices of the evidence pages `E0` |
| **Inspector** | question + `E0` + retrieval-confidence features | `STOP` (keep `E0`) or `RETRIEVE-MORE` (look for more evidence in the same document) |
| **Synthesizer** | question + final evidence pages | short answer |

The repository covers the full pipeline:

1. **Supervised initialization** of the Seeker and the Synthesizer (LLaMA-Factory SFT).
2. **Role-Specialized Sequential Optimization**: sequential PPO of the Synthesizer and the Seeker. For 2 cycles, the Synthesizer is optimized with PPO first, then the Seeker is optimized with PPO. 
3. **Selective Visual Evidence Recovery** at inference:
   - **Rank-1 rescue**: the top-ranked retrieved page is appended to the selected evidence.
   - **Inspector-gated `RETRIEVE-MORE` (same-deck second hop)**: the Seeker re-selects from `E0` plus the best-ranked pages of the same deck(s), and the result is unioned with `E0`.
4. **GRPO-Based Inspector Controller Optimization**:
   - The Inspector is a fresh LoRA trained with GRPO.
   - Its rewards are counterfactual branch outcomes (the judge score of `STOP` vs `RETRIEVE-MORE`), cached on the SlideVQA val split.
   - The checkpoint and the action bias are selected on held-out val decks only.

Evaluation covers **SlideVQA**, **ViDoSeek** and **MMLongBench-Doc**. Retrieval always runs over the whole page collection of each benchmark, never within the gold document.

**Paper terms and code identifiers.** Some identifiers in the code predate the paper's terminology:

| Paper | Code |
|---|---|
| Role-Specialized Sequential Optimization | `TFR` (`scripts/train_tfr.sh`, `outputs/TFR_7B/`, `LLaMA-Factory/src/train_restore.py` / `train_restore_7b.py`, eval variants `tfr` / `tfr_r1`) |
| Selective Visual Evidence Recovery | `run_closedloop_eval.py` (`CONTROLLER_MODE=inspector_v2`, `HOP2_MODE=deck`, `RANK1_RESCUE=append`) |
| GRPO-Based Inspector Controller Optimization | `inspector_grpo_v2` (`scripts/train_inspector_grpo.sh`, `LLaMA-Factory/src/train_restore_grpo.py`, `outputs/INSPECTOR_GRPO/`) |
| `RETRIEVE-MORE` action | `DECK` |
| Accuracy | `passing` |

---

## Repository layout

```
.
├── run_closedloop_eval.py            # inference + evaluation (all variants)
├── eval.py, vidorag_agents.py        # agent pipeline, LLM-judge scoring
├── search_engine.py                  # retrieval engine + GMM triage
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
    ├── src/train_restore_7b.py       # Role-Specialized Sequential Optimization, Seeker round (frozen Synthesizer)
    ├── src/train_restore_grpo.py     # GRPO-Based Inspector Controller Optimization
    ├── src/llamafactory/train/ppo/   # trainers, rewards, workflows
    ├── examples/train_lora/*.yaml    # training configs
    └── data/dataset_info.json
```

---

## 1. Environment

The code was developed and tested with:
- Python 3.10, CUDA 12, A100 40GB.
- torch 2.6.0, transformers 5.9.0, peft 0.12.0, trl 0.9.6.

```bash
conda create -n rrg python=3.10 -y && conda activate rrg
pip install -r requirements.txt
```

---

## 2. Data

Every benchmark lives in `data/<name>/`:
- `rag_dataset*.json`: questions in the format `{"examples": [{uid, query, reference_answer, meta_info: {file_name, reference_page, ...}}]}`;
- `img/`: one image per page, named `{doc}_{page}.jpg` with 1-indexed pages.

**SlideVQA.** Use the parquet release from [`NTT-hil-insight/SlideVQA`](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) (original release: [nttmdlab-nlp/SlideVQA](https://github.com/nttmdlab-nlp/SlideVQA)).

```bash
# parquet shards -> data/SlideVQA/data/*.parquet
python data_prep/convert_slidevqa_to_vidorag.py      # img/ + rag_dataset{,_train,_val,_test}.json
```

**ViDoSeek.** Download the dataset released with [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) (see its README for the Hugging Face link).
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

### SFT data (SlideVQA train split)

**Seeker candidate pools**:
- The OCR step needs `fastdeploy` and `opencv-python`, plus the PP-OCRv3 English det/rec models, the cls model and `en_dict.txt` under `models/ppocr/`.
- The agents never see OCR text.

```bash
python data_prep/ocr_slidevqa.py --workers 8                      # data/SlideVQA/ppocr/*.txt
python data_prep/ingest_text.py --dataset SlideVQA                # data/SlideVQA/bge_ingestion/
python data_prep/prepare_sft_data_v2.py --phase retrieve          # retrieval cache
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


*Sequentially optimized agents* are the Seeker and the Synthesizer after Role-Specialized Sequential Optimization.

**Output.** Results are written to `outputs/eval/<dataset>/<ds>_rrg_<variant>_hybrid_<ds>.jsonl`, one row per question with the full agent trace. Metrics are in `*_eval.json`:
- `score`: mean judge score on a 1-5 scale;
- `Accuracy`: a binary model-based evaluation metric, scoring each response as correct (1) or incorrect (0).

```bash
python scripts/merge_eval_shards.py outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa.jsonl \
    outputs/eval/SlideVQA/slidevqa_rrg_full_hybrid_slidevqa_shard{0,1}.jsonl
```

### Reference results 

| Benchmark | Configuration | Score (1-5) | Accuracy (%) |
|---|---|---|---|
| SlideVQA test (2,215) | Role-Specialized Sequential Optimization only | 3.986 | 70.1 |
| | full RRG-RAG (`full`) | **4.084** | **72.8** |
| ViDoSeek (1,142) | Role-Specialized Sequential Optimization only | 4.099 | 73.1 |
| | full RRG-RAG (`full`) | **4.178** | **75.5** |
| MMLongBench-Doc (1,082) | Role-Specialized Sequential Optimization only (`tfr`) | 2.375 | 26.9 |
| | full RRG-RAG (`full`) | **2.421** | **28.0** |

Accuracy is the share of questions with judge score ≥ 4 (`passing` in `*_eval.json`). LLM-judge scores can shift slightly across judge API versions. See the paper for the full tables and ablations.

---

## Acknowledgements

This code builds on [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG), LLaMA-Factory (Apache-2.0, see `LLaMA-Factory/LICENSE`). The datasets keep their original licenses: SlideVQA, ViDoSeek and MMLongBench-Doc.
