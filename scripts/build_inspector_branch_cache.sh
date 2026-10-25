#!/bin/bash
# Counterfactual branch outcomes for Inspector GRPO on the SlideVQA val split:
# every val question runs through the deployed closed-loop pipeline (TFR agents)
# once per Inspector action and each branch answer is judged.
# Needs the SlideVQA retrieval server and outputs/rank_cache/SlideVQA_val.jsonl.
#
# Usage: scripts/build_inspector_branch_cache.sh <gpu> <shard_id> <num_shards> [port=8000]
#   e.g. two shards on two GPUs:
#     scripts/build_inspector_branch_cache.sh 0 0 2 &
#     scripts/build_inspector_branch_cache.sh 1 1 2 &
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU="${1:?usage: $0 <gpu> <shard_id> <num_shards> [port]}"
SHARD="${2:?shard_id}"
NSH="${3:?num_shards}"
PORT="${4:-8000}"
need_judge
need_server "$PORT" SlideVQA
closed_loop_env
need_adapter "$EVAL_SEEKER_PATH" "scripts/train_tfr.sh"
need_adapter "$EVAL_SYNTH_PATH" "scripts/train_tfr.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
export EVAL_DATASET=SlideVQA
export EVAL_QUERY_FILE=rag_dataset_val.json
export EVAL_RETRIEVAL_SERVER="http://localhost:$PORT"
export RANK_CACHE_PATH="${RANK_CACHE_PATH_VAL:-$RANK_CACHE_DIR/SlideVQA_val.jsonl}"
need_file "$RANK_CACHE_PATH" "scripts/build_rank_cache.sh SlideVQA rag_dataset_val.json ..."
OUT_FILE="$INSPECTOR_ROOT/data/branch_cache_val_shard$SHARD.jsonl"
mkdir -p "$(dirname "$OUT_FILE")"

cd "$P"
echo "[branch-cache] gpu=$GPU shard=$SHARD/$NSH out=$OUT_FILE"
python -u build_inspector_branch_cache.py --out "$OUT_FILE" --shard_id "$SHARD" --num_shards "$NSH"
