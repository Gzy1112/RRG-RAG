#!/bin/bash
# Inspector GRPO: fresh LoRA controller choosing STOP / DECK after the Seeker's
# first selection, trained on the cached val-split branch outcomes (train decks).
# Checkpoints: $INSPECTOR_ROOT/<run_name>/checkpoint-{20,40,...,120}/controller
#
# Usage: scripts/train_inspector_grpo.sh <gpu> [run_name=run1]
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU="${1:?usage: $0 <gpu> [run_name]}"
RUN_NAME="${2:-run1}"
OUT_DIR="$INSPECTOR_ROOT/$RUN_NAME"
BASE_CONFIG="$LLAMA_FACTORY/examples/train_lora/qwen2_5_vl_7b_inspector_grpo_v2.yaml"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export VIDORAG_MAPPO_TRAINER=inspector_grpo_v2
export MAPPO_USE_GRPO=1
export MAPPO_GRPO_GROUP_SIZE=8
export GRPO_KL_BETA=0.05
export MAPPO_GRPO_STD_FLOOR=0.25
export MAPPO_GRPO_ADV_CLIP=5.0
export MAPPO_WHITEN_PER_AGENT=0

export INSPECTOR_V2_BRANCH_CACHE="${INSPECTOR_V2_BRANCH_CACHE:-$INSPECTOR_ROOT/data/branch_cache_val_shard0.jsonl,$INSPECTOR_ROOT/data/branch_cache_val_shard1.jsonl}"
export INSPECTOR_V2_RANK_CACHE="${INSPECTOR_V2_RANK_CACHE:-$RANK_CACHE_DIR/SlideVQA_val.jsonl}"
export INSPECTOR_V2_ACTIONS=STOP,DECK
export INSPECTOR_V2_COST=0
export INSPECTOR_V2_INFORMATIVE_WEIGHT=32
export INSPECTOR_V2_SPLIT=train
export CONTROLLER_GRPO_FORMAT_PENALTY=-0.5
export CONTROLLER_GRPO_TEMPERATURE=1.0
export CONTROLLER_GRPO_TOP_P=1.0
export CONTROLLER_GRPO_MAX_NEW_TOKENS=24
export CONTROLLER_GRPO_BATCH=8
export CONTROLLER_GRPO_MAX_STEPS=120
export CONTROLLER_GRPO_LOG_EVERY=10
export CONTROLLER_GRPO_SAVE_EVERY=20
export CONTROLLER_GRPO_LORA_RANK=16
export CONTROLLER_GRPO_LORA_ALPHA=32
export VIDORAG_CONTROLLER_ADAPTER=          # empty = fresh LoRA on the base model
export VIDORAG_CONTROLLER_KL_REF_ADAPTER=

IFS=',' read -ra CACHES <<< "$INSPECTOR_V2_BRANCH_CACHE"
for f in "${CACHES[@]}" "$INSPECTOR_V2_RANK_CACHE"; do
    need_file "$f" "scripts/build_inspector_branch_cache.sh / scripts/build_rank_cache.sh"
done
# The placeholder dataset only has to load (see the YAML header).
need_file "$DATA/SlideVQA/training_data_v4/seeker_sft_v4_train.json"

CONFIG="$OUT_DIR/train_config.yaml"
sed -e "s#^output_dir:.*#output_dir: $OUT_DIR#" \
    -e "s#^model_name_or_path:.*#model_name_or_path: $BASE_MODEL#" "$BASE_CONFIG" > "$CONFIG"

echo "[inspector-grpo] gpu=$GPU out=$OUT_DIR actions=$INSPECTOR_V2_ACTIONS steps=$CONTROLLER_GRPO_MAX_STEPS"
cd "$LLAMA_FACTORY"
python -u src/train_restore_grpo.py "$CONFIG" 2>&1 | tee "$OUT_DIR/training.log"
