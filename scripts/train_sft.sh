#!/bin/bash
# SFT warm start of the Seeker / Synthesizer LoRA adapters (single GPU).
# Data: data/SlideVQA/training_data_v4 (Seeker) and training_data_v7 (Synthesizer),
# built by data_prep/ (see README).
#
# Usage: scripts/train_sft.sh <seeker|synthesizer> [gpu=0]
set -euo pipefail
source "$(dirname "$0")/env.sh"
AGENT="${1:?usage: $0 <seeker|synthesizer> [gpu]}"
export CUDA_VISIBLE_DEVICES="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
case "$AGENT" in
    seeker)      TPL=qwen2_5_vl_7b_seeker_sft_slidevqa_v4.yaml; OUT_DIR="$SEEKER_SFT"
                 need_file "$DATA/SlideVQA/training_data_v4/seeker_sft_v4_train.json" ;;
    synthesizer) TPL=qwen2_5_vl_7b_synthesizer_sft_v7.yaml;   OUT_DIR="$SYNTH_SFT"
                 need_file "$DATA/SlideVQA/training_data_v7/synthesizer_sft_v7_train.json" ;;
    *) die "agent must be seeker or synthesizer" ;;
esac
mkdir -p "$OUT_DIR"
CONFIG="$OUT_DIR/train_config.yaml"
sed -e "s#^model_name_or_path:.*#model_name_or_path: $BASE_MODEL#" \
    -e "s#^output_dir:.*#output_dir: $OUT_DIR#" \
    "$LLAMA_FACTORY/examples/train_lora/$TPL" > "$CONFIG"

echo "[sft] $AGENT gpu=$CUDA_VISIBLE_DEVICES base=$BASE_MODEL out=$OUT_DIR"
cd "$LLAMA_FACTORY"
python src/train.py "$CONFIG" 2>&1 | tee "$OUT_DIR/training.log"
