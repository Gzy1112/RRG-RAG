#!/bin/bash
# Checkpoint + action-bias selection for the Inspector on the val 'holdout' decks
# (no test data is read). Scores every checkpoint offline against the cached
# branch outcomes, then writes $INSPECTOR_ROOT/selected_policy.json.
#
# Usage: scripts/select_inspector_policy.sh <gpu> [run_name=run1]
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU="${1:?usage: $0 <gpu> [run_name]}"
RUN_DIR="$INSPECTOR_ROOT/${2:-run1}"
BRANCH_CACHE="${INSPECTOR_V2_BRANCH_CACHE:-$INSPECTOR_ROOT/data/branch_cache_val_shard0.jsonl,$INSPECTOR_ROOT/data/branch_cache_val_shard1.jsonl}"
DEC="$RUN_DIR/holdout_decisions.jsonl"

ADAPTERS=""
for S in 20 40 60 80 100 120; do
    A="$RUN_DIR/checkpoint-$S/controller"
    need_adapter "$A" "scripts/train_inspector_grpo.sh"
    ADAPTERS="${ADAPTERS}${ADAPTERS:+,}$A"
done
[ -e "$DEC" ] && die "stale decision file $DEC (remove it to re-run)"

cd "$P"
CUDA_VISIBLE_DEVICES="$GPU" python eval_inspector_v2_offline.py \
    --branch_cache "$BRANCH_CACHE" --rank_cache "$RANK_CACHE_DIR/SlideVQA_val.jsonl" \
    --actions STOP,DECK --adapters "$ADAPTERS" --base --base_model "$BASE_MODEL" \
    --out "$DEC" | tee "$RUN_DIR/holdout_offline.txt"
python select_inspector_v2_policy.py "$DEC" --out "$SELECTED_POLICY" | tee "$RUN_DIR/selection.txt"
cat "$SELECTED_POLICY"
