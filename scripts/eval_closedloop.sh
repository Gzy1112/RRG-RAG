#!/bin/bash
# Closed-loop evaluation of the paper configurations on one benchmark.
#
#   full    TFR agents + rank-1 rescue + Inspector GRPO (STOP/DECK, val-selected
#           checkpoint and bias from $SELECTED_POLICY); DECK = same-deck second hop
#           (K=5, any |E0|>=1, union)                                   [default]
#   rule    TFR + rank-1 rescue + fixed rule: DECK iff the Seeker kept >= 2 pages
#   tfr_r1  TFR + rank-1 rescue
#   tfr     TFR agents, no recovery
#
# Needs a retrieval server for the dataset and its rank cache (scripts/build_rank_cache.sh).
# Output: outputs/eval/<dataset>/<ds>_<mode>_hybrid_<ds>[_shardN].jsonl (+ _eval.json:
# 'score' = mean judge score 1-5, 'passing' = share of score >= 4).
#
# Usage: scripts/eval_closedloop.sh <gpu> <SlideVQA|ViDoSeek|MMLongBench> [variant=full] [port=8000] [shard_id num_shards]
#   Shards write separate files; merge them with scripts/merge_eval_shards.py.
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU="${1:?usage: $0 <gpu> <dataset> [variant] [port] [shard_id num_shards]}"
DS="${2:?dataset}"
VARIANT="${3:-full}"
PORT="${4:-8000}"
SHARD_ARGS=()
[ $# -ge 6 ] && SHARD_ARGS=(--shard_id "$5" --num_shards "$6")
MAX_SAMPLES="${MAX_SAMPLES:-100000}"

need_judge
need_server "$PORT" "$DS"
closed_loop_env
need_adapter "$EVAL_SEEKER_PATH" "scripts/train_tfr.sh"
need_adapter "$EVAL_SYNTH_PATH" "scripts/train_tfr.sh"

export CUDA_VISIBLE_DEVICES="$GPU"
export EVAL_DATASET="$DS"
export EVAL_RETRIEVAL_SERVER="http://localhost:$PORT"
if [ "$DS" = SlideVQA ]; then
    export EVAL_QUERY_FILE=rag_dataset_test.json
    export RANK_CACHE_PATH="${RANK_CACHE_PATH:-$RANK_CACHE_DIR/SlideVQA_test.jsonl}"
else
    export EVAL_QUERY_FILE=rag_dataset.json
    export RANK_CACHE_PATH="${RANK_CACHE_PATH:-$RANK_CACHE_DIR/$DS.jsonl}"
fi
need_file "$RANK_CACHE_PATH" "scripts/build_rank_cache.sh"
# Page images are capped at 2048 visual tokens (1.6 MP). Every SlideVQA page is
# below the cap; native ViDoSeek / MMLongBench pages (up to 10 MP) are downscaled.
export VIDORAG_IMAGE_MAX_PIXELS="${VIDORAG_IMAGE_MAX_PIXELS:-1605632}"

case "$VARIANT" in
    full)
        need_file "$SELECTED_POLICY" "scripts/select_inspector_policy.sh"
        readarray -t CFG < <(python - "$SELECTED_POLICY" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x['adapter']); print(x['actions']); print(x['biases'])
PY
)
        export CONTROLLER_MODE=inspector_v2
        export INSPECTOR_V2_ADAPTER="${CFG[0]}"
        export INSPECTOR_V2_ACTIONS="${CFG[1]}"
        export INSPECTOR_V2_ACTION_BIASES="${CFG[2]}"
        export HOP2_MODE=deck HOP2_K=5 HOP2_MIN_SEL=1 HOP2_FIRE=union
        export RANK1_RESCUE=append ;;
    rule)
        export CONTROLLER_MODE=always_stop
        export HOP2_MODE=deck HOP2_K=5 HOP2_MIN_SEL=2 HOP2_FIRE=union
        export RANK1_RESCUE=append ;;
    tfr_r1)
        export CONTROLLER_MODE=always_stop HOP2_MODE=off RANK1_RESCUE=append ;;
    tfr)
        export CONTROLLER_MODE=always_stop HOP2_MODE=off RANK1_RESCUE=off ;;
    *) die "unknown variant $VARIANT (full | rule | tfr_r1 | tfr)" ;;
esac
export EVAL_MODE_NAME="${EVAL_MODE_NAME:-rrg_$VARIANT}"

cd "$P"
echo "[eval] dataset=$DS variant=$VARIANT server=$EVAL_RETRIEVAL_SERVER rank_cache=$RANK_CACHE_PATH"
echo "[eval] seeker=$EVAL_SEEKER_PATH synthesizer=$EVAL_SYNTH_PATH controller=$CONTROLLER_MODE"
python -u run_closedloop_eval.py --mode "$EVAL_MODE_NAME" --max_samples "$MAX_SAMPLES" ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"}
