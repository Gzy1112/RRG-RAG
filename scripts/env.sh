#!/bin/bash
# Shared environment for every script in scripts/. Sourced, not executed.
# Any variable below can be overridden by exporting it before running a script.
# Activate your Python environment (see README) before calling the scripts.

export VIDORAG_PROJECT_ROOT="${VIDORAG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
P="$VIDORAG_PROJECT_ROOT"
DATA="${VIDORAG_DATA_ROOT:-$P/data}"
OUT="${RRG_OUTPUT_ROOT:-$P/outputs}"
LLAMA_FACTORY="$P/LLaMA-Factory"

# ---- models (local snapshots; trained adapters record BASE_MODEL as their base) ----
export BASE_MODEL="${BASE_MODEL:-$P/models/Qwen2.5-VL-7B-Instruct}"
export VIDORAG_BASE_MODEL="$BASE_MODEL"
export VIDORAG_COLQWEN25_BASE="${VIDORAG_COLQWEN25_BASE:-$P/models/colqwen2.5-base}"
export VIDORAG_COLQWEN25_ADAPTER="${VIDORAG_COLQWEN25_ADAPTER:-$P/models/colqwen2.5-v0.2-fixed}"

# ---- adapters written by the training scripts ----
SEEKER_SFT="${SEEKER_SFT:-$OUT/SEEKER_SFT/seeker_sft_v4_7b}"
SYNTH_SFT="${SYNTH_SFT:-$OUT/SYNTHESIZER_SFT/synthesizer_sft_v7_7b}"
TFR_ROOT="${TFR_ROOT:-$OUT/TFR_7B}"
SEEKER_TFR="${SEEKER_TFR:-$TFR_ROOT/cycle2/round_b_seeker}"
SYNTH_TFR="${SYNTH_TFR:-$TFR_ROOT/cycle2/round_a_synth}"
INSPECTOR_ROOT="${INSPECTOR_ROOT:-$OUT/INSPECTOR_GRPO}"
SELECTED_POLICY="${SELECTED_POLICY:-$INSPECTOR_ROOT/selected_policy.json}"
RANK_CACHE_DIR="${RANK_CACHE_DIR:-$OUT/rank_cache}"

# ---- LLM judge (DashScope, OpenAI-compatible endpoint) ----
export JUDGE_MODEL="${JUDGE_MODEL:-qwen3-vl-plus}"
export VIDORAG_JUDGE_MAX_ATTEMPTS="${VIDORAG_JUDGE_MAX_ATTEMPTS:-40}"
export VIDORAG_JUDGE_RETRY_CAP="${VIDORAG_JUDGE_RETRY_CAP:-600}"

# ---- runtime ----
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$LLAMA_FACTORY/src:$P${PYTHONPATH:+:$PYTHONPATH}"

die() { echo "FATAL: $*" >&2; exit 1; }
need_file() { [ -s "$1" ] || die "missing $1${2:+ ($2)}"; }
need_adapter() { [ -s "$1/adapter_config.json" ] || die "missing adapter $1${2:+ ($2)}"; }
need_judge() { [ -n "${DASHSCOPE_API_KEY:-}" ] || die "DASHSCOPE_API_KEY is not set (LLM judge)"; }
need_server() {   # need_server <port> <dataset>
    local health
    health=$(curl -s -m 10 "http://localhost:$1/health")
    echo "$health" | grep -q "\"engine_loaded\":true" && echo "$health" | grep -q "\"dataset\":\"$2\"" \
        || die "no loaded $2 retrieval server on port $1 (scripts/start_retrieval_server.sh): $health"
}

# Closed-loop pipeline settings shared by evaluation and the Inspector branch cache.
closed_loop_env() {
    export EVAL_BASE_MODEL="$BASE_MODEL"
    export EVAL_SEEKER_PATH="${EVAL_SEEKER_PATH:-$SEEKER_TFR}"
    export EVAL_SYNTH_PATH="${EVAL_SYNTH_PATH:-$SYNTH_TFR}"
    export EVAL_NO_CLEAN=1                    # resume partial runs instead of deleting them
    export DISABLE_INSPECTOR_SHORTCIRCUIT=1
    export SKIP_PASSTHROUGH_INSPECTOR=1       # recovery decisions come from the controller only
    export EVAL_SKIP_MAIN_VLM=1
    export RETRIEVE_MODE=merged
    export CLOSED_LOOP_FIRE_MODE=union
    export CLOSED_LOOP_RESELECT_POOL=20
}
