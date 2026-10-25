#!/bin/bash
# TFR: alternating per-agent PPO of the Synthesizer and the Seeker (2 cycles).
#
#   round A  Synthesizer PPO, reward synth_fix5_v11              (1 GPU,  src/train_restore.py)
#   round B  Seeker PPO, reward e2e = 0.4*evidence + 0.6*F1 of the  (2 GPUs, src/train_restore_7b.py:
#            round-A Synthesizer kept frozen                         policy on GPU 0, frozen
#                                                                    Synthesizer on GPU 1)
# Each round warm-starts from the previous adapter of the same agent (cycle 1: SFT).
# Final adapters: $TFR_ROOT/cycle2/round_b_seeker and $TFR_ROOT/cycle2/round_a_synth.
#
# Usage: scripts/train_tfr.sh [gpu_a=0] [gpus_b=0,1]
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU_SYNTH="${1:-0}"
GPU_SEEKER="${2:-0,1}"
SYNTH_DEVICE=cuda:1          # frozen Synthesizer inside round B (second visible GPU)

CYCLES=2
STEPS_SYNTH=30
STEPS_SEEKER=30
LR_SYNTH=1.0e-6
LR_SEEKER=5.0e-6
TPL_SYNTH="$LLAMA_FACTORY/examples/train_lora/qwen2_5_vl_7b_tfr_synth.yaml"
TPL_SEEKER="$LLAMA_FACTORY/examples/train_lora/qwen2_5_vl_7b_tfr_seeker.yaml"

need_adapter "$SYNTH_SFT" "scripts/train_sft.sh synthesizer"
need_adapter "$SEEKER_SFT" "scripts/train_sft.sh seeker"
# Gold answers for the e2e reward (question -> answer).
export VIDORAG_GOLD_ANSWERS="${VIDORAG_GOLD_ANSWERS:-$DATA/SlideVQA/vidorag_gold_answers.json}"
need_file "$VIDORAG_GOLD_ANSWERS" "python data_prep/build_gold_answers.py"
mkdir -p "$TFR_ROOT"

resolve_adapter() {   # final adapter in <dir>, else its latest checkpoint
    local dir="$1" latest
    [ -f "$dir/adapter_config.json" ] && { echo "$dir"; return 0; }
    latest="$(find "$dir" -maxdepth 1 -type d -name 'checkpoint-*' | sed 's#.*/checkpoint-##' \
              | grep -E '^[0-9]+$' | sort -n | tail -1)"
    [ -n "$latest" ] && [ -f "$dir/checkpoint-$latest/adapter_config.json" ] \
        && { echo "$dir/checkpoint-$latest"; return 0; }
    return 1
}

run_round() {   # run_round <synth|seeker> <label> <cycle> <warm_adapter> <steps> <lr> <reward> <gpus> <entry> [frozen_synth]
    local agent="$1" label="$2" cycle="$3" warm="$4" steps="$5" lr="$6" reward="$7"
    local gpus="$8" entry="$9" e2e_synth="${10:-}"
    local out_dir="$TFR_ROOT/cycle$cycle/$label" tpl
    local yaml="$TFR_ROOT/cycle$cycle/$label.yaml" log="$TFR_ROOT/cycle$cycle/${label}_training.log"
    [ "$agent" = synth ] && tpl="$TPL_SYNTH" || tpl="$TPL_SEEKER"
    mkdir -p "$out_dir"
    sed -e "s#^model_name_or_path:.*#model_name_or_path: $BASE_MODEL#" \
        -e "s#^adapter_name_or_path:.*#adapter_name_or_path: $warm#" \
        -e "s#^output_dir:.*#output_dir: $out_dir#" \
        -e "s#^max_steps:.*#max_steps: $steps#" \
        -e "s#^save_steps:.*#save_steps: $(( steps / 2 ))#" \
        -e "s#^learning_rate:.*#learning_rate: $lr#" "$tpl" > "$yaml"
    echo "[tfr] cycle $cycle $label: warm=$warm reward=$reward gpus=$gpus" >&2
    (
        cd "$LLAMA_FACTORY"
        export CUDA_VISIBLE_DEVICES="$gpus"
        export VIDORAG_REWARD_METHOD="$reward"
        unset VIDORAG_MAPPO_TRAINER VIDORAG_MAPPO_INDEPENDENT MAPPO_USE_GRPO
        if [ -n "$e2e_synth" ]; then
            export VIDORAG_E2E_SYNTH_ADAPTER="$e2e_synth"
            export VIDORAG_E2E_W_EVIDENCE=0.4 VIDORAG_E2E_W_ANSWER=0.6
            export VIDORAG_E2E_SYNTH_DEVICE="$SYNTH_DEVICE"
            export VIDORAG_SKIP_DISTRIBUTED_CHECK=1
        fi
        exec python "$entry" "$yaml"
    ) > "$log" 2>&1
    grep -q "Detected ViDoRAG per-agent PPO mode for" "$log" || die "$label did not enter per-agent PPO mode ($log)"
    grep -q "Reward method = $reward" "$log" || die "$label ran with the wrong reward ($log)"
    resolve_adapter "$out_dir" || die "$label produced no adapter ($log)"
}

warm_synth="$SYNTH_SFT"
warm_seeker="$SEEKER_SFT"
for cycle in $(seq 1 "$CYCLES"); do
    out_synth=$(run_round synth round_a_synth "$cycle" "$warm_synth" "$STEPS_SYNTH" "$LR_SYNTH" \
        synth_fix5_v11 "$GPU_SYNTH" src/train_restore.py)
    out_seeker=$(run_round seeker round_b_seeker "$cycle" "$warm_seeker" "$STEPS_SEEKER" "$LR_SEEKER" \
        e2e "$GPU_SEEKER" src/train_restore_7b.py "$out_synth")
    warm_synth="$out_synth"
    warm_seeker="$out_seeker"
    echo "[tfr] cycle $cycle done: synthesizer=$out_synth seeker=$out_seeker"
done
echo "[tfr] final Seeker=$warm_seeker Synthesizer=$warm_synth"
