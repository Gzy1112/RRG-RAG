#!/usr/bin/env python3
"""Simplify Synthesizer SFT v6 data: output = plain answer only (no JSON).

Matches the v4 approach — the model outputs just the answer string directly.
This avoids JSON parsing failures during eval (only 13/300 were valid JSON in v6).

Input:  training_data_v6/synthesizer_sft_v6_train.json (JSON output)
Output: training_data_v7/synthesizer_sft_v7_train.json (plain answer output)

Changes:
  - output: "2Bn"  (instead of {"reason":..., "answer":"2Bn", "reference":[0]})
  - system: Remove JSON format requirement
  - instruction: Remove "provide the accurate answer" -> "provide the final answer"
"""
import json
import os
import shutil

SLIDEVQA_ROOT = os.path.join(
    os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "SlideVQA")
SRC_DIR = os.path.join(SLIDEVQA_ROOT, "training_data_v6")
DST_DIR = os.path.join(SLIDEVQA_ROOT, "training_data_v7")

SIMPLE_SYSTEM = (
    "You are an AI assistant with strong ability to answer questions through images. "
    "Given a question and reference page images, provide the final answer."
)


def simplify_output(output_json_str: str) -> str:
    """Extract the answer from JSON output, return plain string."""
    try:
        parsed = json.loads(output_json_str)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            if answer:
                return str(answer).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: return as-is
    return output_json_str.strip()


def simplify_instruction(instruction: str) -> str:
    """Update instruction wording — no more JSON hints."""
    # Replace the ending prompt
    instruction = instruction.replace(
        "provide the accurate answer.",
        "provide the final answer.",
    )
    return instruction


def convert_file(src_path: str, dst_path: str):
    """Convert one data file from v6 JSON format to v7 plain format."""
    with open(src_path) as f:
        data = json.load(f)

    converted = []
    for sample in data:
        new_sample = dict(sample)
        new_sample["output"] = simplify_output(sample["output"])
        new_sample["instruction"] = simplify_instruction(sample["instruction"])
        new_sample["system"] = SIMPLE_SYSTEM
        converted.append(new_sample)

    with open(dst_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted: {src_path} -> {dst_path} ({len(converted)} samples)")
    return converted


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # Convert train (SFT)
    train = convert_file(
        os.path.join(SRC_DIR, "synthesizer_sft_v6_train.json"),
        os.path.join(DST_DIR, "synthesizer_sft_v7_train.json"),
    )

    # Convert train (PPO — same data, gold answer used for reward)
    shutil.copy(
        os.path.join(DST_DIR, "synthesizer_sft_v7_train.json"),
        os.path.join(DST_DIR, "synthesizer_ppo_v7_train.json"),
    )
    print(f"Copied PPO train: synthesizer_ppo_v7_train.json ({len(train)} samples)")

    # Convert val
    convert_file(
        os.path.join(SRC_DIR, "synthesizer_sft_v6_val.json"),
        os.path.join(DST_DIR, "synthesizer_sft_v7_val.json"),
    )

    # Verify
    print("\n=== Verification ===")
    print(f"Sample 0 output: {repr(train[0]['output'][:80])}")
    print(f"Sample 1 output: {repr(train[1]['output'][:80])}")
    print(f"Sample 2 output: {repr(train[2]['output'][:80])}")

    # Check diversity of outputs
    outputs = set(s["output"] for s in train[:1000])
    print(f"\nUnique outputs (first 1000): {len(outputs)}")


if __name__ == "__main__":
    main()
