#!/usr/bin/env python3
"""Make the public ColQwen2.5 LoRA adapter loadable on top of the remapped base.

search_engine.ColQwen25Engine (and data_prep/ingest_colqwen25.py) load
vidore/colqwen2.5-base with its decoder weights remapped `model.*` ->
`language_model.*` (required by recent transformers). The public adapter
vidore/colqwen2.5-v0.2 still addresses the decoder as `model.*`, so its LoRA
weights would silently not attach. This script renames the adapter keys

    base_model.model.model.*  ->  base_model.model.language_model.*

(tensors are copied unchanged) and writes an adapter_config.json restricted to
the fields understood by peft 0.12.

Usage:
    python data_prep/fix_colqwen25_adapter.py \
        --src models/colqwen2.5-v0.2 --dst models/colqwen2.5-v0.2-fixed
"""
import argparse
import json
import os

from safetensors.torch import load_file, save_file

CONFIG_KEYS = [
    'auto_mapping', 'base_model_name_or_path', 'bias', 'inference_mode',
    'init_lora_weights', 'layers_to_transform', 'lora_alpha', 'lora_dropout',
    'modules_to_save', 'peft_type', 'r', 'revision', 'target_modules', 'task_type',
]
OLD_PREFIX = 'base_model.model.model.'
NEW_PREFIX = 'base_model.model.language_model.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='local snapshot of vidore/colqwen2.5-v0.2')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    sd = load_file(os.path.join(args.src, 'adapter_model.safetensors'))
    renamed = {(NEW_PREFIX + k[len(OLD_PREFIX):] if k.startswith(OLD_PREFIX) else k): v
               for k, v in sd.items()}
    n_renamed = sum(k.startswith(OLD_PREFIX) for k in sd)
    assert len(renamed) == len(sd)

    with open(os.path.join(args.src, 'adapter_config.json')) as f:
        cfg = json.load(f)
    cfg = {k: cfg.get(k) for k in CONFIG_KEYS}

    os.makedirs(args.dst, exist_ok=True)
    save_file(renamed, os.path.join(args.dst, 'adapter_model.safetensors'), metadata={'format': 'pt'})
    with open(os.path.join(args.dst, 'adapter_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'{len(sd)} tensors, {n_renamed} renamed -> {args.dst}')


if __name__ == '__main__':
    main()
