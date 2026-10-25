#!/usr/bin/env python3
"""Generate the ColQwen2.5 page index for one dataset.

Encodes every page image in data/<dataset>/img/ using ColQwen2.5
(base + LoRA adapter with correct key remapping) and saves to .pt file.

Output: data/<dataset>/colqwen25_ingestion_v3.pt
    {page_ids: [str, ...], page_embs: [Tensor[seq_len, 128] bf16, ...]}

Usage:
    CUDA_VISIBLE_DEVICES=0 python data_prep/ingest_colqwen25.py --dataset SlideVQA
    CUDA_VISIBLE_DEVICES=0 python data_prep/ingest_colqwen25.py --dataset ViDoSeek
    CUDA_VISIBLE_DEVICES=0 python data_prep/ingest_colqwen25.py --dataset MMLongBench
"""
import argparse, json, os, sys, glob, time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Local snapshot of vidore/colqwen2.5-base (its safetensors index is read below).
BASE_MODEL_PATH = os.environ.get(
    'VIDORAG_COLQWEN25_BASE', os.path.join(PROJECT_ROOT, 'models/colqwen2.5-base'))
# vidore/colqwen2.5-v0.2 with keys renamed by data_prep/fix_colqwen25_adapter.py.
ADAPTER_PATH = os.environ.get(
    'VIDORAG_COLQWEN25_ADAPTER', os.path.join(PROJECT_ROOT, 'models/colqwen2.5-v0.2-fixed'))
DATA_ROOT = os.environ.get('VIDORAG_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))
IMG_DIR = None     # set from --dataset in main()
OUTPUT_PT = None


def load_model():
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from peft import PeftModel
    from safetensors.torch import load_file

    print(f"[1/3] Loading base model: {BASE_MODEL_PATH}")
    model = ColQwen2_5.from_pretrained(BASE_MODEL_PATH, torch_dtype='auto', device_map='cpu')

    # CRITICAL: Remap checkpoint keys: model.* -> language_model.*
    # Without this, the model has random weights and retrieval fails completely
    idx_path = os.path.join(BASE_MODEL_PATH, 'model.safetensors.index.json')
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    files = {}
    for k, v in weight_map.items():
        files.setdefault(v, []).append(k)
    state_dict = OrderedDict()
    for fname, keys in files.items():
        fpath = os.path.join(BASE_MODEL_PATH, fname)
        sd = load_file(fpath)
        for k in keys:
            new_k = k.replace('model.', 'language_model.', 1) if k.startswith('model.') else k
            state_dict[new_k] = sd[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Base remap: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0 or len(unexpected) > 0:
        print("  WARNING: Key mismatch detected!")
    del state_dict

    model = model.to('cuda:0', dtype=torch.bfloat16)

    print(f"[2/3] Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    processor = ColQwen2_5_Processor.from_pretrained(BASE_MODEL_PATH)
    return model, processor


def ingest(model, processor):
    images = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    print(f"Found {len(images)} page images in {IMG_DIR}")

    page_ids = []
    page_embs = []
    batch_size = 4
    bad = 0

    for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
        batch_paths = images[i:i+batch_size]
        batch_images = []
        batch_ids = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_images.append(img)
                # page_id = filename without extension
                batch_ids.append(os.path.splitext(os.path.basename(p))[0])
            except Exception as e:
                bad += 1
                continue

        if not batch_images:
            continue

        with torch.no_grad():
            inputs = processor.process_images(batch_images).to(model.device)
            embs = model(**inputs)

        for pid, emb in zip(batch_ids, embs):
            page_ids.append(pid)
            page_embs.append(emb.cpu().to(torch.bfloat16))

    print(f"Encoded {len(page_ids)} pages ({bad} bad)")
    return page_ids, page_embs


def main():
    global IMG_DIR, OUTPUT_PT
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='SlideVQA | ViDoSeek | MMLongBench')
    args = ap.parse_args()
    IMG_DIR = os.path.join(DATA_ROOT, args.dataset, 'img')
    OUTPUT_PT = os.path.join(DATA_ROOT, args.dataset, 'colqwen25_ingestion_v3.pt')

    # Remove old broken ingestion
    if os.path.exists(OUTPUT_PT):
        print(f"Removing old ingestion: {OUTPUT_PT}")
        os.remove(OUTPUT_PT)

    t0 = time.time()
    model, processor = load_model()

    print(f"[3/3] Encoding {args.dataset} pages...")
    page_ids, page_embs = ingest(model, processor)

    avg_seq = sum(e.shape[0] for e in page_embs[:100]) / min(100, len(page_embs))
    print(f"  avg seq_len={avg_seq:.0f}, dim={page_embs[0].shape[1]}")

    print(f"Saving to {OUTPUT_PT}...")
    torch.save({'page_ids': page_ids, 'page_embs': page_embs}, OUTPUT_PT)

    size_gb = os.path.getsize(OUTPUT_PT) / (1024**3)
    elapsed = time.time() - t0
    print(f"Done! {size_gb:.2f} GB in {elapsed:.0f}s")
    print(f"  {len(page_ids)} pages, format: {{page_ids, page_embs}}")


if __name__ == '__main__':
    main()
