#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel OCR for SlideVQA using PP-OCRv3 (CPU-based, multi-process).

Usage:
    # Run 4 parallel OCR workers
    python data_prep/ocr_slidevqa.py --workers 4

    # Resume (skips already-processed files automatically)
    python data_prep/ocr_slidevqa.py --workers 4 --resume
"""

import os
import sys
import argparse
import subprocess
import time
import multiprocessing
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("VIDORAG_PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDEVQA_DIR = os.path.join(PROJECT_ROOT, "data", "SlideVQA")
IMG_DIR = os.path.join(SLIDEVQA_DIR, "img")
OUTPUT_DIR = os.path.join(SLIDEVQA_DIR, "ppocr")

# PaddleOCR inference models + English label dict (PPOCR_MODEL_ROOT, default models/ppocr)
PPOCR_ROOT = os.environ.get("PPOCR_MODEL_ROOT", os.path.join(PROJECT_ROOT, "models", "ppocr"))
DET_MODEL_DIR = os.path.join(PPOCR_ROOT, "en_PP-OCRv3_det_infer")
REC_MODEL_DIR = os.path.join(PPOCR_ROOT, "en_PP-OCRv3_rec_infer")
CLS_MODEL_DIR = os.path.join(PPOCR_ROOT, "ch_ppocr_mobile_v2.0_cls_infer")
REC_LABEL_FILE = os.path.join(PPOCR_ROOT, "en_dict.txt")


# ════════════════════════════════════════════════════════════════════════
#  WORKER: runs OCR on a shard of images (single CPU process)
# ════════════════════════════════════════════════════════════════════════
def worker_main(args):
    """Process a shard of images with PP-OCRv3."""
    import cv2
    import numpy as np
    import fastdeploy as fd
    from PIL import Image
    from tqdm import tqdm

    # Read shard file list
    with open(args.shard_file) as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"[Worker {args.worker_id}] {len(files)} files to process", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Initialize PP-OCRv3 models ─────────────────────────────────────
    det_model_file = os.path.join(DET_MODEL_DIR, "inference.pdmodel")
    det_params_file = os.path.join(DET_MODEL_DIR, "inference.pdiparams")
    cls_model_file = os.path.join(CLS_MODEL_DIR, "inference.pdmodel")
    cls_params_file = os.path.join(CLS_MODEL_DIR, "inference.pdiparams")
    rec_model_file = os.path.join(REC_MODEL_DIR, "inference.pdmodel")
    rec_params_file = os.path.join(REC_MODEL_DIR, "inference.pdiparams")

    det_option = fd.RuntimeOption()
    det_option.use_cpu()
    cls_option = fd.RuntimeOption()
    cls_option.use_cpu()
    rec_option = fd.RuntimeOption()
    rec_option.use_cpu()

    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_option
    )
    cls_model = fd.vision.ocr.Classifier(
        cls_model_file, cls_params_file, runtime_option=cls_option
    )
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file, rec_params_file, REC_LABEL_FILE, runtime_option=rec_option
    )

    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False
    cls_model.postprocessor.cls_thresh = 0.9

    ppocr_v3 = fd.vision.ocr.PPOCRv3(
        det_model=det_model, cls_model=cls_model, rec_model=rec_model
    )

    MIN_SCORE = 0.6
    SPACE_THRESHOLD = 45
    LINE_THRESHOLD = 15

    def decode_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def tostr_layout_preserving(result):
        text_boxes = []
        for box, text, score in zip(
            result.boxes, result.text, result.rec_scores
        ):
            if score >= MIN_SCORE:
                coords = [
                    (box[i], box[i + 1]) for i in range(0, len(box), 2)
                ]
                center_x = (coords[0][0] + coords[2][0]) / 2
                center_y = (coords[0][1] + coords[2][1]) / 2
                text_boxes.append((center_x, center_y, text, coords))

        text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))

        merged_text = []
        previous_box = None
        for box in text_boxes:
            if previous_box is not None:
                dx = abs(box[0] - previous_box[0])
                dy = abs(box[1] - previous_box[1])
                if dy < LINE_THRESHOLD:
                    spaces = max(1, int(dx / SPACE_THRESHOLD))
                    merged_text.append("\n" * 0 + " " * spaces)
                else:
                    newlines = max(1, int(dy / LINE_THRESHOLD))
                    merged_text.append("\n" * newlines + " ")
            merged_text.append(box[2])
            previous_box = box

        return "".join(merged_text)

    # ── Process images ─────────────────────────────────────────────────
    processed = 0
    errors = 0

    for fname in tqdm(files, desc=f"OCR-{args.worker_id}", ncols=100):
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        if os.path.exists(txt_path):
            continue

        try:
            img_path = os.path.join(IMG_DIR, fname)
            image = decode_image(img_path)
            result = ppocr_v3.predict(image)
            text = tostr_layout_preserving(result)
            with open(txt_path, "w") as f:
                f.write(text)
            processed += 1
        except Exception as e:
            errors += 1
            print(f"\n[Worker {args.worker_id}] ERROR on {fname}: {e}", flush=True)

    print(
        f"\n[Worker {args.worker_id}] Done! "
        f"Processed: {processed}, Errors: {errors}",
        flush=True,
    )


# ════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════
def orchestrator_main(args):
    num_workers = args.workers

    # ── Collect image files ────────────────────────────────────────────
    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    print(f"Total images: {len(all_files)}")

    existing = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".txt"):
                existing.add(os.path.splitext(f)[0])
    to_process = [f for f in all_files if os.path.splitext(f)[0] not in existing]

    print(f"Already processed: {len(existing)}")
    print(f"To process:        {len(to_process)}")
    print(f"Workers:           {num_workers}")

    if not to_process:
        print("Nothing to do — all OCR files already exist.")
        return

    # ── Split into shards ──────────────────────────────────────────────
    shards = [[] for _ in range(num_workers)]
    for i, f in enumerate(to_process):
        shards[i % num_workers].append(f)

    shard_dir = "/tmp/ocr_shards"
    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = []
    for i, shard in enumerate(shards):
        path = os.path.join(shard_dir, f"shard_{i}.txt")
        with open(path, "w") as f:
            for fname in shard:
                f.write(fname + "\n")
        shard_paths.append(path)
        print(f"  Worker {i}: {len(shard)} files  -> {path}")

    # ── Spawn worker subprocesses ──────────────────────────────────────
    processes = []
    log_files = []

    for i in range(num_workers):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--worker_id", str(i),
            "--shard_file", shard_paths[i],
        ]

        log_path = f"/tmp/ocr_worker_{i}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)

        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append(p)
        print(f"  Started Worker {i} (PID {p.pid}) -> {log_path}")

    print(f"\n{'='*60}")
    print(f"  {num_workers} OCR workers launched. Monitoring progress...")
    print(f"{'='*60}\n")

    # ── Monitor ────────────────────────────────────────────────────────
    start_time = time.time()

    while True:
        alive = sum(1 for p in processes if p.poll() is None)
        if alive == 0:
            break

        current_count = 0
        if os.path.exists(OUTPUT_DIR):
            current_count = sum(
                1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")
            )

        done_this_round = current_count - len(existing)
        elapsed = time.time() - start_time
        rate = done_this_round / elapsed if elapsed > 0 else 0
        remaining = len(to_process) - done_this_round
        eta_min = (remaining / rate / 60) if rate > 0 else float("inf")

        print(
            f"\r  Progress: {done_this_round}/{len(to_process)} "
            f"({done_this_round/len(to_process)*100:.1f}%) | "
            f"Rate: {rate:.1f} files/s | "
            f"Elapsed: {elapsed/60:.1f}min | "
            f"ETA: {eta_min:.1f}min | "
            f"Alive: {alive}/{num_workers}",
            end="",
            flush=True,
        )
        time.sleep(15)

    print()

    # ── Results ────────────────────────────────────────────────────────
    total_time = (time.time() - start_time) / 60
    all_success = True

    for i, p in enumerate(processes):
        rc = p.returncode
        status = "SUCCESS" if rc == 0 else f"FAILED (rc={rc})"
        if rc != 0:
            all_success = False
        print(f"  Worker {i}: {status}")
        log_files[i].close()

    final_count = sum(
        1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")
    ) if os.path.exists(OUTPUT_DIR) else 0

    print(f"\n  Total OCR files: {final_count}/{len(all_files)}")
    print(f"  Total time: {total_time:.1f} minutes")

    if not all_success:
        print("\n  ⚠ Some workers failed. Check logs:")
        for i in range(num_workers):
            print(f"    /tmp/ocr_worker_{i}.log")

    # Clean up
    for sp in shard_paths:
        try:
            os.remove(sp)
        except OSError:
            pass

    if all_success:
        print("\n  ✓ All OCR completed successfully!")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Parallel OCR for SlideVQA using PP-OCRv3"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel CPU workers (default: 4)",
    )

    # Worker args (internal)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_id", type=str, default="")
    parser.add_argument("--shard_file", type=str, default="")

    args = parser.parse_args()

    if args.worker:
        worker_main(args)
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
