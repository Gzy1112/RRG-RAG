#!/bin/bash
# ColQwen2.5 page retrieval server (VL-only, legacy GMM triage 20/10/5).
# Needs data/<dataset>/colqwen25_ingestion_v3.pt (data_prep/ingest_colqwen25.py).
#
# Usage: scripts/start_retrieval_server.sh <SlideVQA|ViDoSeek|MMLongBench> [port=8000] [gpu=0]
set -eu
source "$(dirname "$0")/env.sh"
DS="${1:?usage: $0 <dataset> [port] [gpu]}"
PORT="${2:-8000}"
export CUDA_VISIBLE_DEVICES="${3:-${CUDA_VISIBLE_DEVICES:-0}}"
need_file "$DATA/$DS/colqwen25_ingestion_v3.pt" "python data_prep/ingest_colqwen25.py --dataset $DS"
need_adapter "$VIDORAG_COLQWEN25_ADAPTER" "python data_prep/fix_colqwen25_adapter.py"

cd "$P"
VIDORAG_VL_ONLY=1 VIDORAG_GMM_LEGACY=1 exec python flask_server_vidorag.py \
    --vl_model colqwen2.5-v0.2-fixed --dataset "$DS" --port "$PORT" --topk 100
