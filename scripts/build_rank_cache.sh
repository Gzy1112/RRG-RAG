#!/bin/bash
# Per-question ColQwen2.5 ranking (pre-GMM, top-100) used by rank-1 rescue, the
# same-deck second hop and the Inspector features. Queries a running server.
#
# Usage: scripts/build_rank_cache.sh <dataset> <query_file> <out.jsonl> [port=8000]
#   scripts/build_rank_cache.sh SlideVQA    rag_dataset_val.json  outputs/rank_cache/SlideVQA_val.jsonl
#   scripts/build_rank_cache.sh SlideVQA    rag_dataset_test.json outputs/rank_cache/SlideVQA_test.jsonl
#   scripts/build_rank_cache.sh ViDoSeek    rag_dataset.json      outputs/rank_cache/ViDoSeek.jsonl    8001
#   scripts/build_rank_cache.sh MMLongBench rag_dataset.json      outputs/rank_cache/MMLongBench.jsonl 8002
set -eu
source "$(dirname "$0")/env.sh"
DS="${1:?usage: $0 <dataset> <query_file> <out.jsonl> [port]}"
QF="${2:?query file}"
OUT_FILE="$(realpath -m "${3:?output jsonl}")"
PORT="${4:-8000}"
need_server "$PORT" "$DS"
export RANK_CACHE_DATASET="$DATA/$DS/$QF"
export RETRIEVAL_SERVER_URL="http://localhost:$PORT"
need_file "$RANK_CACHE_DATASET"
mkdir -p "$(dirname "$OUT_FILE")"

cd "$P"
python -u rebuild_rank_cache_colqwen25.py --backend http --resume --out "$OUT_FILE"
