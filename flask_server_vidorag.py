"""
Flask server that serves ViDoRAG's HybridSearchEngine for document retrieval.

Replaces MMOA-RAG's Contriever+FAISS retrieval with ViDoRAG's hybrid
visual-language + text search with optional GMM-based filtering.

Usage:
    # Direct run
    python flask_server_vidorag.py --dataset MMLongBench --port 8000

    # With gunicorn (multi-worker)
    gunicorn -w 1 -b 0.0.0.0:8000 "flask_server_vidorag:create_app()" --timeout 300
"""

import sys
import os
import argparse
import logging
import time
import json
import threading

# search_engine.py and the data/ directory live next to this file.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flask_server_vidorag")

# ---------------------------------------------------------------------------
# Global reference to the search engine (populated at startup)
# ---------------------------------------------------------------------------
_search_engine = None
_engine_lock = threading.Lock()
_config = {
    "dataset": "MMLongBench",
    "topk": 20,
    "use_gmm": True,
    "vl_model": "vidore/colqwen2-v1.0",
    "text_model": "BAAI/bge-m3",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_node(node_dict: dict, dataset_dir: str) -> dict:
    """Convert a llama-index source_node dict into the API response format."""
    inner = node_dict.get("node", {})

    # --- image_path ---
    image_path = inner.get("image_path") or inner.get("image", "")
    if image_path and not os.path.isabs(image_path):
        normalized = os.path.normpath(image_path)
        dataset_marker = os.path.normpath(os.path.join("data", os.path.basename(dataset_dir)))
        if normalized.startswith(dataset_marker + os.sep):
            project_root = os.path.dirname(os.path.dirname(dataset_dir))
            image_path = os.path.abspath(os.path.join(project_root, normalized))
        else:
            image_path = os.path.abspath(os.path.join(dataset_dir, normalized))

    # --- doc_id  (stem of the image filename, e.g. "doc_page_0") ---
    doc_id = ""
    if image_path:
        doc_id = os.path.splitext(os.path.basename(image_path))[0]

    # --- ocr_text ---
    ocr_text = inner.get("text", "")

    # --- score ---
    score = node_dict.get("score")

    # --- metadata ---
    metadata = inner.get("metadata", {})
    # Derive a human-friendly filename / page_num if possible
    if doc_id and "filename" not in metadata:
        parts = doc_id.rsplit("_", 1)
        if len(parts) == 2:
            metadata.setdefault("filename", parts[0])
            try:
                metadata.setdefault("page_num", int(parts[1]))
            except ValueError:
                pass

    return {
        "doc_id": doc_id,
        "image_path": image_path,
        "ocr_text": ocr_text,
        "score": score,
        "metadata": metadata,
    }


def _init_search_engine(cfg: dict):
    """Instantiate the search engine and store it globally.

    When ``VIDORAG_VL_ONLY=1`` is set *and* the VL model name contains
    'colqwen2.5', the engine is a standalone ``ColQwen25Engine`` (no text
    branch) — byte-identical to the in-process retrieval path used by
    ``eval.py`` with ``--embed_model_name_vl vidore/colqwen2.5-*``.
    Otherwise a full ``HybridSearchEngine`` (VL + text) is created.
    """
    global _search_engine

    _vl_only = os.environ.get('VIDORAG_VL_ONLY', '0') == '1'
    _vl_name_lower = (cfg.get('vl_model') or '').lower()
    _use_vl_only = _vl_only and ('colqwen2.5' in _vl_name_lower or 'colqwen2_5' in _vl_name_lower)

    start = time.time()

    if _use_vl_only:
        from search_engine import ColQwen25Engine
        logger.info("Initializing ColQwen25Engine standalone (dataset=%s, VL_ONLY=1) ...", cfg["dataset"])
        _search_engine = ColQwen25Engine(dataset=cfg["dataset"])
        _search_engine.topk = cfg["topk"]
        _search_engine.gmm = cfg["use_gmm"]
        if cfg["use_gmm"]:
            if os.environ.get('VIDORAG_GMM_LEGACY', '0') == '1':
                # Restore the legacy ViDoRAG GMM triage (ColQwen25Engine defaults).
                # The Seeker adapters were trained/evaluated on 5-10 triaged pages;
                # bounding by the server's own --topk (e.g. 100) makes
                # min_output_gmm >= 50, so GMM never bites and every request
                # returns N untriaged pages (a train/eval distribution mismatch).
                _search_engine.input_gmm = 20
                _search_engine.max_output_gmm = 10
                _search_engine.min_output_gmm = 5
            else:
                _search_engine.input_gmm = cfg["topk"] * 2
                _search_engine.max_output_gmm = cfg["topk"]
                _search_engine.min_output_gmm = max(cfg["topk"] // 2, 5)
        elapsed = time.time() - start
        logger.info("ColQwen25Engine standalone ready in %.1f s", elapsed)
    else:
        from search_engine import HybridSearchEngine
        logger.info("Initializing HybridSearchEngine (dataset=%s) ...", cfg["dataset"])
        _search_engine = HybridSearchEngine(
            dataset=cfg["dataset"],
            embed_model_name_vl=cfg["vl_model"],
            embed_model_name_text=cfg["text_model"],
            topk=cfg["topk"],
            gmm=cfg["use_gmm"],
        )
        elapsed = time.time() - start
        logger.info("HybridSearchEngine ready in %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Flask application factory (gunicorn-compatible)
# ---------------------------------------------------------------------------

def create_app(cfg: dict | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    cfg : dict, optional
        Configuration overrides.  When *None*, values are read from
        environment variables (see ``_load_config_from_env``).
    """
    if cfg is None:
        cfg = _load_config_from_env()

    _config.update(cfg)

    app = Flask(__name__)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        """Simple health-check endpoint."""
        return jsonify({
            "status": "ok",
            "dataset": _config["dataset"],
            "engine_loaded": _search_engine is not None,
        })

    @app.route("/search", methods=["POST"])
    def search():
        """Run hybrid retrieval for one or more questions.

        Request JSON
        -------------
        {
            "questions": ["q1", "q2", ...],
            "N": 10,              // optional, override topk
            "use_gmm": true       // optional, override GMM setting
        }

        Response JSON
        --------------
        {
            "results": [
                {
                    "question": "q1",
                    "documents": [
                        {
                            "doc_id": "...",
                            "image_path": "...",
                            "ocr_text": "...",
                            "score": 0.95,
                            "metadata": {...}
                        }
                    ]
                }
            ]
        }
        """
        if _search_engine is None:
            return jsonify({"error": "Search engine not initialized"}), 503

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        questions = data.get("questions")
        if not questions or not isinstance(questions, list):
            return jsonify({
                "error": "'questions' must be a non-empty list of strings"
            }), 400
        if any(not isinstance(question, str) or not question.strip() for question in questions):
            return jsonify({"error": "Every question must be a non-empty string"}), 400

        # Per-request overrides
        topk = data.get("N", _config["topk"])
        use_gmm = data.get("use_gmm", _config["use_gmm"])
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 100:
            return jsonify({"error": "'N' must be an integer between 1 and 100"}), 400
        if not isinstance(use_gmm, bool):
            return jsonify({"error": "'use_gmm' must be a boolean"}), 400

        dataset_dir = os.path.join(
            os.environ.get("VIDORAG_DATA_ROOT", os.path.join(PROJECT_ROOT, "data")),
            _config["dataset"],
        )

        # Temporarily adjust engine settings for this request (thread-safe)
        with _engine_lock:
            original_topk = _search_engine.topk
            original_gmm = _search_engine.gmm
            _search_engine.topk = topk
            _search_engine.gmm = use_gmm
            try:
                results = []
                for question in questions:
                    try:
                        raw = _search_engine.search(question)
                        source_nodes = raw.get("source_nodes", [])
                        documents = [
                            _parse_node(sn, dataset_dir) for sn in source_nodes
                        ]
                    except Exception as exc:
                        logger.exception("Error searching for question: %s", question)
                        documents = []
                        error = {"code": "retrieval_failed", "message": str(exc)}
                    else:
                        error = None

                    result = {
                        "question": question,
                        "documents": documents,
                    }
                    if error is not None:
                        result["error"] = error
                    results.append(result)
            finally:
                _search_engine.topk = original_topk
                _search_engine.gmm = original_gmm

        return jsonify({"results": results})

    # ------------------------------------------------------------------
    # Eagerly load the search engine when the app is first created
    # ------------------------------------------------------------------
    with app.app_context():
        _init_search_engine(_config)

    return app


def _load_config_from_env() -> dict:
    """Read configuration from environment variables."""
    return {
        "dataset": os.environ.get("VIDORAG_DATASET", "MMLongBench"),
        "topk": int(os.environ.get("VIDORAG_TOPK", "20")),
        "use_gmm": os.environ.get("VIDORAG_USE_GMM", "1") in ("1", "true", "True"),
        "vl_model": os.environ.get("VIDORAG_VL_MODEL", "vidore/colqwen2-v1.0"),
        "text_model": os.environ.get("VIDORAG_TEXT_MODEL", "BAAI/bge-m3"),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViDoRAG Hybrid Retrieval Server",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MMLongBench",
        help="Dataset name under data/ (default: MMLongBench)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Default number of results to return (default: 20)",
    )
    parser.add_argument(
        "--use-gmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable GMM-based score filtering (default: enabled)",
    )
    parser.add_argument(
        "--vl_model",
        type=str,
        default="vidore/colqwen2-v1.0",
        help="Visual-language embedding model. Options: vidore/colqwen2-v1.0 (default), "
             "colqwen2.5 (uses ColQwen25Engine for full-corpus MaxSim), "
             "or any model name containing 'colqwen2.5' or 'colqwen2_5'",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="BAAI/bge-m3",
        help="Text embedding model (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = {
        "dataset": args.dataset,
        "topk": args.topk,
        "use_gmm": args.use_gmm,
        "vl_model": args.vl_model,
        "text_model": args.text_model,
    }

    app = create_app(cfg)

    logger.info(
        "Starting server on %s:%d (dataset=%s, topk=%d, gmm=%s)",
        args.host,
        args.port,
        args.dataset,
        args.topk,
        args.use_gmm,
    )
    app.run(host=args.host, port=args.port)
