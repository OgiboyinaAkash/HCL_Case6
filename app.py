"""
PrismRoute — Web Frontend & API Server (Localhost).

Serves the interactive web UI and API endpoints:
  - GET  /            -> Interactive single-page dashboard
  - POST /api/query   -> Process query through full architecture workflow
  - GET  /api/metrics -> Live cost savings and operational telemetry
  - POST /api/clear   -> Reset semantic cache
"""

import os
import sys
from flask import Flask, jsonify, render_template, request

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.service import service

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    """Serves the main interactive dashboard."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """Processes query through the architecture workflow."""
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    scope_id = data.get("scope_id", "default").strip() or "default"

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    result = service.process_query(query, scope_id=scope_id)
    return jsonify(result)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns live telemetry, cache stats, and cost savings."""
    scope_id = request.args.get("scope_id")
    metrics = service.get_metrics(scope_id=scope_id)
    return jsonify(metrics)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Flushes cache and reloads in-memory index."""
    data = request.get_json(silent=True) or {}
    scope_id = data.get("scope_id")
    service.clear_cache(scope_id=scope_id)
    return jsonify({"success": True, "message": "Cache cleared successfully."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print("  PRISMROUTE — WEB FRONTEND RUNNING")
    print(f"  Access in your browser: http://localhost:{port}")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
