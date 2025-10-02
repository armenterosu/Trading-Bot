from __future__ import annotations
"""Simple HTTP metrics server using Flask.

Serves the most recent metrics CSV as JSON at /metrics-json and a plain text at /metrics.
Configure metrics_path in config and ensure files are written there by backtests or runtime.
"""
from typing import Dict, Any
import os
import glob
import pandas as pd
from flask import Flask, jsonify, Response


def latest_metrics_file(metrics_path: str) -> str | None:
    files = glob.glob(os.path.join(metrics_path, "*.csv"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_metrics(path: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path)
        if set(df.columns) >= {"Metric", "Value"}:
            return {str(r.Metric): float(r.Value) if str(r.Value).replace('.', '', 1).isdigit() else r.Value for _, r in df.iterrows()}
        # else equity curve csv
        return {"equity_curve_rows": len(df)}
    except Exception:
        return {}


def create_app(metrics_path: str = "metrics/") -> Flask:
    app = Flask(__name__)

    @app.get("/metrics-json")
    def metrics_json() -> Any:
        f = latest_metrics_file(metrics_path)
        if not f:
            return jsonify({"status": "no_metrics"})
        return jsonify(load_metrics(f))

    @app.get("/metrics")
    def metrics_text() -> Response:
        f = latest_metrics_file(metrics_path)
        if not f:
            return Response("status 0\n", mimetype="text/plain")
        data = load_metrics(f)
        lines = []
        for k, v in data.items():
            try:
                val = float(v)
            except Exception:
                continue
            lines.append(f"{k} {val}")
        return Response("\n".join(lines) + "\n", mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app(os.environ.get("METRICS_PATH", "metrics/"))
    app.run(host="0.0.0.0", port=8000)
