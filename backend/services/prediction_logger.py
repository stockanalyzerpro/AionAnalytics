"""
prediction_logger.py — v3.0
AION Analytics — Nightly Prediction & Policy Logger

Produces clean, structured logs that the dashboard + insights tools can consume.
Logs include:
    • multi-horizon predictions (1d→52w)
    • policy actions (BUY/SELL/HOLD)
    • confidence + score distribution
    • top bullish/bearish symbols
    • context + sentiment overlays
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    log,
)
from backend.core.ai_model import predict_all
from backend.core.policy_engine import apply_policy


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
LOG_DIR: Path = PATHS["nightly_predictions"]
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _file_timestamp() -> str:
    return datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _extract_clean(preds: Dict[str, Any], pol: Dict[str, Any]) -> Dict[str, Any]:
    """Strip down predictions + policy into clean dashboard-ready dicts."""
    out: Dict[str, Any] = {}

    # Multi-horizon predictions
    for h, block in preds.items():
        out[h] = {
            "score": float(block.get("score", 0.0)),
            "confidence": float(block.get("confidence", 0.0)),
            "label": int(block.get("label", 0)),
        }

    # Policy summary
    if isinstance(pol, dict):
        out["policy"] = {
            "intent": pol.get("intent"),
            "score": pol.get("score"),
            "confidence": pol.get("confidence"),
            "exposure_scale": pol.get("exposure_scale"),
            "risk": pol.get("risk"),
        }
    else:
        out["policy"] = None

    return out


def _top_moves(pred_summary: Dict[str, Dict[str, Any]], horizon: str, top_n=20):
    """Rank symbols by strongest positive/negative score."""
    rows = []
    for sym, data in pred_summary.items():
        h = data.get(horizon)
        if not h:
            continue
        score = h.get("score", 0.0)
        rows.append((sym, score))

    rows.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_bullish": rows[:top_n],
        "top_bearish": rows[-top_n:][::-1],
    }


# ---------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------
def log_predictions(save_to_file: bool = True) -> Dict[str, Any]:
    """
    Main nightly logger.
    1) Loads rolling
    2) Generates predictions for all symbols
    3) Applies policy engine
    4) Builds summary stats
    5) Writes full structured log file
    """

    log("[prediction_logger] 🚀 Starting nightly prediction logging…")

    # -------------------------------------------------------------
    # Load rolling
    # -------------------------------------------------------------
    rolling = _read_rolling() or {}
    if not rolling:
        log("[prediction_logger] ⚠️ rolling.json.gz is empty.")
        return {"status": "no_rolling"}

    # -------------------------------------------------------------
    # Run predictions
    # -------------------------------------------------------------
    preds = predict_all(rolling)
    if not preds:
        log("[prediction_logger] ⚠️ No predictions available.")
        return {"status": "no_predictions"}

    # Annotate predictions into rolling
    for sym, node in rolling.items():
        if sym.upper() in preds:
            node["predictions"] = preds[sym.upper()]
            rolling[sym] = node

    # -------------------------------------------------------------
    # Apply policy (updates rolling in-place)
    # -------------------------------------------------------------
    apply_policy()

    # Reload rolling (policy engine saved it)
    rolling = _read_rolling() or {}

    # -------------------------------------------------------------
    # Clean/flatten for logging
    # -------------------------------------------------------------
    summary_per_symbol = {}
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        p = node.get("predictions", {})
        pol = node.get("policy", {})
        summary_per_symbol[sym] = _extract_clean(p, pol)

    # -------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------
    top = {
        "1d": _top_moves(summary_per_symbol, "1d", 20),
        "1w": _top_moves(summary_per_symbol, "1w", 20),
        "4w": _top_moves(summary_per_symbol, "4w", 20),
        "52w": _top_moves(summary_per_symbol, "52w", 20),
    }

    avg_confidence = [
        data[h]["confidence"]
        for data in summary_per_symbol.values()
        for h in ("1d", "1w", "4w") if h in data
    ]

    model_stats = {
        "avg_confidence": float(sum(avg_confidence) / len(avg_confidence)) if avg_confidence else 0.0,
        "symbols": len(summary_per_symbol),
    }

    # -------------------------------------------------------------
    # Build final payload
    # -------------------------------------------------------------
    payload = {
        "timestamp": datetime.now(TIMEZONE).isoformat(),
        "models": model_stats,
        "top_moves": top,
        "symbols": summary_per_symbol,
    }

    # -------------------------------------------------------------
    # Save file
    # -------------------------------------------------------------
    if save_to_file:
        out_path = LOG_DIR / f"predictions_{_file_timestamp()}.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log(f"[prediction_logger] 💾 Saved → {out_path}")
        except Exception as e:
            log(f"[prediction_logger] ❌ Failed writing prediction log: {e}")

    return payload


# CLI
if __name__ == "__main__":
    out = log_predictions()
    print(json.dumps(out, indent=2))
