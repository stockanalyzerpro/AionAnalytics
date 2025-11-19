# dt_backend/ml/signals_rank_builder.py — v2.0
"""
Build intraday signals + ranks from `policy_dt` in rolling.

This is the bridge between the model/policy layer and external
consumers (front-end, rank fetcher, bot runners). It reads:

    rolling[sym]["policy_dt"]

and produces compact JSON artifacts under DT_PATHS["signals_intraday_*"].
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

from dt_backend.core import DT_PATHS, _read_rolling, log


def _signals_dir() -> Path:
    return DT_PATHS["signals_intraday_predictions_dir"]


def _ranks_dir() -> Path:
    return DT_PATHS["signals_intraday_ranks_dir"]


def build_intraday_signals(top_n: int = 200) -> Dict[str, Any]:
    """
    Aggregate `policy_dt` into ranked signals and write JSON artifacts.

    Outputs
    -------
    • predictions.json              (full dict per symbol)
    • prediction_rank_fetch.json.gz (compact rank list for schedulers)
    """
    rolling = _read_rolling()
    if not rolling:
        log("[dt_signals] ⚠️ rolling empty, nothing to build.")
        return {"status": "empty"}

    rows: List[Dict[str, Any]] = []
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        policy = (node or {}).get("policy_dt") or {}
        if not isinstance(policy, dict) or not policy:
            continue

        rows.append(
            {
                "symbol": sym,
                "action": policy.get("action", "HOLD"),
                "score": float(policy.get("score", 0.0) or 0.0),
                "confidence": float(policy.get("confidence", 0.0) or 0.0),
                "expected_return": float(policy.get("expected_return", 0.0) or 0.0),
                "vol_bucket": policy.get("vol_bucket"),
                "regime": policy.get("regime"),
            }
        )

    if not rows:
        log("[dt_signals] ⚠️ no policy rows found.")
        return {"status": "no_rows"}

    # Sort by score descending
    rows.sort(key=lambda r: r["score"], reverse=True)

    top = rows[: max(0, int(top_n))]

    sig_dir = _signals_dir()
    ranks_dir = _ranks_dir()
    sig_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = sig_dir / "intraday_predictions.json"
    ranks_path = ranks_dir / "prediction_rank_fetch.json.gz"

    try:
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(top, f, ensure_ascii=False, indent=2)
        log(f"[dt_signals] ✅ wrote top-{len(top)} predictions → {predictions_path}")
    except Exception as e:
        log(f"[dt_signals] ⚠️ failed to write predictions: {e}")

    try:
        # Compact rank list used by external schedulers
        rank_payload = [
            {"symbol": row["symbol"], "score": row["score"], "action": row["action"]}
            for row in top
        ]
        with gzip.open(ranks_path, "wt", encoding="utf-8") as f:
            json.dump(rank_payload, f, ensure_ascii=False)
        log(f"[dt_signals] ✅ wrote rank fetch payload → {ranks_path}")
    except Exception as e:
        log(f"[dt_signals] ⚠️ failed to write ranks: {e}")

    return {
        "status": "ok",
        "total": len(rows),
        "top_n": len(top),
        "predictions_path": str(predictions_path),
        "ranks_path": str(ranks_path),
    }
