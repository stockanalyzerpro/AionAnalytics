# backend/core/continuous_learning.py — v3.0
"""
Continuous Learning — AION Analytics (Backend EOD Brain)

Responsibilities:
    • Read rolling + brain snapshots.
    • For each symbol, compare:
        - predicted scores (per-horizon) vs realized returns
        - primary context pred_score (1d ensemble) vs realized 1d return
    • Maintain sliding performance history windows:
        - short window (e.g. last 30 samples)
        - long  window (e.g. last 120 samples)
    • Compute drift scores per symbol AND per horizon.
    • Write results back to brain + rolling.
    • Trigger early retraining if drift is severe.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, List

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    save_brain,
    save_rolling,
    log,
    safe_float,
)
from backend.core.ai_model import train_all_models

BRAIN_PATH = PATHS.get("brain_file", PATHS["ml_data"] / "rolling_brain.json.gz")

SHORT_WINDOW = 30
LONG_WINDOW = 120


def _ensure_brain(brain: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(brain, dict):
        brain = {}
    brain.setdefault("_meta", {})
    meta = brain["_meta"]
    meta.setdefault("updated_at", None)
    meta.setdefault("drift_by_horizon", {})
    return brain


def _append_sample(store: List[Dict[str, Any]], sample: Dict[str, Any], max_len: int) -> List[Dict[str, Any]]:
    store.append(sample)
    if len(store) > max_len:
        store = store[-max_len:]
    return store


def _compute_window_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "hit_ratio": 0.5,
            "mae": 0.0,
            "avg_error": 0.0,
            "avg_conf": 0.0,
        }
    hits = [int(s.get("hit", 0)) for s in samples]
    errs = [safe_float(s.get("error", 0.0)) for s in samples]
    confs = [safe_float(s.get("confidence", 0.0)) for s in samples]
    n = len(samples)
    hit_ratio = sum(hits) / float(n)
    mae = sum(abs(e) for e in errs) / float(n)
    avg_err = sum(errs) / float(n)
    avg_conf = sum(confs) / float(n)
    return {
        "n": n,
        "hit_ratio": hit_ratio,
        "mae": mae,
        "avg_error": avg_err,
        "avg_conf": avg_conf,
    }


def run_continuous_learning() -> Dict[str, Any]:
    log("[continuous_learning] 🧠 Running continuous learning v3.0…")

    rolling = _read_rolling() or {}
    brain = _read_brain() or {}
    brain = _ensure_brain(brain)

    horizon_drift: Dict[str, List[float]] = {}
    updated_symbols = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        hist = node.get("history") or []
        if len(hist) < 2:
            continue

        try:
            last = hist[-1]
            prev = hist[-2]
            c0 = safe_float(prev.get("close", 0.0))
            c1 = safe_float(last.get("close", 0.0))
            if c0 <= 0:
                continue
            realized_ret = (c1 - c0) / c0
        except Exception:
            continue

        ctx = node.get("context", {}) or {}
        pred_score_ctx = safe_float(ctx.get("pred_score", 0.0))

        bnode = brain.get(sym, {})
        perf = bnode.get("performance", {})
        short = perf.get("short_window", [])
        longw = perf.get("long_window", [])

        hit_ctx = int((pred_score_ctx >= 0 and realized_ret >= 0) or (pred_score_ctx < 0 and realized_ret < 0))
        err_ctx = realized_ret - pred_score_ctx

        sample_ts = last.get("date") or last.get("ts") or datetime.now(TIMEZONE).isoformat()
        sample_ctx = {
            "ts": sample_ts,
            "pred_score": pred_score_ctx,
            "realized_ret": realized_ret,
            "hit": hit_ctx,
            "error": err_ctx,
            "confidence": safe_float(node.get("predictions", {}).get("1d", {}).get("confidence", 0.0)),
            "horizon": "1d",
        }

        short = _append_sample(short, sample_ctx, SHORT_WINDOW)
        longw = _append_sample(longw, sample_ctx, LONG_WINDOW)

        stats_short = _compute_window_stats(short)
        stats_long = _compute_window_stats(longw)
        drift_ctx = stats_short["hit_ratio"] - stats_long["hit_ratio"]

        perf["short_window"] = short
        perf["long_window"] = longw
        perf["short_stats"] = stats_short
        perf["long_stats"] = stats_long
        perf["drift_score"] = drift_ctx

        bnode["performance"] = perf

        # ---------- Per-horizon drift ----------
        preds = node.get("predictions", {}) or {}
        h_perf = bnode.get("horizon_perf", {})

        for horizon, block in preds.items():
            score_h = safe_float(block.get("score", 0.0))
            conf_h = safe_float(block.get("confidence", 0.0))
            hit_h = int((score_h >= 0 and realized_ret >= 0) or (score_h < 0 and realized_ret < 0))
            err_h = realized_ret - score_h

            hp = h_perf.get(horizon, {})
            sw = hp.get("short_window", [])
            lw = hp.get("long_window", [])

            sample_h = {
                "ts": sample_ts,
                "pred_score": score_h,
                "realized_ret": realized_ret,
                "hit": hit_h,
                "error": err_h,
                "confidence": conf_h,
            }

            sw = _append_sample(sw, sample_h, SHORT_WINDOW)
            lw = _append_sample(lw, sample_h, LONG_WINDOW)

            sw_stats = _compute_window_stats(sw)
            lw_stats = _compute_window_stats(lw)
            drift_h = sw_stats["hit_ratio"] - lw_stats["hit_ratio"]

            hp["short_window"] = sw
            hp["long_window"] = lw
            hp["short_stats"] = sw_stats
            hp["long_stats"] = lw_stats
            hp["drift_score"] = drift_h

            h_perf[horizon] = hp

            horizon_drift.setdefault(horizon, []).append(drift_h)

        bnode["horizon_perf"] = h_perf
        brain[sym] = bnode
        updated_symbols += 1

    # ---------- Global horizon drift ----------
    meta = brain["_meta"]
    gdrift: Dict[str, Any] = {}
    severe_drift = False

    for h, arr in horizon_drift.items():
        if not arr:
            continue
        avg = float(sum(arr) / len(arr))
        n = len(arr)

        # Retrain heuristic:
        # - small negative but within -0.05 → ok
        # - big negative or big positive drift → retrain
        retrain = False
        if n >= 20 and (avg < -0.10 or abs(avg) > 0.25):
            retrain = True
            severe_drift = True

        gdrift[h] = {
            "avg_drift": avg,
            "n": n,
            "retrain_recommended": retrain,
        }

    meta["drift_by_horizon"] = gdrift
    meta["updated_at"] = datetime.now(TIMEZONE).isoformat()

    # Save brain + rolling first
    save_brain(brain)
    save_rolling(rolling)

    # ---------- Early retrain trigger ----------
    retrain_result: Dict[str, Any] = {}
    if severe_drift:
        log("[continuous_learning] 🔁 Severe drift detected — triggering early retrain (no Optuna).")
        try:
            retrain_result = train_all_models(
                dataset_name="training_data_daily.parquet",
                use_optuna=False,
                n_trials=0,
            )
        except Exception as e:
            log(f"[continuous_learning] ⚠️ Early retrain failed: {e}")
            retrain_result = {"error": str(e)}

    log(f"[continuous_learning] ✅ Updated performance for {updated_symbols} symbols.")
    log(f"[continuous_learning] 🌡 Global horizon drift: {gdrift}")

    return {
        "symbols_updated": updated_symbols,
        "drift_by_horizon": gdrift,
        "early_retrain_triggered": severe_drift,
        "early_retrain_result": retrain_result,
    }


if __name__ == "__main__":
    out = run_continuous_learning()
    print(json.dumps(out, indent=2))
