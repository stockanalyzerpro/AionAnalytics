# dt_backend/ml/drift_monitor_dt.py — Intraday model drift monitor.
"""
Aggregates intraday model behavior into a metrics JSON file that can be
consumed by continuous_learning_intraday to adjust ensemble weights.

Writes:
    ml_data_dt/metrics/intraday_model_metrics.json
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Any] = {
        "dtml_data": Path("ml_data_dt")
    }

from dt_backend.core.data_pipeline_dt import _read_rolling, log


def _metrics_path() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    mdir = root / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir / "intraday_model_metrics.json"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _entropy(dist: Dict[str, float]) -> float:
    ent = 0.0
    for p in dist.values():
        if p <= 0.0:
            continue
        ent -= p * math.log(p + 1e-12)
    return ent


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    kl = 0.0
    for k, pk in p.items():
        qk = q.get(k, 1e-12)
        if pk <= 0.0 or qk <= 0.0:
            continue
        kl += pk * math.log(pk / qk)
    return kl


def _collect_prediction_stats() -> Dict[str, Any]:
    rolling = _read_rolling()
    if not rolling:
        log("[drift_dt] ⚠️ rolling empty, nothing to analyze.")
        return {}

    counts = Counter()
    conf_sum = 0.0
    n = 0

    for sym, node in rolling.items():
        if sym.startswith("_") or not isinstance(node, dict):
            continue

        pred = node.get("predictions_dt") or node.get("predictions") or {}
        if not isinstance(pred, dict):
            continue

        label = str(pred.get("label") or pred.get("class") or "").upper()
        if not label:
            continue

        proba = pred.get("proba") or pred.get("probs") or {}
        if isinstance(proba, dict):
            base_conf = max((_safe_float(v) for v in proba.values()), default=0.0)
        else:
            base_conf = _safe_float(pred.get("confidence"), 0.0)

        counts[label] += 1
        conf_sum += base_conf
        n += 1

    if n == 0:
        log("[drift_dt] ⚠️ no predictions found in rolling.")
        return {}

    pred_dist = {k: v / n for k, v in counts.items()}
    avg_conf = conf_sum / n

    # Compare to uniform over observed classes.
    k = len(pred_dist)
    if k > 0:
        uniform = {c: 1.0 / k for c in pred_dist}
        kl = _kl_divergence(pred_dist, uniform)
        ent = _entropy(pred_dist)
    else:
        uniform = {}
        kl = 0.0
        ent = 0.0

    return {
        "lightgbm": {
            "n_samples": n,
            "pred_dist": pred_dist,
            "avg_confidence": avg_conf,
        },
        "meta": {
            "ts": datetime.now(timezone.utc).isoformat(),
            "entropy": ent,
            "kl_vs_uniform": kl,
        },
    }


def run_drift_monitor_intraday() -> Dict[str, Any]:
    metrics = _collect_prediction_stats()
    if not metrics:
        # still write an empty shell for downstream robustness
        metrics = {
            "lightgbm": {"n_samples": 0, "pred_dist": {}, "avg_confidence": 0.0},
            "meta": {
                "ts": datetime.now(timezone.utc).isoformat(),
                "entropy": 0.0,
                "kl_vs_uniform": 0.0,
            },
        }

    path = _metrics_path()
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        log(f"[drift_dt] ✅ wrote drift metrics → {path}")
    except Exception as e:
        log(f"[drift_dt] ⚠️ failed to write metrics at {path}: {e}")

    return metrics


def main() -> None:
    run_drift_monitor_intraday()


if __name__ == "__main__":
    main()
