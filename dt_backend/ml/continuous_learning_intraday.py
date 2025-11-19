
"""Continuous learning / meta-ensemble updater for intraday models.

This is intentionally conservative:
  - If metrics are missing or malformed, it logs and exits.
  - It only nudges ensemble weights based on recent accuracies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtml_data": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models.ensemble.intraday_hybrid_ensemble import EnsembleConfig

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


def _metrics_path() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / "intraday_model_metrics.json"


def _load_metrics() -> Dict[str, Any] | None:
    path = _metrics_path()
    if not path.exists():
        log(f"[continuous_learning_intraday] ℹ️ Metrics file not found at {path} — skipping.")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("metrics JSON must be an object")
        return data
    except Exception as e:
        log(f"[continuous_learning_intraday] ⚠️ Failed to read metrics: {e}")
        return None


def _derive_weights(metrics: Dict[str, Any]) -> EnsembleConfig:
    """Derive weights from recent per-model accuracies.

    Expected structure:
    {
      "lightgbm": {"accuracy": 0.60},
      "lstm": {"accuracy": 0.55},
      "transformer": {"accuracy": 0.52}
    }
    """
    def acc_of(key: str) -> float:
        return float(((metrics.get(key) or {}).get("accuracy", 0.0)) or 0.0)

    acc_lgb = acc_of("lightgbm")
    acc_lstm = acc_of("lstm")
    acc_transf = acc_of("transformer")

    arr = np.array([acc_lgb, acc_lstm, acc_transf], dtype=float)
    arr[arr < 0.0] = 0.0
    s = float(arr.sum())
    if s <= 0.0:
        log("[continuous_learning_intraday] ℹ️ No positive accuracies, keeping current weights.")
        return EnsembleConfig.load()

    arr = arr / s
    return EnsembleConfig(w_lgb=float(arr[0]), w_lstm=float(arr[1]), w_transf=float(arr[2]))


def run_continuous_learning_intraday() -> None:
    metrics = _load_metrics()
    if metrics is None:
        return

    old_cfg = EnsembleConfig.load()
    new_cfg = _derive_weights(metrics)

    if (
        abs(new_cfg.w_lgb - old_cfg.w_lgb) < 1e-6
        and abs(new_cfg.w_lstm - old_cfg.w_lstm) < 1e-6
        and abs(new_cfg.w_transf - old_cfg.w_transf) < 1e-6
    ):
        log("[continuous_learning_intraday] ℹ️ Weights unchanged; nothing to update.")
        return

    new_cfg.save()
    log(
        "[continuous_learning_intraday] ✅ Updated ensemble weights → "
        f"LGB={new_cfg.w_lgb:.3f}, LSTM={new_cfg.w_lstm:.3f}, TRANSF={new_cfg.w_transf:.3f}"
    )


if __name__ == "__main__":
    run_continuous_learning_intraday()
