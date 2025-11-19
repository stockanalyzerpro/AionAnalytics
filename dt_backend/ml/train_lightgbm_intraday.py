
"""Train a fast 3-class LightGBM intraday model.

Expects a parquet built by ml_data_builder_intraday.py with:
  - feature columns
  - label column: 'label' (SELL/HOLD/BUY) or 'label_id' (0/1/2)

Saves artifacts under: DT_PATHS["dtmodels"] / "lightgbm_intraday"
  - model.txt
  - feature_map.json
  - label_map.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtml_data": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models import LABEL_ORDER, LABEL2ID, ID2LABEL, get_model_dir

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


def _resolve_training_data() -> Path:
    base = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    return base / "training_data_intraday.parquet"


def _load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    path = _resolve_training_data()
    if not path.exists():
        raise FileNotFoundError(f"Intraday training data not found at {path}")
    log(f"[train_lightgbm_intraday] 📦 Loading training data from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Training dataframe at {path} is empty.")

    if "label_id" in df.columns:
        y = df["label_id"].astype(int)
    elif "label" in df.columns:
        y = df["label"].map(LABEL2ID)
        if y.isna().any():
            bad = df["label"][y.isna()].unique().tolist()
            raise ValueError(f"Unknown labels in training data: {bad}")
    else:
        raise ValueError("Training data must contain 'label' or 'label_id' column.")

    X = df.drop(columns=[c for c in ("label", "label_id") if c in df.columns])
    return X, y


def _train_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any] | None = None,
) -> lgb.Booster:
    if params is None:
        params = {
            "objective": "multiclass",
            "num_class": len(LABEL_ORDER),
            "metric": ["multi_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "seed": 42,
            "verbosity": -1,
        }
    dtrain = lgb.Dataset(X, label=y.values)
    log(f"[train_lightgbm_intraday] 🚀 Training on {len(X):,} rows, {X.shape[1]} features...")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dtrain],
        valid_names=["train"],
        verbose_eval=50,
    )
    log("[train_lightgbm_intraday] ✅ Training complete.")
    return booster


def _save_artifacts(booster: lgb.Booster, feature_names: list[str]) -> None:
    model_dir = get_model_dir("lightgbm")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.txt"
    fmap_path = model_dir / "feature_map.json"
    label_map_path = model_dir / "label_map.json"

    booster.save_model(str(model_path))
    with fmap_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_order": LABEL_ORDER,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"[train_lightgbm_intraday] 💾 Saved model → {model_path}")
    log(f"[train_lightgbm_intraday] 💾 Saved feature_map → {fmap_path}")
    log(f"[train_lightgbm_intraday] 💾 Saved label_map → {label_map_path}")


def train_lightgbm_intraday() -> Dict[str, Any]:
    """High-level entrypoint to train & persist the intraday LightGBM model."""
    X, y = _load_training_data()
    booster = _train_lgb(X, y)
    _save_artifacts(booster, list(X.columns))
    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_order": LABEL_ORDER,
    }
    log(f"[train_lightgbm_intraday] 📊 Summary: {summary}")
    return summary


if __name__ == "__main__":
    train_lightgbm_intraday()
