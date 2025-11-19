"""
LightGBM intraday classifier for AION (SELL / HOLD / BUY).

This module is **drop-in ready** and focuses purely on inference:
  • locate the trained model + metadata
  • load them safely
  • run predictions for a feature matrix (numpy / pandas)

Training is handled elsewhere (dt_backend/ml/train_lightgbm_intraday.py)
which should write artefacts into:

    DT_PATHS["dtmodels"] / "intraday" / "lightgbm"

Expected artefacts:
  • model.txt            — LightGBM text model
  • feature_map.json     — ordered list of feature names
  • label_map.json       — mapping of class index -> label
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

try:
    from dt_backend.dt_logger import dt_log as log
except Exception:  # pragma: no cover
    def log(msg: str) -> None:  # type: ignore[no-redef]
        print(msg, flush=True)

try:
    from dt_backend.config_dt import DT_PATHS
except Exception:  # pragma: no cover
    # Fallback for ad-hoc / notebook use
    DT_PATHS: Dict[str, Path] = {
        "dtmodels": Path("ml_data_dt") / "models"
    }

# Default 3‑class mapping (can be overridden by label_map.json)
DEFAULT_LABEL_ORDER = ["SELL", "HOLD", "BUY"]
DEFAULT_ID2LABEL = {i: c for i, c in enumerate(DEFAULT_LABEL_ORDER)}


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------
def _model_dir() -> Path:
    base = Path(DT_PATHS["dtmodels"])  # type: ignore[index]
    return base / "intraday" / "lightgbm"


def _safe_read_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------
def load_lightgbm_metadata(model_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load feature_map + label_map for the intraday LightGBM model.

    Returns a dict:
        {
          "feature_map": [...],
          "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
          "label_order": ["SELL","HOLD","BUY"]
        }
    """
    md = model_dir or _model_dir()
    feature_map = _safe_read_json(md / "feature_map.json", [])
    label_map_raw = _safe_read_json(md / "label_map.json", DEFAULT_ID2LABEL)

    # Normalise label map to str->str
    if isinstance(label_map_raw, dict):
        label_map = {str(k): str(v) for k, v in label_map_raw.items()}
    else:
        # tolerate list like ["SELL","HOLD","BUY"]
        label_map = {str(i): str(lbl) for i, lbl in enumerate(label_map_raw)}

    # Build a stable label order
    # Prefer explicit ordering by numeric index if present
    try:
        ids = sorted(int(k) for k in label_map.keys())
        label_order = [label_map[str(i)] for i in ids]
    except Exception:
        # fallback: use values in insertion order
        label_order = list(label_map.values())

    if not label_order:
        label_order = DEFAULT_LABEL_ORDER

    return {
        "feature_map": feature_map,
        "label_map": label_map,
        "label_order": label_order,
    }


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_lightgbm_model(
    model_dir: Path | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the trained LightGBM Booster and associated metadata.

    Returns:
        booster, metadata_dict
    """
    if lgb is None:
        raise RuntimeError(
            "lightgbm is not installed in this environment. "
            "Install `lightgbm` to use the intraday model."
        )

    md = model_dir or _model_dir()
    model_path = md / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found at {model_path}")

    booster = lgb.Booster(model_file=str(model_path))
    meta = load_lightgbm_metadata(md)
    log(
        f"[lightgbm_intraday] Loaded model from {model_path} "
        f"with {len(meta.get('feature_map') or [])} features."
    )
    return booster, meta


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------
def _to_matrix(features, feature_map: Iterable[str] | None) -> np.ndarray:
    """
    Convert input features to a 2D numpy array, respecting feature order.
    """
    if isinstance(features, pd.DataFrame):
        if feature_map:
            df = features.reindex(columns=list(feature_map))
        else:
            df = features
        X = df.to_numpy(dtype=float)
    else:
        X = np.asarray(features, dtype=float)

    # Clean up infinities / NaNs
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def predict_proba(features, model_dir: Path | None = None) -> np.ndarray:
    """
    Run probability predictions for a 2D feature matrix.

    Args
    ----
    features:
        • pandas DataFrame with feature columns
        • or numpy array of shape (n_samples, n_features)

    Returns
    -------
    probs : np.ndarray of shape (n_samples, n_classes)
    """
    booster, meta = load_lightgbm_model(model_dir)
    feature_map = meta.get("feature_map") or []
    X = _to_matrix(features, feature_map or None)

    raw = booster.predict(X)
    raw = np.asarray(raw)
    # LightGBM returns either (n_samples,) for binary or (n_samples, n_class)
    if raw.ndim == 1:
        raw = np.vstack([1.0 - raw, raw]).T

    # Softmax for safety
    raw = raw.astype("float64")
    raw -= raw.max(axis=1, keepdims=True)
    exp = np.exp(raw)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.astype("float32")


def predict_class(
    features,
    model_dir: Path | None = None,
) -> Sequence[str]:
    """
    Convenience wrapper returning a sequence of class labels.
    """
    probs = predict_proba(features, model_dir)
    meta = load_lightgbm_metadata(model_dir)
    order = meta.get("label_order") or DEFAULT_LABEL_ORDER

    idx = probs.argmax(axis=1)
    labels = []
    for i in idx:
        i_int = int(i)
        if 0 <= i_int < len(order):
            labels.append(order[i_int])
        else:
            labels.append("HOLD")
    return labels
