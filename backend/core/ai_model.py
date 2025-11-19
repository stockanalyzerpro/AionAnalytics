"""
ai_model.py — v4.0
AION Analytics — Nightly Multi-Horizon ML Models (Option A)

Trains one classifier per horizon using the dataset built by
backend/services/ml_data_builder.py (v4.0).

Dataset:
    ml_data/nightly/dataset/training_data_daily.parquet
    ml_data/nightly/dataset/feature_list_daily.json

Targets (from ml_data_builder):
    target_dir_1d, target_dir_3d, target_dir_1w, target_dir_2w,
    target_dir_4w, target_dir_13w, target_dir_26w, target_dir_52w

We train one model per horizon (multi-horizon, multi-model):
    model_1d.pkl, model_3d.pkl, ..., model_52w.pkl

Public API:
    train_model(dataset_name="training_data_daily.parquet") -> dict
    train_all_models(...) -> dict  (alias)
    predict_all(rolling: dict | None = None) -> dict[sym] = {horizon: {...}}

Prediction output (per symbol):
    {
      "1d": {
        "score": float,        # directional score in [-1,1]
        "confidence": float,   # probability of predicted class
        "label": int,          # -1, 0, +1
      },
      "3d": {...},
      ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log, _read_rolling

# Optional LightGBM
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ML_DATA_ROOT: Path = PATHS.get("ml_data", Path("ml_data"))
DATASET_DIR: Path = ML_DATA_ROOT / "nightly" / "dataset"
DATASET_FILE: Path = DATASET_DIR / "training_data_daily.parquet"
FEATURE_LIST_FILE: Path = DATASET_DIR / "feature_list_daily.json"

# Model directory (backend nightly models)
MODEL_ROOT: Path = PATHS.get("ml_models", ML_DATA_ROOT / "nightly" / "models")
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# Horizon configuration
HORIZONS = ["1d", "3d", "1w", "2w", "4w", "13w", "26w", "52w"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _model_path(horizon: str) -> Path:
    return MODEL_ROOT / f"model_{horizon}.pkl"


def _load_feature_list() -> Dict[str, Any]:
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(f"Feature list missing at {FEATURE_LIST_FILE}")
    js = json.loads(FEATURE_LIST_FILE.read_text(encoding="utf-8"))
    return js


def _load_dataset(dataset_name: str | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if dataset_name and dataset_name != DATASET_FILE.name:
        df_path = DATASET_DIR / dataset_name
    else:
        df_path = DATASET_FILE

    if not df_path.exists():
        raise FileNotFoundError(f"Dataset parquet missing at {df_path}")

    df = pd.read_parquet(df_path)
    feat_info = _load_feature_list()
    return df, feat_info


def _make_classifier() -> Any:
    """Return a new classifier instance."""
    if HAS_LGBM:
        # 3-class: -1, 0, +1
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    # Fallback
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )


def _coerce_direction_labels(y: pd.Series) -> pd.Series:
    """
    Ensure labels are in {-1, 0, +1}.
    Some legacy data may use 0/1 or 1/2/3 — normalize to -1/0/+1.
    """
    vals = y.values
    uniq = np.unique(vals)

    # If already in {-1, 0, 1}, leave it
    if set(uniq).issubset({-1, 0, 1}):
        return y.astype(int)

    # If binary {0,1}, treat 0→-1, 1→+1
    if set(uniq).issubset({0, 1}):
        mapped = np.where(vals > 0, 1, -1)
        return pd.Series(mapped, index=y.index, name=y.name)

    # Fallback: sign of value
    mapped = np.sign(vals)
    mapped[mapped == 0] = 0
    return pd.Series(mapped.astype(int), index=y.index, name=y.name)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(dataset_name: str = "training_data_daily.parquet") -> Dict[str, Any]:
    """
    Train one model per horizon using the nightly dataset.
    Returns summary metrics for each horizon.
    """
    log(f"[ai_model] 🧠 Training models from {dataset_name}...")
    try:
        df, feat_info = _load_dataset(dataset_name)
    except Exception as e:
        log(f"[ai_model] ❌ Failed to load dataset: {e}")
        return {"status": "error", "error": str(e)}

    id_cols = feat_info.get("id_columns", ["symbol", "name", "sector"])
    feature_cols = feat_info.get("feature_columns", [])
    target_cols = feat_info.get("target_columns", [])

    if not feature_cols:
        log("[ai_model] ⚠️ No feature columns found.")
        return {"status": "error", "error": "no_feature_columns"}

    X = df[feature_cols].copy()
    # Simple numeric imputation
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    summaries: Dict[str, Any] = {}
    any_success = False

    for horizon in HORIZONS:
        target_col = f"target_dir_{horizon}"
        if target_col not in target_cols:
            log(f"[ai_model] ⚠️ Missing target column {target_col}, skipping horizon {horizon}.")
            continue

        y_raw = df[target_col].copy()
        y = _coerce_direction_labels(y_raw)

        if y.nunique() <= 1:
            log(f"[ai_model] ⚠️ Target {target_col} has only one class, skipping.")
            summaries[horizon] = {"status": "skipped", "reason": "single_class"}
            continue

        # Train/test split for basic sanity metrics
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = _make_classifier()
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            log(f"[ai_model] ⚠️ Training failed for horizon {horizon}: {e}")
            summaries[horizon] = {"status": "error", "error": str(e)}
            continue

        # Basic validation accuracy
        try:
            y_pred = clf.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
        except Exception:
            acc = -1.0

        # Save model
        mp = _model_path(horizon)
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
            dump(clf, mp)
            log(f"[ai_model] 💾 Saved model for {horizon} → {mp}")
            any_success = True
            summaries[horizon] = {
                "status": "ok",
                "val_accuracy": acc,
                "classes": sorted(int(c) for c in np.unique(y)),
                "model_path": str(mp),
            }
        except Exception as e:
            log(f"[ai_model] ⚠️ Failed to save model for {horizon}: {e}")
            summaries[horizon] = {"status": "error", "error": str(e)}

    overall_status = "ok" if any_success else "error"
    result = {"status": overall_status, "horizons": summaries}
    log(f"[ai_model] ✅ Training complete. Status={overall_status}")
    return result


def train_all_models(dataset_name: str = "training_data_daily.parquet") -> Dict[str, Any]:
    """
    Backwards-compatible alias for train_model().
    """
    return train_model(dataset_name=dataset_name)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def _load_models() -> Dict[str, Any]:
    """Load all available horizon models from disk."""
    models: Dict[str, Any] = {}
    for horizon in HORIZONS:
        mp = _model_path(horizon)
        if not mp.exists():
            continue
        try:
            models[horizon] = load(mp)
        except Exception as e:
            log(f"[ai_model] ⚠️ Failed to load model {mp}: {e}")
    return models


def _load_features_for_prediction() -> Tuple[pd.DataFrame, List[str]]:
    """
    For nightly predictions we reuse the latest training dataset features.
    This guarantees feature alignment between train and predict.
    """
    df, feat_info = _load_dataset(DATASET_FILE.name)
    feature_cols = feat_info.get("feature_columns", [])
    if not feature_cols:
        raise RuntimeError("No feature_columns in feature_list_daily.json")
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, feature_cols


def predict_all(rolling: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate predictions for all symbols in Rolling for each trained horizon.

    Returns:
        predictions[symbol][horizon] = {
            "score": float in [-1,1],
            "confidence": float in [0,1],
            "label": int (-1,0,1),
        }
    """
    if rolling is None:
        rolling = _read_rolling() or {}

    models = _load_models()
    if not models:
        log("[ai_model] ⚠️ No models found for prediction.")
        return {}

    try:
        df, feature_cols = _load_features_for_prediction()
    except Exception as e:
        log(f"[ai_model] ❌ Failed to load features for prediction: {e}")
        return {}

    if "symbol" not in df.columns:
        log("[ai_model] ❌ Dataset has no 'symbol' column for joining.")
        return {}

    # Index by symbol for quick lookup
    df_idx = df.set_index("symbol")

    preds: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        sym_u = sym.upper()
        if sym_u not in df_idx.index:
            continue

        x_row = df_idx.loc[sym_u, feature_cols]
        # If only one row, ensure proper shape (1, n_features)
        if isinstance(x_row, pd.Series):
            X_sym = x_row.values.reshape(1, -1)
        else:
            X_sym = x_row.to_numpy().reshape(1, -1)

        sym_res: Dict[str, Dict[str, Any]] = {}

        for horizon, model in models.items():
            try:
                # Some models may not support predict_proba (unlikely)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_sym)[0]
                    classes = getattr(model, "classes_", np.array([-1, 0, 1]))
                    # Map proba to our canonical order (-1, 0, 1)
                    class_to_p = {int(c): float(p) for c, p in zip(classes, proba)}
                    p_down = class_to_p.get(-1, 0.0)
                    p_flat = class_to_p.get(0, 0.0)
                    p_up = class_to_p.get(1, 0.0)

                    label = int(model.predict(X_sym)[0])
                    # confidence = probability of the predicted class
                    conf = {
                        -1: p_down,
                        0: p_flat,
                        1: p_up,
                    }.get(label, 0.0)
                    # score: difference between up and down, signed by label
                    score = float((p_up - p_down))
                else:
                    # Fallback: no proba, just label
                    label = int(model.predict(X_sym)[0])
                    conf = 0.5
                    score = float(label)

                sym_res[horizon] = {
                    "score": float(score),
                    "confidence": float(conf),
                    "label": label,
                }

            except Exception as e:
                log(f"[ai_model] ⚠️ Prediction failed for {sym_u} horizon {horizon}: {e}")
                continue

        if sym_res:
            preds[sym_u] = sym_res

    log(f"[ai_model] 🤖 Generated predictions for {len(preds)} symbols across {len(models)} horizons.")
    return preds


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Simple manual test
    summary = train_model()
    print(json.dumps(summary, indent=2))
    rolling = _read_rolling() or {}
    out = predict_all(rolling)
    print(f"Predictions for {len(out)} symbols.")
