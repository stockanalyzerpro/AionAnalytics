
"""Runtime intraday model loader + scorer (Option D ready)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models import LABEL_ORDER, LABEL2ID, ID2LABEL, get_model_dir
from dt_backend.models.ensemble.intraday_hybrid_ensemble import (
    EnsembleConfig,
    IntradayHybridEnsemble,
)

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


@dataclass
class LoadedModels:
    lgb: Optional[lgb.Booster]
    lgb_features: Optional[list[str]]
    lstm: Any | None
    transf: Any | None
    ensemble_cfg: EnsembleConfig


# ----- LightGBM loader -----

def _load_lgbm() -> Tuple[Optional[lgb.Booster], Optional[list[str]]]:
    model_dir = get_model_dir("lightgbm")
    model_path = model_dir / "model.txt"
    fmap_path = model_dir / "feature_map.json"

    if not model_path.exists():
        log(f"[ai_model_intraday] ⚠️ LightGBM model not found at {model_path}")
        return None, None

    booster = lgb.Booster(model_file=str(model_path))
    features: Optional[list[str]] = None
    if fmap_path.exists():
        with fmap_path.open("r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        features = booster.feature_name()

    log("[ai_model_intraday] ✅ Loaded LightGBM intraday model.")
    return booster, features


# ----- Optional deep models -----

def _load_lstm() -> Any | None:
    try:
        from dt_backend.models.lstm_intraday import load_lstm_intraday  # type: ignore
    except Exception:
        log("[ai_model_intraday] ℹ️ LSTM intraday module not available.")
        return None
    try:
        model = load_lstm_intraday()
        log("[ai_model_intraday] ✅ Loaded LSTM intraday model.")
        return model
    except Exception as e:
        log(f"[ai_model_intraday] ⚠️ Failed to load LSTM intraday model: {e}")
        return None


def _load_transformer() -> Any | None:
    try:
        from dt_backend.models.transformer_intraday import load_transformer_intraday  # type: ignore
    except Exception:
        log("[ai_model_intraday] ℹ️ Transformer intraday module not available.")
        return None
    try:
        model = load_transformer_intraday()
        log("[ai_model_intraday] ✅ Loaded Transformer intraday model.")
        return model
    except Exception as e:
        log(f"[ai_model_intraday] ⚠️ Failed to load Transformer intraday model: {e}")
        return None


# ----- Public loader -----

def load_intraday_models() -> LoadedModels:
    lgb_model, lgb_feats = _load_lgbm()
    lstm_model = _load_lstm()
    transf_model = _load_transformer()

    cfg = EnsembleConfig.load()

    if lgb_model is None and lstm_model is None and transf_model is None:
        log("[ai_model_intraday] ❌ No intraday models available.")

    if lgb_model is not None:
        log("[ai_model_intraday] 🔗 LightGBM active.")
    if lstm_model is not None:
        log("[ai_model_intraday] 🔗 LSTM active.")
    if transf_model is not None:
        log("[ai_model_intraday] 🔗 Transformer active.")

    return LoadedModels(
        lgb=lgb_model,
        lgb_features=lgb_feats,
        lstm=lstm_model,
        transf=transf_model,
        ensemble_cfg=cfg,
    )


# ----- Scoring helpers -----

def _predict_lgbm_proba(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
) -> np.ndarray:
    X_local = X.copy()
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X_local.columns]
        if missing:
            for c in missing:
                X_local[c] = 0.0
        X_local = X_local[feature_names]
    raw = booster.predict(X_local.values)
    return np.asarray(raw, dtype=float)


def score_intraday_batch(
    features: pd.DataFrame,
    models: Optional[LoadedModels] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Score a batch of intraday features.

    Args:
        features: DataFrame with one row per symbol / sample.
        models: optional pre-loaded models; if None, they are loaded on demand.

    Returns:
        proba_df: DataFrame with columns LABEL_ORDER.
        label_series: Series of predicted label strings.
    """
    if models is None:
        models = load_intraday_models()

    if models.lgb is None and models.lstm is None and models.transf is None:
        raise RuntimeError("No intraday models available for scoring.")

    X = features.copy()

    p_lgb = None
    if models.lgb is not None:
        p_lgb = _predict_lgbm_proba(models.lgb, X, feature_names=models.lgb_features)

    p_lstm = None
    p_transf = None

    if models.lstm is not None:
        try:
            p_lstm = models.lstm.predict_proba(X)  # type: ignore[attr-defined]
        except Exception as e:
            log(f"[ai_model_intraday] ⚠️ LSTM.predict_proba failed: {e}")
            p_lstm = None

    if models.transf is not None:
        try:
            p_transf = models.transf.predict_proba(X)  # type: ignore[attr-defined]
        except Exception as e:
            log(f"[ai_model_intraday] ⚠️ Transformer.predict_proba failed: {e}")
            p_transf = None

    active = [p for p in (p_lgb, p_lstm, p_transf) if p is not None]
    if not active:
        raise RuntimeError("No valid probability outputs from intraday models.")

    if len(active) == 1:
        proba = active[0]
    else:
        ensemble = IntradayHybridEnsemble(models.ensemble_cfg)
        proba = ensemble.predict_proba(p_lgb=p_lgb, p_lstm=p_lstm, p_transf=p_transf)

    proba_df = pd.DataFrame(proba, index=features.index, columns=LABEL_ORDER)
    idx = np.argmax(proba, axis=1)
    labels = [LABEL_ORDER[int(i)] for i in idx]
    label_series = pd.Series(labels, index=features.index, name="label_pred")
    return proba_df, label_series


if __name__ == "__main__":
    # Tiny smoke test with random features
    import numpy as np
    n = 5
    dummy = pd.DataFrame({f"f{i}": np.random.randn(n) for i in range(10)})
    try:
        proba_df, labels = score_intraday_batch(dummy)
        log(f"[ai_model_intraday] Demo OK — got {len(proba_df)} rows.")
    except Exception as e:
        log(f"[ai_model_intraday] Demo failed: {e}")
