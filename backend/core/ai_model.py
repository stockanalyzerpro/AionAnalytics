"""
ai_model.py — v5.1
AION Analytics — Hybrid Model Engine (LightGBM + LSTM + Transformer + Optuna)

Enhancements over v5.0:
    • Full Optuna hyperparameter tuning per horizon.
    • Backward compatible: if Optuna not installed → training works normally.
    • Tune + Train integrated into the standard nightly flow.
    • Cleaner horizon loop logging + error handling.
    • All deep model code preserved exactly as v5.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ==========================================================
# Imports from backend
# ==========================================================

from backend.config import PATHS, TIMEZONE
from backend.data_pipeline import log, _read_rolling

# ==========================================================
# LightGBM / RF
# ==========================================================

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

# ==========================================================
# Torch (LSTM + Transformer)
# ==========================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ==========================================================
# Optuna
# ==========================================================

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# ==========================================================
# Paths
# ==========================================================

ML_DATA_ROOT: Path = PATHS.get("ml_data", Path("ml_data"))
DATASET_DIR: Path = ML_DATA_ROOT / "nightly" / "dataset"
DATASET_FILE: Path = DATASET_DIR / "training_data_daily.parquet"
FEATURE_LIST_FILE: Path = DATASET_DIR / "feature_list_daily.json"

MODEL_ROOT: Path = PATHS.get("ml_models", ML_DATA_ROOT / "nightly" / "models")
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1d", "3d", "1w", "2w", "4w", "13w", "26w", "52w"]


# ==========================================================
# PATH HELPERS
# ==========================================================

def _model_path(horizon: str) -> Path:
    return MODEL_ROOT / f"model_{horizon}.pkl"


def _lstm_path(horizon: str) -> Path:
    return MODEL_ROOT / f"lstm_{horizon}.pt"


def _transformer_path(horizon: str) -> Path:
    return MODEL_ROOT / f"transformer_{horizon}.pt"


# ==========================================================
# DATASET HELPERS
# ==========================================================

def _load_feature_list() -> Dict[str, Any]:
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(f"Feature list missing at {FEATURE_LIST_FILE}")
    return json.loads(FEATURE_LIST_FILE.read_text(encoding="utf-8"))


def _load_dataset(dataset_name: str | None = None):
    if dataset_name and dataset_name != DATASET_FILE.name:
        df_path = DATASET_DIR / dataset_name
    else:
        df_path = DATASET_FILE

    if not df_path.exists():
        raise FileNotFoundError(f"Dataset missing: {df_path}")

    df = pd.read_parquet(df_path)
    feat_info = _load_feature_list()

    return df, feat_info


# ==========================================================
# Classic classifier factory
# ==========================================================

def _make_classifier(params: Optional[dict] = None):
    """Return LightGBM or RF classifier using tuned params."""
    if HAS_LGBM:
        defaults = dict(
            objective="multiclass",
            num_class=3,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        if params:
            defaults.update(params)
        return lgb.LGBMClassifier(**defaults)

    # RF fallback
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300) if params else 300,
        max_depth=params.get("max_depth", None) if params else None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )


def _coerce_direction_labels(y: pd.Series) -> pd.Series:
    """Ensure direction labels are in -1,0,+1 format."""
    vals = y.values
    uniq = set(np.unique(vals))

    if uniq.issubset({-1, 0, 1}):
        return y.astype(int)
    if uniq.issubset({0, 1}):
        return pd.Series(np.where(vals > 0, 1, -1), index=y.index, name=y.name)

    mapped = np.sign(vals)
    mapped[mapped == 0] = 0
    return pd.Series(mapped.astype(int), index=y.index, name=y.name)


# ==========================================================
# TORCH MODELS: LSTM + TRANSFORMER
# ==========================================================

if HAS_TORCH:

    class LSTMHead(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h, _ = self.lstm(x)
            return self.fc(h[:, -1, :])

    class TransformerHead(nn.Module):
        def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4):
            super().__init__()
            self.embed = nn.Linear(input_dim, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.fc = nn.Linear(d_model, 3)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = self.embed(x)
            h = self.enc(h)
            return self.fc(h[:, -1, :])

    def _torch_train(model, X, y, epochs: int = 4, lr=1e-3, batch=128, device="cpu"):
        label_map = {-1: 0, 0: 1, 1: 2}
        y_idx = np.vectorize(label_map.get)(y)

        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_idx[:split], y_idx[split:]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val   = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val   = torch.tensor(y_val, dtype=torch.long).to(device)

        model = model.to(device)
        optim_ = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        def batches(Xb, yb):
            for i in range(0, len(Xb), batch):
                yield Xb[i:i+batch], yb[i:i+batch]

        model.train()
        for _ in range(epochs):
            for xb, yb in batches(X_train, y_train):
                optim_.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim_.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            preds = logits.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        return acc

    def _torch_predict(model, row, device="cpu"):
        label_map = {0: -1, 1: 0, 2: 1}
        model = model.to(device)
        model.eval()
        x = torch.tensor(row.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        cls = int(np.argmax(probs))
        p_down, p_flat, p_up = probs
        conf = float(max(probs))
        score = float(p_up - p_down)
        return label_map[cls], conf, score


# ==========================================================
# OPTUNA TUNING
# ==========================================================

def _tune_lightgbm(X: np.ndarray, y: np.ndarray, horizon: str, n_trials: int = 20):
    """Run Optuna hyperparameter tuning for one horizon."""
    if not (HAS_OPTUNA and HAS_LGBM):
        return {}

    log(f"[ai_model] 🔍 Optuna tuning horizon={horizon}, trials={n_trials}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        clf = _make_classifier(params)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        return accuracy_score(y_val, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best["num_leaves"] = int(best["num_leaves"])
    best["max_depth"] = int(best["max_depth"])
    best["n_estimators"] = int(best["n_estimators"])

    log(f"[ai_model] 🎯 Best Optuna params for {horizon}: {best}")
    return best


# ==========================================================
# TRAINING: TREE + LSTM + TRANSFORMER + OPTUNA
# ==========================================================

def train_model(dataset_name: str = "training_data_daily.parquet", use_optuna=True, n_trials=20):
    """
    Train all horizons:
        • Tree model (LightGBM/RF)
        • LSTM head (optional)
        • Transformer head (optional)
        • Optional: Optuna tuning per horizon
    """
    log(f"[ai_model] 🧠 Training hybrid models (optuna={use_optuna})")

    df, feat_info = _load_dataset(dataset_name)
    feature_cols = feat_info.get("feature_columns", [])
    target_cols  = feat_info.get("target_columns", [])
    id_cols      = feat_info.get("id_columns", [])

    if not feature_cols:
        log("[ai_model] ❌ No feature columns found.")
        return {"status": "error", "error": "no_features"}

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    summaries = {}

    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    for horizon in HORIZONS:
        tgt = f"target_dir_{horizon}"
        if tgt not in target_cols:
            continue

        y_raw = df[tgt].copy()
        y = _coerce_direction_labels(y_raw)

        if len(np.unique(y)) <= 1:
            summaries[horizon] = {"status": "skipped", "reason": "single_class"}
            continue

        horizon_summary = {"models": {}}

        # ---- Optuna search -----------------------------------------
        params = {}
        if use_optuna:
            try:
                params = _tune_lightgbm(X, y.values, horizon, n_trials=n_trials)
            except Exception as e:
                log(f"[ai_model] ⚠️ Optuna failed: {e}")

        # ---- Train tree model ---------------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y.values, test_size=0.2, random_state=42, stratify=y.values
        )

        try:
            clf = _make_classifier(params)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            acc = float(accuracy_score(y_val, pred))
            mp = _model_path(horizon)
            dump(clf, mp)
            horizon_summary["models"]["tree"] = {
                "status": "ok",
                "val_accuracy": acc,
                "params": params,
                "path": str(mp),
            }
        except Exception as e:
            horizon_summary["models"]["tree"] = {"status": "error", "error": str(e)}

        # ---- Train deep models --------------------------------------
        if HAS_TORCH:
            input_dim = X.shape[1]

            # LSTM
            try:
                lstm = LSTMHead(input_dim)
                acc_lstm = _torch_train(lstm, X, y.values, device=device)
                lp = _lstm_path(horizon)
                torch.save(lstm.state_dict(), lp)
                horizon_summary["models"]["lstm"] = {
                    "status": "ok",
                    "val_accuracy": acc_lstm,
                    "path": str(lp),
                }
            except Exception as e:
                horizon_summary["models"]["lstm"] = {"status": "error", "error": str(e)}

            # Transformer
            try:
                trans = TransformerHead(input_dim)
                acc_trans = _torch_train(trans, X, y.values, device=device)
                tp = _transformer_path(horizon)
                torch.save(trans.state_dict(), tp)
                horizon_summary["models"]["transformer"] = {
                    "status": "ok",
                    "val_accuracy": acc_trans,
                    "path": str(tp),
                }
            except Exception as e:
                horizon_summary["models"]["transformer"] = {"status": "error", "error": str(e)}

        summaries[horizon] = horizon_summary

    return {"status": "ok", "horizons": summaries}


def train_all_models(dataset_name="training_data_daily.parquet", use_optuna=True, n_trials=20):
    """Alias for nightly job."""
    return train_model(dataset_name, use_optuna, n_trials)


# ==========================================================
# PREDICTION
# ==========================================================

def _load_tree_models():
    models = {}
    for horizon in HORIZONS:
        p = _model_path(horizon)
        if p.exists():
            try:
                models[horizon] = load(p)
            except:
                pass
    return models


def _load_deep_models(input_dim):
    if not HAS_TORCH:
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep = {}

    for horizon in HORIZONS:
        lstm_m = None
        trans_m = None

        lp = _lstm_path(horizon)
        if lp.exists():
            lstm_m = LSTMHead(input_dim)
            lstm_m.load_state_dict(torch.load(lp, map_location=device))

        tp = _transformer_path(horizon)
        if tp.exists():
            trans_m = TransformerHead(input_dim)
            trans_m.load_state_dict(torch.load(tp, map_location=device))

        if lstm_m or trans_m:
            deep[horizon] = {"lstm": lstm_m, "transformer": trans_m}

    return deep


def predict_all(rolling: Optional[Dict[str, Any]] = None):
    if rolling is None:
        rolling = _read_rolling() or {}

    df, feat_info = _load_dataset(DATASET_FILE.name)
    feature_cols = feat_info.get("feature_columns", [])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "symbol" not in df.columns:
        return {}

    df_idx = df.set_index("symbol")

    X = df[feature_cols].to_numpy()
    input_dim = X.shape[1]

    tree_models = _load_tree_models()
    deep_models = _load_deep_models(input_dim)

    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    preds = {}

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        s = sym.upper()
        if s not in df_idx.index:
            continue

        x = df_idx.loc[s, feature_cols].to_numpy().reshape(1, -1)
        components = {}
        sym_res = {}

        for h, model in tree_models.items():
            try:
                # TREE
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    classes = model.classes_
                    class_p = dict(zip(classes, proba))
                    p_down = class_p.get(-1, 0.0)
                    p_up = class_p.get(1, 0.0)
                    label_tree = int(model.predict(x)[0])
                    conf_tree = proba[np.argmax(proba)]
                    score_tree = float(p_up - p_down)
                else:
                    label_tree = int(model.predict(x)[0])
                    conf_tree = 0.5
                    score_tree = float(label_tree)

                comp = {
                    "tree": {
                        "label": label_tree,
                        "confidence": conf_tree,
                        "score": score_tree,
                    }
                }

                # LSTM/TRANSFORMER
                lstm_label = lstm_conf = lstm_score = None
                trans_label = trans_conf = trans_score = None

                dm = deep_models.get(h, {})
                if HAS_TORCH:
                    if dm.get("lstm"):
                        lstm_label, lstm_conf, lstm_score = _torch_predict(dm["lstm"], x[0], device)
                        comp["lstm"] = {
                            "label": lstm_label,
                            "confidence": lstm_conf,
                            "score": lstm_score,
                        }
                    if dm.get("transformer"):
                        trans_label, trans_conf, trans_score = _torch_predict(dm["transformer"], x[0], device)
                        comp["transformer"] = {
                            "label": trans_label,
                            "confidence": trans_conf,
                            "score": trans_score,
                        }

                # HYBRID ENSEMBLE
                weights = []
                scores = []
                confs = []

                weights.append(0.5)
                scores.append(score_tree)
                confs.append(conf_tree)

                if lstm_score is not None:
                    weights.append(0.25)
                    scores.append(lstm_score)
                    confs.append(lstm_conf)
                if trans_score is not None:
                    weights.append(0.25)
                    scores.append(trans_score)
                    confs.append(trans_conf)

                w_sum = sum(weights) if weights else 1
                weights = [w / w_sum for w in weights]

                ensemble_score = float(sum(w * s for w, s in zip(weights, scores)))
                ensemble_conf = float(sum(w * c for w, c in zip(weights, confs)))

                if ensemble_score > 0.05:
                    ensemble_label = 1
                elif ensemble_score < -0.05:
                    ensemble_label = -1
                else:
                    ensemble_label = label_tree

                sym_res[h] = {
                    "label": ensemble_label,
                    "confidence": ensemble_conf,
                    "score": ensemble_score,
                    "components": comp,
                }

            except Exception as e:
                log(f"[ai_model] prediction failed for {s} horizon={h}: {e}")

        if sym_res:
            preds[s] = sym_res

    return preds


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    summary = train_model(use_optuna=True, n_trials=10)
    print(json.dumps(summary, indent=2))
