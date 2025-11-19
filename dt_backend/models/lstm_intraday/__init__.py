"""
LSTM-based intraday classifier for AION.

This is a **sequence model** that consumes a rolling window of recent
intraday features per symbol and outputs class probabilities over
{SELL, HOLD, BUY}.

It is designed to share (or closely mirror) the feature set used by
the LightGBM intraday model, but arranged as:

    X_seq: shape (batch, time, features)

Training jobs are responsible for:
  • writing `config.json` with the LSTMConfig used
  • writing `model.pt` with the trained weights
  • writing `feature_map.json` for input feature ordering
  • optionally writing `label_map.json` for custom class ordering
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Sequence

import numpy as np
import torch
from torch import nn

try:
    from dt_backend.dt_logger import dt_log as log
except Exception:  # pragma: no cover
    def log(msg: str) -> None:  # type: ignore[no-redef]
        print(msg, flush=True)

try:
    from dt_backend.config_dt import DT_PATHS
except Exception:  # pragma: no cover
    DT_PATHS: Dict[str, Path] = {
        "dtmodels": Path("ml_data_dt") / "models"
    }

DEFAULT_LABEL_ORDER = ["SELL", "HOLD", "BUY"]
DEFAULT_ID2LABEL = {i: c for i, c in enumerate(DEFAULT_LABEL_ORDER)}


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True
    num_classes: int = 3
    max_seq_len: int = 128

    @classmethod
    def from_json(cls, path: Path) -> "LSTMConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


class LSTMIntradayModel(nn.Module):
    """
    Standard stacked LSTM classifier with a small MLP head.

    Input:
        x: (batch, time, features)
    Output:
        logits: (batch, num_classes)
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        lstm_out = config.hidden_size * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, lstm_out),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_out, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        out, _ = self.lstm(x)
        # Take last timestep representation
        last = out[:, -1, :]
        logits = self.head(last)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Convenience wrapper for numpy inference.

        x: shape (batch, time, features)
        """
        self.eval()
        t = torch.from_numpy(x).float().to(device)
        logits = self.forward(t)
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()


# ------------------------------------------------------------------
# Path + metadata helpers
# ------------------------------------------------------------------
def _model_dir() -> Path:
    base = Path(DT_PATHS["dtmodels"])  # type: ignore[index]
    return base / "intraday" / "lstm"


def _safe_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_lstm_config(model_dir: Path | None = None) -> LSTMConfig:
    """
    Load LSTMConfig from config.json. This must have been written
    by the training job; we do not silently create a default here,
    to avoid mismatches between training and inference.
    """
    md = model_dir or _model_dir()
    cfg_path = md / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"LSTM config.json not found at {cfg_path}")
    cfg = LSTMConfig.from_json(cfg_path)
    return cfg


def _load_feature_map(md: Path) -> Sequence[str]:
    data = _safe_json(md / "feature_map.json", [])
    if isinstance(data, dict) and "features" in data:
        return list(data["features"])
    if isinstance(data, list):
        return list(data)
    return []


def _load_label_order(md: Path) -> Sequence[str]:
    # Try explicit label_map.json
    label_raw = _safe_json(md / "label_map.json", DEFAULT_ID2LABEL)
    if isinstance(label_raw, dict):
        try:
            ids = sorted(int(k) for k in label_raw.keys())
            return [str(label_raw[str(i)]) for i in ids]
        except Exception:
            return list(label_raw.values())
    if isinstance(label_raw, list):
        return [str(x) for x in label_raw]
    return DEFAULT_LABEL_ORDER


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_lstm_model(
    model_dir: Path | None = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[LSTMIntradayModel, Dict[str, Any]]:
    """
    Load the trained LSTMIntradayModel and associated metadata.

    Returns:
        model, metadata_dict
    """
    md = model_dir or _model_dir()
    cfg = load_lstm_config(md)
    model = LSTMIntradayModel(cfg)

    model_path = md / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model.pt not found at {model_path}")

    state = torch.load(model_path, map_location=map_location)
    # Support both raw state_dict and {"model_state": ...}
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    feature_map = _load_feature_map(md)
    label_order = list(_load_label_order(md))

    meta = {
        "config": cfg.__dict__,
        "feature_map": feature_map,
        "label_order": label_order,
    }

    log(
        f"[lstm_intraday] Loaded model from {model_path} "
        f"(features={len(feature_map)}, labels={label_order})."
    )
    return model, meta


# ------------------------------------------------------------------
# Top-level inference helpers
# ------------------------------------------------------------------
@torch.no_grad()
def lstm_predict_proba(
    X_seq: np.ndarray,
    model_dir: Path | None = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, Sequence[str]]:
    """
    Run probability predictions for a batch of sequences.

    Args
    ----
    X_seq:
        numpy array of shape (batch, time, features) matching the
        training configuration (input_size, max_seq_len).
    model_dir:
        Optional override of the model directory.
    device:
        "cpu" or "cuda" (if available).

    Returns
    -------
    probs : np.ndarray, shape (batch, num_classes)
    label_order : sequence of label strings, e.g. ["SELL","HOLD","BUY"]
    """
    model, meta = load_lstm_model(model_dir, map_location=device)
    probs = model.predict_proba(np.asarray(X_seq, dtype="float32"), device=device)
    label_order = meta.get("label_order") or DEFAULT_LABEL_ORDER
    return probs, label_order


def lstm_predict_class(
    X_seq: np.ndarray,
    model_dir: Path | None = None,
    device: str = "cpu",
) -> Sequence[str]:
    """
    Convenience wrapper returning just class labels.
    """
    probs, label_order = lstm_predict_proba(X_seq, model_dir=model_dir, device=device)
    idx = probs.argmax(axis=1)
    labels: list[str] = []
    for i in idx:
        i_int = int(i)
        if 0 <= i_int < len(label_order):
            labels.append(label_order[i_int])
        else:
            labels.append("HOLD")
    return labels
