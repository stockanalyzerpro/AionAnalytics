"""
Transformer-based intraday classifier for AION.

This is a **sequence model** that consumes a window of recent
intraday features per symbol and outputs class probabilities over
{SELL, HOLD, BUY}.

Like the LSTM intraday model, this expects inputs of shape:

    X_seq: (batch, time, features)

But uses a Transformer encoder with positional encoding, which tends
to capture longer-range temporal structure and more "pattern-ish"
behaviour that feels closer to how a human reads a tape.

Training jobs are responsible for:
  • writing `config.json` with the TransformerConfig used
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
class TransformerConfig:
    input_size: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    max_seq_len: int = 128

    @classmethod
    def from_json(cls, path: Path) -> "TransformerConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Transformer-style).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerIntradayModel(nn.Module):
    """
    Transformer encoder classifier for intraday sequences.

    Input:
        x: (batch, time, features)
    Output:
        logits: (batch, num_classes)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        # CLS-style pooling: mean over time (could swap to last-token pooling)
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
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
    return base / "intraday" / "transformer"


def _safe_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_transformer_config(model_dir: Path | None = None) -> TransformerConfig:
    """
    Load TransformerConfig from config.json. This must have been written
    by the training job; we do not silently create a default here.
    """
    md = model_dir or _model_dir()
    cfg_path = md / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Transformer config.json not found at {cfg_path}")
    cfg = TransformerConfig.from_json(cfg_path)
    return cfg


def _load_feature_map(md: Path) -> Sequence[str]:
    data = _safe_json(md / "feature_map.json", [])
    if isinstance(data, dict) and "features" in data:
        return list(data["features"])
    if isinstance(data, list):
        return list(data)
    return []


def _load_label_order(md: Path) -> Sequence[str]:
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
def load_transformer_model(
    model_dir: Path | None = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[TransformerIntradayModel, Dict[str, Any]]:
    """
    Load the trained TransformerIntradayModel and associated metadata.

    Returns:
        model, metadata_dict
    """
    md = model_dir or _model_dir()
    cfg = load_transformer_config(md)
    model = TransformerIntradayModel(cfg)

    model_path = md / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Transformer model.pt not found at {model_path}")

    state = torch.load(model_path, map_location=map_location)
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
        f"[transformer_intraday] Loaded model from {model_path} "
        f"(features={len(feature_map)}, labels={label_order})."
    )
    return model, meta


# ------------------------------------------------------------------
# Top-level inference helpers
# ------------------------------------------------------------------
@torch.no_grad()
def transformer_predict_proba(
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
    model, meta = load_transformer_model(model_dir, map_location=device)
    probs = model.predict_proba(np.asarray(X_seq, dtype="float32"), device=device)
    label_order = meta.get("label_order") or DEFAULT_LABEL_ORDER
    return probs, label_order


def transformer_predict_class(
    X_seq: np.ndarray,
    model_dir: Path | None = None,
    device: str = "cpu",
) -> Sequence[str]:
    """
    Convenience wrapper returning just class labels.
    """
    probs, label_order = transformer_predict_proba(
        X_seq, model_dir=model_dir, device=device
    )
    idx = probs.argmax(axis=1)
    labels: list[str] = []
    for i in idx:
        i_int = int(i)
        if 0 <= i_int < len(label_order):
            labels.append(label_order[i_int])
        else:
            labels.append("HOLD")
    return labels
