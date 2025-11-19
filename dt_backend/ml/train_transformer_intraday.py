# dt_backend/ml/train_transformer_intraday.py
"""
Train an intraday Transformer encoder on sequence datasets produced by
dt_backend.historical_replay.sequence_builder.

Same dataset expectations as train_lstm_intraday.py.

Usage:
    python -m dt_backend.ml.train_transformer_intraday
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dt_backend.core.config_dt import DT_PATHS

try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


# -----------------------------
# Data utilities
# -----------------------------
@dataclass
class TransformerTrainConfig:
    tag: str = "default"
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    train_frac: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _seq_dir(cfg: TransformerTrainConfig) -> Path:
    root = Path(DT_PATHS.get("dtml_data", "ml_data_dt"))
    d = root / "intraday" / "sequences" / cfg.tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_sequence_dataset(cfg: TransformerTrainConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    seq_dir = _seq_dir(cfg)
    files = sorted(seq_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sequence parquet files found in {seq_dir}")

    X_list = []
    y_list = []
    feature_list: List[str] | None = None

    for fp in files:
        df = pd.read_parquet(fp)
        row = df.iloc[0]
        X = np.array(row["X"], dtype="float32")   # [N, T, D]
        y = np.array(row["y"], dtype="float32")   # [N, L]
        X_list.append(X)
        y_list.append(y)

        if feature_list is None:
            feature_list = list(row["features"])

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if feature_list is None:
        feature_list = []

    log(f"[train_tx] Loaded sequences: X={X_all.shape}, y={y_all.shape} from {len(files)} files")
    return X_all, y_all, feature_list


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),  # [T, D]
            torch.from_numpy(self.y[idx]),  # [L]
        )


# -----------------------------
# Positional encoding + model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        x = self.input_proj(x)
        x = self.positional(x)
        out = self.encoder(x)          # [B, T, d_model]
        last = out[:, -1, :]           # last timestep
        return self.head(last)


# -----------------------------
# Training loop
# -----------------------------
def train_transformer_intraday(cfg: TransformerTrainConfig | None = None) -> dict:
    cfg = cfg or TransformerTrainConfig()

    X, y, features = load_sequence_dataset(cfg)
    N = X.shape[0]
    idx = int(cfg.train_frac * N)
    X_train, X_val = X[:idx], X[idx:]
    y_train, y_val = y[:idx], y[idx:]

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device(cfg.device)
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]

    model = TransformerModel(
        input_dim=input_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        output_dim=output_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    log(
        f"[train_tx] Start training on device={device}, "
        f"input_dim={input_dim}, output_dim={output_dim}, N={N}"
    )

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss_sum += loss.item() * Xb.size(0)
        val_loss = val_loss_sum / max(1, len(val_ds))

        log(f"[train_tx] epoch={epoch}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            _save_transformer_model(model, features, cfg)

    return {
        "best_val_loss": best_val,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "n_samples": N,
    }


def _save_transformer_model(model: nn.Module, features: List[str], cfg: TransformerTrainConfig) -> Path:
    root = Path(DT_PATHS.get("dtml_data", "ml_data_dt"))
    model_dir = root / "intraday" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"transformer_intraday_{cfg.tag}.pt"
    meta_path = model_dir / f"transformer_intraday_{cfg.tag}_meta.json"

    torch.save(model.state_dict(), model_path)

    import json
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": cfg.tag,
                "features": features,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_layers": cfg.num_layers,
                "dim_feedforward": cfg.dim_feedforward,
                "dropout": cfg.dropout,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"[train_tx] ✅ saved model → {model_path}")
    return model_path


def main() -> None:
    train_transformer_intraday()


if __name__ == "__main__":
    main()
