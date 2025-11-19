# dt_backend/ml/train_lstm_intraday.py
"""
Train an intraday LSTM model on sequence datasets produced by
dt_backend.historical_replay.sequence_builder.

Assumptions:
    • Each parquet in: DT_PATHS["dtml_data"]/intraday/sequences/<tag>/
      contains one row with:
          X: np.ndarray of shape [N, seq_len, D]
          y: np.ndarray of shape [N, L]
          features: List[str]

    • We treat ALL L label dimensions as regression targets (MSE).
      (If you want multi-task regression + classification, we can extend.)

Usage:
    python -m dt_backend.ml.train_lstm_intraday
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
class LSTMTrainConfig:
    tag: str = "default"        # sequences tag (subfolder name)
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    train_frac: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _sequences_dir(cfg: LSTMTrainConfig) -> Path:
    root = Path(DT_PATHS.get("dtml_data", "ml_data_dt"))
    d = root / "intraday" / "sequences" / cfg.tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_sequence_dataset(cfg: LSTMTrainConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load & concatenate all symbol sequence files into one dataset."""
    seq_dir = _sequences_dir(cfg)
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

    log(f"[train_lstm] Loaded sequences: X={X_all.shape}, y={y_all.shape} from {len(files)} files")
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
# Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, output_dim]
        """
        out, (h_n, c_n) = self.lstm(x)  # out: [B, T, H], h_n: [L, B, H]
        last_hidden = h_n[-1]           # [B, H]
        return self.head(last_hidden)


# -----------------------------
# Training loop
# -----------------------------
def train_lstm_intraday(cfg: LSTMTrainConfig | None = None) -> dict:
    cfg = cfg or LSTMTrainConfig()

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

    model = LSTMModel(
        input_dim=input_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_dim=output_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    log(
        f"[train_lstm] Start training on device={device}, "
        f"input_dim={input_dim}, output_dim={output_dim}, N={N}"
    )

    best_val_loss = math.inf

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

        log(f"[train_lstm] epoch={epoch}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_lstm_model(model, features, cfg)

    return {
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "n_samples": N,
    }


def _save_lstm_model(model: nn.Module, features: List[str], cfg: LSTMTrainConfig) -> Path:
    root = Path(DT_PATHS.get("dtml_data", "ml_data_dt"))
    model_dir = root / "intraday" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"lstm_intraday_{cfg.tag}.pt"
    meta_path = model_dir / f"lstm_intraday_{cfg.tag}_meta.json"

    torch.save(model.state_dict(), model_path)

    import json
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": cfg.tag,
                "features": features,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"[train_lstm] ✅ saved model → {model_path}")
    return model_path


def main() -> None:
    train_lstm_intraday()


if __name__ == "__main__":
    main()
