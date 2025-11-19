# dt_backend/historical_replay/sequence_builder.py
"""
Advanced intraday sequence builder for LSTM / Transformer models.

Supports:
    • Indicator-augmented sequences
    • Multi-horizon labels
    • Multi-task outputs (classification + regression)
    • Normalization (zscore / minmax / robust)
    • Overlapping or stride-based sequence generation
    • Sequence padding or trimming
    • Parquet dataset export
    • Integration with DT_PATHS

This module is designed for AION’s deep-learning expansion.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Sequence, Tuple
from pathlib import Path

from dt_backend.core.config_dt import DT_PATHS


# -----------------------------
# Safe float helper
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


# -----------------------------
# Technical Indicators
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1"] = df["close"].pct_change().fillna(0)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

    # Rolling volatility
    df["vol_10"] = df["ret_1"].rolling(10).std().fillna(0)

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # EMA & SMA windows
    for w in [5, 10, 20]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean().fillna(method="bfill")
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR (simple version)
    df["hl"] = df["high"] - df["low"]
    df["hc"] = (df["high"] - df["close"].shift(1)).abs()
    df["lc"] = (df["low"] - df["close"].shift(1)).abs()
    tr = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().fillna(method="bfill")

    # Volume Z-score
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-9
    )
    df["vol_z"] = df["vol_z"].fillna(0)

    # Cleanup temp
    df.drop(columns=["hl", "hc", "lc"], errors="ignore", inplace=True)

    return df


# -----------------------------
# Label Generation
# -----------------------------
def make_future_labels(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """
    For each horizon H:
        y_ret_H = close.shift(-H)/close - 1
        y_class_H = BUY/HOLD/SELL thresholds
    """
    df = df.copy()

    for H in horizons:
        future_ret = df["close"].shift(-H) / df["close"] - 1
        df[f"y_ret_{H}"] = future_ret.fillna(0)

        # classification label
        y = future_ret.copy()

        # BUY if > +0.2% | SELL if < -0.2% | else HOLD
        df[f"y_cls_{H}"] = np.where(
            y > 0.002,
            2,
            np.where(y < -0.002, 0, 1),
        )

    return df


# -----------------------------
# Normalization
# -----------------------------
def normalize_matrix(X: np.ndarray, mode: str = "zscore") -> np.ndarray:
    if mode == "none":
        return X

    if mode == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-9
        return (X - mean) / std

    if mode == "minmax":
        minv = X.min(axis=0, keepdims=True)
        maxv = X.max(axis=0, keepdims=True)
        return (X - minv) / (maxv - minv + 1e-9)

    if mode == "robust":
        med = np.median(X, axis=0, keepdims=True)
        mad = np.median(np.abs(X - med), axis=0, keepdims=True) + 1e-9
        return (X - med) / mad

    return X


# -----------------------------
# Sequence slicing
# -----------------------------
def slice_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping or stride-based sequences.

    X: [T, D]
    y: [T, L]  (L = number of label columns)
    """
    T = X.shape[0]
    sequences = []
    labels = []

    for start in range(0, T - seq_len, stride):
        end = start + seq_len
        sequences.append(X[start:end])
        labels.append(y[end - 1])  # label at last timestep

    return np.array(sequences, dtype="float32"), np.array(labels, dtype="float32")


# -----------------------------
# Full Builder
# -----------------------------
def build_sequences_for_symbol(
    bars: Sequence[Dict[str, Any]],
    seq_len: int = 60,
    horizons: Sequence[int] = (1, 5, 10),
    norm: str = "zscore",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    bars → DataFrame → indicators → labels → (X, y)
    Returns:
        X: [N, seq_len, D]
        y: [N, L]
        feature_list: list of all features in X
    """

    if not bars:
        return np.zeros((0, seq_len, 1), dtype="float32"), np.zeros((0, 1), dtype="float32"), []

    df = pd.DataFrame(bars)
    df = df.sort_values("ts")

    # Basic required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0

    # Add advanced indicators
    df = add_indicators(df)

    # Add future labels
    df = make_future_labels(df, horizons)

    # Feature columns (auto-detected)
    feature_cols = [
        c for c in df.columns
        if c not in ["ts"] and not c.startswith("y_")
    ]

    X = df[feature_cols].values.astype("float32")
    y_cols = [f"y_ret_{H}" for H in horizons] + [f"y_cls_{H}" for H in horizons]
    y = df[y_cols].values.astype("float32")

    # Normalize inputs
    X = normalize_matrix(X, norm)

    # Slice into sequences
    X_seq, y_seq = slice_sequences(X, y, seq_len=seq_len, stride=stride)

    return X_seq, y_seq, feature_cols


# -----------------------------
# Dataset Writer
# -----------------------------
def write_sequence_dataset(
    symbol: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_list: List[str],
    tag: str = "default",
) -> Path:
    """Write parquet dataset for a single symbol."""

    root = Path(DT_PATHS.get("dtml_data", "ml_data_dt"))
    out_dir = root / "intraday" / "sequences" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "X": [X],  # stored as numpy arrays
        "y": [y],
        "features": [feature_list],
    })

    out_path = out_dir / f"{symbol}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
