"""
ml_data_builder.py — v4.0
AION Analytics — Nightly Multi-Layer ML Dataset Builder

Option D: Full Intelligent ML Stack

This builder:
    • Reads rolling.json.gz (enriched with fundamentals, metrics, context, news, social)
    • Optionally reads:
        - macro_state.json
        - dt_backend intraday "brain" (drift, hit ratios, etc.)
    • Flattens these into a tabular dataset (one row per symbol)
    • Computes multi-horizon targets using price history:
        - 1d, 3d, 1w, 2w, 4w, 13w, 26w, 52w
    • Provides both:
        - raw returns targets (target_ret_*)
        - direction targets (target_dir_*)
        - a default "target" alias for convenience

Output:
    • Parquet: ml_data/nightly/dataset/training_data_daily.parquet
    • JSON:   ml_data/nightly/dataset/feature_list_daily.json

Public API:
    build_ml_dataset(mode: str = "daily") -> pandas.DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, log, safe_float

# Optional dt_backend intraday brain (for hybrid features)
try:
    from dt_backend.core.data_pipeline_dt import _read_brain as _read_dt_brain  # type: ignore
    DT_ENABLED = True
except Exception:
    DT_ENABLED = False

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ML_DATA_ROOT: Path = PATHS.get("ml_data", Path("ml_data"))
DATASET_DIR: Path = ML_DATA_ROOT / "nightly" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = DATASET_DIR / "training_data_daily.parquet"
FEATURE_LIST_FILE = DATASET_DIR / "feature_list_daily.json"

MACRO_STATE_FILE: Path = PATHS.get("macro_state", ML_DATA_ROOT / "macro_state.json")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _compute_returns(history: List[Dict[str, Any]], days: int) -> float:
    """
    Compute % return over `days` using close prices from history.
    History is a list of dicts with at least {"date": ..., "close": ...},
    sorted oldest→newest (we'll enforce that).
    """
    if not history or len(history) <= days:
        return 0.0
    hist_sorted = sorted(history, key=lambda h: h.get("date") or "")
    last = safe_float(hist_sorted[-1].get("close"))
    prev = safe_float(hist_sorted[-(days + 1)].get("close"))
    if prev <= 0:
        return 0.0
    return (last - prev) / prev


def _direction_from_return(ret: float, tol: float = 0.001) -> int:
    """
    Map a return to direction label:
        +1 → up
        -1 → down
         0 → flat / noise
    tol is a "dead zone" threshold (0.1% default).
    """
    if ret > tol:
        return 1
    if ret < -tol:
        return -1
    return 0


def _load_macro_state() -> Dict[str, float]:
    """Load macro_state.json into a flat numeric dict."""
    try:
        if not MACRO_STATE_FILE.exists():
            return {}
        raw = json.loads(MACRO_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[ml_data_builder] ⚠️ Failed to read macro_state.json: {e}")
        return {}

    out: Dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _load_dt_brain() -> Dict[str, Dict[str, Any]]:
    """Optional intraday brain from dt_backend."""
    if not DT_ENABLED:
        return {}
    try:
        dtb = _read_dt_brain() or {}
        return dtb
    except Exception as e:
        log(f"[ml_data_builder] ⚠️ Failed to read dt_backend brain: {e}")
        return {}


def _flatten_numeric(prefix: str, d: Dict[str, Any]) -> Dict[str, float]:
    """Flatten numeric fields from dict `d` under a prefix."""
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out


# ---------------------------------------------------------------------
# Row builder for a single symbol
# ---------------------------------------------------------------------

def _build_row_for_symbol(
    sym: str,
    node: Dict[str, Any],
    macro: Dict[str, float],
    dt_brain: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Build a single training row for a symbol, including:
        - base price/volume
        - fundamentals / metrics
        - context / news / social
        - macro features
        - optional dt_backend drift features
        - multi-horizon return & direction targets
    """
    sym_u = sym.upper()
    history = node.get("history") or []
    if not history or len(history) < 2:
        # Not enough price history to compute anything meaningful
        return None

    history_sorted = sorted(history, key=lambda h: h.get("date") or "")
    latest = history_sorted[-1]

    row: Dict[str, Any] = {
        "symbol": sym_u,
        "name": node.get("name") or node.get("company_name") or sym_u,
        "sector": (node.get("sector")
                   or (node.get("fundamentals") or {}).get("sector")
                   or ""),
    }

    # Basic price/volume
    row["close"] = safe_float(latest.get("close", node.get("close", 0.0)))
    row["volume"] = safe_float(latest.get("volume", node.get("volume", 0.0)))

    # Fundamentals, metrics, context, news, social
    fundamentals = node.get("fundamentals") or {}
    metrics = node.get("metrics") or {}
    context = node.get("context") or {}
    news = node.get("news") or {}
    social = node.get("social") or {}

    row.update(_flatten_numeric("fund_", fundamentals))
    row.update(_flatten_numeric("met_", metrics))
    row.update(_flatten_numeric("ctx_", context))
    row.update(_flatten_numeric("news_", news))
    row.update(_flatten_numeric("soc_", social))

    # Macro: same for all symbols (prefix macro_)
    for k, v in macro.items():
        row[f"macro_{k}"] = float(v)

    # dt_backend intraday brain (optional)
    dt_node = dt_brain.get(sym_u) if dt_brain else None
    if isinstance(dt_node, dict):
        # Select drift / hit ratios / error stats if present
        for key in [
            "drift_score",
            "hit_ratio_short",
            "hit_ratio_long",
            "mae_short",
            "mae_long",
        ]:
            if key in dt_node and isinstance(dt_node[key], (int, float)):
                row[f"dt_{key}"] = float(dt_node[key])

    # -----------------------------------------------------------------
    # Targets: multi-horizon returns + directions (past-looking)
    # -----------------------------------------------------------------
    # Map horizons in days (approximate weekly mapping)
    horizons = {
        "1d": 1,
        "3d": 3,
        "1w": 5,
        "2w": 10,
        "4w": 20,
        "13w": 65,
        "26w": 130,
        "52w": 260,
    }

    default_target_dir = 0
    for label, days in horizons.items():
        ret = _compute_returns(history_sorted, days)
        row[f"target_ret_{label}"] = float(ret)
        d = _direction_from_return(ret)
        row[f"target_dir_{label}"] = int(d)
        # We'll use 1w direction as the default "target"
        if label == "1w":
            default_target_dir = d

    # default alias expected by some training code
    row["target"] = int(default_target_dir)

    return row


# ---------------------------------------------------------------------
# PUBLIC: build_ml_dataset
# ---------------------------------------------------------------------

def build_ml_dataset(mode: str = "daily") -> pd.DataFrame:
    """
    Build the nightly ML dataset.

    mode:
        "daily"  → standard nightly cross-sectional dataset
        (other modes reserved for future use; currently ignored)

    Returns:
        pandas.DataFrame with one row per symbol, including:
            - features: everything except `symbol`, `name`, `sector`, targets
            - targets: target*, target_ret_*, target_dir_*
    Writes:
        training_data_daily.parquet
        feature_list_daily.json
    """
    rolling = _read_rolling()
    if not rolling:
        log("[ml_data_builder] ⚠️ No rolling.json.gz — dataset will be empty.")
        return pd.DataFrame()

    macro = _load_macro_state()
    dt_brain = _load_dt_brain()

    rows: List[Dict[str, Any]] = []
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        row = _build_row_for_symbol(sym, node, macro, dt_brain)
        if row is not None:
            rows.append(row)

    if not rows:
        log("[ml_data_builder] ⚠️ No usable symbols — dataset is empty.")
        df = pd.DataFrame()
        return df

    df = pd.DataFrame(rows)

    # Ensure consistent column ordering: id fields first, then features, then targets
    id_cols = ["symbol", "name", "sector"]
    target_cols = [c for c in df.columns if c.startswith("target")]
    feature_cols = [c for c in df.columns if c not in id_cols + target_cols]

    df = df[id_cols + feature_cols + target_cols]

    # Persist to parquet
    try:
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATASET_FILE, index=False)
        log(f"[ml_data_builder] 💾 Wrote dataset → {DATASET_FILE} ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        log(f"[ml_data_builder] ⚠️ Failed to write dataset parquet: {e}")

    # Persist feature list (for training code / drift monitoring / debugging)
    try:
        feature_list_payload = {
            "generated_at": datetime.now(TIMEZONE).isoformat(),
            "id_columns": id_cols,
            "feature_columns": feature_cols,
            "target_columns": target_cols,
        }
        FEATURE_LIST_FILE.write_text(json.dumps(feature_list_payload, indent=2), encoding="utf-8")
        log(f"[ml_data_builder] 📝 Feature list written → {FEATURE_LIST_FILE}")
    except Exception as e:
        log(f"[ml_data_builder] ⚠️ Failed to write feature list: {e}")

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    out_df = build_ml_dataset(mode="daily")
    print(f"Rows: {len(out_df)}, Columns: {len(out_df.columns)}")
    # Optional preview
    with pd.option_context("display.max_columns", 40, "display.width", 200):
        print(out_df.head())
