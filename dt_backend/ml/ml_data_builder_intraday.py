
# dt_backend/ml/ml_data_builder_intraday.py — v4.0
"""
Intraday ML dataset builder for AION day-trading engine (Option-D ready).

Responsibilities
----------------
  • Read intraday rolling cache
  • Flatten per-symbol `features_dt` snapshots into a tabular dataset
  • Write a parquet file that `train_lightgbm_intraday` and friends can use

Reads
-----
  • dt_backend.core.data_pipeline_dt._read_rolling()
  • dt_backend.core.config_dt.DT_PATHS

Writes
------
  1) If DT_PATHS["dtml_intraday_dataset"] exists:
       → that exact path
  2) Additionally, if DT_PATHS["dtml_data"] exists:
       → DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
     (for compatibility with Phase 1 LightGBM trainer)

  If neither key exists, falls back to:
       ml_data_dt/training_data_intraday.parquet
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dt_backend.core.config_dt import DT_PATHS  # type: ignore
from dt_backend.core.data_pipeline_dt import _read_rolling, log


def _infer_ts(node: Dict[str, Any]) -> datetime | None:
    """
    Infer a reasonable timestamp for this symbol's snapshot.

    Preference order:
      1) node["features_dt"]["ts"]
      2) latest bar timestamp in node["bars_intraday"]
      3) now() as a fallback
    """
    feats = node.get("features_dt") or {}
    ts_raw = feats.get("ts") or feats.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            # Accept both ISO and plain "YYYY-MM-DD HH:MM:SS"
            return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(ts_raw.split(".")[0], "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

    if isinstance(ts_raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts_raw))
        except Exception:
            pass

    # Try latest bar
    bars = node.get("bars_intraday") or []
    latest_ts = None
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        cand = bar.get("ts") or bar.get("t") or bar.get("timestamp")
        if isinstance(cand, str):
            try:
                dt = datetime.fromisoformat(cand.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(cand.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
        elif isinstance(cand, (int, float)):
            try:
                dt = datetime.utcfromtimestamp(float(cand))
            except Exception:
                continue
        else:
            continue
        if latest_ts is None or dt > latest_ts:
            latest_ts = dt

    if latest_ts is not None:
        return latest_ts

    # Last resort – "now"
    return datetime.utcnow()


def _resolve_dataset_paths() -> List[Path]:
    """
    Decide which dataset paths to write.

    - Primary path: DT_PATHS["dtml_intraday_dataset"] if present.
    - Secondary (compat): DT_PATHS["dtml_data"]/training_data_intraday.parquet if present.
    - Fallback: ml_data_dt/training_data_intraday.parquet
    """
    paths: List[Path] = []

    if "dtml_intraday_dataset" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_intraday_dataset"]))

    if "dtml_data" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_data"]) / "training_data_intraday.parquet")

    if not paths:
        paths.append(Path("ml_data_dt") / "training_data_intraday.parquet")

    # Deduplicate while preserving order
    seen = set()
    deduped: List[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def build_intraday_dataset(max_symbols: int | None = None) -> Dict[str, Any]:
    """
    Flatten intraday rolling into a training dataset.

    Parameters
    ----------
    max_symbols:
        Optional cap on number of symbols to include. If provided,
        we take the first N symbols (alphabetically) for a quick,
        lightweight dataset build.

    Returns
    -------
    Summary dict with basic counts and primary path.
    """
    rolling = _read_rolling()
    if not rolling:
        log("[dt_ml_builder] ⚠️ rolling empty, nothing to build.")
        return {"status": "empty", "rows": 0, "symbols": 0}

    items = [(sym, node) for sym, node in rolling.items() if not sym.startswith("_")]
    items.sort(key=lambda kv: kv[0])
    if max_symbols is not None:
        items = items[: max(0, int(max_symbols))]

    rows: List[Dict[str, Any]] = []
    for sym, node in items:
        if not isinstance(node, dict):
            continue
        feats = node.get("features_dt") or {}
        if not isinstance(feats, dict) or not feats:
            # No engineered features – skip symbol for training dataset
            continue

        ts = _infer_ts(node)
        row = {"symbol": sym, "ts": ts}
        # Merge features while avoiding collisions on "symbol"/"ts"
        for k, v in feats.items():
            if k in {"symbol", "ts"}:
                continue
            row[k] = v
        rows.append(row)

    if not rows:
        log("[dt_ml_builder] ⚠️ no feature rows to write.")
        return {"status": "no_rows", "rows": 0, "symbols": 0}

    df = pd.DataFrame.from_records(rows)
    paths = _resolve_dataset_paths()

    try:
        primary_path = paths[0]
        for i, dataset_path in enumerate(paths):
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dataset_path, index=False)
            if i == 0:
                log(
                    f"[dt_ml_builder] ✅ wrote primary dataset → {dataset_path} "
                    f"(rows={len(df)}, symbols={df['symbol'].nunique()})"
                )
            else:
                log(f"[dt_ml_builder] ↳ mirrored dataset → {dataset_path}")

        return {
            "status": "ok",
            "rows": int(len(df)),
            "symbols": int(df["symbol"].nunique()),
            "path": str(primary_path),
        }
    except Exception as e:
        log(f"[dt_ml_builder] ⚠️ failed to write dataset(s) {paths}: {e}")
        return {"status": "error", "error": str(e), "rows": 0, "symbols": 0}
