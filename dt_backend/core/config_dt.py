"""
dt_backend/core/config_dt.py

Central configuration for the intraday (day-trading) engine.

This is the only place that knows about on-disk layout for dt_backend.
Other modules should import DT_PATHS from here instead of hard-coding
relative paths.

The layout follows the Aion_Analytics_Updated_File_Map_LSTM.txt blueprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

# This file lives at: <ROOT>/dt_backend/core/config_dt.py
ROOT: Path = Path(__file__).resolve().parents[2]
DT_BACKEND: Path = ROOT / "dt_backend"

# Dedicated intraday ML data root (sits next to ml_data/ used by backend)
ML_DATA_DT: Path = ROOT / "ml_data_dt"

LOGS_DT: Path = ROOT / "logs" / "dt_backend"


# ---------------------------------------------------------------------------
# Path map
# ---------------------------------------------------------------------------

DT_PATHS: Dict[str, Path] = {
    # roots
    "root": ROOT,
    "dt_backend": DT_BACKEND,
    "core": DT_BACKEND / "core",

    # universe / reference
    "universe_dir": DT_BACKEND / "universe",
    "universe_file": DT_BACKEND / "universe" / "symbol_universe.json",
    "exchanges_file": DT_BACKEND / "universe" / "exchanges.json",

    # raw bars (downloaded or replayed)
    "bars_intraday_dir": DT_BACKEND / "bars" / "intraday",
    "bars_daily_dir": DT_BACKEND / "bars" / "daily",

    # rolling snapshots / caches
    "rolling_intraday_dir": DT_BACKEND / "rolling" / "intraday",
    "rolling_intraday_file": DT_BACKEND / "rolling" / "intraday" / "rolling_intraday.json.gz",
    "rolling_longterm_dir": DT_BACKEND / "rolling" / "longterm",

    # signals + predictions (intraday + optional longterm)
    "signals_intraday_dir": DT_BACKEND / "signals" / "intraday",
    "signals_intraday_predictions_dir": DT_BACKEND / "signals" / "intraday" / "predictions",
    "signals_intraday_ranks_dir": DT_BACKEND / "signals" / "intraday" / "ranks",
    "signals_intraday_boards_dir": DT_BACKEND / "signals" / "intraday" / "boards",

    "signals_longterm_dir": DT_BACKEND / "signals" / "longterm",
    "signals_longterm_predictions_dir": DT_BACKEND / "signals" / "longterm" / "predictions",
    "signals_longterm_boards_dir": DT_BACKEND / "signals" / "longterm" / "boards",

    # historical replay
    "historical_replay_root": DT_BACKEND / "historical_replay",
    "historical_replay_raw": DT_BACKEND / "historical_replay" / "raw",
    "historical_replay_processed": DT_BACKEND / "historical_replay" / "processed",
    "historical_replay_meta": DT_BACKEND / "historical_replay" / "metadata.json",

    # intraday ML datasets / models (for compatibility with existing code)
    # NOTE: many existing modules expect keys named "dtml_data" and "dtmodels".
    "ml_data_dt": ML_DATA_DT,
    "dtml_data": ML_DATA_DT,
    "dtml_intraday_dataset": ML_DATA_DT / "training_data_intraday.parquet",
    "dtmodels": ML_DATA_DT / "models",

    # more granular model directories used by future LSTM / Transformer stack
    "models_root": DT_BACKEND / "models",
    "models_lgbm_intraday_dir": DT_BACKEND / "models" / "lightgbm_intraday",
    "models_lstm_intraday_dir": DT_BACKEND / "models" / "lstm_intraday",
    "models_transformer_intraday_dir": DT_BACKEND / "models" / "transformer_intraday",
    "models_ensemble_dir": DT_BACKEND / "models" / "ensemble",

    # logs
    "logs_dt": LOGS_DT,
}


def ensure_dt_dirs() -> None:
    """
    Best-effort directory creation. Never raises.

    We only create directories; file paths are resolved to their parent.
    """
    for key, path in DT_PATHS.items():
        try:
            target = path if path.suffix == "" else path.parent
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Silent by design to avoid circular import issues if logging
            # depends on this module.
            continue


# Eagerly create directory structure so other modules can assume existence.
ensure_dt_dirs()
