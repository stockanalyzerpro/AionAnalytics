"""
dt_backend.core package

Provides core building blocks for the intraday engine:
  • config_dt       → DT_PATHS, ensure_dt_dirs
  • data_pipeline   → log, _read_rolling, save_rolling, load_universe
  • context_state   → build_intraday_context
  • regime_detector → classify_intraday_regime
  • policy_engine   → apply_intraday_policy
"""

from .config_dt import DT_PATHS, ensure_dt_dirs
from .data_pipeline_dt import (
    log,
    _read_rolling,
    save_rolling,
    load_universe,
    ensure_symbol_node,
)
from .context_state_dt import build_intraday_context
from .regime_detector_dt import classify_intraday_regime
from .policy_engine_dt import apply_intraday_policy

__all__ = [
    "DT_PATHS",
    "ensure_dt_dirs",
    "log",
    "_read_rolling",
    "save_rolling",
    "load_universe",
    "ensure_symbol_node",
    "build_intraday_context",
    "classify_intraday_regime",
    "apply_intraday_policy",
]
