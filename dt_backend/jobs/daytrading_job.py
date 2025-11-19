# dt_backend/jobs/daytrading_job.py — v2.0
"""
Main intraday trading loop for AION dt_backend.

This job wires together the full intraday pipeline:

    bars_intraday → context_dt → features_dt → predictions_dt
                   → regime → policy_dt → signals + (optional) execution
"""

from __future__ import annotations

from typing import Any, Dict

from dt_backend.core import (
    log,
    build_intraday_context,
    classify_intraday_regime,
    apply_intraday_policy,
)
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml import (
    score_intraday_tickers,
    build_intraday_signals,
)
from dt_backend.engines.trade_executor import ExecutionConfig, execute_from_policy


def run_daytrading_cycle(
    execute: bool = False,
    max_symbols: int | None = None,
    max_positions: int = 50,
    execution_cfg: ExecutionConfig | None = None,
) -> Dict[str, Any]:
    """
    Run one full intraday cycle.

    Parameters
    ----------
    execute:
        If True, actually call `execute_from_policy` (paper by default).
    max_symbols:
        Optional cap on number of symbols for features/scoring.
    max_positions:
        Max symbols selected by policy (ranking step).
    execution_cfg:
        Optional ExecutionConfig override for trade sizing.

    Returns
    -------
    Summary dict with keys:
        context, features, scoring, regime, policy, signals, execution
    """
    log("[daytrading_job] 🚀 starting intraday cycle.")

    ctx_summary = build_intraday_context()
    feat_summary = build_intraday_features(max_symbols=max_symbols)
    score_summary = score_intraday_tickers(max_symbols=max_symbols)
    regime_summary = classify_intraday_regime()
    policy_summary = apply_intraday_policy(max_positions=max_positions)
    signals_summary = build_intraday_signals()

    exec_summary: Dict[str, Any] | None = None
    if execute:
        exec_summary = execute_from_policy(execution_cfg)

    log("[daytrading_job] ✅ intraday cycle complete.")

    return {
        "context": ctx_summary,
        "features": feat_summary,
        "scoring": score_summary,
        "regime": regime_summary,
        "policy": policy_summary,
        "signals": signals_summary,
        "execution": exec_summary,
    }
