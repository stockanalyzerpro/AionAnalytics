"""
dt_backend.engines package

Contains intraday "engine" components:

  • indicators          → low-level math utilities
  • feature_engineering → build_intraday_features
  • backtesting_engine  → simple run_intraday_backtest
  • broker_api          → paper broker interface
  • trade_executor      → execute_from_policy using broker_api
"""

from .indicators import sma, ema, rsi, pct_change, realized_vol
from .feature_engineering import build_intraday_features
from .backtesting_engine import run_intraday_backtest, BacktestConfig, BacktestResult
from .broker_api import (
    PAPER_STATE_PATH,
    Position,
    Order,
    get_positions,
    get_cash,
    submit_order,
)
from .trade_executor import ExecutionConfig, execute_from_policy

__all__ = [
    "sma",
    "ema",
    "rsi",
    "pct_change",
    "realized_vol",
    "build_intraday_features",
    "run_intraday_backtest",
    "BacktestConfig",
    "BacktestResult",
    "PAPER_STATE_PATH",
    "Position",
    "Order",
    "get_positions",
    "get_cash",
    "submit_order",
    "ExecutionConfig",
    "execute_from_policy",
]
