# dt_backend/engines/backtesting_engine.py — v1.0
"""
Simple intraday backtesting harness for AION dt_backend.

This is intentionally lightweight and focused on **strategy wiring**
rather than perfect fill modeling. It allows you to:

  • Replay historical intraday sessions symbol by symbol
  • Use the same feature → model → policy pipeline
  • Collect PnL / hit-rate style summaries

It assumes that higher-level jobs prepare the historical bars in
DT_PATHS["historical_replay_processed"].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Iterable

from dt_backend.core import DT_PATHS, log, ensure_symbol_node, save_rolling, _read_rolling
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.core import build_intraday_context, classify_intraday_regime, apply_intraday_policy
from dt_backend.ml import score_intraday_tickers


@dataclass
class BacktestConfig:
    """
    Tuning knobs for a simple backtest.
    """
    max_symbols: int = 200
    top_n: int = 20          # how many symbols to "trade"
    per_trade_notional: float = 1_000.0
    fee_per_trade: float = 0.0


@dataclass
class Trade:
    symbol: str
    action: str          # BUY / SELL
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class BacktestResult:
    trades: List[Trade]
    gross_pnl: float
    net_pnl: float
    n_wins: int
    n_losses: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades": [asdict(t) for t in self.trades],
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
        }


def _load_replay_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        log(f"[dt_backtest] ⚠️ failed to load replay file {path}: {e}")
    return {}


def run_intraday_backtest(config: BacktestConfig | None = None) -> BacktestResult:
    """
    Run a toy intraday backtest using the current pipeline.

    Assumptions
    -----------
    • historical_replay_processed contains a JSON file with:
          { "SYMBOL": [ {bar...}, ... ], ... }
    • We only trade once per symbol: entry at first bar, exit at last bar,
      sign based on policy BUY/SELL.
    """
    if config is None:
        config = BacktestConfig()

    replay_path = DT_PATHS["historical_replay_processed"] / "intraday_snapshot.json"
    data = _load_replay_file(replay_path)
    if not data:
        log(f"[dt_backtest] ⚠️ no replay data at {replay_path}")
        return BacktestResult(trades=[], gross_pnl=0.0, net_pnl=0.0, n_wins=0, n_losses=0)

    # Seed rolling from replay data
    rolling = {}
    symbols = sorted([s for s in data.keys() if isinstance(data[s], list)])[: config.max_symbols]
    for sym in symbols:
        node = ensure_symbol_node(rolling, sym)
        node["bars_intraday"] = data[sym]
        rolling[sym] = node

    # Write initial rolling
    save_rolling(rolling)

    # Build context + features + predictions + policy using existing stack
    build_intraday_context()
    build_intraday_features(max_symbols=config.max_symbols)
    score_intraday_tickers(max_symbols=config.max_symbols)
    classify_intraday_regime()
    policy_summary = apply_intraday_policy(max_positions=config.top_n)

    # Reload rolling with policies
    rolling = _read_rolling()

    trades: List[Trade] = []
    gross_pnl = 0.0
    n_wins = 0
    n_losses = 0

    for sym in policy_summary.get("selected_symbols", []):
        node = rolling.get(sym) or {}
        bars = node.get("bars_intraday") or []
        if len(bars) < 2:
            continue

        first = bars[0]
        last = bars[-1]

        price_in = first.get("c") or first.get("close") or first.get("price")
        price_out = last.get("c") or last.get("close") or last.get("price")

        try:
            price_in = float(price_in)
            price_out = float(price_out)
        except Exception:
            continue

        policy = node.get("policy_dt") or {}
        action = policy.get("action", "HOLD")
        if action not in {"BUY", "SELL"}:
            continue

        direction = 1.0 if action == "BUY" else -1.0
        ret = direction * (price_out / price_in - 1.0)
        pnl = ret * config.per_trade_notional

        trades.append(Trade(symbol=sym, action=action, entry_price=price_in, exit_price=price_out, pnl=pnl))
        gross_pnl += pnl
        if pnl >= 0:
            n_wins += 1
        else:
            n_losses += 1

    net_pnl = gross_pnl - config.fee_per_trade * len(trades)

    result = BacktestResult(
        trades=trades,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        n_wins=n_wins,
        n_losses=n_losses,
    )
    log(f"[dt_backtest] ✅ completed backtest: trades={len(trades)}, net_pnl={net_pnl:.2f}")
    return result
