# dt_backend/engines/trade_executor.py — v1.0
"""
Intraday trade execution layer for AION dt_backend (paper by default).

This module glues together:
  • rolling[sym]["policy_dt"]
  • dt_backend.engines.broker_api (paper account)
  • dt_backend.core for logging and rolling helpers

It does not manage scheduling; higher-level jobs decide *when* to call
`execute_from_policy`. Here we only decide which orders to send and
apply them via the broker API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from dt_backend.core import _read_rolling, save_rolling, log
from dt_backend.engines.broker_api import Order, submit_order, get_cash


@dataclass
class ExecutionConfig:
    """
    Simple sizing rules for intraday execution.
    """
    risk_fraction_per_trade: float = 0.01   # fraction of cash per trade
    max_trades_per_cycle: int = 20
    min_confidence: float = 0.55           # filter low-confidence signals
    min_score_abs: float = 0.0             # optional score threshold


def _last_price_from_node(node: Dict[str, Any]) -> float | None:
    bars = node.get("bars_intraday") or []
    if not bars:
        return None
    last = bars[-1]
    price = last.get("c") or last.get("close") or last.get("price")
    try:
        return float(price)
    except Exception:
        return None


def execute_from_policy(cfg: ExecutionConfig | None = None) -> Dict[str, Any]:
    """
    Execute orders based on current `policy_dt` in rolling.

    Behavior
    --------
    • Fetch current rolling
    • For each symbol with BUY/SELL and confidence/score above thresholds:
        - Compute position size = risk_fraction_per_trade * cash / price
        - Submit simple market order via paper broker
    • HOLD, missing prices, or rejected orders are skipped.
    """
    if cfg is None:
        cfg = ExecutionConfig()

    rolling = _read_rolling()
    if not rolling:
        log("[dt_exec] ⚠️ rolling empty, nothing to execute.")
        return {"status": "empty", "orders": 0}

    cash = get_cash()
    if cash <= 0:
        log("[dt_exec] ⚠️ no cash in paper account, skipping.")
        return {"status": "no_cash", "orders": 0}

    candidates: List[str] = []
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        policy = (node or {}).get("policy_dt") or {}
        action = policy.get("action", "HOLD")
        if action not in {"BUY", "SELL"}:
            continue

        conf = float(policy.get("confidence", 0.0) or 0.0)
        score = float(policy.get("score", 0.0) or 0.0)
        if conf < cfg.min_confidence:
            continue
        if abs(score) < cfg.min_score_abs:
            continue
        candidates.append(sym)

    # Simple ordering: by absolute score descending
    def _score(sym: str) -> float:
        pol = (rolling.get(sym) or {}).get("policy_dt") or {}
        return float(pol.get("score", 0.0) or 0.0)

    candidates.sort(key=lambda s: abs(_score(s)), reverse=True)
    selected = candidates[: cfg.max_trades_per_cycle]

    orders_sent = 0
    filled = 0

    for sym in selected:
        node = rolling.get(sym) or {}
        price = _last_price_from_node(node)
        if price is None or price <= 0:
            continue

        alloc = cash * cfg.risk_fraction_per_trade
        if alloc <= 0:
            continue

        qty = alloc / price
        if qty <= 0:
            continue

        policy = node.get("policy_dt") or {}
        side = policy.get("action", "HOLD")
        if side not in {"BUY", "SELL"}:
            continue

        orders_sent += 1
        order = Order(symbol=sym, side=side, qty=qty)
        res = submit_order(order, last_price=price)
        if res.get("status") == "filled":
            filled += 1

    return {
        "status": "ok",
        "orders_sent": orders_sent,
        "orders_filled": filled,
        "selected": selected,
    }
