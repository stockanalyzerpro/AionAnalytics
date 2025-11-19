# dt_backend/engines/broker_api.py — v1.0
"""
Abstract broker API for AION dt_backend.

This module is deliberately conservative and defaults to **paper trading**
semantics. It does not talk to any real broker out of the box; instead it
provides a simple in-memory / file-backed "paper account" that other
modules (trade_executor, simulators) can use.

You can later subclass or replace this with a real broker integration,
but the interface should remain as close as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

from dt_backend.core import DT_PATHS, log


PAPER_STATE_PATH: Path = DT_PATHS["dt_backend"] / "paper_account_intraday.json"


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float


@dataclass
class Order:
    symbol: str
    side: str           # "BUY" or "SELL"
    qty: float
    limit_price: float | None = None


def _read_state() -> Dict[str, Any]:
    if not PAPER_STATE_PATH.exists():
        return {"cash": 100_000.0, "positions": {}}
    try:
        import json
        with open(PAPER_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {"cash": 100_000.0, "positions": {}}
    except Exception as e:
        log(f"[broker_paper] ⚠️ failed to read state: {e}")
        return {"cash": 100_000.0, "positions": {}}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        import json
        PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"[broker_paper] ⚠️ failed to save state: {e}")


def get_positions() -> Dict[str, Position]:
    state = _read_state()
    positions_raw = state.get("positions") or {}
    out: Dict[str, Position] = {}
    for sym, node in positions_raw.items():
        try:
            out[sym] = Position(
                symbol=sym,
                qty=float(node.get("qty", 0.0)),
                avg_price=float(node.get("avg_price", 0.0)),
            )
        except Exception:
            continue
    return out


def get_cash() -> float:
    state = _read_state()
    try:
        return float(state.get("cash", 0.0))
    except Exception:
        return 0.0


def submit_order(order: Order, last_price: float | None = None) -> Dict[str, Any]:
    """
    Paper-fill a simple market/limit order.

    Logic:
      • If limit_price is set, require last_price to be favorable.
      • Fills immediately at last_price (or limit if provided).
    """
    state = _read_state()
    positions = state.get("positions") or {}
    cash = float(state.get("cash", 0.0))

    if last_price is None:
        log(f"[broker_paper] ⚠️ no last_price for {order.symbol}, skipping order.")
        return {"status": "rejected", "reason": "no_price"}

    fill_price = float(last_price)
    if order.limit_price is not None:
        lp = float(order.limit_price)
        if order.side == "BUY" and fill_price > lp:
            return {"status": "rejected", "reason": "limit_not_reached"}
        if order.side == "SELL" and fill_price < lp:
            return {"status": "rejected", "reason": "limit_not_reached"}

    qty = float(order.qty)
    cost = fill_price * qty

    if order.side == "BUY":
        if cost > cash:
            return {"status": "rejected", "reason": "insufficient_cash"}
        cash -= cost
        pos = positions.get(order.symbol) or {"qty": 0.0, "avg_price": 0.0}
        new_qty = float(pos["qty"]) + qty
        if new_qty <= 0:
            positions.pop(order.symbol, None)
        else:
            new_avg = (
                (float(pos["avg_price"]) * float(pos["qty"]) + cost) / new_qty
                if pos["qty"] > 0
                else fill_price
            )
            positions[order.symbol] = {"qty": new_qty, "avg_price": new_avg}
    else:  # SELL
        pos = positions.get(order.symbol) or {"qty": 0.0, "avg_price": 0.0}
        if qty > pos["qty"]:
            qty = float(pos["qty"])
        cash += fill_price * qty
        new_qty = float(pos["qty"]) - qty
        if new_qty <= 0:
            positions.pop(order.symbol, None)
        else:
            positions[order.symbol] = {"qty": new_qty, "avg_price": float(pos["avg_price"])}

    state["cash"] = cash
    state["positions"] = positions
    _save_state(state)

    log(f"[broker_paper] ✅ filled {order.side} {qty} {order.symbol} @ {fill_price}")
    return {"status": "filled", "symbol": order.symbol, "side": order.side, "qty": qty, "price": fill_price}
