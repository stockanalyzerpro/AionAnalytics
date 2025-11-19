
# dt_backend/engines/feature_engineering.py — v3.0
"""
Intraday feature engineering for AION dt_backend.

This module turns raw intraday bars and intraday context into a compact
feature vector, stored under:

    rolling[sym]["features_dt"]

It leverages:
  • dt_backend.core.data_pipeline_dt  → rolling I/O, logging, node helpers
  • dt_backend.engines.indicators     → low-level numeric indicators

NOTE: This module does *not* define labels; it only computes features.
Label generation belongs to dedicated label / outcome modules.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from dt_backend.core.data_pipeline_dt import (
    _read_rolling,
    save_rolling,
    ensure_symbol_node,
    log,
)
from dt_backend.engines.indicators import sma, ema, rsi, pct_change, realized_vol


def _extract_prices(node: Dict[str, Any]) -> List[float]:
    """
    Extract intraday close/last prices from a rolling node.

    Expected bar schema is flexible but we look for:
      • "c"     → close
      • "close" → close
      • "price" → generic last
    """
    bars = node.get("bars_intraday") or []
    prices: List[float] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        price = (
            bar.get("c")
            if bar.get("c") is not None
            else bar.get("close", bar.get("price"))
        )
        try:
            prices.append(float(price))
        except Exception:
            continue
    return prices


def _compute_returns(prices: List[float]) -> List[float]:
    rets: List[float] = []
    if len(prices) < 2:
        return rets
    for i in range(1, len(prices)):
        rets.append(pct_change(prices[i], prices[i - 1]))
    return rets


def _feature_snapshot_for_symbol(sym: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a feature snapshot for a single symbol.

    Features are intentionally small, interpretable, and stable so models
    can be retrained over time without fragile schema changes.
    """
    prices = _extract_prices(node)
    if len(prices) < 5:
        # Not enough history to build a reliable snapshot
        return {}

    rets = _compute_returns(prices)

    # Moving averages (on price)
    sma_5 = sma(prices, 5)
    sma_20 = sma(prices, 20)
    ema_9 = ema(prices, 9)

    # RSI (14) → we name the feature explicitly rsi_14
    rsi_14 = rsi(prices, 14)

    # Volatility on intraday returns
    intraday_vol = realized_vol(rets)

    last_price = prices[-1]
    prev_open = prices[0]

    feat: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "last_price": last_price,
        "pct_chg_from_open": pct_change(last_price, prev_open),
        "sma_5": sma_5,
        "sma_20": sma_20,
        "ema_9": ema_9,
        "sma_5_over_sma_20": (sma_5 / sma_20 - 1.0) if sma_20 != 0 else 0.0,
        "rsi_14": rsi_14,
        "intraday_vol": intraday_vol,
    }

    # Merge intraday context if present (avoid collisions)
    ctx = node.get("context_dt") or {}
    for k, v in ctx.items():
        if k in feat:
            continue
        feat[k] = v

    return feat


def build_intraday_features(max_symbols: int | None = None) -> Dict[str, Any]:
    """
    Main entry point: compute features_dt for each symbol in rolling.

    Parameters
    ----------
    max_symbols:
        Optional cap on number of symbols (alphabetical). Useful for
        quick dev runs.

    Returns
    -------
    Summary dict.
    """
    rolling = _read_rolling()
    if not rolling:
        log("[dt_features] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    items = [(sym, node) for sym, node in rolling.items() if not sym.startswith("_")]
    items.sort(key=lambda kv: kv[0])
    if max_symbols is not None:
        items = items[: max(0, int(max_symbols))]

    updated = 0
    for sym, node_raw in items:
        node = ensure_symbol_node(rolling, sym)
        feat = _feature_snapshot_for_symbol(sym, node)
        if not feat:
            continue
        node["features_dt"] = feat
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[dt_features] ✅ updated features_dt for {updated} symbols.")
    return {"symbols": len(items), "updated": updated}
