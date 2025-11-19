
"""
dt_backend/core/context_state_dt.py

Builds intraday "context" features for each symbol in the rolling cache.

Philosophy:
  • Humans do not just see prices; they feel trends, volatility, and
    intraday behavior. This module tries to distill a small set of
    intuitive features that mimic that mental model.
"""
from __future__ import annotations

import math
from datetime import datetime, date
from typing import Any, Dict, List

from .data_pipeline_dt import _read_rolling, save_rolling, log, ensure_symbol_node


def _parse_ts(ts_raw: Any) -> datetime | None:
    """Best-effort parsing of timestamps used in intraday bars."""
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            # epoch seconds
            return datetime.utcfromtimestamp(float(ts_raw))
        txt = str(ts_raw)
        # Try ISO first
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:
            # Fallback: common "YYYY-MM-DD HH:MM:SS" format
            return datetime.strptime(txt.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_today_bars(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pull out today's bars from node["bars_intraday"].

    Expected (but flexible) bar schema:
      • "ts" or "t" or "timestamp"  → timestamp
      • "c" or "close" or "price"  → last price
    """
    src = node.get("bars_intraday") or []
    today = date.today()

    out: List[Dict[str, Any]] = []
    for raw_bar in src:
        if not isinstance(raw_bar, dict):
            continue

        ts = _parse_ts(raw_bar.get("ts") or raw_bar.get("t") or raw_bar.get("timestamp"))
        if ts is None or ts.date() != today:
            continue

        price = (
            raw_bar.get("c")
            if raw_bar.get("c") is not None
            else raw_bar.get("close", raw_bar.get("price"))
        )
        price_f = _safe_float(price)
        if price_f is None:
            continue

        out.append({"ts": ts, "price": price_f})

    out.sort(key=lambda b: b["ts"])
    return out


def _pct(a: float, b: float) -> float:
    try:
        if b == 0.0 or not (math.isfinite(a) and math.isfinite(b)):
            return 0.0
        return (a / b) - 1.0
    except Exception:
        return 0.0


def _intraday_stats(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute intuitive intraday stats from a sorted bar list."""
    if len(bars) < 3:
        return {}

    prices = [b["price"] for b in bars]
    first = prices[0]
    last = prices[-1]
    high = max(prices)
    low = min(prices)

    intraday_return = _pct(last, first)
    intraday_range = _pct(high, low) if low > 0 else 0.0

    # Naive realized intraday volatility: std of 1-bar returns
    rets: List[float] = []
    for i in range(1, len(prices)):
        rets.append(_pct(prices[i], prices[i - 1]))
    if rets:
        mu = sum(rets) / len(rets)
        var = sum((x - mu) ** 2 for x in rets) / max(len(rets) - 1, 1)
        intraday_vol = math.sqrt(max(var, 0.0))
    else:
        intraday_vol = 0.0

    return {
        "intraday_return": intraday_return,
        "intraday_range": intraday_range,
        "intraday_vol": intraday_vol,
        "last_price": last,
    }


def _trend_label(r: float, strong: float = 0.01, mild: float = 0.003) -> str:
    if r >= strong:
        return "strong_bull"
    if r >= mild:
        return "bull"
    if r <= -strong:
        return "strong_bear"
    if r <= -mild:
        return "bear"
    return "flat"


def _vol_bucket(vol: float) -> str:
    if vol >= 0.02:
        return "high"
    if vol >= 0.007:
        return "medium"
    return "low"


def build_intraday_context() -> Dict[str, Any]:
    """
    Main entry point.

    Reads the rolling cache, computes context_dt per symbol, writes back.
    Returns a small summary for logging / APIs.

    This function is *best effort* and will never raise in normal usage.
    """
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ build_intraday_context: rolling is empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    updated = 0
    for sym, _ in list(rolling.items()):
        if sym.startswith("_"):
            # reserved for global / meta nodes
            continue

        node = ensure_symbol_node(rolling, sym)
        bars_today = _extract_today_bars(node)
        stats = _intraday_stats(bars_today)
        if not stats:
            continue

        ctx = node.get("context_dt") or {}
        ctx.update(stats)

        trend = _trend_label(stats["intraday_return"])
        vol_bkt = _vol_bucket(stats["intraday_vol"])

        ctx["intraday_trend"] = trend
        ctx["vol_bucket"] = vol_bkt
        ctx["has_intraday_data"] = True

        node["context_dt"] = ctx
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"✅ build_intraday_context updated {updated} symbols.")
    return {"symbols": len(rolling), "updated": updated}
