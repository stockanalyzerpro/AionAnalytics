"""
dt_backend/core/regime_detector_dt.py

Aggregates per-symbol intraday context into a simple market regime label.

The result is written into a reserved global node:

    rolling["_GLOBAL_DT"]["regime"] = {
        "label": "bull" | "bear" | "chop" | "unknown",
        "breadth_up": float,
        "n": int,
        "ts": "...",
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .data_pipeline_dt import _read_rolling, save_rolling, log, ensure_symbol_node


def classify_intraday_regime(min_symbols: int = 100) -> Dict[str, Any]:
    """
    Infer a simple market regime from per-symbol `context_dt`.

    Logic:
      • breadth_up = fraction of symbols with intraday_return > 0
      • if breadth_up > 0.60  → "bull"
      • if breadth_up < 0.40  → "bear"
      • otherwise             → "chop"
    """
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ classify_intraday_regime: rolling empty, nothing to do.")
        return {"regime": "unknown", "breadth": 0.5, "n": 0}

    up = 0
    down = 0
    total = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        ctx = (node or {}).get("context_dt") or {}
        r = ctx.get("intraday_return")
        try:
            r_f = float(r)
        except Exception:
            continue

        total += 1
        if r_f > 0.0:
            up += 1
        elif r_f < 0.0:
            down += 1

    if total < max(min_symbols, 1):
        log(f"⚠️ classify_intraday_regime: only {total} symbols with context.")
        breadth = 0.5
        regime = "unknown"
    else:
        breadth = up / float(total)
        if breadth > 0.60:
            regime = "bull"
        elif breadth < 0.40:
            regime = "bear"
        else:
            regime = "chop"

    node_global = ensure_symbol_node(rolling, "_GLOBAL_DT")
    regime_info = (node_global.get("regime") or {}) | {
        "label": regime,
        "breadth_up": breadth,
        "n": total,
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    node_global["regime"] = regime_info
    rolling["_GLOBAL_DT"] = node_global

    save_rolling(rolling)
    log(f"✅ classify_intraday_regime → {regime} (breadth_up={breadth:.2%}, n={total})")
    return {"regime": regime, "breadth": breadth, "n": total}
