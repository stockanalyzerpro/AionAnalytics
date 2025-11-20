"""
AION Analytics — Advanced Regime Detector (Nightly)
---------------------------------------------------

This module analyzes:
    • Market breadth (percentage of symbols closing green)
    • Volatility proxy (recent daily volatility)
    • Macro signals (VIX & SPY from macro_fetcher)
    • Trend momentum
    • Rolling brain drift

Outputs:
    A dictionary:
        {
            "label": "bull" | "bear" | "chop",
            "breadth_up": float,
            "volatility": float,
            "macro": {...},
            "momentum": float,
            "drift": float,
            "timestamp": ISO string
        }

This is analogous to dt_backend's intraday regime engine,
but adapted for daily / EOD operations.
"""

from __future__ import annotations
import json
import datetime
from pathlib import Path
from typing import Dict, Any

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    safe_float,
    log,
)


# ======================================================================
# HELPERS
# ======================================================================

def _read_macro() -> Dict[str, Any]:
    """Read macro_state produced by macro_fetcher (safe default)."""
    try:
        path = PATHS["market_state"]
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        # If macro fields exist inside "regime", strip them
        if "macro" in raw:
            return raw["macro"]
        return raw
    except Exception:
        return {}


# ======================================================================
# CORE REGIME FUNCTION
# ======================================================================

def detect_regime(rolling: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Determine daily regime using:
        ✓ market breadth
        ✓ daily volatility
        ✓ macro signals
        ✓ momentum
        ✓ rolling brain drift

    This is similar to intraday breadth logic:
        >60% green → bull
        <40% green → bear
        else → chop
    but enhanced with volatility + macro weighting.
    """

    if rolling is None:
        rolling = _read_rolling()

    if not rolling:
        return {"label": "unknown", "timestamp": datetime.datetime.now().isoformat()}

    brain = _read_brain()
    macro = _read_macro()

    returns = []
    vols = []
    drifts = []

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        hist = node.get("history", [])
        if hist and len(hist) >= 2:
            try:
                last = safe_float(hist[-1].get("close"))
                prev = safe_float(hist[-2].get("close"))
                returns.append((last - prev) / prev if prev > 0 else 0.0)
            except Exception:
                pass

        # Simple vol proxy from last 5 days
        if hist and len(hist) >= 6:
            closes = [safe_float(h.get("close")) for h in hist[-6:]]
            diffs = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    diffs.append(abs(closes[i] - closes[i - 1]) / closes[i - 1])
            if diffs:
                vols.append(sum(diffs) / len(diffs))

        # Drift from rolling brain
        if sym in brain:
            drifts.append(safe_float(brain[sym].get("drift_score", 0.0)))

    # Market Breadth
    breadth_up = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0

    # Volatility (higher = more bearish weight)
    vol_proxy = sum(vols) / len(vols) if vols else 0.0

    # Drift
    drift_score = sum(drifts) / len(drifts) if drifts else 0.0

    # Macro Signals (from macro_fetcher):
    vix = safe_float(macro.get("vix", 0.0))
    spy_pct = safe_float(macro.get("spy_daily_pct", 0.0))

    # Momentum proxy = average daily return
    momentum = sum(returns) / len(returns) if returns else 0.0

    # ==================================================================
    # DECISION
    # ==================================================================

    label = "chop"  # default sideways

    # Primary: breadth-based (dt_backend style)
    if breadth_up >= 0.60:
        label = "bull"
    elif breadth_up <= 0.40:
        label = "bear"

    # Volatility override
    if vol_proxy > 0.02:  # high volatility → bearish pressure
        if label == "bull":
            label = "chop"
        else:
            label = "bear"

    # VIX override (macro fear index)
    if vix >= 25:  # elevated VIX → risk-off
        label = "bear"

    # SPY momentum boost
    if spy_pct > 0.5:
        label = "bull"

    # Drift override (model instability → lower confidence)
    if drift_score > 0.10:
        # Only soften bull → chop, not bear → bull
        if label == "bull":
            label = "chop"

    result = {
        "label": label,
        "breadth_up": round(breadth_up, 4),
        "volatility": round(vol_proxy, 4),
        "momentum": round(momentum, 4),
        "macro": {
            "vix": vix,
            "spy_pct": spy_pct,
        },
        "drift": round(drift_score, 4),
        "timestamp": datetime.datetime.now(TIMEZONE).isoformat(),
    }

    # Save snapshot to PATHS
    try:
        PATHS["regime_state"].write_text(json.dumps(result, indent=2), encoding="utf-8")
        log("📈 regime_state.json updated")
    except Exception as e:
        log(f"⚠️ Failed writing regime_state.json: {e}")

    return result
