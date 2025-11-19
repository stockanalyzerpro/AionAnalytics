"""
macro_fetcher.py — v3.0
Aligned with backend/core + nightly_job v4.0 (Hybrid Mode)

Collects and normalizes macroeconomic + market-wide signals:
    • VIX
    • SPY (price + % change)
    • QQQ
    • DXY (Dollar Index)
    • ^TNX (10-year yield)
    • Oil proxy (USO)
    • Gold proxy (GLD)
    • Market breadth from SPY components (proxy)

Saves canonical macro_state.json used by:
    - regime_detector
    - context_state
    - policy_engine
    - supervisor_agent
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, Any

import yfinance as yf

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log, safe_float


# ==============================================================================
# Helper: get last two bars from yfinance
# ==============================================================================

def _yf_last2(symbol: str) -> Dict[str, float]:
    """
    Returns:
        {
            "close": float,
            "pct": float (daily change %),
            "raw": yfinance DF (optional)
        }
    """
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        df = df.dropna()
        if len(df) < 2:
            return {"close": 0.0, "pct": 0.0}

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = safe_float(last["Close"])
        prev_close = safe_float(prev["Close"])

        pct = ((close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

        return {"close": close, "pct": pct}

    except Exception as e:
        log(f"⚠️ yfinance fetch failed for {symbol}: {e}")
        return {"close": 0.0, "pct": 0.0}


# ==============================================================================
# Market breadth proxy (SPY up/down ratio)
# ==============================================================================

def _estimate_breadth() -> float:
    """
    Lightweight breadth proxy:
        if SPY is up → positive breadth
        if SPY down → negative breadth
    """
    spy = _yf_last2("SPY")
    return safe_float(spy["pct"]) / 100.0  # convert % to decimal


# ==============================================================================
# MAIN MACRO BUILDER
# ==============================================================================

def build_macro_features() -> Dict[str, Any]:
    """
    Builds macro_state.json with normalized macro signals.
    Called from nightly_job.py step 3.
    """

    log("🌐 Fetching macro signals (VIX, SPY, QQQ, TNX, DXY, GLD, USO)…")

    # ---------------------------------------------
    # Primary macro instruments
    # ---------------------------------------------
    vix = _yf_last2("^VIX")
    spy = _yf_last2("SPY")
    qqq = _yf_last2("QQQ")
    tnx = _yf_last2("^TNX")
    dxy = _yf_last2("DX-Y.NYB")   # Dollar Index
    gld = _yf_last2("GLD")        # Gold
    uso = _yf_last2("USO")        # Oil proxy

    breadth_proxy = _estimate_breadth()

    macro_state = {
        "timestamp": datetime.now(TIMEZONE).isoformat(),

        # Volatility
        "vix_close": safe_float(vix["close"]),
        "vix_daily_pct": safe_float(vix["pct"]),

        # Broad equities
        "spy_close": safe_float(spy["close"]),
        "spy_daily_pct": safe_float(spy["pct"]),

        "qqq_close": safe_float(qqq["close"]),
        "qqq_daily_pct": safe_float(qqq["pct"]),

        # Rates / yield curve
        "tnx_close": safe_float(tnx["close"]),
        "tnx_daily_pct": safe_float(tnx["pct"]),

        # FX
        "dxy_close": safe_float(dxy["close"]),
        "dxy_daily_pct": safe_float(dxy["pct"]),

        # Commodities
        "gld_close": safe_float(gld["close"]),
        "gld_daily_pct": safe_float(gld["pct"]),

        "uso_close": safe_float(uso["close"]),
        "uso_daily_pct": safe_float(uso["pct"]),

        # Breadth proxy
        "breadth_proxy": safe_float(breadth_proxy),
    }

    # ---------------------------------------------
    # Save macro_state.json
    # ---------------------------------------------
    out_path = PATHS["macro_state"]
    try:
        out_path.write_text(json.dumps(macro_state, indent=2), encoding="utf-8")
        log(f"📈 macro_state.json updated → {out_path}")
    except Exception as e:
        log(f"⚠️ Failed to write macro_state.json: {e}")

    return macro_state


# Standalone CLI
if __name__ == "__main__":
    res = build_macro_features()
    print(json.dumps(res, indent=2))
