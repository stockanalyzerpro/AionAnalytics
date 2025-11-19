"""
AION Analytics — Nightly Context Builder (Advanced Mode)
--------------------------------------------------------

This module builds **market-level** and **symbol-level** context to strengthen
model awareness — similar to human traders.

Context gathers signals from:
    ✓ Rolling cache (fundamentals, metrics, predictions)
    ✓ Macro features (saved by macro_fetcher.py)
    ✓ Breadth / market momentum
    ✓ Sentiment (from news_intel + social sentiment)
    ✓ Rolling Brain (hit ratios, drift)
    ✓ Regime Detector output

Outputs:
    • rolling[sym]["context"] = {...}
    • PATHS["market_state"] (global context snapshot)

This mirrors dt_backend's context_state_dt.py philosophy,
but adapted for nightly EOD behavior.
"""

from __future__ import annotations
import math
import json
import statistics
from pathlib import Path
from typing import Dict, Any, Optional

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    save_rolling,
    safe_float,
    log,
)
from backend.core.regime_detector import detect_regime


# ============================================================
# Helpers
# ============================================================

def _safe_mean(values):
    try:
        return float(statistics.mean(values)) if values else 0.0
    except Exception:
        return 0.0


def _safe_median(values):
    try:
        return float(statistics.median(values)) if values else 0.0
    except Exception:
        return 0.0


def _ensure_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Symbol-Level Context
# ============================================================

def _build_symbol_context(sym: str, node: Dict[str, Any], brain: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-symbol context similar to intraday context,
    but using daily bars + fundamentals.
    """
    ctx = {}

    metrics = node.get("metrics", {})
    fund = node.get("fundamentals", {})
    preds = node.get("predictions", {})
    news = node.get("news", {})
    hist = node.get("history", [])

    # ----- Basic Momentum / Change -----
    if hist and len(hist) >= 2:
        try:
            last = safe_float(hist[-1].get("close"))
            prev = safe_float(hist[-2].get("close"))
            ctx["daily_return"] = (last - prev) / prev if prev > 0 else 0.0
        except Exception:
            ctx["daily_return"] = 0.0
    else:
        ctx["daily_return"] = 0.0

    # ----- Volatility Proxy -----
    if hist and len(hist) >= 10:
        closes = [safe_float(h.get("close")) for h in hist[-10:] if h.get("close")]
        if len(closes) >= 2:
            try:
                returns = [(closes[i] - closes[i-1]) / closes[i-1]
                           for i in range(1, len(closes))]
                ctx["volatility"] = _safe_mean([abs(r) for r in returns])
            except Exception:
                ctx["volatility"] = 0.0
        else:
            ctx["volatility"] = 0.0
    else:
        ctx["volatility"] = 0.0

    # ----- Sentiment Signals -----
    ctx["sentiment"] = safe_float(news.get("sentiment", 0.0))
    ctx["buzz"] = int(news.get("buzz", 0))
    ctx["novelty"] = safe_float(news.get("novelty", 0.0))

    # ----- Fundamentals Highlights -----
    ctx["pe_ratio"] = safe_float(fund.get("pe_ratio") or fund.get("pe"), 0.0)
    ctx["earnings_yield"] = 1.0 / ctx["pe_ratio"] if ctx["pe_ratio"] > 0 else 0.0
    ctx["marketcap"] = safe_float(fund.get("marketcap", 0))

    # ----- Prediction-Based Context -----
    # Flatten prediction structure for nightly preview
    if preds:
        first_horizon = next(iter(preds), None)
        if first_horizon:
            p = preds.get(first_horizon, {})
            ctx["predicted_price"] = safe_float(p.get("predictedPrice", 0.0))
            ctx["pred_score"] = safe_float(p.get("rankingScore", 0.0))
        else:
            ctx["predicted_price"] = 0.0
            ctx["pred_score"] = 0.0
    else:
        ctx["predicted_price"] = 0.0
        ctx["pred_score"] = 0.0

    # ----- Rolling Brain (Hit Ratios / Drift) -----
    bnode = brain.get(sym, {})
    ctx["hit_ratio_30"] = safe_float(bnode.get("hit_ratio_30", 0.5))
    ctx["drift_score"] = safe_float(bnode.get("drift_score", 0.0))

    # ----- Trend Flags -----
    if ctx["daily_return"] > 0.01:
        ctx["trend"] = "bullish"
    elif ctx["daily_return"] < -0.01:
        ctx["trend"] = "bearish"
    else:
        ctx["trend"] = "neutral"

    return ctx


# ============================================================
# Market-Level Context
# ============================================================

def _build_market_context(rolling: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds global market snapshot: breadth, vol proxy, average sentiment, etc.
    Written to PATHS["market_state"].
    """
    returns = []
    vols = []
    sentiments = []
    buzzes = []

    for sym, node in rolling.items():
        if sym.startswith("_"):  # skip global/intraday nodes
            continue

        hist = node.get("history", [])
        if hist and len(hist) >= 2:
            try:
                last = safe_float(hist[-1].get("close"))
                prev = safe_float(hist[-2].get("close"))
                returns.append((last - prev) / prev if prev > 0 else 0.0)
            except Exception:
                pass

        # Volatility proxy (last 5 bars)
        if hist and len(hist) >= 5:
            closes = [safe_float(h.get("close")) for h in hist[-5:] if h.get("close")]
            if len(closes) >= 2:
                try:
                    diffs = [
                        abs(closes[i] - closes[i - 1]) / closes[i - 1]
                        for i in range(1, len(closes))
                    ]
                    vols.append(_safe_mean(diffs))
                except Exception:
                    pass

        news = node.get("news", {})
        sentiments.append(safe_float(news.get("sentiment", 0.0)))
        buzzes.append(int(news.get("buzz", 0)))

    breadth_up = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0

    mkt = {
        "breadth_up": breadth_up,
        "median_return": _safe_median(returns),
        "volatility_proxy": _safe_mean(vols),
        "avg_sentiment": _safe_mean(sentiments),
        "avg_buzz": _safe_mean(buzzes),
    }

    return mkt


# ============================================================
# MAIN ENTRY
# ============================================================

def build_context() -> Dict[str, Any]:
    """
    Full nightly context builder called by:
        backend/jobs/nightly_job.py

    Produces:
        • per symbol context
        • market_state.json
        • enriched rolling cache
    """
    rolling = _read_rolling()
    brain = _read_brain()

    if not rolling:
        log("⚠️ No rolling.json.gz found — context builder exiting.")
        return {}

    # --------------------------------------------------------
    # 1. Symbol-Level Context
    # --------------------------------------------------------
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        ctx = _build_symbol_context(sym, node, brain)
        node["context"] = ctx
        rolling[sym] = node

    # --------------------------------------------------------
    # 2. Market-Level Context
    # --------------------------------------------------------
    market_ctx = _build_market_context(rolling)

    # --------------------------------------------------------
    # 3. Regime Detection (global)
    # --------------------------------------------------------
    regime = detect_regime(rolling)

    market_ctx["regime"] = regime or {}

    # --------------------------------------------------------
    # 4. Write Global Context File
    # --------------------------------------------------------
    try:
        PATHS["market_state"].write_text(
            json.dumps(market_ctx, indent=2), encoding="utf-8"
        )
        log("🧭 market_state.json updated")
    except Exception as e:
        log(f"⚠️ Failed writing market_state.json: {e}")

    # --------------------------------------------------------
    # 5. Save Rolling (with fresh context)
    # --------------------------------------------------------
    save_rolling(rolling)

    return market_ctx
