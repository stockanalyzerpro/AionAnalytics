"""
policy_engine.py — v5.0
AION Analytics — Unified Nightly Policy Engine (Multi-Horizon Aware)

This version merges:
    • Multi-horizon ML predictions (1d → 52w)
    • Context state (trend, volatility, sentiment)
    • News intelligence (sentiment + impact)
    • Social intelligence (heat + sentiment)
    • Macro regime (bull / bear / chop)
    • dt_backend drift brain (model stability)
    • Multi-horizon confidence fusion

Everything is designed to mimic a real human portfolio manager
evaluating short-term edge AND long-term conviction.
"""

from __future__ import annotations

from typing import Dict, Any
import math
from statistics import mean

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
# Horizon weighting (human-like)
# ============================================================

HORIZON_WEIGHTS = {
    "1d": 0.25,
    "3d": 0.25,
    "1w": 0.20,
    "2w": 0.15,
    "4w": 0.10,
    "13w": 0.04,
    "26w": 0.02,
    "52w": 0.01,
}


# ============================================================
# BUILD PREDICTION SUMMARY FROM MULTI-HORIZON
# ============================================================

def _combine_predictions(preds: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    preds = {
      "1d": {"score":.., "confidence":.., "label":..},
      "3d": {...},
      ...
    }

    Returns:
        {
          "policy_score": float,
          "avg_confidence": float,
          "short_term": float,
          "medium_term": float,
          "long_term": float
        }
    """
    if not preds:
        return {
            "policy_score": 0.0,
            "avg_confidence": 0.0,
            "short_term": 0.0,
            "medium_term": 0.0,
            "long_term": 0.0,
        }

    weighted = 0.0
    total_w = 0.0
    confidences = []

    short_list = []
    med_list = []
    long_list = []

    for h, w in HORIZON_WEIGHTS.items():
        block = preds.get(h)
        if not block:
            continue
        score = safe_float(block.get("score", 0.0))
        conf = safe_float(block.get("confidence", 0.0))

        weighted += score * w
        total_w += w
        confidences.append(conf)

        # groupings
        if h in ("1d", "3d"):
            short_list.append(score)
        elif h in ("1w", "2w"):
            med_list.append(score)
        else:
            long_list.append(score)

    if total_w == 0:
        return {
            "policy_score": 0.0,
            "avg_confidence": 0.0,
            "short_term": 0.0,
            "medium_term": 0.0,
            "long_term": 0.0,
        }

    return {
        "policy_score": weighted / total_w,
        "avg_confidence": mean(confidences) if confidences else 0.0,
        "short_term": mean(short_list) if short_list else 0.0,
        "medium_term": mean(med_list) if med_list else 0.0,
        "long_term": mean(long_list) if long_list else 0.0,
    }


# ============================================================
# CONTEXT CONFIDENCE BOOSTER
# ============================================================

def _context_effects(node: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract sentiment, volatility, trend, news, social.
    Maps them into numeric multipliers.
    """

    ctx = node.get("context") or {}
    news = node.get("news") or {}
    social = node.get("social") or {}

    sentiment = safe_float(ctx.get("sentiment", 0.0))
    volatility = safe_float(ctx.get("volatility", 0.0))
    trend = ctx.get("trend", "neutral")

    news_sent = safe_float(news.get("sentiment", 0.0))
    news_imp = safe_float(news.get("impact_score", news.get("novelty", 0.0)))
    soc_sent = safe_float(social.get("sentiment", 0.0))
    soc_heat = safe_float(social.get("heat_score", 0.0))

    # Volatility dampener
    if volatility > 0.05:
        vol_mult = 0.65
    elif volatility > 0.03:
        vol_mult = 0.75
    elif volatility > 0.02:
        vol_mult = 0.90
    else:
        vol_mult = 1.0

    # Trend bonus
    if trend == "bullish":
        trend_mult = 1.15
    elif trend == "bearish":
        trend_mult = 0.85
    else:
        trend_mult = 1.0

    # Sentiment fusion
    sent_mult = 1.0 + ((sentiment + news_sent + soc_sent) * 0.10)

    # News & social impact + heat
    impact_mult = 1.0 + (news_imp * 0.10) + (math.tanh(soc_heat) * 0.05)

    return {
        "vol_mult": vol_mult,
        "trend_mult": trend_mult,
        "sent_mult": sent_mult,
        "impact_mult": impact_mult,
        "volatility": volatility,
        "sentiment": sentiment,
        "news_sentiment": news_sent,
        "news_impact": news_imp,
        "social_sentiment": soc_sent,
        "social_heat": soc_heat,
        "trend": trend,
    }


# ============================================================
# MARKET REGIME EFFECTS
# ============================================================

def _regime_effects(regime: Dict[str, Any]) -> float:
    """
    bull → expand exposure
    bear → compress exposure
    chop → neutral
    """
    label = (regime or {}).get("label", "chop")

    if label == "bull":
        return 1.15
    if label == "bear":
        return 0.70
    return 1.0


# ============================================================
# DRIFT PENALTY (dt_backend stability)
# ============================================================

def _drift_penalty(brain_node: Dict[str, Any]) -> float:
    drift = safe_float(brain_node.get("drift_score", 0.0))
    if drift > 0.15:
        return 0.5
    if drift > 0.10:
        return 0.7
    if drift > 0.05:
        return 0.85
    return 1.0


# ============================================================
# DETERMINE INTENT
# ============================================================

def _determine_intent(score: float, ctx: Dict[str, float], regime: Dict[str, Any]) -> str:
    """
    score = combined nightly multi-horizon ML signal [-1,1]
    """

    market_regime = (regime or {}).get("label", "chop")
    trend = ctx.get("trend", "neutral")

    # BUY logic
    if score >= 0.35 and trend == "bullish":
        if market_regime in ("bull", "chop"):
            return "BUY"

    if score >= 0.45:
        if market_regime != "bear":
            return "BUY"

    # SELL logic
    if score <= -0.25:
        return "SELL"

    if trend == "bearish":
        return "SELL"

    # HOLD otherwise
    return "HOLD"


# ============================================================
# POLICY BUILDER FOR ONE SYMBOL
# ============================================================

def _build_policy(sym: str,
                  node: Dict[str, Any],
                  preds: Dict[str, Any],
                  brain_node: Dict[str, Any],
                  regime: Dict[str, Any]
                  ) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # Combine ML predictions into unified policy score
    # ---------------------------------------------------------
    pred_summary = _combine_predictions(preds)
    base_score = pred_summary["policy_score"]
    avg_conf = pred_summary["avg_confidence"]

    # ---------------------------------------------------------
    # Context multipliers
    # ---------------------------------------------------------
    ctx_eff = _context_effects(node)

    # ---------------------------------------------------------
    # Drift penalty (model instability)
    # ---------------------------------------------------------
    drift_mult = _drift_penalty(brain_node)

    # ---------------------------------------------------------
    # Market regime multiplier
    # ---------------------------------------------------------
    regime_mult = _regime_effects(regime)

    # ---------------------------------------------------------
    # Final policy score = ML × all multipliers
    # ---------------------------------------------------------
    policy_score = (
        base_score *
        ctx_eff["vol_mult"] *
        ctx_eff["trend_mult"] *
        ctx_eff["sent_mult"] *
        ctx_eff["impact_mult"] *
        drift_mult *
        regime_mult
    )

    # Clamp to [-1,1]
    policy_score = max(-1.0, min(1.0, policy_score))

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------
    confidence = avg_conf
    confidence *= (ctx_eff["vol_mult"] * ctx_eff["trend_mult"])
    confidence = max(0.0, min(1.0, confidence))

    # ---------------------------------------------------------
    # Intent
    # ---------------------------------------------------------
    intent = _determine_intent(policy_score, ctx_eff, regime)

    # ---------------------------------------------------------
    # Exposure scale
    # ---------------------------------------------------------
    exposure = abs(policy_score) * confidence

    # ---------------------------------------------------------
    # Risk limit: adapt to market regime & volatility
    # ---------------------------------------------------------
    risk = 0.02  # base 2%

    # High volatility reduces risk
    if ctx_eff["volatility"] > 0.03:
        risk *= 0.7
    elif ctx_eff["volatility"] > 0.02:
        risk *= 0.85

    # Market regime adjustments
    market_regime = (regime or {}).get("label", "chop")
    if market_regime == "bear":
        risk *= 0.7
    elif market_regime == "bull":
        risk *= 1.15

    # ---------------------------------------------------------
    # Trade gate: no BUY in deep bear
    # ---------------------------------------------------------
    trade_gate = not (market_regime == "bear" and intent == "BUY")

    # ---------------------------------------------------------
    # Final policy block
    # ---------------------------------------------------------
    return {
        "intent": intent,
        "score": round(policy_score, 4),
        "confidence": round(confidence, 4),
        "exposure_scale": round(exposure, 4),
        "trade_gate": bool(trade_gate),
        "risk": round(risk, 4),
        "reasons": {
            "multi_horizon_score": round(base_score, 4),
            "avg_confidence": round(avg_conf, 4),
            "trend": ctx_eff["trend"],
            "sentiment": ctx_eff["sentiment"],
            "news_sentiment": ctx_eff["news_sentiment"],
            "news_impact": ctx_eff["news_impact"],
            "social_sentiment": ctx_eff["social_sentiment"],
            "social_heat": ctx_eff["social_heat"],
            "volatility": ctx_eff["volatility"],
            "drift": safe_float(brain_node.get("drift_score", 0.0)),
            "market_regime": market_regime,
        }
    }


# ============================================================
# MAIN ENTRY
# ============================================================

def apply_policy() -> Dict[str, Any]:
    """
    Full nightly policy processor.
    Uses:
        • multi-horizon nightly predictions
        • context, news, social, metrics
        • macro regime
        • dt_backend drift
    """
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ No rolling.json.gz found — policy engine exiting.")
        return {}

    brain = _read_brain()
    regime = detect_regime(rolling)

    updated = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        preds = (node.get("predictions") or {})
        brain_node = brain.get(sym.upper(), {})

        policy = _build_policy(sym, node, preds, brain_node, regime)

        node["policy"] = policy
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"🧩 Unified Policy updated for {updated} symbols (v5.0).")

    return {"updated": updated, "regime": regime}
