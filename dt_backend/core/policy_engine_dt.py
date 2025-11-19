# dt_backend/core/policy_engine_dt.py — Advanced intraday policy engine.
"""
Converts model predictions + intraday context into a structured policy block
attached to each symbol in the rolling cache:

    rolling[sym]["policy_dt"] = {
        "intent": "BUY" | "SELL" | "HOLD",
        "confidence": 0.0–1.0,
        "reason": <short human-readable summary>,
        "ts": <ISO8601 UTC timestamp>
    }

Design goals
------------
• Human-like decision logic (trend, volatility, regime, signal strength)
• Stable — avoid flip-flopping between BUY/SELL on minor noise
• Safe defaults when data is missing
• Pure Python, no heavy dependencies
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log

# Optional regime detector integration
try:  # pragma: no cover - optional dependency
    from .regime_detector_dt import classify_intraday_regime
except Exception:  # pragma: no cover
    def classify_intraday_regime(context_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Fallback regime detector if the real one is unavailable."""
        return {"label": "neutral", "score": 0.0}


@dataclass
class PolicyConfig:
    """Tunable knobs for the intraday policy engine."""

    buy_threshold: float = 0.12      # p_buy - p_sell >= this → BUY
    sell_threshold: float = -0.12    # p_buy - p_sell <= this → SELL
    min_confidence: float = 0.25     # below this → HOLD
    high_conf_boost: float = 1.15    # multiplier when signal is very strong
    vol_penalty_high: float = 0.65   # scale confidence in high vol
    vol_penalty_medium: float = 0.85
    trend_boost_strong: float = 1.25
    trend_boost_mild: float = 1.10
    max_confidence: float = 0.99


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _extract_prediction(node: Dict[str, Any]) -> Tuple[str | None, Dict[str, float]]:
    """Extract predicted label + per-class probabilities from a node.

    Expected schemas (best-effort):
      • node["predictions_dt"] = {
            "label": "BUY",
            "proba": {"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1}
        }
      • or older:
            node["predictions"] = {...}
    """
    pred = node.get("predictions_dt") or node.get("predictions") or {}
    if not isinstance(pred, dict):
        return None, {}

    label = pred.get("label") or pred.get("class")
    proba_raw = pred.get("proba") or pred.get("probs") or {}
    if not isinstance(proba_raw, dict):
        proba_raw = {}

    proba = {
        k.upper(): _safe_float(v, 0.0)
        for k, v in proba_raw.items()
        if isinstance(k, str)
    }

    total = sum(proba.values())
    if total > 0:
        proba = {k: v / total for k, v in proba.items()}

    return (label.upper() if isinstance(label, str) else None), proba


def _classify_trend_and_vol(context_dt: Dict[str, Any]) -> Tuple[str, str]:
    """Return (trend_label, vol_bucket).

    Fallbacks:
      • trend_label in context_dt["intraday_trend"]
      • vol_bucket in context_dt["vol_bucket"]
      • if missing, derive from intraday_return & intraday_vol.
    """
    trend = str(context_dt.get("intraday_trend") or "").strip()
    vol_bkt = str(context_dt.get("vol_bucket") or "").strip()

    intraday_return = _safe_float(context_dt.get("intraday_return"), 0.0)
    intraday_vol = _safe_float(context_dt.get("intraday_vol"), 0.0)

    if not trend:
        if intraday_return >= 0.01:
            trend = "strong_bull"
        elif intraday_return >= 0.003:
            trend = "bull"
        elif intraday_return <= -0.01:
            trend = "strong_bear"
        elif intraday_return <= -0.003:
            trend = "bear"
        else:
            trend = "flat"

    if not vol_bkt:
        if intraday_vol >= 0.02:
            vol_bkt = "high"
        elif intraday_vol >= 0.007:
            vol_bkt = "medium"
        else:
            vol_bkt = "low"

    return trend, vol_bkt


def _intent_from_signal(
    p_buy: float,
    p_hold: float,
    p_sell: float,
    cfg: PolicyConfig,
) -> Tuple[str, float, str]:
    """Decide BUY/SELL/HOLD from probabilities and thresholds."""
    edge = p_buy - p_sell
    base_conf = max(p_buy, p_sell)

    if edge >= cfg.buy_threshold:
        intent = "BUY"
    elif edge <= cfg.sell_threshold:
        intent = "SELL"
    else:
        intent = "HOLD"

    reason = f"edge={edge:.3f}, p_buy={p_buy:.3f}, p_sell={p_sell:.3f}, p_hold={p_hold:.3f}"
    return intent, base_conf, reason


def _adjust_confidence(
    base_conf: float,
    intent: str,
    trend: str,
    vol_bkt: str,
    regime_label: str,
    cfg: PolicyConfig,
) -> Tuple[float, str]:
    """Refine confidence based on trend, volatility, and regime."""
    conf = base_conf
    detail = []

    if conf <= 0.0:
        return 0.0, "no_base_conf"

    # Volatility penalty
    if vol_bkt == "high":
        conf *= cfg.vol_penalty_high
        detail.append("vol=high")
    elif vol_bkt == "medium":
        conf *= cfg.vol_penalty_medium
        detail.append("vol=medium")
    else:
        detail.append("vol=low")

    # Trend alignment (only if non-HOLD)
    if intent == "BUY":
        if trend == "strong_bull":
            conf *= cfg.trend_boost_strong
            detail.append("trend=strong_bull")
        elif trend == "bull":
            conf *= cfg.trend_boost_mild
            detail.append("trend=bull")
    elif intent == "SELL":
        if trend == "strong_bear":
            conf *= cfg.trend_boost_strong
            detail.append("trend=strong_bear")
        elif trend == "bear":
            conf *= cfg.trend_boost_mild
            detail.append("trend=bear")

    # Regime-based adjustment
    regime_label = (regime_label or "").lower()
    if regime_label in {"crash", "stress"}:
        conf *= 0.8
        detail.append(f"regime={regime_label}")
    elif regime_label in {"bull", "uptrend"} and intent == "BUY":
        conf *= 1.05
        detail.append(f"regime={regime_label}")
    elif regime_label in {"bear", "downtrend"} and intent == "SELL":
        conf *= 1.05
        detail.append(f"regime={regime_label}")
    else:
        if regime_label:
            detail.append(f"regime={regime_label}")

    # Clip & enforce minimum for non-HOLD
    conf = max(0.0, min(cfg.max_confidence, conf))
    if intent in {"BUY", "SELL"} and conf < cfg.min_confidence:
        detail.append("conf_below_min → HOLD")
        return 0.0, "; ".join(detail)

    return conf, "; ".join(detail)


def apply_intraday_policy(cfg: PolicyConfig | None = None) -> Dict[str, Any]:
    """Main entry point.

    Reads rolling, applies advanced intraday policy per symbol, writes back.
    Returns a summary dict suitable for logging / API hooks.
    """
    cfg = cfg or PolicyConfig()
    rolling = _read_rolling()
    if not rolling:
        log("[policy_dt] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    updated = 0
    for sym, node_raw in list(rolling.items()):
        if sym.startswith("_"):
            continue

        node = ensure_symbol_node(rolling, sym)
        ctx = node.get("context_dt") or {}
        features = node.get("features_dt") or {}

        label, proba = _extract_prediction(node)
        if not proba:
            # No model output for this symbol
            continue

        p_buy = float(proba.get("BUY", 0.0))
        p_hold = float(proba.get("HOLD", 0.0))
        p_sell = float(proba.get("SELL", 0.0))

        intent, base_conf, base_reason = _intent_from_signal(p_buy, p_hold, p_sell, cfg)

        # Trend / vol / regime adjustments
        trend, vol_bkt = _classify_trend_and_vol(ctx)
        regime_info = classify_intraday_regime(
            {
                "context": ctx,
                "features": features,
                "proba": proba,
                "symbol": sym,
            }
        ) or {}
        regime_label = str(regime_info.get("label", "neutral"))

        conf_adj, detail = _adjust_confidence(
            base_conf=base_conf,
            intent=intent,
            trend=trend,
            vol_bkt=vol_bkt,
            regime_label=regime_label,
            cfg=cfg,
        )

        if conf_adj <= 0.0 or intent == "HOLD":
            final_intent = "HOLD"
        else:
            final_intent = intent

        ts = datetime.now(timezone.utc).isoformat()

        reason = (
            f"{base_reason}; trend={trend}, vol={vol_bkt}, "
            f"regime={regime_label}; adj={detail}"
        )

        node["policy_dt"] = {
            "intent": final_intent,
            "confidence": float(conf_adj),
            "reason": reason,
            "ts": ts,
        }
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[policy_dt] ✅ updated policy_dt for {updated} symbols.")
    return {"symbols": len(rolling), "updated": updated}


def main() -> None:
    apply_intraday_policy()


if __name__ == "__main__":
    main()
