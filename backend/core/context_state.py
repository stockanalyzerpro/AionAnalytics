"""
backend/core/context_state.py — v3.1
Adds `pred_score` for continuous_learning compatibility.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .config import PATHS
from .data_pipeline import (
    _read_rolling,
    save_rolling,
    log,
    safe_float,
)

GLBL_PATH: Path = PATHS["ml_data"] / "market_state.json"
MACRO_DIR: Path = PATHS.get("macro", PATHS["ml_data"] / "macro")
NEWS_DIR: Path = PATHS.get("news", PATHS["ml_data"] / "news")
SOCIAL_DIR: Path = PATHS.get("social", PATHS["ml_data"] / "social")


def _load_json(path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def _latest_with_prefix(folder: Path, prefix: str) -> str | None:
    if not folder.exists():
        return None
    candidates = sorted(
        [p for p in folder.glob(f"{prefix}*.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _news_map(news_js) -> Dict[str, Dict[str, Any]]:
    if not isinstance(news_js, dict):
        return {}
    data = news_js.get("tickers") or news_js.get("data") or {}
    out = {}
    for sym, block in data.items():
        if isinstance(block, dict):
            out[sym.upper()] = {
                "sentiment": safe_float(block.get("sentiment", 0.0)),
                "buzz": int(block.get("buzz", 0)),
                "novelty": safe_float(block.get("novelty", 0.0)),
            }
    return out


def _social_map(js) -> Dict[str, Dict[str, Any]]:
    if not isinstance(js, dict):
        return {}
    data = js.get("data") or js
    out = {}
    for sym, block in data.items():
        if not isinstance(block, dict):
            continue
        out[sym.upper()] = {
            "sentiment_social": safe_float(
                block.get("avg_sentiment", block.get("sentiment", 0.0))
            ),
            "buzz_social": int(block.get("buzz", 0)),
        }
    return out


def _macro_state(js) -> Dict[str, Any]:
    if not isinstance(js, dict):
        return {
            "volatility": 0.0,
            "breadth": 0.5,
            "risk_off": False,
            "regime_hint": "neutral",
        }
    return {
        "volatility": safe_float(js.get("volatility", 0.0)),
        "breadth": safe_float(js.get("breadth", 0.5)),
        "risk_off": bool(js.get("risk_off", False)),
        "regime_hint": js.get("regime", "neutral"),
    }


def build_context() -> Dict[str, Any]:
    log("[context_state] 🧠 Building context (v3.1)…")

    rolling = _read_rolling() or {}
    if not rolling:
        log("[context_state] ⚠ rolling.json.gz missing or empty.")
        return {"symbols": 0, "global": {}}

    news_file = _latest_with_prefix(NEWS_DIR, "news_intel_") \
                or (NEWS_DIR / "news_intel_daily.json")
    news_js = _load_json(news_file)
    news_map = _news_map(news_js)

    social_file = _latest_with_prefix(SOCIAL_DIR, "social_sentiment_")
    social_js = _load_json(social_file)
    social_map = _social_map(social_js)

    macro_file = _latest_with_prefix(MACRO_DIR, "macro_state_") \
                 or (MACRO_DIR / "macro_daily.json")
    macro_js = _load_json(macro_file)
    macro = _macro_state(macro_js)

    global_state = {
        "ts": datetime.utcnow().isoformat(),
        "macro": macro,
        "has_news": bool(news_map),
        "has_social": bool(social_map),
    }

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        news = news_map.get(sym, {})
        soc = social_map.get(sym, {})
        preds = node.get("predictions", {}) or {}

        # sentiment blend
        sent = (
            safe_float(news.get("sentiment", 0.0)) * 0.6 +
            safe_float(soc.get("sentiment_social", 0.0)) * 0.4
        )

        ctx = {
            "sentiment": sent,
            "buzz": int(news.get("buzz", 0)) + int(soc.get("buzz_social", 0)),
            "novelty": safe_float(news.get("novelty", 0.0)),
            "macro_vol": macro.get("volatility", 0.0),
            "macro_breadth": macro.get("breadth", 0.5),
            "risk_off": macro.get("risk_off", False),
        }

        # multi-horizon scores for context
        score_1d = safe_float(preds.get("1d", {}).get("score", 0.0))
        score_1w = safe_float(preds.get("1w", {}).get("score", 0.0))
        score_4w = safe_float(preds.get("4w", {}).get("score", 0.0))

        ctx["short_term_score"] = score_1d
        ctx["mid_term_score"] = score_1w
        ctx["long_term_score"] = score_4w

        # IMPORTANT: pred_score used by continuous_learning
        # treat it as our main short-term directional score
        ctx["pred_score"] = score_1d

        node["context"] = ctx
        rolling[sym] = node

    try:
        GLBL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GLBL_PATH, "w", encoding="utf-8") as f:
            json.dump(global_state, f, indent=2)
    except Exception as e:
        log(f"[context_state] ⚠ Failed writing global: {e}")

    save_rolling(rolling)
    log(f"[context_state] ✅ Context updated → {len(rolling)} symbols")
    return {"symbols": len(rolling), "global": global_state}


if __name__ == "__main__":
    print(json.dumps(build_context(), indent=2))
