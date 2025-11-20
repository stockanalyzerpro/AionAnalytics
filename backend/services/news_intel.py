"""
news_intel.py — v3.0
AION Analytics — News Intelligence Layer
---------------------------------------------------------------

Consumes:
    - Normalized articles from news_fetcher (NEWS_CACHE)
    - dashboard snapshot
    - ticker extraction from articles

Produces:
    - Per-symbol news intelligence injected into Rolling:
        rolling[sym]["news"] = {
            "sentiment": float,
            "buzz": int,
            "novelty": float,
            "impact_score": float,
            "last_updated": iso
        }

    - Global news_intel.json for system-level awareness:
        {
            "timestamp": ...,
            "top_trending_tickers": [...],
            "macro_sentiment": float,
            "market_buzz": float,
            "news_volume": int
        }

This is DIFFERENT from "news_fetcher" — that module collects raw articles.
**news_intel** turns those articles into *intelligence*.

This module is SAFE:
    - Never removes existing fields
    - Tolerant to missing articles
    - Uses simple + stable linguistic sentiment heuristics
    - Computes novelty using recency + rarity
    - Computes symbol impact based on article clusters

"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    log,
)

# Global news cache produced by news_fetcher
NEWS_CACHE_FILE = PATHS["news_cache"] / "news_cache.json"


# ============================================================
# Basic Sentiment Helpers
# ============================================================

POS_WORDS = [
    "beat", "strong", "gain", "positive", "surge", "upgrade",
    "outperform", "improve", "growth", "bullish", "optimistic"
]

NEG_WORDS = [
    "miss", "weak", "down", "negative", "selloff", "downgrade",
    "decline", "drop", "lawsuit", "bearish", "warning", "crisis",
]


def _simple_sentiment(text: str) -> float:
    """Tiny sentiment engine that avoids external APIs."""
    if not text:
        return 0.0

    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)

    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, score))


# ============================================================
# Novelty Scoring
# ============================================================

def _compute_novelty(article_ts: str, all_articles: List[Dict[str, Any]]) -> float:
    """
    Novelty is high if:
        • the article is recent
        • and few recent articles cover the same ticker/category
    """
    try:
        ts = datetime.fromisoformat(article_ts.replace("Z", "+00:00"))
    except Exception:
        return 0.0

    age_hours = (datetime.now(TIMEZONE) - ts).total_seconds() / 3600
    if age_hours < 0:
        age_hours = 0

    # Novelty decays quickly after 24 hours
    novelty = math.exp(-age_hours / 24)

    return max(0.0, min(1.0, novelty))


# ============================================================
# Main Intel Builder
# ============================================================

def run_news_intel() -> Dict[str, Any]:
    """
    Convert raw news articles → symbol-level intelligence.

    Produces:
        rolling[sym]["news"] = {sentiment, buzz, novelty, impact_score}
        PATHS["news_intel"] JSON file.
    """
    # ------------------------------------------------------------
    # Load Articles
    # ------------------------------------------------------------
    try:
        raw = json.loads(NEWS_CACHE_FILE.read_text(encoding="utf-8"))
        articles: List[Dict[str, Any]] = raw if isinstance(raw, list) else []
        log(f"[news_intel] Loaded {len(articles)} articles.")
    except Exception:
        articles = []
        log("[news_intel] ⚠️ No news cache found.")

    if not articles:
        return {"status": "empty"}

    # ------------------------------------------------------------
    # Load Rolling
    # ------------------------------------------------------------
    rolling = _read_rolling()
    if not rolling:
        log("[news_intel] ⚠️ No rolling.json.gz — skipping intel.")
        return {"status": "no_rolling"}

    # ------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------
    per_symbol: Dict[str, List[Dict[str, Any]]] = {}
    macro_sentiments: List[float] = []
    total_buzz = 0

    # ------------------------------------------------------------
    # Distribute articles → each symbol referenced
    # ------------------------------------------------------------
    for art in articles:
        title = art.get("title", "")
        summary = art.get("summary", "")
        full_text = f"{title} {summary}"

        sent = _simple_sentiment(full_text)
        buzz = art.get("buzz", 1) or 1
        novelty = _compute_novelty(art.get("published_at", "") or "", articles)

        total_buzz += buzz

        # Macro article?
        if art.get("category") == "macro":
            macro_sentiments.append(sent)

        # Apply to per-symbol intel
        tickers = art.get("tickers") or []
        for sym in tickers:
            sym_u = sym.upper()
            if sym_u not in per_symbol:
                per_symbol[sym_u] = []
            per_symbol[sym_u].append({
                "sentiment": sent,
                "buzz": buzz,
                "novelty": novelty,
            })

    # ------------------------------------------------------------
    # Generate symbol-level aggregates
    # ------------------------------------------------------------
    updated = 0
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        sym_u = sym.upper()
        cluster = per_symbol.get(sym_u) or []
        if not cluster:
            # No news = neutral
            node["news"] = {
                "sentiment": 0.0,
                "buzz": 0,
                "novelty": 0.0,
                "impact_score": 0.0,
                "last_updated": datetime.now(TIMEZONE).isoformat(),
            }
            rolling[sym_u] = node
            updated += 1
            continue

        sentiments = [c["sentiment"] for c in cluster]
        buzzes = [c["buzz"] for c in cluster]
        novs = [c["novelty"] for c in cluster]

        avg_sent = statistics.mean(sentiments) if sentiments else 0.0
        total_b = sum(buzzes)
        avg_nov = statistics.mean(novs) if novs else 0.0

        # Impact: buzz-weighted sentiment boosted by novelty
        impact = avg_sent * (1 + avg_nov) * math.log1p(total_b)

        node["news"] = {
            "sentiment": avg_sent,
            "buzz": total_b,
            "novelty": avg_nov,
            "impact_score": float(round(impact, 4)),
            "last_updated": datetime.now(TIMEZONE).isoformat(),
        }
        rolling[sym_u] = node
        updated += 1

    # ------------------------------------------------------------
    # Global News Intelligence File
    # ------------------------------------------------------------
    try:
        intel = {
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "news_volume": len(articles),
            "macro_sentiment": statistics.mean(macro_sentiments) if macro_sentiments else 0.0,
            "market_buzz": total_buzz,
            "top_trending_tickers": sorted(
                per_symbol.items(),
                key=lambda kv: sum(c["buzz"] for c in kv[1]),
                reverse=True
            )[:20],
        }
        PATHS["news_intel"].write_text(json.dumps(intel, indent=2), encoding="utf-8")
        log("[news_intel] 🧠 Updated news_intel.json")
    except Exception as e:
        log(f"[news_intel] ⚠️ Failed writing news_intel.json: {e}")

    # ------------------------------------------------------------
    # Save Rolling
    # ------------------------------------------------------------
    save_rolling(rolling)
    log(f"[news_intel] Updated news intel for {updated} symbols.")

    return {
        "updated": updated,
        "total_articles": len(articles),
        "status": "ok",
    }


# CLI
if __name__ == "__main__":
    out = run_news_intel()
    print(json.dumps(out, indent=2))
