"""
news_fetcher.py — v3.0
Aligned with backend/core + nightly_job v4.0

Responsibilities:
    • Periodically fetch business/market news from multiple providers
    • Normalize into a single article schema
    • Categorize articles (stock / macro / general)
    • Extract tickers heuristically
    • Maintain an in-memory + on-disk cache
    • Produce a small dashboard JSON for the rest of the system

Providers (best-effort, all optional):
    - Finnhub (company & market news)
    - NewsAPI (business headlines)
    - rss2json (RSS feeds → JSON)

This module is designed to be SAFE:
    • If any provider fails, others still work.
    • If keys are missing, it logs and continues.
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiohttp

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log

# ====================== API KEYS (optional) ======================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
RSS2JSON_KEY = os.getenv("RSS2JSON_KEY", "")

# ====================== PATHS / CACHE ============================

NEWS_CACHE_DIR = PATHS["news_cache"]
NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MASTER_NEWS_FILE = NEWS_CACHE_DIR / "news_cache.json"
DASHBOARD_FILE = PATHS["news_dashboard_json"]

# In-memory cache of normalized articles
NEWS_CACHE: List[Dict[str, Any]] = []


# ====================== BASIC HELPERS ============================

def _now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_cache() -> None:
    """Load news cache from disk to memory (if present)."""
    global NEWS_CACHE
    if MASTER_NEWS_FILE.exists():
        try:
            NEWS_CACHE = json.loads(MASTER_NEWS_FILE.read_text(encoding="utf-8"))
            log(f"[news_fetcher] Loaded {len(NEWS_CACHE)} articles from cache.")
        except Exception as e:
            log(f"[news_fetcher] ⚠️ Failed to load news cache: {e}")
            NEWS_CACHE = []
    else:
        NEWS_CACHE = []


def _save_daily_snapshot(articles: List[Dict[str, Any]]) -> None:
    """Save a per-day snapshot alongside the master cache."""
    try:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        daily_file = NEWS_CACHE_DIR / f"{date_str}.json"
        daily_file.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[news_fetcher] ⚠️ Failed to write daily news file: {e}")


def save_cache() -> None:
    """Persist NEWS_CACHE to disk (master + daily)."""
    global NEWS_CACHE
    safe = NEWS_CACHE or []
    try:
        MASTER_NEWS_FILE.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_daily_snapshot(safe)
        log(f"[news_fetcher] 💾 Saved {len(safe)} articles to cache.")
    except Exception as e:
        log(f"[news_fetcher] ⚠️ Failed to save news cache: {e}")


# ====================== NORMALIZATION / CATEGORIZATION ===========

_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")


def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    # crude heuristic; later layers (news_intel) can refine
    return list({m.group(0) for m in _TICKER_PATTERN.finditer(text)})


def categorize_article(article: Dict[str, Any]) -> str:
    text = " ".join([
        article.get("title", "") or "",
        article.get("summary", "") or "",
        article.get("source_name", "") or "",
    ]).lower()

    tickers = article.get("tickers") or []
    if tickers:
        return "stock"

    macro_terms = [
        "fed", "federal reserve", "interest rate", "inflation", "tariff",
        "trade war", "sanction", "gdp", "unemployment", "treasury", "yield",
        "bond", "congress", "government", "white house", "ecb", "central bank",
    ]
    if any(term in text for term in macro_terms):
        return "macro"

    return "general"


def _normalize_article(
    *,
    source: str,
    title: str,
    summary: str,
    url: str,
    published_at: str | None,
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    full_text = " ".join(filter(None, [title or "", summary or ""]))
    tickers = extract_tickers(full_text)

    article = {
        "id": raw.get("id") or raw.get("url") or f"{source}:{hash(url)}",
        "source": source,
        "source_name": raw.get("source") or raw.get("source_name") or source,
        "title": title or "",
        "summary": summary or "",
        "url": url or "",
        "published_at": published_at or raw.get("datetime") or _now_iso(),
        "tickers": tickers,
        "category": "",   # filled by categorize_article
        "sentiment": 0.0, # placeholder, refined by news_intel
        "buzz": 1,        # simple proxy; news_intel can adjust
        "novelty": 0.0,   # to be filled down the line
        "raw": raw,
    }

    article["category"] = categorize_article(article)
    return article


# ====================== PROVIDER FETCHERS (ASYNC) ================

async def _fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] | None = None) -> Any:
    try:
        async with session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception as e:
        log(f"[news_fetcher] ⚠️ Request failed for {url}: {e}")
        return None


async def _fetch_finnhub(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []

    url = "https://finnhub.io/api/v1/news"
    params = {"category": "general", "token": FINNHUB_API_KEY}
    js = await _fetch_json(session, url, params=params) or []
    out: List[Dict[str, Any]] = []

    for item in js:
        title = item.get("headline") or ""
        summary = item.get("summary") or ""
        url = item.get("url") or ""
        ts = item.get("datetime")
        pub = datetime.utcfromtimestamp(ts).isoformat() if ts else None

        art = _normalize_article(
            source="finnhub",
            title=title,
            summary=summary,
            url=url,
            published_at=pub,
            raw=item,
        )
        out.append(art)

    return out


async def _fetch_newsapi(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWSAPI_KEY,
    }
    js = await _fetch_json(session, url, params=params) or {}
    arts = js.get("articles") or []
    out: List[Dict[str, Any]] = []

    for item in arts:
        src = (item.get("source") or {}).get("name") or "newsapi"
        title = item.get("title") or ""
        summary = item.get("description") or ""
        url = item.get("url") or ""
        pub = item.get("publishedAt")

        art = _normalize_article(
            source="newsapi",
            title=title,
            summary=summary,
            url=url,
            published_at=pub,
            raw={"source": src, **item},
        )
        out.append(art)

    return out


async def _fetch_rss2json(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    if not RSS2JSON_KEY:
        return []

    # Example: market-focused RSS feed
    feed_url = "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    url = "https://api.rss2json.com/v1/api.json"
    params = {"rss_url": feed_url, "api_key": RSS2JSON_KEY}
    js = await _fetch_json(session, url, params=params) or {}
    items = js.get("items") or []
    out: List[Dict[str, Any]] = []

    for item in items:
        title = item.get("title") or ""
        summary = item.get("description") or ""
        link = item.get("link") or ""
        pub = item.get("pubDate")

        art = _normalize_article(
            source="rss2json",
            title=title,
            summary=summary,
            url=link,
            published_at=pub,
            raw=item,
        )
        out.append(art)

    return out


# ====================== UPDATE PIPELINE ==========================

def _dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by URL or id, newest first."""
    seen = set()
    deduped: List[Dict[str, Any]] = []
    # sort descending by published_at for stability
    articles_sorted = sorted(
        articles,
        key=lambda a: a.get("published_at") or "",
        reverse=True,
    )

    for art in articles_sorted:
        key = art.get("url") or art.get("id")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(art)
    return deduped


def _build_dashboard_snapshot(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a small dashboard object used by:
        - news_intel
        - frontend summaries
    """
    macro = [a for a in articles if a.get("category") == "macro"][:25]
    stock = [a for a in articles if a.get("category") == "stock"][:50]
    general = [a for a in articles if a.get("category") == "general"][:25]

    return {
        "generated_at": _now_iso(),
        "counts": {
            "total": len(articles),
            "macro": len(macro),
            "stock": len(stock),
            "general": len(general),
        },
        "macro": [{"title": a["title"], "url": a["url"]} for a in macro],
        "stock": [{"title": a["title"], "url": a["url"], "tickers": a["tickers"]} for a in stock],
        "general": [{"title": a["title"], "url": a["url"]} for a in general],
    }


async def update_news() -> Dict[str, Any]:
    """
    Fetch latest headlines from all configured providers, merge into NEWS_CACHE,
    and write cache + dashboard files.
    """
    global NEWS_CACHE

    log("[news_fetcher] 🚀 Updating news cache…")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_finnhub(session),
            _fetch_newsapi(session),
            _fetch_rss2json(session),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    new_articles: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, Exception):
            log(f"[news_fetcher] ⚠️ Provider error: {res}")
            continue
        new_articles.extend(res or [])

    if not new_articles:
        log("[news_fetcher] ⚠️ No new articles fetched.")
        return {"fetched": 0, "cached": len(NEWS_CACHE)}

    # Merge with existing cache, then dedupe
    merged = (NEWS_CACHE or []) + new_articles
    merged = _dedupe_articles(merged)

    # Keep a reasonable window (e.g., last 1000 articles)
    NEWS_CACHE = merged[:1000]

    # Save cache + dashboard
    save_cache()

    try:
        dashboard = _build_dashboard_snapshot(NEWS_CACHE)
        DASHBOARD_FILE.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[news_fetcher] 📊 Dashboard snapshot written → {DASHBOARD_FILE}")
    except Exception as e:
        log(f"[news_fetcher] ⚠️ Failed to write dashboard JSON: {e}")
        dashboard = {}

    return {
        "fetched": len(new_articles),
        "cached": len(NEWS_CACHE),
        "dashboard_counts": dashboard.get("counts", {}),
    }


# ====================== BACKGROUND LOOP ==========================

def start_news_loop(interval_seconds: int = 300) -> None:
    """
    Start a background thread that runs update_news() every `interval_seconds`.
    This is safe to call from backend_service on startup.
    """
    async def _loop():
        while True:
            try:
                await update_news()
            except Exception as e:
                log(f"[news_fetcher] ⚠️ News loop error: {e}")
            await asyncio.sleep(interval_seconds)

    t = threading.Thread(target=lambda: asyncio.run(_loop()), daemon=True)
    t.start()
    log("[news_fetcher] 📰 News background loop started.")


# ====================== CLI / TEST ==============================

if __name__ == "__main__":
    load_cache()
    # One-shot update
    asyncio.run(update_news())
