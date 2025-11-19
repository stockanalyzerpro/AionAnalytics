# backend/routers/live_prices_router.py
"""
LIVE PRICES ROUTER — UPDATED FOR PHASE 4 BACKEND

This version:
    • Uses StockAnalysis /s/i/ for fast batch snapshots (same as before)
    • Adds Alpaca IDX 1-minute bar fallback (market hours)
    • Adds YFinance fallback (off-hours & weekends)
    • Ensures valid symbol filtering
    • Ensures clean OHLCV formatting
    • Removes deprecated backend.data_pipeline import path
    • Integrates with backend/services/intraday_fetcher.py
"""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException
import yfinance as yf

# NEW: Uses centralized intraday fetcher logic
from backend.services.intraday_fetcher import fetch_intraday_bars

try:
    from dt_backend.dt_logger import dt_log as log
except:
    def log(msg: str):
        print(msg, flush=True)


# ==========================================================
# --- StockAnalysis LIVE Snapshot Fetcher ------------------
# ==========================================================
BASE_URL = "https://stockanalysis.com/api/screener/s/i/"


def fetch_stockanalysis_snapshot() -> dict:
    """Fetches the whole StockAnalysis /s/i/ snapshot once."""
    try:
        r = requests.get(BASE_URL, timeout=15)
        if r.status_code != 200:
            log(f"[live_prices] ⚠️ StockAnalysis returned {r.status_code}")
            return {}

        j = r.json()
        data = (j.get("data") or {}).get("data", [])
        if not data:
            log("[live_prices] ⚠️ StockAnalysis snapshot returned no data")
            return {}

        out = {}
        for row in data:
            sym = str(row.get("s", "")).upper()
            if not sym:
                continue
            out[sym] = {
                "symbol": sym,
                "name": row.get("n"),
                "price": row.get("price"),
                "change": row.get("change"),
                "industry": row.get("industry"),
                "volume": row.get("volume"),
                "marketCap": row.get("marketCap"),
                "pe_ratio": row.get("peRatio"),
            }
        return out

    except Exception as e:
        log(f"[live_prices] ⚠️ StockAnalysis batch fetch error: {e}")
        return {}


# ==========================================================
# --- Intraday Bars (YF fallback used in your old file) ----
# ==========================================================
def fetch_yf_intraday(symbol: str, minutes: int = 390):
    """Yahoo Finance intraday fallback used previously."""
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes)

    try:
        df = yf.download(
            symbol,
            interval="1m",
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return []

        df = df.reset_index().rename(
            columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return df.to_dict(orient="records")

    except Exception as e:
        log(f"[live_prices] ⚠️ YF intraday failed {symbol}: {e}")
        return []


# ==========================================================
# --- Helper: Validate symbol list -------------------------
# ==========================================================
def _clean_symbols(symbols: List[str]) -> List[str]:
    """
    Filters out StockAnalysis placeholders and malformed symbols.
    """
    good = []
    for s in symbols:
        if (
            isinstance(s, str)
            and s.isalpha()
            and 0 < len(s) <= 6
            and s not in ("TIMESTAMP", "SYMBOLS", "BARS", "PRICES")
        ):
            good.append(s.upper())
    return good


# ==========================================================
# --- Router ------------------------------------------------
# ==========================================================
router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/prices")
async def api_live_prices(
    symbols: Optional[str] = Query(None, description="Comma-separated list (AAPL,MSFT)."),
    limit: int = Query(50, ge=1, le=500),
    include_intraday: bool = Query(False, description="Whether to include OHLCV 1m bars.")
):
    """
    Returns **StockAnalysis live snapshot** + optional **intraday bars**.

    Example:
        /api/live/prices?symbols=AAPL,MSFT&include_intraday=true
    """

    # ------------------------------------------------------
    # 1) Parse symbols or use top of rolling
    # ------------------------------------------------------
    if symbols:
        symbol_list = _clean_symbols([s.strip() for s in symbols.split(",")])
    else:
        # fallback: read rolling tickers
        from dt_backend.core.data_pipeline_dt import _read_rolling
        r = _read_rolling() or {}
        symbol_list = _clean_symbols(list(r.keys())[:limit])

    if not symbol_list:
        raise HTTPException(status_code=400, detail="No valid symbols provided.")

    # ------------------------------------------------------
    # 2) Fetch StockAnalysis snapshot ONCE, then filter
    # ------------------------------------------------------
    snapshot = fetch_stockanalysis_snapshot()
    results = []

    for sym in symbol_list:
        row = snapshot.get(sym) or {}
        entry = {
            "symbol": sym,
            "name": row.get("name"),
            "price": row.get("price"),
            "change": row.get("change"),
            "industry": row.get("industry"),
            "volume": row.get("volume"),
            "marketCap": row.get("marketCap"),
            "pe_ratio": row.get("pe_ratio"),
            "ts": datetime.utcnow().isoformat(),
        }

        # --------------------------------------------------
        # 3) Optional: include 1-minute intraday OHLCV bars
        # --------------------------------------------------
        if include_intraday:
            bars = fetch_intraday_bars(sym)
            if not bars:
                bars = fetch_yf_intraday(sym)
            entry["intraday_bars"] = bars

        results.append(entry)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(results),
        "symbols": symbol_list,
        "results": results,
    }
