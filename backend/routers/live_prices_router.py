# backend/routers/live_prices_router.py — v4.0
"""
LIVE PRICES ROUTER — AION Analytics

This version:
    • Uses StockAnalysis /s/i/ for fast batch snapshots
    • Uses backend.services.intraday_fetcher for live 1m bars (Alpaca/IDX)
    • Uses YFinance fallback outside market hours or on failure
    • Validates symbol list
    • Returns clean OHLCV + snapshot structure
"""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException
import yfinance as yf

from backend.services.intraday_fetcher import fetch_intraday_bars
from backend.core.config import TIMEZONE

try:
    from dt_backend.dt_logger import dt_log as log
except Exception:  # pragma: no cover
    def log(msg: str):
        print(msg, flush=True)


# ==========================================================
# --- StockAnalysis LIVE Snapshot Fetcher ------------------
# ==========================================================
BASE_URL = "https://stockanalysis.com/api/screener/s/i/"


def fetch_stockanalysis_snapshot() -> dict:
    """Fetch the whole StockAnalysis /s/i/ snapshot once and index by symbol."""
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
# --- Intraday Bars (YF fallback) --------------------------
# ==========================================================
def fetch_yf_intraday(symbol: str, minutes: int = 390) -> list[dict]:
    """Yahoo Finance intraday fallback."""
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

        records = df.to_dict(orient="records")
        # ensure ISO timestamps
        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, (datetime, )):
                r["timestamp"] = ts.isoformat()
        return records

    except Exception as e:
        log(f"[live_prices] ⚠️ YF intraday failed {symbol}: {e}")
        return []


# ==========================================================
# --- Market-hours helper ---------------------------------
# ==========================================================
def _is_us_market_hours(now: datetime | None = None) -> bool:
    """
    Rough US equities market-hours check: 9:30–16:00 ET on weekdays.
    We use backend TIMEZONE and treat it as US central/ET aligned.
    """
    now = now or datetime.now(TIMEZONE)
    # weekday: Monday=0, Sunday=6
    if now.weekday() >= 5:
        return False
    hour = now.hour
    minute = now.minute
    total_minutes = hour * 60 + minute
    open_m = 9 * 60 + 30   # 09:30
    close_m = 16 * 60      # 16:00
    return open_m <= total_minutes <= close_m


# ==========================================================
# --- Helper: Validate symbol list -------------------------
# ==========================================================
def _clean_symbols(symbols: List[str]) -> List[str]:
    """
    Filters out StockAnalysis placeholders and malformed symbols.
    Only allows simple alpha-heavy tickers up to length 6.
    """
    good = []
    for s in symbols:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        if (
            s
            and s.isalpha()
            and len(s) <= 6
            and s not in ("TIMESTAMP", "SYMBOLS", "BARS", "PRICES")
        ):
            good.append(s)
    return good


# ==========================================================
# --- Router ------------------------------------------------
# ==========================================================
router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/prices")
async def api_live_prices(
    symbols: Optional[str] = Query(None, description="Comma-separated list (AAPL,MSFT)."),
    limit: int = Query(50, ge=1, le=500),
    include_intraday: bool = Query(False, description="Whether to include OHLCV 1m bars."),
):
    """
    Returns StockAnalysis live snapshot + optional 1-minute intraday bars.

    Example:
        /api/live/prices?symbols=AAPL,MSFT&include_intraday=true
    """
    now = datetime.now(TIMEZONE)

    # ------------------------------------------------------
    # 1) Parse symbols or use top of rolling_dt
    # ------------------------------------------------------
    if symbols:
        symbol_list = _clean_symbols([s.strip() for s in symbols.split(",")])
    else:
        # fallback: read intraday rolling universe
        from dt_backend.core.data_pipeline_dt import _read_rolling as _read_rolling_dt
        r = _read_rolling_dt() or {}
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
            "ts": now.isoformat(),
        }

        # --------------------------------------------------
        # 3) Optional: include 1-minute intraday OHLCV bars
        # --------------------------------------------------
        if include_intraday:
            bars: list[dict] = []
            if _is_us_market_hours(now):
                bars = fetch_intraday_bars(sym) or []
                if not bars:
                    # Fallback to YF if intraday fetcher failed
                    bars = fetch_yf_intraday(sym)
            else:
                # Off-hours → go straight to YF
                bars = fetch_yf_intraday(sym)
            entry["intraday_bars"] = bars

        results.append(entry)

    return {
        "timestamp": now.isoformat(),
        "count": len(results),
        "symbols": symbol_list,
        "results": results,
    }
