# backend/services/intraday_fetcher.py
"""
Intraday Fetcher (Alpaca IDX + YFinance fallback)

- Fetches 1-minute intraday bars for each symbol
- Market hours → Alpaca (uses /v2/stocks/{symbol}/bars?timeframe=1Min&feed=iex)
- Off-hours → YFinance fallback
- Writes bars to: data/raw/intraday_bars/{symbol}.json
- Injects fresh bars into dt_backend rolling cache (bars_intraday)
"""

from __future__ import annotations

import json
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str):
        print(msg, flush=True)

from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node
from dt_backend.core.config_dt import DT_PATHS
from backend.env import ALPACA_KEY, ALPACA_SECRET   # assume you expose env here


# ---------------------------------------------------
# Storage path
# ---------------------------------------------------
def _intraday_storage_dir() -> Path:
    root = Path("data/raw/intraday_bars")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------
# MARKET HOURS CHECK
# ---------------------------------------------------
def _is_market_hours() -> bool:
    now = datetime.now(timezone.utc)
    # 9:30 to 16:00 ET, Mon–Fri
    et = now.astimezone(tz=None)
    if et.weekday() >= 5:  # weekend
        return False
    return (et.hour > 9 or (et.hour == 9 and et.minute >= 30)) and (et.hour < 16)


# ---------------------------------------------------
# ALPACA FETCH
# ---------------------------------------------------
def fetch_bars_alpaca(symbol: str) -> List[Dict[str, Any]]:
    """Return list of {ts, open, high, low, close, volume} bars."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "limit": 400,               # last ~6 hours
        "adjustment": "raw",
        "feed": "iex"               # required for free tier
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            log(f"[intraday_fetcher] Alpaca error {symbol}: {r.text[:200]}")
            return []
        raw = r.json().get("bars", [])
    except Exception as e:
        log(f"[intraday_fetcher] Alpaca fetch failed {symbol}: {e}")
        return []

    bars = []
    for b in raw:
        bars.append({
            "ts": b.get("t"),
            "open": b.get("o"),
            "high": b.get("h"),
            "low": b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v"),
        })
    return bars


# ---------------------------------------------------
# YFINANCE FALLBACK (weekends / after-hours)
# ---------------------------------------------------
def fetch_bars_yf(symbol: str) -> List[Dict[str, Any]]:
    try:
        df = yf.download(tickers=symbol, interval="1m", period="2d", progress=False)
        if df.empty:
            return []
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "ts": ts.isoformat(),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            })
        return bars
    except Exception as e:
        log(f"[intraday_fetcher] YF error {symbol}: {e}")
        return []


# ---------------------------------------------------
# UNIVERSAL FETCH
# ---------------------------------------------------
def fetch_intraday_bars(symbol: str) -> List[Dict[str, Any]]:
    symbol = symbol.upper().strip()

    if _is_market_hours():
        bars = fetch_bars_alpaca(symbol)
        if bars:
            return bars
        # if Alpaca fails → fallback
        return fetch_bars_yf(symbol)
    else:
        return fetch_bars_yf(symbol)


# ---------------------------------------------------
# SAVE + ROLLING INJECTION
# ---------------------------------------------------
def update_rolling_with_bars(symbol: str, bars: List[Dict[str, Any]]) -> None:
    rolling = _read_rolling() or {}
    node = ensure_symbol_node(rolling, symbol)
    node["bars_intraday"] = bars
    rolling[symbol] = node
    save_rolling(rolling)


def fetch_and_update(symbols: List[str]) -> Dict[str, Any]:
    """Fetch & write bars for all symbols, update rolling."""
    out = {"symbols": len(symbols), "fetched": 0, "errors": []}
    store = _intraday_storage_dir()

    for sym in symbols:
        bars = fetch_intraday_bars(sym)
        if not bars:
            out["errors"].append(sym)
            continue

        # Write raw
        fp = store / f"{sym}.json"
        fp.write_text(json.dumps(bars, indent=2), encoding="utf-8")

        # Inject into rolling
        update_rolling_with_bars(sym, bars)
        out["fetched"] += 1

    return out
