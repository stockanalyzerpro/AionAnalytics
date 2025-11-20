# backend/routers/eod_bots_router.py
"""
EOD (swing) bots API — AION Analytics

Exposes:
    • GET /api/eod/status
        → Current PnL & positions per bot (using bot state + latest rolling prices)

    • GET /api/eod/logs/days
        → List of trading days with EOD bot logs

    • GET /api/eod/logs/last-day
        → Logs for latest trading day (all horizons, all bots)

    • GET /api/eod/logs/{day}
        → Logs for a specific day, separated by horizon and bot

    • GET /api/eod/logs/{day}/{horizon}/{bot_name}
        → Logs for a specific bot on a specific day & horizon
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

try:
    from backend.core.config import PATHS
except ImportError:
    from backend.config import PATHS  # type: ignore

router = APIRouter(prefix="/api/eod", tags=["eod-bots"])

ML_DATA = Path(PATHS.get("ml_data", "ml_data"))
STOCK_CACHE = Path(PATHS.get("stock_cache", "data_cache"))

# Where nightly bots keep state + logs
BOT_STATE_DIR = STOCK_CACHE / "master" / "bot"
BOT_LOG_ROOT = ML_DATA / "bot_logs"

# Horizons used by your nightly bots
HORIZONS = ["1w", "2w", "4w"]

# Rolling cache used for pricing
ROLLING_PATH = STOCK_CACHE / "master" / "rolling.json.gz"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _load_gz_dict(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_prices_from_rolling() -> Dict[str, float]:
    """
    Load latest prices from rolling.json.gz.

    Supports:
        {"AAPL": {...}} or {"symbols": {...}}
    and picks price from: price / last / close / c
    """
    data = _load_gz_dict(ROLLING_PATH)
    if not isinstance(data, dict):
        return {}
    symbols = data.get("symbols", data)

    prices: Dict[str, float] = {}
    for sym, node in symbols.items():
        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        try:
            price_f = float(price)
        except Exception:
            continue
        if price_f > 0:
            prices[str(sym).upper()] = price_f
    return prices


def _load_bot_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    if path.suffix.endswith(".gz"):
        return _load_gz_dict(path)
    return _load_json(path)


def _list_log_days() -> List[str]:
    """
    Scan ml_data/bot_logs/<HORIZON>/bot_activity_YYYY-MM-DD.json
    and return a sorted list of unique days.
    """
    days = set()
    if not BOT_LOG_ROOT.exists():
        return []
    for horizon in HORIZONS:
        hdir = BOT_LOG_ROOT / horizon
        if not hdir.exists():
            continue
        for f in hdir.glob("bot_activity_*.json"):
            name = f.name
            if not name.startswith("bot_activity_"):
                continue
            day = name.replace("bot_activity_", "").replace(".json", "")
            days.add(day)
    return sorted(days)


def _latest_day() -> Optional[str]:
    days = _list_log_days()
    return days[-1] if days else None


# -------------------------------------------------------------------
# Status endpoint — current PnL / positions
# -------------------------------------------------------------------

@router.get("/status")
async def eod_status():
    """
    Current state snapshot for EOD (swing) bots.

    Uses:
      • bot state files: stock_cache/master/bot/rolling_<botname>.json.gz
      • latest prices from: stock_cache/master/rolling.json.gz
    """
    prices = _load_prices_from_rolling()
    price_status = "ok" if prices else "no-prices"

    BOT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_files = sorted(BOT_STATE_DIR.glob("rolling_*.json.gz"))

    bots_out: Dict[str, Any] = {}

    for sf in state_files:
        state = _load_gz_dict(sf)
        if not isinstance(state, dict):
            continue

        bot_key = sf.stem.replace("rolling_", "")

        cash = float(state.get("cash", 0.0))
        positions = state.get("positions", {}) or {}

        pos_list = []
        equity = cash

        for sym, pos in positions.items():
            sym_u = str(sym).upper()
            px = prices.get(sym_u)

            entry = float(pos.get("entry", 0.0) or 0.0)
            qty = float(pos.get("qty", 0.0) or 0.0)
            stop = float(pos.get("stop", 0.0) or 0.0)
            target = float(pos.get("target", 0.0) or 0.0)

            unreal = None
            if isinstance(px, (int, float)):
                unreal = (px - entry) * qty
                equity += px * qty

            pos_list.append(
                {
                    "symbol": sym_u,
                    "qty": qty,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "last_price": px,
                    "unrealized_pnl": unreal,
                }
            )

        bots_out[bot_key] = {
            "cash": cash,
            "equity": equity,
            "num_positions": len(pos_list),
            "positions": pos_list,
        }

    return {
        "price_status": price_status,
        "bots": bots_out,
    }


# -------------------------------------------------------------------
# Logs — days / last-day / per-day / per-bot
# -------------------------------------------------------------------

@router.get("/logs/days")
async def eod_log_days():
    """
    List of trading days that have EOD bot logs (for any horizon).
    """
    days = _list_log_days()
    return {"count": len(days), "days": days}


@router.get("/logs/last-day")
async def eod_logs_last_day():
    """
    Logs for the most recent trading day, all horizons and bots.
    """
    day = _latest_day()
    if not day:
        raise HTTPException(status_code=404, detail="No EOD bot logs found.")
    return await eod_logs_for_day(day)


@router.get("/logs/{day}")
async def eod_logs_for_day(day: str):
    """
    Logs for a specific day, separated by horizon and bot.

    Output:
      {
        "date": "YYYY-MM-DD",
        "horizons": {
            "1w": {
                "bots": {
                    "momentum": [...],
                    "mean_revert": [...],
                    ...
                }
            },
            "2w": { ... },
            "4w": { ... }
        }
      }
    """
    if not BOT_LOG_ROOT.exists():
        raise HTTPException(status_code=404, detail="No EOD bot logs root folder found.")

    horizons_out: Dict[str, Any] = {}
    found_any = False

    for horizon in HORIZONS:
        hdir = BOT_LOG_ROOT / horizon
        if not hdir.exists():
            continue

        f = hdir / f"bot_activity_{day}.json"
        if not f.exists():
            continue

        try:
            # Files are gzipped JSON saved via _save_gz
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                js = json.load(fh)
        except Exception:
            js = None

        if not isinstance(js, dict):
            continue

        horizons_out[horizon] = {"bots": js}
        found_any = True

    if not found_any:
        raise HTTPException(
            status_code=404,
            detail=f"No EOD bot logs found for day '{day}'.",
        )

    return {
        "date": day,
        "horizons": horizons_out,
    }


@router.get("/logs/{day}/{horizon}/{bot_name}")
async def eod_logs_for_bot(day: str, horizon: str, bot_name: str):
    """
    Logs for a single bot on a specific day & horizon.

    Example:
      /api/eod/logs/2025-01-11/1w/momentum
    """
    horizon = horizon.lower()
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{horizon}'. Expected one of {HORIZONS}.",
        )

    f = BOT_LOG_ROOT / horizon / f"bot_activity_{day}.json"
    if not f.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EOD bot log file for day '{day}' and horizon '{horizon}'.",
        )

    try:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            js = json.load(fh)
    except Exception:
        js = None

    if not isinstance(js, dict):
        raise HTTPException(status_code=500, detail="Corrupt EOD log file.")

    actions = js.get(bot_name)
    if actions is None:
        raise HTTPException(
            status_code=404,
            detail=f"No log entries for bot '{bot_name}' on '{day}' (horizon '{horizon}').",
        )

    return {
        "date": day,
        "horizon": horizon,
        "bot": bot_name,
        "actions": actions,
    }
