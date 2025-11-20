# backend/routers/intraday_logs_router.py
"""
Intraday Day-Trading Bot Logs API — AION Analytics

Exposes:
    • /api/intraday/logs/last-day
    • /api/intraday/logs/days
    • /api/intraday/logs/{day}
    • /api/intraday/logs/{day}/{bot_name}
    • /api/intraday/pnl/last-day

Reads from:
    ml_data_dt/sim_logs/
    ml_data_dt/sim_summary.json
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

from backend.core.config import PATHS

router = APIRouter(prefix="/api/intraday", tags=["intraday-bot"])

ML_DATA_DT = Path(PATHS.get("ml_data_dt", "ml_data_dt"))
SIM_LOG_DIR = ML_DATA_DT / "sim_logs"
SIM_SUMMARY_FILE = ML_DATA_DT / "sim_summary.json"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _get_available_days() -> List[str]:
    """Return sorted list of unique YYYY-MM-DD available in sim_logs."""
    if not SIM_LOG_DIR.exists():
        return []
    days = set()
    for f in SIM_LOG_DIR.glob("*.json"):
        name = f.name
        # expected: YYYY-MM-DD_botname.json
        parts = name.split("_")
        if len(parts) >= 2:
            day = parts[0]
            days.add(day)
    return sorted(days)

def _get_latest_day() -> Optional[str]:
    days = _get_available_days()
    return days[-1] if days else None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.get("/logs/days")
async def list_log_days():
    """
    Returns list of trading days that have logs.
    Example:
      ["2025-01-09", "2025-01-10", "2025-01-11"]
    """
    days = _get_available_days()
    return {"count": len(days), "days": days}


@router.get("/logs/last-day")
async def get_last_day_logs():
    """
    Returns logs for the most recent trading day — all bots.
    """
    day = _get_latest_day()
    if not day:
        raise HTTPException(404, "No trading-day logs found.")
    return await get_logs_for_day(day)


@router.get("/logs/{day}")
async def get_logs_for_day(day: str):
    """
    Returns logs for all bots for a given day.
    Output:
        {
          "date": "2025-01-11",
          "bots": {
              "momentum_bot": {...},
              "hybrid_bot": {...},
              ...
          }
        }
    """
    files = list(SIM_LOG_DIR.glob(f"{day}_*.json"))
    if not files:
        raise HTTPException(404, f"No logs found for day '{day}'.")

    out: Dict[str, Any] = {"date": day, "bots": {}}
    for f in files:
        bot_name = f.stem.replace(f"{day}_", "")
        js = _read_json(f)
        if js is not None:
            out["bots"][bot_name] = js

    return out


@router.get("/logs/{day}/{bot_name}")
async def get_logs_for_bot(day: str, bot_name: str):
    """
    Returns logs for a single bot for a given day.
    """
    file = SIM_LOG_DIR / f"{day}_{bot_name}.json"
    if not file.exists():
        raise HTTPException(404, f"No logs for bot '{bot_name}' on day '{day}'.")
    js = _read_json(file)
    return {"date": day, "bot": bot_name, "log": js}


@router.get("/pnl/last-day")
async def get_last_day_pnl_summary():
    """
    Returns PnL summary for the latest trading day from sim_summary.json.

    Output format:
        {
          "date": "2025-01-11",
          "bots": {
              "momentum_bot": {"equity": 104.21, "positions": 2, "trades": 8},
              ...
          }
        }
    """
    summary = _read_json(SIM_SUMMARY_FILE)
    if not summary:
        raise HTTPException(404, "No simulation summary found.")

    days = summary.get("days", [])
    if not days:
        raise HTTPException(404, "Simulation summary contains no day entries.")

    last = days[-1]
    return last
