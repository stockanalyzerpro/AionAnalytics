# backend/routers/intraday_router.py
"""
Intraday API router.

Endpoints:
    GET /api/intraday/snapshot
        → top BUY/SELL signals with policy & execution info

    GET /api/intraday/symbol/{symbol}
        → full view for one symbol

    GET /api/intraday/top/{side}
        → top BUY or SELL signals only
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.intraday_service import (
    get_intraday_snapshot,
    get_symbol_view,
    get_top_signals,
)

router = APIRouter(prefix="/api/intraday", tags=["intraday"])


@router.get("/snapshot")
def api_intraday_snapshot(limit: int = Query(50, ge=1, le=200)):
    """
    Quick dashboard snapshot:
        • timestamp
        • top BUY signals
        • top SELL signals
    """
    return get_intraday_snapshot(limit=limit)


@router.get("/symbol/{symbol}")
def api_intraday_symbol(symbol: str):
    """
    Return full intraday view for a single symbol:
        • meta
        • last price
        • context_dt
        • features_dt
        • predictions_dt/predictions
        • policy_dt
        • execution_dt
    """
    view = get_symbol_view(symbol)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in intraday rolling.")
    return view

@router.get("/top/{side}")
def api_intraday_top(
    side: str,
    limit: int = Query(50, ge=1, le=200),
    min_conf: float = Query(0.20, ge=0.0, le=1.0),

from backend.intraday_runner import run_intraday_cycle

@router.post("/refresh")
def api_intraday_refresh():
    """
    Run the full intraday inference cycle (context → features → scoring → policy → execution)
    and return updated results.
    """
    return run_intraday_cycle()

):
    """
    Return top BUY or SELL signals by policy confidence.
    """
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")

    rows = get_top_signals(side=side, limit=limit, min_conf=min_conf)
    return {
        "side": side,
        "limit": limit,
        "min_conf": min_conf,
        "count": len(rows),
        "results": rows,
    }
