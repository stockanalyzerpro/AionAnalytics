# backend/intraday_service.py
"""
Intraday read-only service for AION.

This module sits on top of dt_backend and exposes clean Python helpers to:
    • load the intraday rolling cache
    • inspect per-symbol state (context, features, predictions, policy, execution)
    • compute top BUY/SELL signals by confidence

The FastAPI router (intraday_router.py) calls into these helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    # Prefer your dt logger if available
    from dt_backend.dt_logger import dt_log as log
except Exception:  # fallback
    def log(msg: str) -> None:
        print(msg, flush=True)

from dt_backend.core.data_pipeline_dt import _read_rolling
from dt_backend.core.config_dt import DT_PATHS  # type: ignore


def _load_rolling() -> Dict[str, Any]:
    """Best-effort load of intraday rolling cache."""
    rolling = _read_rolling() or {}
    if not isinstance(rolling, dict):
        log("[intraday_service] ⚠️ rolling is not a dict, resetting to empty.")
        return {}
    return rolling


def _normalize_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _extract_symbol_view(sym: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the raw node into something API-friendly."""
    meta = node.get("meta") or {}
    ctx = node.get("context_dt") or {}
    feats = node.get("features_dt") or {}
    preds = node.get("predictions_dt") or node.get("predictions") or {}
    policy = node.get("policy_dt") or {}
    exec_dt = node.get("execution_dt") or {}

    last_price = None
    bars = node.get("bars_intraday") or []
    if isinstance(bars, list) and bars:
        b = bars[-1]
        last_price = b.get("price") or b.get("c") or b.get("close")

    return {
        "symbol": sym,
        "meta": {
            "name": meta.get("name") or node.get("name"),
            "sector": meta.get("sector") or node.get("sector"),
            "industry": meta.get("industry") or node.get("industry"),
            "marketcap": meta.get("marketcap") or node.get("marketcap"),
        },
        "last_price": last_price,
        "context": ctx,
        "features": feats,
        "predictions": preds,
        "policy": policy,
        "execution": exec_dt,
    }


def get_symbol_view(symbol: str) -> Optional[Dict[str, Any]]:
    """Return a rich view of a single symbol, or None if not found."""
    rolling = _load_rolling()
    sym = _normalize_symbol(symbol)
    node = rolling.get(sym)
    if not isinstance(node, dict):
        return None
    return _extract_symbol_view(sym, node)


def _iter_signal_nodes(rolling: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for sym, node in rolling.items():
        if not isinstance(sym, str):
            continue
        if sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        out.append((sym, node))
    return out


def get_top_signals(
    side: str = "BUY",
    limit: int = 50,
    min_conf: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Return top BUY/SELL signals by policy confidence.

    side: "BUY" or "SELL"
    """
    side = side.upper()
    rolling = _load_rolling()
    rows: List[Tuple[float, Dict[str, Any]]] = []

    for sym, node in _iter_signal_nodes(rolling):
        policy = node.get("policy_dt") or {}
        intent = str(policy.get("intent") or "").upper()
        conf = float(policy.get("confidence") or 0.0)
        if intent != side:
            continue
        if conf < min_conf:
            continue
        view = _extract_symbol_view(sym, node)
        rows.append((conf, view))

    # sort highest confidence first
    rows.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in rows[:limit]]


def get_intraday_snapshot(limit: int = 50) -> Dict[str, Any]:
    """
    Quick dashboard snapshot:
        • timestamp
        • top BUY signals
        • top SELL signals
    """
    now = datetime.now(timezone.utc).isoformat()
    buys = get_top_signals("BUY", limit=limit)
    sells = get_top_signals("SELL", limit=limit)

    return {
        "ts": now,
        "buys": buys,
        "sells": sells,
    }
