# dt_backend/jobs/backfill_intraday_full.py — v2.0
"""
Full intraday backfill / bootstrap job for AION dt_backend.

Goals
-----
• Ensure rolling contains a node for every symbol in the intraday universe
• Optionally hydrate `bars_intraday` from local disk snapshots
• Build intraday context + features
• (Optionally) build dataset and retrain models

This job is **idempotent** and safe to run multiple times; it will
overwrite in-memory rolling but does not mutate historical raw bars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dt_backend.core import (
    DT_PATHS,
    log,
    load_universe,
    ensure_symbol_node,
    save_rolling,
)
from dt_backend.core import build_intraday_context
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml import build_intraday_dataset, train_intraday_models


def _bars_dir() -> Path:
    return DT_PATHS["bars_intraday_dir"]


def _load_bars_for_symbol(sym: str) -> List[Dict[str, Any]]:
    """
    Best-effort loader for local intraday bars for a symbol.

    We expect (if present) a JSON file at:
        dt_backend/bars/intraday/<SYMBOL>.json

    Schema is flexible but typically:
        [ { "ts": "...", "c": 123.4, ... }, ... ]
    """
    path = _bars_dir() / f"{sym}.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [b for b in data if isinstance(b, dict)]
        return []
    except Exception as e:
        log(f"[backfill_intraday_full] ⚠️ failed to load bars for {sym} from {path}: {e}")
        return []


def backfill_intraday_full(
    max_symbols: int | None = None,
    retrain_after: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate a full intraday bootstrap:

      1) Load universe
      2) For each symbol:
           - ensure rolling node
           - hydrate bars_intraday from local disk if present
      3) Build intraday context + features
      4) Build intraday dataset
      5) Optionally retrain models

    Returns a summary dict.
    """
    universe = load_universe()
    if not universe:
        log("[backfill_intraday_full] ⚠️ universe is empty, nothing to backfill.")
        return {"status": "no_universe", "symbols": 0}

    universe = sorted(set(universe))
    if max_symbols is not None:
        universe = universe[: max(0, int(max_symbols))]

    rolling: Dict[str, Any] = {}
    hydrated = 0

    for sym in universe:
        node = ensure_symbol_node(rolling, sym)
        bars = _load_bars_for_symbol(sym)
        if bars:
            node["bars_intraday"] = bars
            hydrated += 1
        rolling[sym] = node

    save_rolling(rolling)
    log(
        f"[backfill_intraday_full] ✅ seeded rolling for {len(universe)} symbols "
        f"(hydrated_bars={hydrated})."
    )

    # Build context + features
    build_intraday_context()
    feat_summary = build_intraday_features(max_symbols=max_symbols)

    # Build dataset
    ds_summary = build_intraday_dataset(max_symbols=max_symbols)

    train_summary: Dict[str, Any] | None = None
    if retrain_after:
        train_summary = train_intraday_models()

    out: Dict[str, Any] = {
        "status": "ok",
        "symbols": len(universe),
        "hydrated_bars": hydrated,
        "features": feat_summary,
        "dataset": ds_summary,
    }
    if train_summary is not None:
        out["training"] = train_summary

    return out
