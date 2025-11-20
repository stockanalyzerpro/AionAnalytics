"""
metrics_fetcher.py — v3.0
Aligned with new backend/core + nightly_job v4.0
--------------------------------------------------

Goal:
    Fetch / refresh StockAnalysis "metrics" for all Rolling symbols.

StockAnalysis provides 200+ indicators; we only fetch a curated set that:
    • is stable across all symbols
    • is used by your ML models
    • overlaps with your fundamentals / backfill bundle

Inputs:
    /s/d/<metric>
        Example: https://stockanalysis.com/api/screener/s/d/rsi

Outputs:
    rolling[sym]["metrics"] = {
        "<metric>": value,
        ...
    }

This module:
    • Uses batch fetching for efficiency
    • Normalizes all keys
    • Merges cleanly into Rolling
    • Never erases existing rolling data
"""

from __future__ import annotations

import json
from typing import Dict, Any, List
from pathlib import Path

import requests

from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    safe_float,
    log,
)
from backend.core.config import PATHS


# ==============================================================================
# CONFIG
# ==============================================================================

SA_BASE = "https://stockanalysis.com/api/screener"

# Best-practice metrics for technical + valuation layers
METRIC_LIST = [
    "rsi",             # → rsi_14
    "ma20", "ma50", "ma200",
    "beta",
    "pbRatio", "psRatio", "pegRatio",
    "ch1w", "ch1m", "ch3m", "ch6m", "ch1y", "chYTD",
    "volatility",      # if missing, will be skipped silently
]

# Normalize camelCase → snake_case
NORMALIZE = {
    "pbRatio": "pb_ratio",
    "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio",
    "rsi": "rsi_14",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def _sa_get_metric_table(metric: str) -> Dict[str, Any]:
    """Fetch /s/d/<metric> and return {SYM: value} map."""
    url = f"{SA_BASE}/s/d/{metric}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return {}
        js = r.json()
        rows = (js or {}).get("data", {}).get("data", [])

        out = {}
        for row in rows:
            # Rows can be [SYM, value] OR {symbol: X, metric: Y}
            if isinstance(row, list) and len(row) >= 2:
                sym = str(row[0]).upper()
                val = row[1]
            else:
                sym = (row.get("symbol") or row.get("s") or "").upper()
                val = row.get(metric)

            if sym:
                out[sym] = val
        return out

    except Exception as e:
        log(f"⚠️ Metric fetch failed for '{metric}': {e}")
        return {}


def _normalize_key(k: str) -> str:
    return NORMALIZE.get(k, k)


# ==============================================================================
# MAIN METRICS REFRESH
# ==============================================================================

def build_latest_metrics() -> Dict[str, Any]:
    """
    Fetch all metric tables and merge into rolling[sym]["metrics"].

    Called from:
        nightly_job.py — BEFORE model training
    """
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ No rolling.json.gz — skipping metrics fetch.")
        return {"status": "no_rolling"}

    log("📊 Fetching latest StockAnalysis metrics…")

    # Step 1 — fetch each table
    metric_tables: Dict[str, Dict[str, Any]] = {}
    for metric in METRIC_LIST:
        tbl = _sa_get_metric_table(metric)
        metric_tables[metric] = tbl

    updated = 0
    total = len(rolling)

    # Step 2 — merge metrics into Rolling
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        sym_u = sym.upper()

        metrics = node.get("metrics", {})
        changed = False

        for metric in METRIC_LIST:
            tbl = metric_tables.get(metric, {})
            if sym_u not in tbl:
                continue

            raw_key = metric
            new_key = _normalize_key(raw_key)
            metrics[new_key] = tbl[sym_u]
            changed = True

        if changed:
            node["metrics"] = metrics
            rolling[sym_u] = node
            updated += 1

    save_rolling(rolling)

    log(f"✅ Metrics updated for {updated}/{total} symbols.")
    return {"status": "ok", "updated": updated, "total": total}
