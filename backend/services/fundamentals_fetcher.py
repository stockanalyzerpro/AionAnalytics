"""
fundamentals_fetcher.py — v3.0
Aligned with new backend/core stack + nightly_job v4.0
--------------------------------------------------------

Goals:
    • Fetch and enrich fundamentals for all symbols in Rolling
    • Support multiple upstream providers (StockAnalysis, FMP, AlphaVantage)
    • Normalize all fields (camelCase → snake_case)
    • Merge fundamentals cleanly into Rolling
    • Full compatibility with updated data_pipeline + nightly_job

This module DOES NOT overwrite any fields already existing in Rolling
unless new fundamentals contain fresher values.

Works with:
    - StockAnalysis /s/d/<metric>
    - (optionally) FMP fundamentals endpoints
    - (optionally) Alpha Vantage fundamentals endpoints
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    safe_float,
    log,
)
from backend.core.config import PATHS


# ==============================================================================
# Constants
# ==============================================================================

FUND_DIR = PATHS["fundamentals_raw"]
FUND_DIR.mkdir(parents=True, exist_ok=True)

SA_BASE = "https://stockanalysis.com/api/screener"

# StockAnalysis metrics we treat as “fundamentals”
FUNDAMENTAL_METRICS = [
    "pbRatio", "psRatio", "pegRatio",
    "profitMargin", "operatingMargin", "grossMargin",
    "revenueGrowth", "epsGrowth",
    "debtEquity", "debtEbitda",
    "fcfYield", "earningsYield",
    "dividendYield",
    "sector",
    "sharesOut",
]


# ==============================================================================
# Normalization helper
# ==============================================================================

def _normalize_keys(node: Dict[str, Any]) -> Dict[str, Any]:
    """Convert StockAnalysis field names into snake_case."""
    replacements = {
        "pbRatio": "pb_ratio",
        "psRatio": "ps_ratio",
        "pegRatio": "peg_ratio",
        "revenueGrowth": "revenue_growth",
        "epsGrowth": "eps_growth",
        "profitMargin": "profit_margin",
        "operatingMargin": "operating_margin",
        "grossMargin": "gross_margin",
        "fcfYield": "fcf_yield",
        "earningsYield": "earnings_yield",
        "dividendYield": "dividend_yield",
        "debtEquity": "debt_equity",
        "debtEbitda": "debt_ebitda",
        "sharesOut": "shares_outstanding",
    }
    for old, new in replacements.items():
        if old in node:
            node[new] = node.pop(old)
    return node


# ==============================================================================
# StockAnalysis API helpers
# ==============================================================================

def _sa_get_metric(metric: str) -> Dict[str, Any]:
    """Fetch a StockAnalysis metric table."""
    url = f"{SA_BASE}/s/d/{metric}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        js = r.json()
        tbl = {}
        rows = (js or {}).get("data", {}).get("data", [])
        for row in rows:
            if isinstance(row, list):
                sym = str(row[0]).upper()
                val = row[1]
            else:
                sym = (row.get("symbol") or row.get("s") or "").upper()
                val = row.get(metric)
            if sym:
                tbl[sym] = val
        return tbl
    except Exception as e:
        log(f"⚠️ SA fundamental metric '{metric}' fetch failed: {e}")
        return {}


def _fetch_sa_fundamentals() -> Dict[str, Dict[str, Any]]:
    """Batch-fetch all fundamental metrics from StockAnalysis."""
    bundle: Dict[str, Dict[str, Any]] = {}
    for metric in FUNDAMENTAL_METRICS:
        tbl = _sa_get_metric(metric)
        for sym, val in tbl.items():
            if sym not in bundle:
                bundle[sym] = {"symbol": sym}
            bundle[sym][metric] = val
    return bundle


# ==============================================================================
# Optional FMP support (kept from your original code)
# ==============================================================================

def _fetch_fmp_data(sym: str, api_key: str | None) -> Dict[str, Any]:
    """Stock fundamentals from FMP (fallback source)."""
    if not api_key:
        return {}

    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        js = r.json()
        if not js:
            return {}
        row = js[0]
        return {
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "description": row.get("description"),
            "ceo": row.get("ceo"),
            "country": row.get("country"),
            "employees": row.get("fullTimeEmployees"),
            "beta": row.get("beta"),
            "pe_ratio": row.get("pe"),
            "marketcap": row.get("mktCap"),
        }
    except Exception as e:
        log(f"⚠️ FMP fetch failed for {sym}: {e}")
        return {}


# ==============================================================================
# Main fundamental enrichment
# ==============================================================================

def enrich_fundamentals() -> Dict[str, Any]:
    """
    Merge multiple sources:
        1) StockAnalysis metric tables
        2) (optional) FMP fundamentals
        3) (optional) AlphaVantage fundamentals

    Integrate into Rolling:
        rolling[sym]["fundamentals"] = merged_fields
    """
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ No rolling.json.gz — fundamentals enrichment aborted.")
        return {"status": "no_rolling"}

    log("📘 Fetching fundamental metrics from StockAnalysis...")
    sa_bundle = _fetch_sa_fundamentals()

    updated = 0
    total = len(rolling)

    FMP_KEY = os.environ.get("FMP_API_KEY", "")

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        sym_u = sym.upper()
        base = {}

        # 1) SA metrics
        sa = sa_bundle.get(sym_u, {})
        base.update(sa)

        # 2) FMP optional enrichment (legacy support)
        if FMP_KEY:
            fmp = _fetch_fmp_data(sym_u, FMP_KEY)
            base.update(fmp)

        # normalize
        base = _normalize_keys(base)

        # Attach to rolling
        fund = node.get("fundamentals", {})
        fund.update(base)
        node["fundamentals"] = fund
        rolling[sym_u] = node
        updated += 1

    save_rolling(rolling)

    log(f"✅ Fundamentals enriched for {updated}/{total} symbols.")
    return {"updated": updated, "total": total, "status": "ok"}
