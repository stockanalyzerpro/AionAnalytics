
"""
dt_backend/core/data_pipeline_dt.py

Lightweight I/O helpers for the intraday engine:

  • `log`                    → tiny stdout logger for dt_backend
  • `_read_rolling`          → load intraday rolling cache (JSON.GZ)
  • `save_rolling`           → atomic write of rolling cache
  • `load_universe`          → list of tickers to trade intraday
  • `ensure_symbol_node`     → ensure a minimal per-symbol node structure

This module is intentionally dependency-light and must *never* raise
for normal operations. Callers rely on safe fallbacks (e.g. empty dict).
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config_dt import DT_PATHS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Simple UTC timestamped logger for dt_backend."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [dt_backend] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Rolling cache helpers
# ---------------------------------------------------------------------------

def _rolling_path() -> Path:
    return DT_PATHS["rolling_intraday_file"]


def _read_rolling() -> Dict[str, Any]:
    """
    Read intraday rolling cache from JSON.GZ.

    Schema is intentionally flexible but we assume a top-level dict:
        { "AAPL": {...}, "MSFT": {...}, ... }

    On any error, returns an empty dict and logs a warning.
    """
    path = _rolling_path()
    if not path.exists():
        return {}

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log(f"⚠️ rolling cache at {path} is not a dict, resetting.")
            return {}
        return data
    except Exception as e:
        log(f"⚠️ failed to read rolling cache {path}: {e}")
        return {}


def save_rolling(rolling: Dict[str, Any]) -> None:
    """
    Atomically write intraday rolling cache as JSON.GZ.

    Uses a .tmp swap file to avoid partial writes.
    """
    path = _rolling_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(rolling or {}, f, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        log(f"⚠️ failed to save rolling cache {path}: {e}")


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def _norm_sym(sym: str) -> str:
    return (sym or "").strip().upper()


def load_universe() -> List[str]:
    """
    Load the intraday trading universe as a flat list of ticker symbols.

    Supports two schemas:
      1) {"symbols": ["AAPL", "MSFT", ...]}
      2) ["AAPL", "MSFT", ...]

    Returns an empty list on error.
    """
    path = DT_PATHS["universe_file"]
    if not path.exists():
        log(f"⚠️ universe file missing at {path} — using empty universe.")
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"⚠️ failed to parse universe file {path}: {e}")
        return []

    if isinstance(raw, dict) and "symbols" in raw:
        items: Iterable[str] = raw.get("symbols", [])
    elif isinstance(raw, list):
        items = raw
    else:
        log(f"⚠️ unexpected universe schema in {path}, expected list or dict['symbols'].")
        return []

    out: List[str] = []
    seen = set()
    for item in items:
        sym = _norm_sym(str(item))
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


# ---------------------------------------------------------------------------
# Rolling node helpers
# ---------------------------------------------------------------------------

def ensure_symbol_node(rolling: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Ensure that `rolling[symbol]` exists and has the standard sub-structure.

    Standard sections:
      • bars_intraday     → list of latest intraday bars
      • features_dt       → feature snapshot for intraday model
      • predictions_dt    → intraday model outputs
      • context_dt        → human-style intraday context
      • policy_dt         → intraday trade policy / intent

    Returns the node dict (mutated in-place inside `rolling`).
    """
    sym = _norm_sym(symbol)
    node = rolling.get(sym)
    if not isinstance(node, dict):
        node = {}

    node.setdefault("bars_intraday", [])
    node.setdefault("features_dt", {})
    node.setdefault("predictions_dt", {})
    node.setdefault("context_dt", {})
    node.setdefault("policy_dt", {})

    rolling[sym] = node
    return node
