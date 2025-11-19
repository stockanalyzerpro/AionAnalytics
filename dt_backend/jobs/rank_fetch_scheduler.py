# dt_backend/jobs/rank_fetch_scheduler.py — v2.0
"""
Intraday rank file maintainer for AION dt_backend.

This job keeps the compact `prediction_rank_fetch.json.gz` file fresh,
which is consumed by external schedulers / bots.

Behavior
--------
• Ensures a rank file exists (seeding from universe if needed)
• Optionally rebuilds ranks in a loop on a fixed interval
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from dt_backend.core import DT_PATHS, log, load_universe
from dt_backend.ml import build_intraday_signals

try:
    # Optional visual progress helper if available
    from utils.progress_bar import progress_bar  # type: ignore
except Exception:  # pragma: no cover
    def progress_bar(iterable, **kwargs):
        for x in iterable:
            yield x


def _ranks_path() -> Path:
    return DT_PATHS["signals_intraday_ranks_dir"] / "prediction_rank_fetch.json.gz"


def _seed_rank_file_from_universe() -> Dict[str, Any]:
    """
    Create a minimal rank file based solely on the universe.

    This is a fallback when no predictions exist yet; every symbol gets
    a neutral score and HOLD action, so downstream jobs can still work.
    """
    universe = load_universe()
    if not universe:
        log("[rank_fetch_scheduler] ⚠️ cannot seed ranks — universe empty.")
        return {"status": "no_universe"}

    universe = sorted(set(universe))
    payload: List[Dict[str, Any]] = [
        {"symbol": sym, "score": 0.0, "action": "HOLD"} for sym in universe
    ]

    path = _ranks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        log(
            f"[rank_fetch_scheduler] ✅ seeded ranks for {len(universe)} symbols "
            f"→ {path}"
        )
        return {"status": "seeded", "symbols": len(universe), "path": str(path)}
    except Exception as e:
        log(f"[rank_fetch_scheduler] ⚠️ failed to seed rank file: {e}")
        return {"status": "error", "error": str(e)}


def ensure_rank_file() -> Dict[str, Any]:
    """
    Ensure that the compact rank file exists.

    If absent, we first try to build from actual policy/predictions
    via `build_intraday_signals`. If that still results in no file,
    we fall back to seeding from the trading universe.
    """
    path = _ranks_path()
    if path.exists():
        return {"status": "exists", "path": str(path)}

    # Try to create via real signals
    sig_summary = build_intraday_signals()
    if path.exists():
        log("[rank_fetch_scheduler] ✅ created ranks via build_intraday_signals.")
        return {"status": "created_from_signals", "signals": sig_summary, "path": str(path)}

    # Fallback: seed from universe
    return _seed_rank_file_from_universe()


def run_rank_scheduler(interval_sec: int = 300, once: bool = False) -> Dict[str, Any]:
    """
    Maintain the rank file on a schedule.

    Parameters
    ----------
    interval_sec:
        Sleep duration between refreshes (ignored if once=True).
    once:
        If True, perform a single refresh and return immediately.

    Returns
    -------
    For once=True, returns the last summary. For continuous mode,
    this function never returns under normal operation.
    """
    log(
        f"[rank_fetch_scheduler] ⚡ Rank Fetch Scheduler started "
        f"(interval={interval_sec}s, once={once})."
    )

    def _cycle() -> Dict[str, Any]:
        ensure_summary = ensure_rank_file()
        signals_summary = build_intraday_signals()
        log("[rank_fetch_scheduler] 🔄 ranks refreshed from latest policy.")
        return {"ensure": ensure_summary, "signals": signals_summary}

    if once:
        return _cycle()

    # Continuous mode
    while True:
        summary = _cycle()
        # crude wait with optional progress bar visualization
        for _ in progress_bar(range(interval_sec), desc="Rank scheduler sleep"):
            time.sleep(1)
        # loop continues indefinitely
