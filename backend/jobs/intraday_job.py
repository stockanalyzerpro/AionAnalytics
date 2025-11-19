# backend/jobs/intraday_job.py
"""
Intraday Job — runs every N minutes.

Steps:
    1. Fetch intraday bars (Alpaca IDX + YF fallback)
    2. Update dt_backend rolling
    3. Run intraday runner (context → features → scoring → policy → execution)
    4. Write logs + snapshot
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.services.intraday_fetcher import fetch_and_update
from backend.intraday_runner import run_intraday_cycle

try:
    from dt_backend.dt_logger import dt_log as log
except:
    def log(msg: str):
        print(msg, flush=True)


def _symbols_from_rolling() -> list:
    from dt_backend.core.data_pipeline_dt import _read_rolling
    r = _read_rolling() or {}
    return [s for s in r.keys() if not s.startswith("_")]


def _log_dir() -> Path:
    p = Path("logs/intraday")
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_intraday_job() -> dict:
    ts = datetime.now(timezone.utc).isoformat()

    symbols = _symbols_from_rolling()
    log(f"[intraday_job] starting job at {ts}, {len(symbols)} symbols")

    # 1. Fetch bars
    fetch_res = fetch_and_update(symbols)

    # 2. Run dt_backend intraday cycle
    cycle_res = run_intraday_cycle()

    # 3. Write snapshot log
    logdir = _log_dir()
    snap = {
        "ts": ts,
        "fetch": fetch_res,
        "cycle": cycle_res,
    }
    (logdir / "last_run.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

    log(f"[intraday_job] complete → fetched={fetch_res['fetched']}, updated={cycle_res['updated_symbols']}")

    return snap


def main():
    print(json.dumps(run_intraday_job(), indent=2))


if __name__ == "__main__":
    main()
