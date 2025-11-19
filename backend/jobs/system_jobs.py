# backend/jobs/system_jobs.py
"""
System Job Registry

Allows backend to expose:
    /api/system/jobs
and return status of:
    - intraday job
    - nightly job
    - replay job
    - rank fetch scheduler
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

SYSTEM_JOBS = {
    "intraday": {
        "name": "Intraday Job (bars + inference)",
        "path": "backend.jobs.intraday_job.run_intraday_job",
        "log_path": "logs/intraday/last_run.json",
    },
    "nightly": {
        "name": "Nightly ML Pipeline",
        "path": "backend.nightly_job.run",
        "log_path": "logs/nightly/last_run.json",
    },
    "replay": {
        "name": "Historical Replay Engine",
        "path": "dt_backend.historical_replay.historical_replay_manager.run_replay_range",
        "log_path": "ml_data_dt/intraday/replay/replay_log.json",
    },
    "rank_scheduler": {
        "name": "Rank Fetch Scheduler",
        "path": "dt_backend.rank_fetch_scheduler.run",
        "log_path": "logs/scheduler/rank_last_run.json",
    },
}


def read_job_log(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "never_run"}

    try:
        return json.loads(p.read_text())
    except:
        return {"status": "log_corrupt"}


def get_system_jobs() -> Dict[str, Any]:
    out = {}
    for key, info in SYSTEM_JOBS.items():
        out[key] = {
            "name": info["name"],
            "job_path": info["path"],
            "log": read_job_log(info["log_path"]),
        }
    return out
