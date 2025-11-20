"""
scheduler_runner.py
Runs automated backend + dt_backend tasks based on scheduler_config.py.

Design:
    • Reads ENABLE, TIMEZONE, SCHEDULE from backend.scheduler_config
    • Each job has:
         {
           "name": "nightly_full",
           "time": "17:30",             # HH:MM in TIMEZONE
           "module": "backend.jobs.nightly_job",
           "args": ["--foo", "bar"],
           "description": "Full nightly rebuild"
         }
    • Checks once every 30 seconds.
    • Triggers each job at the matching HH:MM (only once per minute).
    • Logs to logs/scheduler/scheduler_runner.log via PATHS.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pytz

from .scheduler_config import ENABLE, TIMEZONE, SCHEDULE
from .config import PATHS

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_DIR: Path = PATHS.get("scheduler_logs", PATHS.get("logs", Path("logs")) / "scheduler")  # type: ignore
LOG_FILE: Path = LOG_DIR / "scheduler_runner.log"


def _ensure_log_dir() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def log(msg: str) -> None:
    """Print to stdout and append to scheduler log file."""
    _ensure_log_dir()
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[scheduler_runner] {ts} {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Don't crash scheduler on logging errors
        pass


# ---------------------------------------------------------------------
# Job running
# ---------------------------------------------------------------------

def _build_cmd(job: Dict[str, Any]) -> list[str]:
    """
    Build a python -m command from a job entry.

    Expected keys:
        module:  "backend.jobs.nightly_job"
        args:    ["--mode", "full"]
    """
    module = job.get("module")
    if not module:
        raise ValueError(f"Job {job.get('name')} missing 'module' field.")
    args = job.get("args", []) or []
    cmd = [sys.executable, "-m", module, *args]
    return cmd


def run_job(job: Dict[str, Any]) -> None:
    """Fire-and-forget a job in a subprocess."""
    name = job.get("name", "UNKNOWN")
    desc = job.get("description", "")
    try:
        cmd = _build_cmd(job)
    except Exception as e:
        log(f"❌ Invalid job {name}: {e}")
        return

    log(f"▶️ Starting job '{name}' — {desc} → {cmd!r}")

    try:
        # Non-blocking so scheduler can keep ticking
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        log(f"❌ Failed to launch job '{name}': {e}")


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main(loop_forever: bool = True) -> None:
    """
    Main scheduler loop.

    If loop_forever=False, run a single HH:MM sweep (useful for testing).
    """
    if not ENABLE:
        log("⚠️ Scheduler is disabled via ENABLE=False in scheduler_config.py.")
        return

    tz = pytz.timezone(TIMEZONE)
    log(f"✅ Scheduler starting — timezone={TIMEZONE}, jobs={len(SCHEDULE)}")

    # Keep track of last-run minute per job so we don't run twice in same minute
    last_run_minute: Dict[str, str] = {job["name"]: "" for job in SCHEDULE}

    while True:
        now = datetime.now(tz)
        current_minute = now.strftime("%H:%M")

        for job in SCHEDULE:
            name = job["name"]
            job_time = job["time"]
            if job_time == current_minute and last_run_minute.get(name) != current_minute:
                run_job(job)
                last_run_minute[name] = current_minute

        if not loop_forever:
            break

        # Sleep in short intervals so we don't miss minute ticks
        time.sleep(30)


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Optional: allow a single-pass test with `--once`
    if "--once" in sys.argv:
        main(loop_forever=False)
    else:
        main(loop_forever=True)
