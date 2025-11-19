"""
scheduler_runner.py
Runs automated backend tasks based on scheduler_config.py.
"""

from __future__ import annotations
from datetime import datetime
import pytz
import schedule
from .scheduler_config import ENABLE, TIMEZONE, SCHEDULE
from pathlib import Path
import os, sys, time, subprocess
from .config import PATHS

ROOT = Path(__file__).resolve().parents[1]
JOB_LOCK = ROOT / "data" / "stock_cache" / "master" / "nightly_job.lock"  # local-only lock

# ---------------------------------------------------------------------
# Paths and logging
# ---------------------------------------------------------------------
LOG_DIR = PATHS["scheduler_logs"].parent  # logs/scheduler/
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg: str):
    """Simple timestamped logger with daily rotating log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    log_file = LOG_DIR / f"scheduler_{datetime.now():%Y%m%d}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# ---------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------
def run_job(entry):
    """Execute a backend job via subprocess."""
    script = entry["script"]
    args = entry.get("args", [])
    cmd = ["python", "-m", script.replace("/", ".").replace(".py", "")]
    cmd.extend(args)

    log(f"🚀 Running {entry['name']} → {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        log(f"⚠️ Failed to run {entry['name']}: {e}")
    log(f"✅ Finished {entry['name']}")

# ---------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------
def main():
    if not ENABLE:
        log("⚠️ Scheduler disabled in scheduler_config.py")
        return

    tz = pytz.timezone(TIMEZONE)
    log(f"🧭 Scheduler started — timezone: {TIMEZONE}")

    # Track last run to avoid duplicates
    last_run = {job["name"]: None for job in SCHEDULE}

    while True:
        now = datetime.now(tz)
        current_time = now.strftime("%H:%M")

        for job in SCHEDULE:
            if job["time"] == current_time:
                # Run only once per minute per job
                if last_run[job["name"]] != current_time:
                    log(f"⏰ Triggering {job['name']} at {current_time}")
                    run_job(job)
                    last_run[job["name"]] = current_time
        time.sleep(30)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
