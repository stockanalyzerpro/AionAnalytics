# backend/backend_service.py — v2.0
"""
Main FastAPI backend service for AION Analytics.

Responsibilities:
    • Expose API endpoints via routers:
        - /api/system   (system_status_router)
        - /api/insights (insights_router)
        - /api/live     (live_prices_router)
        - /api/models   (model_router)
    • Start scheduler runner in a background thread.
    • Emit hourly backend heartbeat logs.
    • Optional cloud sync hooks (safe if missing).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.core.config import PATHS, TIMEZONE

# Scheduler runner
try:
    from backend.scheduler_runner import main as scheduler_main
except ImportError:
    try:
        from backend.services.scheduler_runner import main as scheduler_main  # type: ignore
    except ImportError:
        scheduler_main = None  # type: ignore

# Routers
from backend.routers.system_status_router import router as system_router
from backend.routers.insights_router import router as insights_router
from backend.routers.live_prices_router import router as live_router
from backend.routers.model_router import router as model_router

# Optional cloud sync
try:
    from backend import cloud_sync  # type: ignore
except Exception:
    cloud_sync = None  # type: ignore


app = FastAPI(
    title="AION Analytics Backend",
    description="Backend API + scheduler for AION Analytics.",
    version="2.0.0",
)

# CORS – you can tighten this later if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Routers already define their own /api/* prefixes
app.include_router(system_router)
app.include_router(insights_router)
app.include_router(live_router)
app.include_router(model_router)


# ----------------------------------------------------------
# Basic endpoints
# ----------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "AION Analytics Backend",
        "version": "2.0.0",
        "time": datetime.now(TIMEZONE).isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now(TIMEZONE).isoformat(),
    }


# ----------------------------------------------------------
# Background threads
# ----------------------------------------------------------

def _backend_heartbeat():
    """Hourly heartbeat log."""
    while True:
        now = datetime.now(TIMEZONE)
        print(f"[Backend] 💓 Heartbeat — {now.isoformat()}", flush=True)
        time.sleep(3600)


def _scheduler_thread():
    """Run scheduler runner, if present."""
    if scheduler_main is None:
        print("[Scheduler]⚠️ scheduler_runner not found; skipping.", flush=True)
        return
    try:
        print("[Scheduler] 🧭 Starting scheduler runner…", flush=True)
        scheduler_main()
    except Exception as e:
        print(f"[Scheduler] ❌ Scheduler crashed: {e}", flush=True)


def _print_root_path():
    try:
        root = PATHS.get("root", None)
        print(f"[Backend] 📦 Root path: {root}", flush=True)
    except Exception:
        pass


# ----------------------------------------------------------
# Startup
# ----------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    print("────────────────────────────────────────", flush=True)
    print("🚀 AION Analytics Backend — Starting…", flush=True)
    print("────────────────────────────────────────", flush=True)

    _print_root_path()

    # Optional cloud sync
    if cloud_sync is not None:
        try:
            print("[CloudSync] ☁️ Starting background sync tasks…", flush=True)
            cloud_sync.start_background_tasks()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[CloudSync] ⚠️ Cloud sync init failed: {e}", flush=True)

    # Heartbeat + scheduler
    threading.Thread(target=_backend_heartbeat, daemon=True).start()
    threading.Thread(target=_scheduler_thread, daemon=True).start()

    print("[Backend] ✅ Startup complete — ready for requests.", flush=True)


# ----------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("[Backend] ✅ Running at http://127.0.0.1:8000", flush=True)
    uvicorn.run("backend.backend_service:app", host="127.0.0.1", port=8000, reload=True)
