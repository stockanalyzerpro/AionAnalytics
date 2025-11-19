"""
backend_service.py — v1.8.2
Main FastAPI backend service for AION Analytics.
Adds:
  • Emoji boot-sequence logging
  • Hourly backend heartbeat (Eastern Time)
  • Retains scheduler auto-launch + cloud sync integration
"""
from dotenv import load_dotenv
load_dotenv()

import os
import threading
import subprocess
import time
from datetime import datetime
import pytz
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from .config import PATHS  # ✅ unified path import

# -------------------------------------------------------------
# FastAPI Init
# -------------------------------------------------------------
app = FastAPI(title="AION Analytics Backend", version="1.8.2")

# --- CORS setup (for frontend access) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Routers (imported AFTER app init)
# -------------------------------------------------------------
from backend.system_status_router import router as system_router
from backend.live_prices_router import router as live_router
from backend.insights_router import router as insights_router
from backend.dashboard_router import router as dashboard_router

# Register routers
app.include_router(system_router)
app.include_router(live_router)
app.include_router(insights_router)
app.include_router(dashboard_router)

# Predict
@app.get("/predict")
def predict(ticker: str):
    try:
        import ai_model
        res = ai_model.predict_for_symbol(ticker)
        return {"ticker": ticker.upper(), "predictions": res}
    except Exception as e:
        return {"error": str(e)}

# Insights filters
@app.get("/insights/filters")
def insights_filters(horizon: str = "1w"):
    sectors = []
    try:
        from insights_builder import HORIZON_KEYS
    except Exception:
        HORIZON_KEYS = ["1w","1m","1y"]
    try:
        from data_pipeline import STOCK_CACHE, load_all_cached_stocks
        load_all_cached_stocks()
        sectors = sorted({v.get("sector") for v in (STOCK_CACHE or {}).values() if isinstance(v, dict) and v.get("sector")})
    except Exception:
        pass
    return {"horizons": HORIZON_KEYS, "sectors": sectors}

# Insights top picks
@app.get("/insights/top-picks")
def insights_top_picks(
    horizon: str = "1w",
    limit: int = 50,
    sector: Optional[str] = None,
    price_max: Optional[float] = None,
    vol_min: Optional[int] = None,
    conf_min: Optional[float] = None,
):
    import json
    data = []
    try:
        path = PATHS["ml_data"] / f"daily_insights_{horizon}.json"  # ✅ from config
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict) and isinstance(raw.get("picks"), list):
                    data = raw["picks"]
                elif isinstance(raw, list):
                    data = raw
    except Exception:
        data = []

    def keep(r: dict) -> bool:
        if sector and r.get("sector") != sector:
            return False
        if price_max is not None and isinstance(r.get("currentPrice"), (int,float)) and r["currentPrice"] > price_max:
            return False
        if vol_min is not None and isinstance(r.get("volume"), (int,float)) and r["volume"] < vol_min:
            return False
        if conf_min is not None and isinstance(r.get("confidence"), (int,float)) and r["confidence"] < conf_min:
            return False
        return True

    data = [r for r in data if isinstance(r, dict) and keep(r)]
    data.sort(key=lambda r: r.get("rankingScore", r.get("score", r.get("expectedReturnPct", 0))), reverse=True)
    return {"horizon": horizon, "count": len(data), "picks": data[:limit]}

# Intraday last
@app.get("/intraday/last")
def intraday_last(tickers: List[str] = Query(default=[])):
    out = {}
    try:
        from data_pipeline import STOCK_CACHE, load_all_cached_stocks
        load_all_cached_stocks()
        for t in tickers or []:
            rec = (STOCK_CACHE or {}).get(t) or {}
            out[t] = {"price": rec.get("price"), "volume": rec.get("volume")}
    except Exception:
        for t in tickers or []:
            out[t] = {"price": None, "volume": None}
    return {"data": out}

# Optimizer (RL → GNN fallback)
@app.post("/optimizer/run")
def optimizer_run(params: Dict[str, Any]):
    method_req = (params or {}).get("method", "").upper()
    if method_req != "GNN":
        try:
            from rl_env import run_rl_optimizer
            res = run_rl_optimizer(params or {})
            if isinstance(res, dict):
                res.setdefault("method", "RL")
                n = len((res.get("optimized") or {}).keys())
                exp = res.get("expected_return")
                print(f"✅ RL optimizer used ({n} tickers)")
                res.setdefault("summary", f"✅ RL optimizer allocated {n} assets with expected {float(exp)*100:.1f}% return" if isinstance(exp,(int,float)) else f"✅ RL optimizer allocated {n} assets")
                return res
        except Exception:
            pass
    try:
        from gnn_model import run_gnn_optimizer
        res = run_gnn_optimizer(params or {})
        if isinstance(res, dict):
            res.setdefault("method", "GNN")
            n = len((res.get("optimized") or {}).keys())
            exp = res.get("expected_return")
            print(f"✅ GNN optimizer used ({n} tickers)")
            res.setdefault("summary", f"✅ GNN optimizer allocated {n} assets with expected {float(exp)*100:.1f}% return" if isinstance(exp,(int,float)) else f"✅ GNN optimizer allocated {n} assets")
            return res
    except Exception:
        pass
    return {"error": "No optimizer available (rl_env or gnn_model not found or failed)."}

# Explain
@app.get("/explain/{ticker}")
def explain_ticker(ticker: str):
    try:
        from explainability import explain_prediction
        res = explain_prediction(ticker)
        print(f"✅ Explainability generated for {ticker.upper()}")
        return res if isinstance(res, dict) else {"ticker": ticker.upper(), "result": res}
    except Exception:
        try:
            import ai_model
            feats = []
            try:
                feats = ai_model.get_feature_names()
            except Exception:
                feats = ["RSI_14","Momentum_7d","Volatility_20d","MACD","Volume_Delta"]
            weights = [0.32, 0.24, 0.18, 0.14, 0.12]
            top_features = [{"feature": f, "importance": round(weights[i], 3)} for i, f in enumerate(feats[:5])]
            try:
                pred = ai_model.predict_for_symbol(ticker)
                predicted_price, confidence = None, None
                if isinstance(pred, dict):
                    for v in pred.values():
                        if isinstance(v, dict) and "predictedPrice" in v:
                            predicted_price = v.get("predictedPrice")
                            confidence = v.get("confidence")
                            break
                print(f"✅ Explainability (fallback) for {ticker.upper()}")
                return {"ticker": ticker.upper(), "predicted_price": predicted_price, "confidence": confidence, "top_features": top_features}
            except Exception:
                return {"ticker": ticker.upper(), "predicted_price": None, "confidence": None, "top_features": top_features}
        except Exception:
            return {"ticker": ticker.upper(), "predicted_price": None, "confidence": None, "top_features": []}

# ------------------------------------------------------------
# Dashboard Routes v1.5.0
# ------------------------------------------------------------
@app.get("/dashboard/metrics")
def dashboard_metrics():
    cache_path = PATHS["dashboard_cache"] / "metrics.json"  # ✅ from config
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        from dashboard_builder import compute_accuracy
        res = compute_accuracy(days=30, tolerance=10, horizons=("1w","1m"))
        print(f"✅ Dashboard accuracy computed: {res.get('accuracy_30d')} (30d, tol ±{res.get('tolerance')})")
        return res
    except Exception as e:
        return {"error": f"metrics unavailable: {e}"}

@app.get("/dashboard/top-performers")
def dashboard_top_performers(horizon: str = "1w"):
    cache_path = PATHS["dashboard_cache"] / "top_performers.json"  # ✅ from config
    base = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                base = json.load(f)
        except Exception:
            base = {}
    frozen = base.get(horizon) or []
    out, live_map = [], {}
    try:
        from data_pipeline import STOCK_CACHE, load_all_cached_stocks
        load_all_cached_stocks()
        for r in frozen:
            t = r.get("ticker")
            rec = (STOCK_CACHE or {}).get(t) or {}
            live_map[t] = {"price": rec.get("price")}
    except Exception:
        pass
    for r in frozen:
        t = r.get("ticker")
        pred_price = r.get("pred_price")
        now = live_map.get(t, {}).get("price")
        gain = None
        if isinstance(pred_price, (int, float)) and isinstance(now, (int, float)) and pred_price > 0:
            gain = (now - pred_price) / pred_price * 100.0
        out.append({
            "ticker": t,
            "price_then": pred_price,
            "price_now": now,
            "gain_pct": round(gain, 4) if isinstance(gain, (int, float)) else None,
            "frozen_on": r.get("frozen_on"),
        })
    return {"horizon": horizon, "tickers": out, "last_updated": base.get("last_updated")}

# -------------------------------------------------------------
# 🧭 Backend heartbeat (every hour)
# -------------------------------------------------------------
def _backend_heartbeat():
    tz = pytz.timezone("America/New_York")
    while True:
        now = datetime.now(tz)
        print(f"[Backend] ❤️ Alive — {now:%H:%M %Z}", flush=True)
        time.sleep(3600)

# -------------------------------------------------------------
# 🧩 Scheduler launcher
# -------------------------------------------------------------
def _scheduler_heartbeat():
    tz = pytz.timezone("America/New_York")
    while True:
        now = datetime.now(tz)
        print(f"[Scheduler] ❤️ Alive — {now:%H:%M %Z}", flush=True)
        time.sleep(3600)

@app.on_event("startup")
def start_scheduler():
    def _run_scheduler():
        try:
            print("[Scheduler] 🚀 Launching scheduler_runner.py ...", flush=True)
            process = subprocess.Popen(["python", "-m", "backend.scheduler_runner"], stdout=None, stderr=None)
            print(f"[Scheduler] ✅ Running in background (PID {process.pid})", flush=True)
        except Exception as e:
            print(f"[Scheduler] ⚠️ Failed to start: {e}", flush=True)

    threading.Thread(target=_run_scheduler, daemon=True).start()
    threading.Thread(target=_scheduler_heartbeat, daemon=True).start()
    print("[Scheduler] 🕒 Background scheduler thread started.\n", flush=True)

print("[Backend] ⚙️  Loading routers & cloud sync...", flush=True)

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(system_router, prefix="/system", tags=["System"])
app.include_router(live_router, prefix="/live", tags=["Live Prices"])
app.include_router(insights_router, prefix="/insights", tags=["Insights"])

@app.get("/")
def root():
    return {
        "service": "AION Analytics Backend",
        "version": "1.8.2",
        "status": "online",
        "message": "Predict, Learn, Evolve."
    }

@app.on_event("startup")
def _banner():
    print("[Backend] 🚀 Initializing backend service...")
    print("📅 Scheduler auto-start enabled (Eastern Time)")
    print("[Backend] ✅ Startup sequence complete — running at http://127.0.0.1:8080\n", flush=True)

# ==============================================================
# ---------- Smart Bot Bootstrap on Late Start -----------------
# ==============================================================
import pytz
from datetime import datetime, time as dtime

def auto_start_bots_on_late_launch():
    """If system starts after 7:30 ET but before close, run 1w/2w/4w bot cycles once."""
    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    start_cutoff = dtime(7, 30)
    close_cutoff = dtime(16, 0)

    if start_cutoff <= now.time() <= close_cutoff and now.weekday() < 5:
        print("⏰ Late startup detected — running FULL bot cycle now (pre-market recovery).")
        try:
            from backend.trading_bot_nightly_1w import run_all_bots as run_1w
            from backend.trading_bot_nightly_2w import run_all_bots as run_2w
            from backend.trading_bot_nightly_4w import run_all_bots as run_4w
            run_1w("full"); print("✅ 1w bots complete (startup catch-up)")
            run_2w("full"); print("✅ 2w bots complete (startup catch-up)")
            run_4w("full"); print("✅ 4w bots complete (startup catch-up)")
        except Exception as e:
            print(f"⚠️ Auto-start full bots failed: {e}")
    else:
        print("⏳ Startup outside market hours — skipping auto bot start.")

# ==============================================================
# ---------- Smart Intraday Bot Bootstrap on Late Start ---------
# ==============================================================
def auto_start_daytrading_on_late_launch():
    """If system starts after 7:30 ET but before close, run day-trading full cycle once."""
    from dt_backend.daytrading_job import run
    import pytz
    from datetime import datetime, time as dtime
    import threading, time

    def _heartbeat():
        while True:
            print("[Backend] ❤️ Alive —", datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M %Z"))
            time.sleep(60)

    # after app/scheduler start
    threading.Thread(target=_heartbeat, daemon=True).start()

    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    start_cutoff = dtime(7, 30)
    close_cutoff = dtime(16, 0)

    if start_cutoff <= now.time() <= close_cutoff and now.weekday() < 5:
        print("⏰ Late startup detected — running DAY-TRADING full cycle now.")
        try:
            run()
            print("✅ Day-trading bots complete (startup catch-up)")
        except Exception as e:
            print(f"⚠️ Auto-start day-trading bots failed: {e}")
    else:
        print("⏳ Startup outside market hours — skipping day-trading auto start.")

# Run once on backend startup
auto_start_daytrading_on_late_launch()

from . import cloud_sync

def _on_new_release(manifest):
    new_ver = manifest.get("version")
    notes = manifest.get("notes", "")
    print(f"🆕 Update available: v{new_ver} — {notes}")
    path = cloud_sync.download_update_zip(manifest, dest_dir=str(PATHS["updates"]))  # ✅ from config
    print(f"⬇️  Update saved to: {path}")

def _check_current_version():
    current = "1.0.0"
    try:
        manifest = cloud_sync.version_check_once()
        if not manifest:
            print("[cloud_sync] ⚠️ No manifest found on Supabase.")
            return
        latest = manifest.get("version", "unknown")
        if latest == "unknown":
            print(f"[cloud_sync] ⚠️ Could not determine remote version.")
            return
        if cloud_sync._compare_versions(latest, current) <= 0:
            print(f"[cloud_sync] ✅ Running latest version (v{current})")
        else:
            print(f"[cloud_sync] 🚨 New version available: v{latest} > v{current}")
            _on_new_release(manifest)
    except Exception as e:
        print(f"[cloud_sync] ⚠️ Version check failed: {e}")

print("[Backend] ☁️  Cloud Sync active — version check...", flush=True)
_check_current_version()

try:
    cloud_sync.start_background_tasks()
except Exception as e:
    print(f"[cloud_sync] ⚠️ Cloud sync init failed: {e}")

threading.Thread(target=_backend_heartbeat, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    print("[Backend] ✅ Startup complete — running at http://127.0.0.1:8000")
    uvicorn.run("backend.backend_service:app", host="127.0.0.1", port=8000, reload=True)
