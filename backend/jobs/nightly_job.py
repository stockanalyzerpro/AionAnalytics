# nightly_job.py — AION Analytics (Hybrid v4.0)
# Full pipeline: Backfill → Enrich → ML Dataset → Train → Predict
# → Context → Regime → Policy → Continuous Learning → Insights → Supervisor

from __future__ import annotations

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, save_rolling, log
from backend.core import (
    context_state,
    regime_detector,
    policy_engine,
    continuous_learning,
    ai_model,
    supervisor_agent,
)

from backend.services.backfill_history import backfill_symbols
from backend.services.fundamentals_fetcher import enrich_fundamentals
from backend.services.metrics_fetcher import build_latest_metrics
from backend.services.macro_fetcher import build_macro_features
from backend.services.news_intel import run_news_intel
from backend.services.social_sentiment_fetcher import run_social_sentiment
from backend.services.insights_builder import build_daily_insights
from backend.services import ml_data_builder


# ============================================================
# Lock file (one nightly job at a time)
# ============================================================

JOB_LOCK_PATH = PATHS["stock_cache_master"] / "nightly_job.lock"
os.makedirs(JOB_LOCK_PATH.parent, exist_ok=True)


def _ensure_not_directory(p: Path) -> None:
    """If a directory exists where a lock file should be, delete it if empty."""
    if p.exists() and p.is_dir():
        try:
            os.rmdir(p)
            log(f"⚠️ Found directory instead of file at {p}, removing...")
        except Exception as e:
            log(f"⚠️ Could not remove unexpected lock directory {p}: {e}")


def _blocking_acquire_lock(lock_path: Path, poll_secs: float = 0.5) -> bool:
    """
    Blocking file-lock via O_EXCL create; waits until holder releases.
    Safe on Windows & Unix.
    """
    _ensure_not_directory(lock_path)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} ts={datetime.utcnow().isoformat()}")
            return True
        except FileExistsError:
            time.sleep(poll_secs)
        except PermissionError as e:
            log(f"⚠️ Permission error creating job lock: {e}")
            time.sleep(poll_secs)
        except Exception as e:
            log(f"⚠️ job lock create failed: {e}")
            time.sleep(poll_secs)


def _job_lock_acquire() -> bool:
    log(f"[nightly_job] ⛓️  Acquiring nightly lock at {JOB_LOCK_PATH} ...")
    _blocking_acquire_lock(JOB_LOCK_PATH)
    log("[nightly_job] ✅ Lock acquired.")
    return True


def _job_lock_release() -> None:
    try:
        if JOB_LOCK_PATH.exists():
            JOB_LOCK_PATH.unlink()
            log("[nightly_job] 🔓 Lock released.")
    except Exception as e:
        log(f"⚠️ Failed to release nightly lock: {e}")


# ============================================================
# Helpers
# ============================================================

def _phase(title: str) -> None:
    log(f"\n────────── {title} ──────────")


def _write_summary(summary: Dict[str, Any]) -> None:
    """Persist last run summary in logs/nightly."""
    try:
        logs_dir = PATHS["nightly_logs"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = logs_dir / "last_nightly_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"[nightly_job] 📝 Summary written → {out_path}")
    except Exception as e:
        log(f"[nightly_job] ⚠️ Failed to write summary: {e}")


# ============================================================
# MAIN PIPELINE — Option C Hybrid Flow
# ============================================================

def run() -> Dict[str, Any]:
    """
    Full nightly job:

      1) Load & inspect rolling
      2) Backfill / heal
      3) Fundamentals + metrics + macro
      4) Social sentiment + news intel
      5) ML dataset build
      6) Model training
      7) Nightly predictions → inject into rolling
      8) Context builder
      9) Regime detection
     10) Policy engine
     11) Continuous learning (Hybrid Mode C)
     12) Insights builder
     13) Supervisor agent verdict
    """
    t0 = time.time()
    summary: Dict[str, Any] = {
        "started_at": datetime.now(TIMEZONE).isoformat(),
        "steps": {},
    }

    _job_lock_acquire()
    try:
        log("\n🛠️  Nightly Job — AION Analytics (Hybrid v4.0)")
        log("     Pipeline: Backfill → Enrich → ML → Policy → Learn → Insights")

        # --------------------------------------------------
        # 1) Load Rolling
        # --------------------------------------------------
        _phase("Load Rolling cache")
        rolling = _read_rolling() or {}
        summary["symbols_before"] = len(rolling)
        log(f"[nightly_job] 📦 Rolling present — {len(rolling)} tickers loaded.")

        # --------------------------------------------------
        # 2) Backfill / Heal (Rolling-aware)
        # --------------------------------------------------
        _phase("Backfill / heal Rolling")
        try:
            symbols = list(rolling.keys())
            updated = backfill_symbols(symbols)
            summary["steps"]["backfill"] = {"status": "ok", "updated": int(updated)}
        except Exception as e:
            log(f"[nightly_job] ⚠️ backfill_symbols failed: {e}")
            summary["steps"]["backfill"] = {"status": "error", "error": str(e)}

        # Re-read rolling after backfill
        rolling = _read_rolling() or {}
        summary["symbols_after_backfill"] = len(rolling)

        # --------------------------------------------------
        # 3) Fundamentals + Metrics + Macro
        # --------------------------------------------------
        _phase("Fundamentals enrichment")
        try:
            fsum = enrich_fundamentals()
            summary["steps"]["fundamentals"] = {"status": "ok", "summary": fsum}
        except Exception as e:
            log(f"[nightly_job] ⚠️ enrich_fundamentals failed: {e}")
            summary["steps"]["fundamentals"] = {"status": "error", "error": str(e)}

        _phase("Metrics enrichment")
        try:
            msum = build_latest_metrics()
            summary["steps"]["metrics"] = {"status": "ok", "summary": msum}
        except Exception as e:
            log(f"[nightly_job] ⚠️ build_latest_metrics failed: {e}")
            summary["steps"]["metrics"] = {"status": "error", "error": str(e)}

        _phase("Macro features")
        try:
            macro_res = build_macro_features()
            summary["steps"]["macro"] = {"status": "ok", "result": str(macro_res)}
        except Exception as e:
            log(f"[nightly_job] ⚠️ build_macro_features failed: {e}")
            summary["steps"]["macro"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 4) Social Sentiment + News Intel
        # --------------------------------------------------
        _phase("Social sentiment (Reddit, etc.)")
        try:
            run_social_sentiment()
            summary["steps"]["social_sentiment"] = {"status": "ok"}
        except Exception as e:
            log(f"[nightly_job] ⚠️ run_social_sentiment failed: {e}")
            summary["steps"]["social_sentiment"] = {"status": "error", "error": str(e)}

        _phase("News intelligence")
        try:
            news_summary = run_news_intel()
            summary["steps"]["news_intel"] = {"status": "ok", "summary": news_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ run_news_intel failed: {e}")
            summary["steps"]["news_intel"] = {"status": "error", "error": str(e)}

        # Re-read rolling to capture any news-enriched nodes
        rolling = _read_rolling() or {}

        # --------------------------------------------------
        # 5) ML Dataset Build
        # --------------------------------------------------
        _phase("ML dataset build (daily horizon)")
        try:
            df = ml_data_builder.build_ml_dataset("daily")
            summary["steps"]["ml_dataset"] = {
                "status": "ok",
                "rows": int(len(df)),
            }
        except Exception as e:
            log(f"[nightly_job] ⚠️ build_ml_dataset failed: {e}")
            summary["steps"]["ml_dataset"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 6) Model Training
        # --------------------------------------------------
        _phase("Model training (LightGBM / RF)")
        try:
            train_summary = ai_model.train_model(dataset_name="training_data_daily.parquet")
            summary["steps"]["training"] = {"status": train_summary.get("status"), **train_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ train_model failed: {e}")
            summary["steps"]["training"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 7) Nightly Predictions → Inject into Rolling
        # --------------------------------------------------
        _phase("AI predictions → Rolling")
        try:
            rolling = _read_rolling() or {}
            preds = ai_model.predict_all(rolling) or {}
            updated_syms = 0
            for sym, pred_block in preds.items():
                node = rolling.get(sym, {"symbol": sym})
                existing = node.get("predictions") or {}
                # Merge nightly horizon predictions (e.g., "1d")
                existing.update(pred_block)
                node["predictions"] = existing
                rolling[sym] = node
                updated_syms += 1

            save_rolling(rolling)
            log(f"[nightly_job] 🤖 Predictions updated for {updated_syms} symbols.")
            summary["steps"]["predictions"] = {"status": "ok", "symbols": updated_syms}
        except Exception as e:
            log(f"[nightly_job] ⚠️ prediction phase failed: {e}")
            summary["steps"]["predictions"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 8) Context Builder
        # --------------------------------------------------
        _phase("Build nightly context")
        try:
            ctx_summary = context_state.build_context()
            summary["steps"]["context"] = {"status": "ok", "summary": ctx_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ build_context failed: {e}")
            summary["steps"]["context"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 9) Regime Detection
        # --------------------------------------------------
        _phase("Regime detection")
        try:
            regime = regime_detector.detect_regime()
            summary["steps"]["regime"] = {"status": "ok", "regime": regime}
        except Exception as e:
            log(f"[nightly_job] ⚠️ detect_regime failed: {e}")
            summary["steps"]["regime"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 10) Policy Engine
        # --------------------------------------------------
        _phase("Policy engine")
        try:
            pol_summary = policy_engine.apply_policy()
            summary["steps"]["policy"] = {"status": "ok", "summary": pol_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ apply_policy failed: {e}")
            summary["steps"]["policy"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 11) Continuous Learning (Hybrid Mode C)
        # --------------------------------------------------
        _phase("Continuous learning (Hybrid)")
        try:
            cl_summary = continuous_learning.run_continuous_learning()
            summary["steps"]["continuous_learning"] = {"status": "ok", "summary": cl_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ continuous_learning failed: {e}")
            summary["steps"]["continuous_learning"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 12) Insights Builder
        # --------------------------------------------------
        _phase("Insights builder (Top-boards)")
        try:
            ib_summary = build_daily_insights(limit=50)
            summary["steps"]["insights"] = {"status": "ok", "summary": ib_summary}
        except Exception as e:
            log(f"[nightly_job] ⚠️ build_daily_insights failed: {e}")
            summary["steps"]["insights"] = {"status": "error", "error": str(e)}

        # --------------------------------------------------
        # 13) Supervisor Agent Verdict
        # --------------------------------------------------
        _phase("Supervisor agent")
        try:
            verdict = supervisor_agent.run_supervisor()
            summary["steps"]["supervisor"] = {"status": "ok", "verdict": verdict}
            summary["verdict"] = verdict.get("verdict", "UNKNOWN")
        except Exception as e:
            log(f"[nightly_job] ⚠️ supervisor_agent failed: {e}")
            summary["steps"]["supervisor"] = {"status": "error", "error": str(e)}
            summary["verdict"] = "ERROR"

        # --------------------------------------------------
        # Done
        # --------------------------------------------------
        total_time = time.time() - t0
        summary["finished_at"] = datetime.now(TIMEZONE).isoformat()
        summary["elapsed_seconds"] = round(total_time, 2)
        summary["status"] = "ok"

        log(f"\n✅ Nightly job complete in {total_time:.1f}s.")
        _write_summary(summary)
        return summary

    except Exception as e:
        log(f"❌ Nightly job fatal error: {e}")
        traceback.print_exc()
        summary["status"] = "error"
        summary["error"] = str(e)
        _write_summary(summary)
        return summary
    finally:
        _job_lock_release()


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
