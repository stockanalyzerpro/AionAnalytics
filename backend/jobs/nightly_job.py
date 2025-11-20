# backend/jobs/nightly_job.py — v3.3
"""
Nightly Job — AION Analytics (Backend EOD Brain, Hybrid + Optuna)

Pipeline:
    1) Lock + load rolling
    2) Heal / backfill history
    3) Fundamentals
    4) Metrics
    5) Macro features
    6) Social sentiment
    7) News fetch + intel
    8) ML Dataset build (daily)
    9) Hybrid model training (LightGBM + LSTM + Transformer + Optuna weekly)
    10) Predictions → rolling["predictions"]
    11) Context state
    12) Regime detection
    13) Policy engine
    14) Continuous learning (drift, performance, early retrain trigger)
    15) Insights builder
    16) Supervisor agent
    17) Summary log

Run:
    python -m backend.jobs.nightly_job
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Ensure backend is importable when run as module
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, save_rolling, log

# Services
from backend.services.backfill_history import backfill_symbols
from backend.services.fundamentals_fetcher import update_fundamentals
from backend.services.metrics_fetcher import build_metrics
from backend.services.macro_fetcher import build_macro_features
from backend.services.social_sentiment_fetcher import build_social_sentiment
from backend.services.news_fetcher import run_news_fetch
from backend.services.news_intel import build_news_intel
from backend.services.ml_data_builder import build_daily_dataset
from backend.services.insights_builder import build_daily_insights
from backend.services.prediction_logger import log_predictions

# Core ML + policy + learning
from backend.core.ai_model import train_all_models, predict_all
from backend.core.context_state import build_context
from backend.core.regime_detector import detect_regime
from backend.core.policy_engine import apply_policy
from backend.core.continuous_learning import run_continuous_learning
from backend.core.supervisor_agent import run_supervisor_agent

LOCK_FILE = PATHS["nightly_lock"]
SUMMARY_FILE = PATHS["logs"] / "nightly" / "last_nightly_summary.json"


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def _acquire_lock() -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        log(f"[nightly_job]⚠️ Lock present at {LOCK_FILE} — exiting.")
        return False
    LOCK_FILE.write_text(datetime.now(TIMEZONE).isoformat(), encoding="utf-8")
    return True


def _release_lock():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as e:
        log(f"[nightly_job] ⚠️ Failed to release lock: {e}")


def _phase(title: str):
    log("")
    log(f"──────── {title} ────────")


# ----------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------

def run_nightly_job() -> Dict[str, Any]:
    if not _acquire_lock():
        return {"status": "skipped", "reason": "lock_present"}

    summary: Dict[str, Any] = {
        "started_at": datetime.now(TIMEZONE).isoformat(),
        "phases": {},
    }

    try:
        # 1) Load rolling
        _phase("Load rolling cache")
        rolling = _read_rolling() or {}
        summary["phases"]["rolling_symbols_before"] = len(rolling)

        # 2) Heal / backfill history
        _phase("Heal / backfill history")
        try:
            backfill_result = backfill_symbols(rolling)
            summary["phases"]["backfill"] = backfill_result
            rolling = _read_rolling() or rolling
        except Exception as e:
            log(f"[nightly_job] ⚠️ backfill_symbols failed: {e}")
            summary["phases"]["backfill"] = {"error": str(e)}

        summary["phases"]["rolling_symbols_after"] = len(rolling)

        # 3) Fundamentals enrichment
        _phase("Fundamentals fetch")
        try:
            summary["phases"]["fundamentals"] = update_fundamentals(rolling)
            rolling = _read_rolling() or rolling
        except Exception as e:
            log(f"[nightly_job] ⚠️ fundamentals fetch failed: {e}")
            summary["phases"]["fundamentals"] = {"error": str(e)}

        # 4) Metrics
        _phase("Metrics refresh")
        try:
            summary["phases"]["metrics"] = build_metrics(rolling)
            rolling = _read_rolling() or rolling
        except Exception as e:
            log(f"[nightly_job] ⚠️ metrics refresh failed: {e}")
            summary["phases"]["metrics"] = {"error": str(e)}

        # 5) Macro features
        _phase("Macro features")
        try:
            summary["phases"]["macro"] = build_macro_features()
        except Exception as e:
            log(f"[nightly_job] ⚠️ macro features failed: {e}")
            summary["phases"]["macro"] = {"error": str(e)}

        # 6) Social sentiment
        _phase("Social sentiment")
        try:
            summary["phases"]["social"] = build_social_sentiment()
        except Exception as e:
            log(f"[nightly_job] ⚠️ social sentiment failed: {e}")
            summary["phases"]["social"] = {"error": str(e)}

        # 7) News fetch + intel
        _phase("News fetch + intel")
        try:
            run_news_fetch()
            summary["phases"]["news_intel"] = build_news_intel()
        except Exception as e:
            log(f"[nightly_job] ⚠️ news flow failed: {e}")
            summary["phases"]["news_intel"] = {"error": str(e)}

        # 8) ML dataset build (daily)
        _phase("ML dataset build (daily)")
        try:
            ds_info = build_daily_dataset()
            summary["phases"]["dataset"] = ds_info
        except Exception as e:
            log(f"[nightly_job] ⚠️ dataset build failed: {e}")
            summary["phases"]["dataset"] = {"error": str(e)}

        # 9) Model training (hybrid + Optuna weekly)
        _phase("Model training (hybrid + Optuna weekly)")
        try:
            today = datetime.now(TIMEZONE)
            use_optuna = today.weekday() == 0  # Monday = 0
            n_trials = 20 if use_optuna else 0

            summary["phases"]["training"] = train_all_models(
                dataset_name="training_data_daily.parquet",
                use_optuna=use_optuna,
                n_trials=n_trials,
            )
        except Exception as e:
            log(f"[nightly_job] ⚠️ training failed: {e}")
            summary["phases"]["training"] = {"error": str(e)}

        # 10) Predictions → rolling
        _phase("Hybrid predictions → Rolling")
        try:
            rolling = _read_rolling() or rolling
            preds = predict_all(rolling)
            count_syms = 0
            for sym, res in preds.items():
                node = rolling.get(sym, {})
                node["predictions"] = res
                rolling[sym] = node
                count_syms += 1
            save_rolling(rolling)
            log(f"[nightly_job] 🤖 Predictions updated for {count_syms} symbols in Rolling.")
            summary["phases"]["predictions"] = {"symbols": count_syms}
        except Exception as e:
            log(f"[nightly_job] ⚠️ prediction phase failed: {e}")
            summary["phases"]["predictions"] = {"error": str(e)}

        # Optional prediction logging
        _phase("Prediction logging")
        try:
            summary["phases"]["prediction_logger"] = log_predictions()
        except Exception as e:
            log(f"[nightly_job] ⚠️ prediction_logger failed: {e}")
            summary["phases"]["prediction_logger"] = {"error": str(e)}

        # 11) Context state
        _phase("Context state")
        try:
            summary["phases"]["context"] = build_context()
        except Exception as e:
            log(f"[nightly_job] ⚠️ context_state failed: {e}")
            summary["phases"]["context"] = {"error": str(e)}

        # 12) Regime detection
        _phase("Regime detection")
        try:
            summary["phases"]["regime"] = detect_regime()
        except Exception as e:
            log(f"[nightly_job] ⚠️ regime detection failed: {e}")
            summary["phases"]["regime"] = {"error": str(e)}

        # 13) Policy engine
        _phase("Policy engine")
        try:
            summary["phases"]["policy"] = apply_policy()
        except Exception as e:
            log(f"[nightly_job] ⚠️ policy engine failed: {e}")
            summary["phases"]["policy"] = {"error": str(e)}

        # 14) Continuous learning (drift, performance, early retrain trigger)
        _phase("Continuous learning (drift + early retrain)")
        try:
            summary["phases"]["continuous_learning"] = run_continuous_learning()
        except Exception as e:
            log(f"[nightly_job] ⚠️ continuous_learning failed: {e}")
            summary["phases"]["continuous_learning"] = {"error": str(e)}

        # 15) Insights builder
        _phase("Insights builder")
        try:
            summary["phases"]["insights"] = build_daily_insights()
        except Exception as e:
            log(f"[nightly_job] ⚠️ insights builder failed: {e}")
            summary["phases"]["insights"] = {"error": str(e)}

        # 16) Supervisor agent
        _phase("Supervisor agent")
        try:
            summary["phases"]["supervisor"] = run_supervisor_agent()
        except Exception as e:
            log(f"[nightly_job] ⚠️ supervisor agent failed: {e}")
            summary["phases"]["supervisor"] = {"error": str(e)}

        summary["finished_at"] = datetime.now(TIMEZONE).isoformat()
        summary["status"] = "ok"

        # 17) Write summary
        try:
            SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
            SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"[nightly_job] ⚠️ Failed to write summary: {e}")

        return summary

    finally:
        _release_lock()


if __name__ == "__main__":
    out = run_nightly_job()
    print(json.dumps(out, indent=2))
