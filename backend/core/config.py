"""
AION Analytics — Central Configuration Module
----------------------------------------------

This file defines ALL backend paths and environment configuration used by:

    • backend/core (models, policy, context, regime, learning)
    • backend/services (fetchers, builders, logs, insights)
    • backend/jobs (nightly, intraday, system)
    • backend/routers (API responses)
    • dt_backend bridging modules

Rules:
    ✔ No hard-coded paths
    ✔ Everything derives from PROJECT ROOT
    ✔ Safe on Windows (no fcntl)
    ✔ Mirrors dt_backend path style (DT_PATHS)
    ✔ Includes all storage needed for nightly + intraday engines
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
import pytz


# ============================================================
#  PROJECT ROOT
# ============================================================
# backend/core/config.py → backend/core/ → backend/ → Aion_Analytics/
ROOT = Path(__file__).resolve().parents[2]


# ============================================================
#  TIMEZONE
# ============================================================
TIMEZONE = pytz.timezone(os.getenv("AION_TZ", "America/Chicago"))


# ============================================================
#  DATA ROOTS
# ============================================================
DATA_ROOT = ROOT / "data"

RAW_ROOT = DATA_ROOT / "raw"
RAW_DAILY = RAW_ROOT / "daily_bars"
RAW_INTRADAY = RAW_ROOT / "intraday_bars"
RAW_NEWS = RAW_ROOT / "news"
RAW_SOCIAL = RAW_ROOT / "social"
RAW_FUNDAMENTALS = RAW_ROOT / "fundamentals"

UNIVERSE_ROOT = DATA_ROOT / "universe"
CACHE_ROOT = DATA_ROOT / "data_cache"


# ============================================================
#  ROLLING + BRAINS + BACKUPS
# ============================================================
STOCK_CACHE_ROOT = DATA_ROOT / "stock_cache"
STOCK_CACHE_MASTER = STOCK_CACHE_ROOT / "master"

ROLLING_PATH = STOCK_CACHE_MASTER / "rolling.json.gz"
ROLLING_BRAIN = STOCK_CACHE_MASTER / "rolling_brain.json.gz"
ROLLING_BACKUPS = STOCK_CACHE_MASTER / "backups"


# ============================================================
#  ML (NIGHTLY)
# ============================================================
ML_ROOT = ROOT / "ml_data"
ML_MODELS = ML_ROOT / "models"
ML_PREDICTIONS = ML_ROOT / "predictions"
ML_DATASETS = ML_ROOT / "datasets"
ML_TRAINING = ML_ROOT / "training"


# ============================================================
#  ML (INTRADAY / DT-BACKEND)
# ============================================================
MLDT_ROOT = ROOT / "ml_data_dt"
MLDT_INTRADAY = MLDT_ROOT / "intraday"
MLDT_INTRADAY_DATASETS = MLDT_INTRADAY / "dataset"
MLDT_INTRADAY_MODELS = MLDT_INTRADAY / "models"
MLDT_INTRADAY_PREDICTIONS = MLDT_INTRADAY / "predictions"
MLDT_INTRADAY_REPLAY = MLDT_INTRADAY / "replay"
MLDT_INTRADAY_RAW_DAYS = MLDT_INTRADAY_REPLAY / "raw_days"
MLDT_INTRADAY_REPLAY_RESULTS = MLDT_INTRADAY_REPLAY / "replay_results"


# ============================================================
#  INSIGHTS / REPORTS / ANALYTICS
# ============================================================
INSIGHTS_ROOT = ROOT / "insights"
DASHBOARD_ROOT = DATA_ROOT / "dashboard_cache"

ANALYTICS_ROOT = ROOT / "analytics"
ANALYTICS_PNL = ANALYTICS_ROOT / "pnl"
ANALYTICS_PERFORMANCE = ANALYTICS_ROOT / "performance"


# ============================================================
#  LOGS
# ============================================================
LOGS_ROOT = ROOT / "logs"
LOGS_BACKEND = LOGS_ROOT / "backend"
LOGS_NIGHTLY = LOGS_ROOT / "nightly"
LOGS_SCHEDULER = LOGS_ROOT / "scheduler"
LOGS_INTRADAY = LOGS_ROOT / "intraday"


# ============================================================
#  NEWS / SENTIMENT
# ============================================================
NEWS_CACHE = DATA_ROOT / "news_cache"
NEWS_DASHBOARD_JSON = NEWS_CACHE / "news_dashboard_latest.json"
SENTIMENT_MAP = NEWS_CACHE / "sentiment_map_latest.json"


# ============================================================
#  MARKET / REGIME STATE
# ============================================================
MARKET_STATE = ML_ROOT / "market_state.json"
REGIME_STATE = ML_ROOT / "regime_state.json"


# ============================================================
#  CLOUD CACHE (SUPABASE)
# ============================================================
CLOUD_CACHE = DATA_ROOT / "cloud_cache"
UPDATES_DIR = DATA_ROOT / "updates"


# ============================================================
#  PATHS DICTIONARY (BACKEND)
# ============================================================
PATHS: Dict[str, Path] = {
    "root": ROOT,

    # --- Raw Data ---
    "raw_daily": RAW_DAILY,
    "raw_intraday": RAW_INTRADAY,
    "raw_news": RAW_NEWS,
    "raw_social": RAW_SOCIAL,
    "raw_fundamentals": RAW_FUNDAMENTALS,

    # --- Stock / Rolling Cache ---
    "stock_cache": STOCK_CACHE_ROOT,
    "stock_cache_master": STOCK_CACHE_MASTER,
    "rolling": ROLLING_PATH,
    "rolling_brain": ROLLING_BRAIN,
    "rolling_backups": ROLLING_BACKUPS,

    # --- ML (Nightly) ---
    "ml_data": ML_ROOT,
    "ml_models": ML_MODELS,
    "ml_predictions": ML_PREDICTIONS,
    "ml_datasets": ML_DATASETS,
    "ml_training": ML_TRAINING,

    # --- ML (Intraday DT) ---
    "ml_data_dt": MLDT_ROOT,
    "ml_dt_intraday": MLDT_INTRADAY,
    "ml_dt_intraday_models": MLDT_INTRADAY_MODELS,
    "ml_dt_intraday_predictions": MLDT_INTRADAY_PREDICTIONS,
    "ml_dt_intraday_datasets": MLDT_INTRADAY_DATASETS,
    "ml_dt_intraday_replay": MLDT_INTRADAY_REPLAY,
    "ml_dt_intraday_raw_days": MLDT_INTRADAY_RAW_DAYS,
    "ml_dt_intraday_replay_results": MLDT_INTRADAY_REPLAY_RESULTS,

    # --- Insights ---
    "insights": INSIGHTS_ROOT,
    "dashboard_cache": DASHBOARD_ROOT,

    # --- Analytics ---
    "analytics": ANALYTICS_ROOT,
    "analytics_pnl": ANALYTICS_PNL,
    "analytics_performance": ANALYTICS_PERFORMANCE,

    # --- Logs ---
    "logs": LOGS_ROOT,
    "backend_logs": LOGS_BACKEND,
    "nightly_logs": LOGS_NIGHTLY,
    "scheduler_logs": LOGS_SCHEDULER,
    "intraday_logs": LOGS_INTRADAY,

    # --- News ---
    "news_cache": NEWS_CACHE,
    "news_dashboard_json": NEWS_DASHBOARD_JSON,
    "sentiment_map": SENTIMENT_MAP,

    # --- Market / Regime ---
    "market_state": MARKET_STATE,
    "regime_state": REGIME_STATE,

    # --- Cloud (Supabase) ---
    "cloud_cache": CLOUD_CACHE,
    "updates": UPDATES_DIR,
}


# ============================================================
#  AUTO-CREATE DIRECTORIES
# ============================================================
def _create_dirs():
    for key, path in PATHS.items():
        if isinstance(path, Path) and path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)

_create_dirs()


# ============================================================
#  HELPER
# ============================================================
def get_path(key: str) -> Path:
    return PATHS.get(key)
