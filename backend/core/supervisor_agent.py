# backend/supervisor_agent.py — v3.0
from __future__ import annotations

from typing import Dict, Any, List
import json
import datetime
import math
from pathlib import Path
from statistics import mean

from .config import PATHS, TIMEZONE
from .data_pipeline import log, _read_rolling, _read_brain, safe_float

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ML_DATA_ROOT: Path = PATHS.get("ml_data", Path("ml_data"))
DATASET_DIR: Path = ML_DATA_ROOT / "nightly" / "dataset"
DATASET_FILE: Path = DATASET_DIR / "training_data_daily.parquet"
FEATURE_LIST_FILE: Path = DATASET_DIR / "feature_list_daily.json"

MODEL_ROOT: Path = PATHS.get("ml_models", ML_DATA_ROOT / "nightly" / "models")

INSIGHTS_DIR: Path = PATHS.get("insights", PATHS.get("root", Path(".")) / "insights")
MACRO_STATE_FILE: Path = PATHS.get("macro_state", ML_DATA_ROOT / "macro_state.json")
NEWS_INTEL_FILE: Path = PATHS.get("news_intel", PATHS.get("analytics", Path("analytics")) / "news_intel.json")
SOCIAL_INTEL_FILE: Path = PATHS.get("social_intel", PATHS.get("analytics", Path("analytics")) / "social_intel.json")

OVERRIDES_PATH: Path = PATHS["ml_data"] / "supervisor_overrides.json"

# Nightly horizons (must match ai_model / policy_engine)
HORIZONS: List[str] = ["1d", "3d", "1w", "2w", "4w", "13w", "26w", "52w"]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _file_age_minutes(path: Path) -> float:
    """Age of file in minutes; large value if missing."""
    try:
        if not path.exists():
            return 9999.0
        m = path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(m, tz=TIMEZONE)
        diff = datetime.datetime.now(TIMEZONE) - dt
        return diff.total_seconds() / 60.0
    except Exception:
        return 9999.0


def _status_from_age(age_min: float, warn: float, crit: float) -> str:
    if age_min >= crit:
        return "critical"
    if age_min >= warn:
        return "warning"
    return "ok"


def _save_overrides(js: dict) -> None:
    try:
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2)
    except Exception as e:
        log(f"[supervisor_agent] ⚠️ Failed to save overrides: {e}")


def load_overrides() -> Dict[str, Any]:
    """Load the last written supervisor overrides (if any)."""
    try:
        if not OVERRIDES_PATH.exists():
            return {}
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Core health checks
# ---------------------------------------------------------------------

def _check_dataset_health() -> Dict[str, Any]:
    age = _file_age_minutes(DATASET_FILE)
    feat_age = _file_age_minutes(FEATURE_LIST_FILE)

    status = "ok"
    if age >= 1440 or feat_age >= 1440:  # older than 1 day
        status = "critical"
    elif age >= 720 or feat_age >= 720:
        status = "warning"

    return {
        "status": status,
        "dataset_age_min": age,
        "feature_list_age_min": feat_age,
        "dataset_path": str(DATASET_FILE),
        "feature_list_path": str(FEATURE_LIST_FILE),
    }


def _check_model_health() -> Dict[str, Any]:
    """Check that each horizon has a model and how old they are."""
    horizon_info = {}
    ages = []

    for h in HORIZONS:
        mp = MODEL_ROOT / f"model_{h}.pkl"
        age = _file_age_minutes(mp)
        exists = mp.exists()
        ages.append(age if exists else 9999.0)

        horizon_info[h] = {
            "exists": exists,
            "age_min": age,
            "path": str(mp),
        }

    # Status: if any missing or older than 7 days = warning/critical
    missing = [h for h, info in horizon_info.items() if not info["exists"]]
    oldest_age = max(ages) if ages else 9999.0

    if missing or oldest_age >= 10080:  # 7 days
        status = "critical"
    elif oldest_age >= 4320:  # 3 days
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "missing_horizons": missing,
        "oldest_model_age_min": oldest_age,
        "horizons": horizon_info,
    }


def _check_intel_files() -> Dict[str, Any]:
    macro_age = _file_age_minutes(MACRO_STATE_FILE)
    news_age = _file_age_minutes(NEWS_INTEL_FILE)
    soc_age = _file_age_minutes(SOCIAL_INTEL_FILE)

    macro_status = _status_from_age(macro_age, warn=720, crit=1440)
    news_status = _status_from_age(news_age, warn=240, crit=720)
    soc_status = _status_from_age(soc_age, warn=240, crit=720)

    return {
        "macro": {
            "status": macro_status,
            "age_min": macro_age,
            "path": str(MACRO_STATE_FILE),
        },
        "news_intel": {
            "status": news_status,
            "age_min": news_age,
            "path": str(NEWS_INTEL_FILE),
        },
        "social_intel": {
            "status": soc_status,
            "age_min": soc_age,
            "path": str(SOCIAL_INTEL_FILE),
        },
    }


def _check_insights_health() -> Dict[str, Any]:
    """Look at a few canonical insights files."""
    files = [
        "top50_1w.json",
        "top50_2w.json",
        "top50_4w.json",
        "top50_52w.json",
        "top50_social_heat.json",
        "top50_news_novelty.json",
    ]
    info = {}
    ages = []

    for name in files:
        p = INSIGHTS_DIR / name
        age = _file_age_minutes(p)
        ages.append(age if p.exists() else 9999.0)
        info[name] = {
            "exists": p.exists(),
            "age_min": age,
            "path": str(p),
        }

    oldest_age = max(ages) if ages else 9999.0
    if oldest_age >= 10080:  # 7 days
        status = "critical"
    elif oldest_age >= 1440:  # 1 day
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "oldest_insight_age_min": oldest_age,
        "files": info,
    }


def _check_drift_health() -> Dict[str, Any]:
    """Summarize drift from the brain snapshot, if present."""
    try:
        brain = _read_brain() or {}
    except Exception:
        brain = {}

    if not brain:
        return {
            "status": "warning",
            "note": "No drift brain found.",
            "avg_drift": None,
        }

    drifts = []
    for sym, node in brain.items():
        if not isinstance(node, dict):
            continue
        d = safe_float(node.get("drift_score", 0.0))
        if d != 0.0:
            drifts.append(d)

    if not drifts:
        return {
            "status": "ok",
            "avg_drift": 0.0,
            "max_drift": 0.0,
            "count": len(brain),
        }

    avg_d = float(mean(drifts))
    max_d = float(max(drifts))

    if max_d > 0.15:
        status = "critical"
    elif max_d > 0.08:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "avg_drift": avg_d,
        "max_drift": max_d,
        "count": len(brain),
    }


def _check_rolling_coverage() -> Dict[str, Any]:
    """Basic sanity: rolling present, predictions context news social filled."""
    rolling = _read_rolling() or {}
    if not rolling:
        return {
            "status": "critical",
            "symbols": 0,
            "missing_predictions": 0,
            "missing_context": 0,
            "missing_news": 0,
            "missing_social": 0,
        }

    total = 0
    missing_preds = 0
    missing_ctx = 0
    missing_news = 0
    missing_social = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        total += 1
        if not node.get("predictions"):
            missing_preds += 1
        if not node.get("context"):
            missing_ctx += 1
        if not node.get("news"):
            missing_news += 1
        if not node.get("social"):
            missing_social += 1

    ratio_missing_preds = missing_preds / max(total, 1)
    if ratio_missing_preds > 0.5:
        status = "critical"
    elif ratio_missing_preds > 0.1:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "symbols": total,
        "missing_predictions": missing_preds,
        "missing_context": missing_ctx,
        "missing_news": missing_news,
        "missing_social": missing_social,
    }


# ---------------------------------------------------------------------
# Overrides (kill switch / exposure caps)
# ---------------------------------------------------------------------

def compute_overrides_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take PnL / regime metrics and compute:
        - kill_switch (bool)
        - conf_min (float)
        - exposure_cap (float)

    metrics is expected to contain:
        drawdown_7d: float
        regime: str (e.g. "bull", "bear", "panic")
        regime_conf: float [0,1]
    """
    dd = float(metrics.get("drawdown_7d", 0.0) or 0.0)
    regime = (metrics.get("regime") or "neutral").lower()
    rconf = float(metrics.get("regime_conf", 0.0) or 0.0)

    # Base defaults for a calm regime
    overrides: Dict[str, Any]

    # Hard DD guardrails
    if dd <= -0.10:
        # severe drawdown
        overrides = {"kill_switch": True, "conf_min": 0.65, "exposure_cap": 0.3}
    elif dd <= -0.05:
        # moderate drawdown
        overrides = {"kill_switch": True, "conf_min": 0.6, "exposure_cap": 0.5}
    elif regime == "panic" and rconf > 0.7:
        # panic regime, models still running but under full tension
        overrides = {"kill_switch": False, "conf_min": 0.6, "exposure_cap": 0.6}
    elif regime == "bear" and rconf > 0.6:
        overrides = {"kill_switch": False, "conf_min": 0.56, "exposure_cap": 0.7}
    else:
        # normal / bull / chop
        overrides = {"kill_switch": False, "conf_min": 0.52, "exposure_cap": 1.2}

    _save_overrides(overrides)
    log(f"[supervisor_agent] ✅ overrides updated → {overrides}")
    return overrides


# Backwards-compatible alias
def update_overrides(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return compute_overrides_from_metrics(metrics)


# ---------------------------------------------------------------------
# Main supervisor verdict
# ---------------------------------------------------------------------

def supervisor_verdict() -> Dict[str, Any]:
    """
    High-level system verdict for dashboards / system_status_router:

        {
          "status": "ok" | "warning" | "critical",
          "components": {...},
          "overrides": {...}
        }
    """
    log("[supervisor_agent] 🔍 Evaluating system health...")

    dataset = _check_dataset_health()
    models = _check_model_health()
    intel = _check_intel_files()
    insights = _check_insights_health()
    drift = _check_drift_health()
    rolling_cov = _check_rolling_coverage()
    overrides = load_overrides()

    statuses = [
        dataset["status"],
        models["status"],
        intel["macro"]["status"],
        intel["news_intel"]["status"],
        intel["social_intel"]["status"],
        insights["status"],
        drift["status"],
        rolling_cov["status"],
    ]

    if "critical" in statuses:
        overall = "critical"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    verdict = {
        "status": overall,
        "components": {
            "dataset": dataset,
            "models": models,
            "intel": intel,
            "insights": insights,
            "drift": drift,
            "rolling_coverage": rolling_cov,
        },
        "overrides": overrides,
        "generated_at": datetime.datetime.now(TIMEZONE).isoformat(),
    }

    log(f"[supervisor_agent] 🧭 Supervisor verdict: {overall}")
    return verdict


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    v = supervisor_verdict()
    print(json.dumps(v, indent=2))
