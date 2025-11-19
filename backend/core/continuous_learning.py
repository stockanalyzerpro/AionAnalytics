"""
AION Analytics — Continuous Learning (Hybrid Human-like Mode)
-------------------------------------------------------------

Goal:
    Make the system *actually learn* from its own predictions and outcomes,
    adapt quickly to real regime shifts, but stay stable against noise.

Core ideas:
    • Maintain a per-symbol "brain" with recent prediction vs outcome history
    • Use two windows:
        - SHORT (fast learning, e.g. 30 samples)
        - LONG  (slow baseline, e.g. 120 samples)
    • Detect DRIFT when short-term performance < long-term baseline
    • Increase caution (drift_score) when model stops working
    • Decrease drift when model recovers
    • Optionally look at dt_backend (intraday) brain to confirm instability

Inputs (best-effort):
    • rolling.json.gz (latest)
        - node["context"]["pred_score"] (from context_state)
        - node["history"] (close prices)
    • rolling_brain.json.gz
    • regime_state.json (via regime_detector.detect_regime)
    • optional dt_backend rolling_brain_dt (if available)

Outputs:
    • rolling_brain.json.gz:
        brain[sym] = {
            "samples": [... recent performance samples ...],
            "hit_ratio_short": float,
            "hit_ratio_long": float,
            "mae_short": float,
            "mae_long": float,
            "drift_score": float,
            "last_sample_date": "YYYY-MM-DD",
            "last_update": ISO timestamp,
        }

    • This is read by context_state + policy_engine to tune confidence.
"""

from __future__ import annotations
import datetime
from typing import Dict, Any, List

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    save_brain,
    safe_float,
    log,
)
from backend.core.regime_detector import detect_regime

# Optional dt_backend import (for cross-checking drift)
try:
    from dt_backend.core.data_pipeline_dt import _read_brain as _read_dt_brain  # type: ignore
    DT_ENABLED = True
except Exception:
    DT_ENABLED = False

# Window sizes (in #samples, not days)
SHORT_WINDOW = 30   # fast-learning
LONG_WINDOW = 120   # slow baseline
MAX_SAMPLES = 240   # cap per symbol


# ============================================================
# Helper: Time
# ============================================================

def _today_str() -> str:
    return datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.datetime.now(TIMEZONE).isoformat()


# ============================================================
# Helper: Sliding window stats
# ============================================================

def _window_stats(samples: List[Dict[str, Any]], window: int) -> Dict[str, float]:
    """
    Compute hit-ratio and MAE over the most recent `window` samples.
    Each sample:
        {
          "date": "YYYY-MM-DD",
          "pred_score": float,
          "actual_ret": float,
          "hit": bool,
          "error": float
        }
    """
    if not samples:
        return {"hit_ratio": 0.5, "mae": 0.0, "n": 0}

    recent = samples[-window:] if len(samples) > window else samples
    n = len(recent)
    if n == 0:
        return {"hit_ratio": 0.5, "mae": 0.0, "n": 0}

    hits = sum(1 for s in recent if s.get("hit"))
    errors = [abs(float(s.get("error", 0.0))) for s in recent]

    hit_ratio = hits / n if n else 0.5
    mae = sum(errors) / n if n else 0.0
    return {"hit_ratio": hit_ratio, "mae": mae, "n": n}


# ============================================================
# Helper: Build performance sample from rolling node
# ============================================================

def _build_sample(sym: str, node: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Build a single performance sample for a symbol based on:
        - last two closes (to get today's realized return)
        - context["pred_score"] (directional strength)
    """
    hist = node.get("history", [])
    if not hist or len(hist) < 2:
        return None

    ctx = node.get("context") or {}
    pred_score = safe_float(ctx.get("pred_score"), 0.0)

    try:
        last = safe_float(hist[-1].get("close"))
        prev = safe_float(hist[-2].get("close"))
        if prev <= 0:
            return None
        actual_ret = (last - prev) / prev
    except Exception:
        return None

    # "Hit" if prediction direction matches realized return
    # pred_score > 0 ⇒ bullish; < 0 ⇒ bearish (if using that convention).
    # If your pred_score is always positive, this still behaves reasonably:
    # it measures whether "higher score" tended to accompany positive returns.
    hit = (pred_score >= 0 and actual_ret >= 0) or (pred_score < 0 and actual_ret < 0)

    # We need some error metric; we treat pred_score as a proxy for expected return.
    error = float(pred_score - actual_ret)

    sample = {
        "date": _today_str(),
        "pred_score": float(pred_score),
        "actual_ret": float(actual_ret),
        "hit": bool(hit),
        "error": float(error),
    }
    return sample


# ============================================================
# Drift computation (Hybrid Mode)
# ============================================================

def _compute_drift(short_stats: Dict[str, float],
                   long_stats: Dict[str, float],
                   regime_label: str,
                   dt_brain_sym: Dict[str, Any] | None = None) -> float:
    """
    Compute drift_score:
        • Positive when recent performance is worse than long-term baseline
        • Amplified in certain regimes
        • Optionally cross-checks intraday brain (dt_backend)
    """
    hr_s = short_stats.get("hit_ratio", 0.5)
    hr_l = long_stats.get("hit_ratio", 0.5)
    mae_s = short_stats.get("mae", 0.0)
    mae_l = long_stats.get("mae", 0.0)

    # baseline: if short window underperforms long window, that's drift
    drift = max(0.0, hr_l - hr_s)

    # penalize error blow-up
    if mae_l > 0 and mae_s > mae_l:
        drift += min(0.2, (mae_s - mae_l))

    # cross-check with intraday brain, if available
    if dt_brain_sym:
        dt_drift = float(dt_brain_sym.get("drift_score", 0.0) or 0.0)
        # blend, but don't let dt completely override
        drift = (drift * 0.7) + (dt_drift * 0.3)

    # regime shaping:
    #  - in bull/chop regimes, we expect good performance; bad recent -> more drift
    #  - in bear regimes, we allow a bit more "forgiveness" (harder environment)
    if regime_label == "bull":
        drift *= 1.3
    elif regime_label == "chop":
        drift *= 1.1
    elif regime_label == "bear":
        drift *= 0.9

    # clamp
    return float(max(0.0, min(1.0, drift)))


# ============================================================
# Update brain for a single symbol
# ============================================================

def _update_symbol_brain(sym: str,
                         node: Dict[str, Any],
                         brain_node: Dict[str, Any],
                         regime_label: str,
                         dt_brain_sym: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Update brain entry for one symbol:
        - append today's performance sample (if new)
        - recompute stats
        - update drift_score
    """
    samples: List[Dict[str, Any]] = brain_node.get("samples", [])
    last_sample_date = brain_node.get("last_sample_date")

    sample = _build_sample(sym, node)
    if not sample:
        # no new usable sample (no history, etc.)
        brain_node.setdefault("samples", samples)
        brain_node.setdefault("hit_ratio_short", 0.5)
        brain_node.setdefault("hit_ratio_long", 0.5)
        brain_node.setdefault("mae_short", 0.0)
        brain_node.setdefault("mae_long", 0.0)
        brain_node.setdefault("drift_score", 0.0)
        brain_node.setdefault("last_update", _now_iso())
        return brain_node

    # avoid double-counting: only one sample per date
    if last_sample_date == sample["date"]:
        # we've already recorded today's outcome; nothing to do
        return brain_node

    samples.append(sample)
    if len(samples) > MAX_SAMPLES:
        samples = samples[-MAX_SAMPLES:]

    # compute stats
    stats_short = _window_stats(samples, SHORT_WINDOW)
    stats_long = _window_stats(samples, LONG_WINDOW)

    # optional dt brain info for same symbol
    dt_sym = dt_brain_sym if dt_brain_sym else None

    drift_score = _compute_drift(stats_short, stats_long, regime_label, dt_sym)

    brain_node.update(
        samples=samples,
        hit_ratio_short=float(stats_short["hit_ratio"]),
        hit_ratio_long=float(stats_long["hit_ratio"]),
        mae_short=float(stats_short["mae"]),
        mae_long=float(stats_long["mae"]),
        drift_score=float(drift_score),
        last_sample_date=sample["date"],
        last_update=_now_iso(),
    )

    return brain_node


# ============================================================
# MAIN ENTRYPOINTS
# ============================================================

def run_continuous_learning() -> Dict[str, Any]:
    """
    Main entrypoint called from nightly_job *after* prices have updated.

    Steps:
        1) Load rolling + brain
        2) Get current regime
        3) (Optional) load dt_backend brain for cross-check
        4) For each symbol, append one performance sample
        5) Update window stats + drift_score
        6) Save rolling_brain.json.gz
    """
    rolling = _read_rolling()
    brain = _read_brain()

    if not rolling:
        log("⚠️ No rolling.json.gz found — continuous learning skipping.")
        return {"updated": 0, "note": "no_rolling"}

    regime = detect_regime(rolling)
    regime_label = regime.get("label", "unknown")

    # Optional intraday brain (dt_backend)
    dt_brain = {}
    if DT_ENABLED:
        try:
            dt_brain = _read_dt_brain() or {}
        except Exception:
            dt_brain = {}

    updated = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        brain_node = brain.get(sym, {})
        dt_sym = dt_brain.get(sym) if dt_brain else None

        new_brain_node = _update_symbol_brain(sym, node, brain_node, regime_label, dt_sym)
        brain[sym] = new_brain_node
        updated += 1

    save_brain(brain)
    log(f"🧠 Continuous learning updated {updated} symbols (regime={regime_label}).")

    return {
        "updated": updated,
        "regime": regime_label,
        "dt_enabled": DT_ENABLED,
    }


# Alias for backwards compatibility, if anything still calls continuous_learning.run()
def run() -> Dict[str, Any]:
    return run_continuous_learning()
