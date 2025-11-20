# backend/intraday_runner.py
"""
Intraday Runner — Backend-Orchestrated Version of dt_backend Intraday Cycle

This module performs ONE full intraday inference cycle:

    • Load existing rolling
    • Build intraday context
    • Build intraday features
    • Score intraday models (LGBM/LSTM/Transformer if present)
    • Apply policy engine
    • Apply execution engine
    • Return structured JSON summary

This is SAFE, backend-friendly, and designed to be called via API:

    POST /api/intraday/refresh

It does NOT:
    • Fetch bars
    • Modify universe
    • Train models
    • Alter any dt_backend config
    • Overwrite long-term historical data

It ONLY performs inference & writes updated policy/execution nodes.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

# --------------------------
# Try dt_backend logger first
# --------------------------
try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)

# --------------------------
# dt_backend engines
# --------------------------
from dt_backend.core.data_pipeline_dt import (
    _read_rolling,
    save_rolling,
    ensure_symbol_node,
)

from dt_backend.core.context_state_dt import build_intraday_context
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml.ai_model_intraday import (
    score_intraday_batch,
    load_intraday_models,
)

from dt_backend.core.policy_engine_dt import apply_intraday_policy
from dt_backend.core.execution_dt import run_execution_intraday

# Optional regime
try:
    from dt_backend.core.regime_detector_dt import classify_intraday_regime
except Exception:
    def classify_intraday_regime(_context):  # fallback
        return {"label": "neutral", "score": 0.0}


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def run_intraday_cycle() -> Dict[str, Any]:
    """
    Perform a full intraday inference cycle and return structured results.

    This is what the /api/intraday/refresh endpoint will call.
    """

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "errors": [],
        "updated_symbols": 0,
        "top_buys": [],
        "top_sells": [],
    }

    try:
        rolling = _read_rolling()
        if not rolling:
            summary["errors"].append("rolling_cache_empty")
            log("[intraday_runner] ❌ Rolling is empty. Cannot run intraday cycle.")
            return summary

        # ---------------------------------------------------
        # 1) CONTEXT
        # ---------------------------------------------------
        try:
            c = build_intraday_context()
            summary["steps"]["context"] = c
        except Exception as e:
            summary["errors"].append(f"context_error: {e}")
            log(f"[intraday_runner] ⚠️ context error: {e}")
            log(traceback.format_exc())

        # ---------------------------------------------------
        # 2) FEATURES
        # ---------------------------------------------------
        try:
            f = build_intraday_features()
            summary["steps"]["features"] = f
        except Exception as e:
            summary["errors"].append(f"feature_error: {e}")
            log(f"[intraday_runner] ⚠️ feature error: {e}")
            log(traceback.format_exc())

        # Refresh rolling after feature build
        rolling = _read_rolling()

        # ---------------------------------------------------
        # 3) MODEL SCORING
        # ---------------------------------------------------
        try:
            models = load_intraday_models()
            rows, index = [], []

            for sym, node in rolling.items():
                if sym.startswith("_"):
                    continue
                feats = node.get("features_dt") or {}
                if feats:
                    rows.append(feats)
                    index.append(sym)

            import pandas as pd
            df = pd.DataFrame(rows, index=index)

            proba_df, labels = score_intraday_batch(df, models=models)

            updated = 0
            for sym in proba_df.index:
                node = ensure_symbol_node(rolling, sym)
                node["predictions_dt"] = {
                    "label": str(labels.loc[sym]),
                    "proba": proba_df.loc[sym].to_dict(),
                }
                rolling[sym] = node
                updated += 1

            save_rolling(rolling)
            summary["steps"]["scoring"] = {"symbols_scored": updated}

        except Exception as e:
            summary["errors"].append(f"scoring_error: {e}")
            log(f"[intraday_runner] ⚠️ scoring error: {e}")
            log(traceback.format_exc())

        # ---------------------------------------------------
        # 4) POLICY ENGINE
        # ---------------------------------------------------
        try:
            p = apply_intraday_policy()
            summary["steps"]["policy"] = p
        except Exception as e:
            summary["errors"].append(f"policy_error: {e}")
            log(f"[intraday_runner] ⚠️ policy error: {e}")
            log(traceback.format_exc())

        # ---------------------------------------------------
        # 5) EXECUTION ENGINE
        # ---------------------------------------------------
        try:
            e = run_execution_intraday()
            summary["steps"]["execution"] = e
        except Exception as e:
            summary["errors"].append(f"execution_error: {e}")
            log(f"[intraday_runner] ⚠️ execution error: {e}")
            log(traceback.format_exc())

        # ---------------------------------------------------
        # 6) Final summary: Top signals
        # ---------------------------------------------------
        rolling = _read_rolling()
        buys = []
        sells = []

        for sym, node in rolling.items():
            if sym.startswith("_"):
                continue
            policy = node.get("policy_dt") or {}
            intent = str(policy.get("intent") or "").upper()
            conf = float(policy.get("confidence") or 0.0)
            if intent == "BUY":
                buys.append((conf, sym))
            elif intent == "SELL":
                sells.append((conf, sym))

        buys.sort(reverse=True, key=lambda x: x[0])
        sells.sort(reverse=True, key=lambda x: x[0])

        summary["top_buys"] = [{"symbol": s, "confidence": c} for c, s in buys[:25]]
        summary["top_sells"] = [{"symbol": s, "confidence": c} for c, s in sells[:25]]

        summary["updated_symbols"] = len(buys) + len(sells)

        log(f"[intraday_runner] ✅ Finished intraday cycle → updated={summary['updated_symbols']}")

        return summary

    except Exception as fatal:
        summary["errors"].append(f"fatal_error: {fatal}")
        log(f"[intraday_runner] ❌ fatal error: {fatal}")
        log(traceback.format_exc())
        return summary


# ==============================================================================
# CLI mode
# ==============================================================================
def main() -> None:
    out = run_intraday_cycle()
    print("\n===== INTRADAY RUNNER SUMMARY =====")
    print(out)


if __name__ == "__main__":
    main()
