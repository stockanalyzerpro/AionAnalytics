# dt_backend/historical_replay/historical_replay_engine.py
"""
Full intraday replay engine:
    raw bars → context_dt → features_dt → predictions_dt → policy_dt → execution_dt
and then computes:
    - per-symbol PnL
    - gross PnL
    - hit rate
    - trades count
    - daily replay summary

Writes results to:
    ml_data_dt/intraday/replay/replay_results/<date>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log

# Phase 2 engines
from dt_backend.core.context_state_dt import build_intraday_context
from dt_backend.engines.feature_engineering import build_intraday_features

# Phase 1 model scoring
from dt_backend.ml.ai_model_intraday import score_intraday_batch, load_intraday_models

# Phase 3 logic
from dt_backend.core.policy_engine_dt import apply_intraday_policy
from dt_backend.core.execution_dt import run_execution_intraday


@dataclass
class ReplayResult:
    date: str
    n_symbols: int
    n_trades: int
    gross_pnl: float
    avg_pnl_per_trade: float
    hit_rate: float
    meta: Dict[str, Any]


def _paths_for_date(date_str: str) -> Tuple[Path, Path]:
    """Return (raw_day_path, replay_result_output_path)."""
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    raw_file = root / "intraday" / "replay" / "raw_days" / f"{date_str}.json"
    result_file = root / "intraday" / "replay" / "replay_results" / f"{date_str}.json"
    return raw_file, result_file


def _load_raw_day(date_str: str) -> List[Dict[str, Any]]:
    """Load the raw intraday bars for a given date."""
    raw_path, _ = _paths_for_date(date_str)
    if not raw_path.exists():
        log(f"[replay_engine] ⚠️ raw day missing at {raw_path}")
        return []

    try:
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            log(f"[replay_engine] ⚠️ malformed raw file {raw_path}")
            return []
        return data
    except Exception as e:
        log(f"[replay_engine] ⚠️ error reading raw file {raw_path}: {e}")
        return []


def _inject_bars(raw_day: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bootstrap rolling with today's bars. No previous features/context kept."""
    rolling: Dict[str, Any] = {}
    for entry in raw_day:
        sym = str(entry.get("symbol") or "").upper()
        if not sym:
            continue

        bars = entry.get("bars") or []
        node = ensure_symbol_node(rolling, sym)
        node["bars_intraday"] = bars
        rolling[sym] = node
    return rolling


def _extract_prices(bars: List[Dict[str, Any]]) -> List[float]:
    prices = []
    for b in bars:
        try:
            p = float(b.get("price") or b.get("c") or b.get("close"))
            prices.append(p)
        except Exception:
            continue
    return prices


def _symbol_pnl(node: Dict[str, Any]) -> Tuple[float, int, int]:
    """
    Simple PnL:
        BUY  → (end/start - 1)
        SELL → (start/end - 1)
    size multiplier = execution_dt.size
    """
    bars = node.get("bars_intraday") or []
    if len(bars) < 2:
        return 0.0, 0, 0

    prices = _extract_prices(bars)
    if len(prices) < 2:
        return 0.0, 0, 0

    start, end = prices[0], prices[-1]

    exec_dt = node.get("execution_dt") or {}
    side = str(exec_dt.get("side") or "").upper()
    size = float(exec_dt.get("size") or 0.0)

    if side == "BUY":
        ret = (end / start - 1.0) if start > 0 else 0.0
    elif side == "SELL":
        ret = (start / end - 1.0) if end > 0 else 0.0
    else:
        ret = 0.0

    pnl = size * ret
    trades = 1 if side in {"BUY", "SELL"} else 0
    hits = 1 if pnl > 0 else 0
    return pnl, trades, hits


def replay_intraday_day(date_str: str) -> ReplayResult | None:
    """Replay a full intraday session."""
    raw_day = _load_raw_day(date_str)
    if not raw_day:
        return None

    # 1) Inject bars → rolling
    rolling = _inject_bars(raw_day)
    save_rolling(rolling)

    # 2) Build context + features (Phase 2)
    build_intraday_context()
    build_intraday_features()

    # 3) Score with models (Phase 1)
    rolling = _read_rolling()
    rows = []
    index = []

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        feats = node.get("features_dt") or {}
        if not feats:
            continue
        rows.append(feats)
        index.append(sym)

    if not rows:
        log(f"[replay_engine] ⚠️ no features for day {date_str}")
        return None

    import pandas as pd
    X = pd.DataFrame(rows, index=index)

    models = load_intraday_models()
    proba_df, labels = score_intraday_batch(X, models=models)

    for sym in proba_df.index:
        node = rolling.get(sym) or {}
        node["predictions_dt"] = {
            "label": str(labels.loc[sym]),
            "proba": proba_df.loc[sym].to_dict(),
        }
        rolling[sym] = node
    save_rolling(rolling)

    # 4) Policy + Execution (Phase 3)
    apply_intraday_policy()
    run_execution_intraday()

    # 5) PnL aggregation
    rolling = _read_rolling()
    gross = 0.0
    n_trades = 0
    hits = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        pnl, t, h = _symbol_pnl(node)
        gross += pnl
        n_trades += t
        hits += h

    avg = gross / n_trades if n_trades > 0 else 0.0
    hr = hits / n_trades if n_trades > 0 else 0.0

    result = ReplayResult(
        date=date_str,
        n_symbols=len([s for s in rolling if not s.startswith("_")]),
        n_trades=n_trades,
        gross_pnl=gross,
        avg_pnl_per_trade=avg,
        hit_rate=hr,
        meta={},
    )

    # 6) Save output
    _, out_path = _paths_for_date(date_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    log(f"[replay_engine] ✅ {date_str} → PnL={gross:.4f}, trades={n_trades}, hit_rate={hr:.2f}")
    return result


def main() -> None:
    # Default: replay "today"
    today = datetime.now(timezone.utc).date()
    replay_intraday_day(today.isoformat())


if __name__ == "__main__":
    main()
