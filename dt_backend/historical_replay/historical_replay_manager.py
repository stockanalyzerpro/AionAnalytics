# dt_backend/historical_replay/historical_replay_manager.py
"""
Runs intraday replays across multiple days, aggregates performance,
and writes a replay_log.json entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import log
from .historical_replay_engine import replay_intraday_day, ReplayResult


@dataclass
class ReplaySummary:
    start_date: str
    end_date: str
    n_days: int
    total_gross_pnl: float
    avg_pnl_per_day: float
    avg_pnl_per_trade: float
    avg_hit_rate: float
    days: List[Dict[str, Any]]


def _raw_days_dir() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    return root / "intraday" / "replay" / "raw_days"


def _discover_dates() -> List[str]:
    d = _raw_days_dir()
    if not d.exists():
        return []

    return sorted([p.stem for p in d.glob("*.json")])


def run_replay_range(start: Optional[str] = None, end: Optional[str] = None) -> ReplaySummary | None:
    dates = _discover_dates()
    if not dates:
        log("[replay_manager] ⚠️ no raw days found.")
        return None

    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        log("[replay_manager] ⚠️ nothing in requested range.")
        return None

    results: List[ReplayResult] = []
    for ds in dates:
        r = replay_intraday_day(ds)
        if r:
            results.append(r)

    if not results:
        log("[replay_manager] ⚠️ no successful replays.")
        return None

    total_gross = sum(r.gross_pnl for r in results)
    total_trades = sum(r.n_trades for r in results)
    total_hit_weighted = sum(r.hit_rate * max(1, r.n_trades) for r in results)

    n_days = len(results)
    avg_pnl_day = total_gross / n_days if n_days > 0 else 0.0
    avg_pnl_trade = total_gross / total_trades if total_trades > 0 else 0.0
    avg_hit = total_hit_weighted / total_trades if total_trades > 0 else 0.0

    summary = ReplaySummary(
        start_date=dates[0],
        end_date=dates[-1],
        n_days=n_days,
        total_gross_pnl=total_gross,
        avg_pnl_per_day=avg_pnl_day,
        avg_pnl_per_trade=avg_pnl_trade,
        avg_hit_rate=avg_hit,
        days=[asdict(r) for r in results],
    )

    # Append to replay_log.json
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    log_path = root / "intraday" / "replay" / "replay_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if log_path.exists():
            existing = json.load(log_path.open("r", encoding="utf-8"))
        else:
            existing = {"meta": {}, "days": []}
    except Exception:
        existing = {"meta": {}, "days": []}

    existing["days"].append(asdict(summary))
    existing["meta"]["ts_last_run"] = datetime.now(timezone.utc).isoformat()

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    log(
        f"[replay_manager] ✅ replayed {summary.n_days} days "
        f"({summary.start_date}→{summary.end_date}), "
        f"P={total_gross:.4f}, avg/day={avg_pnl_day:.4f}"
    )

    return summary


def main() -> None:
    run_replay_range()


if __name__ == "__main__":
    main()
