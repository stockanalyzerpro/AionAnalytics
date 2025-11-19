# dt_backend/historical_replay/replay_harness.py
"""
Replay Harness v2 — fully integrated with Phase 3 Advanced Mode.

This harness:
    • Discovers all raw intraday days in: 
          ml_data_dt/intraday/replay/raw_days/
    • Runs the full replay engine (bars → ctx → feats → pred → policy → exec → PnL)
    • Builds deep-learning sequences for each symbol using the upgraded sequence_builder
    • Writes sequence parquet datasets:
          ml_data_dt/intraday/sequences/<tag>/<symbol>.parquet
    • Prints summary metrics (PnL, trades, hit rate)
    • Allows specifying:
          - date range
          - sequence tag
          - sequence length
          - horizons
          - normalization mode
          - stride

Usage examples:
    python -m dt_backend.historical_replay.replay_harness
    python -m dt_backend.historical_replay.replay_harness --tag baseline --seq_len 60
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import _read_rolling, log

from dt_backend.historical_replay.historical_replay_manager import _discover_dates
from dt_backend.historical_replay.historical_replay_engine import replay_intraday_day

# New advanced sequence builder
from dt_backend.historical_replay.sequence_builder import (
    build_sequences_for_symbol,
    write_sequence_dataset,
)


# ------------------------------------------------------------------------------
# Sequence Generation after Replay
# ------------------------------------------------------------------------------
def build_sequences_from_rolling(
    tag: str,
    seq_len: int = 60,
    horizons: List[int] = [1, 5, 10],
    norm: str = "zscore",
    stride: int = 1,
) -> Dict[str, Any]:
    """
    After replay runs, rolling holds:
        bars_intraday
        context_dt
        features_dt
        predictions_dt
        policy_dt
        execution_dt

    We only need bars_intraday to build deep-learning sequences.
    """
    rolling = _read_rolling()
    if not rolling:
        log("[replay_harness] ⚠️ rolling empty, skipping sequence-build")
        return {"symbols": 0, "wrote": 0}

    wrote = 0
    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        bars = node.get("bars_intraday") or []
        if not bars or len(bars) < seq_len + max(horizons):
            continue

        # Build sequences
        X_seq, y_seq, feature_list = build_sequences_for_symbol(
            bars=bars,
            seq_len=seq_len,
            horizons=horizons,
            norm=norm,
            stride=stride,
        )

        if X_seq.shape[0] == 0:
            continue

        # Write dataset
        write_sequence_dataset(
            symbol=sym,
            X=X_seq,
            y=y_seq,
            feature_list=feature_list,
            tag=tag,
        )

        wrote += 1

    log(f"[replay_harness] ✅ wrote sequence datasets for {wrote} symbols (tag={tag})")
    return {"symbols": len(rolling), "wrote": wrote}


# ------------------------------------------------------------------------------
# Full pipeline: replay days + generate sequences
# ------------------------------------------------------------------------------
def run_full_replay_and_sequences(
    start: str | None = None,
    end: str | None = None,
    tag: str = "default",
    seq_len: int = 60,
    horizons: List[int] = [1, 5, 10],
    norm: str = "zscore",
    stride: int = 1,
) -> None:

    dates = _discover_dates()
    if not dates:
        log("[replay_harness] ⚠️ no raw intraday days found.")
        return

    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        log("[replay_harness] ⚠️ no days in date range")
        return

    log(
        f"[replay_harness] Starting replay run: "
        f"{dates[0]} → {dates[-1]}, {len(dates)} days"
    )

    total_pnl = 0.0
    total_trades = 0
    total_hits = 0

    # Run replay for each day
    for date in dates:
        res = replay_intraday_day(date)
        if not res:
            continue

        total_pnl += res.gross_pnl
        total_trades += res.n_trades
        total_hits += res.hit_rate * max(1, res.n_trades)

        log(
            f"[replay_harness] Day={date}  PnL={res.gross_pnl:.5f} "
            f"trades={res.n_trades} hit={res.hit_rate:.3f}"
        )

        # After each day: build sequences for deep-learning training
        build_sequences_from_rolling(
            tag=tag,
            seq_len=seq_len,
            horizons=horizons,
            norm=norm,
            stride=stride,
        )

    # Summary
    if total_trades > 0:
        avg_pnl_per_trade = total_pnl / total_trades
        avg_hit = total_hits / total_trades
    else:
        avg_pnl_per_trade = 0.0
        avg_hit = 0.0

    log(
        f"[replay_harness] FINAL SUMMARY → PNL={total_pnl:.5f}, "
        f"trades={total_trades}, avg_pnl/trade={avg_pnl_per_trade:.5f}, "
        f"hit={avg_hit:.3f}"
    )


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AION Replay Harness v2")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--tag", type=str, default="default", help="Sequence dataset tag")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sequence slicing")
    parser.add_argument("--norm", type=str, default="zscore", help="Normalization: none|zscore|minmax|robust")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 10])

    args = parser.parse_args()

    run_full_replay_and_sequences(
        start=args.start,
        end=args.end,
        tag=args.tag,
        seq_len=args.seq_len,
        horizons=args.horizons,
        norm=args.norm,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
