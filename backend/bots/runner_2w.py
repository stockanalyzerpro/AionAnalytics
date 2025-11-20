# backend/bots/runner_2w.py
"""
CLI runner for 2-week EOD swing bot.

Usage:
    python -m backend.bots.runner_2w --mode full
    python -m backend.bots.runner_2w --mode loop
"""

from __future__ import annotations

import argparse

from backend.bots.base_swing_bot import SwingBot
from backend.bots.strategy_2w import CONFIG as BOT_CONFIG


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AION EOD 2w swing bot (AI-powered).")
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "loop"],
        help="full = premarket AI rebalance, loop = intraday risk checks",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bot = SwingBot(BOT_CONFIG)
    bot.run(args.mode)


if __name__ == "__main__":
    main()
