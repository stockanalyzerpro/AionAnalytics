"""
scheduler_config.py

Central schedule for AION Analytics.

Each entry in SCHEDULE is:
    {
        "name": "nightly_full",
        "time": "17:30",                          # HH:MM in TIMEZONE
        "module": "backend.jobs.nightly_job",     # used with python -m
        "args": ["--mode", "full"],               # optional
        "description": "Full nightly rebuild."
    }
"""

from __future__ import annotations

ENABLE   = True
TIMEZONE = "America/Denver"   # Converted to tz in scheduler_runner


SCHEDULE = [
    # ------------------------------------------------------------------
    # Nightly EOD Brain (Backend)
    # ------------------------------------------------------------------
    {
        "name": "nightly_full",
        "time": "17:30",
        "module": "backend.jobs.nightly_job",
        "args": [],
        "description": "Full nightly rebuild: backfill, features, train, predictions, context, regime, policy, learning, insights, supervisor.",
    },
    {
        "name": "evening_insights",
        "time": "18:00",
        "module": "backend.services.insights_builder",
        "args": [],
        "description": "Rebuild nightly insights/top-picks after nightly job.",
    },
    {
        "name": "social_sentiment_evening",
        "time": "20:30",
        "module": "backend.services.social_sentiment_fetcher",
        "args": [],
        "description": "Refresh social sentiment (Reddit/Twitter) once in the evening.",
    },

    # ------------------------------------------------------------------
    # Intraday DT engine (dt_backend) — premarket + hourly loop
    # ------------------------------------------------------------------
    {
        "name": "dt_premarket_full",
        "time": "06:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "Premarket full DT prep: bars, features, intraday dataset, models, signals.",
    },
    {
        "name": "dt_hourly_0930",
        "time": "09:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 9:30.",
    },
    {
        "name": "dt_hourly_1030",
        "time": "10:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 10:30.",
    },
    {
        "name": "dt_hourly_1130",
        "time": "11:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 11:30.",
    },
    {
        "name": "dt_hourly_1230",
        "time": "12:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 12:30.",
    },
    {
        "name": "dt_hourly_1330",
        "time": "13:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 13:30.",
    },
    {
        "name": "dt_hourly_1430",
        "time": "14:30",
        "module": "dt_backend.daytrading_job",
        "args": [],
        "description": "DT hourly refresh around 14:30.",
    },

    # ------------------------------------------------------------------
    # Nightly Swing Bots (1w / 2w / 4w) — FULL premarket pass
    # ------------------------------------------------------------------
       {
        "name": "eod_1w_full",
        "time": "06:00",
        "module": "backend.bots.runner_1w",
        "args": ["--mode", "full"],
        "description": "Premarket rebalance for 1w bot"
    },
    {
        "name": "eod_2w_full",
        "time": "06:00",
        "module": "backend.bots.runner_2w",
        "args": ["--mode", "full"],
        "description": "Premarket rebalance for 2w bot"
    },
    {
        "name": "eod_4w_full",
        "time": "06:00",
        "module": "backend.bots.runner_4w",
        "args": ["--mode", "full"],
        "description": "Premarket rebalance for 4w bot"
    },
    # ------------------------------------------------------------------
    # Hourly Swing Bot Loops (during market) — LOOP mode
    # C mode = BOTH full premarket + hourly loop intraday.
    # ------------------------------------------------------------------
        # 11:35 loop
    {
        "name": "bot_loop_1w_1135",
        "time": "11:35",
        "module": "backend.bots.runner_1w",
        "args": ["--mode", "loop"],
        "description": "1w bots hourly loop: 11:35.",
    },
    {
        "name": "bot_loop_2w_1135",
        "time": "11:35",
        "module": "backend.bots.runner_2w",
        "args": ["--mode", "loop"],
        "description": "2w bots hourly loop: 11:35.",
    },
    {
        "name": "bot_loop_4w_1135",
        "time": "11:35",
        "module": "backend.bots.runner_4w",
        "args": ["--mode", "loop"],
        "description": "4w bots hourly loop: 11:35.",
    },
    # Day Trading Bot — start at open (paper mode)
    {
        "name": "dt_daytrader_start",
        "time": "07:28",  # a bit before open in MT; adjust if you want
        "module": "backend.trading_bot_simulator",
        "args": ["--mode", "paper"],
        "description": "Start intraday day-trading bot (Alpaca PAPER).",
    },
    
    # Day Trading Bot — stop at close
    {
        "name": "dt_daytrader_stop",
        "time": "14:00",
        "module": "backend.trading_bot_simulator",
        "args": ["--mode", "stop"],
        "description": "Gracefully stop day-trading bot at market close.",
    },
    ]
