# backend/bots/strategy_1w.py
"""
Horizon-specific config for 1-week EOD swing bot.
"""

from __future__ import annotations

from backend.bots.base_swing_bot import SwingBotConfig

CONFIG = SwingBotConfig(
    horizon="1w",
    bot_key="eod_1w",
    max_positions=20,
    base_risk_pct=0.20,
    conf_threshold=0.55,
    stop_loss_pct=-0.05,
    take_profit_pct=0.10,
    max_weight_per_name=0.15,
    initial_cash=100.0,
)
