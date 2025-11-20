# backend/bots/strategy_2w.py
"""
Horizon-specific config for 2-week EOD swing bot.
"""

from __future__ import annotations

from backend.bots.base_swing_bot import SwingBotConfig

CONFIG = SwingBotConfig(
    horizon="2w",
    bot_key="eod_2w",
    max_positions=25,
    base_risk_pct=0.18,
    conf_threshold=0.55,
    stop_loss_pct=-0.07,
    take_profit_pct=0.14,
    max_weight_per_name=0.12,
    initial_cash=100.0,
)
