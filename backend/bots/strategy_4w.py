# backend/bots/strategy_4w.py
"""
Horizon-specific config for 4-week EOD swing bot.
"""

from __future__ import annotations

from backend.bots.base_swing_bot import SwingBotConfig

CONFIG = SwingBotConfig(
    horizon="4w",
    bot_key="eod_4w",
    max_positions=25,
    base_risk_pct=0.16,
    conf_threshold=0.55,
    stop_loss_pct=-0.09,
    take_profit_pct=0.20,
    max_weight_per_name=0.10,
    initial_cash=100.0,
)
