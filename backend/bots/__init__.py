# backend/bots/__init__.py

"""
AION Analytics — Backend Bots Package

Contains:
    • Shared EOD swing bot engine (AI-powered)
    • Horizon-specific strategy configs (1w / 2w / 4w)
    • Small CLI runners for scheduler integration
"""

from .base_swing_bot import SwingBot, SwingBotConfig

__all__ = [
    "SwingBot",
    "SwingBotConfig",
]
