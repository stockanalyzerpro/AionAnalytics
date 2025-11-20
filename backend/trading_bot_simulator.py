# backend/trading_bot_simulator.py
# Day Trading Bot v3.0 — Intraday multi-bot engine with risk control, PnL, and optional Alpaca paper trading.
#
# - Reads 1-min bars from: data_dt/rolling_intraday.json.gz
# - Reads AI signals (best-effort) from: ml_data_dt/signals/
# - Simulates 5 bots: momentum, mean-revert, signal-follow, breakout, hybrid
# - Enforces stop-loss / take-profit per position
# - Logs daily trades and updates ml_data_dt/sim_summary.json
# - Can:
#       • run in pure SIM mode  (no broker)
#       • run in PAPER mode     (Alpaca paper account)
#       • expose a STOP mode    (used by scheduler to stop loop)
#
# CLI:
#   python -m backend.trading_bot_simulator --mode sim
#   python -m backend.trading_bot_simulator --mode paper
#   python -m backend.trading_bot_simulator --mode stop     # sets stop flag (scheduler)
#
# Scheduler pattern:
#   - Start once near market open:
#       module="backend.trading_bot_simulator", args=["--mode", "paper"]
#   - Stop at market close:
#       module="backend.trading_bot_simulator", args=["--mode", "stop"]

from __future__ import annotations

import os
import json
import gzip
import time
import glob
import threading
import argparse
from datetime import datetime, timezone, time as dtime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Logging + PATHS / DT_PATHS
# ---------------------------------------------------------------------

try:
    # unified backend logger
    from backend.core.data_pipeline import log  # type: ignore
except Exception:
    try:
        from backend.data_pipeline import log  # type: ignore
    except Exception:
        def log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

try:
    from dt_backend.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS = {}  # type: ignore

try:
    from backend.core.config import PATHS, TIMEZONE  # type: ignore
except Exception:
    try:
        from backend.config import PATHS, TIMEZONE  # type: ignore
    except Exception:
        from pathlib import Path
        PATHS = {
            "root": Path("."),
            "ml_data_dt": Path("ml_data_dt"),
            "logs": Path("logs"),
        }
        TIMEZONE = timezone.utc  # type: ignore

# ---------------------------------------------------------------------
# Safe symbol extractor
# ---------------------------------------------------------------------


def _sym_from_any(val) -> str:
    if isinstance(val, str):
        return val.strip().upper()
    if isinstance(val, dict):
        sym = (
            val.get("symbol")
            or val.get("ticker")
            or val.get("name")
            or ""
        )
        return str(sym).strip().upper()
    return ""


# ----------------------------- CONFIG --------------------------------

# Risk & money management (tune freely)
START_CASH = 100.0
STOP_LOSS_PCT = -0.02  # -2%
TAKE_PROFIT_PCT = 0.04  # +4%
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.20  # 20% of equity per trade
LOOP_SECONDS = 60  # main loop interval for bar-driven checks

# Paths (DT_PATHS-aware)
from pathlib import Path

ROOT = PATHS.get("root", Path("."))
BARS_PATH = str(DT_PATHS.get("rolling_intraday", ROOT / "data_dt" / "rolling_intraday.json.gz"))
SIGNALS_DIR = str(DT_PATHS.get("signals_dir", ROOT / "ml_data_dt" / "signals"))
ML_DATA_DT_DIR = PATHS.get("ml_data_dt", ROOT / "ml_data_dt")

SIM_LOG_DIR = str(ML_DATA_DT_DIR / "sim_logs")
SIM_SUMMARY_FILE = str(ML_DATA_DT_DIR / "sim_summary.json")

# Universe cap (optional — to keep sim light)
MAX_UNIVERSE = 2000  # consider only the first N symbols that have bars

# Stop flag (for scheduler stop job)
STOP_FLAG_FILE = str(PATHS.get("logs", ROOT / "logs") / "bots" / "daytrader_stop.flag")

# Execution mode + Alpaca config
EXECUTION_MODE: str = "sim"  # "sim", "paper", "live"
_ALPACA = None  # lazily initialized REST client
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
ALPACA_PAPER_URL = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
ALPACA_LIVE_URL = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")

# ----------------------------------------------------------------------
# Time / misc helpers
# ----------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_dirs() -> None:
    os.makedirs(SIM_LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SIM_SUMMARY_FILE), exist_ok=True)
    # ensure stop-flag parent
    Path(STOP_FLAG_FILE).parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Market-hours & stop-flag helpers
# ----------------------------------------------------------------------


def is_market_open_now() -> bool:
    """
    Simple US equities session check. Uses TIMEZONE from backend config.
    Treats Monday–Friday, 09:30–16:00 (exchange time) as 'open'.
    """
    try:
        tz = TIMEZONE  # backend config may already be a tzinfo
    except Exception:
        tz = timezone.utc

    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False

    # You are in MT; US equities open at 7:30 MT, close at 14:00 MT.
    # We'll use those local times here.
    open_t = dtime(hour=7, minute=30)
    close_t = dtime(hour=14, minute=0)
    t = now.time()
    return open_t <= t <= close_t


def set_stop_flag() -> None:
    """Used by scheduler stop job."""
    _ensure_dirs()
    Path(STOP_FLAG_FILE).write_text(_now_utc_iso(), encoding="utf-8")
    log(f"[SimTrader] 🛑 Stop flag set at {STOP_FLAG_FILE}")


def clear_stop_flag() -> None:
    try:
        p = Path(STOP_FLAG_FILE)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def stop_flag_set() -> bool:
    try:
        return Path(STOP_FLAG_FILE).exists()
    except Exception:
        return False


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def load_latest_bars(path: str = BARS_PATH) -> Dict[str, dict]:
    """
    Loads your rolling intraday bars and returns a dict:
    { "AAPL": {"price": 188.05, "volume": 412562, "time": "...",
               "closes": [...], "highs": [...], "lows": [...]}, ... }
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to load intraday bars: {e}")
        return {}

    bars = js.get("bars", {}) or {}
    out: Dict[str, dict] = {}
    for sym, arr in bars.items():
        if not arr:
            continue
        last = arr[-1]
        closes = [float(x.get("c", 0.0)) for x in arr[-60:]]
        highs = [float(x.get("h", 0.0)) for x in arr[-60:]]
        lows = [float(x.get("l", 0.0)) for x in arr[-60:]]
        out_sym = _sym_from_any(sym)
        if not out_sym:
            continue
        out[out_sym] = {
            "price": float(last.get("c", 0.0)),
            "volume": int(last.get("v", 0)),
            "time": last.get("t"),
            "closes": closes,
            "highs": highs,
            "lows": lows,
        }
    # Limit universe
    if len(out) > MAX_UNIVERSE:
        out = dict(list(out.items())[:MAX_UNIVERSE])
    return out


def load_latest_signals(signals_dir: str = SIGNALS_DIR) -> Dict[str, dict]:
    """
    Load latest AI intraday signals from ml_data_dt/signals (if present).
    Expected structure (best-effort):
        { "AAPL": {"signal": "BUY", "score": 0.87, "confidence": 0.92, ...}, ... }
    """
    try:
        files = sorted(
            glob.glob(os.path.join(signals_dir, "*.json")),
            key=os.path.getmtime,
        )
    except Exception:
        files = []

    if not files:
        return {}

    latest = files[-1]
    try:
        with open(latest, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to load intraday signals: {e}")
        return {}
    if isinstance(js, dict):
        return js.get("signals", js) or {}
    return {}


# ----------------------------------------------------------------------
# Simple indicators
# ----------------------------------------------------------------------


def _pct_chg(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return (a - b) / b
    except Exception:
        return None


def _sma(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n or n <= 0:
        return None
    return sum(vals[-n:]) / float(n)


def _ema(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n or n <= 0:
        return None
    k = 2 / (n + 1.0)
    ema = vals[-n]
    for v in vals[-n + 1:]:
        ema = v * k + ema * (1 - k)
    return ema


# ----------------------------------------------------------------------
# Execution engine (Algo Mode: sim / paper / live)
# ----------------------------------------------------------------------


def _init_alpaca_if_needed(mode: str) -> None:
    global _ALPACA, EXECUTION_MODE
    EXECUTION_MODE = mode

    if mode not in ("paper", "live"):
        _ALPACA = None
        log(f"[SimTrader] 🧪 Execution mode = {mode.upper()} (no broker).")
        return

    key = ALPACA_API_KEY
    secret = ALPACA_SECRET_KEY
    if not key or not secret:
        log("⚠️ Alpaca API keys not set in env — falling back to SIM mode.")
        EXECUTION_MODE = "sim"
        _ALPACA = None
        return

    base_url = ALPACA_PAPER_URL
    if mode == "live":
        if os.getenv("AION_ALLOW_LIVE", "0") != "1":
            log("⚠️ LIVE trading requested but AION_ALLOW_LIVE!=1 — using PAPER instead.")
        else:
            base_url = ALPACA_LIVE_URL

    try:
        import alpaca_trade_api as tradeapi  # type: ignore
    except Exception as e:
        log(f"⚠️ Could not import alpaca_trade_api ({e}) — falling back to SIM mode.")
        EXECUTION_MODE = "sim"
        _ALPACA = None
        return

    try:
        _ALPACA = tradeapi.REST(
            key_id=key,
            secret_key=secret,
            base_url=base_url,
            api_version="v2",
        )
        log(f"[SimTrader] ☁️ Alpaca client initialized ({mode.upper()} @ {base_url}).")
    except Exception as e:
        log(f"⚠️ Failed to initialize Alpaca client: {e} — staying in SIM mode.")
        EXECUTION_MODE = "sim"
        _ALPACA = None


def _algo_send_order(
    side: str,
    sym: str,
    qty: float,
    price: float,
    bar: dict,
    reason: str,
) -> None:
    """
    Algo-style order selection:
      - High liquidity / stable → MARKET
      - Lower liquidity / choppy → LIMIT at last price
    (Paper/LIVE modes only; SIM just logs.)
    """
    if EXECUTION_MODE not in ("paper", "live") or _ALPACA is None:
        # Pure simulation; nothing to send.
        return

    # Algo heuristics
    volume = float(bar.get("volume") or 0.0)
    closes = bar.get("closes") or []
    if closes:
        hi = max(closes[-5:])
        lo = min(closes[-5:])
        rng = hi - lo
    else:
        rng = 0.0

    vol_ok = volume >= 500_000
    vol_pct = (rng / price) if price > 0 and rng > 0 else 0.0
    stable = vol_pct < 0.004  # <0.4% recent 5m range

    order_type = "market" if (vol_ok and stable) else "limit"

    try:
        if order_type == "market":
            _ALPACA.submit_order(
                symbol=sym,
                qty=qty,
                side=side.lower(),
                type="market",
                time_in_force="day",
            )
            log(f"[Alpaca] {side.upper()} {qty} {sym} @ MARKET ({reason})")
        else:
            _ALPACA.submit_order(
                symbol=sym,
                qty=qty,
                side=side.lower(),
                type="limit",
                limit_price=round(price, 4),
                time_in_force="day",
            )
            log(f"[Alpaca] {side.upper()} {qty} {sym} @ LIMIT {price:.4f} ({reason})")
    except Exception as e:
        log(f"⚠️ Alpaca order failed for {side} {sym} ({reason}): {e}")


# ----------------------------------------------------------------------
# Bot core
# ----------------------------------------------------------------------


class BaseSimBot:
    def __init__(self, name: str):
        self.name = name
        self.cash = START_CASH
        self.positions: Dict[str, dict] = {}  # {sym: {"qty": float, "entry": float, "stop": float, "target": float}}
        self.trades: List[dict] = []
        self.daily_equity: List[Tuple[str, float]] = []  # (iso_ts, equity)
        self.max_positions = MAX_POSITIONS
        self._last_synced_trade_idx: int = 0  # for paper-sync

    # ------- portfolio helpers -------
    def equity(self, price_map: Dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            px = price_map.get(sym)
            if isinstance(px, (int, float)):
                value += pos["qty"] * px
        return value

    def exposure(self, price_map: Dict[str, float]) -> float:
        expo = 0.0
        for sym, pos in self.positions.items():
            px = price_map.get(sym)
            if isinstance(px, (int, float)):
                expo += pos["qty"] * px
        return expo

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions

    def _enter(self, sym: str, price: float, reason: str, bar: Optional[dict] = None):
        if not self.can_open():
            return
        # simple sizing by percentage of equity
        price_map = {sym: price}
        eq = self.equity(price_map)
        risk_capital = eq * POSITION_SIZE_PCT
        if risk_capital <= 0 or price <= 0:
            return
        qty = max(1.0, risk_capital / price)
        stop = price * (1.0 + STOP_LOSS_PCT)
        target = price * (1.0 + TAKE_PROFIT_PCT)

        self.cash -= qty * price
        self.positions[sym] = {"qty": qty, "entry": price, "stop": stop, "target": target}
        trade = {
            "t": _now_utc_iso(),
            "ticker": sym,
            "action": "BUY",
            "qty": qty,
            "price": price,
            "reason": reason,
        }
        self.trades.append(trade)
        log(f"[{self.name}] BUY {qty:.4f} {sym} @ {price:.4f} ({reason})")

        # Optional external execution
        if bar is not None:
            _algo_send_order("buy", sym, qty, price, bar, reason)

    def _exit(self, sym: str, price: float, reason: str, bar: Optional[dict] = None):
        pos = self.positions.get(sym)
        if not pos:
            return
        proceeds = pos["qty"] * price
        self.cash += proceeds
        pnl = (price - pos["entry"]) * pos["qty"]
        trade = {
            "t": _now_utc_iso(),
            "ticker": sym,
            "action": "SELL",
            "qty": pos["qty"],
            "price": price,
            "reason": reason,
            "pnl": round(pnl, 6),
        }
        self.trades.append(trade)
        log(f"[{self.name}] SELL {pos['qty']:.4f} {sym} @ {price:.4f} ({reason}) PnL={pnl:.2f}")
        del self.positions[sym]

        if bar is not None:
            _algo_send_order("sell", sym, pos["qty"], price, bar, reason)

    # ------- risk checks -------
    def risk_checks(self, sym: str, price: float, bar: Optional[dict] = None):
        pos = self.positions.get(sym)
        if not pos:
            return
        if price <= pos["stop"]:
            self._exit(sym, price, "STOP_LOSS", bar)
        elif price >= pos["target"]:
            self._exit(sym, price, "TAKE_PROFIT", bar)

    # ------- strategy hooks (override) -------
    def maybe_enter(self, sym: str, bar: dict, signals: dict):
        """Decide opens (override in subclass)."""
        pass

    def maybe_exit(self, sym: str, bar: dict, signals: dict):
        """Discretionary exits beyond SL/TP (override in subclass)."""
        pass

    # main step (per bot per loop)
    def step(self, market: Dict[str, dict], signals: Dict[str, dict]):
        price_map = {s: d.get("price") for s, d in market.items() if isinstance(d.get("price"), (int, float))}
        # exits + risk
        for sym in list(self.positions.keys()):
            bar = market.get(sym) or {}
            px = price_map.get(sym)
            if not isinstance(px, (int, float)):
                continue
            self.risk_checks(sym, px, bar)
            self.maybe_exit(sym, bar, signals)

        # entries
        for sym, d in market.items():
            if sym in self.positions:
                continue
            if not self.can_open():
                break
            self.maybe_enter(sym, d, signals)

    # sync new trades to external broker (paper/live)
    def sync_trades_to_broker(self, market: Dict[str, dict]):
        # We already sent orders at _enter/_exit, so this is mostly a hook if needed.
        # Currently just advances last_synced index to mark trades as processed.
        self._last_synced_trade_idx = len(self.trades)


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------


class MomentumBot(BaseSimBot):
    def __init__(self):
        super().__init__("momentum_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20:
            return
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        price = float(closes[-1])
        if ema5 is None or ema20 is None:
            return
        # bullish momentum + above ema20
        if ema5 > ema20 and price > ema20:
            self._enter(sym, price, "EMA5>EMA20", bar)

    def maybe_exit(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20 or sym not in self.positions:
            return
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        price = float(closes[-1])
        if ema5 is None or ema20 is None:
            return
        # momentum fade
        if ema5 < ema20:
            self._exit(sym, price, "MOMENTUM_FADE", bar)


class MeanRevertBot(BaseSimBot):
    def __init__(self):
        super().__init__("mean_revert_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 30:
            return
        ma20 = _sma(closes, 20)
        price = float(closes[-1])
        if ma20 is None or ma20 == 0:
            return
        drop = _pct_chg(price, ma20)
        # buy ~1–2% below mean
        if isinstance(drop, float) and drop <= -0.012:
            self._enter(sym, price, "MEAN_REVERT_BUY", bar)

    def maybe_exit(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20 or sym not in self.positions:
            return
        ma20 = _sma(closes, 20)
        price = float(closes[-1])
        if ma20 is None or ma20 == 0:
            return
        # exit when back near/above mean
        diff = _pct_chg(price, ma20)
        if isinstance(diff, float) and diff >= 0.0:
            self._exit(sym, price, "MEAN_REVERT_EXIT", bar)


class SignalFollowBot(BaseSimBot):
    def __init__(self):
        super().__init__("signal_follow_bot")

    def maybe_enter(self, sym, bar, signals):
        info = signals.get(sym) or {}
        sig = str(info.get("signal", "")).upper()
        conf = float(info.get("confidence") or 0.0)
        price = float(bar.get("price") or 0.0)
        if sig == "BUY" and conf >= 0.55 and price > 0:
            self._enter(sym, price, f"AI_BUY_{conf:.2f}", bar)

    def maybe_exit(self, sym, bar, signals):
        if sym not in self.positions:
            return
        info = signals.get(sym) or {}
        sig = str(info.get("signal", "")).upper()
        price = float(bar.get("price") or 0.0)
        if sig == "SELL" and price > 0:
            self._exit(sym, price, "AI_SELL", bar)


class BreakoutBot(BaseSimBot):
    def __init__(self):
        super().__init__("breakout_bot")

    def maybe_enter(self, sym, bar, signals):
        highs = bar.get("highs") or []
        closes = bar.get("closes") or []
        if len(highs) < 20 or not closes:
            return
        price = float(closes[-1])
        hi20 = max(highs[-20:])
        # breakout above recent range
        if price > hi20:
            self._enter(sym, price, "BREAKOUT_UP", bar)

    def maybe_exit(self, sym, bar, signals):
        lows = bar.get("lows") or []
        closes = bar.get("closes") or []
        if len(lows) < 20 or not closes or sym not in self.positions:
            return
        price = float(closes[-1])
        lo20 = min(lows[-20:])
        # break below recent floor
        if price < lo20:
            self._exit(sym, price, "BREAKDOWN_EXIT", bar)


class HybridBot(BaseSimBot):
    """
    Fusion of the 4 strategies:
      - Requires AI BUY OR strong momentum filter
      - Bias long only when price above MA50
      - Avoids entries during high micro drawdown vs MA20
      - Discretionary exit on momentum fade OR AI SELL
    """

    def __init__(self):
        super().__init__("hybrid_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 50:
            return
        price = float(closes[-1])
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        ma50 = _sma(closes, 50)
        ai = (signals.get(sym) or {}).get("signal")
        ai_buy = isinstance(ai, str) and ai.upper() == "BUY"
        if None in (ema5, ema20, ma50):
            return

        mom_ok = (ema5 > ema20) and (price > ema20)
        trend_ok = price > ma50
        dd = _pct_chg(price, ema20)
        safe = True if dd is None else (dd > -0.03)

        if trend_ok and safe and (ai_buy or mom_ok):
            self._enter(sym, price, "HYBRID_BUY", bar)

    def maybe_exit(self, sym, bar, signals):
        if sym not in self.positions:
            return
        closes = bar.get("closes") or []
        if len(closes) < 20:
            return
        price = float(closes[-1])
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        ai = (signals.get(sym) or {}).get("signal")
        ai_sell = isinstance(ai, str) and ai.upper() == "SELL"
        if None in (ema5, ema20):
            return

        fade = ema5 < ema20
        if fade or ai_sell:
            self._exit(sym, price, "HYBRID_EXIT", bar)


# ----------------------------------------------------------------------
# Loop helpers
# ----------------------------------------------------------------------


def _loop_once(bots: List[BaseSimBot]) -> bool:
    market = load_latest_bars(BARS_PATH)
    if not market:
        log("ℹ️ No bars found — skipping loop.")
        return False
    signals = load_latest_signals(SIGNALS_DIR)

    # Step bots
    for bot in bots:
        try:
            bot.step(market, signals)
            bot.sync_trades_to_broker(market)
        except Exception as e:
            log(f"⚠️ Bot step failed ({bot.name}): {e}")

    return True


def _save_daily_log(bot: BaseSimBot, date_tag: str) -> None:
    """Write per-bot trade log + equity curve for the day."""
    _ensure_dirs()
    log_file = os.path.join(SIM_LOG_DIR, f"{date_tag}_{bot.name}.json")
    obj = {
        "bot": bot.name,
        "date": date_tag,
        "trades": bot.trades,
        "positions": bot.positions,
        "daily_equity": bot.daily_equity,
    }
    _write_json(log_file, obj)


def _update_summary(bots: List[BaseSimBot], date_tag: str) -> None:
    """Compact sim summary across all bots."""
    summary = _read_json(SIM_SUMMARY_FILE) or {}
    days = summary.get("days", [])
    day_entry = {
        "date": date_tag,
        "bots": {},
    }

    # approximate closing price map from last bar snapshot
    market = load_latest_bars(BARS_PATH)
    price_map = {s: d.get("price") for s, d in (market or {}).items() if isinstance(d.get("price"), (int, float))}

    for bot in bots:
        eq = bot.equity(price_map)
        day_entry["bots"][bot.name] = {
            "equity": eq,
            "positions": len(bot.positions),
            "trades": len(bot.trades),
        }

    days = [d for d in days if d.get("date") != date_tag]
    days.append(day_entry)
    summary["days"] = days
    _write_json(SIM_SUMMARY_FILE, summary)


# ----------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------


def run_all_bots(exec_mode: str = "sim", run_minutes: Optional[int] = None) -> None:
    """
    Main loop.
      exec_mode: "sim", "paper", "live"
      run_minutes: if provided, stop after N minutes; else run until:
           • market closed OR
           • stop-flag set
    """
    _ensure_dirs()
    clear_stop_flag()
    _init_alpaca_if_needed(exec_mode)
    date_tag = _today_tag()

    bots: List[BaseSimBot] = [
        MomentumBot(),
        MeanRevertBot(),
        SignalFollowBot(),
        BreakoutBot(),
        HybridBot(),
    ]
    log(f"🤖 SimTrader v3.0 started — mode={exec_mode.upper()} | bots={ [b.name for b in bots] } | start_cash=${START_CASH:.2f}")

    loops = 0
    try:
        while True:
            # External stop
            if stop_flag_set():
                log("🛑 Stop flag detected — shutting down day trading bot.")
                break

            # Market hours guard
            if not is_market_open_now():
                log("⏹ Market closed — stopping day trading bots until next session.")
                break

            loops += 1
            market_ok = _loop_once(bots)

            # Save equity checkpoint after each loop to keep logs responsive
            if market_ok:
                market = load_latest_bars(BARS_PATH)
                price_map = {s: d.get("price") for s, d in (market or {}).items() if isinstance(d.get("price"), (int, float))}
                ts = _now_utc_iso()
                for bot in bots:
                    bot.daily_equity.append((ts, bot.equity(price_map)))

            # Bounded run (for tests)
            if run_minutes is not None and loops >= max(1, int(run_minutes)):
                log(f"⏱ run_minutes limit reached ({run_minutes}) — exiting loop.")
                break

            time.sleep(LOOP_SECONDS)
    except KeyboardInterrupt:
        log("🛑 SimTrader interrupted by user.")
    except Exception as e:
        log(f"⚠️ SimTrader crashed: {e}")
    finally:
        log("✅ SimTrader loop ended (market closed or run complete).")
        # End-of-day logging + summary
        for bot in bots:
            if not bot.daily_equity:
                # ensure at least one equity point
                market = load_latest_bars(BARS_PATH)
                price_map = {s: d.get("price") for s, d in (market or {}).items() if isinstance(d.get("price"), (int, float))}
                bot.daily_equity.append((_now_utc_iso(), bot.equity(price_map)))
            _save_daily_log(bot, date_tag)

        _update_summary(bots, date_tag)
        log("✅ SimTrader logs & summary updated.")


# ----------------------------- ENTRYPOINT ------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AION Day Trading Bot (SimTrader v3.0)")
    p.add_argument(
        "--mode",
        type=str,
        default="sim",
        choices=["sim", "paper", "live", "stop"],
        help="Execution mode: sim (no broker), paper (Alpaca paper), live (future), stop (set stop flag and exit).",
    )
    p.add_argument(
        "--minutes",
        type=int,
        default=None,
        help="Optional runtime in minutes (testing). If omitted, runs until market close or stop-flag.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "stop":
        # Used by scheduler to request a graceful shutdown.
        set_stop_flag()
    else:
        # Run indefinitely (subject to market hours + stop flag), per-minute loop.
        run_all_bots(exec_mode=args.mode, run_minutes=args.minutes)
