# backend/bots/base_swing_bot.py
"""
Base EOD Swing Bot Engine — AION Analytics

This module implements a generic, AI-powered EOD swing bot, parameterized by
a SwingBotConfig.

All horizon-specific differences (1w/2w/4w) are captured in the config:
    • horizon
    • bot_key
    • max_positions
    • base_risk_pct
    • conf_threshold
    • stop_loss_pct
    • take_profit_pct
    • max_weight_per_name

Runners (runner_1w/2w/4w) supply an appropriate config and call:
    SwingBot(config).run(mode="full"|"loop")

Paths used:
    • rolling:       PATHS["stock_cache"]/master/rolling.json.gz
    • bot state:     PATHS["stock_cache"]/master/bot/rolling_<bot_key>.json.gz
    • bot logs:      PATHS["ml_data"]/bot_logs/<horizon>/bot_activity_YYYY-MM-DD.json
    • insights:      PATHS["insights"]/top50_<horizon>.json
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------

try:
    from backend.core.config import PATHS
except Exception:
    from backend.config import PATHS  # type: ignore

try:
    from backend.core.data_pipeline import log  # type: ignore
except Exception:
    try:
        from backend.data_pipeline import log  # type: ignore
    except Exception:

        def log(msg: str) -> None:
            print(msg, flush=True)


ROOT = Path(PATHS.get("root", "."))
ML_DATA = Path(PATHS["ml_data"])
STOCK_CACHE = Path(PATHS["stock_cache"])
INSIGHTS_DIR = Path(PATHS.get("insights", ROOT / "insights"))


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_gz_json(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_gz_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------
# Config & state models
# ---------------------------------------------------------------------


@dataclass
class SwingBotConfig:
    """Hyper-parameters and identity for an EOD swing bot."""

    horizon: str              # "1w", "2w", "4w"
    bot_key: str              # e.g. "eod_1w"
    max_positions: int
    base_risk_pct: float      # reserved for future sizing refinements
    conf_threshold: float     # minimum AI confidence to act
    stop_loss_pct: float      # -0.05 → -5%
    take_profit_pct: float    # 0.10 → +10%
    max_weight_per_name: float  # max fraction of equity in one symbol
    initial_cash: float = 100.0


@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    target: float


@dataclass
class BotState:
    cash: float
    positions: Dict[str, Position]
    last_equity: float
    last_updated: str

    @classmethod
    def from_dict(cls, d: dict, initial_cash: float) -> "BotState":
        cash = _safe_float(d.get("cash", initial_cash), initial_cash)
        last_equity = _safe_float(d.get("last_equity", cash), cash)
        last_updated = str(d.get("last_updated") or _now_iso())
        raw_pos = d.get("positions", {}) or {}
        positions: Dict[str, Position] = {}
        for sym, pd in raw_pos.items():
            positions[str(sym).upper()] = Position(
                qty=_safe_float(pd.get("qty"), 0.0),
                entry=_safe_float(pd.get("entry"), 0.0),
                stop=_safe_float(pd.get("stop"), 0.0),
                target=_safe_float(pd.get("target"), 0.0),
            )
        return cls(
            cash=cash,
            positions=positions,
            last_equity=last_equity,
            last_updated=last_updated,
        )

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "last_equity": self.last_equity,
            "last_updated": self.last_updated,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
        }


@dataclass
class Trade:
    t: str
    symbol: str
    side: str  # "BUY" / "SELL"
    qty: float
    price: float
    reason: str
    pnl: Optional[float] = None


# ---------------------------------------------------------------------
# SwingBot engine
# ---------------------------------------------------------------------


class SwingBot:
    """
    Generic AI-powered EOD swing bot.

    Usage:
        from backend.bots.base_swing_bot import SwingBot
        from backend.bots.strategy_1w import CONFIG as CONFIG_1W

        bot = SwingBot(CONFIG_1W)
        bot.run("full")  # or "loop"
    """

    def __init__(self, config: SwingBotConfig) -> None:
        self.cfg = config

        # Derived paths
        self.rolling_file = STOCK_CACHE / "master" / "rolling.json.gz"
        self.bot_state_file = STOCK_CACHE / "master" / "bot" / f"rolling_{config.bot_key}.json.gz"
        self.bot_log_dir = ML_DATA / "bot_logs" / config.horizon
        self.insights_file = INSIGHTS_DIR / f"top50_{config.horizon}.json"

        log(
            f"[{self.cfg.bot_key}] SwingBot initialized — "
            f"horizon={self.cfg.horizon}, max_positions={self.cfg.max_positions}"
        )

    # ------------------------ Data Loading ------------------------ #

    def load_rolling(self) -> Dict[str, dict]:
        js = _read_gz_json(self.rolling_file)
        if not isinstance(js, dict):
            log(f"[{self.cfg.bot_key}] ⚠️ rolling.json.gz missing or invalid.")
            return {}
        symbols = js.get("symbols", js)
        out: Dict[str, dict] = {}
        for sym, node in symbols.items():
            out[str(sym).upper()] = node or {}
        log(f"[{self.cfg.bot_key}] rolling loaded for {len(out)} symbols.")
        return out

    def load_insights(self) -> List[dict]:
        js = _read_json(self.insights_file)
        if not isinstance(js, (dict, list)):
            log(f"[{self.cfg.bot_key}] ⚠️ Insights file {self.insights_file} missing or invalid.")
            return []
        if isinstance(js, dict):
            arr = js.get("insights", [])
        else:
            arr = js
        if not isinstance(arr, list):
            return []
        return arr

    # ---------------------- Feature Extractors -------------------- #

    @staticmethod
    def _extract_price(node: dict) -> Optional[float]:
        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        if price is None:
            return None
        try:
            p = float(price)
            return p if p > 0 else None
        except Exception:
            return None

    def _extract_ai_signal(self, node: dict) -> Tuple[str, float]:
        """
        Try horizon-specific predictions first, fallback to generic keys.
        Structure expected:
            node["predictions"][horizon] = {"signal": "...", "confidence": 0-1}
        """
        preds = node.get("predictions", {}) or {}
        h = preds.get(self.cfg.horizon, {}) or {}
        sig = h.get("signal") or preds.get("signal") or ""
        conf = h.get("confidence") or preds.get("confidence") or 0.0
        return str(sig).upper(), _safe_float(conf, 0.0)

    @staticmethod
    def _extract_regime_bias(node: dict) -> float:
        """
        Simple, robust context/regime bias in [0.5, 1.5].
        """
        ctx = node.get("context", {}) or {}
        trend = str(ctx.get("trend", "")).lower()
        regime_score = _safe_float(ctx.get("regime_score"), 0.0)

        bias = 1.0
        if trend == "bullish":
            bias *= 1.1
        elif trend == "bearish":
            bias *= 0.9

        # regime_score ∈ [-1,1] ⇒ small ±10% modulation
        bias *= 1.0 + 0.1 * max(-1.0, min(1.0, regime_score))
        return max(0.5, min(1.5, bias))

    # ------------------------- State I/O -------------------------- #

    def load_bot_state(self) -> BotState:
        js = _read_gz_json(self.bot_state_file)
        if not isinstance(js, dict):
            log(f"[{self.cfg.bot_key}] ℹ️ No existing state — seeding new bot state.")
            return BotState(
                cash=self.cfg.initial_cash,
                positions={},
                last_equity=self.cfg.initial_cash,
                last_updated=_now_iso(),
            )
        return BotState.from_dict(js, initial_cash=self.cfg.initial_cash)

    def save_bot_state(self, state: BotState) -> None:
        self.bot_state_file.parent.mkdir(parents=True, exist_ok=True)
        _write_gz_json(self.bot_state_file, state.to_dict())
        log(
            f"[{self.cfg.bot_key}] 💾 State saved — "
            f"positions={len(state.positions)} cash={state.cash:.2f} equity={state.last_equity:.2f}"
        )

    # -------------------- AI Ranking & Weights -------------------- #

    def build_ai_ranked_universe(
        self, rolling: Dict[str, dict], insights: List[dict]
    ) -> List[Tuple[str, float]]:
        """
        Combine insights ranking + AI predictions + regime bias
        into a composite score for each symbol.
        """
        # insight rank: lower index = stronger rank
        insight_rank: Dict[str, int] = {}
        for idx, row in enumerate(insights):
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if not sym:
                continue
            insight_rank[sym] = idx

        universe_scores: Dict[str, float] = {}

        for sym, node in rolling.items():
            price = self._extract_price(node)
            if price is None or price <= 0:
                continue

            sig, conf = self._extract_ai_signal(node)
            if sig not in ("BUY", "HOLD"):
                # long-only in this version
                continue

            # base AI score from confidence
            ai_score = conf
            if sig == "BUY":
                ai_score *= 1.1

            # insight rank bonus (if present)
            if sym in insight_rank:
                rank = insight_rank[sym]
                n = max(1, len(insights))
                rank_score = 1.0 - (rank / n)
                ai_score += 0.3 * rank_score

            # regime/context bias
            bias = self._extract_regime_bias(node)
            score = ai_score * bias

            # light penalty for penny-ish names
            if price < 3:
                score *= 0.7

            if score > 0:
                universe_scores[sym] = score

        ranked = sorted(universe_scores.items(), key=lambda kv: kv[1], reverse=True)
        log(f"[{self.cfg.bot_key}] AI-ranked universe size={len(ranked)}")
        return ranked

    def construct_target_weights(self, ranked: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        From ranked (sym, score), produce target weights.
        - keep top max_positions
        - normalize
        - cap each name at max_weight_per_name
        """
        if not ranked:
            return {}

        top = ranked[: self.cfg.max_positions]
        scores = [max(0.0, s) for _, s in top]
        total = sum(scores)
        if total <= 0:
            return {}

        weights: Dict[str, float] = {}
        for sym, s in top:
            w = s / total
            w = min(w, self.cfg.max_weight_per_name)
            weights[sym] = w

        # renormalize after capping
        total = sum(weights.values())
        if total > 0:
            for sym in list(weights.keys()):
                weights[sym] /= total

        log(f"[{self.cfg.bot_key}] Constructed {len(weights)} target weights.")
        return weights

    # ---------------------- Portfolio Logic ----------------------- #

    @staticmethod
    def compute_equity(state: BotState, prices: Dict[str, float]) -> float:
        equity = state.cash
        for sym, pos in state.positions.items():
            px = prices.get(sym)
            if isinstance(px, (int, float)):
                equity += pos.qty * px
        return equity

    def rebalance_full(
        self,
        state: BotState,
        rolling: Dict[str, dict],
        target_weights: Dict[str, float],
    ) -> List[Trade]:
        """
        FULL pre-market rebalance:
            - Sell anything outside universe
            - Adjust positions toward target weights
        """
        trades: List[Trade] = []

        prices: Dict[str, float] = {}
        for sym, node in rolling.items():
            px = self._extract_price(node)
            if px is not None:
                prices[sym] = px

        equity = self.compute_equity(state, prices)
        if equity <= 0:
            equity = state.cash if state.cash > 0 else self.cfg.initial_cash

        log(
            f"[{self.cfg.bot_key}] Starting FULL rebalance — "
            f"equity={equity:.2f}, positions={len(state.positions)}"
        )

        # 1) Sell names not in target
        for sym in list(state.positions.keys()):
            if sym not in target_weights:
                px = prices.get(sym)
                pos = state.positions[sym]
                if isinstance(px, (int, float)) and pos.qty > 0:
                    proceeds = pos.qty * px
                    state.cash += proceeds
                    pnl = (px - pos.entry) * pos.qty
                    trades.append(
                        Trade(
                            t=_now_iso(),
                            symbol=sym,
                            side="SELL",
                            qty=pos.qty,
                            price=px,
                            reason="REMOVE_FROM_UNIVERSE",
                            pnl=pnl,
                        )
                    )
                    log(
                        f"[{self.cfg.bot_key}] SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                        f"(universe exit) PnL={pnl:.2f}"
                    )
                del state.positions[sym]

        # 2) Recompute equity from target universe only
        prices_now = {
            s: prices.get(s)
            for s in target_weights.keys()
            if isinstance(prices.get(s), (int, float))
        }
        equity = self.compute_equity(state, prices_now)
        state.last_equity = equity

        # 3) Move toward target weights
        for sym, target_w in target_weights.items():
            px = prices_now.get(sym)
            if px is None or px <= 0:
                continue

            target_value = equity * target_w
            pos = state.positions.get(sym)
            current_value = pos.qty * px if pos else 0.0
            diff_value = target_value - current_value

            # ignore tiny moves
            if abs(diff_value) < max(10.0, 0.01 * equity):
                continue

            qty_delta = diff_value / px

            if qty_delta > 0:
                # BUY / increase
                qty = qty_delta
                state.cash -= qty * px
                if pos:
                    total_qty = pos.qty + qty
                    if total_qty <= 0:
                        continue
                    new_entry = (pos.entry * pos.qty + px * qty) / total_qty
                    pos.qty = total_qty
                    pos.entry = new_entry
                    pos.stop = new_entry * (1.0 + self.cfg.stop_loss_pct)
                    pos.target = new_entry * (1.0 + self.cfg.take_profit_pct)
                else:
                    state.positions[sym] = Position(
                        qty=qty,
                        entry=px,
                        stop=px * (1.0 + self.cfg.stop_loss_pct),
                        target=px * (1.0 + self.cfg.take_profit_pct),
                    )
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="BUY",
                        qty=qty,
                        price=px,
                        reason="TARGET_REBALANCE",
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] BUY {qty:.4f} {sym} @ {px:.4f} "
                    f"(target_w={target_w:.3f})"
                )
            else:
                # SELL / decrease
                qty = abs(qty_delta)
                if not pos or pos.qty <= 0:
                    continue
                sell_qty = min(qty, pos.qty)
                state.cash += sell_qty * px
                pnl = (px - pos.entry) * sell_qty
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=sell_qty,
                        price=px,
                        reason="TARGET_REBALANCE",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] SELL {sell_qty:.4f} {sym} @ {px:.4f} "
                    f"(target_w={target_w:.3f}) PnL={pnl:.2f}"
                )
                pos.qty -= sell_qty
                if pos.qty <= 0:
                    del state.positions[sym]

        state.last_equity = self.compute_equity(state, prices_now)
        state.last_updated = _now_iso()
        return trades

    def apply_loop_risk_checks(
        self, state: BotState, rolling: Dict[str, dict]
    ) -> List[Trade]:
        """
        Intraday loop:
            - enforce SL/TP
            - AI SELL exits for strong convictions
        """
        trades: List[Trade] = []
        prices: Dict[str, float] = {}

        for sym, node in rolling.items():
            px = self._extract_price(node)
            if px is not None:
                prices[sym] = px

        for sym in list(state.positions.keys()):
            pos = state.positions[sym]
            px = prices.get(sym)
            if px is None:
                continue

            # Stop-loss
            if px <= pos.stop:
                pnl = (px - pos.entry) * pos.qty
                state.cash += pos.qty * px
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=pos.qty,
                        price=px,
                        reason="STOP_LOSS",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] STOP_LOSS SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                    f"PnL={pnl:.2f}"
                )
                del state.positions[sym]
                continue

            # Take-profit
            if px >= pos.target:
                pnl = (px - pos.entry) * pos.qty
                state.cash += pos.qty * px
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=pos.qty,
                        price=px,
                        reason="TAKE_PROFIT",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] TAKE_PROFIT SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                    f"PnL={pnl:.2f}"
                )
                del state.positions[sym]
                continue

            # AI SELL override
            node = rolling.get(sym) or {}
            sig, conf = self._extract_ai_signal(node)
            if sig == "SELL" and conf >= self.cfg.conf_threshold:
                pnl = (px - pos.entry) * pos.qty
                state.cash += pos.qty * px
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=pos.qty,
                        price=px,
                        reason=f"AI_SELL_{conf:.2f}",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] AI SELL EXIT {pos.qty:.4f} {sym} @ {px:.4f} "
                    f"PnL={pnl:.2f}"
                )
                del state.positions[sym]

        state.last_equity = self.compute_equity(state, prices)
        state.last_updated = _now_iso()
        return trades

    # --------------------------- Logging -------------------------- #

    def append_trades_to_daily_log(self, trades: List[Trade]) -> None:
        if not trades:
            return
        self.bot_log_dir.mkdir(parents=True, exist_ok=True)
        day = _today()
        path = self.bot_log_dir / f"bot_activity_{day}.json"

        data = _read_json(path) or {}
        arr = data.get(self.cfg.bot_key, [])
        if not isinstance(arr, list):
            arr = []
        for t in trades:
            arr.append(asdict(t))
        data[self.cfg.bot_key] = arr

        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        log(f"[{self.cfg.bot_key}] 📓 Logged {len(trades)} trades to {path}")

    # --------------------------- Orchestration -------------------- #

    def run_full(self) -> None:
        """Premarket FULL rebalance."""
        rolling = self.load_rolling()
        if not rolling:
            log(f"[{self.cfg.bot_key}] ⚠️ No rolling data — aborting FULL rebalance.")
            return

        insights = self.load_insights()
        ranked = self.build_ai_ranked_universe(rolling, insights)
        target_weights = self.construct_target_weights(ranked)

        state = self.load_bot_state()
        trades = self.rebalance_full(state, rolling, target_weights)
        self.save_bot_state(state)
        self.append_trades_to_daily_log(trades)
        log(
            f"[{self.cfg.bot_key}] ✅ FULL rebalance complete. "
            f"Trades={len(trades)}"
        )

    def run_loop(self) -> None:
        """Intraday LOOP — risk checks only."""
        rolling = self.load_rolling()
        if not rolling:
            log(f"[{self.cfg.bot_key}] ⚠️ No rolling data — aborting LOOP check.")
            return

        state = self.load_bot_state()
        trades = self.apply_loop_risk_checks(state, rolling)
        self.save_bot_state(state)
        self.append_trades_to_daily_log(trades)
        log(
            f"[{self.cfg.bot_key}] ✅ LOOP risk-check complete. "
            f"Trades={len(trades)}"
        )

    def run(self, mode: str = "full") -> None:
        """
        Dispatch method used by runners.
        mode ∈ {"full", "loop"}
        """
        if mode == "loop":
            self.run_loop()
        else:
            self.run_full()
