# dt_backend/core/execution_dt.py — Advanced execution intent layer.
"""
Converts policy_dt (intents) into execution_dt blocks that downstream
executors (e.g. backend trade bots) can consume.

    rolling[sym]["execution_dt"] = {
        "side": "BUY" | "SELL" | "FLAT",
        "size": 0.0–1.0,          # fraction of max capital per symbol
        "confidence_adj": 0.0–1.0,
        "cooldown": bool,
        "valid_until": <ISO8601 UTC>,
        "ts": <ISO8601 UTC>
    }
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log


class ExecConfig:
    """Tunable knobs for execution behavior."""

    # Maximum notional fraction allocated per symbol for a *full* conviction signal.
    max_symbol_fraction: float = 0.15

    # Minimum confidence required to allocate anything.
    min_conf: float = 0.25

    # Hard cap on adjusted confidence (after volatility / regime).
    max_conf_cap: float = 0.95

    # Cooldown window to avoid rapid flips between BUY and SELL.
    cooldown_minutes: int = 10

    # Base validity window for an execution intent.
    valid_minutes: int = 15


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _trend_and_vol(context_dt: Dict[str, Any]) -> Tuple[str, str]:
    trend = str(context_dt.get("intraday_trend") or "").strip()
    vol_bkt = str(context_dt.get("vol_bucket") or "").strip()

    if not trend:
        r = _safe_float(context_dt.get("intraday_return"), 0.0)
        if r >= 0.01:
            trend = "strong_bull"
        elif r >= 0.003:
            trend = "bull"
        elif r <= -0.01:
            trend = "strong_bear"
        elif r <= -0.003:
            trend = "bear"
        else:
            trend = "flat"

    if not vol_bkt:
        vol = _safe_float(context_dt.get("intraday_vol"), 0.0)
        if vol >= 0.02:
            vol_bkt = "high"
        elif vol >= 0.007:
            vol_bkt = "medium"
        else:
            vol_bkt = "low"

    return trend, vol_bkt


def _size_from_conf_and_vol(conf: float, vol_bkt: str, cfg: ExecConfig) -> float:
    if conf <= 0.0:
        return 0.0

    conf = max(0.0, min(cfg.max_conf_cap, conf))

    # Volatility-aware scaling: risk-off when vol is high.
    if vol_bkt == "high":
        scale = 0.4
    elif vol_bkt == "medium":
        scale = 0.7
    else:
        scale = 1.0

    raw_size = conf * cfg.max_symbol_fraction * scale
    return max(0.0, min(cfg.max_symbol_fraction, raw_size))


def _parse_iso_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _cooldown_active(prev_exec: Dict[str, Any], new_side: str, cfg: ExecConfig) -> bool:
    """Return True if we are within cooldown window from a conflicting action."""
    if not prev_exec:
        return False

    prev_side = str(prev_exec.get("side") or "").upper()
    if prev_side not in {"BUY", "SELL"} or new_side not in {"BUY", "SELL"}:
        return False

    # Only enforce cooldown when flipping direction (BUY → SELL or SELL → BUY)
    if prev_side == new_side:
        return False

    ts = _parse_iso_ts(prev_exec.get("ts"))
    if ts is None:
        return False

    now = datetime.now(timezone.utc)
    delta = now - ts
    return delta.total_seconds() < cfg.cooldown_minutes * 60


def run_execution_intraday(cfg: ExecConfig | None = None) -> Dict[str, Any]:
    """Main entry point.

    Reads rolling, converts policy_dt into execution_dt, writes back.
    Returns summary dict.
    """
    cfg = cfg or ExecConfig()
    rolling = _read_rolling()
    if not rolling:
        log("[exec_dt] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    now = datetime.now(timezone.utc)
    updated = 0

    for sym, node_raw in list(rolling.items()):
        if sym.startswith("_"):
            continue

        node = ensure_symbol_node(rolling, sym)
        policy = node.get("policy_dt") or {}
        ctx = node.get("context_dt") or {}

        intent = str(policy.get("intent") or "").upper()
        conf = _safe_float(policy.get("confidence"), 0.0)

        if intent not in {"BUY", "SELL"} or conf < cfg.min_conf:
            side = "FLAT"
            conf_adj = 0.0
        else:
            trend, vol_bkt = _trend_and_vol(ctx)
            size = _size_from_conf_and_vol(conf, vol_bkt, cfg)

            # If size is effectively zero, treat as FLAT.
            if size <= 0.0:
                side = "FLAT"
                conf_adj = 0.0
            else:
                side = intent
                conf_adj = min(cfg.max_conf_cap, conf)

        # Determine cooldown and adjust side/size accordingly.
        prev_exec = node.get("execution_dt") or {}
        cooldown = _cooldown_active(prev_exec, side, cfg)

        if cooldown and side in {"BUY", "SELL"}:
            # Respect cooldown by downgrading to FLAT for this cycle.
            side = "FLAT"

        # Compute final size if side is active; zero otherwise.
        if side in {"BUY", "SELL"}:
            trend, vol_bkt = _trend_and_vol(ctx)
            size = _size_from_conf_and_vol(conf_adj, vol_bkt, cfg)
        else:
            size = 0.0

        valid_until = (now + timedelta(minutes=cfg.valid_minutes)).isoformat()

        node["execution_dt"] = {
            "side": side,
            "size": float(size),
            "confidence_adj": float(conf_adj),
            "cooldown": bool(cooldown),
            "valid_until": valid_until,
            "ts": now.isoformat(),
        }

        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[exec_dt] ✅ updated execution_dt for {updated} symbols.")
    return {"symbols": len(rolling), "updated": updated}


def main() -> None:
    run_execution_intraday()


if __name__ == "__main__":
    main()
