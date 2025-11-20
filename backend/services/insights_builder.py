"""insights_builder.py — v3.0
AION Analytics — Multi-Layer Insights (Nightly, News, Social, ML Hybrid)

This module builds Top-N insight boards for the dashboard and tools.

It is *read-only* with respect to models:
    - It does NOT retrain models
    - It does NOT recompute predictions

It only:
    - Reads rolling.json.gz (already enriched with predictions, news, social, etc.)
    - Aggregates symbols into ranked boards
    - Writes JSON files under PATHS["insights"]

Boards (files):
    • top50_1w.json      — bullish, short-term swing (ML + news + social)
    • top50_2w.json      — bullish, medium-term
    • top50_4w.json      — bullish, position trades
    • top50_52w.json     — long horizon / conviction list
    • top50_social_heat.json — highest social 'heat' (virality)
    • top50_news_novelty.json — high-impact, novel news

Each board item typically includes:
    symbol, name, sector, score, confidence, news_sentiment, social_heat, etc.

Public entrypoint:
    build_daily_insights(limit: int = 50) -> dict
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, log, safe_float


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INSIGHTS_DIR = PATHS.get("insights", PATHS.get("root", Path("insights")))
if isinstance(INSIGHTS_DIR, Path):
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers to extract prediction / sentiment features
# ---------------------------------------------------------------------

def _extract_prediction_features(node: Dict[str, Any]) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Extract a base ML score and optional per-horizon scores.

    Returns:
        base_score: float in [-1, 1] roughly (0 = neutral)
        confidence: float in [0, 1]
        horizons: dict[horizon] = {"score": float, "conf": float}
    """
    preds = node.get("predictions") or {}
    base_score = 0.0
    conf = 0.5
    horizons: Dict[str, Dict[str, float]] = {}

    if isinstance(preds, dict):
        # Case: horizon -> {score, confidence, ...}
        for h, block in preds.items():
            if isinstance(block, dict):
                # collect numeric fields that look like scores
                score_candidates = []
                for k, v in block.items():
                    if not isinstance(v, (int, float)):
                        continue
                    lk = str(k).lower()
                    if any(key in lk for key in ["score", "rank", "prob", "edge", "alpha", "pred"]):
                        score_candidates.append(float(v))
                score = float(score_candidates[0]) if score_candidates else 0.0
                c = safe_float(block.get("confidence", block.get("prob", 0.5)))
                horizons[str(h)] = {"score": score, "conf": c}
            elif isinstance(block, (int, float)):
                horizons[str(h)] = {"score": float(block), "conf": 0.5}

    # Choose base_score:
    if "1d" in horizons:
        base_score = horizons["1d"]["score"]
        conf = horizons["1d"].get("conf", 0.5)
    elif horizons:
        # pick the first horizon deterministically
        h0 = sorted(horizons.keys())[0]
        base_score = horizons[h0]["score"]
        conf = horizons[h0].get("conf", 0.5)

    # Try legacy: flat fields directly on node["predictions"]
    if not horizons and isinstance(preds, dict):
        score_candidates = []
        for k, v in preds.items():
            if not isinstance(v, (int, float)):
                continue
            lk = str(k).lower()
            if any(key in lk for key in ["score", "rank", "prob", "edge", "alpha", "pred"]):
                score_candidates.append(float(v))
        if score_candidates:
            base_score = float(score_candidates[0])

    return base_score, conf, horizons


def _extract_sentiment_features(node: Dict[str, Any]) -> Dict[str, float]:
    """Pulls news + social intel into a compact dict of floats."""
    news = node.get("news") or {}
    social = node.get("social") or {}

    return {
        "news_sentiment": safe_float(news.get("sentiment", 0.0)),
        "news_impact": safe_float(news.get("impact_score", news.get("novelty", 0.0))),
        "social_sentiment": safe_float(social.get("sentiment", 0.0)),
        "social_heat": safe_float(social.get("heat_score", social.get("buzz", 0.0))),
    }


def _horizon_score(horizons: Dict[str, Dict[str, float]], horizon: str, fallback: float) -> float:
    if horizon in horizons:
        return safe_float(horizons[horizon].get("score", fallback))
    return fallback


# ---------------------------------------------------------------------
# Scoring recipes for different boards
# ---------------------------------------------------------------------

def _score_1w_long(base: float, conf: float, sents: Dict[str, float]) -> float:
    """Bullish 1-week: ML score + light sentiment overlay."""
    return (
        base * 0.6
        + conf * 0.2
        + sents["news_sentiment"] * 0.1
        + sents["social_sentiment"] * 0.1
    )


def _score_2w_long(base: float, conf: float, sents: Dict[str, float]) -> float:
    """Bullish 2-week: more weight on news impact + social heat."""
    return (
        base * 0.5
        + conf * 0.2
        + sents["news_impact"] * 0.2
        + math.tanh(sents["social_heat"]) * 0.1
    )


def _score_4w_long(base: float, conf: float, sents: Dict[str, float]) -> float:
    """Bullish 4-week: conviction plays, rely more on ML/confidence."""
    return (
        base * 0.7
        + conf * 0.25
        + sents["news_sentiment"] * 0.05
    )


def _score_52w_long(base: float, conf: float, sents: Dict[str, float]) -> float:
    """Long horizon: heavily confidence-weighted, sentiment as tiebreaker."""
    return (
        base * 0.5
        + conf * 0.45
        + (sents["news_sentiment"] + sents["social_sentiment"]) * 0.05
    )


# ---------------------------------------------------------------------
# Core board builder
# ---------------------------------------------------------------------

def _build_board(rows: List[Dict[str, Any]], limit: int, file_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Sort rows by score desc, take top-N, write JSON file."""
    if not rows:
        top = []
    else:
        top = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)[:limit]

    payload = {
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "limit": limit,
        "count": len(top),
        "meta": meta,
        "items": top,
    }

    out_path = INSIGHTS_DIR / file_name
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[insights_builder] ✅ wrote {file_name} ({len(top)} items)")
    except Exception as e:
        log(f"[insights_builder] ⚠️ failed to write {file_name}: {e}")

    return {
        "file": str(out_path),
        "count": len(top),
    }


# ---------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------

def build_daily_insights(limit: int = 50) -> Dict[str, Any]:
    """Aggregate Rolling into multi-layer Top-N insight boards.

    This is called from nightly_job *after*:
        - predictions
        - context_state
        - news_intel
        - social_sentiment_fetcher

    It is safe to run multiple times; it only reads Rolling and writes
    small JSON artifacts under PATHS["insights"].
    """
    rolling = _read_rolling()
    if not rolling:
        log("[insights_builder] ⚠️ No rolling.json.gz — skipping insights.")
        return {"status": "no_rolling"}

    rows_1w: List[Dict[str, Any]] = []
    rows_2w: List[Dict[str, Any]] = []
    rows_4w: List[Dict[str, Any]] = []
    rows_52w: List[Dict[str, Any]] = []
    rows_social: List[Dict[str, Any]] = []
    rows_news_novel: List[Dict[str, Any]] = []

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        sym_u = sym.upper()
        name = node.get("name") or node.get("company_name") or sym_u
        sector = (
            node.get("sector")
            or (node.get("fundamentals") or {}).get("sector")
            or ""
        )

        base_score, conf, horizons = _extract_prediction_features(node)
        sents = _extract_sentiment_features(node)

        # Skip symbols with no ML score at all
        # (still include them for social/news-only boards)
        has_ml = (base_score != 0.0) or bool(horizons)

        # ---- 1w / 2w / 4w / 52w boards (bullish) ----
        if has_ml:
            score_1w = _score_1w_long(
                _horizon_score(horizons, "1w", base_score),
                conf,
                sents,
            )
            score_2w = _score_2w_long(
                _horizon_score(horizons, "2w", base_score),
                conf,
                sents,
            )
            score_4w = _score_4w_long(
                _horizon_score(horizons, "4w", base_score),
                conf,
                sents,
            )
            score_52w = _score_52w_long(
                _horizon_score(horizons, "52w", base_score),
                conf,
                sents,
            )

            base_payload = {
                "symbol": sym_u,
                "name": name,
                "sector": sector,
                "ml_score": base_score,
                "confidence": conf,
                "news_sentiment": sents["news_sentiment"],
                "news_impact": sents["news_impact"],
                "social_sentiment": sents["social_sentiment"],
                "social_heat": sents["social_heat"],
            }

            rows_1w.append({**base_payload, "score": score_1w})
            rows_2w.append({**base_payload, "score": score_2w})
            rows_4w.append({**base_payload, "score": score_4w})
            rows_52w.append({**base_payload, "score": score_52w})

        # ---- Social heat board (even if no ML) ----
        if sents["social_heat"] != 0.0:
            rows_social.append({
                "symbol": sym_u,
                "name": name,
                "sector": sector,
                "score": sents["social_heat"],
                "social_heat": sents["social_heat"],
                "social_sentiment": sents["social_sentiment"],
                "news_sentiment": sents["news_sentiment"],
            })

        # ---- News novelty / impact board ----
        if sents["news_impact"] != 0.0:
            rows_news_novel.append({
                "symbol": sym_u,
                "name": name,
                "sector": sector,
                "score": sents["news_impact"],
                "news_impact": sents["news_impact"],
                "news_sentiment": sents["news_sentiment"],
                "social_sentiment": sents["social_sentiment"],
            })

    meta_common = {
        "note": "Automatically generated by insights_builder v3.0 (ML + news + social)",
    }

    outputs = {
        "top50_1w": _build_board(rows_1w, limit, "top50_1w.json", {**meta_common, "horizon": "1w"}),
        "top50_2w": _build_board(rows_2w, limit, "top50_2w.json", {**meta_common, "horizon": "2w"}),
        "top50_4w": _build_board(rows_4w, limit, "top50_4w.json", {**meta_common, "horizon": "4w"}),
        "top50_52w": _build_board(rows_52w, limit, "top50_52w.json", {**meta_common, "horizon": "52w"}),
        "top50_social_heat": _build_board(rows_social, limit, "top50_social_heat.json", {**meta_common, "board": "social_heat"}),
        "top50_news_novelty": _build_board(rows_news_novel, limit, "top50_news_novelty.json", {**meta_common, "board": "news_novelty"}),
    }

    log("[insights_builder] ───────────────────────────────────────────────")
    log("[insights_builder] ✅ Insights build complete.")
    for key, meta in outputs.items():
        log(f"   • {key}: {meta['count']:>3} tickers → {meta['file']}")
    log("[insights_builder] ───────────────────────────────────────────────")

    summary = {
        "status": "ok",
        "outputs": outputs,
    }
    return summary


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    out = build_daily_insights(limit=50)
    print(json.dumps(out, indent=2))
