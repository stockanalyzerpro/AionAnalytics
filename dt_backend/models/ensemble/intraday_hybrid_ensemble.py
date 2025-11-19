
"""Intraday hybrid ensemble for combining LightGBM, LSTM, Transformer outputs.

This module is intentionally light-weight and pure-numpy. It knows nothing
about how base models are trained; it only blends per-class probability
outputs that are passed in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from dt_backend.models import LABEL_ORDER, get_model_dir

EPS = 1e-8


@dataclass
class EnsembleConfig:
    """Simple weight container for the intraday ensemble.

    Weights do **not** need to sum to 1; they are normalized internally.
    """

    w_lgb: float = 0.6
    w_lstm: float = 0.2
    w_transf: float = 0.2

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EnsembleConfig":
        if path is None:
            path = get_model_dir("ensemble") / "meta_ensemble.json"
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                w_lgb=float(data.get("w_lgb", 0.6)),
                w_lstm=float(data.get("w_lstm", 0.2)),
                w_transf=float(data.get("w_transf", 0.2)),
            )
        except FileNotFoundError:
            return cls()
        except Exception:
            # On any parse error, fall back to safe defaults
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = get_model_dir("ensemble") / "meta_ensemble.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **asdict(self),
                    "label_order": LABEL_ORDER,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def as_dict(self) -> Dict[str, float]:
        return {
            "w_lgb": float(self.w_lgb),
            "w_lstm": float(self.w_lstm),
            "w_transf": float(self.w_transf),
        }

    @property
    def weights_array(self) -> np.ndarray:
        arr = np.array([self.w_lgb, self.w_lstm, self.w_transf], dtype=float)
        arr[arr < 0.0] = 0.0
        s = float(arr.sum())
        if s <= 0.0:
            # Default to LGBM only, but normalized
            arr = np.array([1.0, 0.0, 0.0], dtype=float)
        return arr / float(arr.sum())


class IntradayHybridEnsemble:
    """Blend per-model probability outputs into a single probability.

    Typical usage:
        cfg = EnsembleConfig.load()
        ens = IntradayHybridEnsemble(cfg)
        p = ens.predict_proba(p_lgb=..., p_lstm=..., p_transf=...)
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()

    @staticmethod
    def _to_logits(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1.0 - EPS)
        return np.log(p)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = exp.sum(axis=1, keepdims=True) + EPS
        return exp / denom

    def blend(
        self,
        p_lgb: Optional[np.ndarray] = None,
        p_lstm: Optional[np.ndarray] = None,
        p_transf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Blend available per-model probabilities.

        All p_* must be (N, C) where C == len(LABEL_ORDER). Missing models
        can be passed as None.
        """
        probs = []
        logits = []
        weights = []

        w_lgb, w_lstm, w_transf = self.config.weights_array

        if p_lgb is not None:
            p = np.asarray(p_lgb, dtype=float)
            probs.append(p)
            logits.append(self._to_logits(p))
            weights.append(w_lgb)
        if p_lstm is not None:
            p = np.asarray(p_lstm, dtype=float)
            probs.append(p)
            logits.append(self._to_logits(p))
            weights.append(w_lstm)
        if p_transf is not None:
            p = np.asarray(p_transf, dtype=float)
            probs.append(p)
            logits.append(self._to_logits(p))
            weights.append(w_transf)

        if not probs:
            raise ValueError("IntradayHybridEnsemble.blend called with no inputs.")

        if len(probs) == 1:
            return probs[0]

        W = np.array(weights, dtype=float)
        W[W < 0.0] = 0.0
        s = float(W.sum())
        if s <= 0.0:
            W = np.ones_like(W) / len(W)
        else:
            W = W / s

        stacked = np.stack(logits, axis=0)  # (M, N, C)
        blended_logits = np.tensordot(W, stacked, axes=(0, 0))  # (N, C)
        return self._softmax(blended_logits)

    def predict_proba(
        self,
        p_lgb: Optional[np.ndarray] = None,
        p_lstm: Optional[np.ndarray] = None,
        p_transf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.blend(p_lgb=p_lgb, p_lstm=p_lstm, p_transf=p_transf)

    def predict_class(
        self,
        p_lgb: Optional[np.ndarray] = None,
        p_lstm: Optional[np.ndarray] = None,
        p_transf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        proba = self.predict_proba(p_lgb=p_lgb, p_lstm=p_lstm, p_transf=p_transf)
        return np.argmax(proba, axis=1)
