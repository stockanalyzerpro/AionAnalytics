
"""dt_backend.models
----------------------
Shared intraday model constants and helpers.

This module is intentionally lightweight and does **not** load any heavy
model objects on import. It just exposes label mappings and a helper
for resolving model artifact directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    # Prefer canonical DT_PATHS if available
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:  # pragma: no cover - safe fallback for tooling / tests
    DT_PATHS: Dict[str, Path] = {
        "dtmodels": Path("dt_backend") / "models"
    }

# 3-class intraday label space
LABEL_ORDER = ["SELL", "HOLD", "BUY"]
LABEL2ID: Dict[str, int] = {c: i for i, c in enumerate(LABEL_ORDER)}
ID2LABEL: Dict[int, str] = {i: c for c, i in LABEL2ID.items()}

def get_model_dir(kind: str) -> Path:
    """Return directory for intraday model artifacts of a given *kind*.

    By convention:

        DT_PATHS["dtmodels"] / f"{kind}_intraday"

    Examples:
        get_model_dir("lightgbm")     → .../lightgbm_intraday
        get_model_dir("lstm")         → .../lstm_intraday
        get_model_dir("transformer")  → .../transformer_intraday
        get_model_dir("ensemble")     → .../ensemble_intraday
    """
    root = Path(DT_PATHS.get("dtmodels", Path("dt_backend") / "models"))
    return root / f"{kind}_intraday"
