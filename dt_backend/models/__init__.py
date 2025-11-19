"""
AION Analytics — Saved Model Artifacts

This package stores trained intraday models and metadata:

Subfolders:
  • ensemble/
  • lightgbm_intraday/
  • lstm_intraday/
  • transformer_intraday/

Each contains:
  - model files
  - config.json
  - feature_map.json
  - label_map.json

This directory is used for loading model artifacts at runtime.
It does NOT contain training logic — training occurs in dt_backend/ml.
"""

__all__ = [
    "ensemble",
    "lightgbm_intraday",
    "lstm_intraday",
    "transformer_intraday",
]
