"""
AION Analytics — Backend Core Modules

Core logic:
  - config
  - data_pipeline
  - ai_model (multi-horizon)
  - policy_engine
  - context_state
  - regime_detector
  - supervisor_agent
"""
from .config import PATHS
from .data_pipeline import log, _read_rolling, save_rolling
