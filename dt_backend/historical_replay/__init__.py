"""
dt_backend.historical_replay package

Components:
  • historical_replay_engine  → snapshot + bar-by-bar replay
  • sequence_builder          → LSTM/Transformer-ready sequence datasets
  • historical_replay_manager → metadata tracking for snapshots
  • replay_harness            → run replays / build sequences across many sessions
"""

from .historical_replay_engine import (
    run_replay_snapshot,
    replay_session_bar_by_bar,
    load_replay_snapshot,
)
from .sequence_builder import build_sequences_from_snapshot
from .historical_replay_manager import (
    SnapshotMeta,
    load_metadata,
    save_metadata,
    register_snapshot,
    list_snapshots,
)
from .replay_harness import (
    evaluate_policy_on_all_snapshots,
    build_sequences_for_all_snapshots,
)

__all__ = [
    "run_replay_snapshot",
    "replay_session_bar_by_bar",
    "load_replay_snapshot",
    "build_sequences_from_snapshot",
    "SnapshotMeta",
    "load_metadata",
    "save_metadata",
    "register_snapshot",
    "list_snapshots",
    "evaluate_policy_on_all_snapshots",
    "build_sequences_for_all_snapshots",
]
