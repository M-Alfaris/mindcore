"""CLST - Cognitive Long-term Storage Transfer.

A protocol for moving, syncing, and compressing long-term memory between
AI agents or between an agent and its external memory vault.

Includes session aggregates for hierarchical memory retrieval with
weighted metadata matching.
"""

from .aggregates import (
    HierarchicalQueryResult,
    SessionAggregate,
    WeightCalculator,
)
from .storage import (
    CLST,
    CompressionResult,
    CompressionStrategy,
    MigrationResult,
    SyncDirection,
    SyncResult,
    TransferManifest,
)


__all__ = [
    # Core CLST
    "CLST",
    "CompressionResult",
    "CompressionStrategy",
    "MigrationResult",
    "SyncDirection",
    "SyncResult",
    "TransferManifest",
    # Session Aggregates
    "SessionAggregate",
    "HierarchicalQueryResult",
    "WeightCalculator",
]
