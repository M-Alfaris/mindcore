"""CLST - Cognitive Long-term Storage Transfer.

A protocol for moving, syncing, and compressing long-term memory between
AI agents or between an agent and its external memory vault.

CLST handles:
- Long-term memory storage
- Memory compression and consolidation
- Cross-agent memory sync
- Memory transfer between instances
- Vocabulary version migrations
- Signal processing from SimpleFLR (complex scoring happens here)

Signal Processing Flow:
    SimpleFLR collects signals → CLST.process_signals() applies weights → Storage updated

Example:
    from mindcore.v2.clst import CLST
    from mindcore.v2.flr import SimpleFLR

    # SimpleFLR handles hot path, CLST handles cold path
    simple_flr = SimpleFLR(storage=storage)
    clst = CLST(storage=storage, vocabulary=vocab)

    # Query hot path
    result = simple_flr.query(user_id="user123", topics=["orders"])

    if result.clst_decision.needs_clst:
        # Query cold path with complex scoring
        memories = clst.search(user_id="user123", topics=["orders"])
        scored = clst.score_memories_complex(memories, query="order status")

    # Process any pending signals
    signals = simple_flr.get_pending_signals()
    if signals:
        clst.process_signals(signals)
        simple_flr.clear_pending_signals()
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
    SignalProcessingResult,
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
    "SignalProcessingResult",
    "SyncDirection",
    "SyncResult",
    "TransferManifest",
    # Session Aggregates
    "SessionAggregate",
    "HierarchicalQueryResult",
    "WeightCalculator",
]
