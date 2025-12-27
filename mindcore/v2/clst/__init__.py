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
- Signal history persistence (audit trail)
- Session segmentation (topic/time-based splitting)

Signal Processing Flow:
    SimpleFLR collects signals → CLST.process_signals() applies weights → Storage updated
    → SignalStore records history (audit trail)

Session Segmentation Flow:
    New memory → SessionManager.should_segment() → Detect topic shift/time gap
    → Create new segment if needed → Maintain session coherence

Example:
    from mindcore.v2.clst import CLST, SessionManager, SignalStore
    from mindcore.v2.flr import SimpleFLR

    # SimpleFLR handles hot path, CLST handles cold path
    simple_flr = SimpleFLR(storage=storage)
    clst = CLST(storage=storage, vocabulary=vocab)

    # Session management
    session_manager = SessionManager(storage=storage)

    # Signal history
    signal_store = SignalStore(db_path="signals.db")

    # Query hot path
    result = simple_flr.query(user_id="user123", topics=["orders"])

    if result.clst_decision.needs_clst:
        # Query cold path with complex scoring
        memories = clst.search(user_id="user123", topics=["orders"])
        scored = clst.score_memories_complex(memories, query="order status")

    # Check for session segmentation
    if new_memory:
        decision = session_manager.should_segment(
            current_session_id="sess_123",
            new_memory=new_memory,
            user_id="user123",
        )
        if decision.should_segment:
            new_segment = session_manager.create_segment(
                parent_session_id="sess_123",
                user_id="user123",
                reason=decision.reason,
            )

    # Process signals with history
    signals = simple_flr.get_pending_signals()
    if signals:
        clst.process_signals(signals, signal_store=signal_store)
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
from .signals import (
    SignalStore,
    StoredSignal,
    SignalStats,
)
from .session_segmentation import (
    SessionManager,
    SessionSegment,
    SegmentDecision,
    SegmentReason,
    SegmentationPolicy,
    TopicDistribution,
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
    # Signal History
    "SignalStore",
    "StoredSignal",
    "SignalStats",
    # Session Segmentation
    "SessionManager",
    "SessionSegment",
    "SegmentDecision",
    "SegmentReason",
    "SegmentationPolicy",
    "TopicDistribution",
]
