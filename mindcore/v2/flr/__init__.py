"""FLR - Fast Learning Recall.

A protocol for rapid retrieval, inference-time memory access, and short-term
contextual recall among AI agents.

FLR Modes:
- SimpleFLR (recommended): Deterministic cache layer that passes signals to CLST
- FLR (legacy): Complex probabilistic scoring with reinforcement processing

SimpleFLR Design:
- O(1) LRU cache lookup
- Deterministic filtering (topics, session, recency)
- Metadata-based CLST decision (is_clst_needed, confidence, priority)
- Collects signals and passes them to CLST for complex processing

Example (SimpleFLR - recommended):
    from mindcore.v2.flr import SimpleFLR

    flr = SimpleFLR(storage=storage)

    result = flr.query(
        user_id="user123",
        topics=["orders"],
        metadata_hints={"is_clst_needed": False, "confidence": 0.9},
    )

    if result.clst_decision.needs_clst:
        # Do full CLST query with complex scoring
        ...

Example (legacy FLR with robust reinforcement):
    from mindcore.v2.flr import FLR, SignalType, SignalSource

    flr = FLR(storage=storage, use_robust_reinforcement=True)

    # Apply detailed signal
    flr.reinforce_robust(
        memory_id="mem_123",
        signal_value=0.8,
        signal_type=SignalType.RELEVANCE,
        source=SignalSource.USER_EXPLICIT,
    )
"""

# Legacy FLR (complex probabilistic scoring)
from .recall import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)

# SimpleFLR (recommended - deterministic cache layer)
from .simple_recall import (
    SimpleFLR,
    SimpleRecallResult,
    CachedMemory,
    CLSTDecision,
    CLSTDecisionPolicy,
    CLSTNeedLevel,
    make_clst_decision,
)

from .reinforcement import (
    RobustReinforcement,
    ReinforcementSignal,
    SignalType,
    SignalSource,
    create_feedback_signal,
    batch_reinforce,
    DEFAULT_SOURCE_WEIGHTS,
    DEFAULT_TYPE_WEIGHTS,
    # Enhanced Reinforcement (2025-12)
    ImportanceAdjuster,
    ImportanceAdjustment,
    CrossMemoryReinforcer,
    RelatedMemorySignal,
    NegativeSignalDecay,
    BatchSignalResult,
    process_signal_batch,
)

from .metadata_feedback import (
    MetadataFeedbackTracker,
    MetadataSignal,
    MetadataEffectiveness,
)

from .usage_detector import (
    UsageDetector,
    UsageDetectionResult,
    MemoryUsage,
)

from .query_optimizer import (
    QueryOptimizer,
    QueryOptimization,
    TopicStats,
)


__all__ = [
    # SimpleFLR (recommended)
    "SimpleFLR",
    "SimpleRecallResult",
    "CachedMemory",
    "CLSTDecision",
    "CLSTDecisionPolicy",
    "CLSTNeedLevel",
    "make_clst_decision",
    # Legacy FLR
    "FLR",
    "ContextWindow",
    "Memory",
    "RecallResult",
    # Robust Reinforcement
    "RobustReinforcement",
    "ReinforcementSignal",
    "SignalType",
    "SignalSource",
    "create_feedback_signal",
    "batch_reinforce",
    "DEFAULT_SOURCE_WEIGHTS",
    "DEFAULT_TYPE_WEIGHTS",
    # Metadata Feedback
    "MetadataFeedbackTracker",
    "MetadataSignal",
    "MetadataEffectiveness",
    # Usage Detection
    "UsageDetector",
    "UsageDetectionResult",
    "MemoryUsage",
    # Query Optimization
    "QueryOptimizer",
    "QueryOptimization",
    "TopicStats",
    # Enhanced Reinforcement (2025-12)
    "ImportanceAdjuster",
    "ImportanceAdjustment",
    "CrossMemoryReinforcer",
    "RelatedMemorySignal",
    "NegativeSignalDecay",
    "BatchSignalResult",
    "process_signal_batch",
]
