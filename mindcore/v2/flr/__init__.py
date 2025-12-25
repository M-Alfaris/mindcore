"""FLR - Fast Learning Recall.

A protocol for rapid retrieval, inference-time memory access, and short-term
contextual recall among AI agents.

Reinforcement Modes:
- Legacy (naive): Simple bounded accumulation with diminishing returns
- Robust: Temporal decay, multi-signal types, exploration bonus, trend tracking

Example (robust reinforcement):
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

from .recall import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
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
    # Core FLR
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
]
