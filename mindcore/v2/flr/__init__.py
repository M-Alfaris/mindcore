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

from .cache import (
    CacheEntry,
    CacheEventType,
    CacheStats,
    SmartCache,
)
from .metadata_feedback import (
    MetadataEffectiveness,
    MetadataFeedbackTracker,
    MetadataSignal,
)
from .preferences import (
    ConflictResolutionResult,
    ConflictResolutionStrategy,
    ConflictStatus,
    PreferenceConflict,
    PreferenceHistory,
    PreferenceManager,
    PreferenceSummary,
    PreferenceUpdate,
)
from .query_optimizer import (
    QueryOptimization,
    QueryOptimizer,
    TopicStats,
)
from .recall import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)
from .reinforcement import (
    DEFAULT_SOURCE_WEIGHTS,
    DEFAULT_TYPE_WEIGHTS,
    BatchSignalResult,
    CrossMemoryReinforcer,
    # Enhanced Reinforcement (2025-12)
    ImportanceAdjuster,
    ImportanceAdjustment,
    NegativeSignalDecay,
    ReinforcementSignal,
    RelatedMemorySignal,
    RobustReinforcement,
    SignalSource,
    SignalType,
    batch_reinforce,
    create_feedback_signal,
    process_signal_batch,
)
from .usage_detector import (
    MemoryUsage,
    UsageDetectionResult,
    UsageDetector,
)


__all__ = [
    "DEFAULT_SOURCE_WEIGHTS",
    "DEFAULT_TYPE_WEIGHTS",
    # Core FLR
    "FLR",
    "BatchSignalResult",
    "CacheEntry",
    "CacheEventType",
    "CacheStats",
    "ConflictResolutionResult",
    "ConflictResolutionStrategy",
    "ConflictStatus",
    "ContextWindow",
    "CrossMemoryReinforcer",
    # Enhanced Reinforcement (2025-12)
    "ImportanceAdjuster",
    "ImportanceAdjustment",
    "Memory",
    "MemoryUsage",
    "MetadataEffectiveness",
    # Metadata Feedback
    "MetadataFeedbackTracker",
    "MetadataSignal",
    "NegativeSignalDecay",
    "PreferenceConflict",
    "PreferenceHistory",
    # Preferences
    "PreferenceManager",
    "PreferenceSummary",
    "PreferenceUpdate",
    "QueryOptimization",
    # Query Optimization
    "QueryOptimizer",
    "RecallResult",
    "ReinforcementSignal",
    "RelatedMemorySignal",
    # Robust Reinforcement
    "RobustReinforcement",
    "SignalSource",
    "SignalType",
    # Smart Cache
    "SmartCache",
    "TopicStats",
    "UsageDetectionResult",
    # Usage Detection
    "UsageDetector",
    "batch_reinforce",
    "create_feedback_signal",
    "process_signal_batch",
]
