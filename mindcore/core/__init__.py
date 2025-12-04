"""Core modules for Mindcore framework."""

from .adaptive_preferences import (
    AdaptiveConfig,
    AdaptivePreferencesLearner,
    PreferenceSignal,
    get_adaptive_learner,
    reset_adaptive_learner,
)
from .cache_invalidation import (
    CacheInvalidationManager,
    InvalidationEvent,
    InvalidationReason,
    InvalidationStats,
    get_cache_invalidation,
    reset_cache_invalidation,
)
from .cache_manager import CacheManager
from .config_loader import ConfigLoader
from .db_manager import DatabaseConnectionError, DatabaseManager
from .disk_cache_manager import DiskCacheManager
from .metrics_manager import MetricsManager
from .multi_agent import (
    AgentProfile,
    AgentVisibility,
    MemorySharingMode,
    MultiAgentConfig,
    MultiAgentManager,
    configure_multi_agent,
    get_multi_agent_manager,
    reset_multi_agent_manager,
)
from .preferences_manager import AsyncPreferencesManager, PreferencesManager
from .prometheus_metrics import (
    MindcoreMetrics,
    get_metrics,
    is_prometheus_available,
    reset_metrics,
    start_metrics_server,
)
from .retention_policy import (
    MemoryTier,
    RetentionConfig,
    RetentionPolicyManager,
    TierMigrationResult,
    get_retention_policy,
    reset_retention_policy,
)
from .schemas import (
    DEFAULT_METADATA_SCHEMA,  # Deprecated: use get_vocabulary() instead
    AssembledContext,
    ContextRequest,
    IngestRequest,
    KnowledgeVisibility,
    Message,
    MessageMetadata,
    MessageRole,
    MetadataSchema,  # Deprecated: use VocabularyManager instead
    ThreadSummary,
    UserPreferences,
)
from .sqlite_manager import SQLiteManager
from .vocabulary import (
    CommunicationStyle,
    EntityType,
    Intent,
    Sentiment,
    VocabularyEntry,
    VocabularyManager,
    VocabularySource,
    get_vocabulary,
    reset_vocabulary,
)
from .worker_monitor import (
    WorkerMetrics,
    WorkerMonitor,
    WorkerStatus,
    get_worker_monitor,
    reset_worker_monitor,
)


# Async database managers (lazy import to avoid requiring async deps)
def get_async_sqlite_manager():
    """Get AsyncSQLiteManager class (requires aiosqlite)."""
    from .async_db import AsyncSQLiteManager

    return AsyncSQLiteManager


def get_async_database_manager():
    """Get AsyncDatabaseManager class (requires asyncpg)."""
    from .async_db import AsyncDatabaseManager

    return AsyncDatabaseManager


# Access Control (lazy imports for optional Casbin)
def get_access_control_manager():
    """Get AccessControlManager class."""
    from .access_control import AccessControlManager

    return AccessControlManager


def get_casbin_access_control():
    """Get CasbinAccessControlManager class (requires: pip install casbin>=1.50.0).

    Example:
        CasbinACM = get_casbin_access_control()
        acm = CasbinACM.with_rbac()  # Built-in RBAC model
    """
    from .access_control import CasbinAccessControlManager

    return CasbinAccessControlManager


__all__ = [
    "DEFAULT_METADATA_SCHEMA",  # Deprecated
    "AdaptiveConfig",
    # Adaptive preferences
    "AdaptivePreferencesLearner",
    "AgentProfile",
    "AgentVisibility",
    "AssembledContext",
    "AsyncPreferencesManager",
    # Cache invalidation
    "CacheInvalidationManager",
    "CacheManager",
    "CommunicationStyle",
    "ConfigLoader",
    "ContextRequest",
    "DatabaseConnectionError",
    "DatabaseManager",
    "DiskCacheManager",
    "EntityType",
    "IngestRequest",
    "Intent",
    "InvalidationEvent",
    "InvalidationReason",
    "InvalidationStats",
    "KnowledgeVisibility",
    "MemorySharingMode",
    "MemoryTier",
    "Message",
    "MessageMetadata",
    "MessageRole",
    "MetadataSchema",  # Deprecated
    "MetricsManager",
    # Prometheus metrics
    "MindcoreMetrics",
    # Multi-agent
    "MultiAgentConfig",
    "MultiAgentManager",
    "PreferenceSignal",
    "PreferencesManager",
    "RetentionConfig",
    # Retention policy
    "RetentionPolicyManager",
    "SQLiteManager",
    "Sentiment",
    "ThreadSummary",
    "TierMigrationResult",
    "UserPreferences",
    "VocabularyEntry",
    # VocabularyManager (central vocabulary control)
    "VocabularyManager",
    "VocabularySource",
    "WorkerMetrics",
    # Worker monitoring
    "WorkerMonitor",
    "WorkerStatus",
    "configure_multi_agent",
    # Access control helpers
    "get_access_control_manager",
    "get_adaptive_learner",
    "get_async_database_manager",
    # Async helpers
    "get_async_sqlite_manager",
    "get_cache_invalidation",
    "get_casbin_access_control",
    "get_metrics",
    "get_multi_agent_manager",
    "get_retention_policy",
    "get_vocabulary",
    "get_worker_monitor",
    "is_prometheus_available",
    "reset_adaptive_learner",
    "reset_cache_invalidation",
    "reset_metrics",
    "reset_multi_agent_manager",
    "reset_retention_policy",
    "reset_vocabulary",
    "reset_worker_monitor",
    "start_metrics_server",
]
