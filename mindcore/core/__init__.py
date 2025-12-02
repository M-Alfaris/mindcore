"""
Core modules for Mindcore framework.
"""

from .config_loader import ConfigLoader
from .db_manager import DatabaseManager, DatabaseConnectionError
from .sqlite_manager import SQLiteManager
from .cache_manager import CacheManager
from .disk_cache_manager import DiskCacheManager
from .metrics_manager import MetricsManager
from .preferences_manager import PreferencesManager, AsyncPreferencesManager
from .schemas import (
    Message,
    MessageMetadata,
    MessageRole,
    KnowledgeVisibility,
    AssembledContext,
    ContextRequest,
    IngestRequest,
    MetadataSchema,  # Deprecated: use VocabularyManager instead
    DEFAULT_METADATA_SCHEMA,  # Deprecated: use get_vocabulary() instead
    ThreadSummary,
    UserPreferences,
)
from .vocabulary import (
    VocabularyManager,
    VocabularySource,
    VocabularyEntry,
    Intent,
    Sentiment,
    CommunicationStyle,
    EntityType,
    get_vocabulary,
    reset_vocabulary,
)
from .worker_monitor import (
    WorkerMonitor,
    WorkerMetrics,
    WorkerStatus,
    get_worker_monitor,
    reset_worker_monitor,
)
from .adaptive_preferences import (
    AdaptivePreferencesLearner,
    AdaptiveConfig,
    PreferenceSignal,
    get_adaptive_learner,
    reset_adaptive_learner,
)
from .retention_policy import (
    RetentionPolicyManager,
    RetentionConfig,
    MemoryTier,
    TierMigrationResult,
    get_retention_policy,
    reset_retention_policy,
)
from .cache_invalidation import (
    CacheInvalidationManager,
    InvalidationReason,
    InvalidationEvent,
    InvalidationStats,
    get_cache_invalidation,
    reset_cache_invalidation,
)
from .multi_agent import (
    MultiAgentConfig,
    MultiAgentManager,
    MemorySharingMode,
    AgentVisibility,
    AgentProfile,
    get_multi_agent_manager,
    reset_multi_agent_manager,
    configure_multi_agent,
)
from .prometheus_metrics import (
    MindcoreMetrics,
    get_metrics,
    reset_metrics,
    start_metrics_server,
    is_prometheus_available,
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
    """
    Get CasbinAccessControlManager class (requires: pip install casbin>=1.50.0).

    Example:
        CasbinACM = get_casbin_access_control()
        acm = CasbinACM.with_rbac()  # Built-in RBAC model
    """
    from .access_control import CasbinAccessControlManager

    return CasbinAccessControlManager


__all__ = [
    "ConfigLoader",
    "DatabaseManager",
    "DatabaseConnectionError",
    "SQLiteManager",
    "CacheManager",
    "DiskCacheManager",
    "MetricsManager",
    "PreferencesManager",
    "AsyncPreferencesManager",
    "Message",
    "MessageMetadata",
    "MessageRole",
    "KnowledgeVisibility",
    "AssembledContext",
    "ContextRequest",
    "IngestRequest",
    "MetadataSchema",  # Deprecated
    "DEFAULT_METADATA_SCHEMA",  # Deprecated
    "ThreadSummary",
    "UserPreferences",
    # VocabularyManager (central vocabulary control)
    "VocabularyManager",
    "VocabularySource",
    "VocabularyEntry",
    "Intent",
    "Sentiment",
    "CommunicationStyle",
    "EntityType",
    "get_vocabulary",
    "reset_vocabulary",
    # Worker monitoring
    "WorkerMonitor",
    "WorkerMetrics",
    "WorkerStatus",
    "get_worker_monitor",
    "reset_worker_monitor",
    # Adaptive preferences
    "AdaptivePreferencesLearner",
    "AdaptiveConfig",
    "PreferenceSignal",
    "get_adaptive_learner",
    "reset_adaptive_learner",
    # Retention policy
    "RetentionPolicyManager",
    "RetentionConfig",
    "MemoryTier",
    "TierMigrationResult",
    "get_retention_policy",
    "reset_retention_policy",
    # Cache invalidation
    "CacheInvalidationManager",
    "InvalidationReason",
    "InvalidationEvent",
    "InvalidationStats",
    "get_cache_invalidation",
    "reset_cache_invalidation",
    # Multi-agent
    "MultiAgentConfig",
    "MultiAgentManager",
    "MemorySharingMode",
    "AgentVisibility",
    "AgentProfile",
    "get_multi_agent_manager",
    "reset_multi_agent_manager",
    "configure_multi_agent",
    # Prometheus metrics
    "MindcoreMetrics",
    "get_metrics",
    "reset_metrics",
    "start_metrics_server",
    "is_prometheus_available",
    # Async helpers
    "get_async_sqlite_manager",
    "get_async_database_manager",
    # Access control helpers
    "get_access_control_manager",
    "get_casbin_access_control",
]
