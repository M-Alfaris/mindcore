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
    MetadataSchema,
    DEFAULT_METADATA_SCHEMA,
    ThreadSummary,
    UserPreferences,
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
    "MetadataSchema",
    "DEFAULT_METADATA_SCHEMA",
    "ThreadSummary",
    "UserPreferences",
    # Async helpers
    "get_async_sqlite_manager",
    "get_async_database_manager",
    # Access control helpers
    "get_access_control_manager",
    "get_casbin_access_control",
]
