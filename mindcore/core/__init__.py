"""
Core modules for Mindcore framework.
"""
from .config_loader import ConfigLoader
from .db_manager import DatabaseManager, DatabaseConnectionError
from .sqlite_manager import SQLiteManager
from .cache_manager import CacheManager
from .disk_cache_manager import DiskCacheManager
from .metrics_manager import MetricsManager
from .schemas import (
    Message,
    MessageMetadata,
    MessageRole,
    AssembledContext,
    ContextRequest,
    IngestRequest,
    MetadataSchema,
    DEFAULT_METADATA_SCHEMA,
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

__all__ = [
    "ConfigLoader",
    "DatabaseManager",
    "DatabaseConnectionError",
    "SQLiteManager",
    "CacheManager",
    "DiskCacheManager",
    "MetricsManager",
    "Message",
    "MessageMetadata",
    "MessageRole",
    "AssembledContext",
    "ContextRequest",
    "IngestRequest",
    "MetadataSchema",
    "DEFAULT_METADATA_SCHEMA",
    # Async helpers
    "get_async_sqlite_manager",
    "get_async_database_manager",
]
