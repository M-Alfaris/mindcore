"""Storage backends for Mindcore v2.

PostgreSQL is the primary production backend.
SQLite is for development/testing only.

Search Extensions:
    For enhanced search (pg_trgm, ParadeDB BM25), see schema/ folder.
    Configure with SearchConfig class.
"""

from .base import BaseStorage
from .config import (
    SEARCH_CONFIG_DEFAULT,
    SEARCH_CONFIG_RECENCY_FOCUSED,
    SEARCH_CONFIG_REINFORCEMENT_FOCUSED,
    SEARCH_CONFIG_TOPIC_FOCUSED,
    SearchConfig,
)
from .partitioning import (
    PartitionInfo,
    PartitioningStatus,
    PartitionInterval,
    PartitionManager,
)
from .postgres import PostgresStorage
from .sqlite import SQLiteStorage


__all__ = [
    "SEARCH_CONFIG_DEFAULT",
    "SEARCH_CONFIG_RECENCY_FOCUSED",
    "SEARCH_CONFIG_REINFORCEMENT_FOCUSED",
    "SEARCH_CONFIG_TOPIC_FOCUSED",
    "BaseStorage",
    "PartitionInfo",
    "PartitionInterval",
    "PartitionManager",
    "PartitioningStatus",
    "PostgresStorage",
    "SQLiteStorage",
    "SearchConfig",
]
