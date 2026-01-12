"""Mindcore: Memory Layer for AI Agents.

A structured memory system with vocabulary-controlled metadata,
fast retrieval (FLR), and long-term storage (CLST) protocols.

Quick Start:
------------
    from mindcore import Mindcore

    # Initialize with PostgreSQL (production) or SQLite (development)
    memory = Mindcore(storage="postgresql://localhost/mindcore")
    # or
    memory = Mindcore(storage="sqlite:///dev.db")

    # Store a memory directly from structured LLM output
    for mem in llm_response["memories_to_store"]:
        memory.store(
            content=mem["content"],
            memory_type=mem["memory_type"],
            user_id="user123",
            topics=mem.get("topics", []),
        )

    # Recall relevant memories
    result = memory.recall(
        query="user preferences",
        user_id="user123",
    )

    # Get JSON schema for LLM structured output configuration
    schema = memory.get_json_schema()

Features:
---------
- FLR (Fast Learning Recall): Hot path for inference-time memory access
- CLST (Cognitive Long-term Storage Transfer): Cold path for persistence
- Vocabulary-controlled metadata with versioning
- Direct memory storage from structured LLM output (no extraction overhead)
- Multi-agent support with access control
- PostgreSQL (production) and SQLite (development) backends
- MCP and REST API servers

IMPORTANT: Your AI agent must output structured metadata directly.
Use get_json_schema() to configure your LLM's structured output format.

See MINDCORE.md for complete documentation.
"""

__version__ = "2.0.0"
__author__ = "Mindcore Contributors"
__license__ = "MIT"

# Utils
# Access Control
from .access import (
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
)

# CLST Protocol
from .clst import (
    CLST,
    CompressionResult,
    CompressionStrategy,
    SyncDirection,  # CLST sync direction (PUSH, PULL, BIDIRECTIONAL)
    SyncResult,  # CLST sync result
    TransferManifest,
)

# Cross-Agent
from .cross_agent import (
    AgentStatus,
    AgentSyncDirection,  # Agent-to-agent sync direction
    AgentSyncResult,  # Agent-to-agent sync result
    CrossAgentLayer,
    RoutingStrategy,
)

# Exceptions
from .exceptions import (
    AccessError,
    AgentNotFoundError,
    ConfigurationError,
    MemoryNotFoundError,
    MemoryValidationError,
    MigrationError,
    MigrationPathError,
    MindcoreError,
    MultiAgentNotEnabledError,
    PermissionDeniedError,
    RollbackError,
    StorageConnectionError,
    StorageError,
    ValidationError,
    VocabularyValidationError,
)

# FLR Protocol
from .flr import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)

# Main class
from .mindcore import Mindcore

# Server
from .server import (
    MCPServer,
    create_app,
    run_server,
)

# Storage
from .storage import (
    BaseStorage,
    PostgresStorage,
    SQLiteStorage,
)

# SVL (Structured Validation Layer)
from .svl import (
    DEFAULT_SVL,
    AccessLevel,
    FieldSchema,
    MemoryType,
    Migration,
    MigrationCheckpoint,
    Sentiment,
    StructuredValidationLayer,
    SharedVocabularyLayer,  # Backwards compatibility alias
)

# Legacy alias
from .svl import DEFAULT_SVL as DEFAULT_VOCABULARY
from .svl import StructuredValidationLayer as VocabularySchema
from .utils import (
    LogCategory,
    configure_logging,
    get_logger,
)


__all__ = [
    "CLST",
    "DEFAULT_SVL",
    "DEFAULT_VOCABULARY",
    "FLR",
    "AccessController",
    "AccessDecision",
    "AccessError",
    "AccessLevel",
    "AgentNotFoundError",
    "AgentProfile",
    "AgentStatus",
    "AgentSyncDirection",
    "AgentSyncResult",
    "BaseStorage",
    "CompressionResult",
    "CompressionStrategy",
    "ConfigurationError",
    "ContextWindow",
    "CrossAgentLayer",
    "FieldSchema",
    "LogCategory",
    "MCPServer",
    "Memory",
    "MemoryNotFoundError",
    "MemoryType",
    "MemoryValidationError",
    "Migration",
    "MigrationCheckpoint",
    "MigrationError",
    "MigrationPathError",
    "Mindcore",
    "MindcoreError",
    "MultiAgentNotEnabledError",
    "Permission",
    "PermissionDeniedError",
    "PostgresStorage",
    "RecallResult",
    "RollbackError",
    "RoutingStrategy",
    "SQLiteStorage",
    "Sentiment",
    "StructuredValidationLayer",
    "SharedVocabularyLayer",  # Backwards compatibility alias
    "StorageConnectionError",
    "StorageError",
    "SyncDirection",
    "SyncResult",
    "TransferManifest",
    "ValidationError",
    "VocabularySchema",
    "VocabularyValidationError",
    "__version__",
    "configure_logging",
    "create_app",
    "get_logger",
    "run_server",
]
