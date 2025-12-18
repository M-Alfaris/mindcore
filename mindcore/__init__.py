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

# Main v2 exports
# Utils
from .utils import (
    LogCategory,
    configure_logging,
    get_logger,
)
from .v2 import (
    # CLST Protocol
    CLST,
    DEFAULT_SVL,
    DEFAULT_VOCABULARY,
    # FLR Protocol
    FLR,
    # Access Control
    AccessController,
    AccessDecision,
    AccessError,
    AccessLevel,
    AgentNotFoundError,
    AgentProfile,
    # Cross-Agent Sync (different from CLST sync)
    AgentSyncDirection,  # Agent-to-agent sync direction
    AgentSyncResult,  # Agent-to-agent sync result
    # Storage
    BaseStorage,
    CompressionResult,
    CompressionStrategy,
    ConfigurationError,
    ContextWindow,
    FieldSchema,
    # Server
    MCPServer,
    Memory,
    MemoryNotFoundError,
    MemoryType,
    MemoryValidationError,
    Migration,
    MigrationCheckpoint,
    MigrationError,
    MigrationPathError,
    # Main class
    Mindcore,
    # Exceptions
    MindcoreError,
    MultiAgentNotEnabledError,
    Permission,
    PermissionDeniedError,
    PostgresStorage,
    RecallResult,
    RollbackError,
    Sentiment,
    SharedVocabularyLayer,
    SQLiteStorage,
    StorageConnectionError,
    StorageError,
    SyncDirection,  # CLST sync direction (PUSH, PULL, BIDIRECTIONAL)
    SyncResult,  # CLST sync result
    TransferManifest,
    ValidationError,
    # Vocabulary (VocabularySchema is now an alias for SharedVocabularyLayer)
    VocabularySchema,
    VocabularyValidationError,
    create_app,
    run_server,
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
    "AgentSyncDirection",
    "AgentSyncResult",
    "BaseStorage",
    "CompressionResult",
    "CompressionStrategy",
    "ConfigurationError",
    "ContextWindow",
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
    "SQLiteStorage",
    "Sentiment",
    "SharedVocabularyLayer",
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
