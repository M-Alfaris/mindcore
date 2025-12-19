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
from .v2 import (
    # Main class
    Mindcore,
    # FLR Protocol
    FLR,
    Memory,
    RecallResult,
    ContextWindow,
    # CLST Protocol
    CLST,
    CompressionResult,
    CompressionStrategy,
    SyncResult,
    TransferManifest,
    # Vocabulary
    VocabularySchema,
    DEFAULT_VOCABULARY,
    MemoryType,
    Sentiment,
    AccessLevel,
    FieldSchema,
    Migration,
    # Access Control
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
    # Storage
    BaseStorage,
    PostgresStorage,
    SQLiteStorage,
    # Server
    MCPServer,
    create_app,
    run_server,
)

# Utils
from .utils import (
    LogCategory,
    configure_logging,
    get_logger,
)


__all__ = [
    # Version
    "__version__",
    # Main
    "Mindcore",
    # FLR
    "FLR",
    "Memory",
    "RecallResult",
    "ContextWindow",
    # CLST
    "CLST",
    "CompressionResult",
    "CompressionStrategy",
    "SyncResult",
    "TransferManifest",
    # Vocabulary
    "VocabularySchema",
    "DEFAULT_VOCABULARY",
    "MemoryType",
    "Sentiment",
    "AccessLevel",
    "FieldSchema",
    "Migration",
    # Access Control
    "AccessController",
    "AccessDecision",
    "AgentProfile",
    "Permission",
    # Storage
    "BaseStorage",
    "PostgresStorage",
    "SQLiteStorage",
    # Server
    "MCPServer",
    "create_app",
    "run_server",
    # Utils
    "LogCategory",
    "configure_logging",
    "get_logger",
]
