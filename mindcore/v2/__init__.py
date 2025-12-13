"""Mindcore v2 - Universal Memory Layer for AI Agents.

A modern memory layer built on two protocols:
- FLR (Fast Learning Recall): Inference-time memory access
- CLST (Cognitive Long-term Storage Transfer): Durable storage

Features:
- Structured output integration (JSON Schema for LLMs)
- Auto-extraction of memories from conversations
- Multi-agent support with access control
- Vocabulary versioning and migrations
- MCP and REST API interfaces

Quick Start:
    from mindcore.v2 import Mindcore

    # Initialize
    memory = Mindcore()

    # Store
    memory.store(
        content="User prefers dark mode",
        memory_type="preference",
        user_id="user123",
        topics=["settings"],
    )

    # Recall
    result = memory.recall(
        query="What are the user's preferences?",
        user_id="user123",
    )

    # Get schema for LLM structured output
    schema = memory.get_json_schema()
"""

from .mindcore import Mindcore
from .vocabulary import (
    AccessLevel,
    DEFAULT_VOCABULARY,
    FieldSchema,
    MemoryType,
    Migration,
    Sentiment,
    VocabularySchema,
)
from .flr import (
    ContextWindow,
    FLR,
    Memory,
    RecallResult,
)
from .clst import (
    CLST,
    CompressionResult,
    CompressionStrategy,
    MigrationResult,
    SyncDirection,
    SyncResult,
    TransferManifest,
)
from .extraction import (
    ExtractionResult,
    MemoryExtractor,
)
from .access import (
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
)
from .storage import (
    BaseStorage,
    PostgresStorage,
    SQLiteStorage,
)
from .server import (
    MCPServer,
    create_app,
    run_server,
)

__version__ = "2.0.0"

__all__ = [
    # Main class
    "Mindcore",
    # Vocabulary
    "AccessLevel",
    "DEFAULT_VOCABULARY",
    "FieldSchema",
    "MemoryType",
    "Migration",
    "Sentiment",
    "VocabularySchema",
    # FLR
    "ContextWindow",
    "FLR",
    "Memory",
    "RecallResult",
    # CLST
    "CLST",
    "CompressionResult",
    "CompressionStrategy",
    "MigrationResult",
    "SyncDirection",
    "SyncResult",
    "TransferManifest",
    # Extraction
    "ExtractionResult",
    "MemoryExtractor",
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
]
