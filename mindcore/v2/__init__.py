"""Mindcore v2 - Universal Memory Layer for AI Agents.

A modern memory layer built on three protocols:
- FLR (Fast Learning Recall): Inference-time memory access
- CLST (Cognitive Long-term Storage Transfer): Durable storage
- SVL (Shared Vocabulary Layer): Unified semantic system

Features:
- Structured output integration (JSON Schema for LLMs)
- Direct memory storage from structured LLM output (no extraction overhead)
- Multi-agent support with access control
- Vocabulary versioning and migrations
- Data source mapping for context enrichment
- MCP and REST API interfaces

IMPORTANT: Direct Structured Output Required
-------------------------------------------
Your AI agent must output structured metadata directly. Configure your LLM
to return memories in a structured format, then store them directly:

    for mem in llm_response["memories_to_store"]:
        memory.store(
            content=mem["content"],
            memory_type=mem["memory_type"],
            user_id="user123",
            topics=mem.get("topics", []),
        )

Quick Start:
    from mindcore.v2 import Mindcore, SharedVocabularyLayer

    # Initialize
    memory = Mindcore()

    # Store directly from structured LLM output
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

    # Get JSON schema for your LLM
    schema = memory.get_json_schema()

    # SVL with data source mapping
    svl = SharedVocabularyLayer(domains=["customer_service"])
    svl.map_source("orders", TableSource(...))
"""

from .access import (
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
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
from .cross_agent import (
    Agent,
    AgentCapability,
    AgentRegistry,
    AgentStatus,
    AgentSyncDirection,
    AgentSyncResult,
    AttentionRouter,
    CrossAgentLayer,
    CrossAgentMemory,
    RouteResult,
    RoutingStrategy,
    ShareResult,
    Team,
)
from .flr import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)
from .mindcore import Mindcore
from .server import (
    MCPServer,
    create_app,
    run_server,
)
from .storage import (
    BaseStorage,
    PostgresStorage,
    SQLiteStorage,
)

# SVL - The unified vocabulary system (replaces VocabularySchema)
from .svl import (
    DEFAULT_SVL,
    DOMAIN_REGISTRY,
    AccessLevel,
    APISource,
    Confidence,
    DataSource,
    DomainLabel,
    # Domains
    DomainVocabulary,
    EmotionalClassification,
    FetchResult,
    FieldSchema,
    FunctionSource,
    MCPSource,
    # Core enums
    MemoryType,
    MessageIntent,
    # Ontology
    MessageType,
    # Migration support
    Migration,
    MigrationCheckpoint,
    PreferenceType,
    SemanticMetadata,
    Sentiment,
    SharedVocabularyLayer,
    SourceMapping,
    SourceRegistry,
    # Sources
    SourceType,
    # Layer
    SVLSchema,
    TableSource,
    TemporalQualifier,
    TriggerCondition,
    Urgency,
    UserRole,
    create_custom_domain,
    create_source,
    get_domain,
    list_domains,
    merge_domains,
)
from .svl import DEFAULT_SVL as DEFAULT_VOCABULARY

# Legacy VocabularySchema - DEPRECATED, use SharedVocabularyLayer instead
# These aliases are provided for backwards compatibility only
from .svl import SharedVocabularyLayer as VocabularySchema


# Backwards compatibility aliases for cross_agent sync types
# DEPRECATED: Use AgentSyncDirection and AgentSyncResult instead
CrossAgentSyncDirection = AgentSyncDirection
CrossAgentSyncResult = AgentSyncResult

# Exceptions - Standardized error handling
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


__version__ = "2.0.0"

# Enterprise features (optional dependencies)
# Import as: from mindcore.v2.enterprise import ...
# or: from mindcore.v2 import enterprise
try:
    from . import enterprise
except ImportError:
    enterprise = None  # Enterprise dependencies not installed

__all__ = [
    "CLST",
    "DEFAULT_SVL",
    "DEFAULT_VOCABULARY",
    "DOMAIN_REGISTRY",
    "FLR",
    "APISource",
    "AccessController",
    "AccessDecision",
    "AccessError",
    "AccessLevel",
    "Agent",
    "AgentCapability",
    "AgentNotFoundError",
    "AgentProfile",
    "AgentRegistry",
    "AgentStatus",
    "AgentSyncDirection",
    "AgentSyncResult",
    "AttentionRouter",
    "BaseStorage",
    "CompressionResult",
    "CompressionStrategy",
    "Confidence",
    "ConfigurationError",
    "ContextWindow",
    "CrossAgentLayer",
    "CrossAgentMemory",
    "CrossAgentSyncDirection",
    "CrossAgentSyncResult",
    "DataSource",
    "DomainLabel",
    "DomainVocabulary",
    "EmotionalClassification",
    "FetchResult",
    "FieldSchema",
    "FunctionSource",
    "MCPServer",
    "MCPSource",
    "Memory",
    "MemoryNotFoundError",
    "MemoryType",
    "MemoryValidationError",
    "MessageIntent",
    "MessageType",
    "Migration",
    "MigrationCheckpoint",
    "MigrationError",
    "MigrationPathError",
    "MigrationResult",
    "Mindcore",
    "MindcoreError",
    "MultiAgentNotEnabledError",
    "Permission",
    "PermissionDeniedError",
    "PostgresStorage",
    "PreferenceType",
    "RecallResult",
    "RollbackError",
    "RouteResult",
    "RoutingStrategy",
    "SQLiteStorage",
    "SVLSchema",
    "SemanticMetadata",
    "Sentiment",
    "ShareResult",
    "SharedVocabularyLayer",
    "SourceMapping",
    "SourceRegistry",
    "SourceType",
    "StorageConnectionError",
    "StorageError",
    "SyncDirection",
    "SyncResult",
    "TableSource",
    "Team",
    "TemporalQualifier",
    "TransferManifest",
    "TriggerCondition",
    "Urgency",
    "UserRole",
    "ValidationError",
    "VocabularySchema",
    "VocabularyValidationError",
    "create_app",
    "create_custom_domain",
    "create_source",
    "enterprise",
    "get_domain",
    "list_domains",
    "merge_domains",
    "run_server",
]
