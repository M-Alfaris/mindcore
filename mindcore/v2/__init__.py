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

from .mindcore import Mindcore

# SVL - The unified vocabulary system (replaces VocabularySchema)
from .svl import (
    # Core enums
    MemoryType,
    Sentiment,
    AccessLevel,
    # Migration support
    Migration,
    FieldSchema,
    # Ontology
    MessageType,
    MessageIntent,
    TemporalQualifier,
    EmotionalClassification,
    UserRole,
    PreferenceType,
    DomainLabel,
    Urgency,
    Confidence,
    SemanticMetadata,
    # Domains
    DomainVocabulary,
    DOMAIN_REGISTRY,
    get_domain,
    list_domains,
    merge_domains,
    create_custom_domain,
    # Sources
    SourceType,
    TriggerCondition,
    DataSource,
    FetchResult,
    TableSource,
    APISource,
    MCPSource,
    FunctionSource,
    SourceMapping,
    SourceRegistry,
    create_source,
    # Layer
    SVLSchema,
    SharedVocabularyLayer,
    DEFAULT_SVL,
)

# Legacy VocabularySchema - kept for backwards compatibility
from .vocabulary import (
    VocabularySchema,
    DEFAULT_VOCABULARY,
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
from .cross_agent import (
    Agent,
    AgentCapability,
    AgentRegistry,
    AgentStatus,
    AttentionRouter,
    CrossAgentLayer,
    CrossAgentMemory,
    RouteResult,
    RoutingStrategy,
    ShareResult,
    SyncDirection as CrossAgentSyncDirection,
    SyncResult as CrossAgentSyncResult,
    Team,
)

__version__ = "2.0.0"

__all__ = [
    # Main class
    "Mindcore",
    # SVL - Unified Vocabulary (primary)
    "MemoryType",
    "Sentiment",
    "AccessLevel",
    "Migration",
    "FieldSchema",
    "MessageType",
    "MessageIntent",
    "TemporalQualifier",
    "EmotionalClassification",
    "UserRole",
    "PreferenceType",
    "DomainLabel",
    "Urgency",
    "Confidence",
    "SemanticMetadata",
    "DomainVocabulary",
    "DOMAIN_REGISTRY",
    "get_domain",
    "list_domains",
    "merge_domains",
    "create_custom_domain",
    "SourceType",
    "TriggerCondition",
    "DataSource",
    "FetchResult",
    "TableSource",
    "APISource",
    "MCPSource",
    "FunctionSource",
    "SourceMapping",
    "SourceRegistry",
    "create_source",
    "SVLSchema",
    "SharedVocabularyLayer",
    "DEFAULT_SVL",
    # Legacy Vocabulary (backwards compatibility)
    "VocabularySchema",
    "DEFAULT_VOCABULARY",
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
    # Cross-Agent
    "Agent",
    "AgentCapability",
    "AgentRegistry",
    "AgentStatus",
    "AttentionRouter",
    "CrossAgentLayer",
    "CrossAgentMemory",
    "CrossAgentSyncDirection",
    "CrossAgentSyncResult",
    "RouteResult",
    "RoutingStrategy",
    "ShareResult",
    "Team",
]
