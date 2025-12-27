"""Shared Vocabulary Layer (SVL) - The unified semantic system for MindCore.

SVL is the single vocabulary system for MindCore, replacing VocabularySchema.
It provides:
1. Universal ontology (message types, intents, temporal, emotional, etc.)
2. Domain-specific vocabulary extensions
3. Memory types, sentiments, access levels
4. Vocabulary versioning and migrations
5. JSON Schema, Pydantic, TypeScript code generation
6. Data source mapping for automatic context enrichment

Example:
    from mindcore.svl import SharedVocabularyLayer, TableSource

    svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

    # Map topics to data sources
    svl.map_source("orders", TableSource(
        name="orders_db",
        connection_string="postgresql://...",
        query_template="SELECT * FROM orders WHERE user_id = :user_id",
        param_mapping={"user_id": "user_id"},
    ))

    # Validate memory
    is_valid, errors = svl.validate_memory({
        "content": "...",
        "memory_type": "semantic",
        "topics": ["orders"],
    })

    # Fetch data for topics
    data = svl.fetch_for_topics(["orders"], context={"user_id": "123"})
"""

# Ontology - Core semantic types
# Domains - Domain-specific vocabulary
# Defaults - Auto-configuration with sensible defaults
from .defaults import (
    CRM_TOPICS,
    DEFAULT_CONFIG,
    # Presets
    ECOMMERCE_TOPICS,
    SUPPORT_TOPICS,
    # Configuration
    DefaultSourceConfig,
    NamingConvention,
    ParamPattern,
    # Smart sources
    SmartTableSource,
    auto_configure_registry,
    create_preset_sources,
    # Auto-configuration
    create_smart_sources,
    discover_tables,
    get_preset_topics,
    # Quick setup
    quick_setup_database,
)
from .domains import (
    # Pre-built domains
    CUSTOMER_SERVICE_DOMAIN,
    DOMAIN_REGISTRY,
    ECOMMERCE_DOMAIN,
    EDUCATION_DOMAIN,
    FINANCE_DOMAIN,
    HEALTHCARE_DOMAIN,
    HR_DOMAIN,
    SAAS_DOMAIN,
    DomainVocabulary,
    create_custom_domain,
    # Functions
    get_domain,
    list_domains,
    merge_domains,
)

# Enforced Metadata - LLM metadata extraction
from .enforced_metadata import (
    ContextDecision,
    EnforcedMetadata,
    HistoricalContextNeeded,
    MetadataExtractor,
)

# Extraction Fallback Strategies
from .extraction_fallback import (
    BatchItem,
    ExtractionAttempt,
    ExtractionFailureType,
    ExtractionResult,
    ExtractionStrategy,
    ResilientMetadataExtractor,
    RuleBasedExtractor,
)

# SVL Gate - The Kernel (mandatory choke point)
from .gate import (
    GateDecision,
    GatePolicy,
    GateResult,
    PolicyViolation,
    RetryConfig,
    SVLGate,
    ValidationError,
)

# Gated Storage - Mandatory gate enforcement wrappers
from .gated_storage import (
    GatedCLST,
    GatedFLR,
    GatedMindcore,
    StoreResult,
)
from .gated_storage import (
    RecallResult as GatedRecallResult,
)

# Layer - Main interface (unified vocabulary)
from .layer import (
    DEFAULT_SVL,
    AccessLevel,
    FieldSchema,
    # Enums (from VocabularySchema)
    MemoryType,
    # Migration
    Migration,
    MigrationCheckpoint,
    Sentiment,
    # Main class
    SharedVocabularyLayer,
    # Schema
    SVLSchema,
)

# LLM Provider Configurations
from .llm_providers import (
    ClaudeConfig,
    # API-Level Context Injection
    ContextInjector,
    FeedbackInjection,
    GeminiConfig,
    GenericConfig,
    LLMProviderConfig,
    OpenAIConfig,
    ReasoningEffort,
    ThinkingLevel,  # Gemini 3 thinking levels
    ThinkingMode,  # Gemini 2.5 thinking modes
    create_injector_from_flr,
    create_injector_from_optimizer,
    get_provider_config,
    get_recommended_config,
)
from .ontology import (
    Confidence,
    # Domain
    DomainLabel,
    # Emotional
    EmotionalClassification,
    MessageIntent,
    # Message roles (LLM conversation roles)
    MessageRole,
    # Message types
    MessageType,
    PreferenceType,
    # Container
    SemanticMetadata,
    # Temporal
    TemporalQualifier,
    # Quality
    Urgency,
    # User context
    UserRole,
    get_confidence_levels,
    get_domain_labels,
    get_emotional_classifications,
    get_message_intents,
    # Helpers
    get_message_roles,
    get_message_types,
    get_preference_types,
    get_temporal_qualifiers,
    get_urgency_levels,
    get_user_roles,
)

# SVL Pipeline - Complete orchestrated data flow
from .pipeline import (
    QueryResult,
    SVLPipeline,
    create_pipeline,
)
from .pipeline import (
    StoreResult as PipelineStoreResult,
)

# Registry - Unified source configuration
from .registry import (
    AsyncFunctionSource,
    AsyncSourceExecutor,
    SourceDefinition,
    SourceDiscovery,
    clear_registered_sources,
    discover_and_register,
    get_registered_sources,
    load_sources_from_json,
    load_sources_from_yaml,
    # Decorator
    source,
)

# Sources - Data source mapping
from .sources import (
    APISource,
    DataSource,
    FetchResult,
    FunctionSource,
    MCPSource,
    # Mapping
    SourceMapping,
    SourceRegistry,
    # Source types
    SourceType,
    # Concrete sources
    TableSource,
    TriggerCondition,
    # Factory
    create_source,
)


__all__ = [
    "CUSTOMER_SERVICE_DOMAIN",
    "DEFAULT_SVL",
    "DOMAIN_REGISTRY",
    "ECOMMERCE_DOMAIN",
    "EDUCATION_DOMAIN",
    "FINANCE_DOMAIN",
    "HEALTHCARE_DOMAIN",
    "HR_DOMAIN",
    "SAAS_DOMAIN",
    "APISource",
    "AccessLevel",
    "AsyncFunctionSource",
    "AsyncSourceExecutor",
    "BatchItem",
    "ClaudeConfig",
    # SVL Gate exports
    "GateDecision",
    "GatePolicy",
    "GateResult",
    "PolicyViolation",
    "RetryConfig",
    "SVLGate",
    "ValidationError",
    # Gated Storage exports
    "GatedCLST",
    "GatedFLR",
    "GatedMindcore",
    "GatedRecallResult",
    "StoreResult",
    # SVL Pipeline exports
    "QueryResult",
    "SVLPipeline",
    "PipelineStoreResult",
    "create_pipeline",
    "Confidence",
    "ContextDecision",
    "ContextInjector",
    "DataSource",
    "DomainLabel",
    "DomainVocabulary",
    "EmotionalClassification",
    "EnforcedMetadata",
    "ExtractionAttempt",
    "ExtractionFailureType",
    "ExtractionResult",
    "ExtractionStrategy",
    "FeedbackInjection",
    "FetchResult",
    "FieldSchema",
    "FunctionSource",
    "GeminiConfig",
    "GenericConfig",
    "HistoricalContextNeeded",
    "LLMProviderConfig",
    "MCPSource",
    "MemoryType",
    "MessageIntent",
    "MessageRole",
    "MessageType",
    "MetadataExtractor",
    "Migration",
    "MigrationCheckpoint",
    "OpenAIConfig",
    "PreferenceType",
    "ReasoningEffort",
    "ResilientMetadataExtractor",
    "RuleBasedExtractor",
    "SVLSchema",
    "SemanticMetadata",
    "Sentiment",
    "SharedVocabularyLayer",
    "SourceDefinition",
    "SourceDiscovery",
    "SourceMapping",
    "SourceRegistry",
    "SourceType",
    "TableSource",
    "TemporalQualifier",
    "ThinkingLevel",
    "ThinkingMode",
    "TriggerCondition",
    "Urgency",
    "UserRole",
    "clear_registered_sources",
    "create_custom_domain",
    "create_injector_from_flr",
    "create_injector_from_optimizer",
    "create_source",
    "discover_and_register",
    "get_confidence_levels",
    "get_domain",
    "get_domain_labels",
    "get_emotional_classifications",
    "get_message_intents",
    "get_message_roles",
    "get_message_types",
    "get_preference_types",
    "get_provider_config",
    "get_recommended_config",
    "get_registered_sources",
    "get_temporal_qualifiers",
    "get_urgency_levels",
    "get_user_roles",
    "list_domains",
    "load_sources_from_json",
    "load_sources_from_yaml",
    "merge_domains",
    "source",
    # Defaults - Auto-configuration
    "DefaultSourceConfig",
    "DEFAULT_CONFIG",
    "NamingConvention",
    "ParamPattern",
    "SmartTableSource",
    "create_smart_sources",
    "auto_configure_registry",
    "discover_tables",
    "ECOMMERCE_TOPICS",
    "CRM_TOPICS",
    "SUPPORT_TOPICS",
    "get_preset_topics",
    "create_preset_sources",
    "quick_setup_database",
]
