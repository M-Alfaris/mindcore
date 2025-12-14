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
    from mindcore.v2.svl import SharedVocabularyLayer, TableSource

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
from .ontology import (
    # Message types
    MessageType,
    MessageIntent,
    # Temporal
    TemporalQualifier,
    # Emotional
    EmotionalClassification,
    # User context
    UserRole,
    PreferenceType,
    # Domain
    DomainLabel,
    # Quality
    Urgency,
    Confidence,
    # Container
    SemanticMetadata,
    # Helpers
    get_message_types,
    get_message_intents,
    get_temporal_qualifiers,
    get_emotional_classifications,
    get_user_roles,
    get_preference_types,
    get_domain_labels,
    get_urgency_levels,
    get_confidence_levels,
)

# Domains - Domain-specific vocabulary
from .domains import (
    DomainVocabulary,
    # Pre-built domains
    CUSTOMER_SERVICE_DOMAIN,
    ECOMMERCE_DOMAIN,
    HEALTHCARE_DOMAIN,
    FINANCE_DOMAIN,
    SAAS_DOMAIN,
    HR_DOMAIN,
    EDUCATION_DOMAIN,
    DOMAIN_REGISTRY,
    # Functions
    get_domain,
    list_domains,
    merge_domains,
    create_custom_domain,
)

# Sources - Data source mapping
from .sources import (
    # Source types
    SourceType,
    TriggerCondition,
    DataSource,
    FetchResult,
    # Concrete sources
    TableSource,
    APISource,
    MCPSource,
    FunctionSource,
    # Mapping
    SourceMapping,
    SourceRegistry,
    # Factory
    create_source,
)

# Layer - Main interface (unified vocabulary)
from .layer import (
    # Enums (from VocabularySchema)
    MemoryType,
    Sentiment,
    AccessLevel,
    # Migration
    Migration,
    FieldSchema,
    # Schema
    SVLSchema,
    # Main class
    SharedVocabularyLayer,
    DEFAULT_SVL,
)

__all__ = [
    # Ontology
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
    "get_message_types",
    "get_message_intents",
    "get_temporal_qualifiers",
    "get_emotional_classifications",
    "get_user_roles",
    "get_preference_types",
    "get_domain_labels",
    "get_urgency_levels",
    "get_confidence_levels",
    # Domains
    "DomainVocabulary",
    "CUSTOMER_SERVICE_DOMAIN",
    "ECOMMERCE_DOMAIN",
    "HEALTHCARE_DOMAIN",
    "FINANCE_DOMAIN",
    "SAAS_DOMAIN",
    "HR_DOMAIN",
    "EDUCATION_DOMAIN",
    "DOMAIN_REGISTRY",
    "get_domain",
    "list_domains",
    "merge_domains",
    "create_custom_domain",
    # Sources
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
    # Layer (unified vocabulary)
    "MemoryType",
    "Sentiment",
    "AccessLevel",
    "Migration",
    "FieldSchema",
    "SVLSchema",
    "SharedVocabularyLayer",
    "DEFAULT_SVL",
]
