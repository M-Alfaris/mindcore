"""Shared Vocabulary Layer (SVL) - Semantic spine of MindCore.

SVL provides a standardized vocabulary that ensures all memories have consistent
metadata and link cleanly into domain-specific tables. Every memory stored or
recalled passes through this vocabulary layer.

Components:
- Ontology: Core semantic definitions (message types, intents, temporal, emotional, etc.)
- Domains: Domain-specific vocabulary extensions (ecommerce, healthcare, etc.)
- Layer: Main SharedVocabularyLayer class

Example:
    from mindcore.v2.svl import SharedVocabularyLayer

    svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

    # Validate metadata
    is_valid, errors = svl.validate_metadata({
        "message_type": "query",
        "message_intent": "check_status",
        "temporal_qualifier": "current",
        "urgency": "high",
    })

    # Get JSON schema for LLM
    schema = svl.get_json_schema()
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

# Layer - Main interface
from .layer import (
    SVLSchema,
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
    # Layer
    "SVLSchema",
    "SharedVocabularyLayer",
    "DEFAULT_SVL",
]
