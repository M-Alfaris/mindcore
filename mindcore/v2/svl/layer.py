"""SharedVocabularyLayer - The semantic spine of MindCore.

SVL provides a standardized vocabulary that ensures all memories have consistent
metadata and link cleanly into domain-specific tables. Every memory stored or
recalled passes through this vocabulary layer.

SVL integrates with VocabularySchema to provide:
1. Universal ontology (message types, intents, temporal, emotional, etc.)
2. Domain-specific vocabulary extensions
3. Consistent metadata format validation
4. JSON Schema generation for LLM structured output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ontology import (
    Confidence,
    DomainLabel,
    EmotionalClassification,
    MessageIntent,
    MessageType,
    PreferenceType,
    SemanticMetadata,
    TemporalQualifier,
    Urgency,
    UserRole,
    get_confidence_levels,
    get_domain_labels,
    get_emotional_classifications,
    get_message_intents,
    get_message_types,
    get_preference_types,
    get_temporal_qualifiers,
    get_urgency_levels,
    get_user_roles,
)
from .domains import (
    DOMAIN_REGISTRY,
    DomainVocabulary,
    get_domain,
    list_domains,
    merge_domains,
)


@dataclass
class SVLSchema:
    """Complete SVL schema combining ontology and domain vocabulary.

    This is the full semantic schema that can be used to:
    - Validate memory metadata
    - Generate JSON Schema for LLM structured output
    - Ensure consistent tagging across all memories
    """

    version: str = "1.0.0"

    # Base topics and categories (can be extended by domains)
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    subcategories: dict[str, list[str]] = field(default_factory=dict)

    # Active domains
    domains: list[str] = field(default_factory=list)

    # Ontology toggles (which SVL fields to enable)
    enable_message_types: bool = True
    enable_message_intents: bool = True
    enable_temporal: bool = True
    enable_emotional: bool = True
    enable_user_roles: bool = True
    enable_preference_types: bool = True
    enable_domain_labels: bool = True
    enable_urgency: bool = True
    enable_confidence: bool = True

    # Custom extensions
    custom_message_types: list[str] = field(default_factory=list)
    custom_intents: list[str] = field(default_factory=list)
    custom_temporal: list[str] = field(default_factory=list)
    custom_emotional: list[str] = field(default_factory=list)
    custom_user_roles: list[str] = field(default_factory=list)
    custom_preference_types: list[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_message_types(self) -> list[str]:
        """Get all valid message types."""
        if not self.enable_message_types:
            return []
        return get_message_types() + self.custom_message_types

    def get_message_intents(self) -> list[str]:
        """Get all valid message intents."""
        if not self.enable_message_intents:
            return []
        base = get_message_intents() + self.custom_intents

        # Add domain-specific intents
        for domain_name in self.domains:
            domain = get_domain(domain_name)
            if domain:
                base.extend(domain.intents)

        return list(set(base))

    def get_temporal_qualifiers(self) -> list[str]:
        """Get all valid temporal qualifiers."""
        if not self.enable_temporal:
            return []
        return get_temporal_qualifiers() + self.custom_temporal

    def get_emotional_classifications(self) -> list[str]:
        """Get all valid emotional classifications."""
        if not self.enable_emotional:
            return []
        return get_emotional_classifications() + self.custom_emotional

    def get_user_roles(self) -> list[str]:
        """Get all valid user roles."""
        if not self.enable_user_roles:
            return []
        return get_user_roles() + self.custom_user_roles

    def get_preference_types(self) -> list[str]:
        """Get all valid preference types."""
        if not self.enable_preference_types:
            return []
        return get_preference_types() + self.custom_preference_types

    def get_domain_labels(self) -> list[str]:
        """Get all valid domain labels."""
        if not self.enable_domain_labels:
            return []
        return get_domain_labels()

    def get_urgency_levels(self) -> list[str]:
        """Get all valid urgency levels."""
        if not self.enable_urgency:
            return []
        return get_urgency_levels()

    def get_confidence_levels(self) -> list[str]:
        """Get all valid confidence levels."""
        if not self.enable_confidence:
            return []
        return get_confidence_levels()

    def get_all_topics(self) -> list[str]:
        """Get all topics including domain-specific ones."""
        all_topics = list(self.topics)

        for domain_name in self.domains:
            domain = get_domain(domain_name)
            if domain:
                all_topics.extend(domain.topics)

        return list(set(all_topics))

    def get_all_categories(self) -> list[str]:
        """Get all categories including domain-specific ones."""
        all_categories = list(self.categories)

        for domain_name in self.domains:
            domain = get_domain(domain_name)
            if domain:
                all_categories.extend(domain.categories)

        return list(set(all_categories))

    def get_all_subcategories(self) -> dict[str, list[str]]:
        """Get all subcategories including domain-specific ones."""
        all_subs = dict(self.subcategories)

        for domain_name in self.domains:
            domain = get_domain(domain_name)
            if domain:
                for cat, subs in domain.subcategories.items():
                    if cat in all_subs:
                        all_subs[cat] = list(set(all_subs[cat] + subs))
                    else:
                        all_subs[cat] = subs

        return all_subs

    def get_entity_types(self) -> list[str]:
        """Get all entity types from active domains."""
        entity_types = []
        for domain_name in self.domains:
            domain = get_domain(domain_name)
            if domain:
                entity_types.extend(domain.entity_types)
        return list(set(entity_types))

    def to_json_schema(self, include_all: bool = False) -> dict[str, Any]:
        """Generate JSON Schema for SVL metadata.

        Args:
            include_all: Include all SVL fields even if disabled

        Returns:
            JSON Schema dict for semantic metadata
        """
        properties: dict[str, Any] = {}

        # Message type
        msg_types = self.get_message_types()
        if msg_types or include_all:
            properties["message_type"] = {
                "type": "string",
                "enum": msg_types if msg_types else get_message_types(),
                "description": "Type of message (query, response, etc.)",
            }

        # Message intent
        intents = self.get_message_intents()
        if intents or include_all:
            properties["message_intent"] = {
                "type": "string",
                "enum": intents if intents else get_message_intents(),
                "description": "Intent behind the message",
            }

        # Temporal qualifier
        temporal = self.get_temporal_qualifiers()
        if temporal or include_all:
            properties["temporal_qualifier"] = {
                "type": "string",
                "enum": temporal if temporal else get_temporal_qualifiers(),
                "description": "Time-based qualifier (daily, past_event, etc.)",
            }

        # Emotional classification
        emotional = self.get_emotional_classifications()
        if emotional or include_all:
            properties["emotional_classification"] = {
                "type": "string",
                "enum": emotional if emotional else get_emotional_classifications(),
                "description": "Emotional content classification",
            }
            properties["emotional_intensity"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Emotional intensity (0-1)",
            }

        # User role
        roles = self.get_user_roles()
        if roles or include_all:
            properties["user_role"] = {
                "type": "string",
                "enum": roles if roles else get_user_roles(),
                "description": "User role classification",
            }

        # Preference type
        pref_types = self.get_preference_types()
        if pref_types or include_all:
            properties["preference_type"] = {
                "type": "string",
                "enum": pref_types if pref_types else get_preference_types(),
                "description": "Type of preference being expressed",
            }

        # Domain label
        domains = self.get_domain_labels()
        if domains or include_all:
            properties["domain_label"] = {
                "type": "string",
                "enum": domains if domains else get_domain_labels(),
                "description": "High-level domain classification",
            }
            properties["subdomain"] = {
                "type": "string",
                "description": "Specific subdomain within the domain",
            }

        # Urgency
        urgency = self.get_urgency_levels()
        if urgency or include_all:
            properties["urgency"] = {
                "type": "string",
                "enum": urgency if urgency else get_urgency_levels(),
                "description": "Urgency level",
            }

        # Confidence
        confidence = self.get_confidence_levels()
        if confidence or include_all:
            properties["confidence"] = {
                "type": "string",
                "enum": confidence if confidence else get_confidence_levels(),
                "description": "Confidence in extracted information",
            }

        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": True,
        }

    def validate_metadata(self, metadata: dict[str, Any] | SemanticMetadata) -> tuple[bool, list[str]]:
        """Validate semantic metadata against this schema.

        Args:
            metadata: Metadata dict or SemanticMetadata to validate

        Returns:
            (is_valid, list of error messages)
        """
        if isinstance(metadata, SemanticMetadata):
            metadata = metadata.to_dict()

        errors = []

        # Validate message_type
        if "message_type" in metadata:
            valid = self.get_message_types()
            if valid and metadata["message_type"] not in valid:
                errors.append(f"Invalid message_type: {metadata['message_type']}")

        # Validate message_intent
        if "message_intent" in metadata:
            valid = self.get_message_intents()
            if valid and metadata["message_intent"] not in valid:
                errors.append(f"Invalid message_intent: {metadata['message_intent']}")

        # Validate temporal_qualifier
        if "temporal_qualifier" in metadata:
            valid = self.get_temporal_qualifiers()
            if valid and metadata["temporal_qualifier"] not in valid:
                errors.append(f"Invalid temporal_qualifier: {metadata['temporal_qualifier']}")

        # Validate emotional_classification
        if "emotional_classification" in metadata:
            valid = self.get_emotional_classifications()
            if valid and metadata["emotional_classification"] not in valid:
                errors.append(f"Invalid emotional_classification: {metadata['emotional_classification']}")

        # Validate emotional_intensity
        if "emotional_intensity" in metadata:
            intensity = metadata["emotional_intensity"]
            if not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 1:
                errors.append(f"Invalid emotional_intensity: {intensity} (must be 0-1)")

        # Validate user_role
        if "user_role" in metadata:
            valid = self.get_user_roles()
            if valid and metadata["user_role"] not in valid:
                errors.append(f"Invalid user_role: {metadata['user_role']}")

        # Validate preference_type
        if "preference_type" in metadata:
            valid = self.get_preference_types()
            if valid and metadata["preference_type"] not in valid:
                errors.append(f"Invalid preference_type: {metadata['preference_type']}")

        # Validate domain_label
        if "domain_label" in metadata:
            valid = self.get_domain_labels()
            if valid and metadata["domain_label"] not in valid:
                errors.append(f"Invalid domain_label: {metadata['domain_label']}")

        # Validate urgency
        if "urgency" in metadata:
            valid = self.get_urgency_levels()
            if valid and metadata["urgency"] not in valid:
                errors.append(f"Invalid urgency: {metadata['urgency']}")

        # Validate confidence
        if "confidence" in metadata:
            valid = self.get_confidence_levels()
            if valid and metadata["confidence"] not in valid:
                errors.append(f"Invalid confidence: {metadata['confidence']}")

        return len(errors) == 0, errors


class SharedVocabularyLayer:
    """The main SVL class - manages semantic vocabulary for MindCore.

    SVL ensures consistent metadata across all memories and provides:
    - Semantic validation
    - Domain-specific vocabulary
    - JSON Schema for LLM structured output
    - Metadata enrichment

    Example:
        svl = SharedVocabularyLayer()
        svl.add_domain("ecommerce")
        svl.add_domain("customer_service")

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

    def __init__(
        self,
        schema: SVLSchema | None = None,
        domains: list[str] | None = None,
    ):
        """Initialize SVL.

        Args:
            schema: Custom SVLSchema or None for defaults
            domains: Domain names to activate
        """
        self.schema = schema or SVLSchema()

        if domains:
            for domain in domains:
                self.add_domain(domain)

    def add_domain(self, domain_name: str) -> None:
        """Add a domain vocabulary.

        Args:
            domain_name: Name of domain to add

        Raises:
            ValueError: If domain not found
        """
        if domain_name not in DOMAIN_REGISTRY:
            available = list_domains()
            raise ValueError(f"Domain '{domain_name}' not found. Available: {available}")

        if domain_name not in self.schema.domains:
            self.schema.domains.append(domain_name)

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain vocabulary."""
        if domain_name in self.schema.domains:
            self.schema.domains.remove(domain_name)

    def get_active_domains(self) -> list[str]:
        """Get list of active domain names."""
        return list(self.schema.domains)

    def get_domain_vocabulary(self, domain_name: str) -> DomainVocabulary | None:
        """Get a specific domain vocabulary."""
        return get_domain(domain_name)

    def add_topics(self, *topics: str) -> None:
        """Add base topics to the schema."""
        for topic in topics:
            if topic not in self.schema.topics:
                self.schema.topics.append(topic)

    def add_categories(self, *categories: str) -> None:
        """Add base categories to the schema."""
        for cat in categories:
            if cat not in self.schema.categories:
                self.schema.categories.append(cat)

    def add_subcategory(self, category: str, *subcategories: str) -> None:
        """Add subcategories to a category."""
        if category not in self.schema.subcategories:
            self.schema.subcategories[category] = []

        for sub in subcategories:
            if sub not in self.schema.subcategories[category]:
                self.schema.subcategories[category].append(sub)

    def validate_metadata(
        self,
        metadata: dict[str, Any] | SemanticMetadata,
    ) -> tuple[bool, list[str]]:
        """Validate semantic metadata.

        Args:
            metadata: Metadata to validate

        Returns:
            (is_valid, list of errors)
        """
        return self.schema.validate_metadata(metadata)

    def get_json_schema(self, include_all: bool = False) -> dict[str, Any]:
        """Get JSON Schema for SVL metadata.

        Args:
            include_all: Include all fields even if disabled

        Returns:
            JSON Schema dict
        """
        return self.schema.to_json_schema(include_all=include_all)

    def get_full_json_schema(self) -> dict[str, Any]:
        """Get complete JSON Schema for memory with SVL metadata.

        Returns:
            JSON Schema for a memory object including SVL semantic_metadata
        """
        svl_schema = self.get_json_schema()

        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()

        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "preference",
                            "entity", "relationship", "temporal", "working"],
                    "description": "Type of memory",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "enum": topics} if topics else {"type": "string"},
                    "description": "Relevant topics",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": categories} if categories else {"type": "string"},
                    "description": "Categories",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Importance score 0-1",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extracted entities",
                },
                "semantic_metadata": svl_schema,
            },
            "required": ["content", "memory_type"],
        }

    def get_prompt_instructions(self) -> str:
        """Generate prompt instructions for LLM.

        Returns:
            Human-readable instructions for using SVL vocabulary
        """
        domains = self.get_active_domains()
        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()

        instructions = f"""## Shared Vocabulary Layer (SVL)

When storing memories, use these standardized values:

### Topics
{', '.join(topics[:20])}{'...' if len(topics) > 20 else ''}

### Categories
{', '.join(categories[:15])}{'...' if len(categories) > 15 else ''}

### Message Types
{', '.join(self.schema.get_message_types()[:10])}

### Message Intents
{', '.join(self.schema.get_message_intents()[:10])}...

### Temporal Qualifiers
{', '.join(self.schema.get_temporal_qualifiers()[:8])}...

### Emotional Classifications
{', '.join(self.schema.get_emotional_classifications()[:8])}...

### User Roles
{', '.join(self.schema.get_user_roles()[:6])}...

### Urgency Levels
{', '.join(self.schema.get_urgency_levels())}

### Confidence Levels
{', '.join(self.schema.get_confidence_levels())}

"""
        if domains:
            instructions += f"""### Active Domains
{', '.join(domains)}
"""

        return instructions

    def enrich_memory(
        self,
        memory: dict[str, Any],
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enrich a memory with SVL metadata defaults.

        Args:
            memory: Memory dict to enrich
            defaults: Default values for semantic_metadata fields

        Returns:
            Enriched memory dict
        """
        result = dict(memory)

        if "semantic_metadata" not in result:
            result["semantic_metadata"] = {}

        if defaults:
            for key, value in defaults.items():
                if key not in result["semantic_metadata"]:
                    result["semantic_metadata"][key] = value

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize SVL state."""
        return {
            "version": self.schema.version,
            "topics": self.schema.topics,
            "categories": self.schema.categories,
            "subcategories": self.schema.subcategories,
            "domains": self.schema.domains,
            "enable_message_types": self.schema.enable_message_types,
            "enable_message_intents": self.schema.enable_message_intents,
            "enable_temporal": self.schema.enable_temporal,
            "enable_emotional": self.schema.enable_emotional,
            "enable_user_roles": self.schema.enable_user_roles,
            "enable_preference_types": self.schema.enable_preference_types,
            "enable_domain_labels": self.schema.enable_domain_labels,
            "enable_urgency": self.schema.enable_urgency,
            "enable_confidence": self.schema.enable_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedVocabularyLayer:
        """Create from serialized state."""
        schema = SVLSchema(
            version=data.get("version", "1.0.0"),
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            subcategories=data.get("subcategories", {}),
            domains=data.get("domains", []),
            enable_message_types=data.get("enable_message_types", True),
            enable_message_intents=data.get("enable_message_intents", True),
            enable_temporal=data.get("enable_temporal", True),
            enable_emotional=data.get("enable_emotional", True),
            enable_user_roles=data.get("enable_user_roles", True),
            enable_preference_types=data.get("enable_preference_types", True),
            enable_domain_labels=data.get("enable_domain_labels", True),
            enable_urgency=data.get("enable_urgency", True),
            enable_confidence=data.get("enable_confidence", True),
        )
        return cls(schema=schema)


# Default SVL instance for quick start
DEFAULT_SVL = SharedVocabularyLayer(
    domains=["customer_service"],
)
