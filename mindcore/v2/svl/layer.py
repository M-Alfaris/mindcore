"""SharedVocabularyLayer - The unified semantic system for MindCore.

SVL is the single vocabulary system for MindCore, providing:
1. Universal ontology (message types, intents, temporal, emotional, etc.)
2. Domain-specific vocabulary extensions
3. Memory types, sentiments, access levels
4. Vocabulary versioning and migrations
5. JSON Schema, Pydantic, TypeScript code generation
6. Data source mapping for automatic context enrichment

This replaces the previous VocabularySchema with a unified semantic layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .domains import (
    DOMAIN_REGISTRY,
    DomainVocabulary,
    get_domain,
    list_domains,
)
from .ontology import (
    SemanticMetadata,
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
from .sources import (
    DataSource,
    FetchResult,
    SourceRegistry,
    TriggerCondition,
)


# =============================================================================
# Core Enums (from VocabularySchema)
# =============================================================================


class MemoryType(str, Enum):
    """Core memory types."""

    EPISODIC = "episodic"  # Events, conversations, interactions
    SEMANTIC = "semantic"  # Facts, knowledge, learned information
    PROCEDURAL = "procedural"  # Workflows, how-to, processes
    PREFERENCE = "preference"  # User preferences, settings
    ENTITY = "entity"  # People, places, things
    RELATIONSHIP = "relationship"  # Connections between entities
    TEMPORAL = "temporal"  # Time-bound info (auto-expires)
    WORKING = "working"  # Current session context (cleared)


class Sentiment(str, Enum):
    """Sentiment values."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class AccessLevel(str, Enum):
    """Memory access levels for multi-agent."""

    PRIVATE = "private"  # Only this agent
    TEAM = "team"  # Agents in same team/group
    SHARED = "shared"  # All agents for this user
    GLOBAL = "global"  # Cross-user (knowledge base)


# =============================================================================
# Migration Support
# =============================================================================


@dataclass
class MigrationCheckpoint:
    """Checkpoint for migration rollback.

    Stores original data to enable rollback after migration.

    Example:
        # After migration
        migrated, checkpoint = svl.migrate_memory(data, "1.0.0", create_checkpoint=True)

        # Later, rollback if needed
        original = svl.rollback_memory(migrated, checkpoint)
    """

    checkpoint_id: str
    from_version: str
    to_version: str
    memory_id: str
    original_data: dict[str, Any]
    migrated_data: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "memory_id": self.memory_id,
            "original_data": self.original_data,
            "migrated_data": self.migrated_data,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationCheckpoint:
        """Create checkpoint from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            from_version=data["from_version"],
            to_version=data["to_version"],
            memory_id=data["memory_id"],
            original_data=data["original_data"],
            migrated_data=data["migrated_data"],
            created_at=created_at,
        )


@dataclass
class Migration:
    """Migration rules between vocabulary versions."""

    from_version: str
    to_version: str

    # Field transformations
    renames: dict[str, str] = field(default_factory=dict)  # old -> new
    merges: dict[str, list[str]] = field(default_factory=dict)  # new -> [old1, old2]
    splits: dict[str, dict[str, str]] = field(default_factory=dict)  # old -> {condition: new}
    deletes: list[str] = field(default_factory=list)

    # New fields with defaults
    added_fields: dict[str, Any] = field(default_factory=dict)  # field -> default

    # Rollback configuration
    reversible: bool = True  # Whether this migration can be rolled back

    def apply_to_topics(self, topics: list[str]) -> list[str]:
        """Apply migration to a list of topics."""
        result = []
        for topic in topics:
            if topic in self.deletes:
                continue
            if topic in self.renames:
                result.append(self.renames[topic])
            else:
                merged = False
                for new_topic, old_topics in self.merges.items():
                    if topic in old_topics:
                        if new_topic not in result:
                            result.append(new_topic)
                        merged = True
                        break
                if not merged:
                    result.append(topic)
        return list(set(result))

    def apply_to_categories(self, categories: list[str]) -> list[str]:
        """Apply migration to categories."""
        return self.apply_to_topics(categories)

    def rollback_topics(
        self,
        current_topics: list[str],
        original_topics: list[str] | None = None,
    ) -> list[str]:
        """Rollback topic migration.

        If original_topics is provided, returns those directly.
        Otherwise, attempts to reverse the migration by:
        - Reversing renames (new -> old)
        - Un-merging topics (if possible)

        Args:
            current_topics: Current (migrated) topics
            original_topics: Original topics from checkpoint (preferred)

        Returns:
            Rolled-back topics list
        """
        if original_topics:
            return list(original_topics)

        # Build reverse rename mapping
        reverse_renames = {v: k for k, v in self.renames.items()}

        result = []
        for topic in current_topics:
            if topic in reverse_renames:
                result.append(reverse_renames[topic])
            else:
                result.append(topic)

        return list(set(result))

    def can_rollback(self) -> bool:
        """Check if this migration can be rolled back.

        Migrations with merges may lose information and are
        harder to rollback without checkpoints.

        Returns:
            True if migration is marked as reversible
        """
        return self.reversible


@dataclass
class FieldSchema:
    """Schema for a custom field."""

    name: str
    field_type: str  # "string", "number", "boolean", "array", "enum"
    required: bool = False
    enum_values: list[str] | None = None
    default: Any = None
    description: str = ""


# =============================================================================
# SVL Schema
# =============================================================================


@dataclass
class SVLSchema:
    """Complete SVL schema - the unified vocabulary configuration.

    This combines all vocabulary features into one schema:
    - Base vocabulary (topics, categories, memory_types, etc.)
    - SVL ontology (message types, intents, temporal, emotional, etc.)
    - Domain vocabularies
    - Custom fields and migrations
    """

    version: str = "1.0.0"

    # Base vocabulary
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    subcategories: dict[str, list[str]] = field(default_factory=dict)

    # Memory configuration
    memory_types: list[str] = field(default_factory=lambda: [t.value for t in MemoryType])
    sentiments: list[str] = field(default_factory=lambda: [s.value for s in Sentiment])
    access_levels: list[str] = field(default_factory=lambda: [a.value for a in AccessLevel])

    # Active domains
    domains: list[str] = field(default_factory=list)

    # Ontology toggles
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
    custom_fields: list[FieldSchema] = field(default_factory=list)
    custom_message_types: list[str] = field(default_factory=list)
    custom_intents: list[str] = field(default_factory=list)
    custom_temporal: list[str] = field(default_factory=list)
    custom_emotional: list[str] = field(default_factory=list)
    custom_user_roles: list[str] = field(default_factory=list)
    custom_preference_types: list[str] = field(default_factory=list)

    # Migrations
    migrations: dict[str, Migration] = field(default_factory=dict)

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ==========================================================================
    # Getters for all vocabulary values
    # ==========================================================================

    def get_memory_types(self) -> list[str]:
        """Get all valid memory types."""
        return self.memory_types

    def get_sentiments(self) -> list[str]:
        """Get all valid sentiments."""
        return self.sentiments

    def get_access_levels(self) -> list[str]:
        """Get all valid access levels."""
        return self.access_levels

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


# =============================================================================
# Main SharedVocabularyLayer Class
# =============================================================================


class SharedVocabularyLayer:
    """The unified semantic vocabulary layer for MindCore.

    SVL manages all vocabulary, semantic metadata, and data source mappings.

    Example:
        svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

        # Map topics to data sources
        svl.map_source("orders", TableSource(
            name="orders_db",
            connection_string="postgresql://...",
            query_template="SELECT * FROM orders WHERE user_id = :user_id",
            param_mapping={"user_id": "user_id"},
        ))

        # Validate metadata
        is_valid, errors = svl.validate_metadata({
            "message_type": "query",
            "urgency": "high",
        })

        # Fetch data for topics
        data = svl.fetch_for_topics(["orders"], context={"user_id": "123"})
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
        self._sources = SourceRegistry()

        if domains:
            for domain in domains:
                self.add_domain(domain)

    # ==========================================================================
    # Domain Management
    # ==========================================================================

    def add_domain(self, domain_name: str, strict: bool = False) -> None:
        """Add a domain vocabulary.

        Args:
            domain_name: Name of the domain to add
            strict: If True, only allow domains from DOMAIN_REGISTRY.
                   If False (default), allow any domain name including custom ones.
        """
        if strict and domain_name not in DOMAIN_REGISTRY:
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

    # ==========================================================================
    # Vocabulary Management
    # ==========================================================================

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

    def add_custom_field(
        self,
        name: str,
        field_type: str,
        required: bool = False,
        enum_values: list[str] | None = None,
        default: Any = None,
        description: str = "",
    ) -> None:
        """Add a custom field to the schema."""
        self.schema.custom_fields.append(
            FieldSchema(
                name=name,
                field_type=field_type,
                required=required,
                enum_values=enum_values,
                default=default,
                description=description,
            )
        )

    # ==========================================================================
    # Data Source Mapping
    # ==========================================================================

    def map_source(
        self,
        term: str,
        source: DataSource | dict,
        term_type: str = "topic",
    ) -> None:
        """Map a vocabulary term to a data source.

        When this term is used in queries, data will be fetched from the source.

        Args:
            term: Topic, category, or domain name
            source: Data source (TableSource, APISource, MCPSource, FunctionSource, or dict)
            term_type: Type of term ("topic", "category", "domain", "intent")
        """
        self._sources.map(term, source, term_type)

    def unmap_source(self, term: str, source_name: str | None = None) -> bool:
        """Remove a source mapping."""
        return self._sources.unmap(term, source_name)

    def get_mapped_terms(self) -> list[str]:
        """Get all terms with source mappings."""
        return self._sources.get_mapped_terms()

    def fetch_for_topics(
        self,
        topics: list[str],
        context: dict[str, Any],
        trigger: TriggerCondition = TriggerCondition.ON_QUERY,
    ) -> dict[str, list[FetchResult]]:
        """Fetch data from all sources mapped to the given topics.

        Args:
            topics: List of topics to fetch data for
            context: Context dict with user_id, query, etc.
            trigger: Current trigger condition

        Returns:
            Dict mapping topic -> list of FetchResults
        """
        return self._sources.fetch_for_terms(topics, context, trigger)

    def set_mcp_client(self, client: Any) -> None:
        """Set MCP client for MCP sources."""
        self._sources.set_mcp_client(client)

    # ==========================================================================
    # Validation
    # ==========================================================================

    def validate_memory(self, memory: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a memory against this vocabulary.

        Args:
            memory: Memory dict to validate

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Required fields
        if "content" not in memory:
            errors.append("Missing required field: content")
        if "memory_type" not in memory:
            errors.append("Missing required field: memory_type")
        elif memory["memory_type"] not in self.schema.memory_types:
            errors.append(f"Invalid memory_type: {memory['memory_type']}")

        # Validate topics
        topics = self.schema.get_all_topics()
        if "topics" in memory and topics:
            invalid_topics = [t for t in memory["topics"] if t not in topics]
            if invalid_topics:
                errors.append(f"Invalid topics: {invalid_topics}")

        # Validate categories
        categories = self.schema.get_all_categories()
        if "categories" in memory and categories:
            invalid_cats = [c for c in memory["categories"] if c not in categories]
            if invalid_cats:
                errors.append(f"Invalid categories: {invalid_cats}")

        # Validate sentiment
        if "sentiment" in memory and memory["sentiment"] not in self.schema.sentiments:
            errors.append(f"Invalid sentiment: {memory['sentiment']}")

        # Validate importance
        if "importance" in memory:
            imp = memory["importance"]
            if not isinstance(imp, int | float) or imp < 0 or imp > 1:
                errors.append(f"Invalid importance: {imp} (must be 0-1)")

        # Validate access_level
        if "access_level" in memory and memory["access_level"] not in self.schema.access_levels:
            errors.append(f"Invalid access_level: {memory['access_level']}")

        # Validate custom fields
        for custom in self.schema.custom_fields:
            if custom.required and custom.name not in memory:
                errors.append(f"Missing required custom field: {custom.name}")
            elif custom.name in memory and custom.enum_values:
                if memory[custom.name] not in custom.enum_values:
                    errors.append(f"Invalid {custom.name}: {memory[custom.name]}")

        # Validate semantic metadata
        if "semantic_metadata" in memory:
            _, meta_errors = self.validate_metadata(memory["semantic_metadata"])
            errors.extend(meta_errors)

        return len(errors) == 0, errors

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
        if isinstance(metadata, SemanticMetadata):
            metadata = metadata.to_dict()

        errors = []

        if "message_type" in metadata:
            valid = self.schema.get_message_types()
            if valid and metadata["message_type"] not in valid:
                errors.append(f"Invalid message_type: {metadata['message_type']}")

        if "message_intent" in metadata:
            valid = self.schema.get_message_intents()
            if valid and metadata["message_intent"] not in valid:
                errors.append(f"Invalid message_intent: {metadata['message_intent']}")

        if "temporal_qualifier" in metadata:
            valid = self.schema.get_temporal_qualifiers()
            if valid and metadata["temporal_qualifier"] not in valid:
                errors.append(f"Invalid temporal_qualifier: {metadata['temporal_qualifier']}")

        if "emotional_classification" in metadata:
            valid = self.schema.get_emotional_classifications()
            if valid and metadata["emotional_classification"] not in valid:
                errors.append(
                    f"Invalid emotional_classification: {metadata['emotional_classification']}"
                )

        if "emotional_intensity" in metadata:
            intensity = metadata["emotional_intensity"]
            if not isinstance(intensity, int | float) or intensity < 0 or intensity > 1:
                errors.append(f"Invalid emotional_intensity: {intensity} (must be 0-1)")

        if "user_role" in metadata:
            valid = self.schema.get_user_roles()
            if valid and metadata["user_role"] not in valid:
                errors.append(f"Invalid user_role: {metadata['user_role']}")

        if "preference_type" in metadata:
            valid = self.schema.get_preference_types()
            if valid and metadata["preference_type"] not in valid:
                errors.append(f"Invalid preference_type: {metadata['preference_type']}")

        if "domain_label" in metadata:
            valid = self.schema.get_domain_labels()
            if valid and metadata["domain_label"] not in valid:
                errors.append(f"Invalid domain_label: {metadata['domain_label']}")

        if "urgency" in metadata:
            valid = self.schema.get_urgency_levels()
            if valid and metadata["urgency"] not in valid:
                errors.append(f"Invalid urgency: {metadata['urgency']}")

        if "confidence" in metadata:
            valid = self.schema.get_confidence_levels()
            if valid and metadata["confidence"] not in valid:
                errors.append(f"Invalid confidence: {metadata['confidence']}")

        return len(errors) == 0, errors

    # ==========================================================================
    # Migration
    # ==========================================================================

    def add_migration(self, migration: Migration) -> None:
        """Add a migration from a previous version."""
        self.schema.migrations[migration.from_version] = migration

    def migrate_memory(
        self,
        memory: dict[str, Any],
        from_version: str,
        create_checkpoint: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], MigrationCheckpoint]:
        """Migrate a memory from an older vocabulary version.

        Args:
            memory: Memory dict to migrate
            from_version: Source vocabulary version
            create_checkpoint: If True, returns (migrated, checkpoint) tuple

        Returns:
            Migrated memory dict, or (migrated, checkpoint) tuple if create_checkpoint=True

        Example:
            # Simple migration
            migrated = svl.migrate_memory(data, "1.0.0")

            # Migration with checkpoint for rollback
            migrated, checkpoint = svl.migrate_memory(data, "1.0.0", create_checkpoint=True)
        """
        import uuid

        if from_version == self.schema.version:
            if create_checkpoint:
                # No migration needed, but return empty checkpoint
                checkpoint = MigrationCheckpoint(
                    checkpoint_id=f"cp_{uuid.uuid4().hex[:12]}",
                    from_version=from_version,
                    to_version=self.schema.version,
                    memory_id=memory.get("memory_id", "unknown"),
                    original_data=memory.copy(),
                    migrated_data=memory.copy(),
                )
                return memory, checkpoint
            return memory

        if from_version not in self.schema.migrations:
            raise ValueError(f"No migration path from {from_version} to {self.schema.version}")

        migration = self.schema.migrations[from_version]
        original = memory.copy()
        result = memory.copy()

        if "topics" in result:
            result["topics"] = migration.apply_to_topics(result["topics"])

        if "categories" in result:
            result["categories"] = migration.apply_to_categories(result["categories"])

        for field_name, default_value in migration.added_fields.items():
            if field_name not in result:
                result[field_name] = default_value

        # Update vocabulary version
        result["vocabulary_version"] = self.schema.version

        # Optionally embed migration metadata for rollback without checkpoint
        if create_checkpoint:
            result["_migration_metadata"] = {
                "from_version": from_version,
                "to_version": self.schema.version,
                "original_topics": original.get("topics", []),
                "original_categories": original.get("categories", []),
            }

            checkpoint = MigrationCheckpoint(
                checkpoint_id=f"cp_{uuid.uuid4().hex[:12]}",
                from_version=from_version,
                to_version=self.schema.version,
                memory_id=memory.get("memory_id", "unknown"),
                original_data=original,
                migrated_data=result.copy(),
            )
            return result, checkpoint

        return result

    def rollback_memory(
        self,
        memory: dict[str, Any],
        checkpoint: MigrationCheckpoint | None = None,
    ) -> dict[str, Any]:
        """Rollback a migrated memory to its original state.

        Uses checkpoint if provided, otherwise attempts rollback using
        embedded migration metadata.

        Args:
            memory: Migrated memory to rollback
            checkpoint: Optional checkpoint from migrate_memory

        Returns:
            Original memory data

        Raises:
            ValueError: If no checkpoint and no embedded metadata

        Example:
            # Rollback with checkpoint
            original = svl.rollback_memory(migrated, checkpoint)

            # Rollback using embedded metadata
            original = svl.rollback_memory(migrated)
        """
        # Prefer checkpoint if provided
        if checkpoint is not None:
            result = checkpoint.original_data.copy()
            result["vocabulary_version"] = checkpoint.from_version
            return result

        # Try embedded metadata
        metadata = memory.get("_migration_metadata")
        if metadata is None:
            raise ValueError(
                "Cannot rollback: no checkpoint provided and no embedded migration metadata. "
                "Use create_checkpoint=True when migrating to enable rollback."
            )

        from_version = metadata["from_version"]
        original_topics = metadata.get("original_topics", [])
        original_categories = metadata.get("original_categories", [])

        result = memory.copy()
        result.pop("_migration_metadata", None)  # Remove metadata

        # Restore original values
        if original_topics:
            result["topics"] = original_topics
        if original_categories:
            result["categories"] = original_categories

        result["vocabulary_version"] = from_version

        # Remove added fields
        if from_version in self.schema.migrations:
            migration = self.schema.migrations[from_version]
            for field_name in migration.added_fields:
                result.pop(field_name, None)

        return result

    def get_migration_path(self, from_version: str) -> list[str]:
        """Get the migration path from a version to current.

        Args:
            from_version: Starting version

        Returns:
            List of versions in migration order [from, ..., current]

        Raises:
            ValueError: If no path exists

        Example:
            path = svl.get_migration_path("1.0.0")
            # Returns ["1.0.0", "2.0.0"] if current is 2.0.0
        """
        if from_version == self.schema.version:
            return [from_version]

        if from_version not in self.schema.migrations:
            raise ValueError(f"No migration path from {from_version} to {self.schema.version}")

        # Build path through chain of migrations
        path = [from_version]
        current = from_version

        while current != self.schema.version:
            if current not in self.schema.migrations:
                raise ValueError(f"Broken migration path: no migration from {current}")
            migration = self.schema.migrations[current]
            path.append(migration.to_version)
            current = migration.to_version

        return path

    # ==========================================================================
    # JSON Schema Generation
    # ==========================================================================

    def get_json_schema(self, include_all: bool = False) -> dict[str, Any]:
        """Get JSON Schema for semantic metadata.

        Args:
            include_all: Include all fields even if disabled

        Returns:
            JSON Schema dict
        """
        properties: dict[str, Any] = {}

        # Always include memory_type
        memory_types = (
            self.schema.memory_types
            if self.schema.memory_types
            else ["episodic", "semantic", "procedural", "preference", "entity"]
        )
        properties["memory_type"] = {
            "type": "string",
            "enum": memory_types,
            "description": "Type of memory (episodic, semantic, procedural, preference, entity)",
        }

        msg_types = self.schema.get_message_types()
        if msg_types or include_all:
            properties["message_type"] = {
                "type": "string",
                "enum": msg_types if msg_types else get_message_types(),
                "description": "Type of message",
            }

        intents = self.schema.get_message_intents()
        if intents or include_all:
            properties["message_intent"] = {
                "type": "string",
                "enum": intents if intents else get_message_intents(),
                "description": "Intent behind the message",
            }

        temporal = self.schema.get_temporal_qualifiers()
        if temporal or include_all:
            properties["temporal_qualifier"] = {
                "type": "string",
                "enum": temporal if temporal else get_temporal_qualifiers(),
                "description": "Time-based qualifier",
            }

        emotional = self.schema.get_emotional_classifications()
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

        roles = self.schema.get_user_roles()
        if roles or include_all:
            properties["user_role"] = {
                "type": "string",
                "enum": roles if roles else get_user_roles(),
                "description": "User role classification",
            }

        pref_types = self.schema.get_preference_types()
        if pref_types or include_all:
            properties["preference_type"] = {
                "type": "string",
                "enum": pref_types if pref_types else get_preference_types(),
                "description": "Type of preference",
            }

        domains = self.schema.get_domain_labels()
        if domains or include_all:
            properties["domain_label"] = {
                "type": "string",
                "enum": domains if domains else get_domain_labels(),
                "description": "Domain classification",
            }

        urgency = self.schema.get_urgency_levels()
        if urgency or include_all:
            properties["urgency"] = {
                "type": "string",
                "enum": urgency if urgency else get_urgency_levels(),
                "description": "Urgency level",
            }

        confidence = self.schema.get_confidence_levels()
        if confidence or include_all:
            properties["confidence"] = {
                "type": "string",
                "enum": confidence if confidence else get_confidence_levels(),
                "description": "Confidence level",
            }

        return {"type": "object", "properties": properties, "additionalProperties": True}

    def get_full_memory_schema(self, include_response: bool = True) -> dict[str, Any]:
        """Get complete JSON Schema for memory with all fields.

        Args:
            include_response: Include response field for agent output

        Returns:
            JSON Schema dict for LLM structured output
        """
        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()
        svl_schema = self.get_json_schema()

        memory_schema = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store",
                },
                "memory_type": {
                    "type": "string",
                    "enum": self.schema.memory_types,
                    "description": "Type of memory",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "enum": topics} if topics else {"type": "string"},
                    "description": "Relevant topics",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": categories}
                    if categories
                    else {"type": "string"},
                    "description": "Categories",
                },
                "sentiment": {
                    "type": "string",
                    "enum": self.schema.sentiments,
                    "description": "Sentiment of the content",
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
                "access_level": {
                    "type": "string",
                    "enum": self.schema.access_levels,
                    "description": "Access level for multi-agent",
                },
                "semantic_metadata": svl_schema,
            },
            "required": ["content", "memory_type"],
        }

        # Add custom fields
        for custom in self.schema.custom_fields:
            field_def = {"description": custom.description}
            if custom.field_type == "enum" and custom.enum_values:
                field_def["type"] = "string"
                field_def["enum"] = custom.enum_values
            elif custom.field_type == "array":
                field_def["type"] = "array"
                field_def["items"] = {"type": "string"}
            else:
                field_def["type"] = custom.field_type
            if custom.default is not None:
                field_def["default"] = custom.default
            memory_schema["properties"][custom.name] = field_def
            if custom.required:
                memory_schema["required"].append(custom.name)

        if include_response:
            return {
                "type": "object",
                "properties": {
                    "response": {"type": "string", "description": "Response to the user"},
                    "memories_to_store": {
                        "type": "array",
                        "items": memory_schema,
                        "description": "Memories to store",
                    },
                    "memory_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "topics": {"type": "array", "items": {"type": "string"}},
                                "memory_types": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                        "description": "Queries to search memories",
                    },
                },
                "required": ["response"],
            }

        return memory_schema

    # ==========================================================================
    # Code Generation
    # ==========================================================================

    def to_pydantic(self) -> str:
        """Generate Pydantic model code."""
        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()

        topics_literal = f"Literal[{', '.join(repr(t) for t in topics)}]" if topics else "str"
        categories_literal = (
            f"Literal[{', '.join(repr(c) for c in categories)}]" if categories else "str"
        )

        return f'''"""Auto-generated Pydantic models from SVL v{self.schema.version}."""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class SemanticMetadata(BaseModel):
    """SVL semantic metadata."""

    message_type: Optional[Literal[{", ".join(repr(m) for m in self.schema.get_message_types()[:10])}]] = None
    message_intent: Optional[str] = None
    temporal_qualifier: Optional[Literal[{", ".join(repr(t) for t in self.schema.get_temporal_qualifiers()[:8])}]] = None
    emotional_classification: Optional[Literal[{", ".join(repr(e) for e in self.schema.get_emotional_classifications()[:8])}]] = None
    emotional_intensity: float = Field(default=0.5, ge=0, le=1)
    user_role: Optional[str] = None
    urgency: Optional[Literal[{", ".join(repr(u) for u in self.schema.get_urgency_levels())}]] = None
    confidence: Optional[Literal[{", ".join(repr(c) for c in self.schema.get_confidence_levels())}]] = None


class Memory(BaseModel):
    """Memory model with SVL vocabulary."""

    content: str
    memory_type: Literal[{", ".join(repr(m) for m in self.schema.memory_types)}]
    topics: list[{topics_literal}] = Field(default_factory=list)
    categories: list[{categories_literal}] = Field(default_factory=list)
    sentiment: Literal[{", ".join(repr(s) for s in self.schema.sentiments)}] = "neutral"
    importance: float = Field(default=0.5, ge=0, le=1)
    entities: list[str] = Field(default_factory=list)
    access_level: Literal[{", ".join(repr(a) for a in self.schema.access_levels)}] = "private"
    semantic_metadata: Optional[SemanticMetadata] = None


class AgentResponse(BaseModel):
    """Response model for LLM structured output."""

    response: str
    memories_to_store: list[Memory] = Field(default_factory=list)
'''

    def to_typescript(self) -> str:
        """Generate TypeScript type definitions."""
        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()

        topics_union = " | ".join(f'"{t}"' for t in topics) if topics else "string"
        categories_union = " | ".join(f'"{c}"' for c in categories) if categories else "string"
        memory_types_union = " | ".join(f'"{m}"' for m in self.schema.memory_types)
        sentiments_union = " | ".join(f'"{s}"' for s in self.schema.sentiments)
        access_levels_union = " | ".join(f'"{a}"' for a in self.schema.access_levels)

        return f"""// Auto-generated TypeScript types from SVL v{self.schema.version}

export type Topic = {topics_union};
export type Category = {categories_union};
export type MemoryType = {memory_types_union};
export type Sentiment = {sentiments_union};
export type AccessLevel = {access_levels_union};

export interface SemanticMetadata {{
  message_type?: string;
  message_intent?: string;
  temporal_qualifier?: string;
  emotional_classification?: string;
  emotional_intensity?: number;
  user_role?: string;
  urgency?: string;
  confidence?: string;
}}

export interface Memory {{
  content: string;
  memory_type: MemoryType;
  topics?: Topic[];
  categories?: Category[];
  sentiment?: Sentiment;
  importance?: number;
  entities?: string[];
  access_level?: AccessLevel;
  semantic_metadata?: SemanticMetadata;
}}

export interface AgentResponse {{
  response: string;
  memories_to_store?: Memory[];
}}
"""

    def to_json_schema(self, include_response: bool = True) -> dict[str, Any]:
        """Alias for get_full_memory_schema() for backwards compatibility.

        Args:
            include_response: Include response field for agent output

        Returns:
            JSON Schema dict for LLM structured output
        """
        return self.get_full_memory_schema(include_response=include_response)

    def to_prompt_instructions(self) -> str:
        """Alias for get_prompt_instructions() for backwards compatibility."""
        return self.get_prompt_instructions()

    def get_prompt_instructions(self) -> str:
        """Generate prompt instructions for LLM."""
        domains = self.get_active_domains()
        topics = self.schema.get_all_topics()
        categories = self.schema.get_all_categories()

        return f"""## Shared Vocabulary Layer (SVL) v{self.schema.version}

When storing memories, use these standardized values:

### Topics
{", ".join(topics[:20])}{"..." if len(topics) > 20 else ""}

### Categories
{", ".join(categories[:15])}{"..." if len(categories) > 15 else ""}

### Memory Types
{", ".join(self.schema.memory_types)}

### Sentiments
{", ".join(self.schema.sentiments)}

### Access Levels
{", ".join(self.schema.access_levels)}

### SVL Semantic Fields
- message_type: {", ".join(self.schema.get_message_types()[:6])}...
- message_intent: {", ".join(self.schema.get_message_intents()[:6])}...
- temporal_qualifier: {", ".join(self.schema.get_temporal_qualifiers()[:6])}...
- emotional_classification: {", ".join(self.schema.get_emotional_classifications()[:6])}...
- urgency: {", ".join(self.schema.get_urgency_levels())}
- confidence: {", ".join(self.schema.get_confidence_levels())}

### Active Domains
{", ".join(domains) if domains else "(none)"}

### Mapped Data Sources
{", ".join(self.get_mapped_terms()) if self.get_mapped_terms() else "(none)"}
"""

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize SVL state."""
        return {
            "version": self.schema.version,
            "topics": self.schema.topics,
            "categories": self.schema.categories,
            "subcategories": self.schema.subcategories,
            "memory_types": self.schema.memory_types,
            "sentiments": self.schema.sentiments,
            "access_levels": self.schema.access_levels,
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
            "custom_fields": [
                {
                    "name": f.name,
                    "field_type": f.field_type,
                    "required": f.required,
                    "enum_values": f.enum_values,
                    "default": f.default,
                    "description": f.description,
                }
                for f in self.schema.custom_fields
            ],
            "description": self.schema.description,
            "sources": self._sources.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedVocabularyLayer:
        """Create from serialized state."""
        custom_fields = [FieldSchema(**f) for f in data.get("custom_fields", [])]

        schema = SVLSchema(
            version=data.get("version", "1.0.0"),
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            subcategories=data.get("subcategories", {}),
            memory_types=data.get("memory_types", [t.value for t in MemoryType]),
            sentiments=data.get("sentiments", [s.value for s in Sentiment]),
            access_levels=data.get("access_levels", [a.value for a in AccessLevel]),
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
            custom_fields=custom_fields,
            description=data.get("description", ""),
        )
        return cls(schema=schema)

    @classmethod
    def from_json(cls, json_str: str) -> SharedVocabularyLayer:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_stats(self) -> dict[str, Any]:
        """Get SVL statistics."""
        return {
            "version": self.schema.version,
            "topics_count": len(self.schema.get_all_topics()),
            "categories_count": len(self.schema.get_all_categories()),
            "domains": self.schema.domains,
            "custom_fields_count": len(self.schema.custom_fields),
            "sources": self._sources.get_stats(),
        }

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


# =============================================================================
# Default Instance
# =============================================================================

DEFAULT_SVL = SharedVocabularyLayer(
    schema=SVLSchema(
        version="1.0.0",
        topics=[
            "greeting",
            "farewell",
            "thanks",
            "help",
            "feedback",
            "issue",
            "bug",
            "error",
            "problem",
            "complaint",
            "billing",
            "payment",
            "refund",
            "subscription",
            "feature",
            "product",
            "service",
            "pricing",
            "api",
            "integration",
            "setup",
            "documentation",
            "account",
            "login",
            "password",
            "settings",
            "profile",
            "order",
            "shipping",
            "delivery",
            "tracking",
        ],
        categories=[
            "support",
            "billing",
            "technical",
            "account",
            "product",
            "feedback",
            "general",
            "urgent",
        ],
        description="Default Mindcore SVL for general use cases",
    ),
    domains=["customer_service"],
)
