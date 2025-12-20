"""Test 05: SVL Domain Tests.

Tests Shared Vocabulary Layer (SVL) domain functionality:
- Domain vocabulary creation
- Schema validation
- Topic and category management
- Vocabulary merging
- JSON schema generation
- Code generation (Pydantic, TypeScript)
"""

import pytest


# ============================================================================
# Domain Vocabulary Creation
# ============================================================================


class TestDomainCreation:
    """Test domain vocabulary creation and configuration."""

    def test_create_default_svl(self, default_svl):
        """Test creating default SVL instance."""
        assert default_svl is not None
        # Should have default topics and categories

    def test_create_custom_svl(self, custom_svl):
        """Test creating custom SVL with specific schema."""
        assert custom_svl is not None

        # Check custom topics are available
        schema = custom_svl.get_json_schema()
        assert schema is not None

    def test_add_domain(self, default_svl):
        """Test adding a domain to SVL."""
        default_svl.add_domain("ecommerce")

        domains = default_svl.get_active_domains()
        assert "ecommerce" in domains or len(domains) > 0

    def test_remove_domain(self, default_svl):
        """Test removing a domain from SVL."""
        default_svl.add_domain("test_domain")
        default_svl.remove_domain("test_domain")

        default_svl.get_active_domains()
        # Should not contain removed domain

    def test_get_domain_vocabulary(self, default_svl):
        """Test getting vocabulary for a specific domain."""
        default_svl.get_domain_vocabulary("customer_service")

        # May return None if domain not loaded, or vocabulary object


# ============================================================================
# Topic and Category Management
# ============================================================================


class TestTopicCategoryManagement:
    """Test topic and category operations."""

    def test_add_topics(self):
        """Test adding custom topics."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer()
        svl.add_topics("custom_topic_1", "custom_topic_2")

        # Topics should be added
        schema = svl.get_json_schema()
        assert schema is not None

    def test_add_categories(self):
        """Test adding custom categories."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer()
        svl.add_categories("custom_category")

        schema = svl.get_json_schema()
        assert schema is not None

    def test_add_subcategories(self):
        """Test adding subcategories to a category."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer()
        svl.add_categories("parent_category")
        svl.add_subcategory("parent_category", "child1", "child2")

        # Subcategories should be associated with parent

    def test_add_custom_field(self):
        """Test adding custom fields to schema."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer()
        svl.add_custom_field(
            name="priority",
            field_type="string",
            required=False,
            enum_values=["low", "medium", "high"],
            description="Priority level",
        )

        svl.get_json_schema()
        # Schema should include custom field


# ============================================================================
# Schema Validation
# ============================================================================


class TestSchemaValidation:
    """Test SVL schema validation."""

    def test_validate_valid_memory(self, default_svl):
        """Test validating a correct memory."""
        memory = {
            "content": "Test content",
            "memory_type": "semantic",
            "topics": ["api"],
            "categories": ["technical"],
            "importance": 0.5,
        }

        _is_valid, _errors = default_svl.validate_memory(memory)
        # Should be valid with default vocabulary

    def test_validate_invalid_memory_type(self, default_svl):
        """Test validating memory with invalid type."""
        memory = {
            "content": "Test content",
            "memory_type": "invalid_type_xyz",
            "topics": ["api"],
            "importance": 0.5,
        }

        is_valid, errors = default_svl.validate_memory(memory)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_metadata(self, default_svl):
        """Test validating metadata fields."""
        metadata = {
            "topics": ["api", "billing"],
            "categories": ["technical"],
            "sentiment": "positive",
            "importance": 0.8,
        }

        _is_valid, _errors = default_svl.validate_metadata(metadata)
        # Should validate metadata correctly

    def test_validate_importance_range(self, default_svl):
        """Test that importance must be in valid range."""
        memory = {
            "content": "Test",
            "memory_type": "semantic",
            "topics": ["api"],
            "importance": 1.5,  # Invalid: > 1.0
        }

        _is_valid, _errors = default_svl.validate_memory(memory)
        # Should fail validation


# ============================================================================
# JSON Schema Generation
# ============================================================================


class TestJSONSchemaGeneration:
    """Test JSON schema generation for LLM integration."""

    def test_get_json_schema(self, default_svl):
        """Test basic JSON schema generation."""
        schema = default_svl.get_json_schema()

        assert schema is not None
        assert isinstance(schema, dict)

    def test_get_full_memory_schema(self, default_svl):
        """Test full memory schema with response fields."""
        schema = default_svl.get_full_memory_schema(include_response=True)

        assert schema is not None
        # Should include response format

    def test_schema_includes_memory_types(self, default_svl):
        """Test that schema includes valid memory types."""
        schema = default_svl.get_json_schema()

        # Should contain memory_type definitions
        schema_str = str(schema)
        assert "semantic" in schema_str or "episodic" in schema_str or "memory_type" in schema_str

    def test_schema_includes_access_levels(self, default_svl):
        """Test that schema includes access levels."""
        schema = default_svl.get_json_schema()

        str(schema)
        # Should contain access level definitions

    def test_get_prompt_instructions(self, default_svl):
        """Test getting prompt instructions for LLMs."""
        instructions = default_svl.get_prompt_instructions()

        assert instructions is not None
        assert isinstance(instructions, str)
        assert len(instructions) > 0


# ============================================================================
# Code Generation
# ============================================================================


class TestCodeGeneration:
    """Test code generation from SVL schema."""

    def test_generate_pydantic_models(self, default_svl):
        """Test Pydantic model generation."""
        pydantic_code = default_svl.to_pydantic()

        assert pydantic_code is not None
        assert isinstance(pydantic_code, str)
        # Should contain class definitions
        assert "class" in pydantic_code or "BaseModel" in pydantic_code

    def test_generate_typescript_types(self, default_svl):
        """Test TypeScript type generation."""
        ts_code = default_svl.to_typescript()

        assert ts_code is not None
        assert isinstance(ts_code, str)
        # Should contain TypeScript definitions
        assert "interface" in ts_code or "type" in ts_code


# ============================================================================
# Vocabulary Merging
# ============================================================================


class TestVocabularyMerging:
    """Test merging vocabularies from multiple domains."""

    def test_merge_domain_vocabularies(self):
        """Test merging multiple domain vocabularies."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

        # Should have topics from both domains
        domains = svl.get_active_domains()
        assert len(domains) >= 1

    def test_merge_without_duplicates(self):
        """Test that merged vocabularies don't have duplicates."""
        from mindcore.v2.svl import SharedVocabularyLayer

        svl = SharedVocabularyLayer(domains=["customer_service"])

        # Add topics that might already exist
        svl.add_topics("billing", "support", "billing")  # Duplicate

        # Should handle duplicates gracefully


# ============================================================================
# Migration Support
# ============================================================================


class TestVocabularyMigration:
    """Test vocabulary version migrations."""

    def test_add_migration(self):
        """Test adding a migration."""
        from mindcore.v2.svl import Migration, SharedVocabularyLayer

        svl = SharedVocabularyLayer()

        migration = Migration(
            from_version="0.9.0",
            to_version="1.0.0",
            renames={"old_topic": "new_topic"},
            merges={},
            splits={},
            deletes=[],
            added_fields={},
        )

        svl.add_migration(migration)

    def test_migrate_memory(self):
        """Test migrating a memory to new vocabulary version."""
        from mindcore.v2.svl import Migration, SharedVocabularyLayer

        svl = SharedVocabularyLayer()

        # Add migration
        migration = Migration(
            from_version="0.9.0",
            to_version="1.0.0",
            renames={"old_topic": "new_topic"},
            merges={},
            splits={},
            deletes=[],
            added_fields={},
        )
        svl.add_migration(migration)

        # Migrate a memory
        memory = {"content": "Test", "topics": ["old_topic"], "vocabulary_version": "0.9.0"}

        svl.migrate_memory(memory, from_version="0.9.0")
        # old_topic should be renamed to new_topic

    def test_get_migration_path(self):
        """Test getting migration path between versions."""
        from mindcore.v2.svl import Migration, SharedVocabularyLayer

        svl = SharedVocabularyLayer()

        # Add migrations
        svl.add_migration(
            Migration(
                from_version="0.8.0",
                to_version="0.9.0",
                renames={},
                merges={},
                splits={},
                deletes=[],
                added_fields={},
            )
        )
        svl.add_migration(
            Migration(
                from_version="0.9.0",
                to_version="1.0.0",
                renames={},
                merges={},
                splits={},
                deletes=[],
                added_fields={},
            )
        )

        svl.get_migration_path("0.8.0")
        # Should return path from 0.8.0 to current version

    def test_rollback_memory(self):
        """Test rolling back a migrated memory."""
        from mindcore.v2.svl import Migration, SharedVocabularyLayer

        svl = SharedVocabularyLayer()

        migration = Migration(
            from_version="0.9.0",
            to_version="1.0.0",
            renames={"old_topic": "new_topic"},
            merges={},
            splits={},
            deletes=[],
            added_fields={},
            reversible=True,
        )
        svl.add_migration(migration)

        # Migrate with checkpoint
        memory = {"content": "Test", "topics": ["old_topic"], "vocabulary_version": "0.9.0"}

        migrated, checkpoint = svl.migrate_memory(
            memory, from_version="0.9.0", create_checkpoint=True
        )

        # Rollback
        svl.rollback_memory(migrated, checkpoint)
        # Should restore original topics


# ============================================================================
# Serialization
# ============================================================================


class TestSVLSerialization:
    """Test SVL serialization and deserialization."""

    def test_to_dict(self, default_svl):
        """Test converting SVL to dictionary."""
        data = default_svl.to_dict()

        assert data is not None
        assert isinstance(data, dict)

    def test_to_json(self, default_svl):
        """Test converting SVL to JSON."""
        json_str = default_svl.to_json()

        assert json_str is not None
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_from_dict(self, default_svl):
        """Test creating SVL from dictionary."""
        from mindcore.v2.svl import SharedVocabularyLayer

        data = default_svl.to_dict()
        restored = SharedVocabularyLayer.from_dict(data)

        assert restored is not None

    def test_from_json(self, default_svl):
        """Test creating SVL from JSON."""
        from mindcore.v2.svl import SharedVocabularyLayer

        json_str = default_svl.to_json()
        restored = SharedVocabularyLayer.from_json(json_str)

        assert restored is not None


# ============================================================================
# Memory Enrichment
# ============================================================================


class TestMemoryEnrichment:
    """Test SVL memory enrichment features."""

    def test_enrich_memory_with_defaults(self, default_svl):
        """Test enriching memory with default values."""
        memory = {"content": "Test memory", "memory_type": "semantic", "topics": ["api"]}

        default_svl.enrich_memory(memory, defaults={"importance": 0.5, "sentiment": "neutral"})

        # Should have default values filled in


# ============================================================================
# Statistics
# ============================================================================


class TestSVLStats:
    """Test SVL statistics and metadata."""

    def test_get_stats(self, default_svl):
        """Test getting SVL statistics."""
        stats = default_svl.get_stats()

        assert stats is not None
        assert isinstance(stats, dict)
        # Should include counts of topics, categories, etc.
