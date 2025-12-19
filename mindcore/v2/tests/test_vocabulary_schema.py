"""Comprehensive tests for vocabulary schema module."""

import json

import pytest

from mindcore.v2.vocabulary import (
    VocabularySchema,
    DEFAULT_VOCABULARY,
    MemoryType,
    Sentiment,
    AccessLevel,
)
from mindcore.v2.vocabulary.schema import FieldSchema, Migration


class TestEnums:
    """Test vocabulary enums."""

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.ENTITY.value == "entity"
        assert MemoryType.RELATIONSHIP.value == "relationship"
        assert MemoryType.TEMPORAL.value == "temporal"
        assert MemoryType.WORKING.value == "working"

    def test_sentiment_values(self):
        """Test Sentiment enum values."""
        assert Sentiment.POSITIVE.value == "positive"
        assert Sentiment.NEGATIVE.value == "negative"
        assert Sentiment.NEUTRAL.value == "neutral"
        assert Sentiment.MIXED.value == "mixed"

    def test_access_level_values(self):
        """Test AccessLevel enum values."""
        assert AccessLevel.PRIVATE.value == "private"
        assert AccessLevel.TEAM.value == "team"
        assert AccessLevel.SHARED.value == "shared"
        assert AccessLevel.GLOBAL.value == "global"


class TestFieldSchema:
    """Test FieldSchema dataclass."""

    def test_create_string_field(self):
        """Test creating a string field."""
        field = FieldSchema(
            name="custom_field",
            field_type="string",
            description="A custom field",
        )

        assert field.name == "custom_field"
        assert field.field_type == "string"
        assert field.required is False
        assert field.default is None

    def test_create_enum_field(self):
        """Test creating an enum field."""
        field = FieldSchema(
            name="priority",
            field_type="enum",
            enum_values=["low", "medium", "high"],
            required=True,
        )

        assert field.field_type == "enum"
        assert field.enum_values == ["low", "medium", "high"]
        assert field.required is True

    def test_create_array_field(self):
        """Test creating an array field."""
        field = FieldSchema(
            name="tags",
            field_type="array",
            default=[],
        )

        assert field.field_type == "array"
        assert field.default == []


class TestMigration:
    """Test Migration dataclass."""

    def test_apply_to_topics_rename(self):
        """Test applying rename migration to topics."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            renames={"old_topic": "new_topic"},
        )

        result = migration.apply_to_topics(["old_topic", "other"])

        assert "new_topic" in result
        assert "old_topic" not in result
        assert "other" in result

    def test_apply_to_topics_delete(self):
        """Test applying delete migration to topics."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            deletes=["deprecated_topic"],
        )

        result = migration.apply_to_topics(["deprecated_topic", "kept"])

        assert "deprecated_topic" not in result
        assert "kept" in result

    def test_apply_to_topics_merge(self):
        """Test applying merge migration to topics."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            merges={"billing": ["payment", "invoice"]},
        )

        result = migration.apply_to_topics(["payment", "invoice", "other"])

        assert "billing" in result
        assert "payment" not in result
        assert "invoice" not in result
        assert "other" in result

    def test_apply_to_categories(self):
        """Test applying migration to categories."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            renames={"old_cat": "new_cat"},
        )

        result = migration.apply_to_categories(["old_cat", "other"])

        assert "new_cat" in result
        assert "old_cat" not in result


class TestVocabularySchemaCreation:
    """Test VocabularySchema creation."""

    def test_create_basic_schema(self):
        """Test creating a basic vocabulary schema."""
        vocab = VocabularySchema(version="1.0.0")

        assert vocab.version == "1.0.0"
        assert vocab.topics == []
        assert len(vocab.memory_types) > 0
        assert len(vocab.sentiments) > 0

    def test_create_with_topics(self):
        """Test creating schema with topics."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support", "sales"],
        )

        assert "billing" in vocab.topics
        assert "support" in vocab.topics
        assert "sales" in vocab.topics

    def test_create_with_categories(self):
        """Test creating schema with categories."""
        vocab = VocabularySchema(
            version="1.0.0",
            categories=["inquiry", "complaint", "feedback"],
        )

        assert "inquiry" in vocab.categories
        assert "complaint" in vocab.categories

    def test_create_with_custom_fields(self):
        """Test creating schema with custom fields."""
        custom = FieldSchema(
            name="priority",
            field_type="enum",
            enum_values=["low", "medium", "high"],
        )
        vocab = VocabularySchema(
            version="1.0.0",
            custom_fields=[custom],
        )

        assert len(vocab.custom_fields) == 1
        assert vocab.custom_fields[0].name == "priority"

    def test_default_vocabulary_exists(self):
        """Test that DEFAULT_VOCABULARY is properly defined."""
        assert DEFAULT_VOCABULARY is not None
        assert DEFAULT_VOCABULARY.version == "1.0.0"
        assert len(DEFAULT_VOCABULARY.topics) > 0
        assert len(DEFAULT_VOCABULARY.categories) > 0


class TestVocabularySchemaJsonSchema:
    """Test JSON Schema generation."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry", "complaint"],
        )

    def test_to_json_schema_structure(self, vocab):
        """Test basic JSON schema structure."""
        schema = vocab.to_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "response" in schema["properties"]
        assert "memories_to_store" in schema["properties"]

    def test_to_json_schema_memory_structure(self, vocab):
        """Test memory schema structure."""
        schema = vocab.to_json_schema()
        memory_schema = schema["properties"]["memories_to_store"]["items"]

        assert "content" in memory_schema["properties"]
        assert "memory_type" in memory_schema["properties"]
        assert "topics" in memory_schema["properties"]

    def test_to_json_schema_topics_enum(self, vocab):
        """Test that topics are enum in schema."""
        schema = vocab.to_json_schema()
        memory_schema = schema["properties"]["memories_to_store"]["items"]
        topics_schema = memory_schema["properties"]["topics"]

        assert topics_schema["items"]["enum"] == ["billing", "support"]

    def test_to_json_schema_without_response(self, vocab):
        """Test generating schema without response field."""
        schema = vocab.to_json_schema(include_response=False)

        assert "response" not in schema.get("properties", schema)
        assert "content" in schema["properties"]

    def test_to_json_schema_with_custom_field(self):
        """Test JSON schema includes custom fields."""
        custom = FieldSchema(
            name="priority",
            field_type="enum",
            enum_values=["low", "medium", "high"],
            required=True,
            description="Priority level",
        )
        vocab = VocabularySchema(
            version="1.0.0",
            custom_fields=[custom],
        )

        schema = vocab.to_json_schema(include_response=False)

        assert "priority" in schema["properties"]
        assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]
        assert "priority" in schema["required"]


class TestVocabularySchemaCodeGen:
    """Test code generation."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry"],
        )

    def test_to_pydantic(self, vocab):
        """Test Pydantic code generation."""
        code = vocab.to_pydantic()

        assert "class Memory(BaseModel)" in code
        assert "class AgentResponse(BaseModel)" in code
        assert "content: str" in code
        assert "memory_type:" in code
        assert "billing" in code

    def test_to_typescript(self, vocab):
        """Test TypeScript code generation."""
        code = vocab.to_typescript()

        assert "export interface Memory" in code
        assert "export interface AgentResponse" in code
        assert "content: string" in code
        assert '"billing"' in code

    def test_to_prompt_instructions(self, vocab):
        """Test prompt instructions generation."""
        instructions = vocab.to_prompt_instructions()

        assert "billing" in instructions
        assert "support" in instructions
        assert "episodic" in instructions
        assert "TOPICS" in instructions
        assert "MEMORY TYPES" in instructions


class TestVocabularySchemaValidation:
    """Test memory validation."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry", "complaint"],
        )

    def test_validate_valid_memory(self, vocab):
        """Test validating a valid memory."""
        memory = {
            "content": "Test memory",
            "memory_type": "episodic",
            "topics": ["billing"],
            "categories": ["inquiry"],
            "importance": 0.7,
            "sentiment": "neutral",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_content(self, vocab):
        """Test validation fails for missing content."""
        memory = {
            "memory_type": "episodic",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("content" in e for e in errors)

    def test_validate_missing_memory_type(self, vocab):
        """Test validation fails for missing memory_type."""
        memory = {
            "content": "Test",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("memory_type" in e for e in errors)

    def test_validate_invalid_memory_type(self, vocab):
        """Test validation fails for invalid memory_type."""
        memory = {
            "content": "Test",
            "memory_type": "invalid_type",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("memory_type" in e for e in errors)

    def test_validate_invalid_topics(self, vocab):
        """Test validation fails for invalid topics."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "topics": ["invalid_topic"],
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("topics" in e for e in errors)

    def test_validate_invalid_categories(self, vocab):
        """Test validation fails for invalid categories."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "categories": ["invalid_category"],
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("categories" in e for e in errors)

    def test_validate_invalid_sentiment(self, vocab):
        """Test validation fails for invalid sentiment."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "sentiment": "very_positive",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("sentiment" in e for e in errors)

    def test_validate_invalid_importance(self, vocab):
        """Test validation fails for importance out of range."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "importance": 1.5,
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("importance" in e for e in errors)

    def test_validate_invalid_importance_negative(self, vocab):
        """Test validation fails for negative importance."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "importance": -0.5,
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("importance" in e for e in errors)

    def test_validate_invalid_access_level(self, vocab):
        """Test validation fails for invalid access_level."""
        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "access_level": "public",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("access_level" in e for e in errors)

    def test_validate_missing_required_custom_field(self):
        """Test validation fails for missing required custom field."""
        custom = FieldSchema(
            name="priority",
            field_type="enum",
            enum_values=["low", "high"],
            required=True,
        )
        vocab = VocabularySchema(
            version="1.0.0",
            custom_fields=[custom],
        )

        memory = {
            "content": "Test",
            "memory_type": "episodic",
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("priority" in e for e in errors)

    def test_validate_invalid_custom_field_enum(self):
        """Test validation fails for invalid custom field value."""
        custom = FieldSchema(
            name="priority",
            field_type="enum",
            enum_values=["low", "high"],
        )
        vocab = VocabularySchema(
            version="1.0.0",
            custom_fields=[custom],
        )

        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "priority": "medium",  # Not in enum
        }

        is_valid, errors = vocab.validate(memory)

        assert is_valid is False
        assert any("priority" in e for e in errors)


class TestVocabularySchemaMigration:
    """Test vocabulary migration."""

    def test_migrate_memory_same_version(self):
        """Test that same version returns unchanged."""
        vocab = VocabularySchema(version="1.0.0")
        memory = {"content": "Test", "memory_type": "episodic"}

        result = vocab.migrate_memory(memory, from_version="1.0.0")

        assert result == memory

    def test_migrate_memory_no_path(self):
        """Test error when no migration path exists."""
        vocab = VocabularySchema(version="2.0.0")

        with pytest.raises(ValueError) as exc_info:
            vocab.migrate_memory({}, from_version="1.0.0")

        assert "migration path" in str(exc_info.value).lower()

    def test_migrate_memory_with_topics(self):
        """Test migrating memory with topic changes."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            renames={"old_topic": "new_topic"},
        )
        vocab = VocabularySchema(
            version="2.0.0",
            migrations={"1.0.0": migration},
        )

        memory = {
            "content": "Test",
            "memory_type": "episodic",
            "topics": ["old_topic", "other"],
        }

        result = vocab.migrate_memory(memory, from_version="1.0.0")

        assert "new_topic" in result["topics"]
        assert "old_topic" not in result["topics"]

    def test_migrate_memory_with_added_fields(self):
        """Test migrating with new fields."""
        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            added_fields={"new_field": "default_value"},
        )
        vocab = VocabularySchema(
            version="2.0.0",
            migrations={"1.0.0": migration},
        )

        memory = {
            "content": "Test",
            "memory_type": "episodic",
        }

        result = vocab.migrate_memory(memory, from_version="1.0.0")

        assert result["new_field"] == "default_value"


class TestVocabularySchemaSerialization:
    """Test serialization and deserialization."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry"],
            description="Test vocabulary",
        )

    def test_to_dict(self, vocab):
        """Test converting to dictionary."""
        data = vocab.to_dict()

        assert data["version"] == "1.0.0"
        assert data["topics"] == ["billing", "support"]
        assert data["categories"] == ["inquiry"]
        assert data["description"] == "Test vocabulary"
        assert "created_at" in data

    def test_to_json(self, vocab):
        """Test converting to JSON."""
        json_str = vocab.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "version": "2.0.0",
            "topics": ["topic1", "topic2"],
            "categories": ["cat1"],
            "description": "Test",
        }

        vocab = VocabularySchema.from_dict(data)

        assert vocab.version == "2.0.0"
        assert vocab.topics == ["topic1", "topic2"]

    def test_from_json(self):
        """Test creating from JSON."""
        json_str = '{"version": "1.0.0", "topics": ["test"]}'

        vocab = VocabularySchema.from_json(json_str)

        assert vocab.version == "1.0.0"
        assert vocab.topics == ["test"]

    def test_roundtrip(self, vocab):
        """Test serialization roundtrip."""
        json_str = vocab.to_json()
        restored = VocabularySchema.from_json(json_str)

        assert restored.version == vocab.version
        assert restored.topics == vocab.topics
        assert restored.categories == vocab.categories

    def test_from_dict_with_custom_fields(self):
        """Test creating from dict with custom fields."""
        data = {
            "version": "1.0.0",
            "custom_fields": [
                {
                    "name": "priority",
                    "field_type": "enum",
                    "required": True,
                    "enum_values": ["low", "high"],
                    "default": None,
                    "description": "Priority",
                }
            ],
        }

        vocab = VocabularySchema.from_dict(data)

        assert len(vocab.custom_fields) == 1
        assert vocab.custom_fields[0].name == "priority"
        assert vocab.custom_fields[0].required is True
