"""Comprehensive tests for extraction module."""

import pytest

from mindcore.v2.extraction import MemoryExtractor, ExtractionResult
from mindcore.v2.vocabulary import VocabularySchema


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_create_result(self):
        """Test creating an extraction result."""
        result = ExtractionResult(
            memories=[],
            extraction_latency_ms=10.5,
        )

        assert result.memories == []
        assert result.extraction_latency_ms == 10.5


class TestMemoryExtractorInit:
    """Test MemoryExtractor initialization."""

    def test_requires_vocabulary(self):
        """Test that vocabulary is required."""
        with pytest.raises(ValueError) as exc_info:
            MemoryExtractor(vocabulary=None)

        assert "vocabulary is required" in str(exc_info.value).lower()

    def test_init_with_vocabulary(self):
        """Test initialization with vocabulary."""
        vocab = VocabularySchema(version="1.0.0", topics=["test"])

        extractor = MemoryExtractor(vocabulary=vocab)

        assert extractor.vocabulary is vocab


class TestMemoryExtractorExtract:
    """Test MemoryExtractor extract method."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support", "product"],
            categories=["inquiry", "complaint"],
        )

    @pytest.fixture
    def extractor(self, vocab):
        """Create extractor."""
        return MemoryExtractor(vocabulary=vocab)

    def test_extract_single_memory(self, extractor):
        """Test extracting a single memory."""
        output = {
            "response": "I'll help you with that.",
            "memories_to_store": [
                {
                    "content": "User prefers email communication",
                    "memory_type": "preference",
                    "topics": ["billing"],
                    "importance": 0.8,
                }
            ],
        }

        result = extractor.extract(output, user_id="user_123")

        assert isinstance(result, ExtractionResult)
        assert len(result.memories) == 1
        assert result.memories[0].content == "User prefers email communication"
        assert result.memories[0].memory_type == "preference"
        assert result.memories[0].user_id == "user_123"

    def test_extract_multiple_memories(self, extractor):
        """Test extracting multiple memories."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Memory 1",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                },
                {
                    "content": "Memory 2",
                    "memory_type": "semantic",
                    "topics": ["support"],
                },
            ],
        }

        result = extractor.extract(output, user_id="user_123")

        assert len(result.memories) == 2

    def test_extract_with_agent_id(self, extractor):
        """Test extraction with agent ID."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test memory",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                }
            ],
        }

        result = extractor.extract(
            output,
            user_id="user_123",
            agent_id="agent_456",
        )

        assert result.memories[0].agent_id == "agent_456"

    def test_extract_empty_list(self, extractor):
        """Test extracting with empty memories list."""
        output = {
            "response": "No memories to store.",
            "memories_to_store": [],
        }

        result = extractor.extract(output, user_id="user_123")

        assert len(result.memories) == 0

    def test_extract_no_memories_key(self, extractor):
        """Test extracting when memories_to_store is missing."""
        output = {
            "response": "Just a response.",
        }

        result = extractor.extract(output, user_id="user_123")

        assert len(result.memories) == 0

    def test_extract_sets_vocabulary_version(self, extractor):
        """Test that extraction sets vocabulary version."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                }
            ],
        }

        result = extractor.extract(output, user_id="user_123")

        assert result.memories[0].vocabulary_version == "1.0.0"

    def test_extract_sets_defaults(self, extractor):
        """Test that extraction sets default values."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Minimal memory",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                }
            ],
        }

        result = extractor.extract(output, user_id="user_123")
        memory = result.memories[0]

        assert memory.sentiment == "neutral"
        assert memory.importance == 0.5
        assert memory.access_level == "private"
        assert memory.topics == ["billing"]
        assert memory.categories == []


class TestMemoryExtractorErrors:
    """Test MemoryExtractor error handling."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
        )

    @pytest.fixture
    def extractor(self, vocab):
        """Create extractor."""
        return MemoryExtractor(vocabulary=vocab)

    def test_fails_on_non_dict_output(self, extractor):
        """Test that extraction fails on non-dict output."""
        with pytest.raises(TypeError) as exc_info:
            extractor.extract("not a dict", user_id="user_123")

        assert "expected dict" in str(exc_info.value).lower()

    def test_fails_on_non_list_memories(self, extractor):
        """Test that extraction fails when memories_to_store is not a list."""
        output = {
            "response": "Done.",
            "memories_to_store": "not a list",
        }

        with pytest.raises(TypeError) as exc_info:
            extractor.extract(output, user_id="user_123")

        assert "must be a list" in str(exc_info.value).lower()

    def test_fails_on_non_dict_memory(self, extractor):
        """Test that extraction fails when memory is not a dict."""
        output = {
            "response": "Done.",
            "memories_to_store": ["not a dict"],
        }

        with pytest.raises(TypeError) as exc_info:
            extractor.extract(output, user_id="user_123")

        assert "expected dict" in str(exc_info.value).lower()

    def test_fails_on_missing_content(self, extractor):
        """Test that extraction fails when content is missing."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "memory_type": "episodic",
                    # content is missing
                }
            ],
        }

        with pytest.raises(KeyError) as exc_info:
            extractor.extract(output, user_id="user_123")

        assert "content" in str(exc_info.value).lower()

    def test_fails_on_missing_memory_type(self, extractor):
        """Test that extraction fails when memory_type is missing."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test content",
                    # memory_type is missing
                }
            ],
        }

        with pytest.raises(KeyError) as exc_info:
            extractor.extract(output, user_id="user_123")

        assert "memory_type" in str(exc_info.value).lower()

    def test_fails_on_invalid_topic(self, extractor):
        """Test that extraction fails on invalid topic."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test",
                    "memory_type": "episodic",
                    "topics": ["invalid_topic"],
                }
            ],
        }

        with pytest.raises(ValueError) as exc_info:
            extractor.extract(output, user_id="user_123")

        assert "validation failed" in str(exc_info.value).lower()


class TestMemoryExtractorValidate:
    """Test MemoryExtractor validate_output method."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        return VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
        )

    @pytest.fixture
    def extractor(self, vocab):
        """Create extractor."""
        return MemoryExtractor(vocabulary=vocab)

    def test_validate_valid_output(self, extractor):
        """Test validating valid output."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                }
            ],
        }

        # Should not raise
        extractor.validate_output(output)

    def test_validate_fails_on_non_dict(self, extractor):
        """Test that validation fails on non-dict."""
        with pytest.raises(TypeError):
            extractor.validate_output("not a dict")

    def test_validate_fails_on_missing_response(self, extractor):
        """Test that validation fails when response is missing."""
        output = {
            "memories_to_store": [],
        }

        with pytest.raises(KeyError) as exc_info:
            extractor.validate_output(output)

        assert "response" in str(exc_info.value).lower()

    def test_validate_fails_on_non_list_memories(self, extractor):
        """Test validation fails when memories is not a list."""
        output = {
            "response": "Done.",
            "memories_to_store": "not a list",
        }

        with pytest.raises(TypeError):
            extractor.validate_output(output)

    def test_validate_fails_on_invalid_memory(self, extractor):
        """Test validation fails on invalid memory."""
        output = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test",
                    "memory_type": "episodic",
                    "topics": ["invalid_topic"],
                }
            ],
        }

        with pytest.raises(ValueError):
            extractor.validate_output(output)
