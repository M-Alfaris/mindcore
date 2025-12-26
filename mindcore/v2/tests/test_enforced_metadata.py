"""Tests for SVL Enforced Metadata - LLM metadata extraction.

Tests cover:
- ContextDecision and EnforcedMetadata dataclasses
- MetadataExtractor: prompts, parsing, validation
- JSON extraction from LLM responses
"""

import json
from datetime import datetime, timezone

import pytest

from mindcore.v2.svl.enforced_metadata import (
    ContextDecision,
    EnforcedMetadata,
    HistoricalContextNeeded,
    MetadataExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create MetadataExtractor without SVL."""
    return MetadataExtractor(svl=None, strict_validation=False)


@pytest.fixture
def strict_extractor():
    """Create MetadataExtractor with strict validation."""
    return MetadataExtractor(svl=None, strict_validation=True)


# =============================================================================
# HistoricalContextNeeded Tests
# =============================================================================


class TestHistoricalContextNeeded:
    """Tests for HistoricalContextNeeded enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert HistoricalContextNeeded.TRUE.value == "True"
        assert HistoricalContextNeeded.FALSE.value == "False"


# =============================================================================
# ContextDecision Tests
# =============================================================================


class TestContextDecision:
    """Tests for ContextDecision dataclass."""

    def test_create_decision(self):
        """Test creating a context decision."""
        decision = ContextDecision(
            historical_context_needed=HistoricalContextNeeded.TRUE,
            suggested_topics=["orders", "shipping"],
            suggested_categories=["support"],
            reasoning="User asks about previous order",
            urgency="high",
            confidence="high",
        )

        assert decision.needs_clst() is True
        assert decision.suggested_topics == ["orders", "shipping"]

    def test_needs_clst_false(self):
        """Test needs_clst returns False when not needed."""
        decision = ContextDecision(
            historical_context_needed=HistoricalContextNeeded.FALSE,
        )

        assert decision.needs_clst() is False

    def test_to_dict(self):
        """Test serialization to dict."""
        decision = ContextDecision(
            historical_context_needed=HistoricalContextNeeded.TRUE,
            suggested_topics=["orders"],
            reasoning="Test",
        )

        data = decision.to_dict()

        assert data["historical_context_needed"] == "True"
        assert data["suggested_topics"] == ["orders"]
        assert data["reasoning"] == "Test"

    def test_from_dict_true(self):
        """Test deserialization with True value."""
        data = {
            "historical_context_needed": "True",
            "suggested_topics": ["orders"],
            "reasoning": "Past reference",
        }

        decision = ContextDecision.from_dict(data)

        assert decision.needs_clst() is True
        assert decision.suggested_topics == ["orders"]

    def test_from_dict_false(self):
        """Test deserialization with False value."""
        data = {
            "historical_context_needed": "False",
            "suggested_topics": [],
        }

        decision = ContextDecision.from_dict(data)

        assert decision.needs_clst() is False

    def test_from_dict_case_insensitive(self):
        """Test deserialization handles case variations."""
        data = {"historical_context_needed": "true"}
        decision = ContextDecision.from_dict(data)
        assert decision.needs_clst() is True

        data = {"historical_context_needed": "TRUE"}
        decision = ContextDecision.from_dict(data)
        assert decision.needs_clst() is True

    def test_from_dict_defaults(self):
        """Test deserialization with missing fields."""
        data = {}

        decision = ContextDecision.from_dict(data)

        assert decision.needs_clst() is False
        assert decision.suggested_topics == []
        assert decision.urgency == "medium"
        assert decision.confidence == "high"


# =============================================================================
# EnforcedMetadata Tests
# =============================================================================


class TestEnforcedMetadata:
    """Tests for EnforcedMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating enforced metadata."""
        metadata = EnforcedMetadata(
            message_id="msg_123",
            user_id="user_456",
            session_id="session_789",
            topics=["orders", "shipping"],
            categories=["support"],
            importance=0.8,
        )

        assert metadata.message_id == "msg_123"
        assert metadata.topics == ["orders", "shipping"]
        assert metadata.importance == 0.8

    def test_default_values(self):
        """Test default values are set."""
        metadata = EnforcedMetadata(
            message_id="msg_1",
            user_id="user_1",
            session_id="session_1",
        )

        assert metadata.message_type == "statement"
        assert metadata.message_intent == "provide_info"
        assert metadata.importance == 0.5
        assert metadata.confidence == 0.8
        assert metadata.urgency == "medium"
        assert metadata.sentiment == "neutral"
        assert metadata.memory_type == "episodic"
        assert metadata.access_level == "private"

    def test_to_dict(self):
        """Test serialization to dict."""
        metadata = EnforcedMetadata(
            message_id="msg_1",
            user_id="user_1",
            session_id="session_1",
            topics=["topic1"],
            importance=0.7,
        )

        data = metadata.to_dict()

        assert data["message_id"] == "msg_1"
        assert data["topics"] == ["topic1"]
        assert data["importance"] == 0.7
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "message_id": "msg_1",
            "user_id": "user_1",
            "session_id": "session_1",
            "topics": ["orders"],
            "categories": ["support"],
            "importance": 0.9,
        }

        metadata = EnforcedMetadata.from_dict(data)

        assert metadata.message_id == "msg_1"
        assert metadata.topics == ["orders"]
        assert metadata.importance == 0.9

    def test_from_dict_generates_message_id(self):
        """Test that from_dict generates message_id if missing."""
        data = {
            "user_id": "user_1",
            "session_id": "session_1",
        }

        metadata = EnforcedMetadata.from_dict(data)

        assert metadata.message_id.startswith("msg_")

    def test_from_dict_parses_datetime(self):
        """Test that from_dict parses datetime strings."""
        data = {
            "message_id": "msg_1",
            "user_id": "user_1",
            "session_id": "session_1",
            "created_at": "2025-01-15T10:30:00+00:00",
        }

        metadata = EnforcedMetadata.from_dict(data)

        assert isinstance(metadata.created_at, datetime)


# =============================================================================
# MetadataExtractor Prompt Tests
# =============================================================================


class TestMetadataExtractorPrompts:
    """Tests for MetadataExtractor prompt generation."""

    def test_get_context_decision_prompt(self, extractor):
        """Test context decision prompt generation."""
        prompt = extractor.get_context_decision_prompt(
            user_message="What was my last order?",
        )

        assert "What was my last order?" in prompt
        assert "historical_context_needed" in prompt
        assert "True" in prompt
        assert "False" in prompt

    def test_get_context_decision_prompt_with_session(self, extractor):
        """Test context decision prompt with session context."""
        prompt = extractor.get_context_decision_prompt(
            user_message="And the status?",
            session_context="User asked about order #12345",
        )

        assert "And the status?" in prompt
        assert "order #12345" in prompt

    def test_get_extraction_prompt(self, extractor):
        """Test metadata extraction prompt generation."""
        prompt = extractor.get_extraction_prompt(
            user_message="I want to cancel my order",
            session_id="session_1",
            user_id="user_1",
        )

        assert "I want to cancel my order" in prompt
        assert "session_1" in prompt
        assert "topics" in prompt.lower()
        assert "categories" in prompt.lower()
        assert "importance" in prompt.lower()

    def test_get_extraction_prompt_with_response(self, extractor):
        """Test extraction prompt with agent response."""
        prompt = extractor.get_extraction_prompt(
            user_message="Cancel my order",
            agent_response="I'll help you cancel your order #12345",
            session_id="session_1",
            user_id="user_1",
        )

        assert "Cancel my order" in prompt
        assert "cancel your order #12345" in prompt

    def test_get_extraction_prompt_no_memory(self, extractor):
        """Test extraction prompt without memory extraction."""
        prompt = extractor.get_extraction_prompt(
            user_message="Hello",
            include_memory_extraction=False,
        )

        assert "Hello" in prompt
        # Memory extraction section should be minimal


# =============================================================================
# MetadataExtractor JSON Schema Tests
# =============================================================================


class TestMetadataExtractorSchema:
    """Tests for MetadataExtractor JSON schema generation."""

    def test_get_json_schema(self, extractor):
        """Test JSON schema generation."""
        schema = extractor.get_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Check required fields
        required = schema["required"]
        assert "topics" in required
        assert "categories" in required
        assert "message_type" in required
        assert "importance" in required

    def test_schema_properties(self, extractor):
        """Test schema has expected properties."""
        schema = extractor.get_json_schema()
        props = schema["properties"]

        assert "historical_context_needed" in props
        assert "topics" in props
        assert "categories" in props
        assert "message_type" in props
        assert "message_intent" in props
        assert "importance" in props
        assert "urgency" in props
        assert "sentiment" in props

    def test_schema_importance_bounds(self, extractor):
        """Test importance has correct bounds in schema."""
        schema = extractor.get_json_schema()
        importance = schema["properties"]["importance"]

        assert importance["minimum"] == 0.0
        assert importance["maximum"] == 1.0


# =============================================================================
# MetadataExtractor Parsing Tests
# =============================================================================


class TestMetadataExtractorParsing:
    """Tests for MetadataExtractor parsing methods."""

    def test_parse_context_decision_dict(self, extractor):
        """Test parsing context decision from dict."""
        data = {
            "historical_context_needed": "True",
            "suggested_topics": ["orders"],
            "reasoning": "User mentions past",
        }

        decision = extractor.parse_context_decision(data)

        assert decision.needs_clst() is True
        assert decision.suggested_topics == ["orders"]

    def test_parse_context_decision_json_string(self, extractor):
        """Test parsing context decision from JSON string."""
        json_str = json.dumps(
            {
                "historical_context_needed": "True",
                "suggested_topics": ["billing"],
            }
        )

        decision = extractor.parse_context_decision(json_str)

        assert decision.needs_clst() is True

    def test_parse_metadata_dict(self, extractor):
        """Test parsing metadata from dict."""
        data = {
            "topics": ["orders"],
            "categories": ["support"],
            "message_type": "query",
            "message_intent": "ask_question",
            "importance": 0.8,
        }

        metadata, _memories = extractor.parse_metadata(
            data,
            user_id="user_1",
            session_id="session_1",
        )

        assert metadata.topics == ["orders"]
        assert metadata.importance == 0.8
        assert metadata.user_id == "user_1"

    def test_parse_metadata_json_string(self, extractor):
        """Test parsing metadata from JSON string."""
        json_str = json.dumps(
            {
                "topics": ["shipping"],
                "categories": ["logistics"],
                "message_type": "statement",
                "importance": 0.5,
            }
        )

        metadata, _ = extractor.parse_metadata(json_str)

        assert metadata.topics == ["shipping"]

    def test_parse_metadata_extracts_memories(self, extractor):
        """Test parsing extracts memories_to_store."""
        data = {
            "topics": ["preferences"],
            "categories": ["user_preference"],
            "importance": 0.7,
            "memories_to_store": [
                {"content": "User prefers dark mode", "importance": 0.8},
                {"content": "User likes email updates", "importance": 0.5},
            ],
        }

        _metadata, memories = extractor.parse_metadata(data)

        assert len(memories) == 2
        assert memories[0]["content"] == "User prefers dark mode"

    def test_parse_metadata_injects_ids(self, extractor):
        """Test parsing injects user_id and session_id."""
        data = {
            "topics": ["test"],
            "categories": ["test"],
            "importance": 0.5,
        }

        metadata, _ = extractor.parse_metadata(
            data,
            user_id="injected_user",
            session_id="injected_session",
        )

        assert metadata.user_id == "injected_user"
        assert metadata.session_id == "injected_session"

    def test_parse_metadata_generates_message_id(self, extractor):
        """Test parsing generates message_id if missing."""
        data = {
            "topics": ["test"],
            "categories": ["test"],
            "importance": 0.5,
        }

        metadata, _ = extractor.parse_metadata(data)

        assert metadata.message_id.startswith("msg_")


# =============================================================================
# MetadataExtractor JSON Extraction Tests
# =============================================================================


class TestJSONExtraction:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_pure_json(self, extractor):
        """Test extracting pure JSON response."""
        response = '{"historical_context_needed": "True", "topics": ["orders"]}'

        data = extractor._extract_json(response)

        assert data["historical_context_needed"] == "True"

    def test_extract_json_with_markdown(self, extractor):
        """Test extracting JSON from markdown code block."""
        response = """Here's my analysis:

```json
{"historical_context_needed": "False", "topics": ["general"]}
```

Let me explain..."""

        data = extractor._extract_json(response)

        assert data["historical_context_needed"] == "False"

    def test_extract_json_with_text(self, extractor):
        """Test extracting JSON embedded in text."""
        response = """Based on the user's query, here is my decision:

{"historical_context_needed": "True", "suggested_topics": ["billing"], "reasoning": "User asks about past charges"}

This is because the user mentioned previous billing."""

        data = extractor._extract_json(response)

        assert data["historical_context_needed"] == "True"
        assert data["suggested_topics"] == ["billing"]

    def test_extract_json_invalid(self, extractor):
        """Test extracting from invalid JSON raises error."""
        response = "This is not JSON at all"

        with pytest.raises(ValueError, match="Could not extract JSON"):
            extractor._extract_json(response)


# =============================================================================
# MetadataExtractor Validation Tests
# =============================================================================


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_validate_returns_tuple(self, extractor):
        """Test validate returns (is_valid, errors) tuple."""
        metadata = EnforcedMetadata(
            message_id="msg_1",
            user_id="user_1",
            session_id="session_1",
            topics=["orders"],
            categories=["support"],
        )

        is_valid, errors = extractor.validate(metadata)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_dict_input(self, extractor):
        """Test validate accepts dict input."""
        data = {
            "message_id": "msg_1",
            "topics": ["orders"],
            "categories": ["support"],
        }

        is_valid, _errors = extractor.validate(data)

        # Without SVL, should pass
        assert is_valid is True

    def test_validate_without_svl_passes(self, extractor):
        """Test validation passes without SVL configured."""
        metadata = EnforcedMetadata(
            message_id="msg_1",
            user_id="user_1",
            session_id="session_1",
            topics=["anything"],
            categories=["anything"],
        )

        is_valid, errors = extractor.validate(metadata)

        assert is_valid is True
        assert errors == []


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_topics_raw_without_svl(self, extractor):
        """Test getting topics without SVL returns empty list."""
        topics = extractor._get_topics_raw()
        assert topics == []

    def test_get_categories_raw_without_svl(self, extractor):
        """Test getting categories without SVL returns empty list."""
        categories = extractor._get_categories_raw()
        assert categories == []

    def test_get_message_types_raw_without_svl(self, extractor):
        """Test getting message types without SVL returns defaults."""
        types = extractor._get_message_types_raw()
        assert "query" in types
        assert "statement" in types

    def test_get_message_intents_raw_without_svl(self, extractor):
        """Test getting message intents without SVL returns defaults."""
        intents = extractor._get_message_intents_raw()
        assert "ask_question" in intents
        assert "provide_info" in intents

    def test_get_memory_types_raw_without_svl(self, extractor):
        """Test getting memory types without SVL returns defaults."""
        types = extractor._get_memory_types_raw()
        assert "episodic" in types
        assert "semantic" in types
        assert "preference" in types

    def test_get_sentiments_raw_without_svl(self, extractor):
        """Test getting sentiments without SVL returns defaults."""
        sentiments = extractor._get_sentiments_raw()
        assert "positive" in sentiments
        assert "negative" in sentiments
        assert "neutral" in sentiments

    def test_get_emotional_raw_without_svl(self, extractor):
        """Test getting emotional classifications without SVL returns defaults."""
        emotional = extractor._get_emotional_raw()
        assert "neutral" in emotional
        assert "joy" in emotional

    def test_get_temporal_raw_without_svl(self, extractor):
        """Test getting temporal qualifiers without SVL returns defaults."""
        temporal = extractor._get_temporal_raw()
        assert "past_event" in temporal
        assert "current" in temporal


# =============================================================================
# Feedback Enhancement Tests
# =============================================================================


class TestFeedbackEnhancement:
    """Tests for feedback-enhanced metadata extraction."""

    def test_get_extraction_prompt_with_feedback(self, extractor):
        """Test extraction prompt enhanced with feedback."""
        feedback = {
            "high_quality_topics": [("orders", 0.9), ("shipping", 0.85)],
            "low_quality_topics": [("misc", 0.2)],
            "high_quality_categories": [("support", 0.88)],
            "low_quality_categories": [("general", 0.15)],
        }

        prompt = extractor.get_extraction_prompt_with_feedback(
            user_message="What's my order status?",
            feedback=feedback,
        )

        assert "orders" in prompt
        assert "shipping" in prompt
        assert "misc" in prompt
        assert "Quality Feedback" in prompt

    def test_get_extraction_prompt_empty_feedback(self, extractor):
        """Test extraction prompt with empty feedback."""
        feedback = {}

        prompt = extractor.get_extraction_prompt_with_feedback(
            user_message="Hello",
            feedback=feedback,
        )

        # Should return base prompt without feedback section
        assert "Hello" in prompt

    def test_get_extraction_prompt_with_guidance(self, extractor):
        """Test extraction prompt with natural language guidance."""
        feedback = {
            "guidance": "Focus on customer support topics",
        }

        prompt = extractor.get_extraction_prompt_with_feedback(
            user_message="I need help",
            feedback=feedback,
        )

        assert "customer support topics" in prompt

    def test_get_effectiveness_enhanced_schema(self, extractor):
        """Test schema enhanced with effectiveness info."""
        feedback = {
            "high_quality_topics": [("orders", 0.9), ("billing", 0.85)],
            "low_quality_topics": [("misc", 0.1)],
        }

        schema = extractor.get_effectiveness_enhanced_schema(feedback)

        # Schema should still be valid
        assert schema["type"] == "object"
        # Topics description should include effectiveness info
        topic_desc = schema["properties"]["topics"]["description"]
        assert "orders" in topic_desc or "Effective" in topic_desc


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_user_message(self, extractor):
        """Test handling empty user message."""
        prompt = extractor.get_extraction_prompt(
            user_message="",
            session_id="s1",
            user_id="u1",
        )

        assert "session_id" in prompt

    def test_special_characters_in_message(self, extractor):
        """Test handling special characters in message."""
        prompt = extractor.get_extraction_prompt(
            user_message='User said: "Hello & goodbye" <script>',
            session_id="s1",
            user_id="u1",
        )

        assert "Hello & goodbye" in prompt

    def test_unicode_in_message(self, extractor):
        """Test handling unicode in message."""
        prompt = extractor.get_extraction_prompt(
            user_message="Hello! Bonjour! こんにちは! 🎉",
            session_id="s1",
            user_id="u1",
        )

        assert "Hello!" in prompt
        assert "こんにちは" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
