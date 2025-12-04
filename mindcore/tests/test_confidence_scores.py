"""Tests for confidence scoring in enrichment."""

import pytest

from mindcore.core.schemas import EnrichmentSource, MessageMetadata


class TestMessageMetadataConfidence:
    """Tests for MessageMetadata confidence fields."""

    def test_default_confidence_score(self):
        """Test that default confidence score is 0.0."""
        metadata = MessageMetadata()
        assert metadata.confidence_score == 0.0

    def test_confidence_score_range(self):
        """Test confidence score values."""
        metadata = MessageMetadata(confidence_score=0.85)
        assert metadata.confidence_score == 0.85

    def test_is_high_confidence(self):
        """Test high confidence detection."""
        high = MessageMetadata(confidence_score=0.8)
        assert high.is_high_confidence is True

        medium = MessageMetadata(confidence_score=0.5)
        assert medium.is_high_confidence is False

        low = MessageMetadata(confidence_score=0.2)
        assert low.is_high_confidence is False

    def test_is_low_confidence(self):
        """Test low confidence detection."""
        low = MessageMetadata(confidence_score=0.2)
        assert low.is_low_confidence is True

        medium = MessageMetadata(confidence_score=0.5)
        assert medium.is_low_confidence is False

        high = MessageMetadata(confidence_score=0.8)
        assert high.is_low_confidence is False

    def test_enrichment_source_values(self):
        """Test enrichment source enum values."""
        assert EnrichmentSource.LLM.value == "llm"
        assert EnrichmentSource.TRIVIAL.value == "trivial"
        assert EnrichmentSource.FALLBACK.value == "fallback"
        assert EnrichmentSource.CACHED.value == "cached"
        assert EnrichmentSource.MANUAL.value == "manual"

    def test_metadata_with_enrichment_source(self):
        """Test metadata with enrichment source."""
        metadata = MessageMetadata(
            topics=["billing"],
            confidence_score=0.9,
            enrichment_source=EnrichmentSource.LLM.value,
            vocabulary_match_rate=0.95,
        )

        assert metadata.enrichment_source == "llm"
        assert metadata.vocabulary_match_rate == 0.95

    def test_metadata_to_dict_includes_confidence(self):
        """Test that to_dict includes confidence fields."""
        metadata = MessageMetadata(
            topics=["orders"],
            confidence_score=0.75,
            enrichment_source="llm",
            enrichment_latency_ms=150.5,
            vocabulary_match_rate=0.9,
        )

        data = metadata.to_dict()

        assert "confidence_score" in data
        assert data["confidence_score"] == 0.75
        assert data["enrichment_source"] == "llm"
        assert data["enrichment_latency_ms"] == 150.5
        assert data["vocabulary_match_rate"] == 0.9

    def test_failed_enrichment_has_zero_confidence(self):
        """Test that failed enrichment has zero confidence."""
        metadata = MessageMetadata(
            enrichment_failed=True,
            enrichment_error="LLM timeout",
            confidence_score=0.0,
            enrichment_source=EnrichmentSource.FALLBACK.value,
        )

        assert metadata.enrichment_failed is True
        assert metadata.confidence_score == 0.0
        assert metadata.is_low_confidence is True


class TestEnrichmentAgentConfidence:
    """Tests for confidence scoring in EnrichmentAgent."""

    def test_confidence_calculation_factors(self):
        """Test that confidence is calculated based on multiple factors."""
        from mindcore.agents.enrichment_agent import EnrichmentAgent

        # Create a mock to test the calculation method
        class MockProvider:
            name = "mock"

        agent = EnrichmentAgent.__new__(EnrichmentAgent)
        agent.vocabulary = None  # Will be set by test

        # Test high confidence scenario
        score = agent._calculate_confidence_score(
            vocab_match_rate=1.0,
            used_default_topics=False,
            used_default_categories=False,
            sentiment_valid=True,
            has_entities=True,
            has_key_phrases=True,
            text_length=100,
        )
        assert score >= 0.8

        # Test low confidence scenario
        score = agent._calculate_confidence_score(
            vocab_match_rate=0.3,
            used_default_topics=True,
            used_default_categories=True,
            sentiment_valid=False,
            has_entities=False,
            has_key_phrases=False,
            text_length=5,
        )
        assert score < 0.5

    def test_vocabulary_match_rate_calculation(self):
        """Test vocabulary match rate calculation."""
        from mindcore.agents.enrichment_agent import EnrichmentAgent

        agent = EnrichmentAgent.__new__(EnrichmentAgent)

        # All matches
        result = agent._calculate_vocabulary_match_rate(
            raw_topics=["billing", "orders"],
            validated_topics=["billing", "orders"],
            raw_categories=["support"],
            validated_categories=["support"],
            raw_intent="ask_question",
            validated_intent="ask_question",
        )
        assert result["match_rate"] == 1.0

        # Partial matches
        result = agent._calculate_vocabulary_match_rate(
            raw_topics=["billing", "unknown_topic"],
            validated_topics=["billing"],
            raw_categories=["support"],
            validated_categories=["support"],
            raw_intent="invalid_intent",
            validated_intent=None,
        )
        assert result["match_rate"] < 1.0
        assert result["topics_matched"] == 1
        assert result["topics_total"] == 2
