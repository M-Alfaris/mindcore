"""Tests for SVL metadata extraction fallback strategies."""

import json
import pytest
from datetime import datetime, timezone

from mindcore.v2.svl.extraction_fallback import (
    ExtractionStrategy,
    ExtractionFailureType,
    ExtractionResult,
    ExtractionAttempt,
    RuleBasedExtractor,
    ResilientMetadataExtractor,
    BatchItem,
)


class TestRuleBasedExtractor:
    """Tests for RuleBasedExtractor."""

    def test_extract_basic_message(self):
        """Test basic extraction without SVL."""
        extractor = RuleBasedExtractor(svl=None)

        result = extractor.extract(
            user_message="I want a refund for my order #12345",
            user_id="user_123",
            session_id="session_abc",
        )

        assert result["user_id"] == "user_123"
        assert result["session_id"] == "session_abc"
        assert "message_id" in result
        assert result["message_type"] in ["query", "command", "statement", "feedback"]
        assert "#12345" in result["entities"]

    def test_detect_question(self):
        """Test question detection."""
        extractor = RuleBasedExtractor(svl=None)

        result = extractor.extract("What is my order status?")
        assert result["message_type"] == "query"
        assert result["message_intent"] == "ask_question"

    def test_detect_command(self):
        """Test command detection."""
        extractor = RuleBasedExtractor(svl=None)

        result = extractor.extract("Please cancel my subscription")
        assert result["message_type"] == "command"
        assert result["message_intent"] == "request_action"

    def test_detect_urgency(self):
        """Test urgency detection."""
        extractor = RuleBasedExtractor(svl=None)

        # Critical urgency
        result = extractor.extract("This is an emergency! I need help immediately!")
        assert result["urgency"] == "critical"

        # Low urgency
        result = extractor.extract("When you can, could you look into this? No rush.")
        assert result["urgency"] == "low"

    def test_detect_sentiment(self):
        """Test sentiment detection."""
        extractor = RuleBasedExtractor(svl=None)

        # Positive
        result = extractor.extract("Great job! Thanks for the help!")
        assert result["sentiment"] == "positive"

        # Negative
        result = extractor.extract("This is terrible. I'm very frustrated.")
        assert result["sentiment"] == "negative"

    def test_extract_entities(self):
        """Test entity extraction."""
        extractor = RuleBasedExtractor(svl=None)

        result = extractor.extract(
            "Contact me at john.doe@example.com about order #ABC123"
        )
        assert "john.doe@example.com" in result["entities"]
        assert "#ABC123" in result["entities"]

    def test_importance_estimation(self):
        """Test importance estimation."""
        extractor = RuleBasedExtractor(svl=None)

        # High importance
        result = extractor.extract("URGENT: Critical issue with production!")
        assert result["importance"] >= 0.8

        # Low importance
        result = extractor.extract("Just curious, btw, random question...")
        assert result["importance"] <= 0.4


class TestResilientMetadataExtractor:
    """Tests for ResilientMetadataExtractor."""

    def test_primary_extraction_success(self):
        """Test that primary extraction works."""
        extractor = ResilientMetadataExtractor(svl=None, max_retries=1)

        # Mock LLM that returns valid JSON
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "topics": ["billing"],
                "categories": ["inquiry"],
                "message_type": "query",
                "message_intent": "ask_question",
                "importance": 0.5,
                "sentiment": "neutral",
            })

        result = extractor.extract_with_fallback(
            user_message="What's my balance?",
            llm_call=mock_llm,
            user_id="user_123",
            session_id="session_abc",
        )

        assert result.success
        assert result.strategy_used == ExtractionStrategy.PRIMARY
        assert result.quality_score == 1.0
        assert not result.needs_review

    def test_retry_with_errors(self):
        """Test that retry with errors works."""
        extractor = ResilientMetadataExtractor(svl=None, max_retries=2)

        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return "invalid json {"
            elif call_count == 2:
                return "still invalid"
            else:
                return json.dumps({
                    "topics": ["billing"],
                    "categories": ["inquiry"],
                    "message_type": "query",
                    "message_intent": "ask_question",
                    "importance": 0.5,
                    "sentiment": "neutral",
                })

        result = extractor.extract_with_fallback(
            user_message="What's my balance?",
            llm_call=mock_llm,
        )

        # Should have tried multiple times
        assert call_count >= 2
        assert len(result.attempts) >= 2

    def test_rule_based_fallback(self):
        """Test rule-based fallback when LLM fails."""
        extractor = ResilientMetadataExtractor(svl=None, max_retries=1)

        # Mock LLM that always fails
        def mock_llm(prompt: str) -> str:
            raise Exception("LLM unavailable")

        result = extractor.extract_with_fallback(
            user_message="I want a refund",
            llm_call=mock_llm,
        )

        assert result.success
        assert result.strategy_used in [
            ExtractionStrategy.RULE_BASED,
            ExtractionStrategy.DEFAULT_ASSIGNMENT,
        ]
        assert result.needs_review
        assert result.quality_score < 1.0

    def test_default_assignment_last_resort(self):
        """Test default assignment as last resort."""
        extractor = ResilientMetadataExtractor(
            svl=None,
            max_retries=0,
            enable_batch_queue=False,
        )

        def mock_llm(prompt: str) -> str:
            raise Exception("Always fails")

        result = extractor.extract_with_fallback(
            user_message="Test message",
            llm_call=mock_llm,
            user_id="user_123",
            skip_strategies=[
                ExtractionStrategy.PRIMARY,
                ExtractionStrategy.RETRY_WITH_ERRORS,
                ExtractionStrategy.RETRY_SIMPLIFIED,
                ExtractionStrategy.BATCH_CONTEXT,
                ExtractionStrategy.RULE_BASED,
            ],
        )

        assert result.success
        assert result.strategy_used == ExtractionStrategy.DEFAULT_ASSIGNMENT
        assert result.quality_score == 0.2
        assert result.needs_review
        assert result.metadata is not None
        assert result.metadata["topics"] == ["untagged"]
        assert result.metadata["categories"] == ["needs_review"]

    def test_batch_context_extraction(self):
        """Test batch context extraction with neighbor messages."""
        extractor = ResilientMetadataExtractor(svl=None, max_retries=0)

        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1

            # Primary fails
            if "PREVIOUS ATTEMPT" not in prompt and "Context" not in prompt:
                raise Exception("Primary fails")

            # Batch context succeeds
            if "Context" in prompt:
                return json.dumps({
                    "topics": ["orders"],
                    "categories": ["inquiry"],
                    "message_type": "query",
                    "importance": 0.6,
                    "sentiment": "neutral",
                })

            raise Exception("Other strategies fail")

        result = extractor.extract_with_fallback(
            user_message="What about the other one?",
            llm_call=mock_llm,
            neighbor_messages=["I ordered two items", "First item arrived"],
            skip_strategies=[
                ExtractionStrategy.PRIMARY,
                ExtractionStrategy.RETRY_WITH_ERRORS,
                ExtractionStrategy.RETRY_SIMPLIFIED,
            ],
        )

        assert result.success
        assert result.strategy_used == ExtractionStrategy.BATCH_CONTEXT

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        extractor = ResilientMetadataExtractor(svl=None, max_retries=0)

        # Successful extraction
        def success_llm(prompt: str) -> str:
            return json.dumps({
                "topics": ["test"],
                "categories": ["test"],
                "message_type": "statement",
                "importance": 0.5,
                "sentiment": "neutral",
            })

        result = extractor.extract_with_fallback(
            user_message="Test",
            llm_call=success_llm,
        )

        stats = extractor.get_stats()
        assert stats["total_extractions"] == 1
        assert stats["primary_success_rate"] > 0


class TestBatchQueue:
    """Tests for batch queue processing."""

    def test_queue_for_batch(self):
        """Test that failed extractions are queued."""
        extractor = ResilientMetadataExtractor(
            svl=None,
            max_retries=0,
            enable_batch_queue=True,
        )

        def fail_llm(prompt: str) -> str:
            raise Exception("Fail")

        result = extractor.extract_with_fallback(
            user_message="Test message",
            llm_call=fail_llm,
            user_id="user_1",
            session_id="session_1",
        )

        assert result.queued_for_batch or result.strategy_used in [
            ExtractionStrategy.RULE_BASED,
            ExtractionStrategy.DEFAULT_ASSIGNMENT,
        ]

    def test_batch_processing(self):
        """Test batch processing of queued items."""
        extractor = ResilientMetadataExtractor(
            svl=None,
            max_retries=0,
            enable_batch_queue=True,
        )

        # Queue some items manually
        extractor._batch_queue.append(BatchItem(
            message_id="msg_1",
            user_message="First message",
            agent_response=None,
            session_id="session_1",
            user_id="user_1",
        ))
        extractor._batch_queue.append(BatchItem(
            message_id="msg_2",
            user_message="Second message",
            agent_response=None,
            session_id="session_1",
            user_id="user_1",
        ))

        def batch_llm(prompt: str) -> str:
            return json.dumps({
                "topics": ["batch_processed"],
                "categories": ["test"],
                "message_type": "statement",
                "importance": 0.5,
                "sentiment": "neutral",
            })

        results = extractor.process_batch_queue(batch_llm)

        assert len(results) == 2
        assert all(r.success for r in results)


class TestExtractionAttempt:
    """Tests for ExtractionAttempt."""

    def test_attempt_creation(self):
        """Test attempt creation."""
        attempt = ExtractionAttempt(
            strategy=ExtractionStrategy.PRIMARY,
            success=True,
            duration_ms=50.0,
        )

        assert attempt.success
        assert attempt.strategy == ExtractionStrategy.PRIMARY
        assert attempt.failure_type is None

    def test_failure_attempt(self):
        """Test failure attempt."""
        attempt = ExtractionAttempt(
            strategy=ExtractionStrategy.PRIMARY,
            success=False,
            failure_type=ExtractionFailureType.INVALID_JSON,
            failure_details="Expected '{' at position 0",
        )

        assert not attempt.success
        assert attempt.failure_type == ExtractionFailureType.INVALID_JSON


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
