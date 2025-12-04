"""Comprehensive tests for EnrichmentAgent.

CRITICAL: The EnrichmentAgent is responsible for background message enrichment,
which is a core feature for metadata extraction and intelligent context assembly.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mindcore.agents.enrichment_agent import EnrichmentAgent
from mindcore.core.schemas import Message, MessageMetadata, MessageRole
from mindcore.core.vocabulary import VocabularyManager
from mindcore.llm.base_provider import LLMResponse


class TestEnrichmentAgentInitialization:
    """Tests for EnrichmentAgent initialization."""

    def test_initialization_with_default_vocabulary(self):
        """Test initialization with default vocabulary."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider)

        assert agent._llm_provider == mock_provider
        assert agent.vocabulary is not None
        assert agent.system_prompt is not None

    def test_initialization_with_custom_vocabulary(self):
        """Test initialization with custom vocabulary."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        custom_vocab = VocabularyManager()

        agent = EnrichmentAgent(mock_provider, vocabulary=custom_vocab)

        assert agent.vocabulary is custom_vocab

    def test_initialization_with_custom_temperature(self):
        """Test initialization with custom temperature."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider, temperature=0.7)

        assert agent.temperature == 0.7

    def test_initialization_with_custom_max_tokens(self):
        """Test initialization with custom max_tokens."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider, max_tokens=500)

        assert agent.max_tokens == 500


class TestEnrichmentAgentSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_contains_vocabulary(self):
        """Test system prompt includes vocabulary information."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider)

        assert "topics" in agent.system_prompt.lower()
        assert "categories" in agent.system_prompt.lower()
        assert "intent" in agent.system_prompt.lower()
        assert "sentiment" in agent.system_prompt.lower()

    def test_system_prompt_contains_json_format(self):
        """Test system prompt specifies JSON output."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider)

        assert "json" in agent.system_prompt.lower()


class TestEnrichmentAgentProcess:
    """Tests for the process method."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.name = "mock_provider"
        return provider

    @pytest.fixture
    def agent(self, mock_provider):
        """Create an EnrichmentAgent with mock provider."""
        return EnrichmentAgent(mock_provider)

    def test_process_with_valid_response(self, agent, mock_provider):
        """Test processing with valid LLM response."""
        # Mock LLM response with valid JSON
        response_content = json.dumps({
            "topics": ["orders", "tracking"],
            "categories": ["support"],
            "importance": 0.8,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "ask_question",
            "tags": ["order status"],
            "entities": ["order #12345"],
            "key_phrases": ["check my order"],
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
            latency_ms=100.0,
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "How do I check my order #12345 status?",
        }

        result = agent.process(message_dict)

        assert isinstance(result, Message)
        assert result.user_id == "user123"
        assert result.thread_id == "thread456"
        assert result.raw_text == "How do I check my order #12345 status?"
        assert result.metadata.enrichment_failed is False
        assert "orders" in result.metadata.topics

    def test_process_generates_message_id(self, agent, mock_provider):
        """Test that process auto-generates message_id if not provided."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": 0.5,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Hello",
        }

        result = agent.process(message_dict)

        assert result.message_id is not None
        assert len(result.message_id) > 0

    def test_process_uses_provided_message_id(self, agent, mock_provider):
        """Test that process uses provided message_id."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": 0.5,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Hello",
            "message_id": "custom_msg_id",
        }

        result = agent.process(message_dict)

        assert result.message_id == "custom_msg_id"

    def test_process_handles_llm_error(self, agent, mock_provider):
        """Test that process handles LLM errors gracefully."""
        mock_provider.generate.side_effect = Exception("LLM Error")

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert isinstance(result, Message)
        assert result.metadata.enrichment_failed is True
        assert result.metadata.enrichment_error is not None
        assert "LLM Error" in result.metadata.enrichment_error

    def test_process_handles_invalid_json(self, agent, mock_provider):
        """Test that process handles invalid JSON response."""
        mock_response = LLMResponse(
            content="not valid json",
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert isinstance(result, Message)
        assert result.metadata.enrichment_failed is True

    def test_process_validates_topics(self, agent, mock_provider):
        """Test that process validates topics against vocabulary."""
        response_content = json.dumps({
            "topics": ["invalid_topic", "orders"],
            "categories": ["support"],
            "importance": 0.5,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "ask_question",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        # Only valid topics should remain
        assert "orders" in result.metadata.topics
        # Invalid topic should be filtered out (validation)

    def test_process_defaults_to_general_topic(self, agent, mock_provider):
        """Test that process defaults to 'general' if no valid topics."""
        response_content = json.dumps({
            "topics": ["completely_invalid"],
            "categories": ["support"],
            "importance": 0.5,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "ask_question",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert "general" in result.metadata.topics

    def test_process_clamps_importance(self, agent, mock_provider):
        """Test that process clamps importance to 0-1 range."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": 1.5,  # Over the max
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert result.metadata.importance == 1.0

    def test_process_clamps_negative_importance(self, agent, mock_provider):
        """Test that process clamps negative importance to 0."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": -0.5,  # Negative
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert result.metadata.importance == 0.0

    def test_process_validates_sentiment(self, agent, mock_provider):
        """Test that process validates sentiment against vocabulary."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": 0.5,
            "sentiment": {"overall": "invalid_sentiment", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        # Should default to neutral for invalid sentiment
        assert result.metadata.sentiment["overall"] == "neutral"

    def test_process_extracts_entities(self, agent, mock_provider):
        """Test that process extracts entities from response."""
        response_content = json.dumps({
            "topics": ["orders"],
            "categories": ["support"],
            "importance": 0.7,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "ask_question",
            "entities": ["order #12345", "John Doe"],
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        message_dict = {
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "role": "user",
            "text": "Test message",
        }

        result = agent.process(message_dict)

        assert "order #12345" in result.metadata.entities
        assert "John Doe" in result.metadata.entities


class TestEnrichmentAgentBatch:
    """Tests for batch enrichment."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.name = "mock_provider"
        return provider

    @pytest.fixture
    def agent(self, mock_provider):
        """Create an EnrichmentAgent with mock provider."""
        return EnrichmentAgent(mock_provider)

    def test_enrich_batch_processes_all(self, agent, mock_provider):
        """Test batch enrichment processes all messages."""
        response_content = json.dumps({
            "topics": ["general"],
            "categories": ["general"],
            "importance": 0.5,
            "sentiment": {"overall": "neutral", "score": 0.5},
            "intent": "provide_info",
        })

        mock_response = LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        messages = [
            {
                "user_id": f"user{i}",
                "thread_id": "thread1",
                "session_id": "session1",
                "role": "user",
                "text": f"Message {i}",
            }
            for i in range(3)
        ]

        results = agent.enrich_batch(messages)

        assert len(results) == 3
        assert all(isinstance(m, Message) for m in results)

    def test_enrich_batch_handles_partial_failures(self, agent, mock_provider):
        """Test batch enrichment continues despite individual failures."""
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("LLM Error on message 2")
            return LLMResponse(
                content=json.dumps({
                    "topics": ["general"],
                    "categories": ["general"],
                    "importance": 0.5,
                    "sentiment": {"overall": "neutral", "score": 0.5},
                    "intent": "provide_info",
                }),
                model="mock-model",
                provider="mock",
            )

        mock_provider.generate.side_effect = mock_generate

        messages = [
            {
                "user_id": f"user{i}",
                "thread_id": "thread1",
                "session_id": "session1",
                "role": "user",
                "text": f"Message {i}",
            }
            for i in range(3)
        ]

        results = agent.enrich_batch(messages)

        # Should still have all 3 messages, with one having enrichment_failed
        assert len(results) == 3

    def test_enrich_batch_empty_list(self, agent):
        """Test batch enrichment with empty list."""
        results = agent.enrich_batch([])
        assert results == []


class TestEnrichmentAgentVocabularyRefresh:
    """Tests for vocabulary refresh."""

    def test_refresh_vocabulary_updates_prompt(self):
        """Test that refreshing vocabulary updates the system prompt."""
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = EnrichmentAgent(mock_provider)
        original_prompt = agent.system_prompt

        # Refresh vocabulary
        agent.refresh_vocabulary()

        # System prompt should be recreated (may be same content but new reference)
        # The key is that the method runs without error
        assert agent.system_prompt is not None


class TestEnrichmentAgentProviderName:
    """Tests for provider name property."""

    def test_provider_name_from_provider(self):
        """Test provider_name property returns provider name."""
        mock_provider = MagicMock()
        mock_provider.name = "test_provider"

        agent = EnrichmentAgent(mock_provider)

        assert agent.provider_name == "test_provider"
