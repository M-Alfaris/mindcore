"""
Tests for Enrichment Agent.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from mindcore.agents import EnrichmentAgent
from mindcore.core.schemas import MessageMetadata
from mindcore.llm import LLMResponse


class TestEnrichmentAgent:
    """Test cases for EnrichmentAgent."""

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        return {
            "topics": ["AI", "machine learning"],
            "categories": ["question", "technical"],
            "importance": 0.8,
            "sentiment": {
                "overall": "neutral",
                "score": 0.5
            },
            "intent": "ask_question",
            "tags": ["AI", "ML", "question"],
            "entities": ["GPT", "OpenAI"],
            "key_phrases": ["best practices", "AI agents"]
        }

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.name = "mock"
        provider.is_available.return_value = True
        return provider

    @pytest.fixture
    def agent(self, mock_provider):
        """Create enrichment agent with mock provider."""
        return EnrichmentAgent(llm_provider=mock_provider)

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.temperature == 0.3
        assert agent.max_tokens == 800
        assert agent.system_prompt is not None
        assert agent.provider_name == "mock"

    def test_process_message(self, agent, mock_provider, mock_openai_response):
        """Test message processing."""
        import json

        # Mock LLM response
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(mock_openai_response),
            model="mock",
            provider="mock"
        )

        # Process message
        message_dict = {
            "user_id": "user_123",
            "thread_id": "thread_456",
            "session_id": "session_789",
            "role": "user",
            "text": "What are best practices for AI agents?"
        }

        message = agent.process(message_dict)

        # Assertions
        assert message.user_id == "user_123"
        assert message.thread_id == "thread_456"
        assert message.role.value == "user"
        assert message.raw_text == message_dict["text"]
        assert isinstance(message.metadata, MessageMetadata)
        assert "AI" in message.metadata.topics
        assert message.metadata.importance == 0.8

    def test_process_handles_api_failure(self, agent, mock_provider):
        """Test graceful handling of API failures."""
        # Mock API failure
        mock_provider.generate.side_effect = Exception("API Error")

        message_dict = {
            "user_id": "user_123",
            "thread_id": "thread_456",
            "session_id": "session_789",
            "role": "user",
            "text": "Test message"
        }

        # Should return message with default metadata
        message = agent.process(message_dict)

        assert message is not None
        assert isinstance(message.metadata, MessageMetadata)
        # Default metadata should be empty
        assert len(message.metadata.topics) == 0

    def test_enrich_batch(self, agent, mock_provider):
        """Test batch enrichment."""
        messages = [
            {
                "user_id": "user_123",
                "thread_id": "thread_456",
                "session_id": "session_789",
                "role": "user",
                "text": f"Message {i}"
            }
            for i in range(3)
        ]

        with patch.object(agent, 'process') as mock_process:
            from mindcore.core.schemas import Message, MessageMetadata
            mock_process.return_value = Message(
                message_id="msg_123",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role="user",
                raw_text="test",
                metadata=MessageMetadata()
            )

            enriched = agent.enrich_batch(messages)
            assert len(enriched) == 3
            assert mock_process.call_count == 3
