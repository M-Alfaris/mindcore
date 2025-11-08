"""
Tests for Enrichment Agent.
"""
import pytest
from unittest.mock import Mock, patch
from mindcore.agents import EnrichmentAgent
from mindcore.core.schemas import MessageMetadata


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
    def agent(self):
        """Create enrichment agent with mock API key."""
        return EnrichmentAgent(api_key="test-api-key")

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.model == "gpt-4o-mini"
        assert agent.temperature == 0.3
        assert agent.system_prompt is not None

    @patch('mindcore.agents.base_agent.BaseAgent._call_openai')
    def test_process_message(self, mock_call, agent, mock_openai_response):
        """Test message processing."""
        # Mock OpenAI response
        import json
        mock_call.return_value = json.dumps(mock_openai_response)

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

    @patch('mindcore.agents.base_agent.BaseAgent._call_openai')
    def test_process_handles_api_failure(self, mock_call, agent):
        """Test graceful handling of API failures."""
        # Mock API failure
        mock_call.side_effect = Exception("API Error")

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

    def test_enrich_batch(self, agent):
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
