"""
Tests for Context Assembler Agent.
"""

import pytest
from unittest.mock import Mock, patch
from mindcore.agents import ContextAssemblerAgent
from mindcore.core.schemas import Message, MessageMetadata, AssembledContext
from mindcore.llm import LLMResponse


class TestContextAssemblerAgent:
    """Test cases for ContextAssemblerAgent."""

    @pytest.fixture
    def mock_context_response(self):
        """Mock context assembly response."""
        return {
            "assembled_context": "User has been discussing AI agent development and best practices.",
            "key_points": [
                "Focus on memory management",
                "Use structured prompts",
                "Implement error handling",
            ],
            "relevant_message_ids": ["msg_1", "msg_2", "msg_3"],
            "metadata": {
                "topics": ["AI", "development"],
                "sentiment": {"overall": "positive", "trend": "stable"},
                "importance": 0.8,
            },
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
        """Create context assembler agent with mock provider."""
        return ContextAssemblerAgent(llm_provider=mock_provider)

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages."""
        messages = []
        for i in range(5):
            msg = Message(
                message_id=f"msg_{i}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role="user" if i % 2 == 0 else "assistant",
                raw_text=f"Message content {i}",
                metadata=MessageMetadata(topics=["AI", "development"], importance=0.5 + (i * 0.1)),
            )
            messages.append(msg)
        return messages

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.temperature == 0.3
        assert agent.max_tokens == 1500
        assert agent.system_prompt is not None
        assert agent.provider_name == "mock"

    def test_process_context(self, agent, mock_provider, sample_messages, mock_context_response):
        """Test context assembly."""
        import json

        # Mock LLM response
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(mock_context_response), model="mock", provider="mock"
        )

        query = "What have we discussed about AI?"
        context = agent.process(sample_messages, query)

        # Assertions
        assert isinstance(context, AssembledContext)
        assert context.assembled_context != ""
        assert len(context.key_points) > 0
        assert len(context.relevant_message_ids) > 0
        assert "topics" in context.metadata

    def test_process_handles_api_failure(self, agent, mock_provider, sample_messages):
        """Test graceful handling of API failures."""
        mock_provider.generate.side_effect = Exception("API Error")

        query = "Test query"
        context = agent.process(sample_messages, query)

        # Should return empty context with error metadata
        assert isinstance(context, AssembledContext)
        assert context.metadata.get("assembly_failed") is True

    def test_format_messages(self, agent, sample_messages):
        """Test message formatting."""
        formatted = agent._format_messages(sample_messages)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Should contain message IDs and content
        assert "msg_0" in formatted
        assert "Message content" in formatted

    def test_assemble_for_prompt(
        self, agent, mock_provider, sample_messages, mock_context_response
    ):
        """Test assembling context for prompt injection."""
        import json

        # Mock LLM response
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(mock_context_response), model="mock", provider="mock"
        )

        query = "AI development"
        formatted = agent.assemble_for_prompt(sample_messages, query)

        assert isinstance(formatted, str)
        assert "Historical Context" in formatted or len(formatted) > 0
