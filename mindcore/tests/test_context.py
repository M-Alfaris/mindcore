"""
Tests for Context Assembler Agent.
"""
import pytest
from unittest.mock import patch
from mindcore.agents import ContextAssemblerAgent
from mindcore.core.schemas import Message, MessageMetadata, AssembledContext


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
                "Implement error handling"
            ],
            "relevant_message_ids": ["msg_1", "msg_2", "msg_3"],
            "metadata": {
                "topics": ["AI", "development"],
                "sentiment": {
                    "overall": "positive",
                    "trend": "stable"
                },
                "importance": 0.8
            }
        }

    @pytest.fixture
    def agent(self):
        """Create context assembler agent with mock API key."""
        return ContextAssemblerAgent(api_key="test-api-key")

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
                metadata=MessageMetadata(
                    topics=["AI", "development"],
                    importance=0.5 + (i * 0.1)
                )
            )
            messages.append(msg)
        return messages

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.model == "gpt-4o-mini"
        assert agent.temperature == 0.3
        assert agent.system_prompt is not None

    @patch('mindcore.agents.base_agent.BaseAgent._call_openai')
    def test_process_context(self, mock_call, agent, sample_messages, mock_context_response):
        """Test context assembly."""
        import json
        mock_call.return_value = json.dumps(mock_context_response)

        query = "What have we discussed about AI?"
        context = agent.process(sample_messages, query)

        # Assertions
        assert isinstance(context, AssembledContext)
        assert context.assembled_context != ""
        assert len(context.key_points) > 0
        assert len(context.relevant_message_ids) > 0
        assert "topics" in context.metadata

    @patch('mindcore.agents.base_agent.BaseAgent._call_openai')
    def test_process_handles_api_failure(self, mock_call, agent, sample_messages):
        """Test graceful handling of API failures."""
        mock_call.side_effect = Exception("API Error")

        query = "Test query"
        context = agent.process(sample_messages, query)

        # Should return empty context with error
        assert isinstance(context, AssembledContext)
        assert "error" in context.metadata or context.assembled_context != ""

    def test_format_messages(self, agent, sample_messages):
        """Test message formatting."""
        formatted = agent._format_messages(sample_messages)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Should contain message IDs and content
        assert "msg_0" in formatted
        assert "Message content" in formatted

    @patch('mindcore.agents.base_agent.BaseAgent._call_openai')
    def test_assemble_for_prompt(self, mock_call, agent, sample_messages, mock_context_response):
        """Test assembling context for prompt injection."""
        import json
        mock_call.return_value = json.dumps(mock_context_response)

        query = "AI development"
        formatted = agent.assemble_for_prompt(sample_messages, query)

        assert isinstance(formatted, str)
        assert "Historical Context" in formatted or len(formatted) > 0
