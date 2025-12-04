"""Tests for SmartContextAgent - the PRIMARY context retrieval mechanism."""

import json
from unittest.mock import Mock, patch

import pytest

from mindcore.agents import ContextTools, SmartContextAgent
from mindcore.core.schemas import AssembledContext, Message, MessageMetadata, MessageRole


class TestSmartContextAgent:
    """Test cases for SmartContextAgent."""

    @pytest.fixture
    def mock_context_tools(self, sample_messages):
        """Create mock context tools."""
        return ContextTools(
            get_recent_messages=Mock(return_value=sample_messages),
            search_history=Mock(return_value=sample_messages[:2]),
            get_session_metadata=Mock(
                return_value={
                    "topics": ["general", "testing"],
                    "categories": ["question"],
                    "intents": ["ask_question"],
                    "message_count": 5,
                }
            ),
            get_historical_summaries=Mock(return_value=[]),
            get_user_preferences=Mock(return_value=None),
            update_user_preference=Mock(return_value=(True, "Updated")),
            lookup_external_data=Mock(return_value=[]),
        )

    @pytest.fixture
    def agent(self, mock_llm_provider, mock_context_tools, mock_vocabulary):
        """Create SmartContextAgent with mocks."""
        with patch("mindcore.agents.smart_context_agent.get_vocabulary", return_value=mock_vocabulary):
            return SmartContextAgent(
                llm_provider=mock_llm_provider,
                context_tools=mock_context_tools,
                vocabulary=mock_vocabulary,
            )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.temperature == 0.2
        assert agent.max_tokens == 1500
        assert agent.max_tool_rounds == 3
        assert agent.system_prompt is not None
        assert agent.provider_name == "mock"
        assert len(agent._tool_definitions) == 7  # 7 tools defined

    def test_tool_definitions_structure(self, agent):
        """Test tool definitions have correct structure."""
        tools = agent._tool_definitions

        # Check each tool has required fields
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

        # Check specific tools exist
        tool_names = [t["function"]["name"] for t in tools]
        assert "get_recent_messages" in tool_names
        assert "search_history" in tool_names
        assert "get_session_metadata" in tool_names
        assert "get_historical_summaries" in tool_names
        assert "get_user_preferences" in tool_names
        assert "update_user_preference" in tool_names
        assert "lookup_external_data" in tool_names

    def test_execute_tool_get_recent_messages(self, agent, mock_context_tools, sample_messages):
        """Test executing get_recent_messages tool."""
        result = agent._execute_tool(
            tool_name="get_recent_messages",
            arguments={"limit": 10},
            user_id="user_123",
            thread_id="thread_456",
        )

        mock_context_tools.get_recent_messages.assert_called_once_with("user_123", "thread_456", 10)
        # The result contains formatted messages with role and content
        assert "Test message content" in result or "No messages found" in result

    def test_execute_tool_search_history(self, agent, mock_context_tools):
        """Test executing search_history tool."""
        result = agent._execute_tool(
            tool_name="search_history",
            arguments={
                "topics": ["general"],
                "categories": ["question"],
                "intent": "ask_question",
                "limit": 20,
            },
            user_id="user_123",
            thread_id="thread_456",
        )

        mock_context_tools.search_history.assert_called_once()
        assert isinstance(result, str)

    def test_execute_tool_get_session_metadata(self, agent, mock_context_tools):
        """Test executing get_session_metadata tool."""
        result = agent._execute_tool(
            tool_name="get_session_metadata",
            arguments={},
            user_id="user_123",
            thread_id="thread_456",
        )

        mock_context_tools.get_session_metadata.assert_called_once_with("user_123", "thread_456")
        # Result should be valid JSON
        data = json.loads(result)
        assert "topics" in data
        assert "message_count" in data

    def test_execute_tool_unknown(self, agent):
        """Test executing unknown tool returns error message."""
        result = agent._execute_tool(
            tool_name="unknown_tool",
            arguments={},
            user_id="user_123",
            thread_id="thread_456",
        )

        assert "Unknown tool" in result

    def test_execute_tool_handles_exception(self, agent, mock_context_tools):
        """Test tool execution handles exceptions gracefully."""
        mock_context_tools.get_recent_messages.side_effect = Exception("Database error")

        result = agent._execute_tool(
            tool_name="get_recent_messages",
            arguments={"limit": 10},
            user_id="user_123",
            thread_id="thread_456",
        )

        assert "Error executing" in result
        assert "Database error" in result

    def test_assess_conversation_state_early(self, agent, mock_context_tools):
        """Test conversation state assessment for early conversations."""
        # Mock only 1 message (below MIN_MESSAGES_FOR_CONTEXT=2)
        mock_context_tools.get_recent_messages.return_value = [
            Message(
                message_id="msg_0",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="Hello",
                metadata=MessageMetadata(),
            )
        ]

        state = agent._assess_conversation_state("user_123", "thread_456")

        assert state["is_early_conversation"] is True
        assert state["message_count"] == 1
        assert state["has_enriched_messages"] is False

    def test_assess_conversation_state_established(self, agent, mock_context_tools, sample_messages):
        """Test conversation state assessment for established conversations."""
        mock_context_tools.get_recent_messages.return_value = sample_messages

        state = agent._assess_conversation_state("user_123", "thread_456")

        assert state["is_early_conversation"] is False
        assert state["message_count"] == 5
        assert state["has_enriched_messages"] is True  # sample_messages have topics

    def test_assess_conversation_state_handles_error(self, agent, mock_context_tools):
        """Test conversation state assessment handles errors gracefully."""
        mock_context_tools.get_recent_messages.side_effect = Exception("DB Error")

        state = agent._assess_conversation_state("user_123", "thread_456")

        # Should return safe defaults
        assert state["is_early_conversation"] is True
        assert state["message_count"] == 0
        assert "error" in state

    def test_handle_early_conversation(self, agent):
        """Test early conversation handling returns context without LLM call."""
        state = {
            "is_early_conversation": True,
            "message_count": 1,
            "has_enriched_messages": False,
            "recent_messages": [],
        }

        context = agent._handle_early_conversation(
            query="Hello",
            user_id="user_123",
            thread_id="thread_456",
            state=state,
        )

        assert isinstance(context, AssembledContext)
        assert "early conversation" in context.assembled_context.lower()
        assert context.metadata["is_early_conversation"] is True
        assert context.metadata["context_source"] == "early_conversation"

    def test_handle_early_conversation_with_messages(self, agent, sample_messages):
        """Test early conversation handling includes available messages."""
        state = {
            "is_early_conversation": True,
            "message_count": 2,
            "has_enriched_messages": False,
            "recent_messages": sample_messages[:2],
        }

        context = agent._handle_early_conversation(
            query="What did I ask?",
            user_id="user_123",
            thread_id="thread_456",
            state=state,
        )

        assert isinstance(context, AssembledContext)
        assert "2 message" in context.assembled_context
        assert len(context.relevant_message_ids) == 2

    def test_create_fallback_context_exhausted(self, agent):
        """Test fallback context when tool calling is exhausted."""
        context = agent._create_fallback_context(
            query="test",
            reason="tool_calling_exhausted",
            message_count=3,
        )

        assert isinstance(context, AssembledContext)
        assert context.metadata["context_source"] == "fallback"
        assert context.metadata["confidence"] == "low"

    def test_create_fallback_context_limited_history(self, agent):
        """Test fallback context with limited history."""
        context = agent._create_fallback_context(
            query="test",
            reason="tool_calling_exhausted",
            message_count=2,  # Below MIN_MESSAGES_FOR_HISTORY_SEARCH
        )

        assert "Limited conversation history" in context.assembled_context

    def test_format_messages(self, agent, sample_messages):
        """Test message formatting for LLM."""
        formatted = agent._format_messages(sample_messages)

        assert isinstance(formatted, str)
        assert "[user]" in formatted.lower() or "[assistant]" in formatted.lower()
        assert "Test message content" in formatted

    def test_format_messages_empty(self, agent):
        """Test formatting empty message list."""
        formatted = agent._format_messages([])
        assert "No messages found" in formatted

    def test_format_messages_limits_length(self, agent):
        """Test message formatting limits to 30 messages."""
        many_messages = [
            Message(
                message_id=f"msg_{i}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text=f"Message {i}",
                metadata=MessageMetadata(),
            )
            for i in range(50)
        ]

        formatted = agent._format_messages(many_messages)
        # Should only contain first 30 messages
        assert "msg_29" in formatted or "Message 29" in formatted
        assert "msg_49" not in formatted

    def test_parse_final_response_valid_json(self, agent):
        """Test parsing valid JSON response."""
        content = json.dumps({
            "relevant_context": "User asked about AI",
            "key_points": ["Point 1", "Point 2"],
            "context_source": "recent",
            "confidence": "high",
        })

        context = agent._parse_final_response(content, "test query")

        assert isinstance(context, AssembledContext)
        assert context.assembled_context == "User asked about AI"
        assert len(context.key_points) == 2
        assert context.metadata["confidence"] == "high"

    def test_parse_final_response_invalid_json(self, agent):
        """Test parsing invalid JSON response falls back gracefully."""
        content = "This is not valid JSON"

        context = agent._parse_final_response(content, "test query")

        assert isinstance(context, AssembledContext)
        assert "parse_error" in context.metadata

    def test_process_early_conversation_skips_llm(self, agent, mock_llm_provider, mock_context_tools):
        """Test that early conversations skip LLM tool calling."""
        # Mock only 1 message
        mock_context_tools.get_recent_messages.return_value = [
            Message(
                message_id="msg_0",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="Hello",
                metadata=MessageMetadata(),
            )
        ]

        context = agent.process(
            query="Hello",
            user_id="user_123",
            thread_id="thread_456",
        )

        # LLM should NOT be called for early conversations
        mock_llm_provider.generate_with_tools.assert_not_called()
        assert context.metadata["is_early_conversation"] is True

    def test_process_with_tool_calling(self, agent, mock_llm_provider, mock_context_tools, sample_messages):
        """Test process with full tool calling flow."""
        mock_context_tools.get_recent_messages.return_value = sample_messages

        # Mock LLM response with tool calls, then final response
        mock_llm_provider.generate_with_tools.side_effect = [
            # First call: LLM wants to call a tool
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_recent_messages",
                            "arguments": '{"limit": 10}',
                        },
                    }
                ],
            },
            # Second call: LLM returns final response
            {
                "content": json.dumps({
                    "relevant_context": "User discussed testing",
                    "key_points": ["Testing focus"],
                    "context_source": "recent",
                    "confidence": "high",
                }),
                "tool_calls": None,
            },
        ]

        context = agent.process(
            query="What did we discuss?",
            user_id="user_123",
            thread_id="thread_456",
        )

        assert isinstance(context, AssembledContext)
        assert context.assembled_context == "User discussed testing"
        assert mock_llm_provider.generate_with_tools.call_count == 2

    def test_process_handles_llm_error(self, agent, mock_llm_provider, mock_context_tools, sample_messages):
        """Test process handles LLM errors gracefully."""
        mock_context_tools.get_recent_messages.return_value = sample_messages
        mock_llm_provider.generate_with_tools.side_effect = Exception("LLM Error")

        context = agent.process(
            query="Test",
            user_id="user_123",
            thread_id="thread_456",
        )

        # Should return fallback context
        assert isinstance(context, AssembledContext)
        assert context.metadata["context_source"] == "fallback"

    def test_process_with_additional_context(self, agent, mock_llm_provider, mock_context_tools, sample_messages):
        """Test process includes additional context."""
        mock_context_tools.get_recent_messages.return_value = sample_messages
        mock_llm_provider.generate_with_tools.return_value = {
            "content": json.dumps({
                "relevant_context": "Context assembled",
                "key_points": [],
                "context_source": "recent",
                "confidence": "medium",
            }),
            "tool_calls": None,
        }

        context = agent.process(
            query="Test",
            user_id="user_123",
            thread_id="thread_456",
            additional_context="This is extra context",
        )

        assert isinstance(context, AssembledContext)
        # Verify the LLM was called with additional context
        call_args = mock_llm_provider.generate_with_tools.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "additional context" in user_message["content"].lower()

    def test_refresh_vocabulary(self, agent, mock_vocabulary):
        """Test vocabulary refresh updates tool definitions."""
        original_tools = agent._tool_definitions.copy()

        # Modify vocabulary return values
        mock_vocabulary.get_topics.return_value = ["new_topic", "another_topic"]

        agent.refresh_vocabulary()

        # Tool definitions should be rebuilt
        assert agent._tool_definitions is not None
        mock_vocabulary.refresh_from_external.assert_called_once()

    def test_call_llm_with_tools_fallback(self, agent, mock_llm_provider):
        """Test fallback when provider doesn't support tools."""
        # Mock generate_with_tools to raise AttributeError to trigger fallback
        mock_llm_provider.generate_with_tools.side_effect = AttributeError("No tools support")
        # Configure the fallback generate method with proper response object
        mock_response = Mock()
        mock_response.content = '{"test": "response"}'
        mock_response.latency_ms = 100.0  # Required for logging
        mock_llm_provider.generate.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = agent._call_llm_with_tools(messages)

        assert "content" in result
        assert result["tool_calls"] is None


class TestContextToolsDataclass:
    """Test the ContextTools dataclass."""

    def test_context_tools_required_fields(self):
        """Test ContextTools requires mandatory callbacks."""
        tools = ContextTools(
            get_recent_messages=Mock(),
            search_history=Mock(),
            get_session_metadata=Mock(),
        )

        assert tools.get_recent_messages is not None
        assert tools.search_history is not None
        assert tools.get_session_metadata is not None
        # Optional fields should be None
        assert tools.get_historical_summaries is None
        assert tools.get_user_preferences is None

    def test_context_tools_optional_fields(self):
        """Test ContextTools with optional callbacks."""
        tools = ContextTools(
            get_recent_messages=Mock(),
            search_history=Mock(),
            get_session_metadata=Mock(),
            get_historical_summaries=Mock(),
            get_user_preferences=Mock(),
            update_user_preference=Mock(),
            lookup_external_data=Mock(),
        )

        assert tools.get_historical_summaries is not None
        assert tools.lookup_external_data is not None
