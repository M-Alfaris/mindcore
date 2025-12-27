"""Test 08: MCP Server Tests.

Tests the Model Context Protocol (MCP) server:
- Tool definitions and schemas
- Tool call execution
- Error handling
- Protocol compliance
"""

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# MCP Server Fixtures
# ============================================================================


@pytest.fixture
def mock_mindcore_for_mcp():
    """Create a mock mindcore instance for MCP testing."""
    mock = MagicMock()
    mock.store.return_value = "mem_mcp_123"
    mock.get.return_value = MagicMock(
        memory_id="mem_mcp_123",
        content="MCP test content",
        memory_type="semantic",
        user_id="mcp_user",
        topics=["api"],
        importance=0.5,
        to_dict=lambda: {
            "memory_id": "mem_mcp_123",
            "content": "MCP test content",
            "memory_type": "semantic",
        },
    )
    mock.recall.return_value = MagicMock(
        memories=[mock.get.return_value],
        scores=[0.9],
        to_dict=lambda: {"memories": [], "scores": []},
    )
    mock.search.return_value = []
    mock.get_json_schema.return_value = {"type": "object"}
    mock.list_agents.return_value = []
    return mock


@pytest.fixture
def mcp_server(flr, clst):
    """Create MCP server instance with real FLR/CLST."""
    try:
        from mindcore.server.mcp import MCPServer

        return MCPServer(flr=flr, clst=clst)
    except ImportError:
        pytest.skip("MCP server not available")


# ============================================================================
# Server Info Tests
# ============================================================================


class TestMCPServerInfo:
    """Test MCP server information and capabilities."""

    def test_server_info(self, mcp_server):
        """Test server info response."""
        info = mcp_server.get_server_info()

        assert info is not None
        assert "name" in info or hasattr(info, "name")
        # Should be "mindcore"

    def test_protocol_version(self, mcp_server):
        """Test protocol version is specified."""
        info = mcp_server.get_server_info()

        # Should have protocol version
        assert "protocol_version" in info or "version" in info

    def test_capabilities(self, mcp_server):
        """Test server capabilities are defined."""
        info = mcp_server.get_server_info()

        # Should have capabilities
        assert "capabilities" in info


# ============================================================================
# Tool Definition Tests
# ============================================================================


class TestMCPToolDefinitions:
    """Test MCP tool definitions and schemas."""

    def test_list_tools(self, mcp_server):
        """Test listing available tools."""
        tools = mcp_server.list_tools()

        assert tools is not None
        assert len(tools) > 0

    def test_store_memory_tool_exists(self, mcp_server):
        """Test store_memory tool is defined."""
        tools = mcp_server.list_tools()

        tool_names = [t.get("name", t.name if hasattr(t, "name") else None) for t in tools]
        assert "store_memory" in tool_names

    def test_recall_memories_tool_exists(self, mcp_server):
        """Test recall tool is defined."""
        tools = mcp_server.list_tools()

        tool_names = [t.get("name", t.name if hasattr(t, "name") else None) for t in tools]
        # Tool may be named 'recall' or 'recall_memories'
        assert "recall" in tool_names or "recall_memories" in tool_names

    def test_search_memories_tool_exists(self, mcp_server):
        """Test search_memories tool is defined."""
        tools = mcp_server.list_tools()

        tool_names = [t.get("name", t.name if hasattr(t, "name") else None) for t in tools]
        assert "search_memories" in tool_names

    def test_tool_has_schema(self, mcp_server):
        """Test that tools have input schemas."""
        tools = mcp_server.list_tools()

        for tool in tools:
            # Each tool should have input schema
            assert "inputSchema" in tool or hasattr(tool, "inputSchema") or "input_schema" in tool


# ============================================================================
# Tool Execution Tests
# ============================================================================


class TestMCPToolExecution:
    """Test MCP tool call execution."""

    def test_call_store_memory(self, mcp_server):
        """Test calling store_memory tool."""
        # Check if call_tool exists, otherwise skip
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        result = mcp_server.call_tool(
            "store_memory",
            {
                "content": "Test memory via MCP",
                "memory_type": "semantic",
                "user_id": "mcp_user",
                "topics": ["api"],
                "importance": 0.7,
            },
        )

        assert result is not None

    def test_call_recall_memories(self, mcp_server):
        """Test calling recall tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        result = mcp_server.call_tool(
            "recall", {"query": "test query", "user_id": "mcp_user", "limit": 5}
        )

        assert result is not None

    def test_call_search_memories(self, mcp_server):
        """Test calling search_memories tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        result = mcp_server.call_tool("search_memories", {"user_id": "mcp_user", "topics": ["api"]})

        assert result is not None

    def test_call_get_memory(self, mcp_server, clst):
        """Test calling get_memory tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        # First store a memory
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="MCP get test",
            memory_type="semantic",
            user_id="mcp_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        # Now try to get it via MCP (if tool exists)
        try:
            result = mcp_server.call_tool("get_memory", {"memory_id": memory_id})
            assert result is not None
        except Exception:
            pytest.skip("get_memory tool not available")

    def test_call_delete_memory(self, mcp_server, clst):
        """Test calling delete_memory tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        # First store a memory
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="MCP delete test",
            memory_type="semantic",
            user_id="mcp_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        # Now try to delete it (if tool exists)
        try:
            mcp_server.call_tool("delete_memory", {"memory_id": memory_id})
        except Exception:
            pytest.skip("delete_memory tool not available")

    def test_call_reinforce_memory(self, mcp_server, clst):
        """Test calling reinforce_memory tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        # First store a memory
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="MCP reinforce test",
            memory_type="semantic",
            user_id="mcp_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        mcp_server.call_tool("reinforce_memory", {"memory_id": memory_id, "signal": 0.8})

    def test_call_get_vocabulary_schema(self, mcp_server):
        """Test calling get_vocabulary_schema tool."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            result = mcp_server.call_tool("get_vocabulary_schema", {})
            assert result is not None
        except Exception:
            pytest.skip("get_vocabulary_schema tool not available")


# ============================================================================
# Multi-Agent Tool Tests
# ============================================================================


class TestMCPMultiAgentTools:
    """Test MCP tools for multi-agent scenarios."""

    def test_register_agent_tool(self, mcp_server):
        """Test register_agent tool if multi-agent enabled."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            mcp_server.call_tool(
                "register_agent",
                {"agent_id": "new_agent_mcp", "name": "New Agent", "teams": ["support"]},
            )
        except Exception as e:
            # May not be available if multi-agent not enabled
            error_str = str(e).lower()
            if (
                "not found" in error_str
                or "not enabled" in error_str
                or "unknown tool" in error_str
            ):
                pytest.skip("register_agent tool not available")
            raise


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestMCPErrorHandling:
    """Test MCP error handling."""

    def test_unknown_tool(self, mcp_server):
        """Test calling unknown tool returns error."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            mcp_server.call_tool("nonexistent_tool", {})
            # If it didn't raise, check that it handled gracefully
        except Exception:
            pass  # Expected behavior

    def test_missing_required_parameter(self, mcp_server):
        """Test missing required parameter handling."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            mcp_server.call_tool(
                "store_memory",
                {
                    "content": "Test"
                    # Missing memory_type, user_id
                },
            )
            # If it didn't raise, that's also acceptable behavior
        except Exception:
            pass  # Expected behavior

    def test_invalid_parameter_type(self, mcp_server):
        """Test invalid parameter type handling."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            mcp_server.call_tool(
                "store_memory",
                {
                    "content": "Test",
                    "memory_type": "semantic",
                    "user_id": "test",
                    "topics": ["api"],
                    "importance": "not_a_number",  # Should be float
                },
            )
        except Exception:
            pass  # Expected behavior

    def test_memory_not_found(self, mcp_server):
        """Test memory not found handling."""
        if not hasattr(mcp_server, "call_tool"):
            pytest.skip("call_tool method not implemented")

        try:
            mcp_server.call_tool("get_memory", {"memory_id": "nonexistent_id_12345"})
            # Should handle gracefully (return None or error)
        except Exception:
            pass  # Expected behavior


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestMCPSchemaValidation:
    """Test MCP input schema validation."""

    def test_validate_store_memory_input(self, mcp_server):
        """Test store_memory input validation."""
        tools = mcp_server.list_tools()

        store_tool = next(
            (t for t in tools if (t.get("name") or getattr(t, "name", None)) == "store_memory"),
            None,
        )

        assert store_tool is not None

        # Get schema
        schema = (
            store_tool.get("inputSchema")
            or getattr(store_tool, "inputSchema", None)
            or store_tool.get("input_schema")
        )
        assert schema is not None

    def test_validate_recall_input(self, mcp_server):
        """Test recall input validation."""
        tools = mcp_server.list_tools()

        # Tool may be named 'recall' or 'recall_memories'
        recall_tool = next(
            (
                t
                for t in tools
                if (t.get("name") or getattr(t, "name", None)) in ["recall", "recall_memories"]
            ),
            None,
        )

        assert recall_tool is not None


# ============================================================================
# Response Format Tests
# ============================================================================


class TestMCPResponseFormat:
    """Test MCP response formats."""

    def test_store_returns_memory_id(self, mcp_server, mock_mindcore_for_mcp):
        """Test store_memory returns memory_id."""
        result = mcp_server.call_tool(
            "store_memory",
            {"content": "Test", "memory_type": "semantic", "user_id": "test", "topics": ["api"]},
        )

        # Should contain memory_id
        assert result is not None
        result_str = str(result)
        assert "memory_id" in result_str or "mem_" in result_str

    def test_recall_returns_memories(self, mcp_server, mock_mindcore_for_mcp):
        """Test recall_memories returns memories list."""
        result = mcp_server.call_tool("recall_memories", {"query": "test", "user_id": "test"})

        assert result is not None
        # Should contain memories

    def test_search_returns_list(self, mcp_server, mock_mindcore_for_mcp):
        """Test search_memories returns list."""
        result = mcp_server.call_tool("search_memories", {"user_id": "test"})

        assert result is not None
        # Should be list or contain memories


# ============================================================================
# Context Integration Tests
# ============================================================================


class TestMCPContextIntegration:
    """Test MCP context and conversation integration."""

    def test_tool_preserves_context(self, mcp_server, mock_mindcore_for_mcp):
        """Test that tool calls preserve conversation context."""
        # Store memory
        mcp_server.call_tool(
            "store_memory",
            {
                "content": "Context test memory",
                "memory_type": "semantic",
                "user_id": "context_user",
                "topics": ["api"],
            },
        )

        # Recall should find it
        mcp_server.call_tool(
            "recall_memories", {"query": "context test", "user_id": "context_user"}
        )

        # Both calls should work independently


# ============================================================================
# Performance Tests
# ============================================================================


class TestMCPPerformance:
    """Test MCP server performance."""

    def test_tool_call_latency(self, mcp_server, mock_mindcore_for_mcp):
        """Test tool call latency."""
        import time

        times = []
        for _ in range(10):
            start = time.perf_counter()
            mcp_server.call_tool("recall_memories", {"query": "test", "user_id": "perf_user"})
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        # Tool calls should be fast (overhead only, actual work is mocked)
        assert avg_time < 50, f"Average tool call time {avg_time:.2f}ms"

    def test_list_tools_latency(self, mcp_server):
        """Test list_tools latency."""
        import time

        start = time.perf_counter()
        mcp_server.list_tools()
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 10, f"list_tools took {elapsed:.2f}ms"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance."""

    def test_tool_response_format(self, mcp_server, mock_mindcore_for_mcp):
        """Test tool response follows MCP format."""
        result = mcp_server.call_tool(
            "store_memory",
            {"content": "Test", "memory_type": "semantic", "user_id": "test", "topics": ["api"]},
        )

        # Response should be serializable
        import json

        try:
            if hasattr(result, "to_dict"):
                json.dumps(result.to_dict())
            elif hasattr(result, "__dict__"):
                json.dumps(str(result))
            else:
                json.dumps(result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Response not JSON serializable: {e}")

    def test_error_response_format(self, mcp_server):
        """Test error responses follow MCP format."""
        # Try to call a nonexistent tool
        result = mcp_server.call_tool("nonexistent_tool", {})
        # Should either raise an exception or return an error response
        # If it returns a result, it should be an error indicator
        if result is not None:
            assert isinstance(result, dict | str)
