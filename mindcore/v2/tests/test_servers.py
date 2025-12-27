"""Tests for REST and MCP Servers.

Tests cover:
- MCPServer: tools, resources, JSON-RPC handling
- REST API create_app (requires FastAPI)
- Server initialization and configuration
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mindcore.v2.flr import FLR, Memory, RecallResult
from mindcore.v2.storage.sqlite import SQLiteStorage


def _has_fastapi():
    """Check if FastAPI is available."""
    try:
        import fastapi

        return True
    except ImportError:
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage():
    """Create temporary SQLite storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    storage.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def flr(storage):
    """Create FLR instance."""
    return FLR(storage=storage)


@pytest.fixture
def clst(storage):
    """Create CLST instance."""
    from mindcore.v2.clst import CLST

    return CLST(storage=storage)


@pytest.fixture
def mcp_server(flr, clst):
    """Create MCP server instance."""
    from mindcore.v2.server.mcp import MCPServer

    return MCPServer(flr=flr, clst=clst)


# =============================================================================
# MCPTool and MCPResource Tests
# =============================================================================


class TestMCPDataclasses:
    """Tests for MCP dataclasses."""

    def test_mcp_tool_creation(self):
        """Test creating MCPTool."""
        from mindcore.v2.server.mcp import MCPTool

        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=lambda: None,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_mcp_resource_creation(self):
        """Test creating MCPResource."""
        from mindcore.v2.server.mcp import MCPResource

        resource = MCPResource(
            uri="mindcore://test",
            name="Test Resource",
            description="A test resource",
            mime_type="application/json",
        )

        assert resource.uri == "mindcore://test"
        assert resource.mime_type == "application/json"

    def test_mcp_server_info_defaults(self):
        """Test MCPServerInfo default values."""
        from mindcore.v2.server.mcp import MCPServerInfo

        info = MCPServerInfo()

        assert info.name == "mindcore"
        assert info.version == "2.0.0"


# =============================================================================
# MCPServer Initialization Tests
# =============================================================================


class TestMCPServerInit:
    """Tests for MCPServer initialization."""

    def test_init_with_flr_only(self, flr):
        """Test init with only FLR creates CLST."""
        from mindcore.v2.server.mcp import MCPServer

        server = MCPServer(flr=flr)

        assert server.clst is not None

    def test_init_with_clst(self, flr, clst):
        """Test init with both FLR and CLST."""
        from mindcore.v2.server.mcp import MCPServer

        server = MCPServer(flr=flr, clst=clst)

        assert server.clst is clst

    def test_init_creates_tools(self, mcp_server):
        """Test init creates tool definitions."""
        assert len(mcp_server._tools) > 0
        assert "store_memory" in mcp_server._tools
        assert "search_memories" in mcp_server._tools
        assert "recall" in mcp_server._tools

    def test_init_creates_resources(self, mcp_server):
        """Test init creates resource definitions."""
        assert len(mcp_server._resources) > 0
        assert "mindcore://vocabulary" in mcp_server._resources
        assert "mindcore://stats" in mcp_server._resources


# =============================================================================
# MCPServer Server Info Tests
# =============================================================================


class TestMCPServerInfo:
    """Tests for get_server_info method."""

    def test_get_server_info(self, mcp_server):
        """Test getting server info."""
        info = mcp_server.get_server_info()

        assert info["name"] == "mindcore"
        assert info["version"] == "2.0.0"
        assert "capabilities" in info
        assert "tools" in info["capabilities"]
        assert "resources" in info["capabilities"]


# =============================================================================
# MCPServer Tools Tests
# =============================================================================


class TestMCPServerTools:
    """Tests for get_tools and list_tools methods."""

    def test_get_tools(self, mcp_server):
        """Test getting tool definitions."""
        tools = mcp_server.get_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 4

        tool_names = [t["name"] for t in tools]
        assert "store_memory" in tool_names
        assert "recall" in tool_names

    def test_list_tools_alias(self, mcp_server):
        """Test list_tools is alias for get_tools."""
        assert mcp_server.list_tools() == mcp_server.get_tools()

    def test_tool_has_schema(self, mcp_server):
        """Test tools have input schemas."""
        tools = mcp_server.get_tools()

        for tool in tools:
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]


# =============================================================================
# MCPServer Resources Tests
# =============================================================================


class TestMCPServerResources:
    """Tests for get_resources method."""

    def test_get_resources(self, mcp_server):
        """Test getting resource definitions."""
        resources = mcp_server.get_resources()

        assert isinstance(resources, list)
        assert len(resources) >= 2

    def test_resource_has_uri(self, mcp_server):
        """Test resources have URIs."""
        resources = mcp_server.get_resources()

        for resource in resources:
            assert "uri" in resource
            assert resource["uri"].startswith("mindcore://")


# =============================================================================
# MCPServer Call Tool Tests
# =============================================================================


class TestMCPServerCallTool:
    """Tests for call_tool method."""

    def test_call_unknown_tool(self, mcp_server):
        """Test calling unknown tool returns error."""
        result = mcp_server.call_tool("unknown_tool", {})

        assert result["isError"] is True
        assert "Unknown tool" in result["error"]

    def test_call_store_memory(self, mcp_server):
        """Test calling store_memory tool."""
        result = mcp_server.call_tool(
            "store_memory",
            {
                "content": "User prefers dark mode",
                "memory_type": "preference",
                "user_id": "user_123",
            },
        )

        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert content["success"] is True
        assert "memory_id" in content

    def test_call_search_memories(self, mcp_server, storage):
        """Test calling search_memories tool."""
        # Store a memory first
        memory = Memory(
            memory_id="search_test",
            content="Test content",
            memory_type="fact",
            user_id="user_123",
        )
        storage.store(memory)

        result = mcp_server.call_tool(
            "search_memories",
            {"user_id": "user_123"},
        )

        assert result["isError"] is False

    def test_call_recall(self, mcp_server):
        """Test calling recall tool."""
        result = mcp_server.call_tool(
            "recall",
            {
                "query": "user preferences",
                "user_id": "user_123",
            },
        )

        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert "memories" in content

    def test_call_reinforce_memory(self, mcp_server, storage):
        """Test calling reinforce_memory tool."""
        # Store a memory first
        memory = Memory(
            memory_id="reinforce_test",
            content="Test",
            memory_type="fact",
            user_id="user_1",
        )
        storage.store(memory)

        result = mcp_server.call_tool(
            "reinforce_memory",
            {"memory_id": "reinforce_test", "signal": 0.5},
        )

        assert result["isError"] is False

    def test_call_get_user_context(self, mcp_server):
        """Test calling get_user_context tool."""
        result = mcp_server.call_tool(
            "get_user_context",
            {"user_id": "user_123"},
        )

        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert "preferences" in content
        assert "facts" in content

    def test_call_tool_with_agent_id(self, mcp_server):
        """Test calling tool with agent_id."""
        # Create server with access controller
        from mindcore.v2.access import AccessController
        from mindcore.v2.server.mcp import MCPServer

        controller = AccessController()
        controller.register_agent("agent_1", "Test Agent")

        server = MCPServer(
            flr=mcp_server.flr,
            clst=mcp_server.clst,
            access_controller=controller,
        )

        result = server.call_tool(
            "search_memories",
            {"user_id": "user_123"},
            agent_id="agent_1",
        )

        assert result["isError"] is False


# =============================================================================
# MCPServer Read Resource Tests
# =============================================================================


class TestMCPServerReadResource:
    """Tests for read_resource method."""

    def test_read_unknown_resource(self, mcp_server):
        """Test reading unknown resource returns error."""
        result = mcp_server.read_resource("mindcore://unknown")

        assert "error" in result

    def test_read_stats_resource(self, mcp_server):
        """Test reading stats resource."""
        result = mcp_server.read_resource("mindcore://stats")

        assert "contents" in result
        content = json.loads(result["contents"][0]["text"])
        assert "flr" in content
        assert "clst" in content

    def test_read_vocabulary_resource_no_vocab(self, mcp_server):
        """Test reading vocabulary resource without vocab."""
        result = mcp_server.read_resource("mindcore://vocabulary")

        assert "contents" in result
        content = json.loads(result["contents"][0]["text"])
        # Empty dict when no vocabulary
        assert content == {}


# =============================================================================
# MCPServer JSON-RPC Tests
# =============================================================================


class TestMCPServerJSONRPC:
    """Tests for JSON-RPC handling."""

    def test_to_json_rpc_response_success(self, mcp_server):
        """Test creating success JSON-RPC response."""
        response = mcp_server.to_json_rpc_response(
            request_id=1,
            result={"data": "test"},
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["data"] == "test"
        assert "error" not in response

    def test_to_json_rpc_response_error(self, mcp_server):
        """Test creating error JSON-RPC response."""
        response = mcp_server.to_json_rpc_response(
            request_id=2,
            result=None,
            error="Something went wrong",
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "error" in response
        assert response["error"]["message"] == "Something went wrong"

    def test_handle_initialize(self, mcp_server):
        """Test handling initialize method."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 1
        assert response["result"]["name"] == "mindcore"

    def test_handle_tools_list(self, mcp_server):
        """Test handling tools/list method."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 2
        assert "tools" in response["result"]

    def test_handle_tools_call(self, mcp_server):
        """Test handling tools/call method."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "query": "test",
                    "user_id": "user_1",
                },
            },
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 3
        assert "result" in response

    def test_handle_resources_list(self, mcp_server):
        """Test handling resources/list method."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": {},
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 4
        assert "resources" in response["result"]

    def test_handle_resources_read(self, mcp_server):
        """Test handling resources/read method."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "mindcore://stats"},
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 5
        assert "contents" in response["result"]

    def test_handle_unknown_method(self, mcp_server):
        """Test handling unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "unknown/method",
            "params": {},
        }

        response = mcp_server.handle_json_rpc(request)

        assert response["id"] == 6
        assert "error" in response


# =============================================================================
# REST API Tests (if FastAPI available)
# =============================================================================


class TestRESTAPI:
    """Tests for REST API create_app."""

    def test_rest_module_imports(self):
        """Test REST module can be imported."""
        from mindcore.v2.server import rest

        assert hasattr(rest, "create_app")

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="FastAPI not installed",
    )
    def test_create_app_success(self, flr, clst):
        """Test create_app succeeds with FastAPI."""
        from mindcore.v2.server.rest import create_app

        app = create_app(flr, clst, rate_limit=None)

        assert app is not None
        assert app.title == "Mindcore API"

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="FastAPI not installed",
    )
    def test_create_app_creates_clst(self, flr):
        """Test create_app creates CLST if not provided."""
        from mindcore.v2.server.rest import create_app

        app = create_app(flr, rate_limit=None)

        assert app is not None


# =============================================================================
# Tool Handler Tests
# =============================================================================


class TestToolHandlers:
    """Tests for individual tool handlers."""

    def test_handle_store_memory(self, mcp_server):
        """Test _handle_store_memory handler."""
        result = mcp_server._handle_store_memory(
            content="Test memory",
            memory_type="fact",
            user_id="user_1",
        )

        assert result["success"] is True
        assert "memory_id" in result

    def test_handle_store_memory_with_topics(self, mcp_server):
        """Test store with topics and categories."""
        result = mcp_server._handle_store_memory(
            content="Order #12345",
            memory_type="episodic",
            user_id="user_1",
            topics=["orders"],
            categories=["support"],
            importance=0.8,
        )

        assert result["success"] is True

    def test_handle_search_memories(self, mcp_server, storage):
        """Test _handle_search_memories handler."""
        memory = Memory(
            memory_id="handler_test",
            content="Handler test",
            memory_type="fact",
            user_id="user_1",
        )
        storage.store(memory)

        result = mcp_server._handle_search_memories(user_id="user_1")

        assert "memories" in result
        assert "count" in result

    def test_handle_recall(self, mcp_server):
        """Test _handle_recall handler."""
        result = mcp_server._handle_recall(
            query="test query",
            user_id="user_1",
        )

        assert "memories" in result
        assert "scores" in result
        assert "latency_ms" in result

    def test_handle_reinforce(self, mcp_server, storage):
        """Test _handle_reinforce handler."""
        memory = Memory(
            memory_id="reinforce_handler",
            content="Reinforce me",
            memory_type="fact",
            user_id="user_1",
        )
        storage.store(memory)

        result = mcp_server._handle_reinforce(
            memory_id="reinforce_handler",
            signal=0.7,
        )

        assert result["success"] is True
        assert result["signal"] == 0.7

    def test_handle_get_user_context(self, mcp_server):
        """Test _handle_get_user_context handler."""
        result = mcp_server._handle_get_user_context(user_id="user_1")

        assert "preferences" in result
        assert "facts" in result
        assert "total_memories" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
