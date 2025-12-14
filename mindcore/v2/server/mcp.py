"""MCP (Model Context Protocol) Server for Mindcore v2.

Provides native LLM integration via the Model Context Protocol standard.
Supports Claude, GPT, Gemini, and other MCP-compatible models.

Based on: https://modelcontextprotocol.io/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from datetime import datetime, timezone

if TYPE_CHECKING:
    from ..flr import FLR
    from ..clst import CLST
    from ..vocabulary import VocabularySchema
    from ..access import AccessController


@dataclass
class MCPTool:
    """MCP Tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class MCPResource:
    """MCP Resource definition."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


@dataclass
class MCPServerInfo:
    """MCP Server information."""

    name: str = "mindcore"
    version: str = "2.0.0"
    protocol_version: str = "2024-11-05"


class MCPServer:
    """Model Context Protocol server for Mindcore.

    Exposes memory operations as MCP tools and resources.

    Example:
        server = MCPServer(
            flr=flr,
            clst=clst,
            vocabulary=vocab,
            access_controller=acl,
        )

        # Start server
        server.serve(port=3000)

        # Or get tool definitions for manual integration
        tools = server.get_tools()
    """

    def __init__(
        self,
        flr: FLR,
        clst: CLST,
        vocabulary: VocabularySchema | None = None,
        access_controller: AccessController | None = None,
    ):
        """Initialize MCP server.

        Args:
            flr: FLR instance for fast recall
            clst: CLST instance for long-term storage
            vocabulary: Optional vocabulary schema
            access_controller: Optional access controller
        """
        self.flr = flr
        self.clst = clst
        self.vocabulary = vocabulary
        self.access_controller = access_controller

        self._tools = self._create_tools()
        self._resources = self._create_resources()

    def get_server_info(self) -> dict[str, Any]:
        """Get MCP server information."""
        info = MCPServerInfo()
        return {
            "name": info.name,
            "version": info.version,
            "protocolVersion": info.protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Get MCP tool definitions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def get_resources(self) -> list[dict[str, Any]]:
        """Get MCP resource definitions."""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type,
            }
            for resource in self._resources.values()
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool.

        Args:
            name: Tool name
            arguments: Tool arguments
            agent_id: Optional agent ID for access control

        Returns:
            Tool result
        """
        if name not in self._tools:
            return {
                "error": f"Unknown tool: {name}",
                "isError": True,
            }

        tool = self._tools[name]

        try:
            # Inject agent_id if access control is enabled
            if self.access_controller and agent_id:
                arguments["_agent_id"] = agent_id

            result = tool.handler(**arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, default=str),
                    }
                ],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(e),
                    }
                ],
                "isError": True,
            }

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read an MCP resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri not in self._resources:
            return {
                "error": f"Unknown resource: {uri}",
            }

        resource = self._resources[uri]

        # Handle different resource types
        if uri == "mindcore://vocabulary":
            content = self.vocabulary.to_dict() if self.vocabulary else {}
        elif uri == "mindcore://stats":
            content = {
                "flr": self.flr.get_stats(),
                "clst": self.clst.get_stats(),
            }
        else:
            content = {}

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": json.dumps(content, default=str),
                }
            ],
        }

    def _create_tools(self) -> dict[str, MCPTool]:
        """Create MCP tool definitions."""
        tools = {}

        # store_memory tool
        tools["store_memory"] = MCPTool(
            name="store_memory",
            description="Store a memory in long-term storage",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "preference", "entity"],
                        "description": "Type of memory",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant topics",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Importance score 0-1",
                    },
                    "access_level": {
                        "type": "string",
                        "enum": ["private", "team", "shared", "global"],
                        "description": "Access level for multi-agent",
                    },
                },
                "required": ["content", "memory_type", "user_id"],
            },
            handler=self._handle_store_memory,
        )

        # search_memories tool
        tools["search_memories"] = MCPTool(
            name="search_memories",
            description="Search stored memories",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by topics",
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by memory types",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max results",
                    },
                },
                "required": ["user_id"],
            },
            handler=self._handle_search_memories,
        )

        # recall tool (FLR fast path)
        tools["recall"] = MCPTool(
            name="recall",
            description="Fast recall of relevant memories for current context",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Current query or context",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "attention_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to prioritize",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max memories to recall",
                    },
                },
                "required": ["query", "user_id"],
            },
            handler=self._handle_recall,
        )

        # reinforce tool
        tools["reinforce_memory"] = MCPTool(
            name="reinforce_memory",
            description="Reinforce a memory as useful (positive) or not useful (negative)",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory identifier",
                    },
                    "signal": {
                        "type": "number",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Reinforcement signal (-1 to 1)",
                    },
                },
                "required": ["memory_id", "signal"],
            },
            handler=self._handle_reinforce,
        )

        # get_user_context tool
        tools["get_user_context"] = MCPTool(
            name="get_user_context",
            description="Get user preferences and recent context",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                },
                "required": ["user_id"],
            },
            handler=self._handle_get_user_context,
        )

        return tools

    def _create_resources(self) -> dict[str, MCPResource]:
        """Create MCP resource definitions."""
        resources = {}

        resources["mindcore://vocabulary"] = MCPResource(
            uri="mindcore://vocabulary",
            name="Vocabulary Schema",
            description="Current vocabulary schema with topics, categories, and memory types",
        )

        resources["mindcore://stats"] = MCPResource(
            uri="mindcore://stats",
            name="Memory Statistics",
            description="Current memory system statistics",
        )

        return resources

    def _handle_store_memory(
        self,
        content: str,
        memory_type: str,
        user_id: str,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        importance: float = 0.5,
        access_level: str = "private",
        entities: list[str] | None = None,
        _agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle store_memory tool call."""
        from ..flr import Memory

        memory = Memory(
            memory_id="",  # Will be generated
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=_agent_id,
            topics=topics or [],
            categories=categories or [],
            importance=importance,
            access_level=access_level,
            entities=entities or [],
            vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
        )

        memory_id = self.clst.store(memory)

        return {
            "success": True,
            "memory_id": memory_id,
        }

    def _handle_search_memories(
        self,
        user_id: str,
        query: str | None = None,
        topics: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        _agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle search_memories tool call."""
        memories = self.clst.search(
            query=query,
            user_id=user_id,
            agent_id=_agent_id,
            topics=topics,
            memory_types=memory_types,
            limit=limit,
        )

        # Filter by access control if enabled
        if self.access_controller and _agent_id:
            from ..access import Permission
            memories = self.access_controller.filter_accessible_memories(
                _agent_id, memories, Permission.READ
            )

        return {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    def _handle_recall(
        self,
        query: str,
        user_id: str,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 5,
        _agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle recall tool call (FLR fast path)."""
        result = self.flr.query(
            query=query,
            user_id=user_id,
            agent_id=_agent_id,
            attention_hints=attention_hints,
            memory_types=memory_types,
            limit=limit,
        )

        return {
            "memories": [m.to_dict() for m in result.memories],
            "scores": result.scores,
            "attention_focus": result.attention_focus,
            "suggested_memory_types": result.suggested_memory_types,
            "latency_ms": result.query_latency_ms,
        }

    def _handle_reinforce(
        self,
        memory_id: str,
        signal: float,
        _agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle reinforce_memory tool call."""
        self.flr.reinforce(memory_id, signal)
        return {
            "success": True,
            "memory_id": memory_id,
            "signal": signal,
        }

    def _handle_get_user_context(
        self,
        user_id: str,
        _agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle get_user_context tool call."""
        # Get recent memories
        recent = self.clst.search(
            user_id=user_id,
            memory_types=["preference", "semantic"],
            limit=20,
        )

        # Filter by access control if enabled
        if self.access_controller and _agent_id:
            from ..access import Permission
            recent = self.access_controller.filter_accessible_memories(
                _agent_id, recent, Permission.READ
            )

        # Extract preferences
        preferences = [m for m in recent if m.memory_type == "preference"]

        # Extract facts
        facts = [m for m in recent if m.memory_type == "semantic"]

        return {
            "preferences": [m.to_dict() for m in preferences[:5]],
            "facts": [m.to_dict() for m in facts[:5]],
            "total_memories": len(recent),
        }

    def to_json_rpc_response(
        self,
        request_id: Any,
        result: Any,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Format response as JSON-RPC 2.0."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
        }

        if error:
            response["error"] = {
                "code": -32603,
                "message": error,
            }
        else:
            response["result"] = result

        return response

    def handle_json_rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC 2.0 request.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response
        """
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = self.get_server_info()
            elif method == "tools/list":
                result = {"tools": self.get_tools()}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = self.call_tool(tool_name, arguments)
            elif method == "resources/list":
                result = {"resources": self.get_resources()}
            elif method == "resources/read":
                uri = params.get("uri")
                result = self.read_resource(uri)
            else:
                return self.to_json_rpc_response(
                    request_id, None, f"Unknown method: {method}"
                )

            return self.to_json_rpc_response(request_id, result)

        except Exception as e:
            return self.to_json_rpc_response(request_id, None, str(e))
