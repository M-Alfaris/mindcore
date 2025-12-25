"""SVL Data Source Mapping - Connect vocabulary to external data sources.

This module enables automatic data fetching when topics, categories, or domains
are accessed. When a memory query uses a mapped topic, the system can automatically
fetch relevant data from configured sources.

Supported source types:
- Database tables (SQL queries)
- REST APIs (HTTP endpoints)
- MCP servers (tool calls)
- Custom functions

Example:
    svl = SharedVocabularyLayer()

    # Map "orders" topic to a database table
    svl.map_source("orders", TableSource(
        connection="postgresql://...",
        table="orders",
        query_template="SELECT * FROM orders WHERE user_id = :user_id",
    ))

    # Map "weather" topic to an API
    svl.map_source("weather", APISource(
        url="https://api.weather.com/current",
        method="GET",
        headers={"Authorization": "Bearer {api_key}"},
    ))

    # Map "search" topic to MCP server
    svl.map_source("search", MCPSource(
        server="brave-search",
        tool="search",
        argument_mapping={"query": "search_query"},
    ))

    # When querying with "orders" topic, data is auto-fetched
    result = svl.fetch_for_topics(["orders"], context={"user_id": "123"})
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# Valid SQL identifier pattern (alphanumeric and underscore only)
_VALID_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SourceType(str, Enum):
    """Types of data sources."""

    TABLE = "table"  # Database table
    API = "api"  # REST API endpoint
    MCP = "mcp"  # MCP server tool
    FUNCTION = "function"  # Custom Python function


class TriggerCondition(str, Enum):
    """When to trigger data fetching."""

    ALWAYS = "always"  # Always fetch when topic is used
    ON_QUERY = "on_query"  # Only on memory queries
    ON_STORE = "on_store"  # Only when storing memories
    ON_DEMAND = "on_demand"  # Only when explicitly requested
    CONDITIONAL = "conditional"  # Based on context conditions


@dataclass
class FetchResult:
    """Result from fetching data from a source."""

    source_name: str
    source_type: SourceType
    data: Any
    success: bool
    error: str | None = None
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "data": self.data,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "metadata": self.metadata,
        }


class DataSource(ABC):
    """Abstract base class for data sources."""

    source_type: SourceType
    name: str
    description: str = ""
    enabled: bool = True
    cache_ttl_seconds: int = 0  # 0 = no caching
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    @abstractmethod
    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Fetch data from the source.

        Args:
            context: Context dict with user_id, query, etc.

        Returns:
            FetchResult with data or error
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize source configuration."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSource:
        """Create from serialized config."""


@dataclass
class TableSource(DataSource):
    """Database table data source.

    Executes SQL queries against a database table when the mapped
    topic/category is accessed.
    """

    source_type: SourceType = field(default=SourceType.TABLE, init=False)
    name: str = ""
    description: str = ""

    # Connection
    connection_string: str = ""  # Database connection string
    table: str = ""  # Table name

    # Query configuration
    query_template: str = ""  # SQL with :param placeholders
    param_mapping: dict[str, str] = field(default_factory=dict)  # context_key -> sql_param

    # Options
    limit: int = 100
    timeout_seconds: int = 30
    enabled: bool = True
    cache_ttl_seconds: int = 60
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    def _validate_identifier(self, identifier: str) -> str:
        """Validate and return a safe SQL identifier.

        Args:
            identifier: Table or column name to validate

        Returns:
            The validated identifier

        Raises:
            ValueError: If identifier contains invalid characters
        """
        if not identifier:
            raise ValueError("SQL identifier cannot be empty")
        if not _VALID_SQL_IDENTIFIER.match(identifier):
            raise ValueError(
                f"Invalid SQL identifier: '{identifier}'. "
                "Only alphanumeric characters and underscores are allowed, "
                "and it must start with a letter or underscore."
            )
        return identifier

    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Execute SQL query and return results."""
        import time

        start = time.time()

        try:
            # Build query params from context
            params = {}
            for context_key, sql_param in self.param_mapping.items():
                if context_key in context:
                    params[sql_param] = context[context_key]

            # Use default query if template not provided
            query = self.query_template
            if not query and self.table:
                # Validate table name and limit to prevent SQL injection
                safe_table = self._validate_identifier(self.table)
                safe_limit = int(self.limit)  # Ensure limit is an integer
                if safe_limit < 1 or safe_limit > 10000:
                    safe_limit = 100  # Default to safe limit
                query = f"SELECT * FROM {safe_table} LIMIT {safe_limit}"

            # Execute based on connection type
            if self.connection_string.startswith("postgresql"):
                data = self._fetch_postgres(query, params)
            elif self.connection_string.startswith("sqlite"):
                data = self._fetch_sqlite(query, params)
            else:
                raise ValueError(f"Unsupported database: {self.connection_string}")

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=data,
                success=True,
                latency_ms=latency,
                metadata={"query": query, "params": params},
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    def _fetch_postgres(self, query: str, params: dict) -> list[dict]:
        """Fetch from PostgreSQL."""
        try:
            import psycopg
        except ImportError:
            raise ImportError("psycopg required: pip install 'psycopg[binary]'")

        with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]

    def _fetch_sqlite(self, query: str, params: dict) -> list[dict]:
        """Fetch from SQLite."""
        import sqlite3
        from pathlib import Path

        # Extract and validate the database path to prevent path traversal
        db_path = self.connection_string.replace("sqlite:///", "")

        # Resolve the path and check for path traversal attempts
        resolved_path = Path(db_path).resolve()

        # Ensure the path doesn't escape to sensitive system directories
        sensitive_prefixes = ("/etc", "/proc", "/sys", "/dev", "/root", "/boot")
        if any(str(resolved_path).startswith(prefix) for prefix in sensitive_prefixes):
            raise ValueError(f"Access to system path is not allowed: {resolved_path}")

        # Ensure it's a .db or .sqlite file (or allow in-memory :memory:)
        if db_path != ":memory:" and not str(resolved_path).endswith((".db", ".sqlite", ".sqlite3")):
            raise ValueError(f"Invalid SQLite database extension: {resolved_path}")

        conn = sqlite3.connect(str(resolved_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "name": self.name,
            "description": self.description,
            "connection_string": self.connection_string,
            "table": self.table,
            "query_template": self.query_template,
            "param_mapping": self.param_mapping,
            "limit": self.limit,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableSource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            connection_string=data.get("connection_string", ""),
            table=data.get("table", ""),
            query_template=data.get("query_template", ""),
            param_mapping=data.get("param_mapping", {}),
            limit=data.get("limit", 100),
            timeout_seconds=data.get("timeout_seconds", 30),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 60),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
        )


@dataclass
class APISource(DataSource):
    """REST API data source.

    Makes HTTP requests to external APIs when the mapped topic is accessed.
    """

    source_type: SourceType = field(default=SourceType.API, init=False)
    name: str = ""
    description: str = ""

    # Endpoint
    url: str = ""  # Base URL with {param} placeholders
    method: str = "GET"  # HTTP method

    # Request configuration
    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)  # Static params
    body_template: dict[str, Any] | None = None  # For POST/PUT

    # Parameter mapping (context -> request)
    url_params: dict[str, str] = field(default_factory=dict)  # context_key -> url_param
    header_params: dict[str, str] = field(default_factory=dict)  # context_key -> header
    body_params: dict[str, str] = field(default_factory=dict)  # context_key -> body_field

    # Options
    timeout_seconds: int = 30
    enabled: bool = True
    cache_ttl_seconds: int = 300
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    # Response handling
    response_path: str = ""  # JSON path to extract (e.g., "data.results")

    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Make HTTP request and return response."""
        import time

        start = time.time()

        try:
            import urllib.parse
            import urllib.request

            # Build URL with params
            url = self.url
            for context_key, url_param in self.url_params.items():
                if context_key in context:
                    url = url.replace(f"{{{url_param}}}", str(context[context_key]))

            # Add query params
            params = dict(self.query_params)
            for context_key, param_name in self.url_params.items():
                if context_key in context and f"{{{param_name}}}" not in self.url:
                    params[param_name] = context[context_key]

            if params:
                url = f"{url}?{urllib.parse.urlencode(params)}"

            # Build headers
            headers = dict(self.headers)
            for context_key, header_name in self.header_params.items():
                if context_key in context:
                    headers[header_name] = str(context[context_key])

            # Build body
            body = None
            if self.method in ("POST", "PUT", "PATCH") and self.body_template:
                body_data = dict(self.body_template)
                for context_key, body_field in self.body_params.items():
                    if context_key in context:
                        body_data[body_field] = context[context_key]
                body = json.dumps(body_data).encode("utf-8")
                headers["Content-Type"] = "application/json"

            # Make request
            req = urllib.request.Request(url, data=body, headers=headers, method=self.method)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            # Extract nested path if specified
            if self.response_path:
                for key in self.response_path.split("."):
                    response_data = response_data.get(key, response_data)

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=response_data,
                success=True,
                latency_ms=latency,
                metadata={"url": url, "method": self.method},
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "query_params": self.query_params,
            "body_template": self.body_template,
            "url_params": self.url_params,
            "header_params": self.header_params,
            "body_params": self.body_params,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
            "response_path": self.response_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APISource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            url=data.get("url", ""),
            method=data.get("method", "GET"),
            headers=data.get("headers", {}),
            query_params=data.get("query_params", {}),
            body_template=data.get("body_template"),
            url_params=data.get("url_params", {}),
            header_params=data.get("header_params", {}),
            body_params=data.get("body_params", {}),
            timeout_seconds=data.get("timeout_seconds", 30),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 300),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
            response_path=data.get("response_path", ""),
        )


@dataclass
class MCPSource(DataSource):
    """MCP (Model Context Protocol) server data source.

    Invokes MCP server tools when the mapped topic is accessed.
    """

    source_type: SourceType = field(default=SourceType.MCP, init=False)
    name: str = ""
    description: str = ""

    # MCP configuration
    server_name: str = ""  # MCP server name
    tool_name: str = ""  # Tool to invoke

    # Argument mapping
    argument_mapping: dict[str, str] = field(default_factory=dict)  # context_key -> tool_arg
    static_arguments: dict[str, Any] = field(default_factory=dict)  # Always included

    # Options
    timeout_seconds: int = 60
    enabled: bool = True
    cache_ttl_seconds: int = 0  # Usually don't cache MCP
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    # MCP client (set externally)
    _mcp_client: Any = field(default=None, repr=False)

    def set_mcp_client(self, client: Any) -> None:
        """Set the MCP client for making calls."""
        self._mcp_client = client

    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Invoke MCP tool and return result."""
        import time

        start = time.time()

        try:
            if not self._mcp_client:
                raise ValueError("MCP client not configured")

            # Build arguments
            arguments = dict(self.static_arguments)
            for context_key, tool_arg in self.argument_mapping.items():
                if context_key in context:
                    arguments[tool_arg] = context[context_key]

            # Call MCP tool
            result = self._mcp_client.call_tool(
                server=self.server_name,
                tool=self.tool_name,
                arguments=arguments,
            )

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=result,
                success=True,
                latency_ms=latency,
                metadata={
                    "server": self.server_name,
                    "tool": self.tool_name,
                    "arguments": arguments,
                },
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "argument_mapping": self.argument_mapping,
            "static_arguments": self.static_arguments,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPSource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            server_name=data.get("server_name", ""),
            tool_name=data.get("tool_name", ""),
            argument_mapping=data.get("argument_mapping", {}),
            static_arguments=data.get("static_arguments", {}),
            timeout_seconds=data.get("timeout_seconds", 60),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 0),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
        )


@dataclass
class FunctionSource(DataSource):
    """Custom Python function data source.

    Calls a Python function when the mapped topic is accessed.
    """

    source_type: SourceType = field(default=SourceType.FUNCTION, init=False)
    name: str = ""
    description: str = ""

    # Function
    function: Callable[[dict[str, Any]], Any] | None = None
    function_name: str = ""  # For serialization reference

    # Options
    enabled: bool = True
    cache_ttl_seconds: int = 0
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Call function and return result."""
        import time

        start = time.time()

        try:
            if not self.function:
                raise ValueError("Function not set")

            result = self.function(context)

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=result,
                success=True,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "name": self.name,
            "description": self.description,
            "function_name": self.function_name,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionSource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            function_name=data.get("function_name", ""),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 0),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
        )


@dataclass
class GenericSource(DataSource):
    """Generic data source created from a dict config.

    Used when source configuration is provided as a plain dict
    rather than a specific DataSource subclass.
    """

    source_type: SourceType = field(default=SourceType.FUNCTION, init=False)
    name: str = ""
    description: str = ""

    # Store raw config
    config: dict[str, Any] = field(default_factory=dict)

    # Options
    enabled: bool = True
    cache_ttl_seconds: int = 0
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    def fetch(self, context: dict[str, Any]) -> FetchResult:
        """Return config as data."""
        import time

        start = time.time()
        latency = (time.time() - start) * 1000

        return FetchResult(
            source_name=self.name,
            source_type=self.source_type,
            data=self.config,
            success=True,
            latency_ms=latency,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenericSource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            config=data.get("config", data),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 0),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
        )


@dataclass
class SourceMapping:
    """Maps a vocabulary term to one or more data sources."""

    term: str  # Topic, category, or domain being mapped
    term_type: str  # "topic", "category", "domain", "intent"
    sources: list[DataSource] = field(default_factory=list)

    # Conditions
    conditions: dict[str, Any] = field(default_factory=dict)  # Additional conditions

    # Aggregation
    aggregate: bool = False  # Combine results from multiple sources
    fail_on_error: bool = False  # Fail if any source fails

    def add_source(self, source: DataSource) -> None:
        """Add a data source to this mapping."""
        self.sources.append(source)

    def remove_source(self, source_name: str) -> bool:
        """Remove a source by name."""
        for i, source in enumerate(self.sources):
            if source.name == source_name:
                self.sources.pop(i)
                return True
        return False

    def fetch_all(
        self,
        context: dict[str, Any],
        trigger: TriggerCondition = TriggerCondition.ON_QUERY,
    ) -> list[FetchResult]:
        """Fetch from all enabled sources matching the trigger.

        Args:
            context: Context for the fetch
            trigger: Current trigger condition

        Returns:
            List of FetchResults from all sources
        """
        results = []

        for source in self.sources:
            if not source.enabled:
                continue

            # Check trigger matches
            if source.trigger not in (TriggerCondition.ALWAYS, trigger):
                continue

            result = source.fetch(context)
            results.append(result)

            # Fail fast if configured
            if self.fail_on_error and not result.success:
                break

        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "term_type": self.term_type,
            "sources": [s.to_dict() for s in self.sources],
            "conditions": self.conditions,
            "aggregate": self.aggregate,
            "fail_on_error": self.fail_on_error,
        }


class SourceRegistry:
    """Registry of all data source mappings.

    Manages mappings between vocabulary terms and data sources,
    with caching and batch fetching support.
    """

    def __init__(self):
        self._mappings: dict[str, SourceMapping] = {}  # term -> mapping
        self._cache: dict[str, tuple[FetchResult, float]] = {}  # cache_key -> (result, timestamp)
        self._mcp_client: Any = None

    def set_mcp_client(self, client: Any) -> None:
        """Set MCP client for all MCP sources."""
        self._mcp_client = client
        for mapping in self._mappings.values():
            for source in mapping.sources:
                if isinstance(source, MCPSource):
                    source.set_mcp_client(client)

    def map(
        self,
        term: str,
        source: DataSource | dict[str, Any],
        term_type: str = "topic",
    ) -> None:
        """Map a vocabulary term to a data source.

        Args:
            term: Topic, category, or domain name
            source: Data source to map (DataSource object or dict config)
            term_type: Type of term ("topic", "category", "domain", "intent")
        """
        if term not in self._mappings:
            self._mappings[term] = SourceMapping(term=term, term_type=term_type)

        # Convert dict to GenericSource
        if isinstance(source, dict):
            source = GenericSource(
                name=source.get("name", term),
                description=source.get("description", ""),
                config=source,
            )

        # Set MCP client if needed
        if isinstance(source, MCPSource) and self._mcp_client:
            source.set_mcp_client(self._mcp_client)

        self._mappings[term].add_source(source)

    def unmap(self, term: str, source_name: str | None = None) -> bool:
        """Remove a mapping.

        Args:
            term: Term to unmap
            source_name: Specific source to remove, or None for all

        Returns:
            True if removed
        """
        if term not in self._mappings:
            return False

        if source_name:
            return self._mappings[term].remove_source(source_name)
        del self._mappings[term]
        return True

    def get_mapping(self, term: str) -> SourceMapping | None:
        """Get mapping for a term."""
        return self._mappings.get(term)

    def get_mapped_terms(self) -> list[str]:
        """Get all mapped terms."""
        return list(self._mappings.keys())

    def fetch_for_terms(
        self,
        terms: list[str],
        context: dict[str, Any],
        trigger: TriggerCondition = TriggerCondition.ON_QUERY,
        use_cache: bool = True,
    ) -> dict[str, list[FetchResult]]:
        """Fetch data for multiple terms.

        Args:
            terms: List of topics/categories to fetch for
            context: Context dict (user_id, query, etc.)
            trigger: Current trigger condition
            use_cache: Whether to use cached results

        Returns:
            Dict mapping term -> list of FetchResults
        """
        import time

        results: dict[str, list[FetchResult]] = {}
        now = time.time()

        for term in terms:
            mapping = self._mappings.get(term)
            if not mapping:
                continue

            term_results = []
            for source in mapping.sources:
                if not source.enabled:
                    continue
                if source.trigger not in (TriggerCondition.ALWAYS, trigger):
                    continue

                # Check cache
                cache_key = f"{term}:{source.name}:{json.dumps(context, sort_keys=True)}"
                if use_cache and cache_key in self._cache:
                    cached_result, cached_time = self._cache[cache_key]
                    if now - cached_time < source.cache_ttl_seconds:
                        cached_result.cached = True
                        term_results.append(cached_result)
                        continue

                # Fetch fresh
                result = source.fetch(context)
                term_results.append(result)

                # Cache if successful
                if result.success and source.cache_ttl_seconds > 0:
                    self._cache[cache_key] = (result, now)

            if term_results:
                results[term] = term_results

        return results

    def clear_cache(self, term: str | None = None) -> int:
        """Clear cache.

        Args:
            term: Specific term to clear, or None for all

        Returns:
            Number of entries cleared
        """
        if term:
            keys_to_delete = [k for k in self._cache if k.startswith(f"{term}:")]
            for k in keys_to_delete:
                del self._cache[k]
            return len(keys_to_delete)
        count = len(self._cache)
        self._cache.clear()
        return count

    def to_dict(self) -> dict[str, Any]:
        """Serialize all mappings."""
        return {
            "mappings": {term: mapping.to_dict() for term, mapping in self._mappings.items()},
        }

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_mappings": len(self._mappings),
            "mapped_terms": list(self._mappings.keys()),
            "cache_size": len(self._cache),
            "sources_by_type": self._count_sources_by_type(),
        }

    def _count_sources_by_type(self) -> dict[str, int]:
        """Count sources by type."""
        counts: dict[str, int] = {}
        for mapping in self._mappings.values():
            for source in mapping.sources:
                key = source.source_type.value
                counts[key] = counts.get(key, 0) + 1
        return counts


def create_source(source_type: str | SourceType, **kwargs: Any) -> DataSource:
    """Factory function to create data sources.

    Args:
        source_type: Type of source to create
        **kwargs: Source-specific configuration

    Returns:
        DataSource instance
    """
    if isinstance(source_type, str):
        source_type = SourceType(source_type)

    if source_type == SourceType.TABLE:
        return TableSource(**kwargs)
    if source_type == SourceType.API:
        return APISource(**kwargs)
    if source_type == SourceType.MCP:
        return MCPSource(**kwargs)
    if source_type == SourceType.FUNCTION:
        return FunctionSource(**kwargs)
    raise ValueError(f"Unknown source type: {source_type}")
