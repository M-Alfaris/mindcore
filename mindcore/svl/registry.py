"""SVL Source Registry - Unified source configuration with decorators and auto-discovery.

This module provides a declarative way to define and register data sources
tied to vocabulary terms (topics, categories, domains, intents).

Features:
- @source decorator for registering functions/classes as data sources
- Auto-discovery of sources from a folder
- YAML configuration loader for simple sources
- Async execution support with parallel fetching
- Automatic registration with SharedVocabularyLayer

Example - Decorator-based sources:
    from mindcore.svl import source, TriggerCondition

    @source(term="orders", term_type="topic", trigger=TriggerCondition.ON_QUERY)
    async def get_orders(context: dict) -> list[dict]:
        user_id = context.get("user_id")
        return await db.fetch_orders(user_id)

    @source(term="user_profile", term_type="topic", cache_ttl=300)
    def get_user_profile(context: dict) -> dict:
        return db.get_user(context["user_id"])

Example - Class-based sources:
    @source(term="products", term_type="topic")
    class ProductSource(TableSource):
        name = "products_db"
        query_template = "SELECT * FROM products WHERE category = :category"
        param_mapping = {"category": "category"}

Example - YAML configuration:
    sources:
      - term: "weather"
        type: api
        url: "https://api.weather.com/current"
        params:
          location: "{location}"
        cache_ttl: 300
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from .sources import (
    APISource,
    DataSource,
    FetchResult,
    MCPSource,
    SourceRegistry,
    SourceType,
    TableSource,
    TriggerCondition,
)


# Type variable for decorated functions/classes
T = TypeVar("T")

# Global registry for decorated sources (collected during module import)
_DECORATED_SOURCES: list[SourceDefinition] = []


@dataclass
class SourceDefinition:
    """Definition of a registered source from decorator or config."""

    term: str
    term_type: str  # "topic", "category", "domain", "intent"
    source: DataSource
    priority: int = 0  # Higher priority sources are fetched first
    tags: list[str] = field(default_factory=list)  # For filtering/grouping


@dataclass
class AsyncFunctionSource(DataSource):
    """Data source that wraps an async function.

    Supports both sync and async functions, executing them appropriately.
    """

    source_type: SourceType = field(default=SourceType.FUNCTION, init=False)
    name: str = ""
    description: str = ""

    # Function (sync or async)
    function: Callable[[dict[str, Any]], Any] | None = None
    is_async: bool = False

    # Options
    enabled: bool = True
    cache_ttl_seconds: int = 0
    trigger: TriggerCondition = TriggerCondition.ON_QUERY
    timeout_seconds: float = 30.0

    def _do_fetch(self, context: dict[str, Any]) -> FetchResult:
        """Implementation of abstract method - delegates to fetch logic."""
        return self._execute_fetch(context)

    def _execute_fetch(self, context: dict[str, Any]) -> FetchResult:
        """Execute function (sync or async) and return result."""
        start = time.time()

        try:
            if not self.function:
                raise ValueError("Function not set")

            if self.is_async:
                # Run async function in event loop
                result = self._run_async(context)
            else:
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

    def _run_async(self, context: dict[str, Any]) -> Any:
        """Run async function, handling event loop scenarios."""
        if not self.function:
            raise ValueError("Function not set")

        try:
            # Check if there's an existing event loop
            asyncio.get_running_loop()
            # If we're already in an async context, we need to handle differently
            # Create a new task and wait for it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._call_async(context))
                return future.result(timeout=self.timeout_seconds)
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            return asyncio.run(self._call_async(context))

    async def _call_async(self, context: dict[str, Any]) -> Any:
        """Call the async function with timeout."""
        if not self.function:
            raise ValueError("Function not set")
        return await asyncio.wait_for(self.function(context), timeout=self.timeout_seconds)

    async def fetch_async(self, context: dict[str, Any]) -> FetchResult:
        """Async version of fetch for use in async contexts."""
        start = time.time()

        try:
            if not self.function:
                raise ValueError("Function not set")

            if self.is_async:
                result = await asyncio.wait_for(
                    self.function(context), timeout=self.timeout_seconds
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.function, context)

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=result,
                success=True,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=f"Timeout after {self.timeout_seconds}s",
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
            "is_async": self.is_async,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "trigger": self.trigger.value,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsyncFunctionSource:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            is_async=data.get("is_async", False),
            enabled=data.get("enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 0),
            trigger=TriggerCondition(data.get("trigger", "on_query")),
            timeout_seconds=data.get("timeout_seconds", 30.0),
        )


def source(
    term: str,
    term_type: str = "topic",
    *,
    name: str | None = None,
    description: str = "",
    trigger: TriggerCondition = TriggerCondition.ON_QUERY,
    cache_ttl: int = 0,
    timeout: float = 30.0,
    enabled: bool = True,
    priority: int = 0,
    tags: list[str] | None = None,
) -> Callable[[T], T]:
    """Decorator to register a function or class as a data source.

    Can be used to decorate:
    - Sync functions: def get_data(context: dict) -> Any
    - Async functions: async def get_data(context: dict) -> Any
    - DataSource subclasses: class MySource(TableSource)

    Args:
        term: Vocabulary term to bind to (e.g., "orders", "user_profile")
        term_type: Type of term ("topic", "category", "domain", "intent")
        name: Source name (defaults to function/class name)
        description: Human-readable description
        trigger: When to trigger fetching
        cache_ttl: Cache TTL in seconds (0 = no caching)
        timeout: Timeout for async operations in seconds
        enabled: Whether source is enabled
        priority: Priority for ordering (higher = first)
        tags: Optional tags for filtering

    Returns:
        Decorator function

    Example:
        @source(term="orders", term_type="topic", cache_ttl=60)
        async def get_orders(context: dict) -> list[dict]:
            return await db.fetch_orders(context["user_id"])

        @source(term="products", term_type="topic")
        class ProductSource(TableSource):
            name = "products"
            query_template = "SELECT * FROM products"
    """

    def decorator(obj: T) -> T:
        source_name = name or getattr(obj, "__name__", str(obj))

        # Determine source type
        if isinstance(obj, type) and issubclass(obj, DataSource):
            # Class-based source - instantiate it
            # Check for class-level name attribute (set before instantiation)
            class_name = getattr(obj, "name", None)
            if class_name and isinstance(class_name, str):
                # Use class-defined name
                source_name = class_name

            instance = obj()
            # Set name if not set by class or decorator
            if not instance.name:
                instance.name = source_name
            instance.description = instance.description or description
            instance.trigger = trigger
            instance.cache_ttl_seconds = cache_ttl
            instance.enabled = enabled
            data_source = instance

        elif callable(obj):
            # Function-based source
            is_async = asyncio.iscoroutinefunction(obj)
            data_source = AsyncFunctionSource(
                name=source_name,
                description=description or (obj.__doc__ or "").strip(),
                function=obj,
                is_async=is_async,
                trigger=trigger,
                cache_ttl_seconds=cache_ttl,
                timeout_seconds=timeout,
                enabled=enabled,
            )
        else:
            raise TypeError(
                f"@source can only decorate functions or DataSource classes, got {type(obj)}"
            )

        # Register in global list
        definition = SourceDefinition(
            term=term,
            term_type=term_type,
            source=data_source,
            priority=priority,
            tags=tags or [],
        )
        _DECORATED_SOURCES.append(definition)

        # Attach metadata to the original object for introspection
        obj._svl_source_definition = definition  # type: ignore

        return obj

    return decorator


def get_registered_sources() -> list[SourceDefinition]:
    """Get all sources registered via @source decorator."""
    return list(_DECORATED_SOURCES)


def clear_registered_sources() -> None:
    """Clear all registered sources (useful for testing)."""
    _DECORATED_SOURCES.clear()


class SourceDiscovery:
    """Discovers and loads sources from a folder structure.

    Scans a directory for:
    - Python modules with @source decorated functions/classes
    - YAML/JSON configuration files

    Folder structure:
        sources/
        ├── __init__.py
        ├── config.yaml          # Simple sources config
        ├── topics/              # Organized by term type
        │   ├── orders.py
        │   └── products.py
        ├── categories/
        │   └── support.py
        └── custom/              # Any organization
            └── my_sources.py
    """

    def __init__(self, sources_path: str | Path):
        """Initialize discovery for a sources directory.

        Args:
            sources_path: Path to the sources directory
        """
        self.sources_path = Path(sources_path)
        self._discovered: list[SourceDefinition] = []
        self._errors: list[tuple[str, Exception]] = []

    def discover(self) -> list[SourceDefinition]:
        """Discover all sources in the directory.

        Returns:
            List of discovered SourceDefinitions
        """
        self._discovered = []
        self._errors = []

        if not self.sources_path.exists():
            return []

        # Clear global registry before discovery
        clear_registered_sources()

        # 1. Discover Python modules
        self._discover_python_modules()

        # 2. Load YAML/JSON configs
        self._load_config_files()

        # Get all registered sources (from decorators)
        self._discovered.extend(get_registered_sources())

        # Sort by priority (higher first)
        self._discovered.sort(key=lambda s: s.priority, reverse=True)

        return self._discovered

    def _discover_python_modules(self) -> None:
        """Discover and import Python modules with @source decorators."""
        for py_file in self.sources_path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                self._import_module(py_file)
            except Exception as e:
                self._errors.append((str(py_file), e))

    def _import_module(self, module_path: Path) -> None:
        """Import a Python module to trigger @source decorators."""
        # Create a unique module name
        relative = module_path.relative_to(self.sources_path)
        module_name = f"svl_sources.{relative.with_suffix('').as_posix().replace('/', '.')}"

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    def _load_config_files(self) -> None:
        """Load sources from YAML/JSON configuration files."""
        # Load YAML files
        for yaml_file in self.sources_path.rglob("*.yaml"):
            try:
                sources = load_sources_from_yaml(yaml_file)
                self._discovered.extend(sources)
            except Exception as e:
                self._errors.append((str(yaml_file), e))

        for yml_file in self.sources_path.rglob("*.yml"):
            try:
                sources = load_sources_from_yaml(yml_file)
                self._discovered.extend(sources)
            except Exception as e:
                self._errors.append((str(yml_file), e))

        # Load JSON files
        for json_file in self.sources_path.rglob("*.json"):
            if json_file.name.startswith("_"):
                continue
            try:
                sources = load_sources_from_json(json_file)
                self._discovered.extend(sources)
            except Exception as e:
                self._errors.append((str(json_file), e))

    def get_errors(self) -> list[tuple[str, Exception]]:
        """Get list of errors encountered during discovery."""
        return self._errors

    def register_to(self, registry: SourceRegistry) -> int:
        """Register all discovered sources to a SourceRegistry.

        Args:
            registry: Target registry

        Returns:
            Number of sources registered
        """
        count = 0
        for definition in self._discovered:
            registry.map(
                term=definition.term,
                source=definition.source,
                term_type=definition.term_type,
            )
            count += 1
        return count


def load_sources_from_yaml(yaml_path: str | Path) -> list[SourceDefinition]:
    """Load sources from a YAML configuration file.

    YAML format:
        sources:
          - term: "orders"
            term_type: "topic"
            type: "table"
            connection_string: "postgresql://..."
            query_template: "SELECT * FROM orders WHERE user_id = :user_id"
            param_mapping:
              user_id: "user_id"
            cache_ttl: 60
            trigger: "on_query"

          - term: "weather"
            type: "api"
            url: "https://api.weather.com/current"
            method: "GET"
            headers:
              Authorization: "Bearer {api_key}"
            cache_ttl: 300

    Args:
        yaml_path: Path to YAML file

    Returns:
        List of SourceDefinitions
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML config: pip install pyyaml")

    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    return _parse_config(config, str(yaml_path))


def load_sources_from_json(json_path: str | Path) -> list[SourceDefinition]:
    """Load sources from a JSON configuration file.

    Args:
        json_path: Path to JSON file

    Returns:
        List of SourceDefinitions
    """
    import json

    json_path = Path(json_path)
    with open(json_path) as f:
        config = json.load(f)

    return _parse_config(config, str(json_path))


def _parse_config(config: dict[str, Any], source_file: str) -> list[SourceDefinition]:
    """Parse a configuration dict into SourceDefinitions.

    Args:
        config: Configuration dict with 'sources' key
        source_file: Path to source file (for error messages)

    Returns:
        List of SourceDefinitions
    """
    definitions = []
    sources_config = config.get("sources", [])

    for idx, src in enumerate(sources_config):
        try:
            definition = _parse_source_config(src, f"{source_file}[{idx}]")
            definitions.append(definition)
        except Exception as e:
            raise ValueError(f"Error parsing source {idx} in {source_file}: {e}") from e

    return definitions


def _parse_source_config(src: dict[str, Any], location: str) -> SourceDefinition:
    """Parse a single source configuration into a SourceDefinition.

    Args:
        src: Source configuration dict
        location: Location string for error messages

    Returns:
        SourceDefinition
    """
    term = src.get("term")
    if not term:
        raise ValueError(f"Missing 'term' in source config at {location}")

    term_type = src.get("term_type", "topic")
    source_type = src.get("type", "").lower()
    trigger = TriggerCondition(src.get("trigger", "on_query"))
    cache_ttl = src.get("cache_ttl", src.get("cache_ttl_seconds", 0))
    priority = src.get("priority", 0)
    tags = src.get("tags", [])

    # Build the appropriate DataSource
    if source_type == "table":
        data_source = TableSource(
            name=src.get("name", f"{term}_table"),
            description=src.get("description", ""),
            connection_string=src.get("connection_string", ""),
            table=src.get("table", ""),
            query_template=src.get("query_template", src.get("query", "")),
            param_mapping=src.get("param_mapping", src.get("params", {})),
            limit=src.get("limit", 100),
            timeout_seconds=src.get("timeout", 30),
            enabled=src.get("enabled", True),
            cache_ttl_seconds=cache_ttl,
            trigger=trigger,
        )
    elif source_type == "api":
        data_source = APISource(
            name=src.get("name", f"{term}_api"),
            description=src.get("description", ""),
            url=src.get("url", ""),
            method=src.get("method", "GET"),
            headers=src.get("headers", {}),
            query_params=src.get("query_params", {}),
            body_template=src.get("body_template", src.get("body")),
            url_params=src.get("url_params", src.get("params", {})),
            header_params=src.get("header_params", {}),
            body_params=src.get("body_params", {}),
            timeout_seconds=src.get("timeout", 30),
            enabled=src.get("enabled", True),
            cache_ttl_seconds=cache_ttl,
            trigger=trigger,
            response_path=src.get("response_path", ""),
        )
    elif source_type == "mcp":
        data_source = MCPSource(
            name=src.get("name", f"{term}_mcp"),
            description=src.get("description", ""),
            server_name=src.get("server_name", src.get("server", "")),
            tool_name=src.get("tool_name", src.get("tool", "")),
            argument_mapping=src.get("argument_mapping", src.get("args", {})),
            static_arguments=src.get("static_arguments", {}),
            timeout_seconds=src.get("timeout", 60),
            enabled=src.get("enabled", True),
            cache_ttl_seconds=cache_ttl,
            trigger=trigger,
        )
    else:
        raise ValueError(
            f"Unknown source type '{source_type}' at {location}. "
            f"Supported types: table, api, mcp"
        )

    return SourceDefinition(
        term=term,
        term_type=term_type,
        source=data_source,
        priority=priority,
        tags=tags,
    )


class AsyncSourceExecutor:
    """Executes multiple sources in parallel using asyncio.

    Provides efficient parallel fetching for multiple sources,
    with configurable concurrency limits and timeout handling.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        default_timeout: float = 30.0,
    ):
        """Initialize executor.

        Args:
            max_concurrency: Maximum concurrent fetches
            default_timeout: Default timeout for operations
        """
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self._semaphore: asyncio.Semaphore | None = None

    async def fetch_all(
        self,
        sources: list[tuple[str, DataSource]],
        context: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, list[FetchResult]]:
        """Fetch from multiple sources in parallel.

        Args:
            sources: List of (term, DataSource) tuples
            context: Context dict for fetching
            timeout: Optional timeout override

        Returns:
            Dict mapping term -> list of FetchResults
        """
        timeout = timeout or self.default_timeout
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        # Create tasks for all sources
        tasks = []
        for term, source in sources:
            task = self._fetch_with_semaphore(term, source, context, timeout)
            tasks.append(task)

        # Execute all tasks
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by term
        results: dict[str, list[FetchResult]] = {}
        for (term, _), result in zip(sources, results_list, strict=False):
            if term not in results:
                results[term] = []

            if isinstance(result, Exception):
                # Convert exception to failed FetchResult
                results[term].append(
                    FetchResult(
                        source_name="unknown",
                        source_type=SourceType.FUNCTION,
                        data=None,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                results[term].append(result)

        return results

    async def _fetch_with_semaphore(
        self,
        term: str,
        source: DataSource,
        context: dict[str, Any],
        timeout: float,
    ) -> FetchResult:
        """Fetch from a single source with semaphore control."""
        if self._semaphore is None:
            raise RuntimeError("Semaphore not initialized")

        async with self._semaphore:
            try:
                # Check if source supports async
                if isinstance(source, AsyncFunctionSource):
                    return await asyncio.wait_for(source.fetch_async(context), timeout=timeout)
                # Run sync fetch in thread pool
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, source.fetch, context),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return FetchResult(
                    source_name=source.name,
                    source_type=source.source_type,
                    data=None,
                    success=False,
                    error=f"Timeout after {timeout}s",
                )
            except Exception as e:
                return FetchResult(
                    source_name=source.name,
                    source_type=source.source_type,
                    data=None,
                    success=False,
                    error=str(e),
                )


def discover_and_register(
    sources_path: str | Path,
    registry: SourceRegistry | None = None,
) -> tuple[SourceRegistry, list[tuple[str, Exception]]]:
    """Convenience function to discover sources and register them.

    Args:
        sources_path: Path to sources directory
        registry: Optional existing registry (creates new if None)

    Returns:
        Tuple of (registry, errors)
    """
    if registry is None:
        registry = SourceRegistry()

    discovery = SourceDiscovery(sources_path)
    discovery.discover()
    discovery.register_to(registry)

    return registry, discovery.get_errors()
