"""Tests for the SVL Registry - unified source configuration."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mindcore.v2.svl import (
    AsyncFunctionSource,
    AsyncSourceExecutor,
    FetchResult,
    SharedVocabularyLayer,
    SourceDefinition,
    SourceDiscovery,
    SourceType,
    TableSource,
    TriggerCondition,
    clear_registered_sources,
    get_registered_sources,
    load_sources_from_yaml,
    source,
)


class TestSourceDecorator:
    """Tests for the @source decorator."""

    def setup_method(self):
        """Clear registered sources before each test."""
        clear_registered_sources()

    def test_decorator_sync_function(self):
        """Test @source decorator with sync function."""

        @source(term="test_topic", term_type="topic")
        def my_sync_source(context: dict) -> dict:
            return {"data": "test", "user_id": context.get("user_id")}

        # Check source was registered
        registered = get_registered_sources()
        assert len(registered) == 1

        definition = registered[0]
        assert definition.term == "test_topic"
        assert definition.term_type == "topic"
        assert isinstance(definition.source, AsyncFunctionSource)
        assert definition.source.is_async is False

        # Test fetching
        result = definition.source.fetch({"user_id": "123"})
        assert result.success is True
        assert result.data == {"data": "test", "user_id": "123"}

    def test_decorator_async_function(self):
        """Test @source decorator with async function."""

        @source(term="async_topic", term_type="topic", cache_ttl=60)
        async def my_async_source(context: dict) -> list:
            return [{"id": 1}, {"id": 2}]

        registered = get_registered_sources()
        assert len(registered) == 1

        definition = registered[0]
        assert definition.term == "async_topic"
        assert isinstance(definition.source, AsyncFunctionSource)
        assert definition.source.is_async is True
        assert definition.source.cache_ttl_seconds == 60

    def test_decorator_with_class(self):
        """Test @source decorator with DataSource subclass."""

        @source(term="products", term_type="topic")
        class ProductSource(TableSource):
            name = "products"
            connection_string = "sqlite:///test.db"
            query_template = "SELECT * FROM products"

        registered = get_registered_sources()
        assert len(registered) == 1

        definition = registered[0]
        assert definition.term == "products"
        assert isinstance(definition.source, TableSource)
        assert definition.source.name == "products"

    def test_decorator_options(self):
        """Test @source decorator with all options."""

        @source(
            term="detailed",
            term_type="category",
            name="custom_name",
            description="Custom description",
            trigger=TriggerCondition.ON_DEMAND,
            cache_ttl=300,
            timeout=60.0,
            enabled=True,
            priority=10,
            tags=["important", "cached"],
        )
        def detailed_source(context: dict) -> dict:
            return {}

        registered = get_registered_sources()
        definition = registered[0]

        assert definition.term == "detailed"
        assert definition.term_type == "category"
        assert definition.priority == 10
        assert definition.tags == ["important", "cached"]
        assert definition.source.name == "custom_name"
        assert definition.source.description == "Custom description"
        assert definition.source.trigger == TriggerCondition.ON_DEMAND
        assert definition.source.cache_ttl_seconds == 300

    def test_multiple_sources(self):
        """Test registering multiple sources."""

        @source(term="source1", priority=1)
        def s1(ctx):
            return "s1"

        @source(term="source2", priority=2)
        def s2(ctx):
            return "s2"

        @source(term="source3", priority=3)
        def s3(ctx):
            return "s3"

        registered = get_registered_sources()
        assert len(registered) == 3


class TestAsyncFunctionSource:
    """Tests for AsyncFunctionSource."""

    def test_sync_function_execution(self):
        """Test executing a sync function."""

        def sync_func(context: dict) -> dict:
            return {"result": context.get("value", 0) * 2}

        source = AsyncFunctionSource(
            name="sync_test",
            function=sync_func,
            is_async=False,
        )

        result = source.fetch({"value": 21})
        assert result.success is True
        assert result.data == {"result": 42}
        assert result.source_name == "sync_test"
        assert result.source_type == SourceType.FUNCTION

    def test_async_function_execution(self):
        """Test executing an async function."""

        async def async_func(context: dict) -> dict:
            await asyncio.sleep(0.01)
            return {"async": True, "user": context.get("user")}

        source = AsyncFunctionSource(
            name="async_test",
            function=async_func,
            is_async=True,
        )

        result = source.fetch({"user": "test_user"})
        assert result.success is True
        assert result.data == {"async": True, "user": "test_user"}

    def test_function_error_handling(self):
        """Test error handling in function execution."""

        def error_func(context: dict) -> dict:
            raise ValueError("Test error")

        source = AsyncFunctionSource(
            name="error_test",
            function=error_func,
            is_async=False,
        )

        result = source.fetch({})
        assert result.success is False
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_fetch_async(self):
        """Test async fetch method."""

        async def async_func(context: dict) -> list:
            await asyncio.sleep(0.01)
            return [1, 2, 3]

        source = AsyncFunctionSource(
            name="async_fetch_test",
            function=async_func,
            is_async=True,
        )

        result = await source.fetch_async({})
        assert result.success is True
        assert result.data == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_fetch_async_timeout(self):
        """Test async fetch with timeout."""

        async def slow_func(context: dict) -> dict:
            await asyncio.sleep(10)  # Long delay
            return {}

        source = AsyncFunctionSource(
            name="timeout_test",
            function=slow_func,
            is_async=True,
            timeout_seconds=0.1,
        )

        result = await source.fetch_async({})
        assert result.success is False
        assert "Timeout" in result.error


class TestAsyncSourceExecutor:
    """Tests for AsyncSourceExecutor."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution of multiple sources."""
        execution_order = []

        async def source1(ctx):
            execution_order.append("start_1")
            await asyncio.sleep(0.1)
            execution_order.append("end_1")
            return "result_1"

        async def source2(ctx):
            execution_order.append("start_2")
            await asyncio.sleep(0.05)
            execution_order.append("end_2")
            return "result_2"

        s1 = AsyncFunctionSource(name="s1", function=source1, is_async=True)
        s2 = AsyncFunctionSource(name="s2", function=source2, is_async=True)

        executor = AsyncSourceExecutor(max_concurrency=10)
        results = await executor.fetch_all(
            sources=[("topic1", s1), ("topic2", s2)],
            context={},
        )

        # Both should have started before either ended
        assert "start_1" in execution_order
        assert "start_2" in execution_order
        assert results["topic1"][0].data == "result_1"
        assert results["topic2"][0].data == "result_2"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        concurrent_count = []
        max_concurrent = 0

        async def counting_source(ctx):
            nonlocal max_concurrent
            concurrent_count.append(1)
            current = len(concurrent_count)
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            concurrent_count.pop()
            return current

        sources = [
            (
                f"topic{i}",
                AsyncFunctionSource(name=f"s{i}", function=counting_source, is_async=True),
            )
            for i in range(5)
        ]

        executor = AsyncSourceExecutor(max_concurrency=2)
        await executor.fetch_all(sources=sources, context={})

        assert max_concurrent <= 2


class TestSourceDiscovery:
    """Tests for SourceDiscovery."""

    def setup_method(self):
        """Clear registered sources before each test."""
        clear_registered_sources()

    def test_discover_python_modules(self):
        """Test discovering sources from Python modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir) / "sources"
            sources_dir.mkdir()

            # Create a Python module with sources
            module_content = """
from mindcore.v2.svl import source

@source(term="discovered_topic", term_type="topic")
def discovered_source(context: dict) -> dict:
    return {"discovered": True}
"""
            (sources_dir / "test_sources.py").write_text(module_content)

            discovery = SourceDiscovery(sources_dir)
            discovered = discovery.discover()

            assert len(discovered) >= 1
            terms = [d.term for d in discovered]
            assert "discovered_topic" in terms

    def test_discover_yaml_config(self):
        """Test discovering sources from YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)

            yaml_content = """
sources:
  - term: "yaml_topic"
    term_type: "topic"
    type: "api"
    url: "https://api.example.com/data"
    method: "GET"
    cache_ttl: 60
"""
            (sources_dir / "config.yaml").write_text(yaml_content)

            discovery = SourceDiscovery(sources_dir)
            discovered = discovery.discover()

            assert len(discovered) == 1
            assert discovered[0].term == "yaml_topic"
            assert discovered[0].source.url == "https://api.example.com/data"

    def test_register_to_registry(self):
        """Test registering discovered sources to a registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)

            yaml_content = """
sources:
  - term: "reg_topic"
    type: "api"
    url: "https://api.example.com"
"""
            (sources_dir / "config.yaml").write_text(yaml_content)

            from mindcore.v2.svl import SourceRegistry

            registry = SourceRegistry()
            discovery = SourceDiscovery(sources_dir)
            discovery.discover()
            count = discovery.register_to(registry)

            assert count == 1
            assert "reg_topic" in registry.get_mapped_terms()


class TestLoadSourcesFromYaml:
    """Tests for YAML configuration loading."""

    def test_load_table_source(self):
        """Test loading a table source from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("""
sources:
  - term: "users"
    type: "table"
    connection_string: "postgresql://localhost/db"
    query_template: "SELECT * FROM users WHERE id = :id"
    param_mapping:
      user_id: "id"
    cache_ttl: 120
""")
            f.flush()

            sources = load_sources_from_yaml(f.name)

            assert len(sources) == 1
            assert sources[0].term == "users"
            assert isinstance(sources[0].source, TableSource)
            assert sources[0].source.cache_ttl_seconds == 120

    def test_load_api_source(self):
        """Test loading an API source from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("""
sources:
  - term: "weather"
    type: "api"
    url: "https://api.weather.com/current"
    method: "GET"
    headers:
      Authorization: "Bearer token"
    response_path: "data.current"
    trigger: "on_demand"
""")
            f.flush()

            sources = load_sources_from_yaml(f.name)

            assert len(sources) == 1
            assert sources[0].term == "weather"
            assert sources[0].source.url == "https://api.weather.com/current"
            assert sources[0].source.response_path == "data.current"
            assert sources[0].source.trigger == TriggerCondition.ON_DEMAND

    def test_load_mcp_source(self):
        """Test loading an MCP source from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("""
sources:
  - term: "search"
    type: "mcp"
    server_name: "brave-search"
    tool_name: "brave_web_search"
    argument_mapping:
      query: "search_query"
""")
            f.flush()

            sources = load_sources_from_yaml(f.name)

            assert len(sources) == 1
            assert sources[0].term == "search"
            assert sources[0].source.server_name == "brave-search"
            assert sources[0].source.tool_name == "brave_web_search"

    def test_load_multiple_sources(self):
        """Test loading multiple sources from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("""
sources:
  - term: "topic1"
    type: "api"
    url: "https://api1.com"
  - term: "topic2"
    type: "api"
    url: "https://api2.com"
  - term: "topic3"
    type: "table"
    connection_string: "sqlite:///test.db"
    query_template: "SELECT * FROM data"
""")
            f.flush()

            sources = load_sources_from_yaml(f.name)
            assert len(sources) == 3


class TestSharedVocabularyLayerIntegration:
    """Tests for SVL integration with source discovery."""

    def setup_method(self):
        """Clear registered sources before each test."""
        clear_registered_sources()

    def test_discover_sources_method(self):
        """Test SharedVocabularyLayer.discover_sources()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)

            yaml_content = """
sources:
  - term: "integration_topic"
    type: "api"
    url: "https://api.example.com"
"""
            (sources_dir / "config.yaml").write_text(yaml_content)

            svl = SharedVocabularyLayer()
            count, errors = svl.discover_sources(str(sources_dir))

            assert count == 1
            assert len(errors) == 0
            assert "integration_topic" in svl.get_mapped_terms()

    def test_fetch_from_discovered_sources(self):
        """Test fetching from discovered sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)

            # Create a Python module with a testable source
            module_content = """
from mindcore.v2.svl import source

@source(term="fetchable", term_type="topic")
def fetchable_source(context: dict) -> dict:
    return {"fetched": True, "user": context.get("user_id")}
"""
            (sources_dir / "fetchable.py").write_text(module_content)

            svl = SharedVocabularyLayer()
            count, errors = svl.discover_sources(str(sources_dir))

            assert count >= 1
            assert "fetchable" in svl.get_mapped_terms()

            # Fetch data
            results = svl.fetch_for_topics(["fetchable"], {"user_id": "test123"})
            assert "fetchable" in results
            assert results["fetchable"][0].success is True
            assert results["fetchable"][0].data["fetched"] is True

    @pytest.mark.asyncio
    async def test_fetch_async_from_discovered_sources(self):
        """Test async fetching from discovered sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)

            module_content = """
from mindcore.v2.svl import source
import asyncio

@source(term="async_fetchable", term_type="topic")
async def async_source(context: dict) -> dict:
    await asyncio.sleep(0.01)
    return {"async": True}
"""
            (sources_dir / "async_source.py").write_text(module_content)

            svl = SharedVocabularyLayer()
            svl.discover_sources(str(sources_dir))

            results = await svl.fetch_for_topics_async(
                ["async_fetchable"],
                {},
                max_concurrency=5,
            )

            assert "async_fetchable" in results
            assert results["async_fetchable"][0].success is True


class TestSourceDefinition:
    """Tests for SourceDefinition dataclass."""

    def test_source_definition_creation(self):
        """Test creating a SourceDefinition."""
        source = AsyncFunctionSource(
            name="test",
            function=lambda ctx: {},
            is_async=False,
        )

        definition = SourceDefinition(
            term="my_topic",
            term_type="topic",
            source=source,
            priority=5,
            tags=["tag1", "tag2"],
        )

        assert definition.term == "my_topic"
        assert definition.term_type == "topic"
        assert definition.priority == 5
        assert definition.tags == ["tag1", "tag2"]

    def test_source_definition_defaults(self):
        """Test SourceDefinition default values."""
        source = AsyncFunctionSource(name="test", function=lambda ctx: {})

        definition = SourceDefinition(
            term="topic",
            term_type="category",
            source=source,
        )

        assert definition.priority == 0
        assert definition.tags == []
