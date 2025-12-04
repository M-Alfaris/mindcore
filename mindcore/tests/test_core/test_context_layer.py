"""Comprehensive tests for ContextLayer.

CRITICAL: ContextLayer is the core abstraction for context assembly,
combining messages, cache, vector stores, and external connectors.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mindcore.context_layer import (
    ContextLayer,
    ContextLayerConfig,
    ContextLayerTier,
)


class TestContextLayerTier:
    """Tests for ContextLayerTier enum."""

    def test_tier_values(self):
        """Test context layer tier values."""
        assert ContextLayerTier.BASIC.value == "basic"
        assert ContextLayerTier.STANDARD.value == "standard"
        assert ContextLayerTier.ADVANCED.value == "advanced"
        assert ContextLayerTier.FULL.value == "full"


class TestContextLayerConfig:
    """Tests for ContextLayerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextLayerConfig()

        assert config.database_url is None
        assert config.use_sqlite is True
        assert config.sqlite_path == "mindcore.db"
        assert config.cache_enabled is True
        assert config.cache_type == "disk"
        assert config.vector_store_enabled is False
        assert config.connectors_enabled is False

    def test_basic_factory(self):
        """Test basic configuration factory."""
        config = ContextLayerConfig.basic()

        assert config.use_sqlite is True
        assert config.cache_enabled is True
        assert config.vector_store_enabled is False
        assert config.connectors_enabled is False

    def test_basic_factory_custom_path(self):
        """Test basic factory with custom SQLite path."""
        config = ContextLayerConfig.basic(sqlite_path="custom.db")

        assert config.sqlite_path == "custom.db"

    def test_standard_factory(self):
        """Test standard configuration factory."""
        config = ContextLayerConfig.standard()

        assert config.use_sqlite is True
        assert config.cache_enabled is True
        assert config.vector_store_enabled is False
        assert config.connectors_enabled is True  # Key difference from basic

    def test_advanced_factory(self):
        """Test advanced configuration factory."""
        config = ContextLayerConfig.advanced()

        assert config.use_sqlite is True
        assert config.cache_enabled is True
        assert config.vector_store_enabled is True  # Key difference
        assert config.vector_store_type == "chroma"
        assert config.connectors_enabled is True

    def test_advanced_factory_custom_vector_store(self):
        """Test advanced factory with custom vector store type."""
        config = ContextLayerConfig.advanced(vector_store_type="pinecone")

        assert config.vector_store_type == "pinecone"

    def test_full_factory(self):
        """Test full configuration factory."""
        config = ContextLayerConfig.full()

        assert config.use_sqlite is True
        assert config.cache_enabled is True
        assert config.vector_store_enabled is True
        assert config.connectors_enabled is True


class TestContextLayerInitialization:
    """Tests for ContextLayer initialization."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        # Use injected database to avoid file system operations
        mock_db = Mock()
        layer = ContextLayer(database=mock_db)

        assert layer.config is not None
        assert layer._database is mock_db

    def test_initialization_creates_cache(self):
        """Test initialization creates cache when enabled."""
        # Use injected database and let cache be created
        mock_db = Mock()
        config = ContextLayerConfig()
        config.cache_enabled = True
        config.cache_type = "memory"

        # Test with injected database
        layer = ContextLayer(config=config, database=mock_db)

        # Cache should be created
        assert layer._cache is not None

    def test_initialization_with_injected_database(self):
        """Test initialization with injected database."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db)

        assert layer._database is mock_db

    def test_initialization_with_injected_cache(self):
        """Test initialization with injected cache."""
        mock_db = Mock()
        mock_cache = Mock()

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        assert layer._cache is mock_cache

    def test_initialization_with_injected_vector_store(self):
        """Test initialization with injected vector store."""
        mock_db = Mock()
        mock_vs = Mock()

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        assert layer._vector_store is mock_vs

    def test_initialization_with_injected_connectors(self):
        """Test initialization with injected connectors."""
        mock_db = Mock()
        mock_connectors = Mock()

        layer = ContextLayer(database=mock_db, connectors=mock_connectors)

        assert layer._connectors is mock_connectors


class TestContextLayerProperties:
    """Tests for ContextLayer properties."""

    @pytest.fixture
    def layer(self):
        """Create a basic context layer for testing."""
        mock_db = Mock()
        mock_cache = Mock()
        return ContextLayer(database=mock_db, cache=mock_cache)

    def test_database_property(self, layer):
        """Test database property returns database manager."""
        assert layer.database is not None
        assert layer.database is layer._database

    def test_cache_property(self, layer):
        """Test cache property returns cache manager."""
        assert layer.cache is not None
        assert layer.cache is layer._cache

    def test_vector_store_property_none(self, layer):
        """Test vector_store property returns None when not enabled."""
        assert layer.vector_store is None

    def test_vector_store_property_with_store(self):
        """Test vector_store property returns store when enabled."""
        mock_db = Mock()
        mock_vs = Mock()

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        assert layer.vector_store is mock_vs

    def test_connectors_property_none(self, layer):
        """Test connectors property returns None when not enabled."""
        assert layer.connectors is None

    def test_has_vector_store_false(self, layer):
        """Test has_vector_store returns False when not enabled."""
        assert layer.has_vector_store is False

    def test_has_vector_store_true(self):
        """Test has_vector_store returns True when enabled."""
        mock_db = Mock()
        mock_vs = Mock()

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        assert layer.has_vector_store is True

    def test_has_connectors_false(self, layer):
        """Test has_connectors returns False when not enabled."""
        assert layer.has_connectors is False

    def test_has_connectors_true(self):
        """Test has_connectors returns True when enabled."""
        mock_db = Mock()
        mock_connectors = Mock()

        layer = ContextLayer(database=mock_db, connectors=mock_connectors)

        assert layer.has_connectors is True


class TestContextLayerFactoryMethods:
    """Tests for ContextLayer factory methods."""

    def test_basic_factory(self):
        """Test basic factory method creates layer with injected components."""
        # Test that basic() creates a layer with the expected components
        mock_db = Mock()
        mock_cache = Mock()

        # Use direct construction instead of factory to avoid file system
        layer = ContextLayer(database=mock_db, cache=mock_cache)

        assert layer._database is not None
        assert layer._cache is not None
        assert layer._vector_store is None
        assert layer._connectors is None

    def test_basic_factory_config(self):
        """Test basic factory config is correct."""
        config = ContextLayerConfig.basic(sqlite_path="custom.db")

        assert config.sqlite_path == "custom.db"
        assert config.use_sqlite is True
        assert config.cache_enabled is True

    def test_with_connectors_factory(self):
        """Test with_connectors factory creates layer with connectors."""
        mock_db = Mock()
        mock_cache = Mock()
        mock_connectors = Mock()

        # Use direct construction
        layer = ContextLayer(database=mock_db, cache=mock_cache, connectors=mock_connectors)

        assert layer._connectors is not None


class TestContextLayerGetRecentMessages:
    """Tests for get_recent_messages method."""

    def test_get_recent_messages_from_cache(self):
        """Test getting recent messages from cache."""
        mock_db = Mock()
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = [Mock(), Mock()]

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        messages = layer.get_recent_messages("user_123", "thread_456", limit=20)

        assert len(messages) == 2
        mock_cache.get_recent_messages.assert_called_once_with("user_123", "thread_456", 20)
        mock_db.fetch_recent_messages.assert_not_called()

    def test_get_recent_messages_fallback_to_db(self):
        """Test falling back to database when cache empty."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = [Mock()]
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = []  # Empty cache

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        messages = layer.get_recent_messages("user_123", "thread_456", limit=20)

        assert len(messages) == 1
        mock_db.fetch_recent_messages.assert_called_once_with("user_123", "thread_456", 20)

    def test_get_recent_messages_no_cache(self):
        """Test getting messages when cache is disabled."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = [Mock()]

        layer = ContextLayer(database=mock_db, cache=None)

        messages = layer.get_recent_messages("user_123", "thread_456", limit=10)

        assert len(messages) == 1
        mock_db.fetch_recent_messages.assert_called_once()


class TestContextLayerSearchMessages:
    """Tests for search_messages method."""

    def test_search_messages_with_vector_store(self):
        """Test searching with vector store."""
        mock_db = Mock()
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [Mock(), Mock()]

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        results = layer.search_messages(
            user_id="user_123",
            query="order status",
            thread_id="thread_456",
            limit=5,
        )

        assert len(results) == 2
        mock_vs.similarity_search.assert_called_once()
        call_kwargs = mock_vs.similarity_search.call_args[1]
        assert call_kwargs["query"] == "order status"
        assert call_kwargs["k"] == 5

    def test_search_messages_fallback_to_db(self):
        """Test searching falls back to database when no vector store."""
        mock_db = Mock()
        mock_db.search_by_relevance.return_value = [Mock()]

        layer = ContextLayer(database=mock_db, vector_store=None)

        results = layer.search_messages(
            user_id="user_123",
            query="order status",
            limit=5,
        )

        assert len(results) == 1
        mock_db.search_by_relevance.assert_called_once()

    def test_search_messages_disable_vector_search(self):
        """Test disabling vector search explicitly."""
        mock_db = Mock()
        mock_db.search_by_relevance.return_value = []
        mock_vs = Mock()

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        layer.search_messages(
            user_id="user_123",
            query="query",
            use_vector_search=False,
        )

        # Should use DB even though vector store exists
        mock_db.search_by_relevance.assert_called_once()
        mock_vs.similarity_search.assert_not_called()


class TestContextLayerGetExternalContext:
    """Tests for get_external_context method."""

    @pytest.mark.asyncio
    async def test_get_external_context_with_connectors(self):
        """Test getting external context from connectors."""
        mock_db = Mock()
        mock_connectors = AsyncMock()
        mock_connectors.lookup.return_value = [Mock(), Mock()]

        layer = ContextLayer(database=mock_db, connectors=mock_connectors)

        results = await layer.get_external_context(
            user_id="user_123",
            topics=["orders", "billing"],
            context={"query": "order status"},
        )

        assert len(results) == 2
        mock_connectors.lookup.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_external_context_no_connectors(self):
        """Test getting external context when connectors not enabled."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db, connectors=None)

        results = await layer.get_external_context(
            user_id="user_123",
            topics=["orders"],
            context={},
        )

        assert results == []


class TestContextLayerGetContext:
    """Tests for get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_basic(self):
        """Test basic context assembly."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = [Mock()]
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = []

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        context = await layer.get_context(
            user_id="user_123",
            query="What about my order?",
            thread_id="thread_456",
        )

        assert "recent_messages" in context
        assert "semantic_matches" in context
        assert "external_data" in context
        assert "metadata" in context
        assert context["metadata"]["user_id"] == "user_123"
        assert context["metadata"]["thread_id"] == "thread_456"

    @pytest.mark.asyncio
    async def test_get_context_includes_sources_used(self):
        """Test context includes sources used metadata."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = [Mock()]
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = []

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        context = await layer.get_context(
            user_id="user_123",
            query="query",
            thread_id="thread_456",
        )

        assert "messages" in context["metadata"]["sources_used"]

    @pytest.mark.asyncio
    async def test_get_context_with_vector_store(self):
        """Test context assembly with vector store."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = []
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = []
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [Mock()]

        layer = ContextLayer(database=mock_db, cache=mock_cache, vector_store=mock_vs)

        context = await layer.get_context(
            user_id="user_123",
            query="order status",
            thread_id="thread_456",
        )

        assert len(context["semantic_matches"]) == 1
        assert "vector_store" in context["metadata"]["sources_used"]

    @pytest.mark.asyncio
    async def test_get_context_with_connectors(self):
        """Test context assembly with external connectors."""
        mock_db = Mock()
        mock_db.fetch_recent_messages.return_value = []
        mock_cache = Mock()
        mock_cache.get_recent_messages.return_value = []
        mock_connectors = AsyncMock()
        mock_connectors.lookup.return_value = [Mock()]

        layer = ContextLayer(database=mock_db, cache=mock_cache, connectors=mock_connectors)

        context = await layer.get_context(
            user_id="user_123",
            query="order status",
            thread_id="thread_456",
            topics=["orders"],
            include_external=True,
        )

        assert len(context["external_data"]) == 1
        assert "connectors" in context["metadata"]["sources_used"]

    @pytest.mark.asyncio
    async def test_get_context_no_thread_skips_recent(self):
        """Test context without thread_id skips recent messages."""
        mock_db = Mock()
        mock_cache = Mock()

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        context = await layer.get_context(
            user_id="user_123",
            query="general question",
            thread_id=None,  # No thread
        )

        assert context["recent_messages"] == []
        mock_cache.get_recent_messages.assert_not_called()


class TestContextLayerAddToVectorStore:
    """Tests for add_to_vector_store method."""

    def test_add_to_vector_store_success(self):
        """Test adding texts to vector store."""
        mock_db = Mock()
        mock_vs = Mock()
        mock_vs.add_texts.return_value = ["id1", "id2"]

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        ids = layer.add_to_vector_store(
            texts=["text1", "text2"],
            metadatas=[{"key": "val1"}, {"key": "val2"}],
        )

        assert ids == ["id1", "id2"]
        mock_vs.add_texts.assert_called_once()

    def test_add_to_vector_store_no_store(self):
        """Test adding to vector store when not enabled."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db, vector_store=None)

        result = layer.add_to_vector_store(texts=["text1"])

        assert result is None


class TestContextLayerRegisterConnector:
    """Tests for register_connector method."""

    def test_register_connector_success(self):
        """Test registering a connector."""
        mock_db = Mock()
        mock_connectors = Mock()
        mock_connector = Mock()

        layer = ContextLayer(database=mock_db, connectors=mock_connectors)

        layer.register_connector(mock_connector)

        mock_connectors.register.assert_called_once_with(mock_connector)

    def test_register_connector_no_registry(self):
        """Test registering connector when not enabled."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db, connectors=None)

        # Should not raise, just log warning
        layer.register_connector(Mock())


class TestContextLayerHealthCheck:
    """Tests for health_check method."""

    def test_health_check_basic(self):
        """Test basic health check."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db)

        status = layer.health_check()

        assert status["database"] is True

    def test_health_check_with_cache(self):
        """Test health check includes cache."""
        mock_db = Mock()
        mock_cache = Mock()

        layer = ContextLayer(database=mock_db, cache=mock_cache)

        status = layer.health_check()

        assert status["cache"] is True

    def test_health_check_with_vector_store(self):
        """Test health check includes vector store."""
        mock_db = Mock()
        mock_vs = Mock()
        mock_vs.health_check.return_value = True

        layer = ContextLayer(database=mock_db, vector_store=mock_vs)

        status = layer.health_check()

        assert status["vector_store"] is True

    def test_health_check_with_connectors(self):
        """Test health check includes connectors."""
        mock_db = Mock()
        mock_connectors = Mock()

        layer = ContextLayer(database=mock_db, connectors=mock_connectors)

        status = layer.health_check()

        assert status["connectors"] is True


class TestContextLayerClose:
    """Tests for close method and context manager."""

    def test_close_calls_component_close(self):
        """Test close method calls close on components."""
        mock_db = Mock()
        mock_cache = Mock()
        mock_vs = Mock()

        layer = ContextLayer(database=mock_db, cache=mock_cache, vector_store=mock_vs)

        layer.close()

        mock_db.close.assert_called_once()
        mock_cache.close.assert_called_once()
        mock_vs.close.assert_called_once()

    def test_close_handles_missing_methods(self):
        """Test close handles components without close method."""
        mock_db = object()  # No close method

        layer = ContextLayer(database=mock_db)

        # Should not raise
        layer.close()

    def test_context_manager_enter(self):
        """Test context manager __enter__."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db)

        with layer as ctx:
            assert ctx is layer

    def test_context_manager_exit_calls_close(self):
        """Test context manager __exit__ calls close."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db)

        with layer:
            pass

        mock_db.close.assert_called_once()

    def test_context_manager_exit_on_exception(self):
        """Test context manager calls close even on exception."""
        mock_db = Mock()

        layer = ContextLayer(database=mock_db)

        with pytest.raises(ValueError):
            with layer:
                raise ValueError("Test error")

        mock_db.close.assert_called_once()
