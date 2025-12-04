"""Tests for AsyncMindcoreClient - Async interface for high-performance applications."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mindcore.async_client import AsyncMindcoreClient
from mindcore.core.schemas import AssembledContext, Message, MessageMetadata, MessageRole


class TestAsyncMindcoreClient:
    """Test cases for AsyncMindcoreClient."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock ConfigLoader."""
        config = Mock()
        config.get_database_config.return_value = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test",
            "password": "test",
        }
        config.get_cache_config.return_value = {
            "max_size": 50,
            "ttl": 3600,
        }
        config.get_llm_config.return_value = {
            "provider": "openai",
            "llama_cpp": {},
            "openai": {
                "api_key": "test-key",
                "model": "gpt-4o-mini",
            },
            "defaults": {
                "temperature": 0.3,
                "max_tokens_enrichment": 800,
                "max_tokens_context": 1500,
            },
        }
        return config

    @pytest.fixture
    def mock_db(self):
        """Create a mock async database."""
        db = AsyncMock()
        db.connect = AsyncMock()
        db.close = AsyncMock()
        db.insert_message = AsyncMock(return_value=True)
        db.fetch_recent_messages = AsyncMock(return_value=[])
        db.get_message_by_id = AsyncMock(return_value=None)
        db.update_message_metadata = AsyncMock()
        db.search_by_relevance = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache manager."""
        cache = Mock()
        cache.add_message = Mock()
        cache.update_message = Mock()
        cache.get_recent_messages = Mock(return_value=[])
        cache.get_session_metadata = Mock(return_value={})
        cache.clear_thread = Mock()
        cache.clear_all = Mock()
        cache.close = Mock()
        return cache

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.name = "mock"
        provider.is_available.return_value = True
        provider.close = Mock()
        provider.generate = Mock()
        return provider

    @pytest.fixture
    def mock_metadata_agent(self, sample_message):
        """Create a mock metadata enrichment agent."""
        agent = Mock()
        agent.process = Mock(return_value=sample_message)
        return agent

    @pytest.fixture
    def mock_context_agent(self):
        """Create a mock context assembler agent."""
        agent = Mock()
        agent.process = Mock(
            return_value=AssembledContext(
                assembled_context="Test context",
                key_points=["Point 1"],
                relevant_message_ids=["msg_1"],
                metadata={"source": "test"},
            )
        )
        return agent

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = AsyncMindcoreClient(use_sqlite=True)

        assert client._use_sqlite is True
        assert client._connected is False
        assert client.db is None
        assert client.cache is None

    @pytest.mark.asyncio
    async def test_client_context_manager(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test client works as async context manager."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                assert client._connected is True
                assert client.db is not None

            # After exit, resources should be cleaned up
            mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_sqlite(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test connecting with SQLite database."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
        ):
            client = AsyncMindcoreClient(use_sqlite=True)
            await client.connect()

            assert client._connected is True
            mock_db.connect.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test connect() is idempotent."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
        ):
            client = AsyncMindcoreClient(use_sqlite=True)
            await client.connect()
            await client.connect()  # Second call should be no-op

            # Should only connect once
            assert mock_db.connect.call_count == 1

            await client.close()

    @pytest.mark.asyncio
    async def test_ingest_message_not_connected(self):
        """Test ingest_message raises error when not connected."""
        client = AsyncMindcoreClient(use_sqlite=True)

        with pytest.raises(RuntimeError, match="not connected"):
            await client.ingest_message({"user_id": "test"})

    @pytest.mark.asyncio
    async def test_ingest_message(
        self, mock_config, mock_db, mock_cache, mock_llm_provider, mock_metadata_agent, sample_message
    ):
        """Test message ingestion with enrichment."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent", return_value=mock_metadata_agent),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                message_dict = {
                    "user_id": "user_123",
                    "thread_id": "thread_456",
                    "session_id": "session_789",
                    "role": "user",
                    "text": "Hello, AI!",
                }

                result = await client.ingest_message(message_dict)

                assert isinstance(result, Message)
                mock_metadata_agent.process.assert_called_once()
                mock_db.insert_message.assert_called_once()
                mock_cache.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_message_validation_error(
        self, mock_config, mock_db, mock_cache, mock_llm_provider
    ):
        """Test message ingestion with invalid data."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                # Missing required fields
                with pytest.raises(ValueError, match="Invalid message"):
                    await client.ingest_message({"user_id": "test"})

    @pytest.mark.asyncio
    async def test_ingest_message_fast(
        self, mock_config, mock_db, mock_cache, mock_llm_provider
    ):
        """Test fast message ingestion (background enrichment)."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                message_dict = {
                    "user_id": "user_123",
                    "thread_id": "thread_456",
                    "session_id": "session_789",
                    "role": "user",
                    "text": "Hello, AI!",
                }

                result = await client.ingest_message_fast(message_dict)

                assert isinstance(result, Message)
                # Message should have empty metadata (not enriched yet)
                assert result.metadata is not None
                mock_db.insert_message.assert_called_once()
                mock_cache.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_not_connected(self):
        """Test get_context raises error when not connected."""
        client = AsyncMindcoreClient(use_sqlite=True)

        with pytest.raises(RuntimeError, match="not connected"):
            await client.get_context("user_123", "thread_456", "test query")

    @pytest.mark.asyncio
    async def test_get_context(
        self, mock_config, mock_db, mock_cache, mock_llm_provider, mock_context_agent, sample_messages
    ):
        """Test context retrieval."""
        mock_cache.get_recent_messages.return_value = sample_messages

        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent", return_value=mock_context_agent),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                context = await client.get_context(
                    user_id="user_123",
                    thread_id="thread_456",
                    query="What did we discuss?",
                )

                assert isinstance(context, AssembledContext)
                assert context.assembled_context == "Test context"
                mock_cache.get_recent_messages.assert_called()

    @pytest.mark.asyncio
    async def test_get_context_merges_cache_and_db(
        self, mock_config, mock_db, mock_cache, mock_llm_provider, mock_context_agent
    ):
        """Test context retrieval merges cache and database messages."""
        # Cache has 2 messages
        cached_msgs = [
            Message(
                message_id="msg_1",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="Cached message 1",
                metadata=MessageMetadata(),
            ),
            Message(
                message_id="msg_2",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.ASSISTANT,
                raw_text="Cached message 2",
                metadata=MessageMetadata(),
            ),
        ]

        # DB has different messages
        db_msgs = [
            Message(
                message_id="msg_3",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="DB message 3",
                metadata=MessageMetadata(),
            ),
        ]

        mock_cache.get_recent_messages.return_value = cached_msgs
        mock_db.fetch_recent_messages.return_value = db_msgs

        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent", return_value=mock_context_agent),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                await client.get_context(
                    user_id="user_123",
                    thread_id="thread_456",
                    query="What did we discuss?",
                    max_messages=50,
                )

                # Should call context agent with merged messages
                mock_context_agent.process.assert_called_once()
                call_args = mock_context_agent.process.call_args
                messages = call_args[0][0]
                # Should have all 3 unique messages
                assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_get_context_validation_error(
        self, mock_config, mock_db, mock_cache, mock_llm_provider
    ):
        """Test context retrieval with invalid parameters."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                with pytest.raises(ValueError, match="Invalid query"):
                    await client.get_context(
                        user_id="",  # Empty user_id
                        thread_id="thread_456",
                        query="test",
                    )

    @pytest.mark.asyncio
    async def test_get_message(self, mock_config, mock_db, mock_cache, mock_llm_provider, sample_message):
        """Test getting a single message by ID."""
        mock_db.get_message_by_id.return_value = sample_message

        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                result = await client.get_message("msg_test_001")

                assert result == sample_message
                mock_db.get_message_by_id.assert_called_once_with("msg_test_001")

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test cache clearing."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                # Clear specific thread
                client.clear_cache(user_id="user_123", thread_id="thread_456")
                mock_cache.clear_thread.assert_called_once_with("user_123", "thread_456")

                # Clear all
                client.clear_cache()
                mock_cache.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_name_property(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test provider_name property."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            client = AsyncMindcoreClient(use_sqlite=True)

            # Before connect
            assert client.provider_name == "not_initialized"

            await client.connect()

            # After connect
            assert client.provider_name == "mock"

            await client.close()

    @pytest.mark.asyncio
    async def test_llm_provider_property(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test llm_provider property."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            async with AsyncMindcoreClient(use_sqlite=True) as client:
                assert client.llm_provider == mock_llm_provider

    @pytest.mark.asyncio
    async def test_close_cancels_enrichment_task(self, mock_config, mock_db, mock_cache, mock_llm_provider):
        """Test close() cancels background enrichment task."""
        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.DiskCacheManager", return_value=mock_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            client = AsyncMindcoreClient(use_sqlite=True)
            await client.connect()

            # Start a background task
            client._enrichment_task = asyncio.create_task(asyncio.sleep(100))

            await client.close()

            # Task should be cancelled
            assert client._enrichment_task.cancelled() or client._enrichment_task.done()

    @pytest.mark.asyncio
    async def test_in_memory_cache(self, mock_config, mock_db, mock_llm_provider):
        """Test using in-memory cache instead of disk cache."""
        mock_memory_cache = Mock()
        mock_memory_cache.add_message = Mock()
        mock_memory_cache.get_recent_messages = Mock(return_value=[])

        with (
            patch("mindcore.async_client.ConfigLoader", return_value=mock_config),
            patch("mindcore.async_client.AsyncSQLiteManager", return_value=mock_db),
            patch("mindcore.async_client.CacheManager", return_value=mock_memory_cache),
            patch("mindcore.async_client.create_provider", return_value=mock_llm_provider),
            patch("mindcore.async_client.MetadataAgent"),
            patch("mindcore.async_client.ContextAgent"),
        ):
            client = AsyncMindcoreClient(use_sqlite=True, persistent_cache=False)
            await client.connect()

            assert client.cache == mock_memory_cache

            await client.close()


class TestAsyncMindcoreClientAlias:
    """Test AsyncMindcore alias."""

    def test_alias_exists(self):
        """Test AsyncMindcore is an alias for AsyncMindcoreClient."""
        from mindcore.async_client import AsyncMindcore, AsyncMindcoreClient

        assert AsyncMindcore is AsyncMindcoreClient
