"""Tests for Async Database Managers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mindcore.core.async_db import AsyncDatabaseManager, AsyncSQLiteManager
from mindcore.core.schemas import Message, MessageMetadata, MessageRole


class TestAsyncSQLiteManager:
    """Tests for AsyncSQLiteManager class."""

    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Hello, this is a test message",
            metadata=MessageMetadata(
                topics=["general", "testing"],
                categories=["question"],
                intent="ask_question",
                importance=0.7,
            ),
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_messages(self):
        """Create multiple sample messages for testing."""
        messages = []
        base_time = datetime.now(timezone.utc)
        for i in range(5):
            msg = Message(
                message_id=f"msg_{i:03d}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                raw_text=f"Test message {i}",
                metadata=MessageMetadata(
                    topics=["general", "testing"] if i % 2 == 0 else ["orders"],
                    categories=["question"] if i % 2 == 0 else ["response"],
                    intent="ask_question" if i % 2 == 0 else "provide_info",
                    importance=0.5 + (i * 0.1),
                ),
                created_at=base_time,
            )
            messages.append(msg)
        return messages

    # =====================
    # Initialization Tests
    # =====================

    def test_initialization(self):
        """Test AsyncSQLiteManager initializes correctly."""
        manager = AsyncSQLiteManager(":memory:")
        assert manager.db_path == ":memory:"
        assert manager._connection is None
        assert manager._initialized is False

    def test_initialization_with_custom_path(self):
        """Test AsyncSQLiteManager with custom database path."""
        manager = AsyncSQLiteManager("/tmp/test.db")
        assert manager.db_path == "/tmp/test.db"

    # =====================
    # Connection Tests
    # =====================

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager pattern."""
        manager = AsyncSQLiteManager(":memory:")

        async with manager as db:
            assert db._connection is not None
            assert db._initialized is True

        # After exit, connection should be closed
        assert manager._connection is None

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connect method."""
        manager = AsyncSQLiteManager(":memory:")
        await manager.connect()

        assert manager._connection is not None
        assert manager._initialized is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        """Test connect raises ImportError if aiosqlite not installed."""
        manager = AsyncSQLiteManager(":memory:")

        with patch.dict("sys.modules", {"aiosqlite": None}):
            with patch("builtins.__import__", side_effect=ImportError("No aiosqlite")):
                with pytest.raises(ImportError, match="aiosqlite is required"):
                    await manager.connect()

    def test_ensure_connected_raises_when_not_connected(self):
        """Test _ensure_connected raises RuntimeError when not connected."""
        manager = AsyncSQLiteManager(":memory:")

        with pytest.raises(RuntimeError, match="Database not connected"):
            manager._ensure_connected()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method."""
        manager = AsyncSQLiteManager(":memory:")
        await manager.connect()
        await manager.close()

        assert manager._connection is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        """Test close when not connected does nothing."""
        manager = AsyncSQLiteManager(":memory:")
        await manager.close()  # Should not raise

    # =====================
    # Message CRUD Tests
    # =====================

    @pytest.mark.asyncio
    async def test_insert_message(self, sample_message):
        """Test inserting a message."""
        async with AsyncSQLiteManager(":memory:") as db:
            result = await db.insert_message(sample_message)
            assert result is True

            msg = await db.get_message_by_id(sample_message.message_id)
            assert msg is not None
            assert msg.message_id == sample_message.message_id

    @pytest.mark.asyncio
    async def test_insert_message_with_dict_metadata(self):
        """Test inserting message with dict metadata."""
        async with AsyncSQLiteManager(":memory:") as db:
            msg = Message(
                message_id="msg_dict",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="Dict metadata test",
                metadata={"topics": ["test"]},  # type: ignore
            )
            result = await db.insert_message(msg)
            assert result is True

    @pytest.mark.asyncio
    async def test_insert_message_exception(self):
        """Test insert_message handles exception."""
        async with AsyncSQLiteManager(":memory:") as db:
            with patch.object(db._connection, "execute", side_effect=Exception("DB Error")):
                msg = Message(
                    message_id="msg_error",
                    user_id="user_123",
                    thread_id="thread_456",
                    session_id="session_789",
                    role=MessageRole.USER,
                    raw_text="Test",
                )
                result = await db.insert_message(msg)
                assert result is False

    @pytest.mark.asyncio
    async def test_get_message_by_id(self, sample_message):
        """Test getting a message by ID."""
        async with AsyncSQLiteManager(":memory:") as db:
            await db.insert_message(sample_message)

            msg = await db.get_message_by_id(sample_message.message_id)
            assert msg is not None
            assert msg.user_id == sample_message.user_id
            assert msg.raw_text == sample_message.raw_text

    @pytest.mark.asyncio
    async def test_get_message_by_id_not_found(self):
        """Test getting non-existent message returns None."""
        async with AsyncSQLiteManager(":memory:") as db:
            msg = await db.get_message_by_id("nonexistent")
            assert msg is None

    @pytest.mark.asyncio
    async def test_fetch_recent_messages(self, sample_messages):
        """Test fetching recent messages."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.fetch_recent_messages("user_123", "thread_456", limit=10)
            assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_fetch_recent_messages_with_limit(self, sample_messages):
        """Test fetching with limit."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.fetch_recent_messages("user_123", "thread_456", limit=2)
            assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_fetch_recent_messages_empty(self):
        """Test fetching from empty database."""
        async with AsyncSQLiteManager(":memory:") as db:
            messages = await db.fetch_recent_messages("no_user", "no_thread")
            assert messages == []

    @pytest.mark.asyncio
    async def test_update_message_metadata(self, sample_message):
        """Test updating message metadata."""
        async with AsyncSQLiteManager(":memory:") as db:
            await db.insert_message(sample_message)

            new_metadata = MessageMetadata(
                topics=["updated"],
                importance=0.9,
            )
            result = await db.update_message_metadata(sample_message.message_id, new_metadata)
            assert result is True

            msg = await db.get_message_by_id(sample_message.message_id)
            assert "updated" in msg.metadata.topics

    @pytest.mark.asyncio
    async def test_update_message_metadata_not_found(self):
        """Test updating metadata for non-existent message."""
        async with AsyncSQLiteManager(":memory:") as db:
            result = await db.update_message_metadata(
                "nonexistent",
                MessageMetadata(topics=["test"]),
            )
            assert result is False

    # =====================
    # Search Tests
    # =====================

    @pytest.mark.asyncio
    async def test_search_by_relevance(self, sample_messages):
        """Test search by relevance."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                thread_id="thread_456",
                limit=10,
            )
            assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_search_by_relevance_with_topics(self, sample_messages):
        """Test search with topic filter."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                topics=["orders"],
                limit=10,
            )
            for msg in messages:
                assert "orders" in msg.metadata.topics

    @pytest.mark.asyncio
    async def test_search_by_relevance_with_categories(self, sample_messages):
        """Test search with category filter."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                categories=["question"],
                limit=10,
            )
            for msg in messages:
                assert "question" in msg.metadata.categories

    @pytest.mark.asyncio
    async def test_search_by_relevance_with_intent(self, sample_messages):
        """Test search with intent filter."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                intent="ask_question",
                limit=10,
            )
            for msg in messages:
                assert msg.metadata.intent == "ask_question"

    @pytest.mark.asyncio
    async def test_search_by_relevance_with_min_importance(self, sample_messages):
        """Test search with minimum importance."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                min_importance=0.7,
                limit=10,
            )
            for msg in messages:
                assert msg.metadata.importance >= 0.7

    @pytest.mark.asyncio
    async def test_search_by_relevance_with_session(self, sample_messages):
        """Test search with session filter."""
        async with AsyncSQLiteManager(":memory:") as db:
            for msg in sample_messages:
                await db.insert_message(msg)

            messages = await db.search_by_relevance(
                user_id="user_123",
                session_id="session_789",
                limit=10,
            )
            for msg in messages:
                assert msg.session_id == "session_789"


class TestAsyncDatabaseManager:
    """Tests for AsyncDatabaseManager (PostgreSQL) class."""

    @pytest.fixture
    def db_config(self):
        """Sample database configuration."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "mindcore_test",
            "user": "postgres",
            "password": "postgres",
            "min_connections": 1,
            "max_connections": 5,
        }

    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Hello, this is a test message",
            metadata=MessageMetadata(
                topics=["general", "testing"],
                categories=["question"],
                intent="ask_question",
                importance=0.7,
            ),
            created_at=datetime.now(timezone.utc),
        )

    # =====================
    # Initialization Tests
    # =====================

    def test_initialization(self, db_config):
        """Test AsyncDatabaseManager initializes correctly."""
        manager = AsyncDatabaseManager(db_config)
        assert manager.config == db_config
        assert manager._pool is None

    def test_ensure_pool_raises_when_not_connected(self, db_config):
        """Test _ensure_pool raises RuntimeError when not connected."""
        manager = AsyncDatabaseManager(db_config)

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            manager._ensure_pool()

    # =====================
    # Connection Tests (Mocked)
    # =====================

    @pytest.mark.asyncio
    async def test_connect_import_error(self, db_config):
        """Test connect raises ImportError if asyncpg not installed."""
        manager = AsyncDatabaseManager(db_config)

        with patch.dict("sys.modules", {"asyncpg": None}):
            with patch("builtins.__import__", side_effect=ImportError("No asyncpg")):
                with pytest.raises(ImportError, match="asyncpg is required"):
                    await manager.connect()

    @pytest.mark.asyncio
    async def test_context_manager(self, db_config):
        """Test async context manager pattern (mocked)."""
        manager = AsyncDatabaseManager(db_config)

        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock())
        mock_pool.close = AsyncMock()

        with patch("mindcore.core.async_db.AsyncDatabaseManager.connect") as mock_connect:
            with patch.object(manager, "_pool", mock_pool):
                mock_connect.return_value = None
                manager._pool = mock_pool

                async with manager as db:
                    assert db._pool is not None

    @pytest.mark.asyncio
    async def test_close(self, db_config):
        """Test close method (mocked)."""
        manager = AsyncDatabaseManager(db_config)
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        manager._pool = mock_pool

        await manager.close()
        mock_pool.close.assert_called_once()
        assert manager._pool is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self, db_config):
        """Test close when not connected does nothing."""
        manager = AsyncDatabaseManager(db_config)
        await manager.close()  # Should not raise

    # =====================
    # Message Operations (Mocked)
    # =====================

    @pytest.mark.asyncio
    async def test_insert_message_mocked(self, db_config, sample_message):
        """Test insert_message with mocked pool."""
        manager = AsyncDatabaseManager(db_config)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        result = await manager.insert_message(sample_message)
        assert result is True

    @pytest.mark.asyncio
    async def test_insert_message_exception(self, db_config, sample_message):
        """Test insert_message handles exceptions."""
        manager = AsyncDatabaseManager(db_config)

        # Create a context manager that raises on execute
        async def raise_on_execute(*args):
            raise Exception("DB Error")

        mock_conn = AsyncMock()
        mock_conn.execute = raise_on_execute

        class MockAcquire:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=MockAcquire())

        manager._pool = mock_pool

        result = await manager.insert_message(sample_message)
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_recent_messages_mocked(self, db_config):
        """Test fetch_recent_messages with mocked pool."""
        manager = AsyncDatabaseManager(db_config)

        mock_rows = [
            {
                "message_id": "msg_001",
                "user_id": "user_123",
                "thread_id": "thread_456",
                "session_id": "session_789",
                "role": "user",
                "raw_text": "Test message",
                "metadata": '{"topics": ["test"]}',
                "created_at": datetime.now(timezone.utc),
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        messages = await manager.fetch_recent_messages("user_123", "thread_456", limit=10)
        assert len(messages) == 1
        assert messages[0].message_id == "msg_001"

    @pytest.mark.asyncio
    async def test_fetch_recent_messages_with_dict_metadata(self, db_config):
        """Test fetch handles dict metadata (not JSON string)."""
        manager = AsyncDatabaseManager(db_config)

        mock_rows = [
            {
                "message_id": "msg_001",
                "user_id": "user_123",
                "thread_id": "thread_456",
                "session_id": "session_789",
                "role": "user",
                "raw_text": "Test message",
                "metadata": {"topics": ["test"]},  # Dict, not string
                "created_at": datetime.now(timezone.utc),
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        messages = await manager.fetch_recent_messages("user_123", "thread_456")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_get_message_by_id_mocked(self, db_config):
        """Test get_message_by_id with mocked pool."""
        manager = AsyncDatabaseManager(db_config)

        mock_row = {
            "message_id": "msg_001",
            "user_id": "user_123",
            "thread_id": "thread_456",
            "session_id": "session_789",
            "role": "user",
            "raw_text": "Test message",
            "metadata": '{"topics": ["test"]}',
            "created_at": datetime.now(timezone.utc),
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        msg = await manager.get_message_by_id("msg_001")
        assert msg is not None
        assert msg.message_id == "msg_001"

    @pytest.mark.asyncio
    async def test_get_message_by_id_not_found(self, db_config):
        """Test get_message_by_id returns None when not found."""
        manager = AsyncDatabaseManager(db_config)

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        msg = await manager.get_message_by_id("nonexistent")
        assert msg is None

    @pytest.mark.asyncio
    async def test_update_message_metadata_mocked(self, db_config):
        """Test update_message_metadata with mocked pool."""
        manager = AsyncDatabaseManager(db_config)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        result = await manager.update_message_metadata(
            "msg_001",
            MessageMetadata(topics=["updated"]),
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_message_metadata_not_found(self, db_config):
        """Test update_message_metadata when message not found."""
        manager = AsyncDatabaseManager(db_config)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        result = await manager.update_message_metadata(
            "nonexistent",
            MessageMetadata(topics=["test"]),
        )
        assert result is False

    # =====================
    # Search Tests (Mocked)
    # =====================

    @pytest.mark.asyncio
    async def test_search_by_relevance_mocked(self, db_config):
        """Test search_by_relevance with mocked pool."""
        manager = AsyncDatabaseManager(db_config)

        mock_rows = [
            {
                "message_id": "msg_001",
                "user_id": "user_123",
                "thread_id": "thread_456",
                "session_id": "session_789",
                "role": "user",
                "raw_text": "Test message",
                "metadata": '{"topics": ["test"], "importance": 0.8}',
                "created_at": datetime.now(timezone.utc),
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        messages = await manager.search_by_relevance(
            user_id="user_123",
            topics=["test"],
            limit=10,
        )
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_search_by_relevance_all_filters(self, db_config):
        """Test search_by_relevance with all filter options."""
        manager = AsyncDatabaseManager(db_config)

        mock_rows = []

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        messages = await manager.search_by_relevance(
            user_id="user_123",
            topics=["test"],
            categories=["question"],
            intent="ask_question",
            min_importance=0.5,
            thread_id="thread_456",
            session_id="session_789",
            limit=10,
        )
        assert messages == []

    @pytest.mark.asyncio
    async def test_search_by_relevance_exception(self, db_config):
        """Test search_by_relevance handles exceptions."""
        manager = AsyncDatabaseManager(db_config)

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=Exception("DB Error"))

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        manager._pool = mock_pool

        messages = await manager.search_by_relevance(user_id="user_123")
        assert messages == []
