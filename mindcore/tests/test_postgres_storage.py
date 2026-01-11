"""Tests for PostgreSQL Storage backend.

Tests cover:
- Connection pool management
- CRUD operations (store, get, update, delete)
- Search with filters
- Session aggregates
- Statistics

Uses mocking since actual PostgreSQL is not available in test environment.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from mindcore.clst.aggregates import SessionAggregate
from mindcore.exceptions import MemoryNotFoundError, StorageError
from mindcore.flr import Memory
from mindcore.storage.postgres import PostgresStorage


# =============================================================================
# Mock PostgreSQL Module
# =============================================================================


class MockCursor:
    """Mock psycopg cursor."""

    def __init__(self):
        self.description = [
            ("memory_id",),
            ("content",),
            ("memory_type",),
            ("user_id",),
            ("agent_id",),
            ("topics",),
            ("categories",),
            ("sentiment",),
            ("importance",),
            ("entities",),
            ("access_level",),
            ("session_id",),
            ("message_index",),
            ("created_at",),
            ("last_accessed",),
            ("expires_at",),
            ("reinforcement_score",),
            ("access_count",),
            ("vocabulary_version",),
            ("embedding",),
            ("search_vector",),
        ]
        self._results = []
        self._result_index = 0
        self.rowcount = 1
        self.lastrowid = None
        self._executed = []

    def execute(self, sql, params=None):
        self._executed.append((sql, params))
        return self

    def fetchone(self):
        if self._results and self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return None

    def fetchall(self):
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnection:
    """Mock psycopg connection."""

    def __init__(self):
        self._cursor = MockCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def execute(self, sql):
        pass

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnectionPool:
    """Mock psycopg_pool ConnectionPool."""

    def __init__(self, *args, **kwargs):
        self._connection = MockConnection()
        self._stats = {
            "pool_size": 10,
            "pool_available": 8,
            "requests_waiting": 0,
        }

    def connection(self):
        return self

    def wait(self):
        pass

    def close(self):
        pass

    def get_stats(self):
        return self._stats

    def __enter__(self):
        return self._connection

    def __exit__(self, *args):
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_psycopg():
    """Mock psycopg and psycopg_pool modules."""
    mock_pool_module = MagicMock()
    mock_pool_module.ConnectionPool = MockConnectionPool

    mock_psycopg_module = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "psycopg": mock_psycopg_module,
            "psycopg_pool": mock_pool_module,
        },
    ):
        yield mock_pool_module


@pytest.fixture
def storage(mock_psycopg):
    """Create mocked PostgresStorage instance."""
    with patch.object(PostgresStorage, "_initialize_schema"):
        storage = PostgresStorage(
            connection_string="postgresql://user:pass@localhost/test",
            pool_size=5,
            max_overflow=10,
            connection_timeout=30.0,
        )
        # Replace pool with our mock
        storage._pool = MockConnectionPool()
        yield storage


@pytest.fixture
def test_memory():
    """Create a test memory."""
    return Memory(
        memory_id="test_mem_1",
        content="User prefers dark mode",
        memory_type="preference",
        user_id="user_123",
        agent_id="agent_1",
        topics=["settings", "ui"],
        categories=["user_preference"],
        entities=["dark mode"],
        importance=0.8,
        created_at=datetime.now(timezone.utc),
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPostgresStorageInit:
    """Tests for PostgresStorage initialization."""

    def test_init_stores_pool_config(self, storage):
        """Test initialization stores pool configuration."""
        assert storage._pool_size == 5
        assert storage._max_overflow == 10
        assert storage._connection_timeout == 30.0

    def test_default_pool_size(self, storage):
        """Test default pool size is set from fixture."""
        assert storage._pool_size == 5  # Set in storage fixture


# =============================================================================
# Store Tests
# =============================================================================


class TestStore:
    """Tests for store method."""

    def test_store_memory(self, storage, test_memory):
        """Test storing a memory."""
        result = storage.store(test_memory)

        assert result == test_memory.memory_id

    def test_store_generates_id(self, storage):
        """Test store generates ID if not provided."""
        memory = Memory(
            memory_id="",  # Empty ID
            content="Test content",
            memory_type="fact",
            user_id="user_1",
        )

        result = storage.store(memory)

        assert result.startswith("mem_")
        assert memory.memory_id == result

    def test_store_sets_created_at(self, storage):
        """Test store sets created_at if not provided."""
        memory = Memory(
            memory_id="test_1",
            content="Test",
            memory_type="fact",
            user_id="user_1",
        )
        memory.created_at = None

        storage.store(memory)

        assert memory.created_at is not None


# =============================================================================
# Get Tests
# =============================================================================


class TestGet:
    """Tests for get method."""

    def test_get_memory_found(self, storage, test_memory):
        """Test retrieving existing memory."""
        # Setup mock to return data
        now = datetime.now(timezone.utc)
        storage._pool._connection._cursor._results = [
            (
                "test_mem_1",
                "User prefers dark mode",
                "preference",
                "user_123",
                "agent_1",
                ["settings", "ui"],
                ["user_preference"],
                "neutral",
                0.8,
                ["dark mode"],
                "private",
                None,
                0,
                now,
                None,
                None,
                0.0,
                0,
                "1.0.0",
                None,
                None,
            )
        ]

        result = storage.get("test_mem_1")

        assert result is not None
        assert result.memory_id == "test_mem_1"
        assert result.content == "User prefers dark mode"

    def test_get_memory_not_found(self, storage):
        """Test retrieving nonexistent memory returns None."""
        storage._pool._connection._cursor._results = []

        result = storage.get("nonexistent")

        assert result is None


# =============================================================================
# Update Tests
# =============================================================================


class TestUpdate:
    """Tests for update method."""

    def test_update_memory(self, storage, test_memory):
        """Test updating a memory."""
        storage._pool._connection._cursor.rowcount = 1

        # Should not raise
        storage.update(test_memory)

    def test_update_not_found_raises(self, storage, test_memory):
        """Test updating nonexistent memory raises error."""
        storage._pool._connection._cursor.rowcount = 0

        with pytest.raises(MemoryNotFoundError):
            storage.update(test_memory)


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    """Tests for delete method."""

    def test_delete_memory(self, storage):
        """Test deleting a memory."""
        storage._pool._connection._cursor.rowcount = 1

        # Should not raise
        storage.delete("test_mem_1")

    def test_delete_not_found_raises(self, storage):
        """Test deleting nonexistent memory raises error."""
        storage._pool._connection._cursor.rowcount = 0

        with pytest.raises(MemoryNotFoundError):
            storage.delete("nonexistent")


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search method."""

    def test_search_basic(self, storage):
        """Test basic search."""
        now = datetime.now(timezone.utc)
        storage._pool._connection._cursor._results = [
            (
                "mem_1",
                "Content 1",
                "fact",
                "user_1",
                None,
                [],
                [],
                "neutral",
                0.5,
                [],
                "private",
                None,
                0,
                now,
                None,
                None,
                0.0,
                0,
                "1.0.0",
                None,
                None,
            ),
        ]

        results = storage.search(user_id="user_1")

        assert len(results) == 1
        assert results[0].memory_id == "mem_1"

    def test_search_with_query(self, storage):
        """Test search with full-text query."""
        storage._pool._connection._cursor._results = []

        storage.search(query="dark mode", user_id="user_1")

        # Verify query was included in SQL
        executed = storage._pool._connection._cursor._executed
        assert any("search_vector" in str(e[0]) for e in executed)

    def test_search_with_topics_filter(self, storage):
        """Test search with topics filter."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            topics=["settings", "ui"],
        )

        # Verify topics filter was included
        executed = storage._pool._connection._cursor._executed
        assert any("topics" in str(e[0]) for e in executed)

    def test_search_with_categories_filter(self, storage):
        """Test search with categories filter."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            categories=["user_preference"],
        )

        executed = storage._pool._connection._cursor._executed
        assert any("categories" in str(e[0]) for e in executed)

    def test_search_with_memory_types(self, storage):
        """Test search with memory type filter."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            memory_types=["preference", "fact"],
        )

        executed = storage._pool._connection._cursor._executed
        assert any("memory_type" in str(e[0]) for e in executed)

    def test_search_with_date_range(self, storage):
        """Test search with date range filter."""
        now = datetime.now(timezone.utc)
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            start_date=now,
            end_date=now,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("created_at >=" in str(e[0]) for e in executed)

    def test_search_with_min_importance(self, storage):
        """Test search with minimum importance filter."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            min_importance=0.7,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("importance >=" in str(e[0]) for e in executed)

    def test_search_with_access_levels(self, storage):
        """Test search with access level filter."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            access_levels=["private", "team"],
        )

        executed = storage._pool._connection._cursor._executed
        assert any("access_level" in str(e[0]) for e in executed)

    def test_search_with_pagination(self, storage):
        """Test search with limit and offset."""
        storage._pool._connection._cursor._results = []

        storage.search(
            user_id="user_1",
            limit=50,
            offset=10,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("LIMIT" in str(e[0]) and "OFFSET" in str(e[0]) for e in executed)


# =============================================================================
# Search by Version Tests
# =============================================================================


class TestSearchByVersion:
    """Tests for search_by_version method."""

    def test_search_by_version(self, storage):
        """Test searching by vocabulary version."""
        storage._pool._connection._cursor._results = []

        storage.search_by_version(
            version="2.0.0",
            user_id="user_1",
        )

        executed = storage._pool._connection._cursor._executed
        assert any("vocabulary_version" in str(e[0]) for e in executed)


# =============================================================================
# Reinforcement Tests
# =============================================================================


class TestUpdateReinforcement:
    """Tests for update_reinforcement method."""

    def test_update_reinforcement(self, storage):
        """Test updating reinforcement score."""
        storage._pool._connection._cursor.rowcount = 1

        storage.update_reinforcement("test_mem_1", 0.5)

        executed = storage._pool._connection._cursor._executed
        assert any("reinforcement_score" in str(e[0]) for e in executed)

    def test_update_reinforcement_not_found(self, storage):
        """Test updating reinforcement for nonexistent memory."""
        storage._pool._connection._cursor.rowcount = 0

        with pytest.raises(MemoryNotFoundError):
            storage.update_reinforcement("nonexistent", 0.5)

    def test_update_reinforcement_invalid_type(self, storage):
        """Test updating reinforcement with invalid signal type."""
        with pytest.raises(TypeError):
            storage.update_reinforcement("test_mem_1", "invalid")


# =============================================================================
# Transfer Tests
# =============================================================================


class TestTransfers:
    """Tests for transfer data operations."""

    def test_store_transfer(self, storage):
        """Test storing transfer data."""
        data = [{"memory_id": "mem_1", "content": "Test"}]

        storage.store_transfer("transfer_123", data)

        executed = storage._pool._connection._cursor._executed
        assert any("INSERT INTO transfers" in str(e[0]) for e in executed)

    def test_get_transfer_found(self, storage):
        """Test retrieving transfer data."""
        storage._pool._connection._cursor._results = [
            ([{"memory_id": "mem_1", "content": "Test"}],),
        ]

        result = storage.get_transfer("transfer_123")

        assert result is not None
        assert result[0]["memory_id"] == "mem_1"

    def test_get_transfer_not_found(self, storage):
        """Test retrieving nonexistent transfer."""
        storage._pool._connection._cursor._results = []

        result = storage.get_transfer("nonexistent")

        assert result is None


# =============================================================================
# Stats Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self, storage):
        """Test getting storage statistics."""
        # The get_stats method makes multiple queries:
        # 1. SELECT COUNT(*) - total (fetchone)
        # 2. SELECT memory_type, COUNT(*) - by_type (fetchall)
        # 3. SELECT COUNT(DISTINCT user_id) - unique_users (fetchone)
        # 4. SELECT COUNT(DISTINCT agent_id) - unique_agents (fetchone)
        # 5. SELECT pg_size_pretty - db_size (fetchone)
        fetchone_results = [
            (100,),  # total memories
            (10,),  # unique users
            (5,),  # unique agents
            ("10 MB",),  # db size
        ]
        fetchone_idx = [0]

        def mock_fetchone():
            if fetchone_idx[0] < len(fetchone_results):
                result = fetchone_results[fetchone_idx[0]]
                fetchone_idx[0] += 1
                return result
            return (0,)

        def mock_fetchall():
            return [("fact", 50), ("preference", 30)]

        storage._pool._connection._cursor.fetchone = mock_fetchone
        storage._pool._connection._cursor.fetchall = mock_fetchall

        stats = storage.get_stats()

        assert "total_memories" in stats
        assert "connection_pool" in stats
        assert stats["connection_pool"]["pool_size"] == 5


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    """Tests for close method."""

    def test_close(self, storage):
        """Test closing storage releases resources."""
        storage.close()

        # Pool close should be called
        # (Mock doesn't track this, but method shouldn't raise)


# =============================================================================
# Row Conversion Tests
# =============================================================================


class TestRowConversion:
    """Tests for row to object conversion."""

    def test_row_to_memory(self, storage):
        """Test converting database row to Memory."""
        now = datetime.now(timezone.utc)
        row = (
            "mem_1",
            "Test content",
            "fact",
            "user_1",
            "agent_1",
            ["topic1"],
            ["cat1"],
            "neutral",
            0.5,
            ["entity1"],
            "private",
            "session_1",
            0,
            now,
            None,
            None,
            0.0,
            0,
            "1.0.0",
            None,
            None,
        )
        description = [
            ("memory_id",),
            ("content",),
            ("memory_type",),
            ("user_id",),
            ("agent_id",),
            ("topics",),
            ("categories",),
            ("sentiment",),
            ("importance",),
            ("entities",),
            ("access_level",),
            ("session_id",),
            ("message_index",),
            ("created_at",),
            ("last_accessed",),
            ("expires_at",),
            ("reinforcement_score",),
            ("access_count",),
            ("vocabulary_version",),
            ("embedding",),
            ("search_vector",),
        ]

        memory = storage._row_to_memory(row, description)

        assert memory.memory_id == "mem_1"
        assert memory.content == "Test content"
        assert memory.topics == ["topic1"]

    def test_row_to_session_aggregate(self, storage):
        """Test converting row dict to SessionAggregate."""
        data = {
            "session_id": "session_1",
            "user_id": "user_1",
            "agent_id": "agent_1",
            "topic_weights": {"orders": 0.8},
            "category_weights": {"support": 0.9},
            "entity_weights": {},
            "intent_weights": {},
            "sentiment_weights": {"neutral": 1.0},
            "importance_min": 0.3,
            "importance_max": 0.9,
            "importance_avg": 0.6,
            "importance_sum": 1.8,
            "confidence_min": 0.5,
            "confidence_max": 0.95,
            "confidence_avg": 0.75,
            "confidence_sum": 2.25,
            "memory_count": 3,
            "message_count": 3,
            "started_at": datetime.now(timezone.utc),
            "last_activity_at": datetime.now(timezone.utc),
            "dominant_topic": "orders",
            "dominant_category": "support",
            "dominant_sentiment": "neutral",
            "max_urgency": "medium",
            "access_level": "private",
            "summary_text": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        aggregate = storage._row_to_session_aggregate(data)

        assert aggregate.session_id == "session_1"
        assert aggregate.topic_weights == {"orders": 0.8}
        assert aggregate.memory_count == 3


# =============================================================================
# Session Aggregate Tests
# =============================================================================


class TestSessionAggregates:
    """Tests for session aggregate operations."""

    def test_store_session_aggregate(self, storage):
        """Test storing a session aggregate."""
        aggregate = SessionAggregate(
            session_id="session_1",
            user_id="user_1",
            agent_id="agent_1",
        )

        result = storage.store_session_aggregate(aggregate)

        assert result == "session_1"

    def test_get_session_aggregate_found(self, storage):
        """Test retrieving existing session aggregate."""
        now = datetime.now(timezone.utc)
        storage._pool._connection._cursor._results = [
            (
                "session_1",
                "user_1",
                "agent_1",
                {},
                {},
                {},
                {},
                {},
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                now,
                now,
                None,
                None,
                None,
                None,
                "private",
                None,
                None,
                now,
                now,
            )
        ]
        storage._pool._connection._cursor.description = [
            ("session_id",),
            ("user_id",),
            ("agent_id",),
            ("topic_weights",),
            ("category_weights",),
            ("entity_weights",),
            ("intent_weights",),
            ("sentiment_weights",),
            ("importance_min",),
            ("importance_max",),
            ("importance_avg",),
            ("importance_sum",),
            ("confidence_min",),
            ("confidence_max",),
            ("confidence_avg",),
            ("confidence_sum",),
            ("memory_count",),
            ("message_count",),
            ("started_at",),
            ("last_activity_at",),
            ("dominant_topic",),
            ("dominant_category",),
            ("dominant_sentiment",),
            ("max_urgency",),
            ("access_level",),
            ("summary_text",),
            ("summary_embedding",),
            ("created_at",),
            ("updated_at",),
        ]

        result = storage.get_session_aggregate("session_1")

        assert result is not None
        assert result.session_id == "session_1"

    def test_get_session_aggregate_not_found(self, storage):
        """Test retrieving nonexistent session aggregate."""
        storage._pool._connection._cursor._results = []

        result = storage.get_session_aggregate("nonexistent")

        assert result is None

    def test_query_sessions(self, storage):
        """Test querying sessions."""
        storage._pool._connection._cursor._results = []
        storage._pool._connection._cursor.description = []

        storage.query_sessions(
            user_id="user_1",
            topic_hints=["orders"],
            min_importance_avg=0.5,
            limit=10,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("session_aggregates" in str(e[0]) for e in executed)

    def test_query_memories_by_sessions(self, storage):
        """Test querying memories by session IDs."""
        storage._pool._connection._cursor._results = []

        storage.query_memories_by_sessions(
            session_ids=["session_1", "session_2"],
            min_importance=0.5,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("session_id = ANY" in str(e[0]) for e in executed)

    def test_query_memories_by_sessions_empty(self, storage):
        """Test querying memories with empty session list."""
        results = storage.query_memories_by_sessions(session_ids=[])

        assert results == []

    def test_get_next_message_index(self, storage):
        """Test getting next message index for session."""
        storage._pool._connection._cursor._results = [(5,)]

        result = storage.get_next_message_index("session_1")

        assert result == 5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_store_with_embedding(self, storage):
        """Test storing memory with embedding."""
        memory = Memory(
            memory_id="mem_with_embedding",
            content="Test",
            memory_type="semantic",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        result = storage.store(memory)

        assert result == "mem_with_embedding"

    def test_store_with_session(self, storage):
        """Test storing memory with session ID."""
        storage._pool._connection._cursor._results = [(0,)]  # For message_index

        memory = Memory(
            memory_id="mem_with_session",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            session_id="session_1",
        )

        result = storage.store(memory)

        assert result == "mem_with_session"

    def test_search_filters_expired(self, storage):
        """Test search filters out expired memories."""
        storage._pool._connection._cursor._results = []

        storage.search(user_id="user_1")

        # Verify expiration check in query
        executed = storage._pool._connection._cursor._executed
        assert any("expires_at" in str(e[0]) for e in executed)


# =============================================================================
# Enhanced Search Tests (pg_trgm, SQL ranking)
# =============================================================================


class TestSearchRanked:
    """Tests for search_ranked() with SQL-based ranking."""

    @pytest.fixture
    def storage_with_extensions(self, storage):
        """Create storage with mocked extensions available."""
        # Set extension flags to simulate extensions being available
        storage._has_pg_trgm = True
        storage._has_pg_search = False
        storage._has_rank_memory = True
        storage._has_rank_session = True
        return storage

    @pytest.fixture
    def storage_without_extensions(self, storage):
        """Create storage without extensions."""
        # Set extension flags to simulate extensions not available
        storage._has_pg_trgm = False
        storage._has_pg_search = False
        storage._has_rank_memory = False
        storage._has_rank_session = False
        return storage

    def test_search_ranked_with_extensions(self, storage_with_extensions):
        """Test search_ranked uses SQL ranking when extensions available."""
        storage = storage_with_extensions
        storage._pool._connection._cursor._results = []

        results = storage.search_ranked(
            query="test query",
            user_id="user_1",
            attention_hints=["topic1", "topic2"],
            limit=10,
        )

        assert isinstance(results, list)
        # Should have executed SQL with rank_memory function
        executed = storage._pool._connection._cursor._executed
        assert any("rank_memory" in str(e[0]) for e in executed)

    def test_search_ranked_raises_without_pg_trgm(self, storage_without_extensions):
        """Test search_ranked raises StorageError without pg_trgm extension."""
        storage = storage_without_extensions

        with pytest.raises(StorageError, match="pg_trgm extension required"):
            storage.search_ranked(
                query="test query",
                user_id="user_1",
                attention_hints=["topic1"],
                limit=10,
            )

    def test_search_ranked_raises_without_rank_memory(self, storage_without_extensions):
        """Test search_ranked raises StorageError without rank_memory function."""
        storage = storage_without_extensions
        storage._has_pg_trgm = True  # Has extension but not function

        with pytest.raises(StorageError, match="rank_memory.*function required"):
            storage.search_ranked(
                query="test query",
                user_id="user_1",
                attention_hints=["topic1"],
                limit=10,
            )

    def test_search_ranked_with_attention_hints(self, storage_with_extensions):
        """Test search_ranked uses attention hints in ranking."""
        storage = storage_with_extensions
        storage._pool._connection._cursor._results = []

        storage.search_ranked(
            query="order status",
            user_id="user_1",
            attention_hints=["orders", "shipping"],
            limit=20,
        )

        executed = storage._pool._connection._cursor._executed
        # Verify attention hints appear in query
        assert any("orders" in str(e) and "shipping" in str(e) for e in executed)

    def test_search_ranked_with_min_similarity(self, storage_with_extensions):
        """Test search_ranked respects min_similarity threshold."""
        storage = storage_with_extensions
        storage._pool._connection._cursor._results = []

        storage.search_ranked(
            query="test",
            user_id="user_1",
            min_similarity=0.3,
            limit=10,
        )

        executed = storage._pool._connection._cursor._executed
        # Should have similarity threshold in query
        assert any("similarity" in str(e[0]) for e in executed)

    def test_search_capabilities_property(self, storage_with_extensions):
        """Test search_capabilities returns correct status."""
        storage = storage_with_extensions

        capabilities = storage.search_capabilities

        assert capabilities["trigram_search"] is True
        assert capabilities["sql_memory_ranking"] is True
        assert capabilities["bm25_search"] is False

    def test_search_capabilities_without_extensions(self, storage_without_extensions):
        """Test search_capabilities when extensions unavailable."""
        storage = storage_without_extensions

        capabilities = storage.search_capabilities

        assert capabilities["trigram_search"] is False
        assert capabilities["sql_memory_ranking"] is False
        assert capabilities["bm25_search"] is False

    def test_search_bm25_raises_without_pg_search(self, storage_without_extensions):
        """Test search_bm25 raises StorageError without pg_search extension."""
        storage = storage_without_extensions

        with pytest.raises(StorageError, match="pg_search extension required"):
            storage.search_bm25(
                query="test query",
                user_id="user_1",
                limit=10,
            )

    def test_search_bm25_raises_without_rank_memory(self, storage_without_extensions):
        """Test search_bm25 raises StorageError without rank_memory function."""
        storage = storage_without_extensions
        storage._has_pg_search = True  # Has extension but not function

        with pytest.raises(StorageError, match="rank_memory.*function required"):
            storage.search_bm25(
                query="test query",
                user_id="user_1",
                limit=10,
            )


class TestQuerySessionsRanked:
    """Tests for query_sessions_ranked() with SQL-based session ranking."""

    @pytest.fixture
    def storage_with_extensions(self, storage):
        """Create storage with mocked extensions available."""
        # Set extension flags to simulate extensions being available
        storage._has_pg_trgm = True
        storage._has_pg_search = False
        storage._has_rank_memory = True
        storage._has_rank_session = True
        return storage

    def test_query_sessions_ranked_with_extensions(self, storage_with_extensions):
        """Test query_sessions_ranked uses SQL ranking."""
        storage = storage_with_extensions
        storage._pool._connection._cursor._results = []

        results = storage.query_sessions_ranked(
            user_id="user_1",
            topic_hints=["orders", "support"],
            limit=5,
        )

        assert isinstance(results, list)
        executed = storage._pool._connection._cursor._executed
        assert any("rank_session" in str(e[0]) for e in executed)

    def test_query_sessions_ranked_with_category_hints(self, storage_with_extensions):
        """Test query_sessions_ranked uses category hints."""
        storage = storage_with_extensions
        storage._pool._connection._cursor._results = []

        storage.query_sessions_ranked(
            user_id="user_1",
            category_hints=["support", "billing"],
            limit=5,
        )

        executed = storage._pool._connection._cursor._executed
        assert any("support" in str(e) and "billing" in str(e) for e in executed)

    def test_query_sessions_ranked_raises_without_rank_session(self, storage_with_extensions):
        """Test query_sessions_ranked raises StorageError without rank_session function."""
        storage = storage_with_extensions
        storage._has_rank_session = False  # Disable the function

        with pytest.raises(StorageError, match="rank_session.*function required"):
            storage.query_sessions_ranked(
                user_id="user_1",
                topic_hints=["orders"],
                limit=5,
            )


class TestSearchConfig:
    """Tests for SearchConfig configuration."""

    def test_default_config(self):
        """Test default SearchConfig values."""
        from mindcore.storage.config import SearchConfig

        config = SearchConfig()

        assert config.use_trigram_search is True
        assert config.use_bm25_search is False
        assert config.trigram_similarity_threshold == 0.2
        assert "content" in config.ranking_weights
        assert "topic" in config.ranking_weights

    def test_config_validation_invalid_threshold(self):
        """Test SearchConfig validation rejects invalid threshold."""
        from mindcore.storage.config import SearchConfig

        with pytest.raises(ValueError, match="trigram_similarity_threshold"):
            SearchConfig(trigram_similarity_threshold=1.5)

    def test_config_validation_invalid_weights(self):
        """Test SearchConfig validation rejects invalid weights."""
        from mindcore.storage.config import SearchConfig

        with pytest.raises(ValueError, match="ranking_weights"):
            SearchConfig(
                ranking_weights={
                    "content": 0.5,
                    # Missing other required keys
                }
            )

    def test_config_to_sql_weights_json(self):
        """Test SearchConfig.to_sql_weights_json() produces valid JSON."""
        from mindcore.storage.config import SearchConfig

        config = SearchConfig()
        json_str = config.to_sql_weights_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["content"] == 0.15
        assert parsed["topic"] == 0.25

    def test_preset_configs(self):
        """Test preset configurations are valid."""
        from mindcore.storage.config import (
            SEARCH_CONFIG_DEFAULT,
            SEARCH_CONFIG_RECENCY_FOCUSED,
            SEARCH_CONFIG_TOPIC_FOCUSED,
        )

        # All should be valid (no exceptions during import)
        assert SEARCH_CONFIG_DEFAULT.use_trigram_search is True
        assert SEARCH_CONFIG_RECENCY_FOCUSED.ranking_weights["recency"] > 0.15
        assert SEARCH_CONFIG_TOPIC_FOCUSED.ranking_weights["topic"] > 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
