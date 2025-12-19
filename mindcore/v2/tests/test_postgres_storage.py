"""Tests for PostgreSQL storage backend.

These tests require a running PostgreSQL database.
Set MINDCORE_TEST_POSTGRES_URL environment variable to run these tests.

Example:
    export MINDCORE_TEST_POSTGRES_URL="postgresql://user:pass@localhost/mindcore_test"
    pytest mindcore/v2/tests/test_postgres_storage.py -v
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

# Check if psycopg is available
try:
    import psycopg
    from psycopg_pool import ConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

from mindcore.v2.flr import Memory


# Get PostgreSQL connection string from environment
POSTGRES_URL = os.environ.get("MINDCORE_TEST_POSTGRES_URL")

# Skip all tests if PostgreSQL is not configured
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL or not PSYCOPG_AVAILABLE,
    reason="PostgreSQL not configured. Set MINDCORE_TEST_POSTGRES_URL environment variable.",
)


@pytest.fixture(scope="module")
def postgres_storage():
    """Create PostgreSQL storage for tests."""
    from mindcore.v2.storage import PostgresStorage

    storage = PostgresStorage(POSTGRES_URL)

    # Clean up any existing test data
    with storage._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE user_id LIKE 'test_%'")
            cur.execute("DELETE FROM transfers WHERE transfer_id LIKE 'test_%'")
            conn.commit()

    yield storage

    # Clean up after tests
    with storage._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE user_id LIKE 'test_%'")
            cur.execute("DELETE FROM transfers WHERE transfer_id LIKE 'test_%'")
            conn.commit()

    storage.close()


class TestPostgresStorageBasicOps:
    """Test basic CRUD operations."""

    def test_store_memory(self, postgres_storage):
        """Test storing a memory."""
        memory = Memory(
            memory_id="",
            content="PostgreSQL test memory content",
            memory_type="episodic",
            user_id="test_user_1",
        )

        memory_id = postgres_storage.store(memory)

        assert memory_id is not None
        assert memory_id.startswith("mem_")

    def test_store_with_id(self, postgres_storage):
        """Test storing memory with predefined ID."""
        memory = Memory(
            memory_id="test_custom_id",
            content="Custom ID test",
            memory_type="semantic",
            user_id="test_user_1",
        )

        memory_id = postgres_storage.store(memory)
        assert memory_id == "test_custom_id"

    def test_store_with_all_fields(self, postgres_storage):
        """Test storing memory with all fields."""
        now = datetime.now(timezone.utc)
        memory = Memory(
            memory_id="test_mem_full",
            content="Full memory with all fields",
            memory_type="preference",
            user_id="test_user_1",
            agent_id="test_agent",
            topics=["topic1", "topic2"],
            categories=["cat1"],
            sentiment="positive",
            importance=0.9,
            entities=["entity1"],
            access_level="team",
            created_at=now,
            vocabulary_version="2.0.0",
        )

        memory_id = postgres_storage.store(memory)
        retrieved = postgres_storage.get(memory_id)

        assert retrieved is not None
        assert retrieved.topics == ["topic1", "topic2"]
        assert retrieved.sentiment == "positive"
        assert retrieved.importance == 0.9
        assert retrieved.vocabulary_version == "2.0.0"

    def test_get_memory(self, postgres_storage):
        """Test retrieving a memory."""
        memory = Memory(
            memory_id="test_mem_get",
            content="Get test memory",
            memory_type="episodic",
            user_id="test_user_1",
        )
        postgres_storage.store(memory)

        retrieved = postgres_storage.get("test_mem_get")

        assert retrieved is not None
        assert retrieved.content == "Get test memory"
        assert retrieved.memory_type == "episodic"

    def test_get_nonexistent(self, postgres_storage):
        """Test getting non-existent memory."""
        result = postgres_storage.get("nonexistent_pg_id")
        assert result is None

    def test_update_memory(self, postgres_storage):
        """Test updating a memory."""
        memory = Memory(
            memory_id="test_mem_update",
            content="Original PostgreSQL content",
            memory_type="episodic",
            user_id="test_user_1",
        )
        postgres_storage.store(memory)

        # Update
        memory.content = "Updated PostgreSQL content"
        memory.importance = 0.8
        result = postgres_storage.update(memory)

        assert result is True

        retrieved = postgres_storage.get("test_mem_update")
        assert retrieved.content == "Updated PostgreSQL content"
        assert retrieved.importance == 0.8

    def test_update_nonexistent(self, postgres_storage):
        """Test updating non-existent memory."""
        memory = Memory(
            memory_id="nonexistent_pg",
            content="Test",
            memory_type="episodic",
            user_id="test_user_1",
        )

        result = postgres_storage.update(memory)
        assert result is False

    def test_delete_memory(self, postgres_storage):
        """Test deleting a memory."""
        memory = Memory(
            memory_id="test_mem_delete",
            content="To be deleted",
            memory_type="episodic",
            user_id="test_user_1",
        )
        postgres_storage.store(memory)

        result = postgres_storage.delete("test_mem_delete")
        assert result is True

        retrieved = postgres_storage.get("test_mem_delete")
        assert retrieved is None

    def test_delete_nonexistent(self, postgres_storage):
        """Test deleting non-existent memory."""
        result = postgres_storage.delete("nonexistent_pg_delete")
        assert result is False


class TestPostgresStorageSearch:
    """Test search functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, postgres_storage):
        """Set up test data for search tests."""
        memories = [
            Memory(
                memory_id="test_search_1",
                content="User prefers email communication",
                memory_type="preference",
                user_id="test_search_user",
                topics=["communication"],
                categories=["settings"],
                importance=0.8,
            ),
            Memory(
                memory_id="test_search_2",
                content="Customer had billing issue",
                memory_type="episodic",
                user_id="test_search_user",
                topics=["billing"],
                categories=["support"],
                importance=0.7,
            ),
            Memory(
                memory_id="test_search_3",
                content="Technical support request",
                memory_type="episodic",
                user_id="test_search_user",
                agent_id="test_agent",
                topics=["support"],
                access_level="team",
            ),
        ]

        for m in memories:
            postgres_storage.store(m)

        yield

        # Cleanup
        for m in memories:
            postgres_storage.delete(m.memory_id)

    def test_search_by_user(self, postgres_storage):
        """Test searching by user ID."""
        results = postgres_storage.search(user_id="test_search_user")

        assert len(results) >= 3
        for m in results:
            assert m.user_id == "test_search_user"

    def test_search_by_query(self, postgres_storage):
        """Test full-text search."""
        results = postgres_storage.search(
            query="billing",
            user_id="test_search_user",
        )

        assert len(results) >= 1
        assert any("billing" in m.content.lower() for m in results)

    def test_search_by_topics(self, postgres_storage):
        """Test searching by topics using JSONB."""
        results = postgres_storage.search(
            user_id="test_search_user",
            topics=["billing"],
        )

        assert len(results) >= 1
        for m in results:
            assert "billing" in m.topics

    def test_search_by_memory_type(self, postgres_storage):
        """Test searching by memory type."""
        results = postgres_storage.search(
            user_id="test_search_user",
            memory_types=["preference"],
        )

        assert len(results) >= 1
        for m in results:
            assert m.memory_type == "preference"

    def test_search_by_importance(self, postgres_storage):
        """Test searching by minimum importance."""
        results = postgres_storage.search(
            user_id="test_search_user",
            min_importance=0.75,
        )

        for m in results:
            assert m.importance >= 0.75

    def test_search_with_limit(self, postgres_storage):
        """Test search with limit."""
        results = postgres_storage.search(
            user_id="test_search_user",
            limit=2,
        )
        assert len(results) <= 2


class TestPostgresStorageReinforcement:
    """Test reinforcement score updates."""

    def test_update_reinforcement(self, postgres_storage):
        """Test positive reinforcement update."""
        memory = Memory(
            memory_id="test_reinforce_pg",
            content="Reinforcement test",
            memory_type="episodic",
            user_id="test_user_reinforce",
            reinforcement_score=0.0,
        )
        postgres_storage.store(memory)

        result = postgres_storage.update_reinforcement("test_reinforce_pg", 0.5)
        assert result is True

        retrieved = postgres_storage.get("test_reinforce_pg")
        assert abs(retrieved.reinforcement_score - 0.5) < 0.01

        # Cleanup
        postgres_storage.delete("test_reinforce_pg")

    def test_update_reinforcement_accumulates(self, postgres_storage):
        """Test that reinforcement accumulates."""
        memory = Memory(
            memory_id="test_reinforce_acc",
            content="Accumulation test",
            memory_type="episodic",
            user_id="test_user_reinforce",
        )
        postgres_storage.store(memory)

        postgres_storage.update_reinforcement("test_reinforce_acc", 0.3)
        postgres_storage.update_reinforcement("test_reinforce_acc", 0.4)

        retrieved = postgres_storage.get("test_reinforce_acc")
        assert abs(retrieved.reinforcement_score - 0.7) < 0.01

        # Cleanup
        postgres_storage.delete("test_reinforce_acc")


class TestPostgresStorageTransfers:
    """Test transfer data storage."""

    def test_store_and_get_transfer(self, postgres_storage):
        """Test storing and retrieving transfer data."""
        data = [
            {"memory_id": "mem_1", "content": "Test 1"},
            {"memory_id": "mem_2", "content": "Test 2"},
        ]

        postgres_storage.store_transfer("test_transfer_1", data)

        retrieved = postgres_storage.get_transfer("test_transfer_1")

        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0]["memory_id"] == "mem_1"

        # Cleanup
        with postgres_storage._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM transfers WHERE transfer_id = %s", ("test_transfer_1",))
                conn.commit()

    def test_get_transfer_nonexistent(self, postgres_storage):
        """Test getting non-existent transfer."""
        result = postgres_storage.get_transfer("nonexistent_transfer")
        assert result is None


class TestPostgresStorageStats:
    """Test storage statistics."""

    def test_get_stats(self, postgres_storage):
        """Test getting storage stats."""
        # Add some test data
        memory = Memory(
            memory_id="test_stats_mem",
            content="Stats test",
            memory_type="episodic",
            user_id="test_stats_user",
            agent_id="test_stats_agent",
        )
        postgres_storage.store(memory)

        stats = postgres_storage.get_stats()

        assert "total_memories" in stats
        assert "by_memory_type" in stats
        assert "unique_users" in stats
        assert "unique_agents" in stats
        assert "database_size" in stats
        assert "pool_size" in stats

        # Cleanup
        postgres_storage.delete("test_stats_mem")


class TestPostgresStorageVersion:
    """Test vocabulary version operations."""

    def test_search_by_version(self, postgres_storage):
        """Test searching by vocabulary version."""
        v1 = Memory(
            memory_id="test_v1_mem",
            content="Version 1 memory",
            memory_type="episodic",
            user_id="test_version_user",
            vocabulary_version="1.0.0",
        )
        v2 = Memory(
            memory_id="test_v2_mem",
            content="Version 2 memory",
            memory_type="episodic",
            user_id="test_version_user",
            vocabulary_version="2.0.0",
        )

        postgres_storage.store(v1)
        postgres_storage.store(v2)

        results = postgres_storage.search_by_version(
            "1.0.0",
            user_id="test_version_user",
        )

        assert len(results) >= 1
        for m in results:
            assert m.vocabulary_version == "1.0.0"

        # Cleanup
        postgres_storage.delete("test_v1_mem")
        postgres_storage.delete("test_v2_mem")


class TestPostgresStorageExpiration:
    """Test memory expiration."""

    def test_expired_memory_not_returned(self, postgres_storage):
        """Test that expired memories are filtered out."""
        now = datetime.now(timezone.utc)

        # Expired memory
        expired = Memory(
            memory_id="test_expired_pg",
            content="Expired",
            memory_type="episodic",
            user_id="test_expiry_user",
            expires_at=now - timedelta(hours=1),
        )

        # Valid memory
        valid = Memory(
            memory_id="test_valid_pg",
            content="Valid",
            memory_type="episodic",
            user_id="test_expiry_user",
            expires_at=now + timedelta(days=1),
        )

        postgres_storage.store(expired)
        postgres_storage.store(valid)

        results = postgres_storage.search(user_id="test_expiry_user")

        # Should only return the valid memory
        memory_ids = [m.memory_id for m in results]
        assert "test_valid_pg" in memory_ids
        assert "test_expired_pg" not in memory_ids

        # Cleanup
        postgres_storage.delete("test_expired_pg")
        postgres_storage.delete("test_valid_pg")
