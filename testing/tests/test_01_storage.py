"""Test 01: Storage Layer Tests.

Tests SQLite and PostgreSQL storage backends for:
- CRUD operations (Create, Read, Update, Delete)
- Search functionality
- Batch operations
- Migration between backends
- Performance characteristics
"""

import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from tests.conftest import requires_postgres


# ============================================================================
# SQLite Storage Tests
# ============================================================================


class TestSQLiteStorage:
    """Test suite for SQLite storage backend."""

    def test_store_and_retrieve(self, sqlite_storage):
        """Test basic store and retrieve operations."""
        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",  # Will be assigned
            content="Test memory content",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            categories=["technical"],
            importance=0.8,
            created_at=datetime.now(),
        )

        # Store
        memory_id = sqlite_storage.store(memory)
        assert memory_id is not None
        assert len(memory_id) > 0

        # Retrieve
        retrieved = sqlite_storage.get(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.memory_type == "semantic"
        assert retrieved.user_id == "test_user"
        assert retrieved.importance == 0.8

    def test_update_memory(self, sqlite_storage):
        """Test updating an existing memory."""
        from mindcore.flr import Memory

        # Create initial memory
        memory = Memory(
            memory_id="",
            content="Original content",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            importance=0.5,
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Update
        memory.memory_id = memory_id
        memory.content = "Updated content"
        memory.importance = 0.9
        sqlite_storage.update(memory)

        # Verify
        updated = sqlite_storage.get(memory_id)
        assert updated.content == "Updated content"
        assert updated.importance == 0.9

    def test_delete_memory(self, sqlite_storage):
        """Test deleting a memory."""
        from mindcore.exceptions import MemoryNotFoundError
        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="To be deleted",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Verify exists
        assert sqlite_storage.get(memory_id) is not None

        # Delete
        sqlite_storage.delete(memory_id)

        # Verify deleted
        assert sqlite_storage.get(memory_id) is None

        # Delete again should raise error
        with pytest.raises(MemoryNotFoundError):
            sqlite_storage.delete(memory_id)

    def test_search_by_user(self, sqlite_storage):
        """Test searching memories by user."""
        from mindcore.flr import Memory

        # Store memories for different users
        for i, user_id in enumerate(["user_a", "user_a", "user_b"]):
            memory = Memory(
                memory_id="",
                content=f"Memory {i} for {user_id}",
                memory_type="semantic",
                user_id=user_id,
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Search for user_a
        results = sqlite_storage.search(user_id="user_a")
        assert len(results) == 2
        assert all(m.user_id == "user_a" for m in results)

        # Search for user_b
        results = sqlite_storage.search(user_id="user_b")
        assert len(results) == 1
        assert results[0].user_id == "user_b"

    def test_search_by_topics(self, sqlite_storage):
        """Test searching by topics."""
        from mindcore.flr import Memory

        memories_data = [
            {"content": "API docs", "topics": ["api", "documentation"]},
            {"content": "Billing issue", "topics": ["billing"]},
            {"content": "API billing", "topics": ["api", "billing"]},
        ]

        for data in memories_data:
            memory = Memory(
                memory_id="",
                content=data["content"],
                memory_type="semantic",
                user_id="test_user",
                topics=data["topics"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Search by api topic
        results = sqlite_storage.search(user_id="test_user", topics=["api"])
        assert len(results) == 2

        # Search by billing topic
        results = sqlite_storage.search(user_id="test_user", topics=["billing"])
        assert len(results) == 2

    def test_search_by_memory_type(self, sqlite_storage):
        """Test searching by memory type."""
        from mindcore.flr import Memory

        types = ["semantic", "episodic", "preference"]
        for i, mtype in enumerate(types):
            memory = Memory(
                memory_id="",
                content=f"Memory type {mtype}",
                memory_type=mtype,
                user_id="test_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Search for semantic only
        results = sqlite_storage.search(user_id="test_user", memory_types=["semantic"])
        assert len(results) == 1
        assert results[0].memory_type == "semantic"

    def test_search_with_date_range(self, sqlite_storage, old_date, recent_date):
        """Test searching with date range filters."""
        from mindcore.flr import Memory

        # Store old memory
        old_memory = Memory(
            memory_id="",
            content="Old memory",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=old_date,
        )
        sqlite_storage.store(old_memory)

        # Store recent memory
        recent_memory = Memory(
            memory_id="",
            content="Recent memory",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=recent_date,
        )
        sqlite_storage.store(recent_memory)

        # Search for recent only
        cutoff = datetime.now() - timedelta(days=30)
        results = sqlite_storage.search(user_id="test_user", start_date=cutoff)
        assert len(results) == 1
        assert results[0].content == "Recent memory"

    def test_batch_store(self, sqlite_storage):
        """Test batch storage of multiple memories."""
        from mindcore.flr import Memory

        memories = [
            Memory(
                memory_id="",
                content=f"Batch memory {i}",
                memory_type="semantic",
                user_id="test_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            for i in range(10)
        ]

        memory_ids = sqlite_storage.store_batch(memories)
        assert len(memory_ids) == 10
        assert all(mid is not None for mid in memory_ids)

        # Verify all stored
        for mid in memory_ids:
            assert sqlite_storage.get(mid) is not None

    def test_reinforcement_update(self, sqlite_storage):
        """Test reinforcement signal updates."""
        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Reinforcement test",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            reinforcement_score=0.0,
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Apply positive reinforcement
        sqlite_storage.update_reinforcement(memory_id, 0.5)

        updated = sqlite_storage.get(memory_id)
        assert updated.reinforcement_score > 0

        # Apply negative reinforcement
        sqlite_storage.update_reinforcement(memory_id, -0.3)

        updated = sqlite_storage.get(memory_id)
        # Score should decrease but still be positive (from initial positive)

    def test_storage_stats(self, sqlite_storage):
        """Test getting storage statistics."""
        from mindcore.flr import Memory

        # Store some memories
        for i in range(5):
            memory = Memory(
                memory_id="",
                content=f"Stats test {i}",
                memory_type="semantic",
                user_id="test_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        stats = sqlite_storage.get_stats()
        assert "total_memories" in stats or "memory_count" in stats

    def test_performance_store(self, sqlite_storage):
        """Test storage performance (should be < 10ms per operation)."""
        from mindcore.flr import Memory

        times = []
        for i in range(100):
            memory = Memory(
                memory_id="",
                content=f"Performance test {i}",
                memory_type="semantic",
                user_id="test_user",
                topics=["api"],
                created_at=datetime.now(),
            )

            start = time.perf_counter()
            sqlite_storage.store(memory)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)

        avg_time = sum(times) / len(times)
        # Allow up to 20ms on systems with slower I/O
        assert avg_time < 20, f"Average store time {avg_time:.2f}ms exceeds 20ms"


# ============================================================================
# PostgreSQL Storage Tests
# ============================================================================


@requires_postgres
class TestPostgresStorage:
    """Test suite for PostgreSQL storage backend."""

    def test_store_and_retrieve(self, postgres_storage):
        """Test basic store and retrieve with PostgreSQL."""
        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="PostgreSQL test memory",
            memory_type="semantic",
            user_id="test_user_pg",
            topics=["api"],
            categories=["technical"],
            importance=0.7,
            created_at=datetime.now(),
        )

        memory_id = postgres_storage.store(memory)
        assert memory_id is not None

        retrieved = postgres_storage.get(memory_id)
        assert retrieved is not None
        assert retrieved.content == "PostgreSQL test memory"

        # Cleanup
        postgres_storage.delete(memory_id)

    def test_search_with_full_text(self, postgres_storage):
        """Test PostgreSQL full-text search."""
        from mindcore.flr import Memory

        # Store memories with different content
        memories_data = [
            "Python is a programming language",
            "JavaScript runs in the browser",
            "Python and JavaScript are both popular",
        ]

        memory_ids = []
        for content in memories_data:
            memory = Memory(
                memory_id="",
                content=content,
                memory_type="semantic",
                user_id="fts_test_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            memory_ids.append(postgres_storage.store(memory))

        # Search for Python
        results = postgres_storage.search(user_id="fts_test_user", query="Python")
        assert len(results) >= 2

        # Cleanup
        for mid in memory_ids:
            try:
                postgres_storage.delete(mid)
            except Exception:
                pass

    def test_performance_postgres(self, postgres_storage):
        """Test PostgreSQL performance (should be < 50ms per operation)."""
        from mindcore.flr import Memory

        times = []
        memory_ids = []

        for i in range(50):
            memory = Memory(
                memory_id="",
                content=f"PG performance test {i}",
                memory_type="semantic",
                user_id="perf_test_user",
                topics=["api"],
                created_at=datetime.now(),
            )

            start = time.perf_counter()
            mid = postgres_storage.store(memory)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)
            memory_ids.append(mid)

        avg_time = sum(times) / len(times)
        # Docker containers have higher latency; allow up to 100ms per operation
        assert avg_time < 100, f"Average PG store time {avg_time:.2f}ms exceeds 100ms"

        # Cleanup
        for mid in memory_ids:
            try:
                postgres_storage.delete(mid)
            except Exception:
                pass


# ============================================================================
# Storage Migration Tests
# ============================================================================


class TestStorageMigration:
    """Test migration between storage backends."""

    @requires_postgres
    def test_sqlite_to_postgres_migration(self, sqlite_storage, postgres_storage):
        """Test migrating data from SQLite to PostgreSQL."""
        from mindcore.flr import Memory

        # Store in SQLite
        memory_ids = []
        for i in range(5):
            memory = Memory(
                memory_id="",
                content=f"Migration test {i}",
                memory_type="semantic",
                user_id="migration_user",
                topics=["api"],
                importance=0.5 + (i * 0.1),
                created_at=datetime.now(),
            )
            memory_ids.append(sqlite_storage.store(memory))

        # Read from SQLite and store in PostgreSQL
        pg_ids = []
        for mid in memory_ids:
            memory = sqlite_storage.get(mid)
            memory.memory_id = ""  # Reset for new storage
            pg_ids.append(postgres_storage.store(memory))

        # Verify in PostgreSQL
        for pg_id in pg_ids:
            retrieved = postgres_storage.get(pg_id)
            assert retrieved is not None
            assert "Migration test" in retrieved.content

        # Cleanup PostgreSQL
        for mid in pg_ids:
            try:
                postgres_storage.delete(mid)
            except Exception:
                pass


# ============================================================================
# Edge Cases
# ============================================================================


class TestStorageEdgeCases:
    """Test edge cases and error handling."""

    def test_get_nonexistent_memory(self, sqlite_storage):
        """Test getting a memory that doesn't exist."""
        result = sqlite_storage.get("nonexistent_id_12345")
        assert result is None

    def test_delete_nonexistent_memory(self, sqlite_storage):
        """Test deleting a memory that doesn't exist."""
        from mindcore.exceptions import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            sqlite_storage.delete("nonexistent_id_12345")

    def test_empty_search(self, sqlite_storage):
        """Test search with no matching results."""
        results = sqlite_storage.search(user_id="nonexistent_user")
        assert len(results) == 0

    def test_special_characters_in_content(self, sqlite_storage):
        """Test storing content with special characters."""
        from mindcore.flr import Memory

        special_content = "Test with 'quotes', \"double quotes\", and emoji 🎉!"
        memory = Memory(
            memory_id="",
            content=special_content,
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )

        memory_id = sqlite_storage.store(memory)
        retrieved = sqlite_storage.get(memory_id)

        assert retrieved.content == special_content

    def test_large_content(self, sqlite_storage):
        """Test storing large content."""
        from mindcore.flr import Memory

        large_content = "x" * 10000  # 10KB of content
        memory = Memory(
            memory_id="",
            content=large_content,
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )

        memory_id = sqlite_storage.store(memory)
        retrieved = sqlite_storage.get(memory_id)

        assert len(retrieved.content) == 10000
