"""Comprehensive tests for SQLite storage backend."""

import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest

from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.flr import Memory


class TestSQLiteStorageInit:
    """Test SQLite storage initialization."""

    def test_create_database(self):
        """Test creating new database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            assert os.path.exists(db_path)
            storage.close()
        finally:
            os.unlink(db_path)

    def test_schema_created(self):
        """Test that schema is properly created."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            conn = storage._get_connection()

            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            assert "memories" in tables
            assert "transfers" in tables
            assert "memories_fts" in tables

            storage.close()
        finally:
            os.unlink(db_path)


class TestSQLiteStorageBasicOps:
    """Test basic CRUD operations."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_store_memory(self, storage):
        """Test storing a memory."""
        memory = Memory(
            memory_id="",
            content="Test memory content",
            memory_type="episodic",
            user_id="user_123",
        )

        memory_id = storage.store(memory)

        assert memory_id is not None
        assert memory_id.startswith("mem_")

    def test_store_with_id(self, storage):
        """Test storing memory with predefined ID."""
        memory = Memory(
            memory_id="custom_id_123",
            content="Test content",
            memory_type="semantic",
            user_id="user_456",
        )

        memory_id = storage.store(memory)

        assert memory_id == "custom_id_123"

    def test_store_with_all_fields(self, storage):
        """Test storing memory with all fields."""
        now = datetime.now(timezone.utc)
        memory = Memory(
            memory_id="mem_full",
            content="Full memory",
            memory_type="preference",
            user_id="user_123",
            agent_id="agent_456",
            topics=["topic1", "topic2"],
            categories=["cat1"],
            sentiment="positive",
            importance=0.9,
            entities=["entity1"],
            access_level="team",
            created_at=now,
            vocabulary_version="2.0.0",
        )

        memory_id = storage.store(memory)
        retrieved = storage.get(memory_id)

        assert retrieved is not None
        assert retrieved.topics == ["topic1", "topic2"]
        assert retrieved.sentiment == "positive"
        assert retrieved.importance == 0.9
        assert retrieved.vocabulary_version == "2.0.0"

    def test_get_memory(self, storage):
        """Test retrieving a memory."""
        memory = Memory(
            memory_id="mem_get",
            content="Get test",
            memory_type="episodic",
            user_id="user_123",
        )
        storage.store(memory)

        retrieved = storage.get("mem_get")

        assert retrieved is not None
        assert retrieved.content == "Get test"
        assert retrieved.memory_type == "episodic"

    def test_get_nonexistent(self, storage):
        """Test getting non-existent memory."""
        result = storage.get("nonexistent_id")
        assert result is None

    def test_update_memory(self, storage):
        """Test updating a memory."""
        memory = Memory(
            memory_id="mem_update",
            content="Original content",
            memory_type="episodic",
            user_id="user_123",
        )
        storage.store(memory)

        # Update
        memory.content = "Updated content"
        memory.importance = 0.8
        result = storage.update(memory)

        assert result is True

        retrieved = storage.get("mem_update")
        assert retrieved.content == "Updated content"
        assert retrieved.importance == 0.8

    def test_update_nonexistent(self, storage):
        """Test updating non-existent memory."""
        memory = Memory(
            memory_id="nonexistent",
            content="Test",
            memory_type="episodic",
            user_id="user_123",
        )

        result = storage.update(memory)
        assert result is False

    def test_delete_memory(self, storage):
        """Test deleting a memory."""
        memory = Memory(
            memory_id="mem_delete",
            content="To be deleted",
            memory_type="episodic",
            user_id="user_123",
        )
        storage.store(memory)

        result = storage.delete("mem_delete")
        assert result is True

        retrieved = storage.get("mem_delete")
        assert retrieved is None

    def test_delete_nonexistent(self, storage):
        """Test deleting non-existent memory."""
        result = storage.delete("nonexistent_id")
        assert result is False


class TestSQLiteStorageSearch:
    """Test search functionality."""

    @pytest.fixture
    def storage(self):
        """Create storage with test data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)

        # Add test data
        memories = [
            Memory(
                memory_id="mem_1",
                content="User prefers email communication",
                memory_type="preference",
                user_id="user_1",
                topics=["communication"],
                categories=["settings"],
                importance=0.8,
            ),
            Memory(
                memory_id="mem_2",
                content="Customer had billing issue",
                memory_type="episodic",
                user_id="user_1",
                topics=["billing"],
                categories=["support"],
                importance=0.7,
            ),
            Memory(
                memory_id="mem_3",
                content="Technical support request",
                memory_type="episodic",
                user_id="user_1",
                agent_id="agent_1",
                topics=["support"],
                access_level="team",
            ),
            Memory(
                memory_id="mem_4",
                content="User 2 preference",
                memory_type="preference",
                user_id="user_2",
                topics=["settings"],
            ),
        ]

        for m in memories:
            storage.store(m)

        yield storage
        storage.close()
        os.unlink(db_path)

    def test_search_by_user(self, storage):
        """Test searching by user ID."""
        results = storage.search(user_id="user_1")

        assert len(results) == 3
        for m in results:
            assert m.user_id == "user_1"

    def test_search_by_query(self, storage):
        """Test full-text search."""
        results = storage.search(query="billing", user_id="user_1")

        assert len(results) >= 1
        assert any("billing" in m.content.lower() for m in results)

    def test_search_by_topics(self, storage):
        """Test searching by topics."""
        results = storage.search(
            user_id="user_1",
            topics=["billing"],
        )

        assert len(results) >= 1
        for m in results:
            assert "billing" in m.topics

    def test_search_by_multiple_topics(self, storage):
        """Test searching by multiple topics (OR)."""
        results = storage.search(
            user_id="user_1",
            topics=["billing", "communication"],
        )

        assert len(results) >= 2

    def test_search_by_categories(self, storage):
        """Test searching by categories."""
        results = storage.search(
            user_id="user_1",
            categories=["support"],
        )

        assert len(results) >= 1
        for m in results:
            assert "support" in m.categories

    def test_search_by_memory_type(self, storage):
        """Test searching by memory type."""
        results = storage.search(
            user_id="user_1",
            memory_types=["preference"],
        )

        assert len(results) >= 1
        for m in results:
            assert m.memory_type == "preference"

    def test_search_by_agent(self, storage):
        """Test searching by agent ID."""
        results = storage.search(agent_id="agent_1")

        assert len(results) == 1
        assert results[0].agent_id == "agent_1"

    def test_search_by_importance(self, storage):
        """Test searching by minimum importance."""
        results = storage.search(
            user_id="user_1",
            min_importance=0.75,
        )

        for m in results:
            assert m.importance >= 0.75

    def test_search_by_access_level(self, storage):
        """Test searching by access level."""
        results = storage.search(
            user_id="user_1",
            access_levels=["team"],
        )

        assert len(results) >= 1
        for m in results:
            assert m.access_level == "team"

    def test_search_with_limit(self, storage):
        """Test search with limit."""
        results = storage.search(user_id="user_1", limit=2)
        assert len(results) <= 2

    def test_search_with_offset(self, storage):
        """Test search with offset."""
        all_results = storage.search(user_id="user_1")
        offset_results = storage.search(user_id="user_1", offset=1)

        if len(all_results) > 1:
            assert len(offset_results) == len(all_results) - 1


class TestSQLiteStorageDateFilters:
    """Test date-based filtering."""

    @pytest.fixture
    def storage(self):
        """Create storage with dated data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)

        now = datetime.now(timezone.utc)

        # Old memory
        old = Memory(
            memory_id="mem_old",
            content="Old memory",
            memory_type="episodic",
            user_id="user_1",
            created_at=now - timedelta(days=30),
        )

        # Recent memory
        recent = Memory(
            memory_id="mem_recent",
            content="Recent memory",
            memory_type="episodic",
            user_id="user_1",
            created_at=now - timedelta(days=1),
        )

        storage.store(old)
        storage.store(recent)

        yield storage, now
        storage.close()
        os.unlink(db_path)

    def test_search_with_start_date(self, storage):
        """Test filtering by start date."""
        storage_obj, now = storage
        start = now - timedelta(days=7)

        results = storage_obj.search(
            user_id="user_1",
            start_date=start,
        )

        assert len(results) == 1
        assert results[0].memory_id == "mem_recent"

    def test_search_with_end_date(self, storage):
        """Test filtering by end date."""
        storage_obj, now = storage
        end = now - timedelta(days=15)

        results = storage_obj.search(
            user_id="user_1",
            end_date=end,
        )

        assert len(results) == 1
        assert results[0].memory_id == "mem_old"

    def test_search_with_date_range(self, storage):
        """Test filtering by date range."""
        storage_obj, now = storage
        start = now - timedelta(days=35)
        end = now - timedelta(days=25)

        results = storage_obj.search(
            user_id="user_1",
            start_date=start,
            end_date=end,
        )

        assert len(results) == 1
        assert results[0].memory_id == "mem_old"


class TestSQLiteStorageExpiration:
    """Test memory expiration."""

    @pytest.fixture
    def storage(self):
        """Create storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_expired_memory_not_returned(self, storage):
        """Test that expired memories are filtered out."""
        now = datetime.now(timezone.utc)

        # Expired memory
        expired = Memory(
            memory_id="mem_expired",
            content="Expired",
            memory_type="episodic",
            user_id="user_1",
            expires_at=now - timedelta(hours=1),
        )

        # Valid memory
        valid = Memory(
            memory_id="mem_valid",
            content="Valid",
            memory_type="episodic",
            user_id="user_1",
            expires_at=now + timedelta(days=1),
        )

        storage.store(expired)
        storage.store(valid)

        results = storage.search(user_id="user_1")

        assert len(results) == 1
        assert results[0].memory_id == "mem_valid"

    def test_no_expiration_returned(self, storage):
        """Test that memories without expiration are returned."""
        memory = Memory(
            memory_id="mem_no_expire",
            content="No expiration",
            memory_type="episodic",
            user_id="user_1",
            expires_at=None,
        )
        storage.store(memory)

        results = storage.search(user_id="user_1")

        assert len(results) == 1


class TestSQLiteStorageReinforcement:
    """Test reinforcement score updates."""

    @pytest.fixture
    def storage(self):
        """Create storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_update_reinforcement_positive(self, storage):
        """Test positive reinforcement update."""
        memory = Memory(
            memory_id="mem_reinforce",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            reinforcement_score=0.0,
        )
        storage.store(memory)

        result = storage.update_reinforcement("mem_reinforce", 0.5)
        assert result is True

        retrieved = storage.get("mem_reinforce")
        assert retrieved.reinforcement_score == 0.5

    def test_update_reinforcement_negative(self, storage):
        """Test negative reinforcement update."""
        memory = Memory(
            memory_id="mem_reinforce",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            reinforcement_score=0.5,
        )
        storage.store(memory)

        storage.update_reinforcement("mem_reinforce", -0.3)

        retrieved = storage.get("mem_reinforce")
        assert abs(retrieved.reinforcement_score - 0.2) < 0.01

    def test_update_reinforcement_accumulates(self, storage):
        """Test that reinforcement accumulates."""
        memory = Memory(
            memory_id="mem_acc",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
        )
        storage.store(memory)

        storage.update_reinforcement("mem_acc", 0.3)
        storage.update_reinforcement("mem_acc", 0.4)

        retrieved = storage.get("mem_acc")
        assert abs(retrieved.reinforcement_score - 0.7) < 0.01

    def test_update_reinforcement_nonexistent(self, storage):
        """Test reinforcing non-existent memory."""
        result = storage.update_reinforcement("nonexistent", 0.5)
        assert result is False


class TestSQLiteStorageVersion:
    """Test vocabulary version operations."""

    @pytest.fixture
    def storage(self):
        """Create storage with versioned data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)

        # Add memories with different versions
        v1 = Memory(
            memory_id="mem_v1",
            content="Version 1",
            memory_type="episodic",
            user_id="user_1",
            vocabulary_version="1.0.0",
        )
        v2 = Memory(
            memory_id="mem_v2",
            content="Version 2",
            memory_type="episodic",
            user_id="user_1",
            vocabulary_version="2.0.0",
        )
        v2_other = Memory(
            memory_id="mem_v2_other",
            content="Version 2 other user",
            memory_type="episodic",
            user_id="user_2",
            vocabulary_version="2.0.0",
        )

        storage.store(v1)
        storage.store(v2)
        storage.store(v2_other)

        yield storage
        storage.close()
        os.unlink(db_path)

    def test_search_by_version(self, storage):
        """Test searching by vocabulary version."""
        results = storage.search_by_version("1.0.0")

        assert len(results) == 1
        assert results[0].vocabulary_version == "1.0.0"

    def test_search_by_version_with_user(self, storage):
        """Test searching by version with user filter."""
        results = storage.search_by_version("2.0.0", user_id="user_1")

        assert len(results) == 1
        assert results[0].user_id == "user_1"

    def test_search_by_version_limit(self, storage):
        """Test version search with limit."""
        results = storage.search_by_version("2.0.0", limit=1)
        assert len(results) <= 1


class TestSQLiteStorageTransfers:
    """Test transfer data storage."""

    @pytest.fixture
    def storage(self):
        """Create storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_store_transfer(self, storage):
        """Test storing transfer data."""
        data = [
            {"memory_id": "mem_1", "content": "Test 1"},
            {"memory_id": "mem_2", "content": "Test 2"},
        ]

        # Should not raise
        storage.store_transfer("transfer_123", data)

    def test_get_transfer(self, storage):
        """Test retrieving transfer data."""
        data = [
            {"memory_id": "mem_1", "content": "Test 1"},
            {"memory_id": "mem_2", "content": "Test 2"},
        ]
        storage.store_transfer("transfer_123", data)

        retrieved = storage.get_transfer("transfer_123")

        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0]["memory_id"] == "mem_1"

    def test_get_transfer_nonexistent(self, storage):
        """Test getting non-existent transfer."""
        result = storage.get_transfer("nonexistent")
        assert result is None


class TestSQLiteStorageStats:
    """Test storage statistics."""

    @pytest.fixture
    def storage(self):
        """Create storage with data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)

        # Add diverse data
        memories = [
            Memory(
                memory_id="mem_1",
                content="Test 1",
                memory_type="episodic",
                user_id="user_1",
            ),
            Memory(
                memory_id="mem_2",
                content="Test 2",
                memory_type="preference",
                user_id="user_1",
                agent_id="agent_1",
            ),
            Memory(
                memory_id="mem_3",
                content="Test 3",
                memory_type="episodic",
                user_id="user_2",
                agent_id="agent_2",
            ),
        ]

        for m in memories:
            storage.store(m)

        yield storage, db_path
        storage.close()
        os.unlink(db_path)

    def test_get_stats(self, storage):
        """Test getting storage stats."""
        storage_obj, _ = storage

        stats = storage_obj.get_stats()

        assert stats["total_memories"] == 3
        assert stats["unique_users"] == 2
        assert stats["unique_agents"] == 2
        assert "by_memory_type" in stats
        assert stats["by_memory_type"]["episodic"] == 2
        assert stats["by_memory_type"]["preference"] == 1
        assert stats["database_size_bytes"] > 0


class TestSQLiteStorageThreadSafety:
    """Test thread safety."""

    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            errors = []

            def write_memory(thread_id):
                try:
                    for i in range(10):
                        memory = Memory(
                            memory_id=f"mem_{thread_id}_{i}",
                            content=f"Thread {thread_id} memory {i}",
                            memory_type="episodic",
                            user_id="user_1",
                        )
                        storage.store(memory)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=write_memory, args=(i,))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            # Verify all memories were stored
            results = storage.search(user_id="user_1", limit=100)
            assert len(results) == 50

            storage.close()
        finally:
            os.unlink(db_path)

    def test_concurrent_reads_writes(self):
        """Test concurrent read and write operations."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)

            # Pre-populate
            for i in range(10):
                memory = Memory(
                    memory_id=f"mem_pre_{i}",
                    content=f"Pre memory {i}",
                    memory_type="episodic",
                    user_id="user_1",
                )
                storage.store(memory)

            errors = []

            def writer():
                try:
                    for i in range(10):
                        memory = Memory(
                            memory_id=f"mem_new_{i}",
                            content=f"New memory {i}",
                            memory_type="episodic",
                            user_id="user_1",
                        )
                        storage.store(memory)
                except Exception as e:
                    errors.append(e)

            def reader():
                try:
                    for _ in range(10):
                        storage.search(user_id="user_1")
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            storage.close()
        finally:
            os.unlink(db_path)
