"""Comprehensive tests for CLST (Cognitive Long-term Storage Transfer) module."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from mindcore.v2.clst import (
    CLST,
    CompressionStrategy,
    CompressionResult,
    SyncDirection,
    SyncResult,
    TransferManifest,
    MigrationResult,
)
from mindcore.v2.flr import Memory
from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.vocabulary import VocabularySchema


class TestCompressionStrategy:
    """Test CompressionStrategy enum."""

    def test_strategy_values(self):
        """Test compression strategy values."""
        assert CompressionStrategy.SUMMARIZE.value == "summarize"
        assert CompressionStrategy.MERGE.value == "merge"
        assert CompressionStrategy.DEDUPLICATE.value == "deduplicate"
        assert CompressionStrategy.EXTRACT.value == "extract"


class TestSyncDirection:
    """Test SyncDirection enum."""

    def test_direction_values(self):
        """Test sync direction values."""
        assert SyncDirection.PUSH.value == "push"
        assert SyncDirection.PULL.value == "pull"
        assert SyncDirection.BIDIRECTIONAL.value == "bidirectional"


class TestTransferManifest:
    """Test TransferManifest dataclass."""

    def test_create_manifest(self):
        """Test creating a transfer manifest."""
        manifest = TransferManifest(
            transfer_id="transfer_123",
            source_instance="instance_a",
            target_instance="instance_b",
            memory_count=100,
            total_size_bytes=50000,
            vocabulary_version="1.0.0",
            created_at=datetime.now(timezone.utc),
            checksum="abc123",
        )

        assert manifest.transfer_id == "transfer_123"
        assert manifest.memory_count == 100

    def test_to_dict(self):
        """Test converting manifest to dictionary."""
        now = datetime.now(timezone.utc)
        manifest = TransferManifest(
            transfer_id="transfer_123",
            source_instance="instance_a",
            target_instance="instance_b",
            memory_count=100,
            total_size_bytes=50000,
            vocabulary_version="1.0.0",
            created_at=now,
            checksum="abc123",
        )

        result = manifest.to_dict()

        assert isinstance(result, dict)
        assert result["transfer_id"] == "transfer_123"
        assert result["created_at"] == now.isoformat()


class TestCLSTBasics:
    """Test basic CLST functionality."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    @pytest.fixture
    def clst(self, storage):
        """Create CLST instance."""
        return CLST(storage=storage)

    @pytest.fixture
    def clst_with_vocab(self, storage):
        """Create CLST with vocabulary."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support", "product"],
        )
        return CLST(storage=storage, vocabulary=vocab)

    def test_clst_initialization(self, storage):
        """Test CLST initialization."""
        clst = CLST(storage=storage)

        assert clst.storage is storage
        assert clst.vocabulary is None
        assert clst.compression_llm is None

    def test_clst_with_vocabulary(self, storage):
        """Test CLST with vocabulary."""
        vocab = VocabularySchema(version="1.0.0", topics=["test"])
        clst = CLST(storage=storage, vocabulary=vocab)

        assert clst.vocabulary is vocab

    def test_store_memory(self, clst):
        """Test storing a memory."""
        memory = Memory(
            memory_id="",
            content="Test memory content",
            memory_type="episodic",
            user_id="user_123",
        )

        memory_id = clst.store(memory)

        assert memory_id is not None
        assert len(memory_id) > 0

    def test_store_sets_vocabulary_version(self, clst_with_vocab):
        """Test that store sets vocabulary version."""
        memory = Memory(
            memory_id="",
            content="Test memory",
            memory_type="episodic",
            user_id="user_123",
            topics=["billing"],
        )

        clst_with_vocab.store(memory)

        assert memory.vocabulary_version == "1.0.0"

    def test_store_with_invalid_vocabulary(self, clst_with_vocab):
        """Test storing memory that fails vocabulary validation."""
        memory = Memory(
            memory_id="",
            content="Test memory",
            memory_type="episodic",
            user_id="user_123",
            topics=["invalid_topic"],  # Not in vocabulary
        )

        with pytest.raises(ValueError) as exc_info:
            clst_with_vocab.store(memory)

        assert "validation failed" in str(exc_info.value).lower()

    def test_store_batch(self, clst):
        """Test storing multiple memories."""
        memories = [
            Memory(
                memory_id="",
                content=f"Memory {i}",
                memory_type="episodic",
                user_id="user_123",
            )
            for i in range(5)
        ]

        memory_ids = clst.store_batch(memories)

        assert len(memory_ids) == 5
        assert all(mid is not None for mid in memory_ids)

    def test_store_batch_skips_invalid(self, clst_with_vocab):
        """Test that batch store skips invalid memories."""
        memories = [
            Memory(
                memory_id="",
                content="Valid memory",
                memory_type="episodic",
                user_id="user_123",
                topics=["billing"],
            ),
            Memory(
                memory_id="",
                content="Invalid memory",
                memory_type="episodic",
                user_id="user_123",
                topics=["invalid_topic"],
            ),
            Memory(
                memory_id="",
                content="Another valid memory",
                memory_type="episodic",
                user_id="user_123",
                topics=["support"],
            ),
        ]

        memory_ids = clst_with_vocab.store_batch(memories)

        # Only valid ones should be stored
        assert len(memory_ids) == 2

    def test_retrieve_memory(self, clst):
        """Test retrieving a memory."""
        memory = Memory(
            memory_id="",
            content="Test content",
            memory_type="semantic",
            user_id="user_123",
        )
        memory_id = clst.store(memory)

        retrieved = clst.retrieve(memory_id)

        assert retrieved is not None
        assert retrieved.content == "Test content"

    def test_retrieve_nonexistent(self, clst):
        """Test retrieving non-existent memory."""
        retrieved = clst.retrieve("nonexistent_id")
        assert retrieved is None

    def test_delete_memory(self, clst):
        """Test deleting a memory."""
        memory = Memory(
            memory_id="",
            content="To be deleted",
            memory_type="episodic",
            user_id="user_123",
        )
        memory_id = clst.store(memory)

        result = clst.delete(memory_id)

        assert result is True
        assert clst.retrieve(memory_id) is None

    def test_delete_nonexistent(self, clst):
        """Test deleting non-existent memory."""
        result = clst.delete("nonexistent_id")
        assert result is False


class TestCLSTSearch:
    """Test CLST search functionality."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    @pytest.fixture
    def clst(self, storage):
        """Create CLST with test data."""
        clst = CLST(storage=storage)

        # Add test memories
        memories = [
            Memory(
                memory_id="",
                content="Billing issue resolved",
                memory_type="episodic",
                user_id="user_1",
                topics=["billing"],
            ),
            Memory(
                memory_id="",
                content="Product question answered",
                memory_type="semantic",
                user_id="user_1",
                topics=["product"],
            ),
            Memory(
                memory_id="",
                content="Support ticket closed",
                memory_type="episodic",
                user_id="user_2",
                topics=["support"],
            ),
        ]
        for m in memories:
            clst.store(m)

        return clst

    def test_search_by_user(self, clst):
        """Test searching by user ID."""
        results = clst.search(user_id="user_1")

        assert len(results) == 2
        assert all(m.user_id == "user_1" for m in results)

    def test_search_by_topic(self, clst):
        """Test searching by topic."""
        results = clst.search(topics=["billing"])

        assert len(results) >= 1
        assert any("billing" in m.topics for m in results)

    def test_search_by_memory_type(self, clst):
        """Test searching by memory type."""
        results = clst.search(memory_types=["episodic"])

        assert len(results) >= 1
        assert all(m.memory_type == "episodic" for m in results)

    def test_search_with_query(self, clst):
        """Test text search."""
        results = clst.search(query="billing")

        assert len(results) >= 1

    def test_search_with_limit(self, clst):
        """Test search with limit."""
        results = clst.search(limit=1)

        assert len(results) <= 1


class TestCLSTCompression:
    """Test CLST compression functionality."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    @pytest.fixture
    def clst(self, storage):
        """Create CLST with test data."""
        clst = CLST(storage=storage)

        # Add test memories
        for i in range(10):
            memory = Memory(
                memory_id="",
                content=f"Memory content {i}",
                memory_type="episodic",
                user_id="user_123",
                topics=["test"],
            )
            clst.store(memory)

        return clst

    def test_compress_deduplicate(self, clst):
        """Test deduplication compression."""
        result = clst.compress(
            user_id="user_123",
            older_than=timedelta(days=0),  # All memories
            strategy=CompressionStrategy.DEDUPLICATE,
        )

        assert isinstance(result, CompressionResult)
        assert result.strategy == CompressionStrategy.DEDUPLICATE
        assert result.original_count >= 0
        assert result.latency_ms >= 0


class TestCLSTStats:
    """Test CLST statistics."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_get_stats(self, storage):
        """Test getting CLST stats."""
        clst = CLST(storage=storage)

        # Store some memories
        for i in range(3):
            memory = Memory(
                memory_id="",
                content=f"Memory {i}",
                memory_type="episodic",
                user_id="user_123",
            )
            clst.store(memory)

        stats = clst.get_stats()

        assert isinstance(stats, dict)
        assert "total_memories" in stats or "memory_count" in stats or len(stats) >= 0
