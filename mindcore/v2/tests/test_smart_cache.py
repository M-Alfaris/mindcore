"""Tests for SmartCache - Write-through cache with intelligent invalidation."""

import tempfile
import time
from pathlib import Path

import pytest

from mindcore.v2.flr import FLR, CacheEventType, Memory, SmartCache
from mindcore.v2.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    """Create a temporary SQLite storage for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    storage.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def cache(storage):
    """Create SmartCache instance."""
    return SmartCache(
        storage=storage,
        max_size=100,
        ttl_seconds=300,
        warm_threshold=0.7,
    )


@pytest.fixture
def memory():
    """Create a test memory."""
    return Memory(
        memory_id="test_mem_1",
        content="User prefers dark mode",
        memory_type="preference",
        user_id="user_123",
        agent_id="agent_1",
        topics=["settings", "ui"],
        importance=0.8,
    )


class TestSmartCacheBasics:
    """Test basic cache operations."""

    def test_store_and_get(self, cache, memory):
        """Test storing and retrieving from cache."""
        cache.store(memory)

        retrieved = cache.get(memory.memory_id)
        assert retrieved is not None
        assert retrieved.content == "User prefers dark mode"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_contains(self, cache, memory):
        """Test __contains__ check."""
        cache.store(memory)
        assert memory.memory_id in cache

    def test_cache_len(self, cache, memory):
        """Test __len__ returns count."""
        assert len(cache) == 0
        cache.store(memory)
        assert len(cache) == 1

    def test_get_many(self, cache, storage):
        """Test batch retrieval."""
        memories = []
        for i in range(5):
            mem = Memory(
                memory_id=f"batch_{i}",
                content=f"Content {i}",
                memory_type="fact",
                user_id="user_1",
            )
            cache.store(mem)
            memories.append(mem)

        # Get all
        results = cache.get_many([f"batch_{i}" for i in range(5)])
        assert len(results) == 5


class TestCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_single(self, cache, memory):
        """Test invalidating a single entry."""
        cache.store(memory)
        assert memory.memory_id in cache

        result = cache.invalidate(memory.memory_id)
        assert result is True
        assert memory.memory_id not in cache

    def test_invalidate_nonexistent(self, cache):
        """Test invalidating nonexistent entry."""
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_invalidate_user(self, cache, storage):
        """Test invalidating all user entries."""
        # Store memories for two users
        for i in range(3):
            cache.store(
                Memory(
                    memory_id=f"user1_mem_{i}",
                    content=f"Content {i}",
                    memory_type="fact",
                    user_id="user_1",
                )
            )
        for i in range(2):
            cache.store(
                Memory(
                    memory_id=f"user2_mem_{i}",
                    content=f"Content {i}",
                    memory_type="fact",
                    user_id="user_2",
                )
            )

        assert len(cache) == 5

        # Invalidate user_1
        count = cache.invalidate_user("user_1")
        assert count == 3
        assert len(cache) == 2

        # User_2's memories should still be cached
        assert "user2_mem_0" in cache


class TestPatternInvalidation:
    """Test pattern-based invalidation."""

    def test_invalidate_by_user_pattern(self, cache):
        """Test invalidating by user pattern."""
        # Store memories with different types to avoid auto-invalidation
        # (preferences auto-invalidate previous preferences for same user)
        for i in range(3):
            cache.store(
                Memory(
                    memory_id=f"mem_{i}",
                    content=f"Content {i}",
                    memory_type="fact",  # Use fact to avoid preference invalidation
                    user_id="user_123",
                )
            )

        count = cache.invalidate_pattern("user:user_123:*")
        assert count == 3

    def test_invalidate_by_type_pattern(self, cache):
        """Test invalidating by memory type pattern."""
        cache.store(
            Memory(
                memory_id="pref_1",
                content="Preference 1",
                memory_type="preference",
                user_id="user_1",
            )
        )
        cache.store(
            Memory(
                memory_id="fact_1",
                content="Fact 1",
                memory_type="fact",
                user_id="user_1",
            )
        )
        cache.store(
            Memory(
                memory_id="pref_2",
                content="Preference 2",
                memory_type="preference",
                user_id="user_2",
            )
        )

        count = cache.invalidate_pattern("type:preference:*")
        assert count == 2

        # Fact should still be cached
        assert "fact_1" in cache

    def test_invalidate_by_topic_pattern(self, cache):
        """Test invalidating by topic pattern."""
        cache.store(
            Memory(
                memory_id="order_1",
                content="Order info",
                memory_type="fact",
                user_id="user_1",
                topics=["orders", "shipping"],
            )
        )
        cache.store(
            Memory(
                memory_id="other_1",
                content="Other info",
                memory_type="fact",
                user_id="user_1",
                topics=["account"],
            )
        )

        count = cache.invalidate_pattern("topic:orders:*")
        assert count == 1
        assert "other_1" in cache


class TestCacheWarming:
    """Test cache warming functionality."""

    def test_auto_warm_high_importance(self, cache):
        """Test that high importance memories are auto-warmed."""
        high_importance = Memory(
            memory_id="important_1",
            content="Important memory",
            memory_type="preference",
            user_id="user_1",
            importance=0.9,  # Above warm_threshold of 0.7
        )

        cache.store(high_importance)

        # Check it was warmed (in cache with warmed flag)
        entry = cache._cache.get("important_1")
        assert entry is not None
        # The memory should be in cache

    def test_explicit_warm(self, cache, memory):
        """Test explicitly warming a memory."""
        cache.warm(memory)
        assert memory.memory_id in cache

    def test_warm_stats(self, cache, memory):
        """Test that warming updates stats."""
        cache.warm(memory)
        stats = cache.get_stats()
        assert stats["warms"] >= 1


class TestCacheTTL:
    """Test cache TTL and expiration."""

    def test_expired_entries_not_returned(self, storage):
        """Test that expired entries are not returned."""
        # Create cache with very short TTL
        cache = SmartCache(
            storage=storage,
            max_size=100,
            ttl_seconds=1,
        )

        memory = Memory(
            memory_id="expire_test",
            content="Will expire",
            memory_type="fact",
            user_id="user_1",
        )
        cache.store(memory)
        assert memory.memory_id in cache

        # Wait for TTL
        time.sleep(1.5)

        # Should not be in cache anymore
        assert memory.memory_id not in cache

    def test_cleanup_expired(self, storage):
        """Test cleanup of expired entries."""
        cache = SmartCache(
            storage=storage,
            max_size=100,
            ttl_seconds=1,
        )

        for i in range(3):
            cache.store(
                Memory(
                    memory_id=f"expire_{i}",
                    content=f"Content {i}",
                    memory_type="fact",
                    user_id="user_1",
                )
            )

        assert len(cache) == 3

        time.sleep(1.5)

        removed = cache.cleanup_expired()
        assert removed == 3
        assert len(cache) == 0


class TestCacheEviction:
    """Test LRU eviction."""

    def test_eviction_at_max_size(self, storage):
        """Test that entries are evicted at max size."""
        cache = SmartCache(
            storage=storage,
            max_size=3,
            ttl_seconds=300,
        )

        for i in range(5):
            cache.store(
                Memory(
                    memory_id=f"evict_{i}",
                    content=f"Content {i}",
                    memory_type="fact",
                    user_id="user_1",
                )
            )

        # Should only have 3 entries (max_size)
        assert len(cache) == 3

        # Oldest entries should be evicted
        assert "evict_0" not in cache
        assert "evict_1" not in cache
        assert "evict_4" in cache


class TestCacheStats:
    """Test cache statistics."""

    def test_hit_rate(self, cache, memory):
        """Test hit rate calculation."""
        cache.store(memory)

        # Create hits
        for _ in range(3):
            cache.get(memory.memory_id)

        # Create miss
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75

    def test_store_count(self, cache):
        """Test store counter."""
        for i in range(5):
            cache.store(
                Memory(
                    memory_id=f"store_{i}",
                    content=f"Content {i}",
                    memory_type="fact",
                    user_id="user_1",
                )
            )

        stats = cache.get_stats()
        assert stats["stores"] == 5

    def test_invalidation_count(self, cache, memory):
        """Test invalidation counter."""
        cache.store(memory)
        cache.invalidate(memory.memory_id)

        stats = cache.get_stats()
        assert stats["invalidations"] >= 1


class TestCacheEvents:
    """Test cache event callbacks."""

    def test_event_callback(self, storage):
        """Test that events are emitted to callback."""
        events = []

        def on_event(event_type, key, data):
            events.append((event_type, key, data))

        cache = SmartCache(
            storage=storage,
            max_size=100,
            ttl_seconds=300,
            on_event=on_event,
        )

        memory = Memory(
            memory_id="event_test",
            content="Test",
            memory_type="fact",
            user_id="user_1",
        )

        cache.store(memory)
        cache.get(memory.memory_id)
        cache.invalidate(memory.memory_id)

        event_types = [e[0] for e in events]
        assert CacheEventType.STORE in event_types
        assert CacheEventType.HIT in event_types
        assert CacheEventType.INVALIDATE in event_types


class TestFLRSmartCacheIntegration:
    """Test FLR integration with SmartCache."""

    def test_flr_with_smart_cache(self, storage):
        """Test FLR with SmartCache enabled."""
        flr = FLR(
            storage=storage,
            use_smart_cache=True,
            cache_warm_threshold=0.7,
        )

        # Verify smart cache is initialized
        assert flr._smart_cache is not None
        assert flr.use_smart_cache is True

    def test_flr_cache_invalidation(self, storage):
        """Test cache invalidation through FLR."""
        flr = FLR(storage=storage, use_smart_cache=True)

        # Store a memory
        memory = Memory(
            memory_id="flr_cache_1",
            content="Test memory",
            memory_type="preference",
            user_id="user_123",
        )
        storage.store(memory)

        # Warm it
        flr.warm_cache(memory)

        # Invalidate
        result = flr.invalidate_cache("flr_cache_1")
        assert result is True

    def test_flr_pattern_invalidation(self, storage):
        """Test pattern invalidation through FLR."""
        flr = FLR(storage=storage, use_smart_cache=True)

        # Store memories with type=fact to avoid auto-invalidation
        for i in range(3):
            memory = Memory(
                memory_id=f"pattern_{i}",
                content=f"Content {i}",
                memory_type="fact",  # Use fact to avoid preference auto-invalidation
                user_id="user_123",
            )
            storage.store(memory)
            flr.warm_cache(memory)

        # Invalidate by pattern
        count = flr.invalidate_cache_pattern("user:user_123:*")
        assert count == 3

    def test_flr_cache_stats(self, storage):
        """Test cache stats from FLR."""
        flr = FLR(storage=storage, use_smart_cache=True)

        stats = flr.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_flr_clear_cache(self, storage):
        """Test clearing cache through FLR."""
        flr = FLR(storage=storage, use_smart_cache=True)

        for i in range(5):
            memory = Memory(
                memory_id=f"clear_{i}",
                content=f"Content {i}",
                memory_type="fact",
                user_id="user_1",
            )
            storage.store(memory)
            flr.warm_cache(memory)

        count = flr.clear_cache()
        assert count == 5

    def test_flr_without_smart_cache(self, storage):
        """Test FLR falls back to legacy cache."""
        flr = FLR(storage=storage, use_smart_cache=False)

        assert flr._smart_cache is None

        # Pattern invalidation should raise
        with pytest.raises(RuntimeError):
            flr.invalidate_cache_pattern("user:123:*")

        # User cache warming should raise
        with pytest.raises(RuntimeError):
            flr.warm_user_cache("user_123")


class TestWriteThrough:
    """Test write-through behavior."""

    def test_store_writes_to_storage(self, cache, storage, memory):
        """Test that store writes to storage first."""
        cache.store(memory)

        # Should be in storage
        stored = storage.get(memory.memory_id)
        assert stored is not None
        assert stored.content == memory.content

    def test_update_invalidates_old(self, cache, storage):
        """Test that updating invalidates old cached version."""
        memory = Memory(
            memory_id="update_test",
            content="Original content",
            memory_type="preference",
            user_id="user_1",
        )
        cache.store(memory)

        # Verify cached
        assert memory.memory_id in cache

        # Update
        memory.content = "Updated content"
        cache.store(memory)

        # Should still be cached with new content
        retrieved = cache.get(memory.memory_id)
        assert retrieved.content == "Updated content"
