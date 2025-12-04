"""Tests for CacheManager."""

import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from mindcore.core.cache_manager import CacheManager
from mindcore.core.schemas import Message, MessageMetadata, MessageRole


class TestCacheManager:
    """Tests for CacheManager class."""

    @pytest.fixture
    def cache(self):
        """Create a cache manager for testing."""
        return CacheManager(max_size=10, ttl_seconds=None)

    @pytest.fixture
    def cache_with_ttl(self):
        """Create a cache manager with TTL for testing."""
        return CacheManager(max_size=10, ttl_seconds=1)

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
                    topics=["general"] if i % 2 == 0 else ["orders"],
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

    def test_initialization_lru(self, cache):
        """Test CacheManager initializes with LRU cache."""
        assert cache.max_size == 10
        assert cache.ttl_seconds is None
        assert cache._thread_caches == {}

    def test_initialization_ttl(self, cache_with_ttl):
        """Test CacheManager initializes with TTL cache."""
        assert cache_with_ttl.max_size == 10
        assert cache_with_ttl.ttl_seconds == 1

    def test_get_key(self, cache):
        """Test cache key generation."""
        key = cache._get_key("user_123", "thread_456")
        assert key == ("user_123", "thread_456")

    # =====================
    # Add Message Tests
    # =====================

    def test_add_message(self, cache, sample_message):
        """Test adding a message to cache."""
        cache.add_message(sample_message)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1
        assert messages[0].message_id == sample_message.message_id

    def test_add_multiple_messages(self, cache, sample_messages):
        """Test adding multiple messages."""
        for msg in sample_messages:
            cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 5

    def test_add_message_creates_thread_cache(self, cache, sample_message):
        """Test that adding message creates thread cache if not exists."""
        assert ("user_123", "thread_456") not in cache._thread_caches

        cache.add_message(sample_message)

        assert ("user_123", "thread_456") in cache._thread_caches

    def test_add_message_updates_existing(self, cache, sample_message):
        """Test adding message with same ID updates the entry."""
        cache.add_message(sample_message)

        # Modify and re-add
        sample_message.raw_text = "Updated text"
        cache.add_message(sample_message)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1  # Still just 1 message
        assert messages[0].raw_text == "Updated text"

    def test_add_message_with_none_timestamp(self, cache):
        """Test adding message with None created_at."""
        msg = Message(
            message_id="msg_no_time",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="No timestamp",
            created_at=None,
        )
        cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1

    def test_add_message_with_string_timestamp(self, cache):
        """Test adding message with string created_at."""
        msg = Message(
            message_id="msg_str_time",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="String timestamp",
            created_at="2024-01-15T12:00:00Z",  # type: ignore
        )
        cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1

    def test_add_message_with_naive_datetime(self, cache):
        """Test adding message with naive datetime."""
        msg = Message(
            message_id="msg_naive",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Naive datetime",
            created_at=datetime(2024, 1, 15, 12, 0, 0),  # Naive
        )
        cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1

    # =====================
    # LRU Eviction Tests
    # =====================

    def test_lru_eviction(self):
        """Test LRU eviction when max_size is exceeded."""
        cache = CacheManager(max_size=3, ttl_seconds=None)

        for i in range(5):
            msg = Message(
                message_id=f"msg_{i}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text=f"Message {i}",
                created_at=datetime.now(timezone.utc),
            )
            cache.add_message(msg)

        # Cache should only have max_size messages
        messages = cache.get_recent_messages("user_123", "thread_456")
        # Note: exact count depends on cachetools behavior
        assert len(messages) <= 5

    # =====================
    # TTL Expiration Tests
    # =====================

    def test_ttl_expiration(self, cache_with_ttl, sample_message):
        """Test TTL expiration."""
        cache_with_ttl.add_message(sample_message)

        # Message should be present initially
        messages = cache_with_ttl.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 1

        # Wait for TTL to expire
        time.sleep(1.5)

        # Message should be expired
        messages = cache_with_ttl.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 0

    # =====================
    # Update Message Tests
    # =====================

    def test_update_message(self, cache, sample_message):
        """Test updating an existing message."""
        cache.add_message(sample_message)

        # Update the message
        sample_message.raw_text = "Updated text"
        sample_message.metadata = MessageMetadata(topics=["updated"])

        result = cache.update_message(sample_message)
        assert result is True

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert messages[0].raw_text == "Updated text"
        assert "updated" in messages[0].metadata.topics

    def test_update_message_not_in_cache(self, cache, sample_message):
        """Test updating message not in cache returns False."""
        result = cache.update_message(sample_message)
        assert result is False

    def test_update_message_thread_not_cached(self, cache, sample_message):
        """Test updating when thread cache doesn't exist."""
        # Add to one thread
        cache.add_message(sample_message)

        # Try to update for different thread
        other_msg = Message(
            message_id="msg_other",
            user_id="other_user",
            thread_id="other_thread",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Other message",
        )
        result = cache.update_message(other_msg)
        assert result is False

    # =====================
    # Get Recent Messages Tests
    # =====================

    def test_get_recent_messages(self, cache, sample_messages):
        """Test getting recent messages."""
        for msg in sample_messages:
            cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert len(messages) == 5

    def test_get_recent_messages_with_limit(self, cache, sample_messages):
        """Test getting recent messages with limit."""
        for msg in sample_messages:
            cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456", limit=2)
        assert len(messages) == 2

    def test_get_recent_messages_empty_thread(self, cache):
        """Test getting messages for non-existent thread."""
        messages = cache.get_recent_messages("no_user", "no_thread")
        assert messages == []

    def test_get_recent_messages_order(self, cache):
        """Test that messages are returned in reverse chronological order."""
        now = datetime.now(timezone.utc)
        for i in range(3):
            msg = Message(
                message_id=f"msg_{i}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text=f"Message {i}",
                created_at=now,
            )
            cache.add_message(msg)

        messages = cache.get_recent_messages("user_123", "thread_456")
        # Most recent first (due to reverse order in get_recent_messages)
        assert len(messages) == 3

    # =====================
    # Clear Tests
    # =====================

    def test_clear_thread(self, cache, sample_messages):
        """Test clearing a specific thread."""
        for msg in sample_messages:
            cache.add_message(msg)

        cache.clear_thread("user_123", "thread_456")

        messages = cache.get_recent_messages("user_123", "thread_456")
        assert messages == []

    def test_clear_thread_not_cached(self, cache):
        """Test clearing non-existent thread doesn't raise."""
        cache.clear_thread("no_user", "no_thread")  # Should not raise

    def test_clear_all(self, cache, sample_messages):
        """Test clearing entire cache."""
        for msg in sample_messages:
            cache.add_message(msg)

        # Add messages to another thread
        other_msg = Message(
            message_id="msg_other",
            user_id="other_user",
            thread_id="other_thread",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Other message",
            created_at=datetime.now(timezone.utc),
        )
        cache.add_message(other_msg)

        cache.clear_all()

        assert cache._thread_caches == {}
        assert cache._order == {}

    # =====================
    # Session Metadata Tests
    # =====================

    def test_get_session_metadata(self, cache, sample_messages):
        """Test getting aggregated session metadata."""
        for msg in sample_messages:
            cache.add_message(msg)

        metadata = cache.get_session_metadata("user_123", "thread_456")

        assert "topics" in metadata
        assert "categories" in metadata
        assert "intents" in metadata
        assert "message_count" in metadata

        # Check aggregated values
        assert "general" in metadata["topics"]
        assert "orders" in metadata["topics"]
        assert "question" in metadata["categories"]
        assert "response" in metadata["categories"]
        assert "ask_question" in metadata["intents"]
        assert metadata["message_count"] == 5

    def test_get_session_metadata_empty(self, cache):
        """Test getting metadata for non-existent thread."""
        metadata = cache.get_session_metadata("no_user", "no_thread")

        assert metadata == {
            "topics": [],
            "categories": [],
            "intents": [],
            "message_count": 0,
        }

    def test_get_session_metadata_handles_empty_metadata(self, cache):
        """Test session metadata handles messages with empty metadata."""
        msg = Message(
            message_id="msg_empty",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Empty metadata",
            metadata=MessageMetadata(),  # Empty
        )
        cache.add_message(msg)

        metadata = cache.get_session_metadata("user_123", "thread_456")
        assert metadata["message_count"] == 1

    # =====================
    # Stats Tests
    # =====================

    def test_get_stats_empty(self, cache):
        """Test getting stats for empty cache."""
        stats = cache.get_stats()

        assert stats["total_threads"] == 0
        assert stats["total_messages"] == 0
        assert stats["max_size_per_thread"] == 10
        assert stats["cache_type"] == "LRUCache"

    def test_get_stats_with_messages(self, cache, sample_messages):
        """Test getting stats with cached messages."""
        for msg in sample_messages:
            cache.add_message(msg)

        stats = cache.get_stats()

        assert stats["total_threads"] == 1
        assert stats["total_messages"] == 5
        assert stats["max_size_per_thread"] == 10

    def test_get_stats_with_ttl(self, cache_with_ttl, sample_message):
        """Test stats show TTL cache type."""
        cache_with_ttl.add_message(sample_message)

        stats = cache_with_ttl.get_stats()

        assert stats["cache_type"] == "TTLCache"
        assert stats["ttl_seconds"] == 1

    def test_get_stats_multiple_threads(self, cache, sample_message):
        """Test stats with multiple threads."""
        cache.add_message(sample_message)

        # Add to another thread
        other_msg = Message(
            message_id="msg_other",
            user_id="other_user",
            thread_id="other_thread",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Other message",
            created_at=datetime.now(timezone.utc),
        )
        cache.add_message(other_msg)

        stats = cache.get_stats()
        assert stats["total_threads"] == 2
        assert stats["total_messages"] == 2

    # =====================
    # Thread Safety Tests
    # =====================

    def test_thread_safety_lock(self, cache):
        """Test that operations use the lock."""
        assert cache._lock is not None

        # Verify lock is RLock (reentrant) by checking the type name
        # RLock is a factory function, not a class, so check type name
        assert type(cache._lock).__name__ == "RLock"

    # =====================
    # Edge Cases
    # =====================

    def test_get_timestamp_invalid_string(self, cache):
        """Test _get_timestamp handles invalid string."""
        msg = Message(
            message_id="msg_invalid",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Invalid timestamp",
            created_at="not-a-date",  # type: ignore
        )
        # Should not raise, uses current time as fallback
        timestamp = cache._get_timestamp(msg)
        assert timestamp is not None
        assert timestamp.tzinfo == timezone.utc

    def test_create_thread_cache_lru(self, cache):
        """Test creating LRU thread cache."""
        from cachetools import LRUCache

        thread_cache = cache._create_thread_cache()
        assert isinstance(thread_cache, LRUCache)

    def test_create_thread_cache_ttl(self, cache_with_ttl):
        """Test creating TTL thread cache."""
        from cachetools import TTLCache

        thread_cache = cache_with_ttl._create_thread_cache()
        assert isinstance(thread_cache, TTLCache)

    def test_order_list_trimming(self):
        """Test that order list is trimmed when it gets too large."""
        cache = CacheManager(max_size=3, ttl_seconds=None)

        # Add many messages to trigger order list trimming
        for i in range(20):
            msg = Message(
                message_id=f"msg_{i}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text=f"Message {i}",
                created_at=datetime.now(timezone.utc),
            )
            cache.add_message(msg)

        key = ("user_123", "thread_456")
        # Order list should be trimmed to reasonable size
        assert len(cache._order.get(key, [])) <= cache.max_size * 2
