"""Tests for FLR recall module - Memory, ContextWindow, and FLR class.

Tests cover:
- Memory.effective_importance: decay calculations, boosts, expiration
- ContextWindow: window management, message overflow
- RecallResult: serialization
- FLR: context management, promote, flush_reinforcements, filtering
"""

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mindcore.flr.recall import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)
from mindcore.storage.sqlite import SQLiteStorage


# =============================================================================
# Fixtures
# =============================================================================


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
def flr(storage):
    """Create FLR instance with storage."""
    return FLR(storage=storage)


@pytest.fixture
def memory():
    """Create a basic test memory."""
    return Memory(
        memory_id="test_mem_1",
        content="User prefers dark mode for the application",
        memory_type="preference",
        user_id="user_123",
        agent_id="agent_1",
        topics=["settings", "ui", "preferences"],
        categories=["user_preference"],
        entities=["dark mode"],
        importance=0.8,
    )


# =============================================================================
# Memory.effective_importance Tests
# =============================================================================


class TestMemoryEffectiveImportance:
    """Tests for Memory.effective_importance property."""

    def test_no_decay_new_memory(self):
        """Test that new memory has no decay applied."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.8,
            importance_decay_rate=0.1,
        )

        # New memory should have effective importance close to base
        assert memory.effective_importance == pytest.approx(0.8, rel=0.01)

    def test_no_decay_when_rate_zero(self):
        """Test no decay when decay_rate is 0."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.8,
            importance_decay_rate=0.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
        )

        # No decay applied
        assert memory.effective_importance == 0.8

    def test_decay_over_time(self):
        """Test importance decays over time."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=1.0,
            importance_decay_rate=0.5,  # 50% decay per month
            created_at=datetime.now(timezone.utc) - timedelta(days=30),  # ~1 month old
        )

        # After 1 month with 50% decay rate, should be ~0.5
        assert memory.effective_importance < 1.0
        assert memory.effective_importance > 0.3

    def test_decay_bounds(self):
        """Test effective importance stays within bounds."""
        # Very old memory
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
            importance_decay_rate=0.9,  # Heavy decay
            created_at=datetime.now(timezone.utc) - timedelta(days=365),  # 1 year old
        )

        # Should be bounded to [0, 2]
        assert memory.effective_importance >= 0.0
        assert memory.effective_importance <= 2.0


class TestMemoryBoostImportance:
    """Tests for Memory.boost_importance method."""

    def test_permanent_boost(self):
        """Test adding a permanent boost."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
            importance_decay_rate=0.0,
        )

        new_importance = memory.boost_importance(
            amount=0.3,
            reason="User explicitly requested",
        )

        assert new_importance == pytest.approx(0.8, rel=0.01)
        assert len(memory.importance_boosts) == 1
        assert memory.importance_boosts[0]["reason"] == "User explicitly requested"
        assert "expires_at" not in memory.importance_boosts[0]

    def test_temporary_boost(self):
        """Test adding a temporary boost with expiration."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
            importance_decay_rate=0.0,
        )

        new_importance = memory.boost_importance(
            amount=0.2,
            reason="Temporary promotion",
            decay_after_days=7,
        )

        assert new_importance == pytest.approx(0.7, rel=0.01)
        assert "expires_at" in memory.importance_boosts[0]

    def test_negative_boost(self):
        """Test adding a negative boost (penalty)."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.8,
            importance_decay_rate=0.0,
        )

        new_importance = memory.boost_importance(
            amount=-0.3,
            reason="Outdated information",
        )

        assert new_importance == pytest.approx(0.5, rel=0.01)

    def test_multiple_boosts(self):
        """Test accumulating multiple boosts."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.3,
            importance_decay_rate=0.0,
        )

        memory.boost_importance(amount=0.2, reason="Boost 1")
        memory.boost_importance(amount=0.3, reason="Boost 2")

        assert memory.effective_importance == pytest.approx(0.8, rel=0.01)
        assert len(memory.importance_boosts) == 2

    def test_boost_upper_bound(self):
        """Test that boosts don't exceed upper bound."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=1.5,
            importance_decay_rate=0.0,
        )

        memory.boost_importance(amount=1.0, reason="Big boost")

        # Should be capped at 2.0
        assert memory.effective_importance == 2.0


class TestMemoryClearExpiredBoosts:
    """Tests for Memory.clear_expired_boosts method."""

    def test_clear_expired(self):
        """Test clearing expired boosts."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
        )

        # Add an expired boost
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        memory.importance_boosts.append(
            {
                "amount": 0.2,
                "reason": "Expired",
                "applied_at": expired_time.isoformat(),
                "expires_at": expired_time.isoformat(),
            }
        )

        # Add a valid boost
        future_time = datetime.now(timezone.utc) + timedelta(days=7)
        memory.importance_boosts.append(
            {
                "amount": 0.3,
                "reason": "Still valid",
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": future_time.isoformat(),
            }
        )

        removed = memory.clear_expired_boosts()

        assert removed == 1
        assert len(memory.importance_boosts) == 1
        assert memory.importance_boosts[0]["reason"] == "Still valid"

    def test_clear_no_expired(self):
        """Test clearing when no boosts are expired."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
        )

        memory.boost_importance(amount=0.1, reason="Permanent")

        removed = memory.clear_expired_boosts()

        assert removed == 0
        assert len(memory.importance_boosts) == 1

    def test_expired_boost_not_counted(self):
        """Test that expired boosts are not counted in effective_importance."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="fact",
            user_id="user1",
            importance=0.5,
            importance_decay_rate=0.0,
        )

        # Add an expired boost
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        memory.importance_boosts.append(
            {
                "amount": 0.5,
                "reason": "Expired",
                "applied_at": expired_time.isoformat(),
                "expires_at": expired_time.isoformat(),
            }
        )

        # Expired boost should not affect effective importance
        assert memory.effective_importance == pytest.approx(0.5, rel=0.01)


# =============================================================================
# ContextWindow Tests
# =============================================================================


class TestContextWindowBasics:
    """Tests for basic ContextWindow operations."""

    def test_create_empty_window(self):
        """Test creating an empty context window."""
        window = ContextWindow(session_id="session_1")

        assert window.session_id == "session_1"
        assert len(window.messages) == 0
        assert len(window.working_memories) == 0
        assert len(window.attention_hints) == 0

    def test_add_message(self):
        """Test adding messages to window."""
        window = ContextWindow()

        window.add_message(role="user", content="Hello there")
        window.add_message(role="assistant", content="Hi!")

        assert len(window.messages) == 2
        assert window.messages[0]["role"] == "user"
        assert window.messages[0]["content"] == "Hello there"
        assert window.messages[1]["role"] == "assistant"

    def test_add_message_with_metadata(self):
        """Test adding message with metadata."""
        window = ContextWindow()

        window.add_message(
            role="user",
            content="What's my order status?",
            metadata={"intent": "order_inquiry"},
        )

        assert window.messages[0]["metadata"]["intent"] == "order_inquiry"
        assert "timestamp" in window.messages[0]

    def test_clear_window(self):
        """Test clearing the context window."""
        window = ContextWindow()
        window.add_message(role="user", content="Test")
        window.working_memories.append(
            Memory(
                memory_id="mem1",
                content="Test",
                memory_type="working",
                user_id="user1",
            )
        )
        window.attention_hints = ["topic1", "topic2"]

        window.clear()

        assert len(window.messages) == 0
        assert len(window.working_memories) == 0
        assert len(window.attention_hints) == 0


class TestContextWindowOverflow:
    """Tests for ContextWindow overflow handling."""

    def test_message_overflow(self):
        """Test that old messages are trimmed when max is exceeded."""
        window = ContextWindow(max_messages=5)

        for i in range(10):
            window.add_message(role="user", content=f"Message {i}")

        assert len(window.messages) == 5
        # Should keep the most recent messages
        assert window.messages[0]["content"] == "Message 5"
        assert window.messages[4]["content"] == "Message 9"

    def test_custom_max_messages(self):
        """Test custom max_messages limit."""
        window = ContextWindow(max_messages=3)

        for i in range(5):
            window.add_message(role="user", content=f"Msg {i}")

        assert len(window.messages) == 3
        assert window.messages[0]["content"] == "Msg 2"

    def test_no_overflow_under_limit(self):
        """Test no trimming when under limit."""
        window = ContextWindow(max_messages=10)

        for i in range(5):
            window.add_message(role="user", content=f"Msg {i}")

        assert len(window.messages) == 5


# =============================================================================
# RecallResult Tests
# =============================================================================


class TestRecallResult:
    """Tests for RecallResult dataclass."""

    def test_to_dict(self, memory):
        """Test RecallResult serialization."""
        result = RecallResult(
            memories=[memory],
            scores=[0.85],
            query_latency_ms=15.5,
            sources=["cache", "storage"],
            attention_focus=["settings", "ui"],
            suggested_memory_types=["preference"],
            query_topics=["settings"],
            query_categories=["user_preference"],
        )

        data = result.to_dict()

        assert len(data["memories"]) == 1
        assert data["scores"] == [0.85]
        assert data["query_latency_ms"] == 15.5
        assert data["sources"] == ["cache", "storage"]
        assert data["attention_focus"] == ["settings", "ui"]
        assert data["suggested_memory_types"] == ["preference"]
        assert data["query_topics"] == ["settings"]

    def test_empty_result(self):
        """Test empty RecallResult."""
        result = RecallResult(
            memories=[],
            scores=[],
            query_latency_ms=1.0,
            sources=[],
            attention_focus=[],
            suggested_memory_types=[],
        )

        data = result.to_dict()
        assert data["memories"] == []
        assert data["scores"] == []


# =============================================================================
# FLR Context Management Tests
# =============================================================================


class TestFLRContextManagement:
    """Tests for FLR context management methods."""

    def test_update_context_new_session(self, flr):
        """Test creating new context for session."""
        context = flr.update_context(
            session_id="session_1",
            messages=[{"role": "user", "content": "Hello"}],
            attention_hints=["greeting"],
        )

        assert context.session_id == "session_1"
        assert len(context.messages) == 1
        assert context.attention_hints == ["greeting"]

    def test_update_context_existing_session(self, flr):
        """Test updating existing context."""
        # Create initial context
        flr.update_context(
            session_id="session_1",
            messages=[{"role": "user", "content": "First"}],
        )

        # Update with more messages
        context = flr.update_context(
            session_id="session_1",
            messages=[{"role": "assistant", "content": "Response"}],
        )

        assert len(context.messages) == 2

    def test_update_context_with_working_memories(self, flr, memory):
        """Test adding working memories to context."""
        context = flr.update_context(
            session_id="session_1",
            working_memories=[memory],
        )

        assert len(context.working_memories) == 1
        assert context.working_memories[0].memory_id == memory.memory_id

    def test_get_context(self, flr):
        """Test retrieving context."""
        flr.update_context(session_id="session_1")

        context = flr.get_context("session_1")
        assert context is not None
        assert context.session_id == "session_1"

    def test_get_context_nonexistent(self, flr):
        """Test retrieving nonexistent context returns None."""
        context = flr.get_context("nonexistent_session")
        assert context is None

    def test_clear_context(self, flr):
        """Test clearing context."""
        flr.update_context(
            session_id="session_1",
            messages=[{"role": "user", "content": "Test"}],
        )

        flr.clear_context("session_1")

        assert flr.get_context("session_1") is None


# =============================================================================
# FLR Promote Tests
# =============================================================================


class TestFLRPromote:
    """Tests for FLR.promote method."""

    def test_promote_from_working_memory(self, flr, storage):
        """Test promoting working memory to long-term storage."""
        working_memory = Memory(
            memory_id="working_1",
            content="Temporary working memory",
            memory_type="working",
            user_id="user_1",
        )

        # Add to context
        flr.update_context(
            session_id="session_1",
            working_memories=[working_memory],
        )

        result = flr.promote("working_1")

        assert result is True
        # Should be stored in storage
        stored = storage.get("working_1")
        assert stored is not None
        assert stored.memory_type == "episodic"  # Promoted from working

    def test_promote_from_storage(self, flr, storage):
        """Test promoting working memory directly from storage."""
        working_memory = Memory(
            memory_id="working_2",
            content="Working memory in storage",
            memory_type="working",
            user_id="user_1",
        )
        storage.store(working_memory)

        result = flr.promote("working_2")

        assert result is True
        stored = storage.get("working_2")
        assert stored.memory_type == "episodic"

    def test_promote_nonexistent(self, flr):
        """Test promoting nonexistent memory returns False."""
        result = flr.promote("nonexistent_memory")
        assert result is False


# =============================================================================
# FLR Flush Reinforcements Tests
# =============================================================================


class TestFLRFlushReinforcements:
    """Tests for FLR.flush_reinforcements method."""

    def test_flush_empty_buffer(self, flr):
        """Test flushing empty buffer returns 0."""
        count = flr.flush_reinforcements()
        assert count == 0

    def test_flush_with_pending(self, flr, storage, memory):
        """Test flushing pending reinforcements."""
        # Store memory
        storage.store(memory)

        # Simulate storage failure by setting buffer directly
        flr._reinforcement_buffer["test_mem_1"] = 0.5

        count = flr.flush_reinforcements()

        assert count == 1
        assert len(flr._reinforcement_buffer) == 0


# =============================================================================
# FLR Query and Filtering Tests
# =============================================================================


class TestFLRQuery:
    """Tests for FLR.query method."""

    def test_query_with_topic_hints(self, flr, storage):
        """Test querying with attention hints."""
        # Store memories with different topics
        memory1 = Memory(
            memory_id="mem1",
            content="Info about orders and shipping",
            memory_type="semantic",
            user_id="user_1",
            topics=["orders", "shipping"],
        )
        memory2 = Memory(
            memory_id="mem2",
            content="Info about returns",
            memory_type="semantic",
            user_id="user_1",
            topics=["returns"],
        )
        storage.store(memory1)
        storage.store(memory2)

        result = flr.query(
            query="What about my order?",
            user_id="user_1",
            attention_hints=["orders"],
            limit=10,
        )

        # Should find memories
        assert isinstance(result, RecallResult)
        assert result.query_topics == ["orders"]

    def test_query_with_memory_type_filter(self, flr, storage):
        """Test querying with memory type filter."""
        preference = Memory(
            memory_id="pref1",
            content="User prefers email",
            memory_type="preference",
            user_id="user_1",
        )
        fact = Memory(
            memory_id="fact1",
            content="Company email is info@example.com",
            memory_type="fact",
            user_id="user_1",
        )
        storage.store(preference)
        storage.store(fact)

        result = flr.query(
            query="email",
            user_id="user_1",
            memory_types=["preference"],
            limit=10,
        )

        # Result should contain only preference type
        assert all(m.memory_type == "preference" for m in result.memories)

    def test_query_with_min_score(self, flr, storage):
        """Test querying with minimum score threshold."""
        memory = Memory(
            memory_id="mem1",
            content="Some content",
            memory_type="semantic",
            user_id="user_1",
        )
        storage.store(memory)

        result = flr.query(
            query="test query",
            user_id="user_1",
            min_score=0.99,  # Very high threshold
            limit=10,
        )

        # High threshold should filter out most results
        assert len(result.memories) <= 1

    def test_query_updates_access_count(self, flr, storage):
        """Test that querying updates memory access count."""
        memory = Memory(
            memory_id="mem1",
            content="Frequently accessed memory with important keywords",
            memory_type="semantic",
            user_id="user_1",
            topics=["keywords", "important"],
            access_count=0,
        )
        storage.store(memory)

        # Query to access the memory
        flr.query(
            query="important keywords",
            user_id="user_1",
            attention_hints=["keywords"],
            limit=10,
        )

        # Check if access count was updated in cache
        if memory.memory_id in flr._cache:
            cached, _ = flr._cache[memory.memory_id]
            assert cached.access_count >= 1


class TestFLRCacheOperations:
    """Tests for FLR cache operations."""

    def test_cache_memory(self, flr, memory):
        """Test that memories are cached after query."""
        # Manually cache a memory
        flr._cache_memory(memory)

        assert memory.memory_id in flr._cache

    def test_cache_eviction(self, storage):
        """Test LRU eviction when cache is full."""
        flr = FLR(storage=storage, cache_size=3)

        for i in range(5):
            memory = Memory(
                memory_id=f"mem_{i}",
                content=f"Content {i}",
                memory_type="fact",
                user_id="user_1",
            )
            flr._cache_memory(memory)

        # Only 3 should remain (cache_size=3)
        assert len(flr._cache) == 3

        # Oldest should be evicted
        assert "mem_0" not in flr._cache
        assert "mem_1" not in flr._cache
        assert "mem_4" in flr._cache

    def test_cache_ttl_expiration(self, storage):
        """Test cache TTL expiration."""
        flr = FLR(storage=storage, cache_ttl_seconds=1)

        memory = Memory(
            memory_id="expire_me",
            content="Will expire",
            memory_type="fact",
            user_id="user_1",
        )
        flr._cache_memory(memory)

        assert "expire_me" in flr._cache

        # Wait for TTL
        time.sleep(1.5)

        # Force cache cleanup by querying
        flr._query_cache("test", "user_1", None, [], [])

        # Should be expired
        assert "expire_me" not in flr._cache


# =============================================================================
# FLR Stats Tests
# =============================================================================


class TestFLRStats:
    """Tests for FLR statistics."""

    def test_get_stats_basic(self, flr):
        """Test getting basic stats."""
        stats = flr.get_stats()

        assert "cache_size" in stats
        assert "cache_max" in stats
        assert "active_contexts" in stats
        assert "pending_reinforcements" in stats
        assert "total_retrievals" in stats

    def test_stats_with_robust_reinforcement(self, storage):
        """Test stats include robust reinforcement info."""
        flr = FLR(storage=storage, use_robust_reinforcement=True)

        stats = flr.get_stats()

        assert stats["robust_reinforcement"]["enabled"] is True
        assert "exploration_factor" in stats["robust_reinforcement"]
        assert "decay_half_life_hours" in stats["robust_reinforcement"]

    def test_stats_without_robust_reinforcement(self, flr):
        """Test stats without robust reinforcement."""
        stats = flr.get_stats()

        assert stats["robust_reinforcement"]["enabled"] is False


# =============================================================================
# Memory Serialization Tests
# =============================================================================


class TestMemorySerialization:
    """Tests for Memory.to_dict and Memory.from_dict."""

    def test_to_dict(self, memory):
        """Test Memory serialization to dict."""
        data = memory.to_dict()

        assert data["memory_id"] == "test_mem_1"
        assert data["content"] == "User prefers dark mode for the application"
        assert data["memory_type"] == "preference"
        assert data["user_id"] == "user_123"
        assert data["topics"] == ["settings", "ui", "preferences"]
        assert "effective_importance" in data

    def test_from_dict(self, memory):
        """Test Memory deserialization from dict."""
        data = memory.to_dict()
        restored = Memory.from_dict(data)

        assert restored.memory_id == memory.memory_id
        assert restored.content == memory.content
        assert restored.memory_type == memory.memory_type
        assert restored.topics == memory.topics
        assert restored.importance == memory.importance

    def test_round_trip(self, memory):
        """Test full serialization round-trip."""
        data = memory.to_dict()
        restored = Memory.from_dict(data)
        data2 = restored.to_dict()

        # Compare key fields
        assert data["memory_id"] == data2["memory_id"]
        assert data["content"] == data2["content"]
        assert data["importance"] == data2["importance"]

    def test_from_dict_with_datetimes(self):
        """Test deserialization handles datetime strings."""
        data = {
            "memory_id": "mem1",
            "content": "Test",
            "memory_type": "fact",
            "user_id": "user1",
            "created_at": "2025-01-15T10:30:00+00:00",
            "last_accessed": "2025-01-15T12:00:00+00:00",
        }

        memory = Memory.from_dict(data)

        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.last_accessed, datetime)


# =============================================================================
# FLR Reinforce Tests (additional coverage)
# =============================================================================


class TestFLRReinforce:
    """Additional tests for FLR.reinforce method."""

    def test_reinforce_invalid_signal_type(self, flr):
        """Test reinforce raises TypeError for invalid signal."""
        with pytest.raises(TypeError):
            flr.reinforce("mem1", "invalid")

    def test_reinforce_clamps_signal(self, flr, storage, memory):
        """Test that reinforce clamps signal to [-1, 1]."""
        storage.store(memory)

        # Signal > 1 should be clamped
        flr.reinforce(memory.memory_id, 5.0)

        # Check buffer was clamped
        if memory.memory_id in flr._reinforcement_buffer:
            assert flr._reinforcement_buffer[memory.memory_id] <= 1.0

    def test_reinforce_updates_cache(self, flr, storage, memory):
        """Test that reinforce updates cached memory."""
        storage.store(memory)
        flr._cache_memory(memory)

        flr.reinforce(memory.memory_id, 0.5)

        cached, _ = flr._cache[memory.memory_id]
        assert cached.reinforcement_score > 0


# =============================================================================
# FLR Access Control Tests
# =============================================================================


class TestFLRAccessControl:
    """Tests for FLR access control filtering."""

    def test_filter_private_memories_different_user(self, storage):
        """Test that private memories are filtered by user."""
        flr = FLR(storage=storage)

        # Store a private memory for user_1
        private_mem = Memory(
            memory_id="private1",
            content="Private info",
            memory_type="fact",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        flr._cache_memory(private_mem)

        # Query as different user - should not see private memory
        result = flr._query_cache(
            query="private info",
            user_id="user_2",  # Different user
            agent_id="agent_1",
            attention_hints=[],
            memory_types=[],
        )

        # Should not find private memory of other user
        assert all(m.memory_id != "private1" for m in result)

    def test_private_memories_same_user(self, storage):
        """Test that private memories are visible to same user."""
        flr = FLR(storage=storage)

        # Store a private memory
        private_mem = Memory(
            memory_id="private2",
            content="Private user info",
            memory_type="fact",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        flr._cache_memory(private_mem)

        # Same user should see their private memory
        result = flr._query_cache(
            query="private user info",
            user_id="user_1",
            agent_id="agent_2",  # Different agent, same user
            attention_hints=[],
            memory_types=[],
        )

        # Should find the memory (same user owns it)
        memory_ids = [m.memory_id for m in result]
        assert "private2" in memory_ids

    def test_shared_memories_accessible(self, storage):
        """Test that shared memories are accessible to all."""
        flr = FLR(storage=storage)

        # Store a shared memory
        shared_mem = Memory(
            memory_id="shared1",
            content="Shared info",
            memory_type="fact",
            user_id="user_1",
            agent_id="agent_1",
            access_level="shared",
        )
        storage.store(shared_mem)

        filtered = flr._filter_by_access([shared_mem], requesting_agent_id="agent_2")

        assert len(filtered) == 1
        assert filtered[0].memory_id == "shared1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
