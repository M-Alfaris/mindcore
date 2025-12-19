"""Comprehensive tests for FLR (Fast Learning Recall) module."""

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest

from mindcore.v2.flr import FLR, Memory, RecallResult, ContextWindow
from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.access import AccessController, Permission


class TestMemory:
    """Test Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a Memory object."""
        memory = Memory(
            memory_id="mem_123",
            content="Test memory content",
            memory_type="episodic",
            user_id="user_456",
        )

        assert memory.memory_id == "mem_123"
        assert memory.content == "Test memory content"
        assert memory.memory_type == "episodic"
        assert memory.user_id == "user_456"
        assert memory.importance == 0.5  # default
        assert memory.access_level == "private"  # default

    def test_memory_with_metadata(self):
        """Test Memory with full metadata."""
        memory = Memory(
            memory_id="mem_123",
            content="Test content",
            memory_type="preference",
            user_id="user_456",
            agent_id="agent_789",
            topics=["settings", "ui"],
            categories=["user_preference"],
            sentiment="positive",
            importance=0.9,
            entities=["dark_mode"],
            access_level="team",
        )

        assert memory.agent_id == "agent_789"
        assert "settings" in memory.topics
        assert memory.sentiment == "positive"
        assert memory.importance == 0.9

    def test_memory_to_dict(self):
        """Test converting Memory to dictionary."""
        memory = Memory(
            memory_id="mem_123",
            content="Test content",
            memory_type="semantic",
            user_id="user_456",
            topics=["topic1"],
        )

        result = memory.to_dict()

        assert isinstance(result, dict)
        assert result["memory_id"] == "mem_123"
        assert result["content"] == "Test content"
        assert result["topics"] == ["topic1"]

    def test_memory_from_dict(self):
        """Test creating Memory from dictionary."""
        data = {
            "memory_id": "mem_123",
            "content": "Test content",
            "memory_type": "procedural",
            "user_id": "user_456",
            "topics": ["coding"],
            "importance": 0.7,
        }

        memory = Memory.from_dict(data)

        assert memory.memory_id == "mem_123"
        assert memory.memory_type == "procedural"
        assert memory.topics == ["coding"]
        assert memory.importance == 0.7


class TestContextWindow:
    """Test ContextWindow for active context management."""

    def test_context_window_creation(self):
        """Test creating a context window."""
        ctx = ContextWindow(session_id="session_123")

        assert ctx.session_id == "session_123"
        assert ctx.messages == []
        assert ctx.working_memories == []

    def test_add_message(self):
        """Test adding messages to context window."""
        ctx = ContextWindow()

        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there!")

        assert len(ctx.messages) == 2
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[1]["content"] == "Hi there!"

    def test_message_limit(self):
        """Test that messages are trimmed when over limit."""
        ctx = ContextWindow(max_messages=5)

        for i in range(10):
            ctx.add_message("user", f"Message {i}")

        assert len(ctx.messages) == 5
        assert ctx.messages[0]["content"] == "Message 5"

    def test_clear(self):
        """Test clearing context window."""
        ctx = ContextWindow()
        ctx.add_message("user", "Hello")
        ctx.attention_hints = ["topic1"]

        ctx.clear()

        assert ctx.messages == []
        assert ctx.attention_hints == []


class TestFLRBasics:
    """Test basic FLR functionality."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_flr_initialization(self, storage):
        """Test FLR initialization."""
        flr = FLR(
            storage=storage,
            cache_size=500,
            cache_ttl_seconds=120,
        )

        assert flr.cache_size == 500
        assert flr.cache_ttl == 120
        assert flr.access_controller is None

    def test_flr_with_access_controller(self, storage):
        """Test FLR with access controller."""
        ac = AccessController()
        flr = FLR(storage=storage, access_controller=ac)

        assert flr.access_controller is ac

    def test_store_to_cache(self, flr):
        """Test storing memory to hot cache."""
        memory = Memory(
            memory_id="mem_123",
            content="Test memory",
            memory_type="episodic",
            user_id="user_456",
        )

        flr._cache_memory(memory)

        assert "mem_123" in flr._cache

    def test_query_empty(self, flr):
        """Test querying with no memories."""
        result = flr.query(
            query="test query",
            user_id="user_123",
        )

        assert isinstance(result, RecallResult)
        assert result.memories == []
        assert result.scores == []

    def test_query_with_memories(self, flr, storage):
        """Test querying with stored memories."""
        # Store some memories
        memory = Memory(
            memory_id="mem_1",
            content="User prefers dark mode",
            memory_type="preference",
            user_id="user_123",
            topics=["settings"],
        )
        storage.store(memory)

        result = flr.query(
            query="dark mode preferences",
            user_id="user_123",
        )

        assert len(result.memories) > 0

    def test_query_respects_user_id(self, flr, storage):
        """Test that query respects user_id filter."""
        # Store memory for different users
        mem1 = Memory(
            memory_id="mem_1",
            content="Memory for user 1",
            memory_type="episodic",
            user_id="user_1",
        )
        mem2 = Memory(
            memory_id="mem_2",
            content="Memory for user 2",
            memory_type="episodic",
            user_id="user_2",
        )
        storage.store(mem1)
        storage.store(mem2)

        result = flr.query(query="memory", user_id="user_1")

        # Should only return user_1's memory
        for m in result.memories:
            assert m.user_id == "user_1"


class TestFLRScoring:
    """Test FLR memory scoring."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_topic_matching_boosts_score(self, flr, storage):
        """Test that matching topics boost relevance score."""
        # Store memory with topic
        memory = Memory(
            memory_id="mem_1",
            content="Information about billing",
            memory_type="semantic",
            user_id="user_123",
            topics=["billing"],
        )
        storage.store(memory)

        # Query with matching attention hint
        result = flr.query(
            query="billing information",
            user_id="user_123",
            attention_hints=["billing"],
        )

        assert len(result.memories) > 0
        # Score should be higher due to topic match
        assert result.scores[0] > 0

    def test_importance_affects_score(self, flr, storage):
        """Test that importance affects scoring."""
        # Store memories with different importance
        high_importance = Memory(
            memory_id="mem_high",
            content="Important memory",
            memory_type="semantic",
            user_id="user_123",
            importance=0.9,
        )
        low_importance = Memory(
            memory_id="mem_low",
            content="Important memory",  # Same content
            memory_type="semantic",
            user_id="user_123",
            importance=0.1,
        )
        storage.store(high_importance)
        storage.store(low_importance)

        result = flr.query(query="important", user_id="user_123")

        # Both should be returned but high importance should score higher
        if len(result.memories) >= 2:
            high_idx = next(i for i, m in enumerate(result.memories)
                            if m.memory_id == "mem_high")
            low_idx = next(i for i, m in enumerate(result.memories)
                           if m.memory_id == "mem_low")
            # Higher score should come first (memories are sorted by score)
            assert high_idx < low_idx or result.scores[high_idx] >= result.scores[low_idx]


class TestFLRReinforcement:
    """Test FLR reinforcement learning."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_reinforce_positive(self, flr, storage):
        """Test positive reinforcement."""
        memory = Memory(
            memory_id="mem_123",
            content="Useful memory",
            memory_type="semantic",
            user_id="user_456",
        )
        storage.store(memory)
        flr._cache_memory(memory)

        # Reinforce positively
        flr.reinforce("mem_123", signal=1.0)

        # Check buffer has the reinforcement
        assert "mem_123" in flr._reinforcement_buffer
        assert flr._reinforcement_buffer["mem_123"] == 1.0

    def test_reinforce_negative(self, flr, storage):
        """Test negative reinforcement."""
        memory = Memory(
            memory_id="mem_123",
            content="Not useful memory",
            memory_type="semantic",
            user_id="user_456",
        )
        storage.store(memory)

        flr.reinforce("mem_123", signal=-0.5)

        assert flr._reinforcement_buffer["mem_123"] == -0.5

    def test_reinforcement_accumulates(self, flr, storage):
        """Test that reinforcement signals accumulate."""
        memory = Memory(
            memory_id="mem_123",
            content="Memory",
            memory_type="semantic",
            user_id="user_456",
        )
        storage.store(memory)

        flr.reinforce("mem_123", signal=0.3)
        flr.reinforce("mem_123", signal=0.4)

        assert abs(flr._reinforcement_buffer["mem_123"] - 0.7) < 0.01


class TestFLRTeamAccess:
    """Test FLR team-based access control."""

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
    def access_controller(self):
        """Create access controller with agents."""
        ac = AccessController()

        # Register agents in same team
        ac.register_agent("agent_1", "Agent 1", teams=["support_team"])
        ac.register_agent("agent_2", "Agent 2", teams=["support_team"])

        # Register agent in different team
        ac.register_agent("agent_3", "Agent 3", teams=["sales_team"])

        return ac

    @pytest.fixture
    def flr(self, storage, access_controller):
        """Create FLR with access controller."""
        return FLR(storage=storage, access_controller=access_controller)

    def test_check_team_access_same_team(self, flr):
        """Test team access check for agents in same team."""
        result = flr._check_team_access("agent_1", "agent_2")
        assert result is True

    def test_check_team_access_different_team(self, flr):
        """Test team access check for agents in different teams."""
        result = flr._check_team_access("agent_1", "agent_3")
        assert result is False

    def test_check_team_access_no_controller(self, storage):
        """Test team access with no access controller."""
        flr = FLR(storage=storage)  # No access controller

        result = flr._check_team_access("agent_1", "agent_2")
        assert result is False

    def test_check_team_access_none_agent(self, flr):
        """Test team access with None agent IDs."""
        assert flr._check_team_access(None, "agent_1") is False
        assert flr._check_team_access("agent_1", None) is False

    def test_check_team_access_unknown_agent(self, flr):
        """Test team access with unknown agent."""
        result = flr._check_team_access("agent_1", "unknown_agent")
        assert result is False


class TestFLRCaching:
    """Test FLR hot cache functionality."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()
        os.unlink(db_path)

    def test_cache_size_limit(self, storage):
        """Test that cache respects size limit."""
        flr = FLR(storage=storage, cache_size=3)

        # Add more memories than cache size
        for i in range(5):
            memory = Memory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                memory_type="episodic",
                user_id="user_123",
            )
            flr._cache_memory(memory)

        # Only last 3 should remain (LRU eviction)
        assert len(flr._cache) <= 3

    def test_cache_ttl_expiration(self, storage):
        """Test that cache entries expire."""
        flr = FLR(storage=storage, cache_ttl_seconds=1)

        memory = Memory(
            memory_id="mem_123",
            content="Test memory",
            memory_type="episodic",
            user_id="user_123",
        )
        flr._cache_memory(memory)

        # Wait for expiration
        time.sleep(1.5)

        # Query should trigger cache cleanup
        flr._query_cache("test", "user_123", None, [], [])

        # Memory should be expired
        assert "mem_123" not in flr._cache

    def test_cache_memory_update_timestamp(self, storage):
        """Test that caching updates timestamp."""
        flr = FLR(storage=storage)

        memory = Memory(
            memory_id="mem_123",
            content="Test memory",
            memory_type="episodic",
            user_id="user_123",
        )

        flr._cache_memory(memory)

        # Check timestamp is recent
        _, timestamp = flr._cache["mem_123"]
        assert time.time() - timestamp < 1


class TestFLRStats:
    """Test FLR statistics."""

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
        """Test getting FLR statistics."""
        flr = FLR(storage=storage)

        # Add some cached memories
        for i in range(3):
            memory = Memory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                memory_type="episodic",
                user_id="user_123",
            )
            flr._cache_memory(memory)

        stats = flr.get_stats()

        assert "cache_size" in stats
        assert "cache_max" in stats
        assert "active_contexts" in stats
        assert stats["cache_size"] == 3


class TestFLRContextManagement:
    """Test FLR context window management."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_update_context_creates_new(self, flr):
        """Test that update_context creates new context."""
        ctx = flr.update_context(
            session_id="session_123",
            attention_hints=["topic1"],
        )

        assert ctx is not None
        assert ctx.session_id == "session_123"
        assert ctx.attention_hints == ["topic1"]

    def test_update_context_adds_messages(self, flr):
        """Test adding messages to context."""
        flr.update_context(
            session_id="session_123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )

        ctx = flr.get_context("session_123")
        assert len(ctx.messages) == 2
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[1]["content"] == "Hi there!"

    def test_update_context_adds_working_memories(self, flr):
        """Test adding working memories to context."""
        memory = Memory(
            memory_id="mem_work",
            content="Working memory",
            memory_type="working",
            user_id="user_123",
        )

        flr.update_context(
            session_id="session_123",
            working_memories=[memory],
        )

        ctx = flr.get_context("session_123")
        assert len(ctx.working_memories) == 1
        assert ctx.working_memories[0].memory_id == "mem_work"

    def test_get_context(self, flr):
        """Test getting existing context."""
        flr.update_context(session_id="session_123")

        ctx = flr.get_context("session_123")

        assert ctx is not None
        assert ctx.session_id == "session_123"

    def test_get_context_nonexistent(self, flr):
        """Test getting non-existent context."""
        ctx = flr.get_context("nonexistent")
        assert ctx is None

    def test_clear_context(self, flr):
        """Test clearing context."""
        flr.update_context(
            session_id="session_123",
            messages=[{"role": "user", "content": "Hello"}],
            attention_hints=["topic1"],
        )

        flr.clear_context("session_123")

        ctx = flr.get_context("session_123")
        assert ctx.messages == []
        assert ctx.attention_hints == []

    def test_clear_context_nonexistent(self, flr):
        """Test clearing non-existent context (should not error)."""
        # Should not raise
        flr.clear_context("nonexistent")


class TestFLRPromote:
    """Test FLR memory promotion."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_promote_working_memory(self, flr, storage):
        """Test promoting a working memory to long-term storage."""
        # Create working memory in context
        memory = Memory(
            memory_id="mem_work",
            content="Promote me",
            memory_type="working",
            user_id="user_123",
        )

        flr.update_context(
            session_id="session_123",
            working_memories=[memory],
        )

        # Promote
        result = flr.promote("mem_work")

        assert result is True

        # Check it's in storage
        stored = storage.get("mem_work")
        assert stored is not None
        assert stored.memory_type == "episodic"  # Promoted to episodic

        # Check it's removed from working memories
        ctx = flr.get_context("session_123")
        assert len(ctx.working_memories) == 0

    def test_promote_nonexistent(self, flr):
        """Test promoting non-existent memory."""
        result = flr.promote("nonexistent")
        assert result is False


class TestFLRFlushReinforcements:
    """Test FLR reinforcement flushing."""

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
    def flr(self, storage):
        """Create FLR instance."""
        return FLR(storage=storage)

    def test_flush_reinforcements(self, flr, storage):
        """Test flushing reinforcements to storage."""
        # Store memory first
        memory = Memory(
            memory_id="mem_123",
            content="Test",
            memory_type="episodic",
            user_id="user_123",
        )
        storage.store(memory)

        # Add reinforcement signals
        flr.reinforce("mem_123", signal=0.5)
        flr.reinforce("mem_123", signal=0.3)

        # Flush
        count = flr.flush_reinforcements()

        assert count == 1

        # Verify in storage
        stored = storage.get("mem_123")
        assert abs(stored.reinforcement_score - 0.8) < 0.01

        # Buffer should be cleared
        assert len(flr._reinforcement_buffer) == 0

    def test_flush_empty_buffer(self, flr):
        """Test flushing empty buffer."""
        count = flr.flush_reinforcements()
        assert count == 0

    def test_flush_multiple_memories(self, flr, storage):
        """Test flushing multiple memory reinforcements."""
        # Store memories
        for i in range(3):
            memory = Memory(
                memory_id=f"mem_{i}",
                content=f"Test {i}",
                memory_type="episodic",
                user_id="user_123",
            )
            storage.store(memory)
            flr.reinforce(f"mem_{i}", signal=0.1 * (i + 1))

        count = flr.flush_reinforcements()

        assert count == 3


class TestFLRCacheAccessControl:
    """Test FLR cache access control."""

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
    def access_controller(self):
        """Create access controller with agents."""
        ac = AccessController()
        ac.register_agent("agent_1", "Agent 1", teams=["team_a"])
        ac.register_agent("agent_2", "Agent 2", teams=["team_a"])
        ac.register_agent("agent_3", "Agent 3", teams=["team_b"])
        return ac

    @pytest.fixture
    def flr(self, storage, access_controller):
        """Create FLR with access controller."""
        return FLR(storage=storage, access_controller=access_controller)

    def test_cache_private_memory_own(self, flr):
        """Test accessing own private memory from cache."""
        memory = Memory(
            memory_id="mem_private",
            content="Private memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        flr._cache_memory(memory)

        # Query as the same user
        result = flr._query_cache(
            query="private",
            user_id="user_1",
            agent_id="agent_1",
            attention_hints=[],
            memory_types=[],
        )

        assert len(result) >= 1

    def test_cache_private_memory_other_user(self, flr):
        """Test that other users can't access private memory from cache."""
        memory = Memory(
            memory_id="mem_private",
            content="Private memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        flr._cache_memory(memory)

        # Query as different user
        result = flr._query_cache(
            query="private",
            user_id="user_2",
            agent_id="agent_2",
            attention_hints=[],
            memory_types=[],
        )

        assert len(result) == 0

    def test_cache_team_memory_same_team(self, flr):
        """Test accessing team memory from same team."""
        memory = Memory(
            memory_id="mem_team",
            content="Team memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="team",
        )
        flr._cache_memory(memory)

        # Query as different user but same team agent
        result = flr._query_cache(
            query="team",
            user_id="user_1",  # Same user to pass user check
            agent_id="agent_2",  # Different agent but same team
            attention_hints=[],
            memory_types=[],
        )

        assert len(result) >= 1

    def test_cache_team_memory_different_team(self, flr):
        """Test that different team can't access team memory."""
        memory = Memory(
            memory_id="mem_team",
            content="Team memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",  # team_a
            access_level="team",
        )
        flr._cache_memory(memory)

        # Query as different team
        result = flr._query_cache(
            query="team",
            user_id="user_2",
            agent_id="agent_3",  # team_b
            attention_hints=[],
            memory_types=[],
        )

        assert len(result) == 0

    def test_cache_memory_type_filter(self, flr):
        """Test cache query with memory type filter."""
        mem1 = Memory(
            memory_id="mem_1",
            content="Episodic memory",
            memory_type="episodic",
            user_id="user_1",
        )
        mem2 = Memory(
            memory_id="mem_2",
            content="Preference memory",
            memory_type="preference",
            user_id="user_1",
        )
        flr._cache_memory(mem1)
        flr._cache_memory(mem2)

        result = flr._query_cache(
            query="memory",
            user_id="user_1",
            agent_id=None,
            attention_hints=[],
            memory_types=["preference"],
        )

        assert len(result) == 1
        assert result[0].memory_type == "preference"
