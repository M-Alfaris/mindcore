"""Comprehensive tests for cross-agent memory sharing."""

import os
import tempfile

import pytest

from mindcore.v2.cross_agent.sharing import (
    CrossAgentMemory,
    ShareResult,
    SyncResult,
    SyncDirection,
)
from mindcore.v2.cross_agent.registry import Agent, AgentRegistry
from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.flr import Memory


class TestShareResult:
    """Test ShareResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful share result."""
        result = ShareResult(
            success=True,
            memory_id="mem_123",
            source_agent="agent_1",
            target_agents=["agent_2", "agent_3"],
            access_level="team",
        )

        assert result.success is True
        assert result.memory_id == "mem_123"
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating a failed share result."""
        result = ShareResult(
            success=False,
            memory_id="mem_123",
            source_agent="agent_1",
            target_agents=[],
            access_level="team",
            error="Memory not found",
        )

        assert result.success is False
        assert result.error == "Memory not found"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ShareResult(
            success=True,
            memory_id="mem_123",
            source_agent="agent_1",
            target_agents=["agent_2"],
            access_level="shared",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["memory_id"] == "mem_123"
        assert "shared_at" in data


class TestSyncResult:
    """Test SyncResult dataclass."""

    def test_create_sync_result(self):
        """Test creating a sync result."""
        result = SyncResult(
            success=True,
            source_agent="agent_1",
            target_agent="agent_2",
            direction=SyncDirection.ONE_WAY,
            memories_synced=5,
            memories_skipped=2,
            conflicts=0,
            sync_duration_ms=123.45,
        )

        assert result.success is True
        assert result.memories_synced == 5
        assert result.conflicts == 0

    def test_sync_result_to_dict(self):
        """Test converting sync result to dict."""
        result = SyncResult(
            success=True,
            source_agent="agent_1",
            target_agent="agent_2",
            direction=SyncDirection.BIDIRECTIONAL,
            memories_synced=3,
            memories_skipped=1,
            conflicts=0,
            sync_duration_ms=50.0,
        )

        data = result.to_dict()

        assert data["direction"] == "bidirectional"
        assert data["memories_synced"] == 3
        assert "synced_at" in data


class TestCrossAgentMemoryInit:
    """Test CrossAgentMemory initialization."""

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
    def registry(self):
        """Create agent registry with agents."""
        registry = AgentRegistry()

        # Register agents in teams
        registry.register_agent(
            agent_id="support_bot",
            name="Support Bot",
            teams=["support_team"],
        )
        registry.register_agent(
            agent_id="sales_bot",
            name="Sales Bot",
            teams=["sales_team"],
        )
        registry.register_agent(
            agent_id="support_bot_2",
            name="Support Bot 2",
            teams=["support_team"],
        )

        return registry

    def test_create_cross_agent_memory(self, storage, registry):
        """Test creating CrossAgentMemory instance."""
        cam = CrossAgentMemory(storage=storage, registry=registry)

        assert cam.storage is storage
        assert cam.registry is registry
        assert cam._share_history == []
        assert cam._sync_history == []


class TestShareMemory:
    """Test share_memory functionality."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        registry.register_agent("agent_3", "Agent 3", teams=["team_b"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_share_memory_success(self, cam, storage):
        """Test successfully sharing a memory."""
        # Create memory owned by agent_1
        memory = Memory(
            memory_id="mem_123",
            content="Shareable memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        storage.store(memory)

        result = cam.share_memory(
            memory_id="mem_123",
            source_agent="agent_1",
            access_level="team",
        )

        assert result.success is True
        assert result.access_level == "team"
        # Verify memory was updated
        updated_memory = storage.get("mem_123")
        assert updated_memory.access_level == "team"

    def test_share_memory_agent_not_found(self, cam):
        """Test sharing with non-existent agent."""
        result = cam.share_memory(
            memory_id="mem_123",
            source_agent="nonexistent",
            access_level="team",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_share_memory_memory_not_found(self, cam):
        """Test sharing non-existent memory."""
        result = cam.share_memory(
            memory_id="nonexistent",
            source_agent="agent_1",
            access_level="team",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_share_memory_not_owner(self, cam, storage):
        """Test that only owner can share."""
        # Create memory owned by agent_1
        memory = Memory(
            memory_id="mem_123",
            content="Test memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
        )
        storage.store(memory)

        # Try to share as agent_2
        result = cam.share_memory(
            memory_id="mem_123",
            source_agent="agent_2",
            access_level="team",
        )

        assert result.success is False
        assert "owner" in result.error.lower()


class TestSyncAgents:
    """Test agent synchronization."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        registry.register_agent("agent_3", "Agent 3", teams=["team_b"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_sync_agents_one_way(self, cam, storage):
        """Test one-way sync between agents."""
        # Create memories for agent_1
        memory = Memory(
            memory_id="mem_1",
            content="Memory from agent 1",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="team",
        )
        storage.store(memory)

        result = cam.sync_agents(
            source_agent="agent_1",
            target_agent="agent_2",
            user_id="user_1",
            direction=SyncDirection.ONE_WAY,
        )

        assert result.success is True
        assert result.direction == SyncDirection.ONE_WAY
        assert result.sync_duration_ms > 0

    def test_sync_agents_bidirectional(self, cam, storage):
        """Test bidirectional sync."""
        # Create memories for both agents
        mem1 = Memory(
            memory_id="mem_1",
            content="From agent 1",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="team",
        )
        mem2 = Memory(
            memory_id="mem_2",
            content="From agent 2",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_2",
            access_level="team",
        )
        storage.store(mem1)
        storage.store(mem2)

        result = cam.sync_agents(
            source_agent="agent_1",
            target_agent="agent_2",
            user_id="user_1",
            direction=SyncDirection.BIDIRECTIONAL,
        )

        assert result.success is True
        assert result.direction == SyncDirection.BIDIRECTIONAL

    def test_sync_agents_source_not_found(self, cam):
        """Test sync with non-existent source agent."""
        result = cam.sync_agents(
            source_agent="nonexistent",
            target_agent="agent_2",
            user_id="user_1",
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "source" in result.errors[0].lower()

    def test_sync_agents_target_not_found(self, cam):
        """Test sync with non-existent target agent."""
        result = cam.sync_agents(
            source_agent="agent_1",
            target_agent="nonexistent",
            user_id="user_1",
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "target" in result.errors[0].lower()

    def test_sync_agents_different_teams(self, cam, storage):
        """Test sync between agents in different teams."""
        # Create memory with shared access
        memory = Memory(
            memory_id="mem_1",
            content="Shared memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="shared",
        )
        storage.store(memory)

        result = cam.sync_agents(
            source_agent="agent_1",
            target_agent="agent_3",  # Different team
            user_id="user_1",
        )

        # Should work with warning about different teams
        assert "team" in " ".join(result.errors).lower() if result.errors else True


class TestGetAccessibleMemories:
    """Test getting accessible memories."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        registry.register_agent("agent_3", "Agent 3", teams=["team_b"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_get_accessible_memories_own(self, cam, storage):
        """Test getting own memories."""
        memory = Memory(
            memory_id="mem_1",
            content="My memory",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="private",
        )
        storage.store(memory)

        memories = cam.get_accessible_memories(
            agent_id="agent_1",
            user_id="user_1",
        )

        assert len(memories) >= 1
        assert any(m.memory_id == "mem_1" for m in memories)

    def test_get_accessible_memories_inactive_agent(self, cam, registry):
        """Test that inactive agents get no memories."""
        from mindcore.v2.cross_agent.registry import AgentStatus

        agent = registry.get_agent("agent_1")
        agent.status = AgentStatus.INACTIVE

        memories = cam.get_accessible_memories(
            agent_id="agent_1",
            user_id="user_1",
        )

        assert memories == []

    def test_get_accessible_memories_nonexistent_agent(self, cam):
        """Test getting memories for non-existent agent."""
        memories = cam.get_accessible_memories(
            agent_id="nonexistent",
            user_id="user_1",
        )

        assert memories == []


class TestGetMemoryVisibility:
    """Test memory visibility checking."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_get_memory_visibility_nonexistent(self, cam):
        """Test visibility for non-existent memory."""
        visibility = cam.get_memory_visibility("nonexistent")
        assert visibility == {}


class TestHistory:
    """Test share and sync history."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_get_share_history(self, cam, storage):
        """Test getting share history."""
        # Create and share a memory
        memory = Memory(
            memory_id="mem_1",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
        )
        storage.store(memory)
        cam.share_memory("mem_1", "agent_1", "team")

        history = cam.get_share_history()

        assert len(history) >= 1

    def test_get_share_history_filtered(self, cam, storage):
        """Test getting share history filtered by agent."""
        # Create and share memories
        mem1 = Memory(
            memory_id="mem_1",
            content="Test 1",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
        )
        storage.store(mem1)
        cam.share_memory("mem_1", "agent_1", "team")

        history = cam.get_share_history(agent_id="agent_1")

        for result in history:
            assert result.source_agent == "agent_1" or "agent_1" in result.target_agents

    def test_get_sync_history(self, cam, storage):
        """Test getting sync history."""
        memory = Memory(
            memory_id="mem_1",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="team",
        )
        storage.store(memory)

        cam.sync_agents("agent_1", "agent_2", "user_1")

        history = cam.get_sync_history()

        assert len(history) >= 1

    def test_get_sync_history_filtered(self, cam, storage):
        """Test getting sync history filtered by agent."""
        cam.sync_agents("agent_1", "agent_2", "user_1")

        history = cam.get_sync_history(agent_id="agent_1")

        for result in history:
            assert result.source_agent == "agent_1" or result.target_agent == "agent_1"


class TestStats:
    """Test statistics."""

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
    def registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent("agent_1", "Agent 1", teams=["team_a"])
        registry.register_agent("agent_2", "Agent 2", teams=["team_a"])
        return registry

    @pytest.fixture
    def cam(self, storage, registry):
        """Create CrossAgentMemory instance."""
        return CrossAgentMemory(storage=storage, registry=registry)

    def test_get_stats(self, cam, storage):
        """Test getting sharing stats."""
        # Create some activity
        memory = Memory(
            memory_id="mem_1",
            content="Test",
            memory_type="episodic",
            user_id="user_1",
            agent_id="agent_1",
            access_level="team",
        )
        storage.store(memory)
        cam.share_memory("mem_1", "agent_1", "team")
        cam.sync_agents("agent_1", "agent_2", "user_1")

        stats = cam.get_stats()

        assert "total_shares" in stats
        assert "successful_shares" in stats
        assert "total_syncs" in stats
        assert "successful_syncs" in stats
        assert "total_memories_synced" in stats
        assert stats["total_shares"] >= 1
        assert stats["total_syncs"] >= 1
