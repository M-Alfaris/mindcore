"""Tests for Cross-Agent Memory Sharing.

Tests cover:
- ShareResult and SyncResult dataclasses
- CrossAgentMemory sharing operations
- Conflict resolution strategies
- Sync operations between agents
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mindcore.cross_agent.registry import Agent, AgentRegistry, AgentStatus
from mindcore.cross_agent.sharing import (
    AgentSyncDirection,
    AgentSyncResult,
    ConflictInfo,
    ConflictResolution,
    CrossAgentMemory,
    ShareResult,
    SyncDirection,
    SyncResult,
)
from mindcore.flr import Memory
from mindcore.storage.sqlite import SQLiteStorage


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage():
    """Create temporary SQLite storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    storage.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def registry():
    """Create agent registry with test agents."""
    reg = AgentRegistry()
    reg.register_agent(
        agent_id="support_bot",
        name="Support Agent",
        teams=["support", "customer_service"],
    )
    reg.register_agent(
        agent_id="sales_bot",
        name="Sales Agent",
        teams=["sales"],
    )
    reg.register_agent(
        agent_id="billing_bot",
        name="Billing Agent",
        teams=["billing", "support"],  # Shares "support" with support_bot
    )
    return reg


@pytest.fixture
def sharing(storage, registry):
    """Create CrossAgentMemory instance."""
    return CrossAgentMemory(storage=storage, registry=registry)


@pytest.fixture
def test_memory(storage):
    """Create and store a test memory."""
    memory = Memory(
        memory_id="test_mem_1",
        content="User prefers dark mode",
        memory_type="preference",
        user_id="user_123",
        agent_id="support_bot",
        topics=["settings", "ui"],
        importance=0.8,
        access_level="private",
    )
    storage.store(memory)
    return memory


# =============================================================================
# Enum and Alias Tests
# =============================================================================


class TestEnumsAndAliases:
    """Tests for enums and backwards-compatible aliases."""

    def test_sync_direction_values(self):
        """Test AgentSyncDirection enum values."""
        assert AgentSyncDirection.ONE_WAY.value == "one_way"
        assert AgentSyncDirection.BIDIRECTIONAL.value == "bidirectional"
        assert AgentSyncDirection.MERGE.value == "merge"

    def test_sync_direction_alias(self):
        """Test SyncDirection is alias for AgentSyncDirection."""
        assert SyncDirection is AgentSyncDirection

    def test_conflict_resolution_values(self):
        """Test ConflictResolution enum values."""
        assert ConflictResolution.SOURCE_WINS.value == "source_wins"
        assert ConflictResolution.TARGET_WINS.value == "target_wins"
        assert ConflictResolution.NEWEST_WINS.value == "newest_wins"
        assert ConflictResolution.HIGHEST_IMPORTANCE.value == "highest_importance"
        assert ConflictResolution.MERGE_METADATA.value == "merge_metadata"
        assert ConflictResolution.SKIP.value == "skip"


# =============================================================================
# ShareResult Tests
# =============================================================================


class TestShareResult:
    """Tests for ShareResult dataclass."""

    def test_share_result_creation(self):
        """Test creating a ShareResult."""
        result = ShareResult(
            success=True,
            memory_id="mem_1",
            source_agent="agent_a",
            target_agents=["agent_b", "agent_c"],
            access_level="shared",
        )

        assert result.success is True
        assert result.memory_id == "mem_1"
        assert result.source_agent == "agent_a"
        assert result.target_agents == ["agent_b", "agent_c"]
        assert result.access_level == "shared"
        assert result.error is None

    def test_share_result_with_error(self):
        """Test ShareResult with error."""
        result = ShareResult(
            success=False,
            memory_id="mem_1",
            source_agent="agent_a",
            target_agents=[],
            access_level="shared",
            error="Memory not found",
        )

        assert result.success is False
        assert result.error == "Memory not found"

    def test_share_result_to_dict(self):
        """Test ShareResult serialization."""
        result = ShareResult(
            success=True,
            memory_id="mem_1",
            source_agent="agent_a",
            target_agents=["agent_b"],
            access_level="team",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["memory_id"] == "mem_1"
        assert "shared_at" in data


# =============================================================================
# ConflictInfo Tests
# =============================================================================


class TestConflictInfo:
    """Tests for ConflictInfo dataclass."""

    def test_conflict_info_creation(self):
        """Test creating ConflictInfo."""
        source_mem = Memory(
            memory_id="src_1",
            content="Source content",
            memory_type="fact",
            user_id="user_1",
        )
        target_mem = Memory(
            memory_id="tgt_1",
            content="Target content",
            memory_type="fact",
            user_id="user_1",
        )

        info = ConflictInfo(
            memory_id="src_1",
            source_memory=source_mem,
            target_memory=target_mem,
            resolution=ConflictResolution.SOURCE_WINS,
            resolved_memory=source_mem,
            reason="Source wins by strategy",
        )

        assert info.memory_id == "src_1"
        assert info.resolution == ConflictResolution.SOURCE_WINS

    def test_conflict_info_to_dict(self):
        """Test ConflictInfo serialization."""
        source_mem = Memory(
            memory_id="src_1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
        )
        target_mem = Memory(
            memory_id="tgt_1",
            content="Target",
            memory_type="fact",
            user_id="user_1",
        )

        info = ConflictInfo(
            memory_id="src_1",
            source_memory=source_mem,
            target_memory=target_mem,
            resolution=ConflictResolution.NEWEST_WINS,
            resolved_memory=source_mem,
            reason="Source is newer",
        )

        data = info.to_dict()

        assert data["memory_id"] == "src_1"
        assert data["source_content"] == "Source"
        assert data["target_content"] == "Target"
        assert data["resolution"] == "newest_wins"


# =============================================================================
# AgentSyncResult Tests
# =============================================================================


class TestSyncResult:
    """Tests for AgentSyncResult dataclass."""

    def test_sync_result_creation(self):
        """Test creating a SyncResult."""
        result = AgentSyncResult(
            success=True,
            source_agent="agent_a",
            target_agent="agent_b",
            direction=AgentSyncDirection.ONE_WAY,
            memories_synced=10,
            memories_skipped=2,
            conflicts=1,
            conflict_resolution=ConflictResolution.SOURCE_WINS,
        )

        assert result.success is True
        assert result.memories_synced == 10
        assert result.conflicts == 1

    def test_sync_result_to_dict(self):
        """Test SyncResult serialization."""
        result = AgentSyncResult(
            success=True,
            source_agent="a",
            target_agent="b",
            direction=AgentSyncDirection.BIDIRECTIONAL,
            memories_synced=5,
            memories_skipped=0,
            conflicts=0,
            sync_duration_ms=15.5,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["direction"] == "bidirectional"
        assert data["sync_duration_ms"] == 15.5
        assert "synced_at" in data

    def test_sync_result_alias(self):
        """Test SyncResult is alias for AgentSyncResult."""
        assert SyncResult is AgentSyncResult


# =============================================================================
# CrossAgentMemory Share Tests
# =============================================================================


class TestShareMemory:
    """Tests for CrossAgentMemory.share_memory method."""

    def test_share_memory_success(self, sharing, storage, test_memory):
        """Test successful memory sharing."""
        result = sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="support_bot",
            access_level="team",
        )

        assert result.success is True
        assert result.access_level == "team"

        # Verify memory was updated
        updated = storage.get("test_mem_1")
        assert updated.access_level == "team"

    def test_share_memory_agent_not_found(self, sharing, test_memory):
        """Test sharing with unknown agent."""
        result = sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="unknown_agent",
            access_level="shared",
        )

        assert result.success is False
        assert "not found" in result.error

    def test_share_memory_not_found(self, sharing):
        """Test sharing nonexistent memory."""
        result = sharing.share_memory(
            memory_id="nonexistent",
            source_agent="support_bot",
            access_level="shared",
        )

        assert result.success is False
        assert "not found" in result.error

    def test_share_memory_not_owner(self, sharing, storage, test_memory):
        """Test sharing memory not owned by agent."""
        result = sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="sales_bot",  # Different agent
            access_level="shared",
        )

        assert result.success is False
        assert "owner" in result.error

    def test_share_history(self, sharing, storage, test_memory):
        """Test share history is recorded."""
        sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="support_bot",
            access_level="team",
        )

        history = sharing.get_share_history(agent_id="support_bot")

        assert len(history) >= 1
        assert history[-1].memory_id == "test_mem_1"


# =============================================================================
# CrossAgentMemory Sync Tests
# =============================================================================


class TestSyncAgents:
    """Tests for CrossAgentMemory.sync_agents method."""

    def test_sync_source_not_found(self, sharing):
        """Test sync with unknown source agent."""
        result = sharing.sync_agents(
            source_agent="unknown",
            target_agent="sales_bot",
            user_id="user_1",
        )

        assert result.success is False
        assert "not found" in result.errors[0]

    def test_sync_target_not_found(self, sharing):
        """Test sync with unknown target agent."""
        result = sharing.sync_agents(
            source_agent="support_bot",
            target_agent="unknown",
            user_id="user_1",
        )

        assert result.success is False
        assert "not found" in result.errors[0]

    def test_sync_no_memories(self, sharing):
        """Test sync with no memories to sync."""
        result = sharing.sync_agents(
            source_agent="support_bot",
            target_agent="billing_bot",  # Same team
            user_id="user_1",
        )

        # Success even with 0 memories
        assert result.success is True
        assert result.memories_synced == 0

    def test_sync_history(self, sharing):
        """Test sync history is recorded."""
        sharing.sync_agents(
            source_agent="support_bot",
            target_agent="billing_bot",
            user_id="user_1",
        )

        history = sharing.get_sync_history(agent_id="support_bot")

        assert len(history) >= 1


# =============================================================================
# Conflict Resolution Tests
# =============================================================================


class TestConflictResolution:
    """Tests for conflict resolution strategies."""

    def test_resolve_source_wins(self, sharing):
        """Test SOURCE_WINS resolution."""
        source = Memory(
            memory_id="m1",
            content="Source content",
            memory_type="fact",
            user_id="user_1",
            importance=0.5,
        )
        target = Memory(
            memory_id="m2",
            content="Target content",
            memory_type="fact",
            user_id="user_1",
            importance=0.8,
        )

        resolved, reason = sharing._resolve_conflict(source, target, ConflictResolution.SOURCE_WINS)

        assert resolved == source
        assert "source_wins" in reason.lower()

    def test_resolve_target_wins(self, sharing):
        """Test TARGET_WINS resolution."""
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
        )

        resolved, _reason = sharing._resolve_conflict(
            source, target, ConflictResolution.TARGET_WINS
        )

        assert resolved == target

    def test_resolve_skip(self, sharing):
        """Test SKIP resolution."""
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
        )

        resolved, _reason = sharing._resolve_conflict(source, target, ConflictResolution.SKIP)

        assert resolved is None

    def test_resolve_newest_wins_source_newer(self, sharing):
        """Test NEWEST_WINS with source newer."""
        now = datetime.now(timezone.utc)
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
            created_at=now,
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
            created_at=now.replace(year=now.year - 1),
        )

        resolved, reason = sharing._resolve_conflict(source, target, ConflictResolution.NEWEST_WINS)

        assert resolved == source
        assert "newer" in reason.lower()

    def test_resolve_newest_wins_target_newer(self, sharing):
        """Test NEWEST_WINS with target newer."""
        now = datetime.now(timezone.utc)
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
            created_at=now.replace(year=now.year - 1),
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
            created_at=now,
        )

        resolved, _reason = sharing._resolve_conflict(
            source, target, ConflictResolution.NEWEST_WINS
        )

        assert resolved == target

    def test_resolve_highest_importance_source(self, sharing):
        """Test HIGHEST_IMPORTANCE with source higher."""
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
            importance=0.9,
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
            importance=0.5,
        )

        resolved, reason = sharing._resolve_conflict(
            source, target, ConflictResolution.HIGHEST_IMPORTANCE
        )

        assert resolved == source
        assert "importance" in reason.lower()

    def test_resolve_highest_importance_target(self, sharing):
        """Test HIGHEST_IMPORTANCE with target higher."""
        source = Memory(
            memory_id="m1",
            content="Source",
            memory_type="fact",
            user_id="user_1",
            importance=0.3,
        )
        target = Memory(
            memory_id="m2",
            content="Target",
            memory_type="fact",
            user_id="user_1",
            importance=0.9,
        )

        resolved, _reason = sharing._resolve_conflict(
            source, target, ConflictResolution.HIGHEST_IMPORTANCE
        )

        assert resolved == target

    def test_resolve_merge_metadata(self, sharing):
        """Test MERGE_METADATA resolution."""
        source = Memory(
            memory_id="src_1",
            content="Shared content",
            memory_type="fact",
            user_id="user_1",
            agent_id="agent_a",
            topics=["topic_a", "topic_b"],
            categories=["cat_a"],
            entities=["entity_a"],
            importance=0.6,
            reinforcement_score=0.2,
            access_level="team",
        )
        target = Memory(
            memory_id="tgt_1",
            content="Shared content",
            memory_type="fact",
            user_id="user_1",
            agent_id="agent_b",
            topics=["topic_b", "topic_c"],
            categories=["cat_b"],
            entities=["entity_b"],
            importance=0.8,
            reinforcement_score=0.5,
            access_level="shared",
        )

        resolved, _reason = sharing._resolve_conflict(
            source, target, ConflictResolution.MERGE_METADATA
        )

        assert resolved is not None
        # Topics merged
        assert "topic_a" in resolved.topics
        assert "topic_b" in resolved.topics
        assert "topic_c" in resolved.topics
        # Categories merged
        assert "cat_a" in resolved.categories
        assert "cat_b" in resolved.categories
        # Higher importance kept
        assert resolved.importance == 0.8
        # Higher reinforcement kept
        assert resolved.reinforcement_score == 0.5


# =============================================================================
# Content Hash Tests
# =============================================================================


class TestContentHash:
    """Tests for content hashing."""

    def test_content_hash_consistency(self, sharing):
        """Test content hash is consistent."""
        content = "User prefers dark mode"
        hash1 = sharing._content_hash(content)
        hash2 = sharing._content_hash(content)

        assert hash1 == hash2

    def test_content_hash_case_insensitive(self, sharing):
        """Test content hash is case-insensitive."""
        hash1 = sharing._content_hash("Hello World")
        hash2 = sharing._content_hash("hello world")

        assert hash1 == hash2

    def test_content_hash_trims_whitespace(self, sharing):
        """Test content hash trims whitespace."""
        hash1 = sharing._content_hash("Hello")
        hash2 = sharing._content_hash("  Hello  ")

        assert hash1 == hash2


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for sharing statistics."""

    def test_get_stats(self, sharing, storage, test_memory):
        """Test getting sharing stats."""
        # Do some operations
        sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="support_bot",
            access_level="team",
        )
        sharing.sync_agents(
            source_agent="support_bot",
            target_agent="billing_bot",
            user_id="user_1",
        )

        stats = sharing.get_stats()

        assert "total_shares" in stats
        assert "successful_shares" in stats
        assert "total_syncs" in stats
        assert "successful_syncs" in stats
        assert "total_memories_synced" in stats


# =============================================================================
# History Tests
# =============================================================================


class TestHistory:
    """Tests for share and sync history."""

    def test_share_history_filter_by_agent(self, sharing, storage, test_memory):
        """Test filtering share history by agent."""
        sharing.share_memory(
            memory_id="test_mem_1",
            source_agent="support_bot",
            access_level="team",
        )

        # Filter by agent
        history = sharing.get_share_history(agent_id="support_bot", limit=10)
        assert len(history) >= 1

        # Different agent should have empty history
        sharing.get_share_history(agent_id="sales_bot")
        # May or may not be empty depending on target_agents

    def test_sync_history_filter_by_agent(self, sharing):
        """Test filtering sync history by agent."""
        sharing.sync_agents(
            source_agent="support_bot",
            target_agent="billing_bot",
            user_id="user_1",
        )

        history = sharing.get_sync_history(agent_id="support_bot", limit=10)
        assert len(history) >= 1

    def test_history_limit(self, sharing, storage, test_memory):
        """Test history respects limit."""
        for i in range(5):
            sharing.share_memory(
                memory_id="test_mem_1",
                source_agent="support_bot",
                access_level="team",
            )

        history = sharing.get_share_history(limit=3)
        assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
