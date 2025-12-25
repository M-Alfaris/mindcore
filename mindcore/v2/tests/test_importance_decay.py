"""Tests for Memory importance decay functionality."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindcore.v2.flr import FLR, Memory
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
def flr(storage):
    """Create FLR instance."""
    return FLR(storage)


class TestEffectiveImportance:
    """Test effective importance calculation with decay."""

    def test_new_memory_effective_importance_equals_base(self):
        """Test that new memory has effective_importance equal to base importance."""
        memory = Memory(
            memory_id="test_1",
            content="Test content",
            memory_type="fact",
            user_id="user_1",
            importance=0.8,
            importance_decay_rate=0.1,
        )

        # New memory should have effective importance close to base
        assert abs(memory.effective_importance - 0.8) < 0.01

    def test_old_memory_decays(self):
        """Test that old memories have reduced effective importance."""
        # Create a memory from 6 months ago
        old_date = datetime.now(timezone.utc) - timedelta(days=180)

        memory = Memory(
            memory_id="test_2",
            content="Old preference",
            memory_type="preference",
            user_id="user_1",
            importance=0.8,
            importance_decay_rate=0.1,  # 10% decay per month
            created_at=old_date,
        )

        # After 6 months at 10% decay rate, effective should be ~0.8 * 0.9^6 ≈ 0.42
        assert memory.effective_importance < 0.6
        assert memory.effective_importance > 0.3

    def test_no_decay_when_rate_is_zero(self):
        """Test that importance doesn't decay when decay_rate is 0."""
        old_date = datetime.now(timezone.utc) - timedelta(days=365)

        memory = Memory(
            memory_id="test_3",
            content="Permanent fact",
            memory_type="fact",
            user_id="user_1",
            importance=0.8,
            importance_decay_rate=0.0,  # No decay
            created_at=old_date,
        )

        # No decay - should equal base importance
        assert memory.effective_importance == 0.8

    def test_very_old_memory_decays_significantly(self):
        """Test that 2 year old memories have very low effective importance."""
        old_date = datetime.now(timezone.utc) - timedelta(days=730)  # 2 years

        memory = Memory(
            memory_id="test_4",
            content="Very old preference",
            memory_type="preference",
            user_id="user_1",
            importance=0.8,
            importance_decay_rate=0.1,
            created_at=old_date,
        )

        # After 24 months at 10% decay, effective should be very low
        # 0.8 * 0.9^24 ≈ 0.065
        assert memory.effective_importance < 0.15
        assert memory.effective_importance > 0.0


class TestImportanceBoosts:
    """Test importance boosting functionality."""

    def test_boost_importance(self):
        """Test that boost_importance adds a boost."""
        memory = Memory(
            memory_id="test_5",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
        )

        new_importance = memory.boost_importance(
            amount=0.2,
            reason="customer_mentioned_again",
            decay_after_days=30,
        )

        # Should now be 0.5 + 0.2 = 0.7
        assert abs(new_importance - 0.7) < 0.05
        assert len(memory.importance_boosts) == 1
        assert memory.importance_boosts[0]["reason"] == "customer_mentioned_again"

    def test_multiple_boosts_stack(self):
        """Test that multiple boosts add up."""
        memory = Memory(
            memory_id="test_6",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
        )

        memory.boost_importance(amount=0.1, reason="mentioned")
        memory.boost_importance(amount=0.2, reason="positive_feedback")

        # Should be 0.5 + 0.1 + 0.2 = 0.8
        assert abs(memory.effective_importance - 0.8) < 0.05
        assert len(memory.importance_boosts) == 2

    def test_permanent_boost_no_expiration(self):
        """Test boost without decay_after_days is permanent."""
        memory = Memory(
            memory_id="test_7",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
        )

        memory.boost_importance(
            amount=0.3,
            reason="verified_important",
            decay_after_days=None,  # Permanent
        )

        assert abs(memory.effective_importance - 0.8) < 0.05
        assert "expires_at" not in memory.importance_boosts[0]

    def test_expired_boosts_not_counted(self):
        """Test that expired boosts don't contribute to effective importance."""
        memory = Memory(
            memory_id="test_8",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
            importance_boosts=[
                {
                    "amount": 0.2,
                    "reason": "old_boost",
                    "applied_at": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                    "expires_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                }
            ],
        )

        # Expired boost shouldn't count
        assert abs(memory.effective_importance - 0.5) < 0.05

    def test_clear_expired_boosts(self):
        """Test that clear_expired_boosts removes expired entries."""
        now = datetime.now(timezone.utc)

        memory = Memory(
            memory_id="test_9",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
            importance_boosts=[
                {
                    "amount": 0.1,
                    "reason": "expired_boost",
                    "applied_at": (now - timedelta(days=60)).isoformat(),
                    "expires_at": (now - timedelta(days=30)).isoformat(),
                },
                {
                    "amount": 0.2,
                    "reason": "active_boost",
                    "applied_at": now.isoformat(),
                    "expires_at": (now + timedelta(days=30)).isoformat(),
                },
                {
                    "amount": 0.1,
                    "reason": "permanent_boost",
                    "applied_at": (now - timedelta(days=100)).isoformat(),
                },
            ],
        )

        removed = memory.clear_expired_boosts()

        assert removed == 1
        assert len(memory.importance_boosts) == 2
        reasons = [b["reason"] for b in memory.importance_boosts]
        assert "expired_boost" not in reasons
        assert "active_boost" in reasons
        assert "permanent_boost" in reasons

    def test_effective_importance_capped_at_max(self):
        """Test that effective importance is capped at 2.0."""
        memory = Memory(
            memory_id="test_10",
            content="Test content",
            memory_type="preference",
            user_id="user_1",
            importance=1.0,
        )

        # Add many boosts
        for i in range(10):
            memory.boost_importance(amount=0.5, reason=f"boost_{i}")

        # Should be capped at 2.0
        assert memory.effective_importance == 2.0


class TestDecayAndBoostsCombined:
    """Test combined decay and boost behavior."""

    def test_boosted_old_memory(self):
        """Test that boosts can counteract decay."""
        old_date = datetime.now(timezone.utc) - timedelta(days=180)

        memory = Memory(
            memory_id="test_11",
            content="Old but recently mentioned",
            memory_type="preference",
            user_id="user_1",
            importance=0.8,
            importance_decay_rate=0.1,
            created_at=old_date,
        )

        # Without boost, should be around 0.42
        decayed_importance = memory.effective_importance

        # Apply a boost
        memory.boost_importance(amount=0.4, reason="mentioned_recently")

        # Should be significantly higher now
        assert memory.effective_importance > decayed_importance + 0.3


class TestStoragePersistence:
    """Test that importance decay fields persist through storage."""

    def test_store_and_retrieve_with_decay(self, storage):
        """Test that decay fields are persisted."""
        memory = Memory(
            memory_id="persist_1",
            content="Test persistence",
            memory_type="preference",
            user_id="user_1",
            importance=0.7,
            importance_decay_rate=0.15,
            importance_boosts=[
                {
                    "amount": 0.1,
                    "reason": "test",
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        storage.store(memory)
        retrieved = storage.get("persist_1")

        assert retrieved is not None
        assert retrieved.importance == 0.7
        assert retrieved.importance_decay_rate == 0.15
        assert len(retrieved.importance_boosts) == 1
        assert retrieved.importance_boosts[0]["reason"] == "test"

    def test_update_preserves_boosts(self, storage):
        """Test that updating memory preserves boosts."""
        memory = Memory(
            memory_id="persist_2",
            content="Original content",
            memory_type="preference",
            user_id="user_1",
            importance=0.5,
        )

        storage.store(memory)

        # Add a boost and update
        memory.boost_importance(amount=0.2, reason="reinforced")
        memory.content = "Updated content"
        storage.update(memory)

        # Retrieve and verify
        retrieved = storage.get("persist_2")
        assert len(retrieved.importance_boosts) == 1
        assert retrieved.content == "Updated content"


class TestMemorySerialization:
    """Test Memory to_dict and from_dict with new fields."""

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes decay fields."""
        memory = Memory(
            memory_id="serial_1",
            content="Test",
            memory_type="fact",
            user_id="user_1",
            importance=0.5,
            importance_decay_rate=0.2,
            importance_boosts=[{"amount": 0.1, "reason": "test"}],
        )

        data = memory.to_dict()

        assert "importance_decay_rate" in data
        assert data["importance_decay_rate"] == 0.2
        assert "importance_boosts" in data
        assert len(data["importance_boosts"]) == 1
        assert "effective_importance" in data

    def test_from_dict_restores_fields(self):
        """Test that from_dict restores decay fields."""
        data = {
            "memory_id": "serial_2",
            "content": "Test",
            "memory_type": "fact",
            "user_id": "user_1",
            "importance": 0.5,
            "importance_decay_rate": 0.15,
            "importance_boosts": [{"amount": 0.2, "reason": "restored"}],
            "effective_importance": 0.7,  # This should be ignored (computed)
        }

        memory = Memory.from_dict(data)

        assert memory.importance_decay_rate == 0.15
        assert len(memory.importance_boosts) == 1
        assert memory.importance_boosts[0]["reason"] == "restored"


class TestFLRScoring:
    """Test that FLR uses effective_importance for scoring."""

    def test_scoring_uses_effective_importance(self, storage, flr):
        """Test that FLR scoring considers effective importance."""
        # Create an old memory with high base importance
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        old_memory = Memory(
            memory_id="score_1",
            content="User prefers dark mode settings",
            memory_type="preference",
            user_id="user_1",
            importance=0.9,
            importance_decay_rate=0.15,
            created_at=old_date,
            topics=["settings"],
        )

        # Create a new memory with lower base importance
        new_memory = Memory(
            memory_id="score_2",
            content="User prefers dark mode UI",
            memory_type="preference",
            user_id="user_1",
            importance=0.6,
            importance_decay_rate=0.1,
            topics=["settings"],
        )

        storage.store(old_memory)
        storage.store(new_memory)

        # Query for dark mode
        result = flr.query(
            query="dark mode",
            user_id="user_1",
            attention_hints=["settings"],
        )

        # The new memory should score higher even with lower base importance
        # because the old memory has decayed significantly
        assert len(result.memories) >= 2

        # Find both memories in results
        old_idx = next(i for i, m in enumerate(result.memories) if m.memory_id == "score_1")
        new_idx = next(i for i, m in enumerate(result.memories) if m.memory_id == "score_2")

        # New memory should have a higher or similar score despite lower base importance
        # (the old memory's importance has decayed from 0.9 to about 0.15)
        assert result.scores[new_idx] >= result.scores[old_idx] - 0.1
