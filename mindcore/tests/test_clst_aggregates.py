"""Tests for CLST Session Aggregates - Weighted metadata aggregation.

Tests cover:
- SessionAggregate: incremental updates, weight calculations
- WeightCalculator: batch weight computation, importance stats
- Relevance scoring and top topics/categories
- Serialization and deserialization
"""

from datetime import datetime, timedelta, timezone

import pytest

from mindcore.clst.aggregates import (
    HierarchicalQueryResult,
    SessionAggregate,
    WeightCalculator,
)
from mindcore.flr import Memory


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def memory_list():
    """Create a list of test memories."""
    now = datetime.now(timezone.utc)
    return [
        Memory(
            memory_id="mem1",
            content="I need to check my order status",
            memory_type="episodic",
            user_id="user_1",
            topics=["orders", "shipping"],
            categories=["support"],
            entities=["order #12345"],
            importance=0.8,
            sentiment="neutral",
            created_at=now - timedelta(hours=1),
        ),
        Memory(
            memory_id="mem2",
            content="The package should arrive tomorrow",
            memory_type="episodic",
            user_id="user_1",
            topics=["shipping", "delivery"],
            categories=["support", "logistics"],
            entities=["package"],
            importance=0.6,
            sentiment="positive",
            created_at=now - timedelta(minutes=30),
        ),
        Memory(
            memory_id="mem3",
            content="User prefers express shipping",
            memory_type="preference",
            user_id="user_1",
            topics=["shipping", "preferences"],
            categories=["user_preference"],
            entities=[],
            importance=0.9,
            sentiment="neutral",
            created_at=now,
        ),
    ]


@pytest.fixture
def session_aggregate():
    """Create a basic session aggregate."""
    return SessionAggregate(
        session_id="session_1",
        user_id="user_1",
        agent_id="agent_1",
    )


# =============================================================================
# SessionAggregate Basic Tests
# =============================================================================


class TestSessionAggregateBasics:
    """Tests for basic SessionAggregate operations."""

    def test_create_empty_aggregate(self):
        """Test creating an empty session aggregate."""
        agg = SessionAggregate(
            session_id="session_1",
            user_id="user_1",
        )

        assert agg.session_id == "session_1"
        assert agg.user_id == "user_1"
        assert agg.memory_count == 0
        assert agg.topic_weights == {}
        assert agg.access_level == "private"

    def test_default_values(self):
        """Test default values are set correctly."""
        agg = SessionAggregate(
            session_id="s1",
            user_id="u1",
        )

        assert agg.importance_min == 1.0
        assert agg.importance_max == 0.0
        assert agg.importance_avg == 0.0
        assert agg.dominant_topic is None
        assert agg.dominant_category is None


class TestSessionAggregateUpdate:
    """Tests for SessionAggregate.update_from_memory method."""

    def test_update_single_memory(self, session_aggregate, memory_list):
        """Test updating with a single memory."""
        memory = memory_list[0]
        session_aggregate.update_from_memory(memory)

        assert session_aggregate.memory_count == 1
        assert session_aggregate.message_count == 1
        assert "orders" in session_aggregate.topic_weights
        assert "shipping" in session_aggregate.topic_weights
        assert "support" in session_aggregate.category_weights

    def test_update_multiple_memories(self, session_aggregate, memory_list):
        """Test updating with multiple memories."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        assert session_aggregate.memory_count == 3
        # Shipping appears in all 3 memories
        assert "shipping" in session_aggregate.topic_weights

    def test_importance_stats_update(self, session_aggregate, memory_list):
        """Test importance statistics are updated correctly."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        assert session_aggregate.importance_min == 0.6
        assert session_aggregate.importance_max == 0.9
        # Average of 0.8, 0.6, 0.9 = 0.7666...
        assert 0.76 <= session_aggregate.importance_avg <= 0.77

    def test_dominant_values_updated(self, session_aggregate, memory_list):
        """Test dominant topic/category are set correctly."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        # Shipping appears in all 3 memories, should be dominant
        assert session_aggregate.dominant_topic is not None

    def test_time_bounds_updated(self, session_aggregate, memory_list):
        """Test time bounds are tracked correctly."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        assert session_aggregate.started_at is not None
        assert session_aggregate.last_activity_at is not None
        assert session_aggregate.started_at <= session_aggregate.last_activity_at

    def test_access_level_upgrade(self, session_aggregate):
        """Test access level is upgraded to highest."""
        private_mem = Memory(
            memory_id="m1",
            content="Private",
            memory_type="fact",
            user_id="user_1",
            access_level="private",
        )
        team_mem = Memory(
            memory_id="m2",
            content="Team",
            memory_type="fact",
            user_id="user_1",
            access_level="team",
        )
        shared_mem = Memory(
            memory_id="m3",
            content="Shared",
            memory_type="fact",
            user_id="user_1",
            access_level="shared",
        )

        session_aggregate.update_from_memory(private_mem)
        assert session_aggregate.access_level == "private"

        session_aggregate.update_from_memory(team_mem)
        assert session_aggregate.access_level == "team"

        session_aggregate.update_from_memory(shared_mem)
        assert session_aggregate.access_level == "shared"


class TestTermWeightCalculation:
    """Tests for incremental term weight calculation."""

    def test_weight_normalization(self, session_aggregate):
        """Test weights are normalized to [0, 1]."""
        for i in range(5):
            memory = Memory(
                memory_id=f"m{i}",
                content=f"Content {i}",
                memory_type="fact",
                user_id="user_1",
                topics=["common_topic", f"unique_{i}"],
                importance=0.5 + (i * 0.1),
            )
            session_aggregate.update_from_memory(memory)

        # All weights should be in [0, 1]
        for weight in session_aggregate.topic_weights.values():
            assert 0.0 <= weight <= 1.0

        # Highest weighted topic should be 1.0 after normalization
        assert max(session_aggregate.topic_weights.values()) == 1.0

    def test_ema_weight_update(self, session_aggregate):
        """Test exponential moving average updates."""
        # Add first memory with topic
        mem1 = Memory(
            memory_id="m1",
            content="First",
            memory_type="fact",
            user_id="user_1",
            topics=["topic_a"],
            importance=0.5,
        )
        session_aggregate.update_from_memory(mem1)
        session_aggregate.topic_weights.get("topic_a", 0)

        # Add second memory with same topic
        mem2 = Memory(
            memory_id="m2",
            content="Second",
            memory_type="fact",
            user_id="user_1",
            topics=["topic_a"],
            importance=0.8,
        )
        session_aggregate.update_from_memory(mem2)
        updated_weight = session_aggregate.topic_weights.get("topic_a", 0)

        # Weight should still be 1.0 after normalization (only one topic)
        assert updated_weight == 1.0


# =============================================================================
# Relevance Scoring Tests
# =============================================================================


class TestRelevanceScoring:
    """Tests for SessionAggregate.calculate_relevance_score method."""

    def test_relevance_with_matching_topics(self, session_aggregate, memory_list):
        """Test relevance score with matching topic hints."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        score = session_aggregate.calculate_relevance_score(
            topic_hints=["shipping"],
        )

        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have positive score for matching topic

    def test_relevance_with_no_hints(self, session_aggregate, memory_list):
        """Test relevance score with no hints."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        score = session_aggregate.calculate_relevance_score()

        # Should still get score from importance and recency
        assert 0.0 <= score <= 1.0

    def test_relevance_with_non_matching_topics(self, session_aggregate, memory_list):
        """Test relevance score with non-matching topics."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        score_matching = session_aggregate.calculate_relevance_score(
            topic_hints=["shipping"],
        )
        score_non_matching = session_aggregate.calculate_relevance_score(
            topic_hints=["unrelated_topic"],
        )

        # Matching topics should score higher
        assert score_matching >= score_non_matching

    def test_relevance_with_category_hints(self, session_aggregate, memory_list):
        """Test relevance score with category hints."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        score = session_aggregate.calculate_relevance_score(
            category_hints=["support"],
        )

        assert 0.0 <= score <= 1.0
        assert score > 0

    def test_relevance_min_importance_filter(self, session_aggregate, memory_list):
        """Test relevance score respects min_importance."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        # Session has avg importance ~0.77
        score_low_threshold = session_aggregate.calculate_relevance_score(
            min_importance=0.5,
        )
        score_high_threshold = session_aggregate.calculate_relevance_score(
            min_importance=0.9,
        )

        # Both should be valid scores
        assert 0.0 <= score_low_threshold <= 1.0
        assert 0.0 <= score_high_threshold <= 1.0
        # When importance avg (0.77) is below threshold (0.9), no importance bonus
        # is added but score is still calculated from other factors


# =============================================================================
# Top Topics/Categories Tests
# =============================================================================


class TestTopRankings:
    """Tests for get_top_topics and get_top_categories."""

    def test_get_top_topics(self, session_aggregate, memory_list):
        """Test getting top weighted topics."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        top = session_aggregate.get_top_topics(limit=3)

        assert len(top) <= 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
        # Should be sorted by weight descending
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]

    def test_get_top_categories(self, session_aggregate, memory_list):
        """Test getting top weighted categories."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        top = session_aggregate.get_top_categories(limit=2)

        assert len(top) <= 2
        assert all(isinstance(c, tuple) and len(c) == 2 for c in top)

    def test_top_topics_empty_aggregate(self, session_aggregate):
        """Test getting top topics from empty aggregate."""
        top = session_aggregate.get_top_topics(limit=5)
        assert top == []


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSessionAggregateSerialization:
    """Tests for SessionAggregate serialization."""

    def test_to_dict(self, session_aggregate, memory_list):
        """Test serialization to dictionary."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        data = session_aggregate.to_dict()

        assert data["session_id"] == "session_1"
        assert data["user_id"] == "user_1"
        assert data["memory_count"] == 3
        assert "topic_weights" in data
        assert "importance_avg" in data

    def test_from_dict(self, session_aggregate, memory_list):
        """Test deserialization from dictionary."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        data = session_aggregate.to_dict()
        restored = SessionAggregate.from_dict(data)

        assert restored.session_id == session_aggregate.session_id
        assert restored.memory_count == session_aggregate.memory_count
        assert restored.topic_weights == session_aggregate.topic_weights

    def test_round_trip(self, session_aggregate, memory_list):
        """Test serialization round-trip."""
        for memory in memory_list:
            session_aggregate.update_from_memory(memory)

        data = session_aggregate.to_dict()
        restored = SessionAggregate.from_dict(data)
        data2 = restored.to_dict()

        # Key fields should match
        assert data["session_id"] == data2["session_id"]
        assert data["memory_count"] == data2["memory_count"]
        assert data["importance_avg"] == data2["importance_avg"]


# =============================================================================
# WeightCalculator Tests
# =============================================================================


class TestWeightCalculator:
    """Tests for WeightCalculator class."""

    def test_calculate_weights_from_memories(self, memory_list):
        """Test batch weight calculation."""
        weights = WeightCalculator.calculate_weights_from_memories(memory_list)

        assert "topic_weights" in weights
        assert "category_weights" in weights
        assert "entity_weights" in weights

        # Shipping appears in all memories
        assert "shipping" in weights["topic_weights"]

    def test_calculate_weights_empty_list(self):
        """Test weight calculation with empty list."""
        weights = WeightCalculator.calculate_weights_from_memories([])

        assert weights["topic_weights"] == {}
        assert weights["category_weights"] == {}
        assert weights["entity_weights"] == {}

    def test_weights_normalized(self, memory_list):
        """Test weights are normalized to [0, 1]."""
        weights = WeightCalculator.calculate_weights_from_memories(memory_list)

        for w in weights["topic_weights"].values():
            assert 0.0 <= w <= 1.0
        for w in weights["category_weights"].values():
            assert 0.0 <= w <= 1.0

    def test_calculate_importance_stats(self, memory_list):
        """Test importance statistics calculation."""
        stats = WeightCalculator.calculate_importance_stats(memory_list)

        assert stats["min"] == 0.6
        assert stats["max"] == 0.9
        assert 0.76 <= stats["avg"] <= 0.77
        assert stats["sum"] == pytest.approx(2.3, rel=0.01)

    def test_calculate_importance_stats_empty(self):
        """Test importance stats with empty list."""
        stats = WeightCalculator.calculate_importance_stats([])

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["avg"] == 0.0
        assert stats["sum"] == 0.0


class TestRebuildSessionAggregate:
    """Tests for WeightCalculator.rebuild_session_aggregate."""

    def test_rebuild_from_memories(self, memory_list):
        """Test rebuilding aggregate from memories."""
        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="rebuilt_session",
            user_id="user_1",
            memories=memory_list,
            agent_id="agent_1",
        )

        assert agg.session_id == "rebuilt_session"
        assert agg.memory_count == 3
        assert agg.importance_min == 0.6
        assert agg.importance_max == 0.9
        assert "shipping" in agg.topic_weights
        assert agg.dominant_topic is not None

    def test_rebuild_empty_memories(self):
        """Test rebuilding with empty memories list."""
        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="empty_session",
            user_id="user_1",
            memories=[],
        )

        assert agg.session_id == "empty_session"
        assert agg.memory_count == 0
        assert agg.topic_weights == {}

    def test_rebuild_sets_time_bounds(self, memory_list):
        """Test that rebuild sets correct time bounds."""
        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="s1",
            user_id="u1",
            memories=memory_list,
        )

        assert agg.started_at is not None
        assert agg.last_activity_at is not None
        # Started at should be the earliest memory
        assert agg.started_at <= agg.last_activity_at

    def test_rebuild_sets_dominant_sentiment(self, memory_list):
        """Test that rebuild sets dominant sentiment."""
        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="s1",
            user_id="u1",
            memories=memory_list,
        )

        # Should have sentiment weights
        assert len(agg.sentiment_weights) > 0
        assert agg.dominant_sentiment is not None

    def test_rebuild_access_level(self, memory_list):
        """Test that rebuild sets highest access level."""
        # Modify one memory to have shared access
        memory_list[0].access_level = "shared"

        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="s1",
            user_id="u1",
            memories=memory_list,
        )

        assert agg.access_level == "shared"


# =============================================================================
# HierarchicalQueryResult Tests
# =============================================================================


class TestHierarchicalQueryResult:
    """Tests for HierarchicalQueryResult dataclass."""

    def test_to_dict(self, memory_list):
        """Test serialization."""
        agg = WeightCalculator.rebuild_session_aggregate(
            session_id="s1",
            user_id="u1",
            memories=memory_list,
        )

        result = HierarchicalQueryResult(
            memories=memory_list,
            sessions=[agg],
            sessions_searched=5,
            memories_returned=3,
            query_latency_ms=15.5,
            session_scores={"s1": 0.85},
        )

        data = result.to_dict()

        assert len(data["memories"]) == 3
        assert len(data["sessions"]) == 1
        assert data["sessions_searched"] == 5
        assert data["memories_returned"] == 3
        assert data["query_latency_ms"] == 15.5
        assert data["session_scores"]["s1"] == 0.85


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_memory_without_topics(self, session_aggregate):
        """Test handling memory with no topics."""
        memory = Memory(
            memory_id="m1",
            content="No topics",
            memory_type="fact",
            user_id="user_1",
            topics=[],
            importance=0.5,
        )
        session_aggregate.update_from_memory(memory)

        assert session_aggregate.memory_count == 1
        assert session_aggregate.topic_weights == {}

    def test_memory_without_categories(self, session_aggregate):
        """Test handling memory with no categories."""
        memory = Memory(
            memory_id="m1",
            content="No categories",
            memory_type="fact",
            user_id="user_1",
            topics=["topic1"],
            categories=[],
            importance=0.5,
        )
        session_aggregate.update_from_memory(memory)

        assert session_aggregate.category_weights == {}

    def test_very_old_memory_decay(self, session_aggregate):
        """Test decay handling for very old memories."""
        old_memory = Memory(
            memory_id="m1",
            content="Old memory",
            memory_type="fact",
            user_id="user_1",
            topics=["old_topic"],
            importance=0.5,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        session_aggregate.update_from_memory(old_memory)

        # Should still be tracked
        assert "old_topic" in session_aggregate.topic_weights

    def test_memory_without_created_at(self, session_aggregate):
        """Test handling memory without created_at timestamp."""
        memory = Memory(
            memory_id="m1",
            content="No timestamp",
            memory_type="fact",
            user_id="user_1",
            topics=["topic1"],
            importance=0.5,
        )
        # Remove created_at by setting to None
        memory.created_at = None

        session_aggregate.update_from_memory(memory)

        assert session_aggregate.memory_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
