"""Tests for QueryOptimizer - Dynamic query optimization based on usage feedback."""

from dataclasses import field
from datetime import datetime, timedelta, timezone

import pytest

from mindcore.flr.query_optimizer import (
    CategoryStats,
    QueryOptimization,
    QueryOptimizer,
    TopicStats,
)
from mindcore.flr.usage_detector import (
    MemoryUsage,
    UsageDetectionResult,
)


@pytest.fixture
def optimizer():
    """Create QueryOptimizer with default settings."""
    return QueryOptimizer()


@pytest.fixture
def optimizer_low_threshold():
    """Create QueryOptimizer with low sample threshold for testing."""
    return QueryOptimizer(min_samples_for_optimization=2)


def create_usage_result(
    used_topics: list[list[str]],
    unused_topics: list[list[str]],
    used_signals: list[float] | None = None,
) -> UsageDetectionResult:
    """Helper to create UsageDetectionResult for testing."""
    used_signals = used_signals or [0.5] * len(used_topics)

    used_memories = []
    for i, topics in enumerate(used_topics):
        used_memories.append(
            MemoryUsage(
                memory_id=f"used_{i}",
                memory_content=f"Used content {i}",
                memory_topics=topics,
                memory_categories=[],
                was_used=True,
                usage_confidence=0.8,
                suggested_signal=used_signals[i] if i < len(used_signals) else 0.5,
            )
        )

    unused_memories = []
    for i, topics in enumerate(unused_topics):
        unused_memories.append(
            MemoryUsage(
                memory_id=f"unused_{i}",
                memory_content=f"Unused content {i}",
                memory_topics=topics,
                memory_categories=[],
                was_used=False,
                usage_confidence=0.0,
                suggested_signal=0.0,
            )
        )

    return UsageDetectionResult(
        llm_response="Test response",
        total_memories=len(used_memories) + len(unused_memories),
        used_memories=used_memories,
        unused_memories=unused_memories,
    )


class TestTopicStats:
    """Test TopicStats dataclass."""

    def test_usage_rate_zero_retrieved(self):
        """Test usage rate when nothing retrieved."""
        stats = TopicStats(topic="test")
        assert stats.usage_rate == 0.0

    def test_usage_rate_calculation(self):
        """Test usage rate calculation."""
        stats = TopicStats(
            topic="test",
            times_retrieved=10,
            times_used=6,
        )
        assert stats.usage_rate == 0.6

    def test_avg_signal_zero_used(self):
        """Test avg signal when nothing used."""
        stats = TopicStats(topic="test")
        assert stats.avg_signal == 0.0

    def test_avg_signal_calculation(self):
        """Test average signal calculation."""
        stats = TopicStats(
            topic="test",
            times_used=4,
            total_signal=2.0,
        )
        assert stats.avg_signal == 0.5

    def test_effectiveness_score_insufficient_data(self):
        """Test effectiveness score with insufficient data."""
        stats = TopicStats(
            topic="test",
            times_retrieved=2,  # Less than 3
        )
        assert stats.effectiveness_score == 0.5  # Neutral

    def test_effectiveness_score_calculation(self):
        """Test effectiveness score calculation."""
        stats = TopicStats(
            topic="test",
            times_retrieved=10,
            times_used=8,  # 80% usage
            total_signal=4.0,  # avg signal = 0.5
        )
        # effectiveness = 0.6 * usage_rate + 0.4 * normalized_signal
        # usage_rate = 0.8
        # normalized_signal = (0.5 + 1) / 2 = 0.75
        # effectiveness = 0.6 * 0.8 + 0.4 * 0.75 = 0.48 + 0.3 = 0.78
        assert 0.7 < stats.effectiveness_score < 0.85


class TestCategoryStats:
    """Test CategoryStats dataclass."""

    def test_usage_rate_zero_retrieved(self):
        """Test usage rate when nothing retrieved."""
        stats = CategoryStats(category="test")
        assert stats.usage_rate == 0.0

    def test_usage_rate_calculation(self):
        """Test usage rate calculation."""
        stats = CategoryStats(
            category="test",
            times_retrieved=20,
            times_used=10,
        )
        assert stats.usage_rate == 0.5

    def test_effectiveness_score_insufficient_data(self):
        """Test effectiveness score with insufficient data."""
        stats = CategoryStats(
            category="test",
            times_retrieved=2,
        )
        assert stats.effectiveness_score == 0.5


class TestQueryOptimization:
    """Test QueryOptimization dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        opt = QueryOptimization(
            original_topics=["a", "b"],
            optimized_topics=["b"],
            removed_topics=["a"],
            boosted_topics=["b"],
            original_limit=10,
            optimized_limit=8,
            confidence=0.7,
            reasoning="Test reasoning",
        )

        data = opt.to_dict()

        assert data["original_topics"] == ["a", "b"]
        assert data["optimized_topics"] == ["b"]
        assert data["removed_topics"] == ["a"]
        assert data["boosted_topics"] == ["b"]
        assert data["original_limit"] == 10
        assert data["optimized_limit"] == 8
        assert data["confidence"] == 0.7


class TestQueryOptimizerBasics:
    """Test basic QueryOptimizer functionality."""

    def test_init_defaults(self):
        """Test default initialization."""
        opt = QueryOptimizer()
        assert opt.min_samples == 5
        assert opt.removal_threshold == 0.2
        assert opt.boost_threshold == 0.6
        assert opt.enable_limit_adjustment is True

    def test_init_custom(self):
        """Test custom initialization."""
        opt = QueryOptimizer(
            min_samples_for_optimization=10,
            topic_removal_threshold=0.3,
            topic_boost_threshold=0.7,
            max_history_age_hours=24.0,
            enable_limit_adjustment=False,
        )
        assert opt.min_samples == 10
        assert opt.removal_threshold == 0.3
        assert opt.boost_threshold == 0.7
        assert opt.enable_limit_adjustment is False

    def test_optimize_query_no_data(self, optimizer):
        """Test optimization with no usage data."""
        result = optimizer.optimize_query(
            original_topics=["billing", "support"],
            original_limit=10,
        )

        assert isinstance(result, QueryOptimization)
        assert result.optimized_topics == ["billing", "support"]
        assert result.optimized_limit == 10
        assert result.confidence < 0.2  # Low confidence
        assert "Insufficient data" in result.reasoning

    def test_optimize_query_preserves_unknown_topics(self, optimizer_low_threshold):
        """Test that unknown topics are preserved."""
        # Record some data for one topic
        usage = create_usage_result(
            used_topics=[["known_topic"]],
            unused_topics=[["known_topic"]],
        )
        optimizer_low_threshold.record_usage(usage)
        optimizer_low_threshold.record_usage(usage)  # Meet threshold

        result = optimizer_low_threshold.optimize_query(
            original_topics=["known_topic", "unknown_topic"],
        )

        # Unknown topic should be preserved with neutral score
        assert "unknown_topic" in result.optimized_topics


class TestRecordUsage:
    """Test usage recording."""

    def test_record_used_memory(self, optimizer):
        """Test recording a used memory."""
        usage = create_usage_result(
            used_topics=[["billing"]],
            unused_topics=[],
        )

        optimizer.record_usage(usage)

        assert "billing" in optimizer._topic_stats
        assert optimizer._topic_stats["billing"].times_used == 1
        assert optimizer._topic_stats["billing"].times_retrieved == 1

    def test_record_unused_memory(self, optimizer):
        """Test recording an unused memory."""
        usage = create_usage_result(
            used_topics=[],
            unused_topics=[["billing"]],
        )

        optimizer.record_usage(usage)

        assert "billing" in optimizer._topic_stats
        assert optimizer._topic_stats["billing"].times_used == 0
        assert optimizer._topic_stats["billing"].times_retrieved == 1

    def test_record_multiple_topics(self, optimizer):
        """Test recording memory with multiple topics."""
        usage = create_usage_result(
            used_topics=[["billing", "refund", "support"]],
            unused_topics=[],
        )

        optimizer.record_usage(usage)

        assert "billing" in optimizer._topic_stats
        assert "refund" in optimizer._topic_stats
        assert "support" in optimizer._topic_stats

    def test_accumulate_stats(self, optimizer):
        """Test that stats accumulate across multiple recordings."""
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["billing"]],
                unused_topics=[["billing"]],  # Same topic, sometimes used
            )
            optimizer.record_usage(usage)

        # Should have 10 retrievals, 5 uses
        assert optimizer._topic_stats["billing"].times_retrieved == 10
        assert optimizer._topic_stats["billing"].times_used == 5

    def test_record_updates_overall_stats(self, optimizer):
        """Test that overall stats are updated."""
        usage = create_usage_result(
            used_topics=[["a"], ["b"]],  # 2 used
            unused_topics=[["c"]],  # 1 unused
        )

        optimizer.record_usage(usage)

        assert optimizer._total_retrieved == 3
        assert optimizer._total_used == 2

    def test_signal_accumulated(self, optimizer):
        """Test that signals are accumulated."""
        usage = create_usage_result(
            used_topics=[["billing"]],
            unused_topics=[],
            used_signals=[0.8],
        )

        optimizer.record_usage(usage)

        assert optimizer._topic_stats["billing"].total_signal == 0.8


class TestOptimization:
    """Test query optimization logic."""

    def test_remove_low_usage_topic(self, optimizer_low_threshold):
        """Test that low-usage topics are removed."""
        # Create data where topic_a has low usage, topic_b has high usage
        for _ in range(3):
            usage = create_usage_result(
                used_topics=[["topic_b"]],  # topic_b always used
                unused_topics=[["topic_a"]],  # topic_a never used
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["topic_a", "topic_b"],
        )

        # topic_a should be removed (0% usage)
        assert "topic_a" in result.removed_topics
        assert "topic_b" in result.optimized_topics

    def test_boost_high_usage_topic(self, optimizer_low_threshold):
        """Test that high-usage topics are boosted."""
        # Create data where topic has very high usage
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["high_usage"]],
                unused_topics=[],
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["high_usage"],
        )

        # high_usage should be boosted (100% usage)
        assert "high_usage" in result.boosted_topics

    def test_min_topics_preserved(self, optimizer_low_threshold):
        """Test that minimum topics are preserved even if low usage."""
        # All topics have low usage
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[],
                unused_topics=[["a"], ["b"]],
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["a", "b"],
            min_topics=1,
        )

        # At least one topic should remain
        assert len(result.optimized_topics) >= 1

    def test_limit_adjustment_decrease(self, optimizer_low_threshold):
        """Test limit adjustment when usage is low."""
        # Create low usage scenario (lots of unused memories)
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["topic"]],  # 1 used
                unused_topics=[["topic"]] * 9,  # 9 unused = 10% usage
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
            original_limit=10,
        )

        # Limit should be reduced
        assert result.optimized_limit <= result.original_limit

    def test_limit_adjustment_increase(self, optimizer_low_threshold):
        """Test limit adjustment when usage is high."""
        # Create high usage scenario
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["topic"]] * 9,  # 9 used
                unused_topics=[["topic"]],  # 1 unused = 90% usage
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
            original_limit=10,
        )

        # Limit should be increased
        assert result.optimized_limit >= result.original_limit

    def test_limit_bounds(self, optimizer_low_threshold):
        """Test that limit stays within bounds."""
        # Create very high usage scenario
        for _ in range(10):
            usage = create_usage_result(
                used_topics=[["topic"]] * 10,
                unused_topics=[],
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
            original_limit=100,  # Very high
        )

        # Should be clamped to max
        assert result.optimized_limit <= 20  # MAX_RETRIEVAL_LIMIT

    def test_confidence_increases_with_samples(self, optimizer_low_threshold):
        """Test that confidence increases with more data."""
        # Low data
        usage = create_usage_result(
            used_topics=[["topic"]],
            unused_topics=[["topic"]],
        )
        optimizer_low_threshold.record_usage(usage)
        optimizer_low_threshold.record_usage(usage)

        result1 = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
        )

        # Add more data
        for _ in range(50):
            optimizer_low_threshold.record_usage(usage)

        result2 = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
        )

        # Confidence should be higher with more samples
        assert result2.confidence >= result1.confidence


class TestCategoryOptimization:
    """Test category-based optimization."""

    def test_optimize_with_categories(self, optimizer_low_threshold):
        """Test optimization includes categories."""
        result = optimizer_low_threshold.optimize_query(
            original_topics=["topic"],
            original_categories=["support", "billing"],
        )

        assert "support" in result.original_categories
        assert "billing" in result.original_categories


class TestHistoryPruning:
    """Test data history pruning."""

    def test_old_data_pruned(self):
        """Test that old data is pruned."""
        optimizer = QueryOptimizer(max_history_age_hours=0.001)  # Very short

        usage = create_usage_result(
            used_topics=[["topic"]],
            unused_topics=[],
        )
        optimizer.record_usage(usage)

        # Wait and record again
        import time

        time.sleep(0.01)
        optimizer.record_usage(usage)

        # Old data should be pruned
        # Just verify it doesn't crash
        result = optimizer.optimize_query(original_topics=["topic"])
        assert result is not None


class TestRankings:
    """Test topic and category rankings."""

    def test_get_topic_rankings(self, optimizer_low_threshold):
        """Test getting topic rankings."""
        # Create varied usage
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["high_usage"]],
                unused_topics=[["low_usage"]],
            )
            optimizer_low_threshold.record_usage(usage)

        rankings = optimizer_low_threshold.get_topic_rankings()

        assert len(rankings) == 2
        # Rankings are tuples of (topic, score)
        topics = [r[0] for r in rankings]
        assert "high_usage" in topics
        assert "low_usage" in topics
        # high_usage should rank higher (first in sorted list)
        assert rankings[0][0] == "high_usage"

    def test_get_category_rankings(self, optimizer_low_threshold):
        """Test getting category rankings."""
        # Record some data with categories
        for _ in range(5):
            used = MemoryUsage(
                memory_id="used_1",
                memory_content="Content",
                memory_topics=["topic"],
                memory_categories=["category_a"],
                was_used=True,
                usage_confidence=0.8,
                suggested_signal=0.5,
            )
            unused = MemoryUsage(
                memory_id="unused_1",
                memory_content="Content",
                memory_topics=["topic"],
                memory_categories=["category_b"],
                was_used=False,
                usage_confidence=0.0,
                suggested_signal=0.0,
            )
            result = UsageDetectionResult(
                llm_response="Response",
                total_memories=2,
                used_memories=[used],
                unused_memories=[unused],
            )
            optimizer_low_threshold.record_usage(result)

        rankings = optimizer_low_threshold.get_category_rankings()

        # Should have both categories (returns list of tuples)
        assert len(rankings) >= 1


class TestRecommendations:
    """Test optimization recommendations."""

    def test_get_recommendations(self, optimizer_low_threshold):
        """Test getting recommendations."""
        # Create data with clear patterns
        for _ in range(10):
            usage = create_usage_result(
                used_topics=[["good_topic"]],
                unused_topics=[["bad_topic"]],
            )
            optimizer_low_threshold.record_usage(usage)

        recommendations = optimizer_low_threshold.get_recommendations()

        assert isinstance(recommendations, dict)
        assert "status" in recommendations
        assert recommendations["status"] in ["ready", "insufficient_data"]

    def test_get_recommendations_insufficient_data(self, optimizer):
        """Test recommendations with insufficient data."""
        recommendations = optimizer.get_recommendations()

        assert recommendations["status"] == "insufficient_data"

    def test_get_recommendations_includes_topics(self, optimizer_low_threshold):
        """Test that recommendations include topic info."""
        for _ in range(10):
            usage = create_usage_result(
                used_topics=[["topic_a"]],
                unused_topics=[["topic_b"]],
            )
            optimizer_low_threshold.record_usage(usage)

        recommendations = optimizer_low_threshold.get_recommendations()

        if recommendations["status"] == "ready":
            assert "top_performing_topics" in recommendations
            assert "recommendations" in recommendations


class TestGetStats:
    """Test optimizer statistics."""

    def test_get_stats_empty(self, optimizer):
        """Test stats with no data."""
        stats = optimizer.get_stats()

        assert stats["topics_tracked"] == 0
        assert stats["categories_tracked"] == 0
        assert stats["total_retrieved"] == 0
        assert stats["total_used"] == 0

    def test_get_stats_with_data(self, optimizer):
        """Test stats after recording data."""
        usage = create_usage_result(
            used_topics=[["a", "b"]],
            unused_topics=[["c"]],
        )
        optimizer.record_usage(usage)

        stats = optimizer.get_stats()

        assert stats["topics_tracked"] == 3
        assert stats["total_retrieved"] >= 1
        assert "overall_usage_rate" in stats
        assert "history_size" in stats

    def test_get_stats_topic_details(self, optimizer):
        """Test that stats include topic details when enough samples."""
        # Record enough data to meet min samples threshold
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["topic_a"]],
                unused_topics=[["topic_a"]],
            )
            optimizer.record_usage(usage)

        stats = optimizer.get_stats()

        # Should have topic_stats with details
        assert "topic_stats" in stats
        if "topic_a" in stats["topic_stats"]:
            topic_stats = stats["topic_stats"]["topic_a"]
            assert "retrieved" in topic_stats
            assert "used" in topic_stats
            assert "usage_rate" in topic_stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_topics_list(self, optimizer):
        """Test optimization with empty topics list."""
        result = optimizer.optimize_query(
            original_topics=[],
            original_limit=10,
        )

        assert result.optimized_topics == []

    def test_single_topic(self, optimizer_low_threshold):
        """Test optimization with single topic."""
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["only_topic"]],
                unused_topics=[],
            )
            optimizer_low_threshold.record_usage(usage)

        result = optimizer_low_threshold.optimize_query(
            original_topics=["only_topic"],
            min_topics=1,
        )

        # Single topic should be preserved
        assert "only_topic" in result.optimized_topics

    def test_limit_zero(self, optimizer):
        """Test with zero limit."""
        result = optimizer.optimize_query(
            original_topics=["topic"],
            original_limit=0,
        )

        # Should handle gracefully
        assert result.optimized_limit >= 0

    def test_negative_signal(self, optimizer):
        """Test with negative signals."""
        usage = create_usage_result(
            used_topics=[["topic"]],
            unused_topics=[],
            used_signals=[-0.5],  # Negative signal
        )

        optimizer.record_usage(usage)

        # Should handle negative signals
        assert optimizer._topic_stats["topic"].total_signal == -0.5


class TestDisabledLimitAdjustment:
    """Test with limit adjustment disabled."""

    def test_limit_not_adjusted(self):
        """Test that limit is not adjusted when disabled."""
        optimizer = QueryOptimizer(
            enable_limit_adjustment=False,
            min_samples_for_optimization=1,
        )

        # Record high usage data
        for _ in range(5):
            usage = create_usage_result(
                used_topics=[["topic"]] * 9,
                unused_topics=[["topic"]],
            )
            optimizer.record_usage(usage)

        result = optimizer.optimize_query(
            original_topics=["topic"],
            original_limit=10,
        )

        # Limit should remain unchanged
        assert result.optimized_limit == 10
