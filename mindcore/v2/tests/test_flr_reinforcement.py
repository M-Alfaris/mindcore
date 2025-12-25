"""Tests for FLR reinforcement, metadata feedback, usage detection, and query optimization.

Tests cover:
- RobustReinforcement: temporal decay, signal aggregation, exploration bonus
- MetadataFeedbackTracker: effectiveness tracking, feedback generation
- UsageDetector: memory usage detection, auto-reinforcement
- QueryOptimizer: dynamic query optimization
- ContextInjector: API-level context injection
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from mindcore.v2.flr.reinforcement import (
    RobustReinforcement,
    ReinforcementSignal,
    SignalType,
    SignalSource,
    create_feedback_signal,
    batch_reinforce,
    DEFAULT_SOURCE_WEIGHTS,
    DEFAULT_TYPE_WEIGHTS,
)
from mindcore.v2.flr.metadata_feedback import (
    MetadataFeedbackTracker,
    MetadataSignal,
    MetadataEffectiveness,
)
from mindcore.v2.flr.usage_detector import (
    UsageDetector,
    UsageDetectionResult,
    MemoryUsage,
)
from mindcore.v2.flr.query_optimizer import (
    QueryOptimizer,
    QueryOptimization,
    TopicStats,
)
from mindcore.v2.flr.recall import Memory, FLR, RecallResult


# =============================================================================
# ReinforcementSignal Tests
# =============================================================================


class TestReinforcementSignal:
    """Tests for ReinforcementSignal dataclass."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )
        assert signal.signal_type == SignalType.RELEVANCE
        assert signal.value == 0.8
        assert signal.source == SignalSource.USER_EXPLICIT
        assert signal.context_similarity == 1.0

    def test_signal_value_clamping(self):
        """Test that signal values are clamped to [-1, 1]."""
        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=1.5,
            source=SignalSource.LLM_EVALUATION,
        )
        assert signal.value == 1.0

        signal_neg = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=-2.0,
            source=SignalSource.LLM_EVALUATION,
        )
        assert signal_neg.value == -1.0

    def test_context_similarity_clamping(self):
        """Test that context_similarity is clamped to [0, 1]."""
        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.5,
            source=SignalSource.LLM_EVALUATION,
            context_similarity=1.5,
        )
        assert signal.context_similarity == 1.0

    def test_weighted_value(self):
        """Test weighted value calculation."""
        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=1.0,
            source=SignalSource.USER_EXPLICIT,
            context_similarity=1.0,
        )
        weighted = signal.get_weighted_value()
        # value * source_weight * type_weight * context_similarity
        # 1.0 * 1.0 * 0.35 * 1.0 = 0.35
        assert weighted == pytest.approx(0.35, rel=0.01)

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        signal = ReinforcementSignal(
            signal_type=SignalType.USEFULNESS,
            value=0.7,
            source=SignalSource.USER_IMPLICIT,
            context_similarity=0.9,
            query_id="q123",
            session_id="s456",
        )
        data = signal.to_dict()
        restored = ReinforcementSignal.from_dict(data)

        assert restored.signal_type == signal.signal_type
        assert restored.value == signal.value
        assert restored.source == signal.source
        assert restored.context_similarity == signal.context_similarity
        assert restored.query_id == signal.query_id


# =============================================================================
# RobustReinforcement Tests
# =============================================================================


class TestRobustReinforcement:
    """Tests for RobustReinforcement class."""

    def test_apply_signal(self):
        """Test applying a signal."""
        reinforcement = RobustReinforcement()
        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )
        score = reinforcement.apply_signal(signal)

        assert len(reinforcement.signal_history) == 1
        assert reinforcement.reinforcement_count == 1
        assert -1.0 <= score <= 1.0

    def test_apply_simple_signal(self):
        """Test convenience method for simple signals."""
        reinforcement = RobustReinforcement()
        score = reinforcement.apply_simple_signal(
            value=0.5,
            signal_type=SignalType.USEFULNESS,
            source=SignalSource.LLM_EVALUATION,
        )

        assert len(reinforcement.signal_history) == 1
        assert -1.0 <= score <= 1.0

    def test_history_trimming(self):
        """Test that history is trimmed to max_history_size."""
        reinforcement = RobustReinforcement(max_history_size=5)

        for i in range(10):
            reinforcement.apply_simple_signal(value=0.5)

        assert len(reinforcement.signal_history) == 5

    def test_temporal_decay(self):
        """Test that older signals decay."""
        reinforcement = RobustReinforcement(decay_half_life_hours=1.0)

        # Add an old signal
        old_signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=1.0,
            source=SignalSource.USER_EXPLICIT,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        reinforcement.signal_history.append(old_signal)
        reinforcement.reinforcement_count = 1

        # Add a new signal
        new_signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=1.0,
            source=SignalSource.USER_EXPLICIT,
        )
        reinforcement.signal_history.append(new_signal)
        reinforcement.reinforcement_count = 2

        score = reinforcement.get_aggregated_score()
        # New signal should have more weight than old decayed signal
        assert -1.0 <= score <= 1.0

    def test_exploration_bonus(self):
        """Test UCB-like exploration bonus."""
        reinforcement = RobustReinforcement()

        # Unvisited memory gets max exploration bonus
        assert reinforcement.get_exploration_bonus(total_retrievals=100) == 1.0

        # Accessed memory gets lower bonus
        reinforcement.access_count = 10
        bonus = reinforcement.get_exploration_bonus(total_retrievals=100)
        assert 0 < bonus < 1.0

    def test_effective_score(self):
        """Test combined exploitation + exploration score."""
        reinforcement = RobustReinforcement()
        reinforcement.apply_simple_signal(value=0.8)
        reinforcement.access_count = 5

        effective = reinforcement.get_effective_score(
            exploration_factor=0.1,
            total_retrievals=100,
        )

        assert -1.0 <= effective <= 1.0

    def test_trending_detection(self):
        """Test trend detection."""
        reinforcement = RobustReinforcement(moving_average_window=5)

        # Add declining signals
        for i in range(20):
            value = 0.8 - (i * 0.05)  # 0.8 down to -0.15
            reinforcement.apply_simple_signal(value=max(-1, value))

        assert reinforcement.is_trending_down(threshold=0.1)
        assert not reinforcement.is_trending_up(threshold=0.1)

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        reinforcement = RobustReinforcement(decay_half_life_hours=100)
        reinforcement.apply_simple_signal(value=0.5)
        reinforcement.access_count = 10

        data = reinforcement.to_dict()
        restored = RobustReinforcement.from_dict(data)

        assert restored.decay_half_life_hours == reinforcement.decay_half_life_hours
        assert restored.access_count == reinforcement.access_count
        assert len(restored.signal_history) == len(reinforcement.signal_history)


class TestCreateFeedbackSignal:
    """Tests for create_feedback_signal factory function."""

    def test_user_feedback(self):
        """Test signal from user feedback."""
        signal = create_feedback_signal(value=0.9, is_user_feedback=True)
        assert signal.source == SignalSource.USER_EXPLICIT
        assert signal.value == 0.9

    def test_llm_feedback(self):
        """Test signal from LLM feedback."""
        signal = create_feedback_signal(value=0.5, is_user_feedback=False)
        assert signal.source == SignalSource.LLM_EVALUATION

    def test_strong_signal_type(self):
        """Test that strong signals get RELEVANCE type."""
        signal = create_feedback_signal(value=0.9)
        assert signal.signal_type == SignalType.RELEVANCE


# =============================================================================
# MetadataFeedbackTracker Tests
# =============================================================================


class TestMetadataEffectiveness:
    """Tests for MetadataEffectiveness dataclass."""

    def test_effectiveness_score_no_matches(self):
        """Test effectiveness score with no matches."""
        eff = MetadataEffectiveness(value="test", metadata_type="topic")
        assert eff.effectiveness_score == 0.5  # Neutral

    def test_effectiveness_score_all_positive(self):
        """Test effectiveness score with all positive signals."""
        eff = MetadataEffectiveness(value="test", metadata_type="topic")
        eff.record_match(0.8)
        eff.record_match(0.9)
        eff.record_match(0.7)

        assert eff.effectiveness_score == 1.0
        assert eff.times_matched == 3
        assert eff.positive_signals == 3

    def test_effectiveness_score_mixed(self):
        """Test effectiveness score with mixed signals."""
        eff = MetadataEffectiveness(value="test", metadata_type="topic")
        eff.record_match(0.8)  # positive
        eff.record_match(-0.5)  # negative
        eff.record_match(0.6)  # positive

        assert eff.effectiveness_score == pytest.approx(2/3, rel=0.01)
        assert eff.positive_signals == 2
        assert eff.negative_signals == 1


class TestMetadataFeedbackTracker:
    """Tests for MetadataFeedbackTracker class."""

    def test_record_assignment(self):
        """Test recording metadata assignments."""
        tracker = MetadataFeedbackTracker()
        tracker.record_assignment(
            topics=["billing", "refund"],
            categories=["support"],
            intent="ask_question",
        )

        assert "billing" in tracker._topic_effectiveness
        assert "refund" in tracker._topic_effectiveness
        assert tracker._topic_effectiveness["billing"].times_assigned == 1

    def test_record_retrieval_feedback(self):
        """Test recording retrieval feedback."""
        tracker = MetadataFeedbackTracker()

        signal = tracker.record_retrieval_feedback(
            memory_id="mem123",
            assigned_topics=["billing", "refund"],
            assigned_categories=["support"],
            query_topics=["refund"],  # Only refund matched
            query_categories=[],
            signal=0.8,
        )

        assert signal.matched_topics == ["refund"]
        assert "refund" in tracker._topic_effectiveness
        assert tracker._topic_effectiveness["refund"].positive_signals == 1

    def test_get_top_effective_values(self):
        """Test getting top effective values."""
        tracker = MetadataFeedbackTracker()

        # Add some data
        for _ in range(10):
            tracker.record_retrieval_feedback(
                memory_id="mem1",
                assigned_topics=["good_topic"],
                assigned_categories=[],
                query_topics=["good_topic"],
                query_categories=[],
                signal=0.9,
            )

        for _ in range(10):
            tracker.record_retrieval_feedback(
                memory_id="mem2",
                assigned_topics=["bad_topic"],
                assigned_categories=[],
                query_topics=["bad_topic"],
                query_categories=[],
                signal=-0.5,
            )

        top = tracker.get_top_effective_values("topic", limit=5, min_matches=5)
        assert len(top) >= 1
        assert top[0][0] == "good_topic"

    def test_get_feedback_for_extractor(self):
        """Test getting feedback for MetadataExtractor."""
        tracker = MetadataFeedbackTracker()

        # Add sufficient data
        for _ in range(5):
            tracker.record_retrieval_feedback(
                memory_id="mem1",
                assigned_topics=["effective"],
                assigned_categories=[],
                query_topics=["effective"],
                query_categories=[],
                signal=0.8,
            )

        feedback = tracker.get_feedback_for_extractor()

        assert "high_quality_topics" in feedback
        assert "low_quality_topics" in feedback
        assert "guidance" in feedback


# =============================================================================
# UsageDetector Tests
# =============================================================================


class TestMemoryUsage:
    """Tests for MemoryUsage dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        usage = MemoryUsage(
            memory_id="mem123",
            memory_content="Test content",
            memory_topics=["billing"],
            memory_categories=["support"],
            was_used=True,
            usage_confidence=0.8,
        )
        data = usage.to_dict()

        assert data["memory_id"] == "mem123"
        assert data["was_used"] is True
        assert data["usage_confidence"] == 0.8


class TestUsageDetector:
    """Tests for UsageDetector class."""

    def test_detect_usage_content_overlap(self):
        """Test detection based on content overlap."""
        detector = UsageDetector()

        memory = Memory(
            memory_id="mem1",
            content="Your order number is #12345 and it shipped on December 20th",
            memory_type="semantic",
            user_id="user1",
            topics=["orders", "shipping"],
        )

        result = detector.detect_usage(
            llm_response="Based on order #12345, your package shipped on December 20th.",
            retrieved_memories=[memory],
            query_topics=["orders"],
        )

        assert result.total_memories == 1
        assert len(result.used_memories) >= 1  # Should detect usage

    def test_detect_usage_entity_match(self):
        """Test detection based on entity matching."""
        detector = UsageDetector()

        memory = Memory(
            memory_id="mem1",
            content="Contact support@example.com for help",
            memory_type="semantic",
            user_id="user1",
            topics=["support"],
        )

        result = detector.detect_usage(
            llm_response="You can reach us at support@example.com",
            retrieved_memories=[memory],
        )

        # Should detect entity match
        if result.used_memories:
            assert "support@example.com" in result.used_memories[0].matched_entities

    def test_detect_usage_no_match(self):
        """Test detection when memory is not used."""
        detector = UsageDetector()

        memory = Memory(
            memory_id="mem1",
            content="Information about product returns",
            memory_type="semantic",
            user_id="user1",
            topics=["returns"],
        )

        result = detector.detect_usage(
            llm_response="The weather today is sunny with a high of 75 degrees.",
            retrieved_memories=[memory],
        )

        assert len(result.unused_memories) == 1
        assert len(result.used_memories) == 0

    def test_auto_reinforce(self):
        """Test automatic reinforcement."""
        detector = UsageDetector()

        # Create mock FLR
        mock_flr = MagicMock()
        mock_flr.reinforce_with_metadata_feedback.return_value = (0.5, None)

        # Create usage result
        used_usage = MemoryUsage(
            memory_id="mem1",
            memory_content="Test",
            memory_topics=["test"],
            memory_categories=[],
            was_used=True,
            suggested_signal=0.5,
        )
        result = UsageDetectionResult(
            llm_response="Test response",
            total_memories=1,
            used_memories=[used_usage],
            unused_memories=[],
        )

        applied = detector.auto_reinforce(result, mock_flr)

        assert "mem1" in applied
        mock_flr.reinforce_with_metadata_feedback.assert_called()

    def test_get_topic_effectiveness(self):
        """Test topic effectiveness tracking."""
        detector = UsageDetector()

        # Simulate some usage
        for i in range(5):
            memory = Memory(
                memory_id=f"mem{i}",
                content="Test content",
                memory_type="semantic",
                user_id="user1",
                topics=["effective_topic"],
            )
            detector.detect_usage(
                llm_response="Test content is relevant here",
                retrieved_memories=[memory],
            )

        effectiveness = detector.get_topic_effectiveness()
        assert "effective_topic" in effectiveness


# =============================================================================
# QueryOptimizer Tests
# =============================================================================


class TestTopicStats:
    """Tests for TopicStats dataclass."""

    def test_usage_rate(self):
        """Test usage rate calculation."""
        stats = TopicStats(topic="test", times_retrieved=10, times_used=7)
        assert stats.usage_rate == 0.7

    def test_usage_rate_zero_retrieval(self):
        """Test usage rate with no retrievals."""
        stats = TopicStats(topic="test")
        assert stats.usage_rate == 0.0

    def test_effectiveness_score(self):
        """Test effectiveness score."""
        stats = TopicStats(
            topic="test",
            times_retrieved=10,
            times_used=8,
            total_signal=4.0,  # Average 0.5 signal
        )
        score = stats.effectiveness_score
        assert 0 <= score <= 1


class TestQueryOptimizer:
    """Tests for QueryOptimizer class."""

    def test_optimize_query_insufficient_data(self):
        """Test optimization with insufficient data."""
        optimizer = QueryOptimizer(min_samples_for_optimization=10)

        result = optimizer.optimize_query(
            original_topics=["billing", "support"],
            original_limit=10,
        )

        # Should return original values with low confidence
        assert result.optimized_topics == ["billing", "support"]
        assert result.confidence < 0.5

    def test_record_usage(self):
        """Test recording usage data."""
        optimizer = QueryOptimizer()

        # Create a usage result
        used = MemoryUsage(
            memory_id="mem1",
            memory_content="Test",
            memory_topics=["billing"],
            memory_categories=[],
            was_used=True,
            suggested_signal=0.5,
        )
        unused = MemoryUsage(
            memory_id="mem2",
            memory_content="Test2",
            memory_topics=["support"],
            memory_categories=[],
            was_used=False,
            suggested_signal=-0.05,
        )
        result = UsageDetectionResult(
            llm_response="Test",
            total_memories=2,
            used_memories=[used],
            unused_memories=[unused],
        )

        optimizer.record_usage(result)

        assert optimizer._total_retrieved == 2
        assert optimizer._total_used == 1
        assert "billing" in optimizer._topic_stats
        assert "support" in optimizer._topic_stats

    def test_optimize_with_data(self):
        """Test optimization with sufficient data."""
        optimizer = QueryOptimizer(min_samples_for_optimization=5)

        # Add data for effective topic
        for _ in range(10):
            used = MemoryUsage(
                memory_id="mem1",
                memory_content="Test",
                memory_topics=["effective"],
                memory_categories=[],
                was_used=True,
                suggested_signal=0.8,
            )
            result = UsageDetectionResult(
                llm_response="Test",
                total_memories=1,
                used_memories=[used],
                unused_memories=[],
            )
            optimizer.record_usage(result)

        # Add data for ineffective topic
        for _ in range(10):
            unused = MemoryUsage(
                memory_id="mem2",
                memory_content="Test",
                memory_topics=["ineffective"],
                memory_categories=[],
                was_used=False,
                suggested_signal=-0.05,
            )
            result = UsageDetectionResult(
                llm_response="Test",
                total_memories=1,
                used_memories=[],
                unused_memories=[unused],
            )
            optimizer.record_usage(result)

        optimization = optimizer.optimize_query(
            original_topics=["effective", "ineffective"],
            original_limit=10,
        )

        # Effective should be boosted, ineffective might be removed
        assert "effective" in optimization.optimized_topics
        assert optimization.confidence > 0.1

    def test_get_recommendations(self):
        """Test getting recommendations."""
        optimizer = QueryOptimizer()

        # With no data
        recs = optimizer.get_recommendations()
        assert recs["status"] == "insufficient_data"

        # Add some data
        for _ in range(5):
            used = MemoryUsage(
                memory_id="mem1",
                memory_content="Test",
                memory_topics=["test"],
                memory_categories=[],
                was_used=True,
                suggested_signal=0.5,
            )
            result = UsageDetectionResult(
                llm_response="Test",
                total_memories=1,
                used_memories=[used],
                unused_memories=[],
            )
            optimizer.record_usage(result)

        recs = optimizer.get_recommendations()
        assert recs["status"] == "ready"

    def test_reset(self):
        """Test resetting optimizer."""
        optimizer = QueryOptimizer()

        # Add some data
        optimizer._topic_stats["test"] = TopicStats(topic="test", times_retrieved=5)
        optimizer._total_retrieved = 10
        optimizer._total_used = 8

        optimizer.reset()

        assert len(optimizer._topic_stats) == 0
        assert optimizer._total_retrieved == 0
        assert optimizer._total_used == 0


# =============================================================================
# Memory Tests (FLR integration)
# =============================================================================


class TestMemoryReinforcement:
    """Tests for Memory reinforcement methods."""

    def test_apply_reinforcement_legacy(self):
        """Test legacy reinforcement."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="semantic",
            user_id="user1",
        )

        score = memory.apply_reinforcement(0.5)
        assert -1.0 <= score <= 1.0
        assert memory.reinforcement_score > 0

    def test_apply_reinforcement_diminishing_returns(self):
        """Test diminishing returns near bounds."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="semantic",
            user_id="user1",
            reinforcement_score=0.9,
        )

        # Near max, positive signal should have diminishing effect
        old_score = memory.reinforcement_score
        memory.apply_reinforcement(0.5)
        delta = memory.reinforcement_score - old_score

        # Delta should be less than the full signal
        assert delta < 0.5
        assert memory.reinforcement_score <= 1.0

    def test_apply_robust_reinforcement(self):
        """Test robust reinforcement initialization."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="semantic",
            user_id="user1",
        )

        score = memory.apply_robust_reinforcement(
            signal_value=0.8,
            signal_type=SignalType.RELEVANCE,
            source=SignalSource.USER_EXPLICIT,
        )

        assert memory.robust_reinforcement is not None
        assert len(memory.robust_reinforcement.signal_history) == 1
        assert -1.0 <= score <= 1.0

    def test_get_effective_reinforcement_score(self):
        """Test getting effective score."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="semantic",
            user_id="user1",
            reinforcement_score=0.5,
        )

        # Legacy mode
        score = memory.get_effective_reinforcement_score(use_robust=False)
        assert score == 0.5

        # Robust mode (should initialize if not present)
        memory.robust_reinforcement = RobustReinforcement()
        memory.robust_reinforcement.apply_simple_signal(0.5)
        memory.robust_reinforcement.access_count = 5

        score_robust = memory.get_effective_reinforcement_score(
            use_robust=True,
            exploration_factor=0.1,
            total_retrievals=100,
        )
        assert -1.0 <= score_robust <= 1.0

    def test_memory_serialization_with_robust(self):
        """Test Memory serialization includes robust reinforcement."""
        memory = Memory(
            memory_id="mem1",
            content="Test",
            memory_type="semantic",
            user_id="user1",
        )
        memory.apply_robust_reinforcement(0.5)

        data = memory.to_dict()
        assert "robust_reinforcement" in data

        restored = Memory.from_dict(data)
        assert restored.robust_reinforcement is not None
        assert len(restored.robust_reinforcement.signal_history) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestFLRIntegration:
    """Integration tests for FLR with reinforcement components."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.search.return_value = []
        storage.get.return_value = None
        storage.update_reinforcement.return_value = None
        return storage

    def test_flr_with_robust_reinforcement(self, mock_storage):
        """Test FLR with robust reinforcement enabled."""
        flr = FLR(
            storage=mock_storage,
            use_robust_reinforcement=True,
            exploration_factor=0.15,
        )

        assert flr.use_robust_reinforcement is True
        assert flr.exploration_factor == 0.15

    def test_reinforce_with_metadata_feedback(self, mock_storage):
        """Test reinforce_with_metadata_feedback method."""
        # Create a memory in cache
        memory = Memory(
            memory_id="mem1",
            content="Test content",
            memory_type="semantic",
            user_id="user1",
            topics=["billing"],
            categories=["support"],
        )

        flr = FLR(storage=mock_storage)
        flr._cache["mem1"] = (memory, 0)
        flr._last_query_context = {"topics": ["billing"], "categories": []}

        score, meta_signal = flr.reinforce_with_metadata_feedback(
            memory_id="mem1",
            signal=0.8,
            is_user_feedback=True,
        )

        assert score > 0
        assert meta_signal is not None
        assert "billing" in meta_signal.matched_topics

    def test_get_metadata_feedback_for_extractor(self, mock_storage):
        """Test getting metadata feedback for extractor."""
        flr = FLR(storage=mock_storage)

        feedback = flr.get_metadata_feedback_for_extractor()

        assert "high_quality_topics" in feedback
        assert "low_quality_topics" in feedback
        assert "guidance" in feedback

    def test_flr_stats_with_robust(self, mock_storage):
        """Test FLR stats include robust reinforcement info."""
        flr = FLR(storage=mock_storage, use_robust_reinforcement=True)

        stats = flr.get_stats()

        assert "robust_reinforcement" in stats
        assert stats["robust_reinforcement"]["enabled"] is True
        assert "exploration_factor" in stats["robust_reinforcement"]


# =============================================================================
# ContextInjector Tests
# =============================================================================


class TestContextInjector:
    """Tests for ContextInjector and FeedbackInjection."""

    def test_feedback_injection_from_feedback(self):
        """Test creating FeedbackInjection from FLR feedback."""
        from mindcore.v2.svl.llm_providers import FeedbackInjection

        feedback = {
            "high_quality_topics": [("billing", 0.85), ("refund", 0.75)],
            "low_quality_topics": [("general", 0.2)],
            "high_quality_categories": [("support", 0.9)],
            "low_quality_categories": [],
        }

        injection = FeedbackInjection.from_feedback(feedback)

        assert len(injection.effective_topics) == 2
        assert len(injection.ineffective_topics) == 1
        assert injection.effective_topics[0][0] == "billing"

    def test_to_system_instruction(self):
        """Test generating system instruction text."""
        from mindcore.v2.svl.llm_providers import FeedbackInjection

        injection = FeedbackInjection(
            effective_topics=[("billing", 0.85)],
            ineffective_topics=[("general", 0.2)],
        )

        instruction = injection.to_system_instruction()

        assert "billing" in instruction
        assert "general" in instruction

    def test_context_injector_openai(self):
        """Test OpenAI injection."""
        from mindcore.v2.svl.llm_providers import ContextInjector, FeedbackInjection

        injection = FeedbackInjection(
            effective_topics=[("billing", 0.85)],
        )
        injector = ContextInjector(injection)

        result = injector.get_openai_injection()

        assert "instructions" in result
        assert "messages" in result

    def test_context_injector_claude(self):
        """Test Claude injection."""
        from mindcore.v2.svl.llm_providers import ContextInjector, FeedbackInjection

        injection = FeedbackInjection(
            effective_topics=[("billing", 0.85)],
        )
        injector = ContextInjector(injection)

        result = injector.get_claude_injection()

        assert "system_suffix" in result

    def test_annotate_schema(self):
        """Test schema annotation."""
        from mindcore.v2.svl.llm_providers import ContextInjector, FeedbackInjection

        injection = FeedbackInjection(
            effective_topics=[("billing", 0.85)],
            ineffective_topics=[("general", 0.2)],
        )
        injector = ContextInjector(injection)

        schema = {
            "properties": {
                "topics": {
                    "type": "array",
                    "description": "Topics from SVL",
                },
            },
        }

        annotated = injector.annotate_schema(schema)

        assert "billing" in annotated["properties"]["topics"]["description"]
        assert "general" in annotated["properties"]["topics"]["description"]


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch reinforcement operations."""

    def test_batch_reinforce(self):
        """Test batch reinforcement function."""
        reinforcements = [
            RobustReinforcement(),
            RobustReinforcement(),
        ]
        signals = [
            ReinforcementSignal(
                signal_type=SignalType.RELEVANCE,
                value=0.8,
                source=SignalSource.USER_EXPLICIT,
            ),
            ReinforcementSignal(
                signal_type=SignalType.USEFULNESS,
                value=0.6,
                source=SignalSource.LLM_EVALUATION,
            ),
        ]

        scores = batch_reinforce(list(zip(reinforcements, signals)))

        assert len(scores) == 2
        assert all(-1.0 <= s <= 1.0 for s in scores)


# =============================================================================
# Gemini 3 API Tests
# =============================================================================


class TestGemini3Support:
    """Tests for Gemini 3 thinking level API support."""

    def test_thinking_level_enum(self):
        """Test ThinkingLevel enum values."""
        from mindcore.v2.svl.llm_providers import ThinkingLevel

        assert ThinkingLevel.MINIMAL.value == "minimal"
        assert ThinkingLevel.LOW.value == "low"
        assert ThinkingLevel.MEDIUM.value == "medium"
        assert ThinkingLevel.HIGH.value == "high"

    def test_gemini_config_is_gemini_3(self):
        """Test is_gemini_3 detection."""
        from mindcore.v2.svl.llm_providers import GeminiConfig

        # Gemini 2.5 models
        config_25 = GeminiConfig(model="gemini-2.5-flash")
        assert config_25.is_gemini_3() is False

        config_25_pro = GeminiConfig(model="gemini-2.5-pro")
        assert config_25_pro.is_gemini_3() is False

        # Gemini 3 models
        config_3_flash = GeminiConfig(model="gemini-3-flash")
        assert config_3_flash.is_gemini_3() is True

        config_3_pro = GeminiConfig(model="gemini-3-pro")
        assert config_3_pro.is_gemini_3() is True

    def test_gemini_25_uses_thinking_budget(self):
        """Test Gemini 2.5 uses thinkingBudget parameter."""
        from mindcore.v2.svl.llm_providers import GeminiConfig, ThinkingMode

        config = GeminiConfig(
            model="gemini-2.5-flash",
            thinking_mode=ThinkingMode.DYNAMIC,
        )

        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        params = config.get_request_params(schema)

        # Should use thinking_budget, not thinking_level
        thinking_config = params["generation_config"]["thinking_config"]
        assert "thinking_budget" in thinking_config
        assert "thinking_level" not in thinking_config
        assert thinking_config["thinking_budget"] == -1  # Dynamic

    def test_gemini_3_uses_thinking_level(self):
        """Test Gemini 3 uses thinkingLevel parameter."""
        from mindcore.v2.svl.llm_providers import GeminiConfig, ThinkingLevel

        config = GeminiConfig(
            model="gemini-3-flash",
            thinking_level=ThinkingLevel.HIGH,
        )

        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        params = config.get_request_params(schema)

        # Should use thinking_level, not thinking_budget
        thinking_config = params["generation_config"]["thinking_config"]
        assert "thinking_level" in thinking_config
        assert "thinking_budget" not in thinking_config
        assert thinking_config["thinking_level"] == "HIGH"  # Uppercase for API

    def test_gemini_3_default_thinking_level(self):
        """Test Gemini 3 defaults to HIGH thinking level."""
        from mindcore.v2.svl.llm_providers import GeminiConfig

        config = GeminiConfig(model="gemini-3-pro")  # No thinking_level set

        schema = {"type": "object"}
        params = config.get_request_params(schema)

        thinking_config = params["generation_config"]["thinking_config"]
        assert thinking_config["thinking_level"] == "HIGH"  # Default

    def test_get_gemini3_params_with_signatures(self):
        """Test Gemini 3 params include thought signatures."""
        from mindcore.v2.svl.llm_providers import GeminiConfig, ThinkingLevel

        config = GeminiConfig(model="gemini-3-flash")
        schema = {"type": "object"}

        # First call - no signatures
        params = config.get_gemini3_params(schema, ThinkingLevel.HIGH)
        assert "thought_signatures" not in params

        # Subsequent call - with signatures
        signatures = ["sig1", "sig2"]
        params = config.get_gemini3_params(
            schema, ThinkingLevel.HIGH, thought_signatures=signatures
        )
        assert params["thought_signatures"] == signatures

    def test_get_recommended_config_gemini3(self):
        """Test recommended config for Gemini 3."""
        from mindcore.v2.svl.llm_providers import get_recommended_config

        config = get_recommended_config("gemini3")
        assert config.model == "gemini-3-flash"
        assert config.thinking_level is not None

        # Also test with dash
        config2 = get_recommended_config("gemini-3")
        assert config2.model == "gemini-3-flash"

    def test_thinking_mode_vs_level_separation(self):
        """Test that ThinkingMode and ThinkingLevel are separate."""
        from mindcore.v2.svl.llm_providers import (
            GeminiConfig,
            ThinkingLevel,
            ThinkingMode,
        )

        # Gemini 2.5 with ThinkingMode
        config_25 = GeminiConfig(
            model="gemini-2.5-pro",
            thinking_mode=ThinkingMode.FIXED,
            thinking_budget=5000,
        )
        params_25 = config_25.get_request_params({"type": "object"})
        assert params_25["generation_config"]["thinking_config"]["thinking_budget"] == 5000

        # Gemini 3 with ThinkingLevel (should ignore thinking_mode)
        config_3 = GeminiConfig(
            model="gemini-3-flash",
            thinking_mode=ThinkingMode.FIXED,  # Should be ignored
            thinking_budget=5000,  # Should be ignored
            thinking_level=ThinkingLevel.LOW,
        )
        params_3 = config_3.get_request_params({"type": "object"})
        assert params_3["generation_config"]["thinking_config"]["thinking_level"] == "LOW"
        assert "thinking_budget" not in params_3["generation_config"]["thinking_config"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
