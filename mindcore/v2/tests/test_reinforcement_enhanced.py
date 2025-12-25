"""Tests for enhanced reinforcement features (2025-12)."""

from datetime import datetime, timedelta, timezone

import pytest

from mindcore.v2.flr.reinforcement import (
    BatchSignalResult,
    CrossMemoryReinforcer,
    # Enhanced features
    ImportanceAdjuster,
    ImportanceAdjustment,
    NegativeSignalDecay,
    ReinforcementSignal,
    RelatedMemorySignal,
    RobustReinforcement,
    SignalSource,
    SignalType,
    process_signal_batch,
)


class TestImportanceAdjuster:
    """Tests for ImportanceAdjuster."""

    def test_no_adjustment_insufficient_signals(self):
        """Test that no adjustment is made with insufficient signals."""
        adjuster = ImportanceAdjuster()
        reinforcement = RobustReinforcement()

        # Only 2 signals, need 3
        reinforcement.apply_simple_signal(0.8)
        reinforcement.apply_simple_signal(0.7)

        result = adjuster.calculate_adjustment(
            current_importance=0.5,
            reinforcement=reinforcement,
            min_signals=3,
        )

        assert result.adjustment == 0.0
        assert result.new_importance == 0.5
        assert "Insufficient" in result.reason

    def test_positive_adjustment(self):
        """Test positive importance adjustment."""
        adjuster = ImportanceAdjuster()
        reinforcement = RobustReinforcement()

        # Add strong positive signals
        for _ in range(5):
            reinforcement.apply_simple_signal(
                0.9,
                source=SignalSource.USER_EXPLICIT,
            )

        result = adjuster.calculate_adjustment(
            current_importance=0.5,
            reinforcement=reinforcement,
        )

        assert result.adjustment > 0
        assert result.new_importance > 0.5
        assert "Increased" in result.reason

    def test_negative_adjustment(self):
        """Test negative importance adjustment."""
        adjuster = ImportanceAdjuster()
        reinforcement = RobustReinforcement()

        # Add negative signals
        for _ in range(5):
            reinforcement.apply_simple_signal(
                -0.8,
                source=SignalSource.USER_EXPLICIT,
            )

        result = adjuster.calculate_adjustment(
            current_importance=0.5,
            reinforcement=reinforcement,
        )

        assert result.adjustment < 0
        assert result.new_importance < 0.5
        assert "Decreased" in result.reason

    def test_bounds_enforcement(self):
        """Test that importance stays within bounds."""
        adjuster = ImportanceAdjuster(
            min_importance=0.1,
            max_importance=0.9,
        )
        reinforcement = RobustReinforcement()

        # Try to push importance above max
        for _ in range(20):
            reinforcement.apply_simple_signal(1.0)

        result = adjuster.calculate_adjustment(
            current_importance=0.85,
            reinforcement=reinforcement,
        )

        assert result.new_importance <= 0.9

        # Try to push importance below min
        reinforcement2 = RobustReinforcement()
        for _ in range(20):
            reinforcement2.apply_simple_signal(-1.0)

        result2 = adjuster.calculate_adjustment(
            current_importance=0.15,
            reinforcement=reinforcement2,
        )

        assert result2.new_importance >= 0.1


class TestCrossMemoryReinforcer:
    """Tests for CrossMemoryReinforcer."""

    def test_no_propagation_weak_signal(self):
        """Test that weak signals don't propagate."""
        reinforcer = CrossMemoryReinforcer(min_signal_for_propagation=0.5)

        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.3,  # Below threshold
            source=SignalSource.USER_EXPLICIT,
        )

        candidate_memories = [
            {"memory_id": "mem_2", "topics": ["billing"], "session_id": "s1", "entities": []},
        ]

        result = reinforcer.propagate_signal(
            source_memory_id="mem_1",
            source_topics=["billing"],
            source_session_id="s1",
            source_entities=[],
            signal=signal,
            candidate_memories=candidate_memories,
        )

        assert len(result) == 0

    def test_propagation_topic_overlap(self):
        """Test propagation based on topic overlap."""
        reinforcer = CrossMemoryReinforcer(
            min_signal_for_propagation=0.5,
            topic_attenuation=0.3,
        )

        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )

        # Need > 50% topic overlap for "topic_overlap" relationship
        # source: ["billing", "orders"] (2 topics)
        # target: ["billing", "orders"] (2 topics) -> overlap = 2/2 = 100% > 50%
        candidate_memories = [
            {
                "memory_id": "mem_2",
                "topics": ["billing", "orders"],
                "session_id": "s2",
                "entities": [],
            },
            {"memory_id": "mem_3", "topics": ["unrelated"], "session_id": "s3", "entities": []},
        ]

        result = reinforcer.propagate_signal(
            source_memory_id="mem_1",
            source_topics=["billing", "orders"],
            source_session_id="s1",
            source_entities=[],
            signal=signal,
            candidate_memories=candidate_memories,
        )

        # Should propagate to mem_2 (topic overlap) but not mem_3 (no overlap)
        assert len(result) == 1
        assert result[0].memory_id == "mem_2"
        assert result[0].relationship == "topic_overlap"
        assert result[0].signal.value < signal.value  # Attenuated

    def test_propagation_session_match(self):
        """Test propagation based on session match."""
        reinforcer = CrossMemoryReinforcer(
            min_signal_for_propagation=0.5,
            session_attenuation=0.4,
        )

        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )

        candidate_memories = [
            {"memory_id": "mem_2", "topics": ["billing"], "session_id": "s1", "entities": []},
        ]

        result = reinforcer.propagate_signal(
            source_memory_id="mem_1",
            source_topics=["billing"],
            source_session_id="s1",  # Same session
            source_entities=[],
            signal=signal,
            candidate_memories=candidate_memories,
        )

        assert len(result) == 1
        assert result[0].relationship == "same_session"

    def test_propagation_entity_overlap(self):
        """Test propagation based on entity overlap."""
        reinforcer = CrossMemoryReinforcer(
            min_signal_for_propagation=0.5,
            entity_attenuation=0.5,
        )

        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )

        # Need > 50% entity overlap for "entity_overlap" relationship
        # source_entities: ["Order #12345", "John Doe"] (2 entities)
        # target_entities: ["Order #12345", "John Doe"] (2 entities)
        # overlap = 2/2 = 100% > 50%
        candidate_memories = [
            {
                "memory_id": "mem_2",
                "topics": ["orders"],
                "session_id": "s2",
                "entities": ["Order #12345", "John Doe"],
            },
        ]

        result = reinforcer.propagate_signal(
            source_memory_id="mem_1",
            source_topics=["shipping"],
            source_session_id="s1",
            source_entities=["Order #12345", "John Doe"],
            signal=signal,
            candidate_memories=candidate_memories,
        )

        assert len(result) == 1
        assert result[0].relationship == "entity_overlap"

    def test_skip_source_memory(self):
        """Test that source memory is skipped."""
        reinforcer = CrossMemoryReinforcer(min_signal_for_propagation=0.5)

        signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.8,
            source=SignalSource.USER_EXPLICIT,
        )

        candidate_memories = [
            {"memory_id": "mem_1", "topics": ["billing"], "session_id": "s1", "entities": []},
            {"memory_id": "mem_2", "topics": ["billing"], "session_id": "s1", "entities": []},
        ]

        result = reinforcer.propagate_signal(
            source_memory_id="mem_1",  # Same as first candidate
            source_topics=["billing"],
            source_session_id="s1",
            source_entities=[],
            signal=signal,
            candidate_memories=candidate_memories,
        )

        # Should only return mem_2, not mem_1
        assert len(result) == 1
        assert result[0].memory_id == "mem_2"


class TestNegativeSignalDecay:
    """Tests for NegativeSignalDecay."""

    def test_decay_removes_old_negative_signals(self):
        """Test that old negative signals are removed."""
        decay = NegativeSignalDecay(
            negative_decay_multiplier=2.0,
            recovery_threshold=-0.3,
        )

        reinforcement = RobustReinforcement(decay_half_life_hours=24)

        # Add an old negative signal
        old_signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=-0.8,
            source=SignalSource.USER_EXPLICIT,
            timestamp=datetime.now(timezone.utc) - timedelta(days=30),
        )
        reinforcement.signal_history.append(old_signal)

        # Add a recent positive signal
        recent_signal = ReinforcementSignal(
            signal_type=SignalType.RELEVANCE,
            value=0.5,
            source=SignalSource.USER_EXPLICIT,
            timestamp=datetime.now(timezone.utc),
        )
        reinforcement.signal_history.append(recent_signal)

        initial_count = len(reinforcement.signal_history)
        removed = decay.apply_to_reinforcement(reinforcement)

        assert removed >= 1
        assert len(reinforcement.signal_history) < initial_count

    def test_positive_signals_preserved(self):
        """Test that positive signals are not affected."""
        decay = NegativeSignalDecay()
        reinforcement = RobustReinforcement()

        # Add positive signals of various ages
        for days_ago in [1, 7, 30]:
            signal = ReinforcementSignal(
                signal_type=SignalType.RELEVANCE,
                value=0.8,
                source=SignalSource.USER_EXPLICIT,
                timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
            )
            reinforcement.signal_history.append(signal)

        initial_count = len(reinforcement.signal_history)
        removed = decay.apply_to_reinforcement(reinforcement)

        assert removed == 0
        assert len(reinforcement.signal_history) == initial_count


class TestBatchSignalProcessing:
    """Tests for batch signal processing."""

    def test_process_batch_basic(self):
        """Test basic batch processing."""
        signals = [
            (
                "mem_1",
                RobustReinforcement(),
                ReinforcementSignal(
                    signal_type=SignalType.RELEVANCE,
                    value=0.8,
                    source=SignalSource.USER_EXPLICIT,
                ),
                0.5,  # importance
                ["billing"],  # topics
                "s1",  # session_id
                [],  # entities
            ),
            (
                "mem_2",
                RobustReinforcement(),
                ReinforcementSignal(
                    signal_type=SignalType.USEFULNESS,
                    value=0.6,
                    source=SignalSource.LLM_EVALUATION,
                ),
                0.5,
                ["orders"],
                "s1",
                [],
            ),
        ]

        results = process_signal_batch(signals)

        assert len(results) == 2
        assert all(isinstance(r, BatchSignalResult) for r in results)
        assert results[0].memory_id == "mem_1"
        assert results[1].memory_id == "mem_2"

    def test_process_batch_with_importance_adjuster(self):
        """Test batch processing with importance adjuster."""
        reinforcement = RobustReinforcement()

        # Pre-populate with signals to meet threshold
        for _ in range(5):
            reinforcement.apply_simple_signal(0.8)

        signals = [
            (
                "mem_1",
                reinforcement,
                ReinforcementSignal(
                    signal_type=SignalType.RELEVANCE,
                    value=0.9,
                    source=SignalSource.USER_EXPLICIT,
                ),
                0.5,
                ["billing"],
                "s1",
                [],
            ),
        ]

        adjuster = ImportanceAdjuster()
        results = process_signal_batch(signals, importance_adjuster=adjuster)

        assert len(results) == 1
        assert results[0].importance_adjustment is not None
        assert results[0].importance_adjustment.adjustment > 0

    def test_process_batch_with_cross_memory(self):
        """Test batch processing with cross-memory reinforcement."""
        signals = [
            (
                "mem_1",
                RobustReinforcement(),
                ReinforcementSignal(
                    signal_type=SignalType.RELEVANCE,
                    value=0.9,
                    source=SignalSource.USER_EXPLICIT,
                ),
                0.5,
                ["billing"],
                "s1",
                [],
            ),
        ]

        all_memories = [
            {"memory_id": "mem_2", "topics": ["billing"], "session_id": "s1", "entities": []},
            {"memory_id": "mem_3", "topics": ["billing"], "session_id": "s2", "entities": []},
        ]

        reinforcer = CrossMemoryReinforcer(min_signal_for_propagation=0.5)
        results = process_signal_batch(
            signals,
            cross_memory_reinforcer=reinforcer,
            all_memories=all_memories,
        )

        assert len(results) == 1
        assert len(results[0].related_signals) > 0


class TestRobustReinforcementEnhancements:
    """Tests for enhancements to RobustReinforcement."""

    def test_neutral_decay_application(self):
        """Test apply_neutral_decay method."""
        reinforcement = RobustReinforcement()

        # Apply a positive signal first
        reinforcement.apply_simple_signal(0.5)
        initial_count = len(reinforcement.signal_history)

        # Apply neutral decay
        reinforcement.apply_neutral_decay(hours_since_access=24)

        # Should have added a small negative signal
        assert len(reinforcement.signal_history) == initial_count + 1
        assert reinforcement.signal_history[-1].value < 0

    def test_signal_breakdown(self):
        """Test get_signal_breakdown method."""
        reinforcement = RobustReinforcement()

        # Add various signal types
        reinforcement.apply_simple_signal(0.8, signal_type=SignalType.RELEVANCE)
        reinforcement.apply_simple_signal(0.6, signal_type=SignalType.USEFULNESS)
        reinforcement.apply_simple_signal(-0.3, signal_type=SignalType.RELEVANCE)

        breakdown = reinforcement.get_signal_breakdown()

        assert SignalType.RELEVANCE in breakdown
        assert SignalType.USEFULNESS in breakdown
        # RELEVANCE has two signals (0.8 + -0.3) / 2 = 0.25
        assert breakdown[SignalType.RELEVANCE] == pytest.approx(0.25, abs=0.01)
        assert breakdown[SignalType.USEFULNESS] == pytest.approx(0.6, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
