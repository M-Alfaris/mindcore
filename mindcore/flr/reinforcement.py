"""Robust Reinforcement Learning for FLR.

This module provides a production-grade reinforcement system for memory recall
with temporal decay, multi-signal types, exploration balancing, and trend tracking.

Key Features:
- Temporal Decay: Reinforcement scores decay over time
- Multi-Signal Types: Different signal types with configurable weights
- Signal History: Track individual signals with timestamps for trend analysis
- Exploration Factor: UCB-like exploration to avoid exploitation traps
- Context-Aware Signals: Weight signals by context relevance
- Source Weighting: Different weights for different signal sources
- Trend Detection: Moving average for momentum tracking

Example:
    from mindcore.flr.reinforcement import (
        RobustReinforcement,
        ReinforcementSignal,
        SignalType,
        SignalSource,
    )

    # Create reinforcement tracker
    reinforcement = RobustReinforcement(decay_half_life_hours=168)

    # Apply a signal
    signal = ReinforcementSignal(
        signal_type=SignalType.RELEVANCE,
        value=0.8,
        source=SignalSource.USER_EXPLICIT,
        context_similarity=0.9,
    )
    new_score = reinforcement.apply_signal(signal)

    # Get effective score with exploration bonus
    effective = reinforcement.get_effective_score(exploration_factor=0.1)

    # Check if memory is trending up
    if reinforcement.is_trending_up():
        print("Memory is gaining relevance")

Enhancements (2025-12):
- Importance Adjustment: Reinforcement signals affect memory importance scores
- Cross-Memory Signals: Related memories receive attenuated reinforcement
- Negative Signal Decay: Negative signals decay faster to allow recovery
- Signal Batching: Efficient batch processing of multiple signals

References:
- UCB1: https://en.wikipedia.org/wiki/Multi-armed_bandit#UCB1
- Exponential Decay: Standard RL temporal discounting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    """Types of reinforcement signals.

    Different signal types capture different aspects of memory quality.
    """

    RELEVANCE = "relevance"  # How relevant was the memory to the query?
    USEFULNESS = "usefulness"  # Did the memory help solve the task?
    CORRECTNESS = "correctness"  # Was the information in the memory accurate?
    TIMELINESS = "timeliness"  # Was the memory appropriately recent/current?
    COMPLETENESS = "completeness"  # Did the memory provide sufficient detail?


class SignalSource(str, Enum):
    """Sources of reinforcement signals.

    Different sources have different reliability weights.
    """

    USER_EXPLICIT = "user_explicit"  # Direct user feedback (thumbs up/down)
    USER_IMPLICIT = "user_implicit"  # Inferred from user behavior
    LLM_EVALUATION = "llm_evaluation"  # LLM self-assessment
    AUTOMATED_METRIC = "automated_metric"  # System metrics (retrieval time, etc.)
    CROSS_AGENT = "cross_agent"  # Feedback from other agents


# Default weights for signal sources (0-1 scale)
DEFAULT_SOURCE_WEIGHTS: dict[SignalSource, float] = {
    SignalSource.USER_EXPLICIT: 1.0,  # Most reliable
    SignalSource.USER_IMPLICIT: 0.7,  # Good but not explicit
    SignalSource.LLM_EVALUATION: 0.5,  # Useful but potentially biased
    SignalSource.AUTOMATED_METRIC: 0.3,  # Objective but limited scope
    SignalSource.CROSS_AGENT: 0.6,  # Trust other agents moderately
}

# Default weights for signal types (0-1 scale)
DEFAULT_TYPE_WEIGHTS: dict[SignalType, float] = {
    SignalType.RELEVANCE: 0.35,  # Most important for retrieval
    SignalType.USEFULNESS: 0.30,  # Important for task completion
    SignalType.CORRECTNESS: 0.20,  # Critical for trust
    SignalType.TIMELINESS: 0.10,  # Context-dependent
    SignalType.COMPLETENESS: 0.05,  # Nice to have
}


@dataclass
class ReinforcementSignal:
    """A single reinforcement signal with metadata."""

    signal_type: SignalType
    value: float  # -1.0 to 1.0
    source: SignalSource
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context information
    context_similarity: float = 1.0  # How similar was the retrieval context? (0-1)
    query_id: str | None = None  # Which query triggered this?
    session_id: str | None = None  # Which session?

    # Optional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and clamp values."""
        self.value = max(-1.0, min(1.0, self.value))
        self.context_similarity = max(0.0, min(1.0, self.context_similarity))

    def get_weighted_value(
        self,
        source_weights: dict[SignalSource, float] | None = None,
        type_weights: dict[SignalType, float] | None = None,
    ) -> float:
        """Get the weighted value of this signal.

        Args:
            source_weights: Custom source weights (defaults to DEFAULT_SOURCE_WEIGHTS)
            type_weights: Custom type weights (defaults to DEFAULT_TYPE_WEIGHTS)

        Returns:
            Weighted signal value
        """
        source_weights = source_weights or DEFAULT_SOURCE_WEIGHTS
        type_weights = type_weights or DEFAULT_TYPE_WEIGHTS

        source_weight = source_weights.get(self.source, 0.5)
        type_weight = type_weights.get(self.signal_type, 0.2)

        # Apply context similarity as a multiplier
        return self.value * source_weight * type_weight * self.context_similarity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_type": self.signal_type.value,
            "value": self.value,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "context_similarity": self.context_similarity,
            "query_id": self.query_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReinforcementSignal:
        """Create from dictionary."""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            value=data["value"],
            source=SignalSource(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context_similarity=data.get("context_similarity", 1.0),
            query_id=data.get("query_id"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RobustReinforcement:
    """Robust reinforcement tracker for a single memory.

    Features:
    - Temporal decay with configurable half-life
    - Multi-signal aggregation with type/source weights
    - Signal history for trend detection
    - UCB-like exploration bonus
    - Moving average for momentum tracking
    """

    # Configuration
    decay_half_life_hours: float = 168.0  # 1 week half-life
    max_history_size: int = 100  # Keep last N signals
    moving_average_window: int = 10  # Window for trend detection

    # State
    signal_history: list[ReinforcementSignal] = field(default_factory=list)
    access_count: int = 0  # Total times this memory was retrieved
    reinforcement_count: int = 0  # Total times reinforcement was received

    # Cached aggregates (recalculated on demand)
    _cached_score: float | None = None
    _cache_timestamp: datetime | None = None
    _cache_ttl_seconds: float = 60.0  # Recalculate every minute

    # Custom weights (optional)
    source_weights: dict[SignalSource, float] | None = None
    type_weights: dict[SignalType, float] | None = None

    def apply_signal(self, signal: ReinforcementSignal) -> float:
        """Apply a reinforcement signal.

        Args:
            signal: The reinforcement signal to apply

        Returns:
            The new aggregated reinforcement score
        """
        self.signal_history.append(signal)
        self.reinforcement_count += 1

        # Trim history if too large
        if len(self.signal_history) > self.max_history_size:
            # Keep most recent signals
            self.signal_history = self.signal_history[-self.max_history_size :]

        # Invalidate cache
        self._cached_score = None

        return self.get_aggregated_score()

    def apply_simple_signal(
        self,
        value: float,
        signal_type: SignalType = SignalType.RELEVANCE,
        source: SignalSource = SignalSource.LLM_EVALUATION,
        context_similarity: float = 1.0,
    ) -> float:
        """Convenience method to apply a simple signal.

        Args:
            value: Signal value (-1 to 1)
            signal_type: Type of signal
            source: Source of signal
            context_similarity: Context similarity (0-1)

        Returns:
            New aggregated score
        """
        signal = ReinforcementSignal(
            signal_type=signal_type,
            value=value,
            source=source,
            context_similarity=context_similarity,
        )
        return self.apply_signal(signal)

    def record_access(self) -> None:
        """Record that this memory was accessed/retrieved."""
        self.access_count += 1

    def get_aggregated_score(self) -> float:
        """Get the aggregated reinforcement score with temporal decay.

        This is the main scoring method that:
        1. Applies temporal decay to each signal
        2. Weights signals by source and type
        3. Aggregates with diminishing returns

        Returns:
            Aggregated score in range [-1, 1]
        """
        # Check cache
        now = datetime.now(timezone.utc)
        if (
            self._cached_score is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            return self._cached_score

        if not self.signal_history:
            return 0.0

        # Calculate decay constant: score = initial * e^(-λt)
        # Half-life: 0.5 = e^(-λ * half_life) => λ = ln(2) / half_life
        decay_constant = math.log(2) / (self.decay_half_life_hours * 3600)  # per second

        weighted_sum = 0.0
        weight_sum = 0.0

        for signal in self.signal_history:
            # Calculate time since signal
            signal_time = signal.timestamp
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)

            age_seconds = (now - signal_time).total_seconds()
            age_seconds = max(age_seconds, 0)  # Handle future timestamps

            # Apply exponential decay
            decay_factor = math.exp(-decay_constant * age_seconds)

            # Get weighted signal value
            signal_value = signal.get_weighted_value(
                source_weights=self.source_weights,
                type_weights=self.type_weights,
            )

            # Accumulate with decay
            weighted_sum += signal_value * decay_factor
            weight_sum += decay_factor

        # Normalize and apply bounds
        if weight_sum > 0:
            raw_score = weighted_sum / weight_sum
        else:
            raw_score = 0.0

        # Apply diminishing returns (tanh-like compression)
        # This prevents extreme scores from single strong signals
        compressed_score = math.tanh(raw_score * 2) * 0.9 + raw_score * 0.1
        final_score = max(-1.0, min(1.0, compressed_score))

        # Cache result
        self._cached_score = final_score
        self._cache_timestamp = now

        return final_score

    def get_exploration_bonus(self, total_retrievals: int = 1000) -> float:
        """Get UCB-like exploration bonus.

        Less-accessed memories get a higher bonus to encourage exploration.
        Based on UCB1: bonus = sqrt(2 * ln(N) / n)

        Args:
            total_retrievals: Total retrievals across all memories (N)

        Returns:
            Exploration bonus (0 to ~1)
        """
        if self.access_count == 0:
            return 1.0  # Maximum exploration for unvisited

        if total_retrievals <= 0:
            total_retrievals = 1

        # UCB1 formula
        bonus = math.sqrt(2 * math.log(total_retrievals) / self.access_count)

        # Normalize to 0-1 range (cap at 1)
        return min(1.0, bonus)

    def get_effective_score(
        self,
        exploration_factor: float = 0.1,
        total_retrievals: int = 1000,
    ) -> float:
        """Get effective score combining exploitation and exploration.

        Args:
            exploration_factor: Weight for exploration bonus (0-1)
            total_retrievals: Total retrievals for UCB calculation

        Returns:
            Effective score for ranking
        """
        base_score = self.get_aggregated_score()
        exploration_bonus = self.get_exploration_bonus(total_retrievals)

        # Combine: (1 - ε) * exploitation + ε * exploration
        effective = (1 - exploration_factor) * base_score + exploration_factor * exploration_bonus

        return max(-1.0, min(1.0, effective))

    def get_moving_average(self, window: int | None = None) -> float:
        """Get moving average of recent signals for trend detection.

        Args:
            window: Number of recent signals to average

        Returns:
            Average of recent signal values
        """
        window = window or self.moving_average_window

        if not self.signal_history:
            return 0.0

        recent = self.signal_history[-window:]
        if not recent:
            return 0.0

        values = [s.value for s in recent]
        return sum(values) / len(values)

    def is_trending_up(self, threshold: float = 0.1) -> bool:
        """Check if reinforcement is trending upward.

        Compares recent signals to older signals.

        Args:
            threshold: Minimum difference to consider trending

        Returns:
            True if trending up
        """
        if len(self.signal_history) < self.moving_average_window * 2:
            return False

        # Compare recent half to older half
        mid = len(self.signal_history) // 2
        old_avg = sum(s.value for s in self.signal_history[:mid]) / mid
        new_avg = sum(s.value for s in self.signal_history[mid:]) / (len(self.signal_history) - mid)

        return (new_avg - old_avg) > threshold

    def is_trending_down(self, threshold: float = 0.1) -> bool:
        """Check if reinforcement is trending downward.

        Args:
            threshold: Minimum difference to consider trending

        Returns:
            True if trending down
        """
        if len(self.signal_history) < self.moving_average_window * 2:
            return False

        mid = len(self.signal_history) // 2
        old_avg = sum(s.value for s in self.signal_history[:mid]) / mid
        new_avg = sum(s.value for s in self.signal_history[mid:]) / (len(self.signal_history) - mid)

        return (old_avg - new_avg) > threshold

    def get_signal_breakdown(self) -> dict[SignalType, float]:
        """Get breakdown of scores by signal type.

        Returns:
            Dict mapping signal type to its contribution
        """
        breakdown: dict[SignalType, float] = {st: 0.0 for st in SignalType}
        counts: dict[SignalType, int] = {st: 0 for st in SignalType}

        for signal in self.signal_history:
            breakdown[signal.signal_type] += signal.value
            counts[signal.signal_type] += 1

        # Average per type
        for st in SignalType:
            if counts[st] > 0:
                breakdown[st] /= counts[st]

        return breakdown

    def apply_neutral_decay(self, hours_since_access: float) -> None:
        """Apply implicit negative signal for lack of reinforcement.

        When a memory is retrieved but not reinforced, this is implicit
        neutral/negative feedback that should be recorded.

        Args:
            hours_since_access: Hours since last access without reinforcement
        """
        if hours_since_access <= 0:
            return

        # Small negative signal proportional to time without reinforcement
        # Caps at -0.1 per day of no reinforcement
        decay_value = min(0.1, hours_since_access / 24 * 0.1)

        self.apply_simple_signal(
            value=-decay_value,
            signal_type=SignalType.RELEVANCE,
            source=SignalSource.AUTOMATED_METRIC,
            context_similarity=0.5,  # Low weight since we don't know context
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "decay_half_life_hours": self.decay_half_life_hours,
            "signal_history": [s.to_dict() for s in self.signal_history],
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
            "source_weights": (
                {k.value: v for k, v in self.source_weights.items()}
                if self.source_weights
                else None
            ),
            "type_weights": (
                {k.value: v for k, v in self.type_weights.items()} if self.type_weights else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustReinforcement:
        """Create from dictionary."""
        source_weights = None
        if data.get("source_weights"):
            source_weights = {SignalSource(k): v for k, v in data["source_weights"].items()}

        type_weights = None
        if data.get("type_weights"):
            type_weights = {SignalType(k): v for k, v in data["type_weights"].items()}

        reinforcement = cls(
            decay_half_life_hours=data.get("decay_half_life_hours", 168.0),
            access_count=data.get("access_count", 0),
            reinforcement_count=data.get("reinforcement_count", 0),
            source_weights=source_weights,
            type_weights=type_weights,
        )

        # Load signal history
        for sig_data in data.get("signal_history", []):
            reinforcement.signal_history.append(ReinforcementSignal.from_dict(sig_data))

        return reinforcement

    def get_legacy_score(self) -> float:
        """Get a simple score compatible with legacy reinforcement_score field.

        This provides backward compatibility with the naive implementation.

        Returns:
            Score in range [-1, 1]
        """
        return self.get_aggregated_score()


# Convenience function for batch reinforcement
def batch_reinforce(
    memories_signals: list[tuple[RobustReinforcement, ReinforcementSignal]],
) -> list[float]:
    """Apply reinforcement signals to multiple memories.

    Args:
        memories_signals: List of (reinforcement, signal) tuples

    Returns:
        List of new scores
    """
    return [r.apply_signal(s) for r, s in memories_signals]


# Factory for creating signals from simple feedback
def create_feedback_signal(
    value: float,
    is_user_feedback: bool = False,
    context_similarity: float = 1.0,
    query_id: str | None = None,
    session_id: str | None = None,
) -> ReinforcementSignal:
    """Create a reinforcement signal from simple feedback.

    Args:
        value: Feedback value (-1 to 1)
        is_user_feedback: Whether this is direct user feedback
        context_similarity: How similar the retrieval context was
        query_id: Associated query ID
        session_id: Associated session ID

    Returns:
        ReinforcementSignal configured appropriately
    """
    source = SignalSource.USER_EXPLICIT if is_user_feedback else SignalSource.LLM_EVALUATION

    # Infer signal type from value magnitude
    if abs(value) > 0.7:
        signal_type = SignalType.RELEVANCE  # Strong signals usually about relevance
    elif value > 0:
        signal_type = SignalType.USEFULNESS
    else:
        signal_type = SignalType.RELEVANCE

    return ReinforcementSignal(
        signal_type=signal_type,
        value=value,
        source=source,
        context_similarity=context_similarity,
        query_id=query_id,
        session_id=session_id,
    )


# =============================================================================
# Enhanced Reinforcement Features (2025-12)
# =============================================================================


@dataclass
class ImportanceAdjustment:
    """Result of importance adjustment from reinforcement."""

    original_importance: float
    new_importance: float
    adjustment: float
    reason: str
    reinforcement_score: float


class ImportanceAdjuster:
    """Adjusts memory importance based on reinforcement signals.

    The key insight: reinforcement should affect not just retrieval ranking,
    but also the underlying importance of the memory. A consistently
    positively-reinforced memory becomes more important over time.

    Example:
        adjuster = ImportanceAdjuster()

        # After reinforcement
        result = adjuster.calculate_adjustment(
            current_importance=0.5,
            reinforcement=memory.reinforcement,
        )

        if result.adjustment != 0:
            memory.importance = result.new_importance
    """

    def __init__(
        self,
        max_adjustment_per_signal: float = 0.05,
        min_importance: float = 0.1,
        max_importance: float = 0.95,
        trend_weight: float = 0.3,
        score_weight: float = 0.7,
    ):
        """Initialize adjuster.

        Args:
            max_adjustment_per_signal: Maximum importance change per signal
            min_importance: Floor for importance (never goes below)
            max_importance: Ceiling for importance (never exceeds)
            trend_weight: Weight for trend direction in adjustment
            score_weight: Weight for absolute score in adjustment
        """
        self.max_adjustment = max_adjustment_per_signal
        self.min_importance = min_importance
        self.max_importance = max_importance
        self.trend_weight = trend_weight
        self.score_weight = score_weight

    def calculate_adjustment(
        self,
        current_importance: float,
        reinforcement: RobustReinforcement,
        min_signals: int = 3,
    ) -> ImportanceAdjustment:
        """Calculate importance adjustment from reinforcement.

        Args:
            current_importance: Current memory importance (0-1)
            reinforcement: RobustReinforcement tracker for the memory
            min_signals: Minimum signals before adjusting

        Returns:
            ImportanceAdjustment with new importance and reasoning
        """
        score = reinforcement.get_aggregated_score()

        # Don't adjust with insufficient data
        if reinforcement.reinforcement_count < min_signals:
            return ImportanceAdjustment(
                original_importance=current_importance,
                new_importance=current_importance,
                adjustment=0.0,
                reason="Insufficient signals for adjustment",
                reinforcement_score=score,
            )

        # Calculate trend component
        trend_adj = 0.0
        if reinforcement.is_trending_up():
            trend_adj = 0.02  # Small boost for upward trend
        elif reinforcement.is_trending_down():
            trend_adj = -0.02  # Small penalty for downward trend

        # Calculate score-based component
        # Map score (-1 to 1) to adjustment (-max to +max)
        score_adj = score * self.max_adjustment

        # Weighted combination
        adjustment = self.trend_weight * trend_adj + self.score_weight * score_adj

        # Apply bounds
        new_importance = current_importance + adjustment
        new_importance = max(self.min_importance, min(self.max_importance, new_importance))
        actual_adjustment = new_importance - current_importance

        # Generate reason
        if actual_adjustment > 0.01:
            reason = f"Increased due to positive reinforcement (score={score:.2f})"
        elif actual_adjustment < -0.01:
            reason = f"Decreased due to negative reinforcement (score={score:.2f})"
        else:
            reason = "No significant change"

        return ImportanceAdjustment(
            original_importance=current_importance,
            new_importance=new_importance,
            adjustment=actual_adjustment,
            reason=reason,
            reinforcement_score=score,
        )


@dataclass
class RelatedMemorySignal:
    """Signal to apply to a related memory."""

    memory_id: str
    signal: ReinforcementSignal
    attenuation: float  # How much the signal was reduced
    relationship: str  # Why these memories are related


class CrossMemoryReinforcer:
    """Propagates reinforcement signals to related memories.

    When a memory receives a strong signal, related memories should
    receive attenuated versions of that signal. This helps:
    - Boost entire topic clusters when one memory is useful
    - Demote related memories when one is found to be wrong

    Relationships are determined by:
    - Shared topics
    - Same session
    - Same user
    - Entity overlap

    Example:
        reinforcer = CrossMemoryReinforcer()

        # After primary memory gets reinforced
        related_signals = reinforcer.propagate_signal(
            source_memory=memory,
            signal=signal,
            candidate_memories=nearby_memories,
        )

        for rel_signal in related_signals:
            apply_to_memory(rel_signal.memory_id, rel_signal.signal)
    """

    def __init__(
        self,
        min_signal_for_propagation: float = 0.5,
        topic_attenuation: float = 0.3,
        session_attenuation: float = 0.4,
        entity_attenuation: float = 0.5,
        max_propagation_depth: int = 1,
    ):
        """Initialize cross-memory reinforcer.

        Args:
            min_signal_for_propagation: Only propagate signals above this threshold
            topic_attenuation: Signal multiplier for topic-related memories
            session_attenuation: Signal multiplier for session-related memories
            entity_attenuation: Signal multiplier for entity-related memories
            max_propagation_depth: How many hops to propagate (1 = direct only)
        """
        self.min_signal = min_signal_for_propagation
        self.topic_attenuation = topic_attenuation
        self.session_attenuation = session_attenuation
        self.entity_attenuation = entity_attenuation
        self.max_depth = max_propagation_depth

    def propagate_signal(
        self,
        source_memory_id: str,
        source_topics: list[str],
        source_session_id: str | None,
        source_entities: list[str],
        signal: ReinforcementSignal,
        candidate_memories: list[dict[str, Any]],
    ) -> list[RelatedMemorySignal]:
        """Propagate a reinforcement signal to related memories.

        Args:
            source_memory_id: ID of the memory that received the signal
            source_topics: Topics of the source memory
            source_session_id: Session ID of source memory
            source_entities: Entities in source memory
            signal: The reinforcement signal received
            candidate_memories: List of candidate memories to check
                Each dict should have: memory_id, topics, session_id, entities

        Returns:
            List of RelatedMemorySignal for related memories
        """
        # Only propagate strong signals
        if abs(signal.value) < self.min_signal:
            return []

        related_signals = []
        source_topics_set = set(source_topics)
        source_entities_set = set(e.lower() for e in source_entities)

        for mem in candidate_memories:
            if mem.get("memory_id") == source_memory_id:
                continue  # Skip source memory

            # Calculate relationship and attenuation
            relationship, attenuation = self._calculate_relationship(
                source_topics_set=source_topics_set,
                source_session_id=source_session_id,
                source_entities_set=source_entities_set,
                target_topics=set(mem.get("topics", [])),
                target_session_id=mem.get("session_id"),
                target_entities=set(e.lower() for e in mem.get("entities", [])),
            )

            if relationship == "none":
                continue

            # Create attenuated signal
            attenuated_signal = ReinforcementSignal(
                signal_type=signal.signal_type,
                value=signal.value * attenuation,
                source=SignalSource.CROSS_AGENT,  # Mark as cross-memory
                context_similarity=attenuation,  # Lower similarity for related
                query_id=signal.query_id,
                session_id=signal.session_id,
                metadata={
                    "propagated_from": source_memory_id,
                    "relationship": relationship,
                    "attenuation": attenuation,
                },
            )

            related_signals.append(
                RelatedMemorySignal(
                    memory_id=mem["memory_id"],
                    signal=attenuated_signal,
                    attenuation=attenuation,
                    relationship=relationship,
                )
            )

        return related_signals

    def _calculate_relationship(
        self,
        source_topics_set: set[str],
        source_session_id: str | None,
        source_entities_set: set[str],
        target_topics: set[str],
        target_session_id: str | None,
        target_entities: set[str],
    ) -> tuple[str, float]:
        """Calculate relationship between source and target memory.

        Returns:
            Tuple of (relationship_type, attenuation_factor)
        """
        # Check topic overlap
        topic_overlap = len(source_topics_set & target_topics)
        topic_total = max(1, len(source_topics_set | target_topics))
        topic_score = topic_overlap / topic_total

        # Check session match
        session_match = source_session_id is not None and source_session_id == target_session_id

        # Check entity overlap
        entity_overlap = len(source_entities_set & target_entities)
        entity_total = max(1, len(source_entities_set | target_entities))
        entity_score = entity_overlap / entity_total if entity_total > 0 else 0

        # Determine primary relationship
        if entity_score > 0.5:
            return "entity_overlap", self.entity_attenuation * entity_score
        if session_match and topic_score > 0.3:
            return "same_session", self.session_attenuation
        if topic_score > 0.5:
            return "topic_overlap", self.topic_attenuation * topic_score

        return "none", 0.0


class NegativeSignalDecay:
    """Enhanced decay for negative signals.

    Negative signals should decay faster than positive signals to allow
    memories to "recover" from temporary negative feedback. This prevents
    a single bad interaction from permanently demoting a memory.

    Example:
        decay = NegativeSignalDecay()

        # Apply faster decay to negative signals in a reinforcement tracker
        decay.apply_faster_decay(reinforcement)
    """

    def __init__(
        self,
        negative_decay_multiplier: float = 2.0,
        recovery_threshold: float = -0.3,
    ):
        """Initialize negative signal decay.

        Args:
            negative_decay_multiplier: How much faster negatives decay
            recovery_threshold: Signals below this decay faster
        """
        self.multiplier = negative_decay_multiplier
        self.threshold = recovery_threshold

    def apply_to_reinforcement(
        self,
        reinforcement: RobustReinforcement,
    ) -> int:
        """Apply faster decay to negative signals.

        Modifies the reinforcement in place by removing old negative signals
        that have effectively decayed to zero.

        Args:
            reinforcement: RobustReinforcement to modify

        Returns:
            Number of signals removed
        """
        now = datetime.now(timezone.utc)
        original_count = len(reinforcement.signal_history)

        # Calculate enhanced decay for negative signals
        # Negative signals decay 2x faster than their normal half-life
        negative_half_life = reinforcement.decay_half_life_hours / self.multiplier
        negative_decay_constant = math.log(2) / (negative_half_life * 3600)

        # Filter signals - remove very old negative signals
        surviving_signals = []
        for signal in reinforcement.signal_history:
            if signal.value >= self.threshold:
                # Positive/neutral signals: keep with normal decay
                surviving_signals.append(signal)
            else:
                # Negative signals: check if they've decayed below threshold
                age_seconds = (now - signal.timestamp).total_seconds()
                remaining_value = signal.value * math.exp(-negative_decay_constant * age_seconds)

                # Keep if still significant
                if abs(remaining_value) > 0.05:
                    surviving_signals.append(signal)

        reinforcement.signal_history = surviving_signals
        reinforcement._cached_score = None  # Invalidate cache

        return original_count - len(surviving_signals)


# =============================================================================
# Batch Processing Utilities
# =============================================================================


@dataclass
class BatchSignalResult:
    """Result of batch signal processing."""

    memory_id: str
    new_score: float
    importance_adjustment: ImportanceAdjustment | None
    related_signals: list[RelatedMemorySignal]


def process_signal_batch(
    signals: list[
        tuple[
            str, RobustReinforcement, ReinforcementSignal, float, list[str], str | None, list[str]
        ]
    ],
    importance_adjuster: ImportanceAdjuster | None = None,
    cross_memory_reinforcer: CrossMemoryReinforcer | None = None,
    all_memories: list[dict[str, Any]] | None = None,
) -> list[BatchSignalResult]:
    """Process a batch of reinforcement signals efficiently.

    Args:
        signals: List of tuples containing:
            (memory_id, reinforcement, signal, current_importance, topics, session_id, entities)
        importance_adjuster: Optional importance adjuster
        cross_memory_reinforcer: Optional cross-memory reinforcer
        all_memories: List of all memories for cross-memory propagation

    Returns:
        List of BatchSignalResult with outcomes
    """
    results = []

    for memory_id, reinforcement, signal, importance, topics, session_id, entities in signals:
        # Apply the signal
        new_score = reinforcement.apply_signal(signal)

        # Calculate importance adjustment if adjuster provided
        importance_adj = None
        if importance_adjuster:
            importance_adj = importance_adjuster.calculate_adjustment(
                current_importance=importance,
                reinforcement=reinforcement,
            )

        # Calculate related signals if reinforcer provided
        related = []
        if cross_memory_reinforcer and all_memories:
            related = cross_memory_reinforcer.propagate_signal(
                source_memory_id=memory_id,
                source_topics=topics,
                source_session_id=session_id,
                source_entities=entities,
                signal=signal,
                candidate_memories=all_memories,
            )

        results.append(
            BatchSignalResult(
                memory_id=memory_id,
                new_score=new_score,
                importance_adjustment=importance_adj,
                related_signals=related,
            )
        )

    return results
