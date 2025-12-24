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
    from mindcore.v2.flr.reinforcement import (
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

References:
- UCB1: https://en.wikipedia.org/wiki/Multi-armed_bandit#UCB1
- Exponential Decay: Standard RL temporal discounting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
            if age_seconds < 0:
                age_seconds = 0  # Handle future timestamps

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
        new_avg = sum(s.value for s in self.signal_history[mid:]) / (
            len(self.signal_history) - mid
        )

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
        new_avg = sum(s.value for s in self.signal_history[mid:]) / (
            len(self.signal_history) - mid
        )

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
                {k.value: v for k, v in self.type_weights.items()}
                if self.type_weights
                else None
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
