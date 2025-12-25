"""Cross-Agent Signal Aggregation.

Aggregates reinforcement signals from multiple agents to create
shared memory importance scores.

Features:
- Trust-weighted signal aggregation
- Conflict resolution for contradictory signals
- Temporal decay for stale signals
- Cross-namespace signal propagation

Signal Flow:
    Agent A (FLR) ──┐
                    ├──► Signal Aggregator ──► Aggregated Score ──► CLST
    Agent B (FLR) ──┤
                    │
    Agent C (FLR) ──┘

Example:
    aggregator = CrossAgentSignalAggregator(
        trust_policy=TrustPolicy.NAMESPACE_WEIGHTED,
    )

    # Agents send signals
    aggregator.add_signal(memory_id, agent_a_id, 0.8, agent_a_scope)
    aggregator.add_signal(memory_id, agent_b_id, 0.6, agent_b_scope)
    aggregator.add_signal(memory_id, agent_c_id, -0.2, agent_c_scope)

    # Get aggregated score
    score = aggregator.get_aggregated_score(memory_id)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .access_control import AccessScope
from .namespace import MemoryNamespace


class TrustPolicy(str, Enum):
    """Policies for weighting signals from different agents."""

    EQUAL = "equal"  # All agents equally weighted
    NAMESPACE_WEIGHTED = "namespace_weighted"  # Same namespace = higher weight
    REPUTATION_BASED = "reputation_based"  # Based on agent accuracy history
    RECENCY_WEIGHTED = "recency_weighted"  # Recent signals weighted more
    HIERARCHICAL = "hierarchical"  # Higher scope = lower weight


@dataclass
class SignalWeight:
    """Configuration for signal weighting.

    Attributes:
        base_weight: Default weight for all signals
        same_team_bonus: Bonus weight for same-team agents
        same_department_bonus: Bonus weight for same-department agents
        same_agent_type_bonus: Bonus weight for same agent type
        decay_half_life_hours: Half-life for temporal decay
        min_weight: Minimum weight (prevents zero influence)
    """

    base_weight: float = 1.0
    same_team_bonus: float = 0.5
    same_department_bonus: float = 0.3
    same_agent_type_bonus: float = 0.2
    decay_half_life_hours: float = 168.0  # 1 week
    min_weight: float = 0.1


@dataclass
class AgentSignal:
    """A reinforcement signal from a specific agent.

    Attributes:
        agent_id: ID of the signaling agent
        agent_scope: Scope of the agent
        value: Signal value (-1 to 1)
        timestamp: When the signal was created
        context: Optional context about the signal
    """

    agent_id: str
    agent_scope: AccessScope
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal value."""
        self.value = max(-1.0, min(1.0, self.value))


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple agents.

    Attributes:
        memory_id: ID of the memory
        signals: All individual signals
        aggregated_value: Combined signal value
        confidence: Confidence in the aggregated value
        last_updated: When last signal was added
    """

    memory_id: str
    signals: dict[str, AgentSignal] = field(default_factory=dict)
    aggregated_value: float = 0.0
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signal_count(self) -> int:
        """Number of unique agent signals."""
        return len(self.signals)

    @property
    def agreement_ratio(self) -> float:
        """Ratio of signals that agree on direction."""
        if not self.signals:
            return 0.0

        positive = sum(1 for s in self.signals.values() if s.value > 0)
        negative = sum(1 for s in self.signals.values() if s.value < 0)
        total = len(self.signals)

        return max(positive, negative) / total if total > 0 else 0.0


@dataclass
class AgentReputation:
    """Tracks an agent's signal accuracy over time.

    Used for reputation-based weighting.
    """

    agent_id: str
    total_signals: int = 0
    accurate_signals: int = 0
    last_signal: datetime | None = None

    @property
    def accuracy(self) -> float:
        """Calculate accuracy rate."""
        if self.total_signals == 0:
            return 0.5  # Neutral for new agents
        return self.accurate_signals / self.total_signals

    def record_accuracy(self, was_accurate: bool) -> None:
        """Record signal accuracy."""
        self.total_signals += 1
        if was_accurate:
            self.accurate_signals += 1
        self.last_signal = datetime.now(timezone.utc)


@dataclass
class CrossAgentSignalAggregator:
    """Aggregates reinforcement signals from multiple agents.

    Supports various trust policies for weighting signals.

    Attributes:
        trust_policy: How to weight signals from different agents
        weight_config: Configuration for signal weights
        signals: Memory ID -> AggregatedSignal mapping
        reputations: Agent ID -> AgentReputation mapping
    """

    trust_policy: TrustPolicy = TrustPolicy.NAMESPACE_WEIGHTED
    weight_config: SignalWeight = field(default_factory=SignalWeight)
    signals: dict[str, AggregatedSignal] = field(default_factory=dict)
    reputations: dict[str, AgentReputation] = field(default_factory=dict)

    def add_signal(
        self,
        memory_id: str,
        agent_id: str,
        value: float,
        agent_scope: AccessScope,
        context: dict[str, Any] | None = None,
        reference_scope: AccessScope | None = None,
    ) -> float:
        """Add a signal from an agent.

        Args:
            memory_id: ID of the memory being reinforced
            agent_id: ID of the signaling agent
            value: Signal value (-1 to 1)
            agent_scope: Scope of the agent
            context: Optional signal context
            reference_scope: Reference for namespace weighting

        Returns:
            New aggregated score for the memory
        """
        # Create or get aggregated signal
        if memory_id not in self.signals:
            self.signals[memory_id] = AggregatedSignal(memory_id=memory_id)

        aggregated = self.signals[memory_id]

        # Create agent signal
        signal = AgentSignal(
            agent_id=agent_id,
            agent_scope=agent_scope,
            value=value,
            context=context or {},
        )

        # Store signal (overwrites previous from same agent)
        aggregated.signals[agent_id] = signal
        aggregated.last_updated = datetime.now(timezone.utc)

        # Recompute aggregated value
        aggregated.aggregated_value = self._compute_aggregation(
            aggregated,
            reference_scope,
        )

        # Update confidence based on agreement and count
        aggregated.confidence = self._compute_confidence(aggregated)

        return aggregated.aggregated_value

    def get_aggregated_score(
        self,
        memory_id: str,
        reference_scope: AccessScope | None = None,
    ) -> float:
        """Get the aggregated score for a memory.

        Args:
            memory_id: Memory ID
            reference_scope: Reference for namespace weighting

        Returns:
            Aggregated score (-1 to 1), or 0 if no signals
        """
        if memory_id not in self.signals:
            return 0.0

        aggregated = self.signals[memory_id]

        # Recompute with potential new reference scope
        return self._compute_aggregation(aggregated, reference_scope)

    def get_signal_details(
        self,
        memory_id: str,
    ) -> AggregatedSignal | None:
        """Get full signal details for a memory."""
        return self.signals.get(memory_id)

    def remove_agent_signals(
        self,
        agent_id: str,
    ) -> list[str]:
        """Remove all signals from a specific agent.

        Useful when an agent is deprecated or removed.

        Args:
            agent_id: Agent to remove

        Returns:
            List of memory IDs that were updated
        """
        updated = []
        for memory_id, aggregated in self.signals.items():
            if agent_id in aggregated.signals:
                del aggregated.signals[agent_id]
                aggregated.aggregated_value = self._compute_aggregation(aggregated, None)
                aggregated.confidence = self._compute_confidence(aggregated)
                updated.append(memory_id)
        return updated

    def get_memories_by_signal_strength(
        self,
        min_score: float = 0.0,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[tuple[str, float, float]]:
        """Get memories sorted by aggregated signal strength.

        Args:
            min_score: Minimum aggregated score
            min_confidence: Minimum confidence
            limit: Maximum results

        Returns:
            List of (memory_id, score, confidence) tuples
        """
        results = []
        for memory_id, aggregated in self.signals.items():
            if aggregated.aggregated_value >= min_score and aggregated.confidence >= min_confidence:
                results.append((
                    memory_id,
                    aggregated.aggregated_value,
                    aggregated.confidence,
                ))

        # Sort by score descending
        results.sort(key=lambda x: -x[1])
        return results[:limit]

    def update_reputation(
        self,
        agent_id: str,
        was_accurate: bool,
    ) -> float:
        """Update an agent's reputation.

        Args:
            agent_id: Agent to update
            was_accurate: Whether their signal was accurate

        Returns:
            New accuracy score
        """
        if agent_id not in self.reputations:
            self.reputations[agent_id] = AgentReputation(agent_id=agent_id)

        self.reputations[agent_id].record_accuracy(was_accurate)
        return self.reputations[agent_id].accuracy

    def _compute_aggregation(
        self,
        aggregated: AggregatedSignal,
        reference_scope: AccessScope | None,
    ) -> float:
        """Compute aggregated value based on trust policy."""
        if not aggregated.signals:
            return 0.0

        if self.trust_policy == TrustPolicy.EQUAL:
            return self._aggregate_equal(aggregated)
        elif self.trust_policy == TrustPolicy.NAMESPACE_WEIGHTED:
            return self._aggregate_namespace_weighted(aggregated, reference_scope)
        elif self.trust_policy == TrustPolicy.REPUTATION_BASED:
            return self._aggregate_reputation_based(aggregated)
        elif self.trust_policy == TrustPolicy.RECENCY_WEIGHTED:
            return self._aggregate_recency_weighted(aggregated)
        elif self.trust_policy == TrustPolicy.HIERARCHICAL:
            return self._aggregate_hierarchical(aggregated)
        else:
            return self._aggregate_equal(aggregated)

    def _aggregate_equal(self, aggregated: AggregatedSignal) -> float:
        """Simple average of all signals."""
        if not aggregated.signals:
            return 0.0
        total = sum(s.value for s in aggregated.signals.values())
        return total / len(aggregated.signals)

    def _aggregate_namespace_weighted(
        self,
        aggregated: AggregatedSignal,
        reference_scope: AccessScope | None,
    ) -> float:
        """Weight signals by namespace proximity."""
        if not aggregated.signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in aggregated.signals.values():
            weight = self.weight_config.base_weight

            if reference_scope:
                # Bonus for same team
                if (reference_scope.team and
                    signal.agent_scope.team == reference_scope.team):
                    weight += self.weight_config.same_team_bonus

                # Bonus for same department
                elif (reference_scope.department and
                      signal.agent_scope.department == reference_scope.department):
                    weight += self.weight_config.same_department_bonus

                # Bonus for same agent type
                if (reference_scope.agent_type and
                    signal.agent_scope.agent_type == reference_scope.agent_type):
                    weight += self.weight_config.same_agent_type_bonus

            weight = max(weight, self.weight_config.min_weight)
            weighted_sum += signal.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _aggregate_reputation_based(self, aggregated: AggregatedSignal) -> float:
        """Weight signals by agent reputation."""
        if not aggregated.signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in aggregated.signals.values():
            reputation = self.reputations.get(signal.agent_id)
            weight = reputation.accuracy if reputation else 0.5
            weight = max(weight, self.weight_config.min_weight)

            weighted_sum += signal.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _aggregate_recency_weighted(self, aggregated: AggregatedSignal) -> float:
        """Weight signals by recency with exponential decay."""
        if not aggregated.signals:
            return 0.0

        now = datetime.now(timezone.utc)
        decay_constant = math.log(2) / (self.weight_config.decay_half_life_hours * 3600)

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in aggregated.signals.values():
            age_seconds = (now - signal.timestamp).total_seconds()
            decay_factor = math.exp(-decay_constant * age_seconds)
            weight = max(decay_factor, self.weight_config.min_weight)

            weighted_sum += signal.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _aggregate_hierarchical(self, aggregated: AggregatedSignal) -> float:
        """Weight signals inversely by scope breadth.

        More specific scopes (team < department < org) get higher weight.
        """
        if not aggregated.signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in aggregated.signals.values():
            # Determine scope level
            if signal.agent_scope.team:
                weight = 1.0  # Team level = highest weight
            elif signal.agent_scope.department:
                weight = 0.7  # Department level
            else:
                weight = 0.4  # Org level = lowest weight

            weight = max(weight, self.weight_config.min_weight)
            weighted_sum += signal.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _compute_confidence(self, aggregated: AggregatedSignal) -> float:
        """Compute confidence based on signal count and agreement."""
        if not aggregated.signals:
            return 0.0

        # Base confidence from number of signals (logarithmic)
        count_factor = min(1.0, math.log(len(aggregated.signals) + 1) / math.log(10))

        # Agreement factor
        agreement_factor = aggregated.agreement_ratio

        # Combined confidence
        return (count_factor * 0.5 + agreement_factor * 0.5)

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregator state."""
        return {
            "trust_policy": self.trust_policy.value,
            "signals": {
                mid: {
                    "aggregated_value": agg.aggregated_value,
                    "confidence": agg.confidence,
                    "signal_count": agg.signal_count,
                }
                for mid, agg in self.signals.items()
            },
            "reputations": {
                aid: {
                    "accuracy": rep.accuracy,
                    "total_signals": rep.total_signals,
                }
                for aid, rep in self.reputations.items()
            },
        }
