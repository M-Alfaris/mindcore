"""Session Aggregates - Weighted metadata aggregation for hierarchical memory retrieval.

This module provides session-level aggregation of memory metadata, enabling:
- Fast hierarchical queries (session → memories)
- Topic/category weight-based relevance scoring
- Importance/confidence statistics for filtering
- Reduced reliance on vector embeddings

The key insight: Topics and categories don't share the same importance/density
within a session. By tracking weights, we can query relevant sessions without
embeddings, then drill down to specific memories.

Example:
    # Query sessions by weighted topics
    sessions = clst.query_sessions(
        user_id="user_123",
        topic_hints=["orders", "shipping"],
        min_importance_avg=0.5,
    )

    # Get memories from relevant sessions
    memories = clst.query_memories_from_sessions(
        session_ids=[s.session_id for s in sessions],
        importance_threshold=0.3,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.flr import Memory


@dataclass
class SessionAggregate:
    """Aggregated metadata for a session.

    Contains weighted distributions of topics, categories, and other metadata
    to enable fast hierarchical queries without embeddings.

    Weight Calculation:
        topic_weight = (frequency * 0.4) + (avg_importance * 0.4) + (recency * 0.2)

        Where:
        - frequency: How often the topic appears in the session
        - avg_importance: Average importance of memories with this topic
        - recency: Exponential decay based on last mention time
    """

    session_id: str
    user_id: str
    agent_id: str | None = None

    # Weighted topic/category distributions (term -> weight 0-1)
    topic_weights: dict[str, float] = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)
    entity_weights: dict[str, float] = field(default_factory=dict)
    intent_weights: dict[str, float] = field(default_factory=dict)
    sentiment_weights: dict[str, float] = field(default_factory=dict)

    # Importance statistics
    importance_min: float = 1.0
    importance_max: float = 0.0
    importance_avg: float = 0.0
    importance_sum: float = 0.0  # For incremental avg calculation

    # Confidence statistics
    confidence_min: float = 1.0
    confidence_max: float = 0.0
    confidence_avg: float = 0.0
    confidence_sum: float = 0.0

    # Counts
    memory_count: int = 0
    message_count: int = 0

    # Time bounds
    started_at: datetime | None = None
    last_activity_at: datetime | None = None

    # Dominant values (most weighted)
    dominant_topic: str | None = None
    dominant_category: str | None = None
    dominant_sentiment: str | None = None
    max_urgency: str | None = None

    # Access control
    access_level: str = "private"  # Highest access level in session

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional: Session summary embedding (much cheaper than per-memory)
    summary_embedding: list[float] | None = None
    summary_text: str | None = None

    def update_from_memory(self, memory: Memory, decay_hours: float = 24.0) -> None:
        """Update aggregate statistics from a new memory.

        This is called incrementally when memories are added to a session,
        avoiding the need to recalculate all weights.

        Args:
            memory: The new memory to incorporate
            decay_hours: Half-life for recency decay (default 24 hours)
        """
        now = datetime.now(timezone.utc)

        # Update time bounds
        if self.started_at is None or (memory.created_at and memory.created_at < self.started_at):
            self.started_at = memory.created_at or now
        self.last_activity_at = now

        # Update counts
        self.memory_count += 1
        self.message_count += 1

        # Update importance statistics
        self.importance_min = min(self.importance_min, memory.importance)
        self.importance_max = max(self.importance_max, memory.importance)
        self.importance_sum += memory.importance
        self.importance_avg = self.importance_sum / self.memory_count

        # Update topic weights
        self._update_term_weights(
            self.topic_weights,
            memory.topics,
            memory.importance,
            now,
            decay_hours,
        )

        # Update category weights
        self._update_term_weights(
            self.category_weights,
            memory.categories,
            memory.importance,
            now,
            decay_hours,
        )

        # Update entity weights
        self._update_term_weights(
            self.entity_weights,
            memory.entities,
            memory.importance,
            now,
            decay_hours,
        )

        # Update sentiment weights
        if memory.sentiment:
            self._update_term_weights(
                self.sentiment_weights,
                [memory.sentiment],
                memory.importance,
                now,
                decay_hours,
            )

        # Update dominant values
        if self.topic_weights:
            self.dominant_topic = max(self.topic_weights, key=self.topic_weights.get)
        if self.category_weights:
            self.dominant_category = max(self.category_weights, key=self.category_weights.get)
        if self.sentiment_weights:
            self.dominant_sentiment = max(self.sentiment_weights, key=self.sentiment_weights.get)

        # Update access level (keep highest)
        access_hierarchy = {"private": 0, "team": 1, "shared": 2, "global": 3}
        current_level = access_hierarchy.get(self.access_level, 0)
        new_level = access_hierarchy.get(memory.access_level, 0)
        if new_level > current_level:
            self.access_level = memory.access_level

        self.updated_at = now

    def _update_term_weights(
        self,
        weights: dict[str, float],
        terms: list[str],
        importance: float,
        now: datetime,
        decay_hours: float,
    ) -> None:
        """Update term weights incrementally.

        Uses exponential moving average to incorporate new terms while
        decaying old ones.
        """
        if not terms:
            return

        # Decay factor for existing weights
        alpha = 0.3  # Learning rate for new observations

        for term in terms:
            if term in weights:
                # Exponential moving average
                old_weight = weights[term]
                new_contribution = importance * alpha
                weights[term] = old_weight * (1 - alpha) + new_contribution
            else:
                # New term - start with importance-based weight
                weights[term] = importance * alpha

        # Normalize weights to [0, 1]
        if weights:
            max_weight = max(weights.values())
            if max_weight > 0:
                for term in weights:
                    weights[term] = round(weights[term] / max_weight, 4)

    def calculate_relevance_score(
        self,
        topic_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        min_importance: float = 0.0,
        recency_weight: float = 0.2,
    ) -> float:
        """Calculate relevance score for this session given query hints.

        Args:
            topic_hints: Topics to match against
            category_hints: Categories to match against
            min_importance: Minimum importance threshold
            recency_weight: Weight for recency in scoring

        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        weights_used = 0

        # Topic matching
        if topic_hints:
            topic_score = 0.0
            for hint in topic_hints:
                if hint in self.topic_weights:
                    topic_score += self.topic_weights[hint]
            if topic_hints:
                topic_score /= len(topic_hints)
            score += topic_score * 0.4
            weights_used += 0.4

        # Category matching
        if category_hints:
            cat_score = 0.0
            for hint in category_hints:
                if hint in self.category_weights:
                    cat_score += self.category_weights[hint]
            if category_hints:
                cat_score /= len(category_hints)
            score += cat_score * 0.2
            weights_used += 0.2

        # Importance score
        if self.importance_avg >= min_importance:
            score += self.importance_avg * 0.25
            weights_used += 0.25

        # Recency score
        if self.last_activity_at:
            now = datetime.now(timezone.utc)
            age_hours = (now - self.last_activity_at).total_seconds() / 3600
            recency_score = math.exp(-age_hours / 168)  # 1 week decay
            score += recency_score * recency_weight
            weights_used += recency_weight

        # Normalize by weights used
        if weights_used > 0:
            score /= weights_used

        return min(1.0, max(0.0, score))

    def get_top_topics(self, limit: int = 5) -> list[tuple[str, float]]:
        """Get top weighted topics."""
        sorted_topics = sorted(self.topic_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:limit]

    def get_top_categories(self, limit: int = 3) -> list[tuple[str, float]]:
        """Get top weighted categories."""
        sorted_cats = sorted(self.category_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_cats[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "topic_weights": self.topic_weights,
            "category_weights": self.category_weights,
            "entity_weights": self.entity_weights,
            "intent_weights": self.intent_weights,
            "sentiment_weights": self.sentiment_weights,
            "importance_min": self.importance_min,
            "importance_max": self.importance_max,
            "importance_avg": self.importance_avg,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "confidence_avg": self.confidence_avg,
            "memory_count": self.memory_count,
            "message_count": self.message_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_activity_at": self.last_activity_at.isoformat()
            if self.last_activity_at
            else None,
            "dominant_topic": self.dominant_topic,
            "dominant_category": self.dominant_category,
            "dominant_sentiment": self.dominant_sentiment,
            "max_urgency": self.max_urgency,
            "access_level": self.access_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "summary_text": self.summary_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionAggregate:
        """Create from dictionary."""
        # Parse datetime fields
        for dt_field in ["started_at", "last_activity_at", "created_at", "updated_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        # Remove embedding if present but not needed for basic deserialization
        data.pop("summary_embedding", None)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HierarchicalQueryResult:
    """Result from hierarchical memory query."""

    memories: list[Any]  # Memory objects
    sessions: list[SessionAggregate]

    # Query statistics
    sessions_searched: int
    memories_returned: int
    query_latency_ms: float

    # Scoring info
    session_scores: dict[str, float] = field(default_factory=dict)

    # Data source results (from SVL)
    source_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memories": [m.to_dict() for m in self.memories],
            "sessions": [s.to_dict() for s in self.sessions],
            "sessions_searched": self.sessions_searched,
            "memories_returned": self.memories_returned,
            "query_latency_ms": self.query_latency_ms,
            "session_scores": self.session_scores,
            "source_data": self.source_data,
        }


class WeightCalculator:
    """Calculate topic/category weights from memory collections."""

    @staticmethod
    def calculate_weights_from_memories(
        memories: list[Any],
        decay_hours: float = 24.0,
    ) -> dict[str, dict[str, float]]:
        """Calculate all weights from a list of memories.

        Returns dict with topic_weights, category_weights, entity_weights, etc.
        """
        now = datetime.now(timezone.utc)

        topic_stats: dict[str, dict[str, float]] = {}
        category_stats: dict[str, dict[str, float]] = {}
        entity_stats: dict[str, dict[str, float]] = {}

        total_memories = len(memories)
        if total_memories == 0:
            return {
                "topic_weights": {},
                "category_weights": {},
                "entity_weights": {},
            }

        for memory in memories:
            # Calculate recency factor
            if memory.created_at:
                age_hours = (now - memory.created_at).total_seconds() / 3600
                recency = math.exp(-age_hours / decay_hours)
            else:
                recency = 0.5

            # Accumulate topic stats
            for topic in memory.topics:
                if topic not in topic_stats:
                    topic_stats[topic] = {"count": 0, "importance_sum": 0, "recency_sum": 0}
                topic_stats[topic]["count"] += 1
                topic_stats[topic]["importance_sum"] += memory.importance
                topic_stats[topic]["recency_sum"] += recency

            # Accumulate category stats
            for category in memory.categories:
                if category not in category_stats:
                    category_stats[category] = {"count": 0, "importance_sum": 0, "recency_sum": 0}
                category_stats[category]["count"] += 1
                category_stats[category]["importance_sum"] += memory.importance
                category_stats[category]["recency_sum"] += recency

            # Accumulate entity stats
            for entity in memory.entities:
                if entity not in entity_stats:
                    entity_stats[entity] = {"count": 0, "importance_sum": 0, "recency_sum": 0}
                entity_stats[entity]["count"] += 1
                entity_stats[entity]["importance_sum"] += memory.importance
                entity_stats[entity]["recency_sum"] += recency

        # Calculate final weights
        def calc_weights(stats: dict[str, dict[str, float]]) -> dict[str, float]:
            weights = {}
            for term, s in stats.items():
                frequency = s["count"] / total_memories
                avg_importance = s["importance_sum"] / s["count"]
                avg_recency = s["recency_sum"] / s["count"]

                weight = frequency * 0.4 + avg_importance * 0.4 + avg_recency * 0.2
                weights[term] = round(weight, 4)

            # Normalize to [0, 1]
            if weights:
                max_w = max(weights.values())
                if max_w > 0:
                    weights = {k: round(v / max_w, 4) for k, v in weights.items()}

            return weights

        return {
            "topic_weights": calc_weights(topic_stats),
            "category_weights": calc_weights(category_stats),
            "entity_weights": calc_weights(entity_stats),
        }

    @staticmethod
    def calculate_importance_stats(memories: list[Any]) -> dict[str, float]:
        """Calculate importance statistics from memories."""
        if not memories:
            return {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "sum": 0.0,
            }

        importances = [m.importance for m in memories]
        return {
            "min": min(importances),
            "max": max(importances),
            "avg": sum(importances) / len(importances),
            "sum": sum(importances),
        }

    @staticmethod
    def rebuild_session_aggregate(
        session_id: str,
        user_id: str,
        memories: list[Any],
        agent_id: str | None = None,
    ) -> SessionAggregate:
        """Rebuild a complete session aggregate from memories.

        Use this for initial creation or full recalculation.
        """
        aggregate = SessionAggregate(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
        )

        if not memories:
            return aggregate

        # Calculate weights
        weights = WeightCalculator.calculate_weights_from_memories(memories)
        aggregate.topic_weights = weights["topic_weights"]
        aggregate.category_weights = weights["category_weights"]
        aggregate.entity_weights = weights["entity_weights"]

        # Calculate importance stats
        stats = WeightCalculator.calculate_importance_stats(memories)
        aggregate.importance_min = stats["min"]
        aggregate.importance_max = stats["max"]
        aggregate.importance_avg = stats["avg"]
        aggregate.importance_sum = stats["sum"]

        # Set counts
        aggregate.memory_count = len(memories)
        aggregate.message_count = len(memories)

        # Set time bounds
        dates = [m.created_at for m in memories if m.created_at]
        if dates:
            aggregate.started_at = min(dates)
            aggregate.last_activity_at = max(dates)

        # Set dominant values
        if aggregate.topic_weights:
            aggregate.dominant_topic = max(aggregate.topic_weights, key=aggregate.topic_weights.get)
        if aggregate.category_weights:
            aggregate.dominant_category = max(
                aggregate.category_weights, key=aggregate.category_weights.get
            )

        # Calculate sentiment distribution
        sentiment_counts: dict[str, int] = {}
        for m in memories:
            if m.sentiment:
                sentiment_counts[m.sentiment] = sentiment_counts.get(m.sentiment, 0) + 1

        if sentiment_counts:
            total = sum(sentiment_counts.values())
            aggregate.sentiment_weights = {
                s: round(c / total, 4) for s, c in sentiment_counts.items()
            }
            aggregate.dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        # Set access level (highest in session)
        access_hierarchy = {"private": 0, "team": 1, "shared": 2, "global": 3}
        max_access = 0
        for m in memories:
            level = access_hierarchy.get(m.access_level, 0)
            if level > max_access:
                max_access = level
                aggregate.access_level = m.access_level

        return aggregate
