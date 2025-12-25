"""Dynamic Query Optimizer for FLR.

Uses feedback from usage detection and metadata effectiveness to
dynamically optimize query parameters for better retrieval.

Optimizations:
1. Topic boosting: Boost topics that lead to used memories
2. Topic filtering: Filter out topics with low effectiveness
3. Limit adjustment: Adjust retrieval limit based on usage rate
4. Attention hint optimization: Reorder hints by effectiveness

Example:
    optimizer = QueryOptimizer()

    # Feed usage data
    optimizer.record_usage(usage_result)

    # Get optimized query params
    optimized = optimizer.optimize_query(
        original_topics=["billing", "refund", "general"],
        original_limit=10,
    )
    # Returns: {"topics": ["refund", "billing"], "limit": 7, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .usage_detector import UsageDetectionResult


@dataclass
class TopicStats:
    """Statistics for a single topic."""

    topic: str
    times_retrieved: int = 0
    times_used: int = 0
    total_signal: float = 0.0
    last_used: datetime | None = None

    @property
    def usage_rate(self) -> float:
        if self.times_retrieved == 0:
            return 0.0
        return self.times_used / self.times_retrieved

    @property
    def avg_signal(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.total_signal / self.times_used

    @property
    def effectiveness_score(self) -> float:
        """Combined effectiveness score (0-1)."""
        if self.times_retrieved < 3:
            return 0.5  # Neutral for insufficient data

        # Combine usage rate and average signal
        usage_component = self.usage_rate * 0.6
        signal_component = (self.avg_signal + 1) / 2 * 0.4  # Normalize -1,1 to 0,1

        return usage_component + signal_component


@dataclass
class QueryOptimization:
    """Result of query optimization."""

    original_topics: list[str]
    optimized_topics: list[str]
    removed_topics: list[str]
    boosted_topics: list[str]

    original_limit: int
    optimized_limit: int

    original_categories: list[str] = field(default_factory=list)
    optimized_categories: list[str] = field(default_factory=list)

    confidence: float = 0.5  # How confident we are in optimization
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_topics": self.original_topics,
            "optimized_topics": self.optimized_topics,
            "removed_topics": self.removed_topics,
            "boosted_topics": self.boosted_topics,
            "original_limit": self.original_limit,
            "optimized_limit": self.optimized_limit,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class QueryOptimizer:
    """Dynamically optimizes FLR queries based on usage feedback.

    Learns from:
    - Which topics lead to used memories
    - Overall usage rates
    - Signal feedback from reinforcement

    Example:
        optimizer = QueryOptimizer()

        # After each query/response cycle
        optimizer.record_usage(usage_result)

        # Before next query
        optimization = optimizer.optimize_query(
            original_topics=["billing", "support"],
            original_limit=10,
        )

        result = flr.query(
            query="...",
            attention_hints=optimization.optimized_topics,
            limit=optimization.optimized_limit,
        )
    """

    def __init__(
        self,
        min_samples_for_optimization: int = 5,
        topic_removal_threshold: float = 0.2,
        topic_boost_threshold: float = 0.6,
        max_history_age_hours: float = 168.0,  # 1 week
        enable_limit_adjustment: bool = True,
    ):
        """Initialize optimizer.

        Args:
            min_samples_for_optimization: Min data points before optimizing
            topic_removal_threshold: Remove topics below this usage rate
            topic_boost_threshold: Boost topics above this usage rate
            max_history_age_hours: Ignore data older than this
            enable_limit_adjustment: Adjust retrieval limit dynamically
        """
        self.min_samples = min_samples_for_optimization
        self.removal_threshold = topic_removal_threshold
        self.boost_threshold = topic_boost_threshold
        self.max_history_age = timedelta(hours=max_history_age_hours)
        self.enable_limit_adjustment = enable_limit_adjustment

        # Topic statistics
        self._topic_stats: dict[str, TopicStats] = {}

        # Overall statistics
        self._total_retrieved: int = 0
        self._total_used: int = 0
        self._usage_history: list[tuple[datetime, float]] = []  # (time, usage_rate)

    def record_usage(self, usage_result: UsageDetectionResult) -> None:
        """Record usage data from a query/response cycle.

        Args:
            usage_result: Result from UsageDetector.detect_usage()
        """
        now = datetime.now(timezone.utc)

        # Update topic stats for used memories
        for usage in usage_result.used_memories:
            for topic in usage.memory_topics:
                self._ensure_topic(topic)
                self._topic_stats[topic].times_retrieved += 1
                self._topic_stats[topic].times_used += 1
                self._topic_stats[topic].total_signal += usage.suggested_signal
                self._topic_stats[topic].last_used = now

        # Update topic stats for unused memories
        for usage in usage_result.unused_memories:
            for topic in usage.memory_topics:
                self._ensure_topic(topic)
                self._topic_stats[topic].times_retrieved += 1
                # Don't increment times_used

        # Update overall stats
        self._total_retrieved += usage_result.total_memories
        self._total_used += len(usage_result.used_memories)
        self._usage_history.append((now, usage_result.usage_rate))

        # Prune old history
        self._prune_old_data()

    def _ensure_topic(self, topic: str) -> None:
        """Ensure topic exists in stats."""
        if topic not in self._topic_stats:
            self._topic_stats[topic] = TopicStats(topic=topic)

    def _prune_old_data(self) -> None:
        """Remove data older than max_history_age."""
        if not self._usage_history:
            return

        cutoff = datetime.now(timezone.utc) - self.max_history_age
        self._usage_history = [
            (t, r) for t, r in self._usage_history if t > cutoff
        ]

    def optimize_query(
        self,
        original_topics: list[str],
        original_limit: int = 10,
        original_categories: list[str] | None = None,
        min_topics: int = 1,
    ) -> QueryOptimization:
        """Optimize query parameters based on learned effectiveness.

        Args:
            original_topics: Original attention hint topics
            original_limit: Original retrieval limit
            original_categories: Original category filters
            min_topics: Minimum topics to keep

        Returns:
            QueryOptimization with adjusted parameters
        """
        original_categories = original_categories or []
        removed = []
        boosted = []
        reasoning_parts = []

        # Check if we have enough data
        total_samples = sum(s.times_retrieved for s in self._topic_stats.values())
        if total_samples < self.min_samples:
            return QueryOptimization(
                original_topics=original_topics,
                optimized_topics=original_topics,
                removed_topics=[],
                boosted_topics=[],
                original_limit=original_limit,
                optimized_limit=original_limit,
                original_categories=original_categories,
                optimized_categories=original_categories,
                confidence=0.1,
                reasoning="Insufficient data for optimization",
            )

        # Score and sort topics
        topic_scores = []
        for topic in original_topics:
            if topic in self._topic_stats:
                stats = self._topic_stats[topic]
                if stats.times_retrieved >= 3:
                    topic_scores.append((topic, stats.effectiveness_score, stats.usage_rate))
                else:
                    topic_scores.append((topic, 0.5, 0.5))  # Neutral
            else:
                topic_scores.append((topic, 0.5, 0.5))  # Unknown

        # Sort by effectiveness
        topic_scores.sort(key=lambda x: x[1], reverse=True)

        # Build optimized topic list
        optimized_topics = []
        for topic, score, usage_rate in topic_scores:
            if usage_rate < self.removal_threshold and len(optimized_topics) >= min_topics:
                removed.append(topic)
                reasoning_parts.append(f"Removed '{topic}' (usage rate: {usage_rate:.0%})")
            else:
                optimized_topics.append(topic)
                if usage_rate >= self.boost_threshold:
                    boosted.append(topic)
                    reasoning_parts.append(f"Boosted '{topic}' (usage rate: {usage_rate:.0%})")

        # Ensure minimum topics
        if len(optimized_topics) < min_topics:
            optimized_topics = original_topics[:min_topics]

        # Optimize limit based on usage rate
        optimized_limit = original_limit
        if self.enable_limit_adjustment and self._usage_history:
            avg_usage_rate = sum(r for _, r in self._usage_history[-20:]) / min(20, len(self._usage_history))

            if avg_usage_rate < 0.3:
                # Low usage - reduce limit to get fewer, more relevant results
                optimized_limit = max(3, int(original_limit * 0.7))
                reasoning_parts.append(f"Reduced limit to {optimized_limit} (low usage rate: {avg_usage_rate:.0%})")
            elif avg_usage_rate > 0.8:
                # High usage - can increase limit
                optimized_limit = min(20, int(original_limit * 1.2))
                reasoning_parts.append(f"Increased limit to {optimized_limit} (high usage rate: {avg_usage_rate:.0%})")

        # Calculate confidence
        confidence = min(1.0, total_samples / 50)  # More samples = more confidence

        return QueryOptimization(
            original_topics=original_topics,
            optimized_topics=optimized_topics,
            removed_topics=removed,
            boosted_topics=boosted,
            original_limit=original_limit,
            optimized_limit=optimized_limit,
            original_categories=original_categories,
            optimized_categories=original_categories,  # TODO: optimize categories too
            confidence=confidence,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else "No significant optimizations",
        )

    def get_topic_rankings(self, min_samples: int = 3) -> list[tuple[str, float]]:
        """Get topics ranked by effectiveness.

        Args:
            min_samples: Minimum retrievals to include

        Returns:
            List of (topic, effectiveness_score) tuples, sorted descending
        """
        rankings = [
            (topic, stats.effectiveness_score)
            for topic, stats in self._topic_stats.items()
            if stats.times_retrieved >= min_samples
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_recommendations(self) -> dict[str, Any]:
        """Get query optimization recommendations.

        Returns:
            Structured recommendations for improving queries
        """
        rankings = self.get_topic_rankings()

        if not rankings:
            return {
                "status": "insufficient_data",
                "message": "Need more query data for recommendations",
            }

        # Top and bottom performers
        top_topics = rankings[:5]
        bottom_topics = rankings[-5:] if len(rankings) > 5 else []

        # Overall health
        avg_usage = self._total_used / self._total_retrieved if self._total_retrieved > 0 else 0

        return {
            "status": "ready",
            "overall_usage_rate": avg_usage,
            "top_performing_topics": [
                {"topic": t, "score": s, "usage_rate": self._topic_stats[t].usage_rate}
                for t, s in top_topics
            ],
            "underperforming_topics": [
                {"topic": t, "score": s, "usage_rate": self._topic_stats[t].usage_rate}
                for t, s in bottom_topics
                if s < 0.4
            ],
            "recommendations": self._generate_recommendations(avg_usage, top_topics, bottom_topics),
        }

    def _generate_recommendations(
        self,
        avg_usage: float,
        top_topics: list[tuple[str, float]],
        bottom_topics: list[tuple[str, float]],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        if avg_usage < 0.3:
            recs.append("Consider reducing retrieval limit - many memories go unused")
            recs.append("Review query formulation - attention hints may be too broad")

        if bottom_topics:
            poor = [t for t, s in bottom_topics if s < 0.3]
            if poor:
                recs.append(f"Consider removing these topics from SVL or refining their definitions: {', '.join(poor)}")

        if top_topics:
            recs.append(f"Best performing topics to prioritize: {', '.join([t for t, _ in top_topics[:3]])}")

        return recs

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "topics_tracked": len(self._topic_stats),
            "total_retrieved": self._total_retrieved,
            "total_used": self._total_used,
            "overall_usage_rate": self._total_used / self._total_retrieved if self._total_retrieved > 0 else 0,
            "history_size": len(self._usage_history),
            "topic_stats": {
                t: {
                    "retrieved": s.times_retrieved,
                    "used": s.times_used,
                    "usage_rate": s.usage_rate,
                    "effectiveness": s.effectiveness_score,
                }
                for t, s in self._topic_stats.items()
                if s.times_retrieved >= 3
            },
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._topic_stats.clear()
        self._total_retrieved = 0
        self._total_used = 0
        self._usage_history.clear()
