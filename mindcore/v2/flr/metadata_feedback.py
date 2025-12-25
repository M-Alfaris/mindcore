"""Metadata Effectiveness Feedback for FLR.

Tracks whether LLM-assigned metadata (topics, categories, intent, etc.)
leads to successful retrievals. This feedback can improve future
metadata assignments by the MetadataExtractor.

The core idea:
1. LLM assigns metadata from SVL vocabulary
2. Memory is stored with that metadata
3. Memory is retrieved based on metadata matching
4. User/LLM feedback indicates if retrieval was helpful
5. We correlate: which metadata assignments → successful retrievals

Example:
    tracker = MetadataFeedbackTracker()

    # After successful retrieval
    tracker.record_retrieval_feedback(
        memory=memory,
        query_topics=["refund", "billing"],
        signal=+0.8,  # Positive = memory was useful
    )

    # Get effectiveness report for MetadataExtractor tuning
    report = tracker.get_effectiveness_report()
    # {"topics": {"refund": 0.85, "billing": 0.72}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MetadataSignal:
    """A feedback signal for metadata effectiveness."""

    memory_id: str
    signal_value: float  # -1 to +1

    # What metadata was on the memory
    assigned_topics: list[str]
    assigned_categories: list[str]
    assigned_intent: str | None = None
    assigned_type: str | None = None

    # What the query was looking for
    query_topics: list[str] = field(default_factory=list)
    query_categories: list[str] = field(default_factory=list)

    # Which assignments matched the query
    matched_topics: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None

    def __post_init__(self):
        """Calculate which assignments matched."""
        if not self.matched_topics:
            self.matched_topics = list(
                set(self.assigned_topics) & set(self.query_topics)
            )
        if not self.matched_categories:
            self.matched_categories = list(
                set(self.assigned_categories) & set(self.query_categories)
            )


@dataclass
class MetadataEffectiveness:
    """Tracks effectiveness of a single metadata value."""

    value: str  # e.g., "refund" or "billing"
    metadata_type: str  # "topic", "category", "intent", "type"

    # Counters
    times_assigned: int = 0  # How many times LLM assigned this
    times_matched: int = 0  # How many times it led to retrieval
    positive_signals: int = 0  # Retrievals that were useful
    negative_signals: int = 0  # Retrievals that weren't useful

    # Accumulated scores
    total_signal: float = 0.0

    def record_assignment(self) -> None:
        """Record that this value was assigned by LLM."""
        self.times_assigned += 1

    def record_match(self, signal: float) -> None:
        """Record that this value matched a query with given feedback."""
        self.times_matched += 1
        self.total_signal += signal

        if signal > 0:
            self.positive_signals += 1
        elif signal < 0:
            self.negative_signals += 1

    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness score (0 to 1).

        Higher = this metadata value leads to useful retrievals.
        """
        if self.times_matched == 0:
            return 0.5  # Neutral if never matched

        # Ratio of positive to total matches
        total_feedback = self.positive_signals + self.negative_signals
        if total_feedback == 0:
            return 0.5

        return self.positive_signals / total_feedback

    @property
    def match_rate(self) -> float:
        """How often this assignment leads to retrieval."""
        if self.times_assigned == 0:
            return 0.0
        return self.times_matched / self.times_assigned

    @property
    def average_signal(self) -> float:
        """Average feedback signal when matched."""
        if self.times_matched == 0:
            return 0.0
        return self.total_signal / self.times_matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "metadata_type": self.metadata_type,
            "times_assigned": self.times_assigned,
            "times_matched": self.times_matched,
            "positive_signals": self.positive_signals,
            "negative_signals": self.negative_signals,
            "total_signal": self.total_signal,
            "effectiveness_score": self.effectiveness_score,
            "match_rate": self.match_rate,
            "average_signal": self.average_signal,
        }


class MetadataFeedbackTracker:
    """Tracks metadata effectiveness for improving LLM assignments.

    This class answers: "Which metadata assignments by the LLM lead to
    successful retrievals?"

    Usage:
        tracker = MetadataFeedbackTracker()

        # When memory is stored (optional, for tracking assignment frequency)
        tracker.record_assignment(memory)

        # When memory is retrieved and feedback is given
        tracker.record_retrieval_feedback(
            memory=memory,
            query_topics=["billing"],
            signal=+0.8,
        )

        # Get report for tuning MetadataExtractor
        report = tracker.get_effectiveness_report()
    """

    def __init__(self, max_history: int = 10000):
        """Initialize tracker.

        Args:
            max_history: Maximum signals to keep in history
        """
        self.max_history = max_history

        # Effectiveness tracking by metadata type and value
        self._topic_effectiveness: dict[str, MetadataEffectiveness] = {}
        self._category_effectiveness: dict[str, MetadataEffectiveness] = {}
        self._intent_effectiveness: dict[str, MetadataEffectiveness] = {}
        self._type_effectiveness: dict[str, MetadataEffectiveness] = {}

        # Recent signals for analysis
        self._signal_history: list[MetadataSignal] = []

    def record_assignment(
        self,
        topics: list[str],
        categories: list[str],
        intent: str | None = None,
        memory_type: str | None = None,
    ) -> None:
        """Record that LLM assigned these metadata values.

        Call this when storing a new memory to track assignment frequency.

        Args:
            topics: Assigned topics
            categories: Assigned categories
            intent: Assigned intent
            memory_type: Assigned memory type
        """
        for topic in topics:
            if topic not in self._topic_effectiveness:
                self._topic_effectiveness[topic] = MetadataEffectiveness(
                    value=topic, metadata_type="topic"
                )
            self._topic_effectiveness[topic].record_assignment()

        for category in categories:
            if category not in self._category_effectiveness:
                self._category_effectiveness[category] = MetadataEffectiveness(
                    value=category, metadata_type="category"
                )
            self._category_effectiveness[category].record_assignment()

        if intent:
            if intent not in self._intent_effectiveness:
                self._intent_effectiveness[intent] = MetadataEffectiveness(
                    value=intent, metadata_type="intent"
                )
            self._intent_effectiveness[intent].record_assignment()

        if memory_type:
            if memory_type not in self._type_effectiveness:
                self._type_effectiveness[memory_type] = MetadataEffectiveness(
                    value=memory_type, metadata_type="type"
                )
            self._type_effectiveness[memory_type].record_assignment()

    def record_retrieval_feedback(
        self,
        memory_id: str,
        assigned_topics: list[str],
        assigned_categories: list[str],
        query_topics: list[str],
        query_categories: list[str],
        signal: float,
        assigned_intent: str | None = None,
        assigned_type: str | None = None,
        session_id: str | None = None,
    ) -> MetadataSignal:
        """Record feedback for a retrieved memory.

        This is the key method - it tells us if the LLM's metadata
        assignment led to a useful retrieval.

        Args:
            memory_id: ID of the retrieved memory
            assigned_topics: Topics the LLM assigned to this memory
            assigned_categories: Categories the LLM assigned
            query_topics: Topics the query was looking for
            query_categories: Categories the query was looking for
            signal: Feedback signal (-1 to +1, positive = useful)
            assigned_intent: Intent the LLM assigned
            assigned_type: Memory type the LLM assigned
            session_id: Current session ID

        Returns:
            The recorded MetadataSignal
        """
        signal = max(-1.0, min(1.0, signal))

        metadata_signal = MetadataSignal(
            memory_id=memory_id,
            signal_value=signal,
            assigned_topics=assigned_topics,
            assigned_categories=assigned_categories,
            assigned_intent=assigned_intent,
            assigned_type=assigned_type,
            query_topics=query_topics,
            query_categories=query_categories,
            session_id=session_id,
        )

        # Update effectiveness for matched topics
        for topic in metadata_signal.matched_topics:
            if topic not in self._topic_effectiveness:
                self._topic_effectiveness[topic] = MetadataEffectiveness(
                    value=topic, metadata_type="topic"
                )
            self._topic_effectiveness[topic].record_match(signal)

        # Update effectiveness for matched categories
        for category in metadata_signal.matched_categories:
            if category not in self._category_effectiveness:
                self._category_effectiveness[category] = MetadataEffectiveness(
                    value=category, metadata_type="category"
                )
            self._category_effectiveness[category].record_match(signal)

        # Track intent effectiveness if it was part of query context
        if assigned_intent:
            if assigned_intent not in self._intent_effectiveness:
                self._intent_effectiveness[assigned_intent] = MetadataEffectiveness(
                    value=assigned_intent, metadata_type="intent"
                )
            self._intent_effectiveness[assigned_intent].record_match(signal)

        # Track type effectiveness
        if assigned_type:
            if assigned_type not in self._type_effectiveness:
                self._type_effectiveness[assigned_type] = MetadataEffectiveness(
                    value=assigned_type, metadata_type="type"
                )
            self._type_effectiveness[assigned_type].record_match(signal)

        # Store in history
        self._signal_history.append(metadata_signal)

        # Trim history if needed
        if len(self._signal_history) > self.max_history:
            self._signal_history = self._signal_history[-self.max_history :]

        return metadata_signal

    def get_effectiveness_report(self) -> dict[str, Any]:
        """Get effectiveness report for all metadata.

        Use this to tune MetadataExtractor - boost confidence for
        high-effectiveness values, reduce for low-effectiveness.

        Returns:
            Report with effectiveness scores by metadata type
        """
        return {
            "topics": {
                topic: eff.to_dict()
                for topic, eff in sorted(
                    self._topic_effectiveness.items(),
                    key=lambda x: x[1].effectiveness_score,
                    reverse=True,
                )
            },
            "categories": {
                cat: eff.to_dict()
                for cat, eff in sorted(
                    self._category_effectiveness.items(),
                    key=lambda x: x[1].effectiveness_score,
                    reverse=True,
                )
            },
            "intents": {
                intent: eff.to_dict()
                for intent, eff in sorted(
                    self._intent_effectiveness.items(),
                    key=lambda x: x[1].effectiveness_score,
                    reverse=True,
                )
            },
            "types": {
                t: eff.to_dict()
                for t, eff in sorted(
                    self._type_effectiveness.items(),
                    key=lambda x: x[1].effectiveness_score,
                    reverse=True,
                )
            },
            "summary": {
                "total_signals": len(self._signal_history),
                "unique_topics_tracked": len(self._topic_effectiveness),
                "unique_categories_tracked": len(self._category_effectiveness),
            },
        }

    def get_top_effective_values(
        self,
        metadata_type: str = "topic",
        limit: int = 10,
        min_matches: int = 5,
    ) -> list[tuple[str, float]]:
        """Get most effective metadata values.

        Args:
            metadata_type: "topic", "category", "intent", or "type"
            limit: Max values to return
            min_matches: Minimum matches required for inclusion

        Returns:
            List of (value, effectiveness_score) tuples
        """
        effectiveness_map = {
            "topic": self._topic_effectiveness,
            "category": self._category_effectiveness,
            "intent": self._intent_effectiveness,
            "type": self._type_effectiveness,
        }

        data = effectiveness_map.get(metadata_type, {})

        # Filter by min matches and sort by effectiveness
        filtered = [
            (value, eff.effectiveness_score)
            for value, eff in data.items()
            if eff.times_matched >= min_matches
        ]

        return sorted(filtered, key=lambda x: x[1], reverse=True)[:limit]

    def get_low_effective_values(
        self,
        metadata_type: str = "topic",
        limit: int = 10,
        min_matches: int = 5,
    ) -> list[tuple[str, float]]:
        """Get least effective metadata values (candidates for review).

        These are values that the LLM assigns but lead to poor retrievals.
        Consider removing them from SVL or redefining their usage.

        Args:
            metadata_type: "topic", "category", "intent", or "type"
            limit: Max values to return
            min_matches: Minimum matches required for inclusion

        Returns:
            List of (value, effectiveness_score) tuples
        """
        effectiveness_map = {
            "topic": self._topic_effectiveness,
            "category": self._category_effectiveness,
            "intent": self._intent_effectiveness,
            "type": self._type_effectiveness,
        }

        data = effectiveness_map.get(metadata_type, {})

        # Filter by min matches and sort by effectiveness (ascending)
        filtered = [
            (value, eff.effectiveness_score)
            for value, eff in data.items()
            if eff.times_matched >= min_matches
        ]

        return sorted(filtered, key=lambda x: x[1])[:limit]

    def get_feedback_for_extractor(self) -> dict[str, Any]:
        """Get structured feedback for MetadataExtractor tuning.

        This returns data that can be injected into the LLM prompt
        to improve future metadata assignments.

        Returns:
            Feedback structure for prompt injection
        """
        return {
            "high_quality_topics": self.get_top_effective_values("topic", 20, 3),
            "low_quality_topics": self.get_low_effective_values("topic", 10, 3),
            "high_quality_categories": self.get_top_effective_values("category", 20, 3),
            "low_quality_categories": self.get_low_effective_values("category", 10, 3),
            "guidance": self._generate_guidance(),
        }

    def _generate_guidance(self) -> str:
        """Generate natural language guidance for MetadataExtractor."""
        lines = []

        # Top effective topics
        top_topics = self.get_top_effective_values("topic", 5, 5)
        if top_topics:
            topics_str = ", ".join([f"'{t[0]}'" for t in top_topics])
            lines.append(f"High-quality topics (use confidently): {topics_str}")

        # Low effective topics
        low_topics = self.get_low_effective_values("topic", 5, 5)
        if low_topics:
            topics_str = ", ".join([f"'{t[0]}'" for t in low_topics])
            lines.append(f"Low-quality topics (use sparingly): {topics_str}")

        # Top effective categories
        top_cats = self.get_top_effective_values("category", 5, 5)
        if top_cats:
            cats_str = ", ".join([f"'{c[0]}'" for c in top_cats])
            lines.append(f"High-quality categories: {cats_str}")

        return "\n".join(lines) if lines else "No feedback available yet."

    def clear(self) -> None:
        """Clear all tracking data."""
        self._topic_effectiveness.clear()
        self._category_effectiveness.clear()
        self._intent_effectiveness.clear()
        self._type_effectiveness.clear()
        self._signal_history.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "topic_effectiveness": {
                k: v.to_dict() for k, v in self._topic_effectiveness.items()
            },
            "category_effectiveness": {
                k: v.to_dict() for k, v in self._category_effectiveness.items()
            },
            "intent_effectiveness": {
                k: v.to_dict() for k, v in self._intent_effectiveness.items()
            },
            "type_effectiveness": {
                k: v.to_dict() for k, v in self._type_effectiveness.items()
            },
            "signal_count": len(self._signal_history),
        }
