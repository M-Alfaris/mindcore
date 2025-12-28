"""Session Segmentation for CLST.

This module handles intelligent session management:
1. Topic shift detection - Detect when conversation topic changes significantly
2. Time-based splitting - Split sessions after inactivity gaps
3. Category transition detection - Track category changes within sessions
4. Session coherence scoring - Measure topical coherence of sessions
5. Automatic session creation - Create new sessions on shifts

Session Segmentation Rules:
- Time gap > threshold → New session
- Topic distribution shift > threshold → New session segment
- Category change (major) → New session segment
- Sessions can have multiple segments (coherent sub-conversations)

Example:
    from mindcore.clst import SessionManager

    manager = SessionManager(storage=storage)

    # Check if we should start a new session
    decision = manager.should_segment(
        current_session_id="sess_123",
        new_memory=memory,
    )

    if decision.should_segment:
        new_session_id = manager.create_segment(
            parent_session_id="sess_123",
            reason=decision.reason,
        )
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.flr import Memory
    from mindcore.storage.base import BaseStorage

logger = logging.getLogger(__name__)


# =============================================================================
# Session Segmentation Types
# =============================================================================


class SegmentReason(str, Enum):
    """Reason for session segmentation."""

    TIME_GAP = "time_gap"  # Inactivity gap exceeded
    TOPIC_SHIFT = "topic_shift"  # Major topic change
    CATEGORY_CHANGE = "category_change"  # Category transition
    USER_EXPLICIT = "user_explicit"  # User explicitly started new session
    COHERENCE_LOW = "coherence_low"  # Session coherence dropped too low
    MAX_SIZE = "max_size"  # Session hit maximum size


@dataclass
class TopicDistribution:
    """Topic weight distribution for comparison."""

    topics: dict[str, float]  # topic -> weight
    dominant_topic: str | None
    entropy: float  # Measure of topic spread
    created_at: datetime

    @classmethod
    def from_memories(cls, memories: list[Any]) -> TopicDistribution:
        """Create distribution from a list of memories."""
        topic_counts: dict[str, float] = {}
        total_importance = 0.0

        for memory in memories:
            importance = getattr(memory, "importance", 0.5)
            for topic in getattr(memory, "topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + importance
                total_importance += importance

        # Normalize to probabilities
        if total_importance > 0:
            topics = {t: c / total_importance for t, c in topic_counts.items()}
        else:
            topics = {}

        # Calculate entropy
        entropy = 0.0
        for p in topics.values():
            if p > 0:
                entropy -= p * math.log2(p)

        # Find dominant topic
        dominant = max(topics.items(), key=lambda x: x[1])[0] if topics else None

        return cls(
            topics=topics,
            dominant_topic=dominant,
            entropy=entropy,
            created_at=datetime.now(timezone.utc),
        )

    def divergence_from(self, other: TopicDistribution) -> float:
        """Calculate Jensen-Shannon divergence from another distribution.

        Returns:
            Divergence score (0 = identical, 1 = completely different)
        """
        all_topics = set(self.topics.keys()) | set(other.topics.keys())
        if not all_topics:
            return 0.0

        # Calculate JS divergence
        divergence = 0.0
        for topic in all_topics:
            p = self.topics.get(topic, 0.0001)  # Small epsilon for missing
            q = other.topics.get(topic, 0.0001)
            m = (p + q) / 2

            if p > 0 and m > 0:
                divergence += 0.5 * p * math.log2(p / m)
            if q > 0 and m > 0:
                divergence += 0.5 * q * math.log2(q / m)

        return min(1.0, divergence)


@dataclass
class SegmentDecision:
    """Decision about whether to segment a session."""

    should_segment: bool
    reason: SegmentReason | None = None
    confidence: float = 1.0

    # Details
    time_gap_minutes: float = 0.0
    topic_divergence: float = 0.0
    old_dominant_topic: str | None = None
    new_dominant_topic: str | None = None
    old_category: str | None = None
    new_category: str | None = None
    coherence_score: float = 1.0

    # Suggested new session ID
    suggested_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_segment": self.should_segment,
            "reason": self.reason.value if self.reason else None,
            "confidence": self.confidence,
            "time_gap_minutes": self.time_gap_minutes,
            "topic_divergence": self.topic_divergence,
            "old_dominant_topic": self.old_dominant_topic,
            "new_dominant_topic": self.new_dominant_topic,
            "coherence_score": self.coherence_score,
            "suggested_session_id": self.suggested_session_id,
        }


@dataclass
class SessionSegment:
    """A coherent segment within a session."""

    segment_id: str
    session_id: str  # Parent session
    user_id: str

    # Time bounds
    started_at: datetime
    ended_at: datetime | None = None

    # Topic profile
    topic_distribution: TopicDistribution | None = None
    dominant_topic: str | None = None
    dominant_category: str | None = None

    # Content
    memory_ids: list[str] = field(default_factory=list)
    memory_count: int = 0

    # Metrics
    coherence_score: float = 1.0
    avg_importance: float = 0.5

    # Lineage
    parent_segment_id: str | None = None  # Previous segment in chain
    segment_reason: SegmentReason | None = None  # Why this segment was created

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "dominant_topic": self.dominant_topic,
            "dominant_category": self.dominant_category,
            "memory_count": self.memory_count,
            "coherence_score": self.coherence_score,
            "parent_segment_id": self.parent_segment_id,
            "segment_reason": self.segment_reason.value if self.segment_reason else None,
        }


@dataclass
class SegmentationPolicy:
    """Policy for session segmentation decisions."""

    # Time-based thresholds
    inactivity_gap_minutes: float = 30.0  # Gap before new session
    max_session_duration_hours: float = 4.0  # Max session length

    # Topic shift thresholds
    topic_divergence_threshold: float = 0.5  # JS divergence threshold
    topic_shift_window_size: int = 5  # Memories to compare

    # Category change
    track_category_changes: bool = True
    major_category_change_triggers_segment: bool = True

    # Coherence
    min_coherence_score: float = 0.3  # Below this, segment
    coherence_window_size: int = 10  # Memories for coherence calc

    # Size limits
    max_session_memories: int = 500  # Max memories per session


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """Manages session segmentation and coherence.

    Handles:
    1. Detecting when to segment sessions (topic shift, time gap, etc.)
    2. Creating new session segments
    3. Calculating session coherence scores
    4. Tracking session lineage (parent-child relationships)
    """

    def __init__(
        self,
        storage: BaseStorage,
        policy: SegmentationPolicy | None = None,
    ):
        """Initialize session manager.

        Args:
            storage: Storage backend for memory access
            policy: Segmentation policy configuration
        """
        self._storage = storage
        self._policy = policy or SegmentationPolicy()

        # Cache of recent session states
        self._session_cache: dict[str, SessionState] = {}

    def should_segment(
        self,
        current_session_id: str,
        new_memory: Memory,
        user_id: str,
    ) -> SegmentDecision:
        """Determine if we should create a new session segment.

        Args:
            current_session_id: Current session ID
            new_memory: Memory about to be added
            user_id: User ID

        Returns:
            SegmentDecision with recommendation
        """
        # Get current session state
        state = self._get_session_state(current_session_id, user_id)

        # Check 1: Time gap
        time_decision = self._check_time_gap(state, new_memory)
        if time_decision.should_segment:
            return time_decision

        # Check 2: Topic shift
        topic_decision = self._check_topic_shift(state, new_memory)
        if topic_decision.should_segment:
            return topic_decision

        # Check 3: Category change
        if self._policy.track_category_changes:
            category_decision = self._check_category_change(state, new_memory)
            if category_decision.should_segment:
                return category_decision

        # Check 4: Coherence
        coherence_decision = self._check_coherence(state, new_memory)
        if coherence_decision.should_segment:
            return coherence_decision

        # Check 5: Size limit
        if state.memory_count >= self._policy.max_session_memories:
            return SegmentDecision(
                should_segment=True,
                reason=SegmentReason.MAX_SIZE,
                confidence=1.0,
                suggested_session_id=self._generate_segment_id(current_session_id),
            )

        # No segmentation needed
        return SegmentDecision(should_segment=False)

    def create_segment(
        self,
        parent_session_id: str,
        user_id: str,
        reason: SegmentReason,
        first_memory: Memory | None = None,
    ) -> SessionSegment:
        """Create a new session segment.

        Args:
            parent_session_id: Parent session ID
            user_id: User ID
            reason: Reason for segmentation
            first_memory: First memory of new segment

        Returns:
            New SessionSegment
        """
        segment_id = self._generate_segment_id(parent_session_id)

        # Get parent segment info
        parent_state = self._session_cache.get(parent_session_id)
        parent_segment_id = parent_state.current_segment_id if parent_state else None

        segment = SessionSegment(
            segment_id=segment_id,
            session_id=parent_session_id,
            user_id=user_id,
            started_at=datetime.now(timezone.utc),
            parent_segment_id=parent_segment_id,
            segment_reason=reason,
        )

        if first_memory:
            segment.dominant_topic = first_memory.topics[0] if first_memory.topics else None
            segment.dominant_category = (
                first_memory.categories[0] if first_memory.categories else None
            )
            segment.memory_ids = [first_memory.memory_id]
            segment.memory_count = 1
            segment.avg_importance = first_memory.importance

        # Close parent segment
        if parent_state and parent_state.current_segment:
            parent_state.current_segment.ended_at = datetime.now(timezone.utc)

        # Update cache
        self._session_cache[segment_id] = SessionState(
            session_id=segment_id,
            user_id=user_id,
            current_segment=segment,
            current_segment_id=segment_id,
        )

        return segment

    def calculate_coherence(
        self,
        session_id: str,
        user_id: str,
        window_size: int | None = None,
    ) -> float:
        """Calculate coherence score for a session.

        Coherence measures how topically focused a session is.
        High coherence = focused conversation on related topics
        Low coherence = scattered, unrelated topics

        Args:
            session_id: Session to analyze
            user_id: User ID
            window_size: Number of recent memories to consider

        Returns:
            Coherence score (0-1, higher is more coherent)
        """
        window_size = window_size or self._policy.coherence_window_size

        # Get recent memories
        memories = self._storage.search(
            user_id=user_id,
            limit=window_size,
        )

        # Filter to session
        session_memories = [m for m in memories if getattr(m, "session_id", None) == session_id]

        if len(session_memories) < 2:
            return 1.0  # Not enough data

        # Calculate topic overlap between consecutive memories
        overlaps = []
        for i in range(len(session_memories) - 1):
            m1_topics = set(getattr(session_memories[i], "topics", []))
            m2_topics = set(getattr(session_memories[i + 1], "topics", []))

            if m1_topics or m2_topics:
                intersection = len(m1_topics & m2_topics)
                union = len(m1_topics | m2_topics)
                overlap = intersection / union if union > 0 else 0
                overlaps.append(overlap)

        if not overlaps:
            return 1.0

        # Average overlap is coherence
        return sum(overlaps) / len(overlaps)

    def get_session_segments(
        self,
        session_id: str,
    ) -> list[SessionSegment]:
        """Get all segments for a session.

        Args:
            session_id: Session ID

        Returns:
            List of segments, ordered by start time
        """
        # This would query storage for segment data
        # For now, return from cache
        segments = []
        for sid, state in self._session_cache.items():
            if state.session_id.startswith(session_id) or session_id.startswith(state.session_id):
                if state.current_segment:
                    segments.append(state.current_segment)

        return sorted(segments, key=lambda s: s.started_at)

    def _get_session_state(
        self,
        session_id: str,
        user_id: str,
    ) -> SessionState:
        """Get or create session state."""
        if session_id in self._session_cache:
            return self._session_cache[session_id]

        # Load from storage
        memories = self._storage.search(
            user_id=user_id,
            limit=self._policy.coherence_window_size,
        )

        # Filter to session
        session_memories = [m for m in memories if getattr(m, "session_id", None) == session_id]

        # Build state
        state = SessionState(
            session_id=session_id,
            user_id=user_id,
            memory_count=len(session_memories),
        )

        if session_memories:
            state.last_activity = max(
                getattr(m, "created_at", datetime.now(timezone.utc)) for m in session_memories
            )
            state.recent_memories = session_memories[: self._policy.topic_shift_window_size]
            state.topic_distribution = TopicDistribution.from_memories(session_memories)
            state.dominant_topic = state.topic_distribution.dominant_topic

            # Get dominant category
            category_counts: dict[str, int] = {}
            for m in session_memories:
                for cat in getattr(m, "categories", []):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            if category_counts:
                state.dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]

        self._session_cache[session_id] = state
        return state

    def _check_time_gap(
        self,
        state: SessionState,
        new_memory: Memory,
    ) -> SegmentDecision:
        """Check if time gap warrants new session."""
        if not state.last_activity:
            return SegmentDecision(should_segment=False)

        new_time = getattr(new_memory, "created_at", datetime.now(timezone.utc))
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=timezone.utc)

        last_time = state.last_activity
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)

        gap_minutes = (new_time - last_time).total_seconds() / 60

        if gap_minutes > self._policy.inactivity_gap_minutes:
            return SegmentDecision(
                should_segment=True,
                reason=SegmentReason.TIME_GAP,
                confidence=min(1.0, gap_minutes / (self._policy.inactivity_gap_minutes * 2)),
                time_gap_minutes=gap_minutes,
                suggested_session_id=self._generate_segment_id(state.session_id),
            )

        return SegmentDecision(
            should_segment=False,
            time_gap_minutes=gap_minutes,
        )

    def _check_topic_shift(
        self,
        state: SessionState,
        new_memory: Memory,
    ) -> SegmentDecision:
        """Check if topic shift warrants new session."""
        if not state.topic_distribution:
            return SegmentDecision(should_segment=False)

        # Create distribution from new memory
        new_topics = getattr(new_memory, "topics", [])
        if not new_topics:
            return SegmentDecision(should_segment=False)

        # Simple single-memory distribution
        new_dist = TopicDistribution(
            topics={t: 1.0 / len(new_topics) for t in new_topics},
            dominant_topic=new_topics[0] if new_topics else None,
            entropy=0.0,
            created_at=datetime.now(timezone.utc),
        )

        divergence = state.topic_distribution.divergence_from(new_dist)

        if divergence > self._policy.topic_divergence_threshold:
            return SegmentDecision(
                should_segment=True,
                reason=SegmentReason.TOPIC_SHIFT,
                confidence=divergence,
                topic_divergence=divergence,
                old_dominant_topic=state.dominant_topic,
                new_dominant_topic=new_dist.dominant_topic,
                suggested_session_id=self._generate_segment_id(state.session_id),
            )

        return SegmentDecision(
            should_segment=False,
            topic_divergence=divergence,
        )

    def _check_category_change(
        self,
        state: SessionState,
        new_memory: Memory,
    ) -> SegmentDecision:
        """Check if category change warrants new session."""
        if not self._policy.major_category_change_triggers_segment:
            return SegmentDecision(should_segment=False)

        old_category = state.dominant_category
        new_categories = getattr(new_memory, "categories", [])
        new_category = new_categories[0] if new_categories else None

        if old_category and new_category and old_category != new_category:
            # Check if it's a major change (completely different category)
            # Could be enhanced with category taxonomy
            return SegmentDecision(
                should_segment=True,
                reason=SegmentReason.CATEGORY_CHANGE,
                confidence=0.8,
                old_category=old_category,
                new_category=new_category,
                suggested_session_id=self._generate_segment_id(state.session_id),
            )

        return SegmentDecision(should_segment=False)

    def _check_coherence(
        self,
        state: SessionState,
        new_memory: Memory,
    ) -> SegmentDecision:
        """Check if coherence is too low."""
        # Calculate current coherence
        coherence = self.calculate_coherence(
            session_id=state.session_id,
            user_id=state.user_id,
        )

        if coherence < self._policy.min_coherence_score:
            return SegmentDecision(
                should_segment=True,
                reason=SegmentReason.COHERENCE_LOW,
                confidence=1.0 - coherence,
                coherence_score=coherence,
                suggested_session_id=self._generate_segment_id(state.session_id),
            )

        return SegmentDecision(
            should_segment=False,
            coherence_score=coherence,
        )

    def _generate_segment_id(self, parent_session_id: str) -> str:
        """Generate a new segment ID."""
        return f"{parent_session_id}_seg_{uuid.uuid4().hex[:8]}"

    def update_session_state(
        self,
        session_id: str,
        memory: Memory,
    ) -> None:
        """Update session state after adding a memory.

        Args:
            session_id: Session ID
            memory: Memory that was added
        """
        if session_id not in self._session_cache:
            return

        state = self._session_cache[session_id]
        state.memory_count += 1
        state.last_activity = getattr(memory, "created_at", datetime.now(timezone.utc))

        # Update recent memories
        if state.recent_memories is None:
            state.recent_memories = []
        state.recent_memories.insert(0, memory)
        state.recent_memories = state.recent_memories[: self._policy.topic_shift_window_size]

        # Recalculate topic distribution
        state.topic_distribution = TopicDistribution.from_memories(state.recent_memories)
        state.dominant_topic = state.topic_distribution.dominant_topic

    def clear_cache(self) -> None:
        """Clear session cache."""
        self._session_cache.clear()


@dataclass
class SessionState:
    """Cached state for a session."""

    session_id: str
    user_id: str
    memory_count: int = 0
    last_activity: datetime | None = None
    dominant_topic: str | None = None
    dominant_category: str | None = None
    topic_distribution: TopicDistribution | None = None
    recent_memories: list[Any] | None = None
    current_segment: SessionSegment | None = None
    current_segment_id: str | None = None
