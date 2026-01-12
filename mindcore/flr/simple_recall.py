"""DeterministicRecall - Fast deterministic cache layer for FLR.

A deterministic, fast cache layer that:
1. Performs O(1) LRU cache lookups
2. Applies deterministic filtering (topics, session, recency)
3. Determines if CLST is needed based on metadata hints
4. Passes reinforcement signals to CLST (doesn't process them)

This provides the hot path for fast memory access.

Example:
    from mindcore.flr import DeterministicRecall

    flr = DeterministicRecall(storage=storage)

    # Fast cache query
    result = flr.query(
        user_id="user123",
        topics=["orders", "shipping"],
        limit=10,
    )

    # Check if CLST is needed
    if result.clst_decision.needs_clst:
        # Do full CLST query
        ...
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.storage.base import BaseStorage

logger = logging.getLogger(__name__)


# =============================================================================
# CLST Decision Logic
# =============================================================================


class CLSTNeedLevel(str, Enum):
    """Level of CLST need based on metadata hints."""

    NONE = "none"  # Cache sufficient, skip CLST
    LOW = "low"  # Maybe useful, but not required
    MEDIUM = "medium"  # Recommended for better results
    HIGH = "high"  # Required for accurate response
    REQUIRED = "required"  # Must use CLST, no fallback


@dataclass
class CLSTDecision:
    """Decision about whether to query CLST based on metadata.

    This is the key insight: FLR determines IF CLST is needed based on
    metadata hints, not complex scoring. The actual scoring happens in CLST.

    Metadata hints considered:
    - is_clst_needed: Explicit hint from LLM
    - confidence: How confident is the cache result (0-1)
    - priority: Query priority (affects urgency threshold)
    - cache_hit_count: How many memories found in cache
    - topic_coverage: Are all requested topics covered by cache
    - temporal_relevance: Is recent context sufficient
    """

    needs_clst: bool
    level: CLSTNeedLevel = CLSTNeedLevel.NONE
    confidence: float = 1.0  # Confidence in cache-only result
    reason: str = ""

    # Metadata hints that influenced decision
    hints_used: list[str] = field(default_factory=list)

    # Cache statistics that influenced decision
    cache_hit_count: int = 0
    topic_coverage: float = 1.0
    age_hours: float = 0.0

    # Pending signals to pass to CLST
    pending_signals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_clst": self.needs_clst,
            "level": self.level.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "hints_used": self.hints_used,
            "cache_hit_count": self.cache_hit_count,
            "topic_coverage": self.topic_coverage,
            "age_hours": self.age_hours,
        }


@dataclass
class CLSTDecisionPolicy:
    """Policy for CLST decision making.

    Configures thresholds for when to escalate to CLST.
    """

    # Minimum cache hits before considering cache-only
    min_cache_hits: int = 1

    # Confidence threshold below which CLST is needed
    confidence_threshold: float = 0.7

    # Maximum age (hours) before considering stale
    max_cache_age_hours: float = 24.0

    # Topic coverage threshold
    min_topic_coverage: float = 0.5

    # Priority thresholds
    high_priority_always_clst: bool = True

    # Force CLST for certain query patterns
    force_clst_patterns: list[str] = field(default_factory=list)


def make_clst_decision(
    cache_hits: int,
    requested_topics: list[str],
    matched_topics: list[str],
    oldest_memory_age_hours: float,
    metadata_hints: dict[str, Any] | None = None,
    policy: CLSTDecisionPolicy | None = None,
) -> CLSTDecision:
    """Deterministic CLST decision based on metadata hints.

    This is a pure function - given the same inputs, always returns
    the same output. No probabilistic scoring.

    Args:
        cache_hits: Number of memories found in cache
        requested_topics: Topics requested in query
        matched_topics: Topics found in cached memories
        oldest_memory_age_hours: Age of oldest cached memory in hours
        metadata_hints: Optional hints from LLM or context
        policy: Decision policy configuration

    Returns:
        CLSTDecision with recommendation
    """
    policy = policy or CLSTDecisionPolicy()
    hints = metadata_hints or {}
    hints_used = []

    # Start with default: no CLST needed
    needs_clst = False
    level = CLSTNeedLevel.NONE
    confidence = 1.0
    reasons = []

    # Check 1: Explicit hint from LLM
    if hints.get("is_clst_needed") is True:
        needs_clst = True
        level = CLSTNeedLevel.REQUIRED
        hints_used.append("is_clst_needed=True")
        reasons.append("LLM explicitly requested CLST")

    elif hints.get("is_clst_needed") is False:
        # LLM says cache is sufficient
        needs_clst = False
        hints_used.append("is_clst_needed=False")

    # Check 2: Confidence scoring hint
    hint_confidence = hints.get("confidence", 1.0)
    if hint_confidence < policy.confidence_threshold:
        needs_clst = True
        confidence = hint_confidence
        hints_used.append(f"confidence={hint_confidence}")
        reasons.append(f"Low confidence ({hint_confidence:.2f})")
        if level == CLSTNeedLevel.NONE:
            level = CLSTNeedLevel.MEDIUM if hint_confidence < 0.5 else CLSTNeedLevel.LOW

    # Check 3: Priority hint
    priority = hints.get("priority", "normal")
    if priority in ("high", "urgent", "critical"):
        if policy.high_priority_always_clst:
            needs_clst = True
            hints_used.append(f"priority={priority}")
            reasons.append("High priority query")
            if level.value < CLSTNeedLevel.HIGH.value:
                level = CLSTNeedLevel.HIGH

    # Check 4: Cache hit count
    if cache_hits < policy.min_cache_hits:
        needs_clst = True
        confidence = min(confidence, cache_hits / max(1, policy.min_cache_hits))
        reasons.append(f"Insufficient cache hits ({cache_hits})")
        if level == CLSTNeedLevel.NONE:
            level = CLSTNeedLevel.MEDIUM

    # Check 5: Topic coverage
    if requested_topics:
        covered = len(set(matched_topics) & set(requested_topics))
        topic_coverage = covered / len(requested_topics) if requested_topics else 1.0
    else:
        topic_coverage = 1.0

    if topic_coverage < policy.min_topic_coverage:
        needs_clst = True
        confidence = min(confidence, topic_coverage)
        reasons.append(f"Low topic coverage ({topic_coverage:.2%})")
        if level == CLSTNeedLevel.NONE:
            level = CLSTNeedLevel.MEDIUM

    # Check 6: Cache staleness
    if oldest_memory_age_hours > policy.max_cache_age_hours:
        needs_clst = True
        staleness = oldest_memory_age_hours / policy.max_cache_age_hours
        confidence = min(confidence, 1.0 / staleness)
        reasons.append(f"Stale cache ({oldest_memory_age_hours:.1f}h old)")
        if level == CLSTNeedLevel.NONE:
            level = CLSTNeedLevel.LOW

    # Check 7: Memory type requirements
    required_types = hints.get("required_memory_types", [])
    if required_types:
        hints_used.append(f"required_types={required_types}")
        # This would need cache contents to check, mark for CLST
        needs_clst = True
        reasons.append("Specific memory types required")
        if level == CLSTNeedLevel.NONE:
            level = CLSTNeedLevel.MEDIUM

    return CLSTDecision(
        needs_clst=needs_clst,
        level=level,
        confidence=confidence,
        reason="; ".join(reasons) if reasons else "Cache sufficient",
        hints_used=hints_used,
        cache_hit_count=cache_hits,
        topic_coverage=topic_coverage,
        age_hours=oldest_memory_age_hours,
    )


# =============================================================================
# Simple Memory Cache
# =============================================================================


@dataclass
class CachedMemory:
    """A cached memory entry."""

    memory_id: str
    content: str
    memory_type: str
    user_id: str
    agent_id: str | None = None

    # Metadata
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    importance: float = 0.5

    # Session context
    session_id: str | None = None
    thread_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_age_hours(self) -> float:
        """Get age in hours."""
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "topics": self.topics,
            "categories": self.categories,
            "importance": self.importance,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class SimpleRecallResult:
    """Result from SimpleFLR query."""

    memories: list[CachedMemory]
    clst_decision: CLSTDecision
    query_latency_ms: float

    # Source info
    from_cache: bool = True

    # Pending reinforcement signals to pass to CLST
    pending_signals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_count": len(self.memories),
            "clst_decision": self.clst_decision.to_dict(),
            "query_latency_ms": self.query_latency_ms,
            "from_cache": self.from_cache,
            "pending_signal_count": len(self.pending_signals),
        }


# =============================================================================
# DeterministicRecall - The Deterministic Cache Layer
# =============================================================================


class DeterministicRecall:
    """Deterministic cache layer for fast memory recall.

    This is a streamlined FLR that:
    1. Provides O(1) LRU cache lookup
    2. Applies deterministic filtering (no probabilistic scoring)
    3. Determines if CLST is needed based on metadata hints
    4. Collects reinforcement signals to pass to CLST

    The key difference from probabilistic approaches:
    - NO probabilistic scoring (word overlap, recency decay, exploration)
    - NO complex reinforcement processing
    - Just cache lookup + deterministic filter + CLST decision

    Example:
        flr = DeterministicRecall(storage=storage)

        result = flr.query(
            user_id="user123",
            topics=["orders"],
            metadata_hints={"is_clst_needed": False, "confidence": 0.9},
        )

        if result.clst_decision.needs_clst:
            # Query CLST for complete results
            clst_result = clst.query(...)

            # Pass pending signals to CLST
            for signal in result.pending_signals:
                clst.apply_signal(signal)
    """

    def __init__(
        self,
        storage: BaseStorage | None = None,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 300,
        decision_policy: CLSTDecisionPolicy | None = None,
    ):
        """Initialize SimpleFLR.

        Args:
            storage: Optional storage backend for cache warming
            cache_size: Maximum cache size (LRU eviction)
            cache_ttl_seconds: Cache TTL in seconds
            decision_policy: CLST decision policy configuration
        """
        self._storage = storage
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_seconds
        self._decision_policy = decision_policy or CLSTDecisionPolicy()

        # LRU cache: user_id -> OrderedDict[memory_id -> (CachedMemory, timestamp)]
        self._cache: dict[str, OrderedDict[str, tuple[CachedMemory, float]]] = {}

        # Pending reinforcement signals to pass to CLST
        self._pending_signals: list[dict] = []

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "clst_decisions_needed": 0,
            "clst_decisions_skipped": 0,
            "signals_collected": 0,
        }

    def query(
        self,
        user_id: str,
        session_id: str | None = None,
        topics: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        metadata_hints: dict[str, Any] | None = None,
    ) -> SimpleRecallResult:
        """Fast deterministic cache query.

        Args:
            user_id: User identifier
            session_id: Filter by session
            topics: Filter by topics
            memory_types: Filter by memory types
            limit: Maximum results
            metadata_hints: Hints for CLST decision (is_clst_needed, confidence, priority)

        Returns:
            SimpleRecallResult with memories and CLST decision
        """
        start_time = time.time()
        topics = topics or []
        memory_types = memory_types or []

        # Step 1: Get from cache (O(1) per user)
        user_cache = self._cache.get(user_id, OrderedDict())
        now = time.time()

        # Step 2: Filter and collect matches
        matches: list[CachedMemory] = []
        matched_topics: set[str] = set()
        oldest_age_hours = 0.0

        # Clean expired and filter
        expired = []
        for memory_id, (memory, timestamp) in user_cache.items():
            # Check TTL
            if now - timestamp > self._cache_ttl:
                expired.append(memory_id)
                continue

            # Filter by session
            if session_id and memory.session_id != session_id:
                continue

            # Filter by topics
            if topics:
                memory_topics = set(memory.topics)
                if not (memory_topics & set(topics)):
                    continue
                matched_topics.update(memory_topics & set(topics))

            # Filter by memory types
            if memory_types and memory.memory_type not in memory_types:
                continue

            matches.append(memory)

            # Track oldest memory age
            age = memory.get_age_hours()
            oldest_age_hours = max(age, oldest_age_hours)

        # Clean expired entries
        for mid in expired:
            del user_cache[mid]

        # Step 3: Sort by recency (deterministic)
        matches.sort(key=lambda m: m.created_at, reverse=True)
        matches = matches[:limit]

        # Step 4: Make CLST decision
        clst_decision = make_clst_decision(
            cache_hits=len(matches),
            requested_topics=topics,
            matched_topics=list(matched_topics),
            oldest_memory_age_hours=oldest_age_hours,
            metadata_hints=metadata_hints,
            policy=self._decision_policy,
        )

        # Update stats
        if matches:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1

        if clst_decision.needs_clst:
            self._stats["clst_decisions_needed"] += 1
        else:
            self._stats["clst_decisions_skipped"] += 1

        # Include pending signals in result
        pending = self._pending_signals.copy()

        latency = (time.time() - start_time) * 1000

        return SimpleRecallResult(
            memories=matches,
            clst_decision=clst_decision,
            query_latency_ms=latency,
            from_cache=True,
            pending_signals=pending,
        )

    def cache_memory(
        self,
        memory: dict[str, Any] | CachedMemory,
    ) -> None:
        """Add a memory to the cache.

        Args:
            memory: Memory to cache (dict or CachedMemory)
        """
        if isinstance(memory, dict):
            cached = CachedMemory(
                memory_id=memory.get("memory_id", ""),
                content=memory.get("content", ""),
                memory_type=memory.get("memory_type", "episodic"),
                user_id=memory.get("user_id", ""),
                agent_id=memory.get("agent_id"),
                topics=memory.get("topics", []),
                categories=memory.get("categories", []),
                importance=memory.get("importance", 0.5),
                session_id=memory.get("session_id"),
                thread_id=memory.get("thread_id"),
                created_at=datetime.fromisoformat(memory["created_at"])
                if isinstance(memory.get("created_at"), str)
                else memory.get("created_at", datetime.now(timezone.utc)),
            )
        else:
            cached = memory

        user_id = cached.user_id

        # Initialize user cache if needed
        if user_id not in self._cache:
            self._cache[user_id] = OrderedDict()

        user_cache = self._cache[user_id]

        # LRU eviction
        while len(user_cache) >= self._cache_size:
            user_cache.popitem(last=False)

        # Add to cache
        user_cache[cached.memory_id] = (cached, time.time())
        user_cache.move_to_end(cached.memory_id)

    def collect_signal(
        self,
        memory_id: str,
        signal_type: str,
        signal_value: float,
        source: str = "user",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Collect a reinforcement signal to pass to CLST.

        SimpleFLR doesn't process signals - it just collects them
        for CLST to process during the full path.

        Args:
            memory_id: Memory to reinforce
            signal_type: Type of signal (relevance, usefulness, etc.)
            signal_value: Signal value (-1 to 1)
            source: Signal source (user, llm, automated)
            context: Additional context
        """
        signal = {
            "memory_id": memory_id,
            "signal_type": signal_type,
            "signal_value": signal_value,
            "source": source,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._pending_signals.append(signal)
        self._stats["signals_collected"] += 1

    def get_pending_signals(self) -> list[dict]:
        """Get pending signals to pass to CLST."""
        return self._pending_signals.copy()

    def clear_pending_signals(self) -> int:
        """Clear pending signals after passing to CLST.

        Returns:
            Number of signals cleared
        """
        count = len(self._pending_signals)
        self._pending_signals.clear()
        return count

    def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user.

        Args:
            user_id: User to invalidate

        Returns:
            Number of entries invalidated
        """
        if user_id in self._cache:
            count = len(self._cache[user_id])
            del self._cache[user_id]
            return count
        return 0

    def invalidate_memory(self, memory_id: str) -> bool:
        """Invalidate a specific memory from all caches.

        Args:
            memory_id: Memory to invalidate

        Returns:
            True if found and removed
        """
        for user_cache in self._cache.values():
            if memory_id in user_cache:
                del user_cache[memory_id]
                return True
        return False

    def warm_cache(
        self,
        user_id: str,
        limit: int = 100,
    ) -> int:
        """Warm cache from storage for a user.

        Args:
            user_id: User to warm cache for
            limit: Maximum memories to load

        Returns:
            Number of memories cached
        """
        if not self._storage:
            return 0

        try:
            # Query recent memories from storage
            memories = self._storage.search(
                query="",
                user_id=user_id,
                limit=limit,
            )

            for memory in memories:
                self.cache_memory(memory.to_dict())

            return len(memories)

        except Exception as e:
            logger.warning("Failed to warm cache for user %s: %s", user_id, e)
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_cached = sum(len(uc) for uc in self._cache.values())
        total_decisions = (
            self._stats["clst_decisions_needed"] + self._stats["clst_decisions_skipped"]
        )

        return {
            "total_cached_memories": total_cached,
            "user_count": len(self._cache),
            "cache_size_limit": self._cache_size,
            "cache_ttl_seconds": self._cache_ttl,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "hit_rate": (
                self._stats["cache_hits"]
                / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0
            ),
            "clst_decisions_needed": self._stats["clst_decisions_needed"],
            "clst_decisions_skipped": self._stats["clst_decisions_skipped"],
            "clst_skip_rate": (
                self._stats["clst_decisions_skipped"] / total_decisions
                if total_decisions > 0
                else 0
            ),
            "signals_collected": self._stats["signals_collected"],
            "pending_signals": len(self._pending_signals),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "clst_decisions_needed": 0,
            "clst_decisions_skipped": 0,
            "signals_collected": 0,
        }


# Backwards compatibility alias
SimpleFLR = DeterministicRecall
