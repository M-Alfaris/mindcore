"""
Cache Invalidation Manager.

Provides intelligent cache invalidation strategies:
- Event-based invalidation
- Selective invalidation by user, thread, topic
- TTL-based automatic expiration
- Cache warming after invalidation
"""
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import time

from .schemas import Message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InvalidationReason(str, Enum):
    """Reasons for cache invalidation."""
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    METADATA_ENRICHED = "metadata_enriched"
    THREAD_SUMMARIZED = "thread_summarized"
    USER_PREFERENCES_CHANGED = "user_preferences_changed"
    TIER_MIGRATION = "tier_migration"
    TTL_EXPIRED = "ttl_expired"
    MANUAL = "manual"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass
class InvalidationEvent:
    """An event that triggered cache invalidation."""
    reason: InvalidationReason
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationStats:
    """Statistics about cache invalidation."""
    total_invalidations: int = 0
    by_reason: Dict[str, int] = field(default_factory=dict)
    last_invalidation: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def record_invalidation(self, reason: InvalidationReason) -> None:
        """Record an invalidation event."""
        self.total_invalidations += 1
        self.by_reason[reason.value] = self.by_reason.get(reason.value, 0) + 1
        self.last_invalidation = datetime.now(timezone.utc)

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_invalidations": self.total_invalidations,
            "by_reason": self.by_reason,
            "last_invalidation": self.last_invalidation.isoformat() if self.last_invalidation else None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheInvalidationManager:
    """
    Manages cache invalidation with intelligent strategies.

    Supports:
    - Event-based invalidation triggers
    - Selective invalidation by various criteria
    - Invalidation callbacks for different cache layers
    - Statistics and monitoring

    Example:
        >>> invalidation = CacheInvalidationManager()
        >>>
        >>> # Register cache to invalidate
        >>> invalidation.register_cache("messages", cache_manager)
        >>>
        >>> # Invalidate when message is enriched
        >>> invalidation.invalidate_message(
        ...     message_id="msg123",
        ...     reason=InvalidationReason.METADATA_ENRICHED
        ... )
    """

    def __init__(
        self,
        enable_cascading: bool = True,
        max_event_history: int = 1000
    ):
        """
        Initialize the cache invalidation manager.

        Args:
            enable_cascading: If True, invalidation cascades to related entries
            max_event_history: Maximum invalidation events to keep in history
        """
        self.enable_cascading = enable_cascading
        self.max_event_history = max_event_history
        self._lock = threading.Lock()

        # Registered caches
        self._caches: Dict[str, Any] = {}

        # Invalidation callbacks
        self._callbacks: Dict[str, List[Callable[[InvalidationEvent], None]]] = {}

        # Event history
        self._event_history: List[InvalidationEvent] = []

        # Statistics
        self._stats = InvalidationStats()

        # Topic -> (user_id, thread_id) mapping for cascading invalidation
        self._topic_index: Dict[str, Set[tuple]] = {}

    def register_cache(
        self,
        name: str,
        cache: Any,
        invalidation_callback: Optional[Callable] = None
    ) -> None:
        """
        Register a cache for invalidation management.

        Args:
            name: Cache identifier
            cache: Cache instance (must have clear_thread, update_message methods)
            invalidation_callback: Optional callback when cache is invalidated
        """
        with self._lock:
            self._caches[name] = cache
            if invalidation_callback:
                if name not in self._callbacks:
                    self._callbacks[name] = []
                self._callbacks[name].append(invalidation_callback)

        logger.info(f"Registered cache '{name}' for invalidation management")

    def unregister_cache(self, name: str) -> None:
        """Unregister a cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
            if name in self._callbacks:
                del self._callbacks[name]

    def index_message(self, message: Message) -> None:
        """
        Index a message's topics for cascading invalidation.

        Args:
            message: Message to index
        """
        if not hasattr(message.metadata, 'topics') or not message.metadata.topics:
            return

        with self._lock:
            key = (message.user_id, message.thread_id)
            for topic in message.metadata.topics:
                if topic not in self._topic_index:
                    self._topic_index[topic] = set()
                self._topic_index[topic].add(key)

    def invalidate_message(
        self,
        message_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        reason: InvalidationReason = InvalidationReason.MESSAGE_UPDATED,
        cascade: bool = True
    ) -> None:
        """
        Invalidate a specific message across all caches.

        Args:
            message_id: Message to invalidate
            user_id: User ID (required for thread-based caches)
            thread_id: Thread ID (required for thread-based caches)
            reason: Reason for invalidation
            cascade: If True and cascading is enabled, invalidate related entries
        """
        event = InvalidationEvent(
            reason=reason,
            user_id=user_id,
            thread_id=thread_id,
            message_id=message_id
        )

        with self._lock:
            self._record_event(event)

            # Invalidate in all registered caches
            for name, cache in self._caches.items():
                try:
                    if hasattr(cache, 'invalidate_message'):
                        cache.invalidate_message(message_id)
                    elif user_id and thread_id and hasattr(cache, 'clear_thread'):
                        # For simple caches, clear the whole thread
                        cache.clear_thread(user_id, thread_id)
                except Exception as e:
                    logger.error(f"Error invalidating message in cache '{name}': {e}")

            # Trigger callbacks
            self._trigger_callbacks(event)

        logger.debug(f"Invalidated message {message_id} ({reason.value})")

    def invalidate_thread(
        self,
        user_id: str,
        thread_id: str,
        reason: InvalidationReason = InvalidationReason.MANUAL
    ) -> None:
        """
        Invalidate all cached data for a thread.

        Args:
            user_id: User identifier
            thread_id: Thread identifier
            reason: Reason for invalidation
        """
        event = InvalidationEvent(
            reason=reason,
            user_id=user_id,
            thread_id=thread_id
        )

        with self._lock:
            self._record_event(event)

            for name, cache in self._caches.items():
                try:
                    if hasattr(cache, 'clear_thread'):
                        cache.clear_thread(user_id, thread_id)
                except Exception as e:
                    logger.error(f"Error invalidating thread in cache '{name}': {e}")

            self._trigger_callbacks(event)

        logger.debug(f"Invalidated thread {user_id}/{thread_id} ({reason.value})")

    def invalidate_by_topic(
        self,
        topic: str,
        reason: InvalidationReason = InvalidationReason.MANUAL
    ) -> int:
        """
        Invalidate all cached data related to a topic.

        Args:
            topic: Topic to invalidate
            reason: Reason for invalidation

        Returns:
            Number of threads invalidated
        """
        invalidated = 0

        with self._lock:
            if topic not in self._topic_index:
                return 0

            threads = list(self._topic_index[topic])

        for user_id, thread_id in threads:
            self.invalidate_thread(user_id, thread_id, reason)
            invalidated += 1

        logger.info(f"Invalidated {invalidated} threads for topic '{topic}'")
        return invalidated

    def invalidate_by_user(
        self,
        user_id: str,
        reason: InvalidationReason = InvalidationReason.USER_PREFERENCES_CHANGED
    ) -> None:
        """
        Invalidate all cached data for a user.

        Args:
            user_id: User identifier
            reason: Reason for invalidation
        """
        event = InvalidationEvent(
            reason=reason,
            user_id=user_id
        )

        with self._lock:
            self._record_event(event)

            for name, cache in self._caches.items():
                try:
                    if hasattr(cache, 'clear_user'):
                        cache.clear_user(user_id)
                    elif hasattr(cache, '_thread_caches'):
                        # For caches with thread structure
                        keys_to_remove = [
                            k for k in cache._thread_caches.keys()
                            if k[0] == user_id
                        ]
                        for key in keys_to_remove:
                            if hasattr(cache, 'clear_thread'):
                                cache.clear_thread(key[0], key[1])
                except Exception as e:
                    logger.error(f"Error invalidating user in cache '{name}': {e}")

            self._trigger_callbacks(event)

        logger.debug(f"Invalidated all cache for user {user_id}")

    def invalidate_stale(
        self,
        max_age_seconds: int = 3600
    ) -> int:
        """
        Invalidate entries older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries invalidated
        """
        # This is a simplified implementation
        # Real implementation would need timestamp tracking per entry
        invalidated = 0
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

        with self._lock:
            # For each cache with order tracking
            for name, cache in self._caches.items():
                if hasattr(cache, '_order'):
                    for key, order in list(cache._order.items()):
                        # Remove entries older than cutoff
                        new_order = [
                            (ts, mid) for ts, mid in order
                            if ts >= cutoff
                        ]
                        if len(new_order) < len(order):
                            invalidated += len(order) - len(new_order)
                            cache._order[key] = new_order

        if invalidated > 0:
            self._stats.record_invalidation(InvalidationReason.TTL_EXPIRED)
            logger.info(f"Invalidated {invalidated} stale entries")

        return invalidated

    def notify_enrichment_complete(
        self,
        message: Message
    ) -> None:
        """
        Notify that a message has been enriched.

        Updates caches and triggers any cascading invalidation.

        Args:
            message: The enriched message
        """
        # Index for topic-based invalidation
        self.index_message(message)

        # Update in caches
        with self._lock:
            for name, cache in self._caches.items():
                try:
                    if hasattr(cache, 'update_message'):
                        cache.update_message(message)
                except Exception as e:
                    logger.error(f"Error updating message in cache '{name}': {e}")

        event = InvalidationEvent(
            reason=InvalidationReason.METADATA_ENRICHED,
            user_id=message.user_id,
            thread_id=message.thread_id,
            message_id=message.message_id,
            topics=getattr(message.metadata, 'topics', []) or []
        )
        self._record_event(event)
        self._trigger_callbacks(event)

    def record_cache_access(self, hit: bool) -> None:
        """Record a cache access for statistics."""
        if hit:
            self._stats.record_hit()
        else:
            self._stats.record_miss()

    def get_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        return self._stats.to_dict()

    def get_event_history(
        self,
        limit: int = 100,
        reason: Optional[InvalidationReason] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent invalidation events.

        Args:
            limit: Maximum events to return
            reason: Filter by reason

        Returns:
            List of event dictionaries
        """
        with self._lock:
            events = self._event_history[-limit:]
            if reason:
                events = [e for e in events if e.reason == reason]

            return [
                {
                    "reason": e.reason.value,
                    "timestamp": e.timestamp.isoformat(),
                    "user_id": e.user_id,
                    "thread_id": e.thread_id,
                    "message_id": e.message_id,
                    "topics": e.topics,
                }
                for e in events
            ]

    def _record_event(self, event: InvalidationEvent) -> None:
        """Record an invalidation event."""
        self._stats.record_invalidation(event.reason)
        self._event_history.append(event)

        # Trim history if needed
        if len(self._event_history) > self.max_event_history:
            self._event_history = self._event_history[-self.max_event_history:]

    def _trigger_callbacks(self, event: InvalidationEvent) -> None:
        """Trigger registered callbacks for an invalidation event."""
        for name, callbacks in self._callbacks.items():
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in invalidation callback for '{name}': {e}")


# Singleton instance
_invalidation_manager: Optional[CacheInvalidationManager] = None
_manager_lock = threading.Lock()


def get_cache_invalidation() -> CacheInvalidationManager:
    """Get the singleton cache invalidation manager."""
    global _invalidation_manager
    if _invalidation_manager is None:
        with _manager_lock:
            if _invalidation_manager is None:
                _invalidation_manager = CacheInvalidationManager()
    return _invalidation_manager


def reset_cache_invalidation() -> None:
    """Reset the singleton manager (for testing)."""
    global _invalidation_manager
    with _manager_lock:
        _invalidation_manager = None
