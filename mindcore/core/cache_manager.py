"""
In-memory cache manager for recent messages.

Uses cachetools for battle-tested caching with TTL and LRU support.
"""
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone
import threading

from cachetools import TTLCache, LRUCache

from .schemas import Message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Thread-safe in-memory cache for recent messages.

    Uses cachetools TTLCache for automatic expiration and LRU eviction.
    Messages are stored per user/thread combination.
    """

    def __init__(self, max_size: int = 50, ttl_seconds: Optional[int] = None):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of messages per user/thread.
            ttl_seconds: Optional time-to-live in seconds for cached messages.
                        If None, uses LRU cache without TTL.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

        # Thread caches: (user_id, thread_id) -> TTLCache/LRUCache of messages
        self._thread_caches: Dict[Tuple[str, str], Any] = {}
        # Track message order per thread for chronological retrieval
        self._order: Dict[Tuple[str, str], List[Tuple[datetime, str]]] = {}

        logger.info(f"Cache manager initialized with max_size={max_size}, ttl_seconds={ttl_seconds}")

    def _get_key(self, user_id: str, thread_id: str) -> Tuple[str, str]:
        """Generate cache key."""
        return (user_id, thread_id)

    def _get_timestamp(self, message: Message) -> datetime:
        """Get timestamp for message, using current time if not set."""
        if message.created_at:
            return message.created_at
        return datetime.now(timezone.utc)

    def _create_thread_cache(self) -> Any:
        """Create a new cache for a thread."""
        if self.ttl_seconds:
            return TTLCache(maxsize=self.max_size, ttl=self.ttl_seconds)
        return LRUCache(maxsize=self.max_size)

    def add_message(self, message: Message) -> None:
        """
        Add a message to cache.

        Args:
            message: Message object to add.
        """
        with self._lock:
            key = self._get_key(message.user_id, message.thread_id)

            # Initialize thread cache if needed
            if key not in self._thread_caches:
                self._thread_caches[key] = self._create_thread_cache()
                self._order[key] = []

            cache = self._thread_caches[key]

            # Add message to cache (cachetools handles eviction automatically)
            cache[message.message_id] = message

            # Track order for chronological retrieval
            timestamp = self._get_timestamp(message)
            order = self._order[key]

            # Remove old entry if message already exists
            order = [(ts, mid) for ts, mid in order if mid != message.message_id]

            # Add new entry and sort
            order.append((timestamp, message.message_id))
            order.sort(key=lambda x: x[0])

            # Trim order list to max_size (in case of edge cases)
            if len(order) > self.max_size * 2:
                order = order[-self.max_size:]

            self._order[key] = order

            logger.debug(f"Added message {message.message_id} to cache")

    def get_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get recent messages from cache in chronological order.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            limit: Maximum number of messages (default: all cached).

        Returns:
            List of Message objects, most recent first.
        """
        with self._lock:
            key = self._get_key(user_id, thread_id)

            if key not in self._thread_caches:
                return []

            cache = self._thread_caches[key]
            order = self._order.get(key, [])

            # Get messages in reverse chronological order (newest first)
            messages = []
            for timestamp, msg_id in reversed(order):
                try:
                    # cachetools handles TTL expiration automatically on access
                    if msg_id in cache:
                        messages.append(cache[msg_id])
                except KeyError:
                    # Message was evicted or expired
                    pass

            if limit:
                messages = messages[:limit]

            logger.debug(f"Retrieved {len(messages)} messages from cache for {user_id}/{thread_id}")
            return messages

    def clear_thread(self, user_id: str, thread_id: str) -> None:
        """
        Clear cache for a specific thread.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
        """
        with self._lock:
            key = self._get_key(user_id, thread_id)
            if key in self._thread_caches:
                del self._thread_caches[key]
            if key in self._order:
                del self._order[key]
            logger.info(f"Cleared cache for {user_id}/{thread_id}")

    def clear_all(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._thread_caches.clear()
            self._order.clear()
            logger.info("Cleared entire cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        with self._lock:
            total_messages = 0
            for cache in self._thread_caches.values():
                try:
                    total_messages += len(cache)
                except Exception:
                    pass

            return {
                "total_threads": len(self._thread_caches),
                "total_messages": total_messages,
                "max_size_per_thread": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "cache_type": "TTLCache" if self.ttl_seconds else "LRUCache"
            }
