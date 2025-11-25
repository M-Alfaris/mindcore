"""
In-memory cache manager for recent messages.
"""
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
import threading
import bisect

from .schemas import Message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Thread-safe in-memory cache for recent messages.

    Messages are stored in chronological order by created_at timestamp.
    Eviction removes oldest messages when cache is full or TTL expires.
    """

    def __init__(self, max_size: int = 50, ttl_seconds: Optional[int] = None):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of messages per user/thread.
            ttl_seconds: Optional time-to-live in seconds for cached messages.
                        If None, messages don't expire based on time.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # Cache stores messages in a dict keyed by message_id
        self._cache: Dict[Tuple[str, str], Dict[str, Message]] = {}
        # Sorted list of (created_at, message_id) for chronological ordering
        self._order: Dict[Tuple[str, str], List[Tuple[datetime, str]]] = {}
        # Track when each message was added to cache (for TTL)
        self._added_at: Dict[Tuple[str, str], Dict[str, datetime]] = {}
        self._lock = threading.RLock()
        logger.info(f"Cache manager initialized with max_size={max_size}, ttl_seconds={ttl_seconds}")

    def _get_key(self, user_id: str, thread_id: str) -> Tuple[str, str]:
        """
        Generate cache key.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.

        Returns:
            Cache key tuple.
        """
        return (user_id, thread_id)

    def _get_timestamp(self, message: Message) -> datetime:
        """Get timestamp for message, using current time if not set."""
        if message.created_at:
            return message.created_at
        return datetime.now(timezone.utc)

    def add_message(self, message: Message) -> None:
        """
        Add a message to cache, maintaining chronological order.

        Args:
            message: Message object to add.
        """
        with self._lock:
            key = self._get_key(message.user_id, message.thread_id)

            if key not in self._cache:
                self._cache[key] = {}
                self._order[key] = []
                self._added_at[key] = {}

            cache = self._cache[key]
            order = self._order[key]
            added_at = self._added_at[key]

            # If message already exists, remove old entry from order list
            if message.message_id in cache:
                old_msg = cache[message.message_id]
                old_ts = self._get_timestamp(old_msg)
                try:
                    order.remove((old_ts, message.message_id))
                except ValueError:
                    pass

            # Add message to cache
            cache[message.message_id] = message
            added_at[message.message_id] = datetime.now(timezone.utc)

            # Insert into sorted order list (by timestamp)
            timestamp = self._get_timestamp(message)
            bisect.insort(order, (timestamp, message.message_id))

            # Evict oldest if over limit
            while len(cache) > self.max_size:
                if order:
                    # Remove oldest (first in sorted list)
                    oldest_ts, oldest_id = order.pop(0)
                    if oldest_id in cache:
                        del cache[oldest_id]
                    if oldest_id in added_at:
                        del added_at[oldest_id]
                    logger.debug(f"Evicted message {oldest_id} from cache (size limit)")
                else:
                    break

            logger.debug(f"Added message {message.message_id} to cache")

    def _is_expired(self, key: Tuple[str, str], msg_id: str) -> bool:
        """Check if a message has expired based on TTL."""
        if self.ttl_seconds is None:
            return False

        added_at = self._added_at.get(key, {}).get(msg_id)
        if added_at is None:
            return False

        expiry_time = added_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time

    def _cleanup_expired(self, key: Tuple[str, str]) -> int:
        """Remove expired messages from a specific thread cache. Returns count removed."""
        if self.ttl_seconds is None:
            return 0

        if key not in self._cache:
            return 0

        cache = self._cache[key]
        order = self._order.get(key, [])
        added_at = self._added_at.get(key, {})

        expired_ids = [msg_id for msg_id in cache if self._is_expired(key, msg_id)]

        for msg_id in expired_ids:
            if msg_id in cache:
                msg = cache[msg_id]
                ts = self._get_timestamp(msg)
                try:
                    order.remove((ts, msg_id))
                except ValueError:
                    pass
                del cache[msg_id]
            if msg_id in added_at:
                del added_at[msg_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired messages from cache")

        return len(expired_ids)

    def get_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        limit: int = None
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

            if key not in self._cache:
                return []

            # Cleanup expired messages first
            self._cleanup_expired(key)

            cache = self._cache[key]
            order = self._order.get(key, [])

            # Get messages in chronological order (newest first = reversed)
            messages = []
            for timestamp, msg_id in reversed(order):
                if msg_id in cache:
                    messages.append(cache[msg_id])

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
            if key in self._cache:
                del self._cache[key]
            if key in self._order:
                del self._order[key]
            if key in self._added_at:
                del self._added_at[key]
            logger.info(f"Cleared cache for {user_id}/{thread_id}")

    def clear_all(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._added_at.clear()
            logger.info("Cleared entire cache")

    def cleanup_all_expired(self) -> int:
        """
        Cleanup expired messages from all thread caches.

        Returns:
            Total number of expired messages removed.
        """
        with self._lock:
            total_removed = 0
            for key in list(self._cache.keys()):
                total_removed += self._cleanup_expired(key)
            return total_removed

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        with self._lock:
            total_messages = sum(len(cache) for cache in self._cache.values())
            return {
                "total_threads": len(self._cache),
                "total_messages": total_messages,
                "max_size_per_thread": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }
