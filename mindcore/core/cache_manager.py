"""
In-memory cache manager for recent messages.
"""
from collections import OrderedDict
from typing import List, Dict, Tuple
import threading

from .schemas import Message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Thread-safe in-memory cache for recent messages.

    Uses LRU-like eviction per user/thread combination.
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of messages per user/thread.
        """
        self.max_size = max_size
        self._cache: Dict[Tuple[str, str], OrderedDict] = {}
        self._lock = threading.RLock()
        logger.info(f"Cache manager initialized with max_size={max_size}")

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

    def add_message(self, message: Message) -> None:
        """
        Add a message to cache.

        Args:
            message: Message object to add.
        """
        with self._lock:
            key = self._get_key(message.user_id, message.thread_id)

            if key not in self._cache:
                self._cache[key] = OrderedDict()

            cache = self._cache[key]

            # Add message (most recent first logic)
            cache[message.message_id] = message
            cache.move_to_end(message.message_id)

            # Evict oldest if over limit
            while len(cache) > self.max_size:
                oldest_key = next(iter(cache))
                removed = cache.pop(oldest_key)
                logger.debug(f"Evicted message {oldest_key} from cache")

            logger.debug(f"Added message {message.message_id} to cache")

    def get_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        limit: int = None
    ) -> List[Message]:
        """
        Get recent messages from cache.

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

            cache = self._cache[key]
            messages = list(cache.values())

            # Return in reverse order (most recent first)
            messages.reverse()

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
                logger.info(f"Cleared cache for {user_id}/{thread_id}")

    def clear_all(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared entire cache")

    def get_stats(self) -> Dict[str, int]:
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
                "max_size_per_thread": self.max_size
            }
