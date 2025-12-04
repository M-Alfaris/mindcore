"""Disk-backed cache manager for persistent message caching.

Uses diskcache (SQLite-backed) for:
- Persistence across restarts
- Overflow to disk when memory is full
- Multi-process safe access
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import diskcache

from mindcore.utils.logger import get_logger

from .schemas import Message, MessageMetadata, MessageRole


logger = get_logger(__name__)


class DiskCacheManager:
    """Persistent disk-backed cache for messages.

    Uses diskcache (SQLite-backed) which provides:
    - Persistence: Survives process restarts
    - Multi-process: Safe concurrent access
    - Overflow: Automatically spills to disk
    - Fast: Memory-mapped SQLite for speed

    Messages are stored per user/thread combination.
    """

    def __init__(
        self,
        max_size: int = 50,
        ttl_seconds: int | None = None,
        cache_dir: str | None = None,
        size_limit: int = 100 * 1024 * 1024,  # 100MB default
    ):
        """Initialize disk cache manager.

        Args:
            max_size: Maximum number of messages per user/thread.
            ttl_seconds: Optional time-to-live in seconds for cached messages.
            cache_dir: Directory for cache files. Defaults to temp directory.
            size_limit: Total cache size limit in bytes (default 100MB).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Set up cache directory
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path(tempfile.gettempdir()) / "mindcore_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diskcache with FanoutCache for better concurrency
        self._cache = diskcache.FanoutCache(
            str(self._cache_dir),
            shards=4,  # Multiple shards for concurrency
            size_limit=size_limit,
            eviction_policy="least-recently-used",
        )

        logger.info(
            f"DiskCacheManager initialized: dir={self._cache_dir}, "
            f"max_size={max_size}, ttl={ttl_seconds}s"
        )

    def _get_key(self, user_id: str, thread_id: str, message_id: str) -> str:
        """Generate cache key for a message."""
        return f"msg:{user_id}:{thread_id}:{message_id}"

    def _get_thread_key(self, user_id: str, thread_id: str) -> str:
        """Generate key for thread message list."""
        return f"thread:{user_id}:{thread_id}"

    def _message_to_dict(self, message: Message) -> dict[str, Any]:
        """Convert Message to serializable dict."""
        return {
            "message_id": message.message_id,
            "user_id": message.user_id,
            "thread_id": message.thread_id,
            "session_id": message.session_id,
            "role": message.role.value if isinstance(message.role, MessageRole) else message.role,
            "raw_text": message.raw_text,
            "metadata": (
                message.metadata.to_dict()
                if hasattr(message.metadata, "to_dict")
                else message.metadata
            ),
            "created_at": message.created_at.isoformat() if message.created_at else None,
        }

    def _dict_to_message(self, data: dict[str, Any]) -> Message:
        """Convert dict back to Message object."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = MessageMetadata(**metadata)

        return Message(
            message_id=data["message_id"],
            user_id=data["user_id"],
            thread_id=data["thread_id"],
            session_id=data["session_id"],
            role=MessageRole(data["role"]) if data.get("role") else MessageRole.USER,
            raw_text=data["raw_text"],
            metadata=metadata,
            created_at=created_at,
        )

    def add_message(self, message: Message) -> None:
        """Add a message to cache.

        Args:
            message: Message object to add.
        """
        key = self._get_key(message.user_id, message.thread_id, message.message_id)
        thread_key = self._get_thread_key(message.user_id, message.thread_id)

        # Serialize message
        msg_data = self._message_to_dict(message)

        # Store message with optional TTL
        expire = self.ttl_seconds if self.ttl_seconds else None
        self._cache.set(key, msg_data, expire=expire)

        # Update thread message list
        thread_messages = self._cache.get(thread_key, default=[])
        if not isinstance(thread_messages, list):
            thread_messages = []

        # Remove if exists (will re-add at end)
        thread_messages = [mid for mid in thread_messages if mid != message.message_id]

        # Add to end and trim to max_size
        thread_messages.append(message.message_id)
        if len(thread_messages) > self.max_size:
            # Remove oldest messages
            removed = thread_messages[: -self.max_size]
            thread_messages = thread_messages[-self.max_size :]
            # Clean up removed message data
            for mid in removed:
                old_key = self._get_key(message.user_id, message.thread_id, mid)
                self._cache.delete(old_key)

        self._cache.set(thread_key, thread_messages, expire=expire)
        logger.debug(f"Added message {message.message_id} to disk cache")

    def update_message(self, message: Message) -> bool:
        """Update an existing message in cache.

        Args:
            message: Message object with updated data.

        Returns:
            True if message was in cache and updated, False otherwise.
        """
        key = self._get_key(message.user_id, message.thread_id, message.message_id)

        if key not in self._cache:
            return False

        # Update with same TTL
        msg_data = self._message_to_dict(message)
        expire = self.ttl_seconds if self.ttl_seconds else None
        self._cache.set(key, msg_data, expire=expire)
        logger.debug(f"Updated message {message.message_id} in disk cache")
        return True

    def get_recent_messages(
        self, user_id: str, thread_id: str, limit: int | None = None
    ) -> list[Message]:
        """Get recent messages from cache in chronological order.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            limit: Maximum number of messages (default: all cached).

        Returns:
            List of Message objects, most recent first.
        """
        thread_key = self._get_thread_key(user_id, thread_id)
        thread_messages = self._cache.get(thread_key, default=[])

        if not thread_messages:
            return []

        # Get messages in reverse order (newest first)
        messages = []
        for msg_id in reversed(thread_messages):
            key = self._get_key(user_id, thread_id, msg_id)
            msg_data = self._cache.get(key)
            if msg_data:
                try:
                    messages.append(self._dict_to_message(msg_data))
                except Exception as e:
                    logger.warning(f"Failed to deserialize message {msg_id}: {e}")

            if limit and len(messages) >= limit:
                break

        logger.debug(
            f"Retrieved {len(messages)} messages from disk cache for {user_id}/{thread_id}"
        )
        return messages

    def clear_thread(self, user_id: str, thread_id: str) -> None:
        """Clear cache for a specific thread.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
        """
        thread_key = self._get_thread_key(user_id, thread_id)
        thread_messages = self._cache.get(thread_key, default=[])

        # Delete all messages
        for msg_id in thread_messages:
            key = self._get_key(user_id, thread_id, msg_id)
            self._cache.delete(key)

        # Delete thread list
        self._cache.delete(thread_key)
        logger.info(f"Cleared disk cache for {user_id}/{thread_id}")

    def clear_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        logger.info("Cleared entire disk cache")

    def get_session_metadata(self, user_id: str, thread_id: str) -> dict[str, Any]:
        """Aggregate metadata from all cached messages in the current session.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.

        Returns:
            Dict with aggregated metadata.
        """
        messages = self.get_recent_messages(user_id, thread_id)

        topics = set()
        categories = set()
        intents = set()

        for message in messages:
            if hasattr(message, "metadata") and message.metadata:
                meta = message.metadata
                if hasattr(meta, "topics") and meta.topics:
                    topics.update(meta.topics)
                if hasattr(meta, "categories") and meta.categories:
                    categories.update(meta.categories)
                if hasattr(meta, "intent") and meta.intent:
                    intents.add(meta.intent)

        return {
            "topics": list(topics),
            "categories": list(categories),
            "intents": list(intents),
            "message_count": len(messages),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "cache_type": "DiskCache (SQLite-backed)",
            "cache_dir": str(self._cache_dir),
            "volume": self._cache.volume(),
            "size_limit": self._cache.size_limit,
            "max_size_per_thread": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }

    def close(self) -> None:
        """Close cache and release resources."""
        self._cache.close()
        logger.info("DiskCacheManager closed")
