"""Smart Cache - Write-through cache with pattern-based invalidation.

Provides intelligent cache management for MindCore memories with:
- Write-through caching (DB first, then cache)
- Pattern-based invalidation (e.g., user:123:*)
- Cache warming for high-importance memories
- Event tracking and statistics
"""

from __future__ import annotations

import fnmatch
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from mindcore.v2.flr.recall import Memory
    from mindcore.v2.storage.base import BaseStorage


class CacheEventType(str, Enum):
    """Types of cache events for monitoring."""

    HIT = "hit"
    MISS = "miss"
    STORE = "store"
    INVALIDATE = "invalidate"
    EVICT = "evict"
    WARM = "warm"
    EXPIRE = "expire"


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    memory: Memory
    cached_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    warmed: bool = False  # Was this entry pre-warmed?

    @property
    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds."""
        return time.time() - self.cached_at


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    invalidations: int = 0
    evictions: int = 0
    warms: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "invalidations": self.invalidations,
            "evictions": self.evictions,
            "warms": self.warms,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
        }


class SmartCache:
    """Write-through cache with intelligent invalidation.

    Features:
    - Write-through: Updates DB first, then cache
    - Pattern invalidation: Invalidate by patterns (user:123:*)
    - Cache warming: Pre-load high-importance memories
    - Event callbacks: Monitor cache behavior

    Example:
        cache = SmartCache(
            storage=storage,
            max_size=1000,
            ttl_seconds=300,
            warm_threshold=0.7,
        )

        # Store with automatic cache update
        await cache.store(memory)

        # Get with cache lookup
        memory = cache.get("mem_123")

        # Invalidate user's cached memories
        cache.invalidate_pattern("user:user_123:*")
    """

    def __init__(
        self,
        storage: BaseStorage,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        warm_threshold: float = 0.7,
        on_event: Callable[[CacheEventType, str, dict[str, Any]], None] | None = None,
    ):
        """Initialize SmartCache.

        Args:
            storage: Storage backend for write-through
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for cache entries
            warm_threshold: Importance threshold for auto-warming (0.0-1.0)
            on_event: Optional callback for cache events
        """
        self.storage = storage
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.warm_threshold = warm_threshold
        self.on_event = on_event

        # Primary cache: memory_id -> CacheEntry
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Secondary indexes for pattern invalidation
        self._user_index: dict[str, set[str]] = {}  # user_id -> set of memory_ids
        self._agent_index: dict[str, set[str]] = {}  # agent_id -> set of memory_ids
        self._topic_index: dict[str, set[str]] = {}  # topic -> set of memory_ids
        self._type_index: dict[str, set[str]] = {}  # memory_type -> set of memory_ids

        # Statistics
        self._stats = CacheStats()

    def store(self, memory: Memory) -> str:
        """Store memory with write-through caching.

        Writes to storage first, then updates cache and invalidates
        related entries.

        Args:
            memory: Memory to store

        Returns:
            Memory ID
        """
        # Write-through: DB first
        memory_id = self.storage.store(memory)

        # Invalidate any existing cache entries for this memory
        # (in case of update)
        if memory_id in self._cache:
            self._remove_from_indexes(memory_id)
            del self._cache[memory_id]

        # Invalidate related entries that might reference stale data
        # E.g., if user's preference changes, invalidate cached queries
        self._invalidate_related(memory)

        # Optionally warm cache for important memories
        if self._should_warm(memory):
            self._warm(memory)
        else:
            # Still cache it, just not marked as "warmed"
            self._cache_memory(memory, warmed=False)

        self._stats.stores += 1
        self._emit_event(
            CacheEventType.STORE,
            memory_id,
            {
                "user_id": memory.user_id,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
            },
        )

        return memory_id

    def get(self, memory_id: str) -> Memory | None:
        """Get memory from cache or storage.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory if found, None otherwise
        """
        # Check cache first
        entry = self._cache.get(memory_id)

        if entry:
            # Check TTL
            if entry.age_seconds > self.ttl_seconds:
                self._expire(memory_id)
                # Fall through to storage lookup
            else:
                # Cache hit
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._cache.move_to_end(memory_id)
                self._stats.hits += 1
                self._emit_event(
                    CacheEventType.HIT,
                    memory_id,
                    {
                        "age_seconds": entry.age_seconds,
                        "access_count": entry.access_count,
                    },
                )
                return entry.memory

        # Cache miss - fetch from storage
        self._stats.misses += 1
        memory = self.storage.get(memory_id)

        if memory:
            self._cache_memory(memory, warmed=False)
            self._emit_event(
                CacheEventType.MISS,
                memory_id,
                {
                    "found_in_storage": True,
                },
            )
        else:
            self._emit_event(
                CacheEventType.MISS,
                memory_id,
                {
                    "found_in_storage": False,
                },
            )

        return memory

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories efficiently.

        Args:
            memory_ids: List of memory IDs

        Returns:
            List of found memories
        """
        results = []
        missing_ids = []

        # Check cache for each
        for memory_id in memory_ids:
            entry = self._cache.get(memory_id)
            if entry and entry.age_seconds <= self.ttl_seconds:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._cache.move_to_end(memory_id)
                results.append(entry.memory)
                self._stats.hits += 1
            else:
                if entry:
                    self._expire(memory_id)
                missing_ids.append(memory_id)
                self._stats.misses += 1

        # Batch fetch missing from storage
        if missing_ids:
            for memory_id in missing_ids:
                memory = self.storage.get(memory_id)
                if memory:
                    self._cache_memory(memory, warmed=False)
                    results.append(memory)

        return results

    def invalidate(self, memory_id: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            memory_id: Memory ID to invalidate

        Returns:
            True if entry was found and invalidated
        """
        if memory_id in self._cache:
            self._remove_from_indexes(memory_id)
            del self._cache[memory_id]
            self._stats.invalidations += 1
            self._emit_event(
                CacheEventType.INVALIDATE,
                memory_id,
                {
                    "pattern": None,
                },
            )
            return True
        return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Supports patterns like:
        - user:123:* - All memories for user 123
        - agent:bot_1:* - All memories from agent bot_1
        - topic:orders:* - All memories with topic "orders"
        - type:preference:* - All preference memories
        - *:user_123:preference - User's preferences specifically

        Args:
            pattern: Glob-style pattern to match

        Returns:
            Number of entries invalidated
        """
        count = 0
        to_invalidate = self._match_pattern(pattern)

        for memory_id in to_invalidate:
            if memory_id in self._cache:
                self._remove_from_indexes(memory_id)
                del self._cache[memory_id]
                count += 1

        if count > 0:
            self._stats.invalidations += count
            self._emit_event(
                CacheEventType.INVALIDATE,
                pattern,
                {
                    "pattern": pattern,
                    "count": count,
                },
            )

        return count

    def invalidate_user(self, user_id: str) -> int:
        """Invalidate all cached memories for a user.

        Args:
            user_id: User ID

        Returns:
            Number of entries invalidated
        """
        memory_ids = self._user_index.get(user_id, set()).copy()
        count = 0

        for memory_id in memory_ids:
            if self.invalidate(memory_id):
                count += 1

        return count

    def warm(self, memory: Memory) -> None:
        """Explicitly warm cache with a memory.

        Use this for memories you know will be accessed soon.

        Args:
            memory: Memory to warm
        """
        self._warm(memory)

    def warm_user(
        self,
        user_id: str,
        limit: int = 50,
        memory_types: list[str] | None = None,
        min_importance: float = 0.5,
    ) -> int:
        """Warm cache with a user's important memories.

        Args:
            user_id: User ID
            limit: Maximum memories to warm
            memory_types: Optional filter by memory types
            min_importance: Minimum importance threshold

        Returns:
            Number of memories warmed
        """
        # Query storage for user's important memories
        memories = self.storage.search(
            user_id=user_id,
            memory_types=memory_types,
            min_importance=min_importance,
            limit=limit,
        )

        count = 0
        for memory in memories:
            if memory.memory_id not in self._cache:
                self._warm(memory)
                count += 1

        return count

    def clear(self) -> int:
        """Clear the entire cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._user_index.clear()
        self._agent_index.clear()
        self._topic_index.clear()
        self._type_index.clear()
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired = [
            mid for mid, entry in self._cache.items() if now - entry.cached_at > self.ttl_seconds
        ]

        for memory_id in expired:
            self._expire(memory_id)

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            **self._stats.to_dict(),
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "warm_threshold": self.warm_threshold,
            "index_sizes": {
                "users": len(self._user_index),
                "agents": len(self._agent_index),
                "topics": len(self._topic_index),
                "types": len(self._type_index),
            },
        }

    def _cache_memory(self, memory: Memory, warmed: bool = False) -> None:
        """Add memory to cache with indexing.

        Args:
            memory: Memory to cache
            warmed: Whether this is a warm operation
        """
        # LRU eviction if at capacity
        while len(self._cache) >= self.max_size:
            evicted_id, _ = self._cache.popitem(last=False)
            self._remove_from_indexes(evicted_id)
            self._stats.evictions += 1
            self._emit_event(CacheEventType.EVICT, evicted_id, {})

        # Create cache entry
        entry = CacheEntry(
            memory=memory,
            cached_at=time.time(),
            warmed=warmed,
        )

        self._cache[memory.memory_id] = entry
        self._cache.move_to_end(memory.memory_id)

        # Update indexes
        self._add_to_indexes(memory)

    def _warm(self, memory: Memory) -> None:
        """Warm cache with memory.

        Args:
            memory: Memory to warm
        """
        self._cache_memory(memory, warmed=True)
        self._stats.warms += 1
        self._emit_event(
            CacheEventType.WARM,
            memory.memory_id,
            {
                "importance": memory.importance,
                "effective_importance": memory.effective_importance,
            },
        )

    def _should_warm(self, memory: Memory) -> bool:
        """Check if memory should be auto-warmed.

        Args:
            memory: Memory to check

        Returns:
            True if should be warmed
        """
        return memory.effective_importance >= self.warm_threshold

    def _expire(self, memory_id: str) -> None:
        """Remove expired entry from cache.

        Args:
            memory_id: Memory ID to expire
        """
        if memory_id in self._cache:
            self._remove_from_indexes(memory_id)
            del self._cache[memory_id]
            self._stats.expirations += 1
            self._emit_event(CacheEventType.EXPIRE, memory_id, {})

    def _invalidate_related(self, memory: Memory) -> None:
        """Invalidate cache entries related to this memory.

        When a memory is stored/updated, we may need to invalidate
        cached query results that included the old version.

        Args:
            memory: The memory being stored
        """
        # For preferences, invalidate user's preference cache
        if memory.memory_type == "preference":
            # Invalidate any cached preferences for this user
            self.invalidate_pattern(f"user:{memory.user_id}:preference")

        # For updates, the memory_id itself is handled in store()

    def _add_to_indexes(self, memory: Memory) -> None:
        """Add memory to secondary indexes.

        Args:
            memory: Memory to index
        """
        # User index
        if memory.user_id:
            if memory.user_id not in self._user_index:
                self._user_index[memory.user_id] = set()
            self._user_index[memory.user_id].add(memory.memory_id)

        # Agent index
        if memory.agent_id:
            if memory.agent_id not in self._agent_index:
                self._agent_index[memory.agent_id] = set()
            self._agent_index[memory.agent_id].add(memory.memory_id)

        # Topic index
        for topic in memory.topics:
            if topic not in self._topic_index:
                self._topic_index[topic] = set()
            self._topic_index[topic].add(memory.memory_id)

        # Type index
        if memory.memory_type:
            if memory.memory_type not in self._type_index:
                self._type_index[memory.memory_type] = set()
            self._type_index[memory.memory_type].add(memory.memory_id)

    def _remove_from_indexes(self, memory_id: str) -> None:
        """Remove memory from secondary indexes.

        Args:
            memory_id: Memory ID to remove
        """
        entry = self._cache.get(memory_id)
        if not entry:
            return

        memory = entry.memory

        # User index
        if memory.user_id and memory.user_id in self._user_index:
            self._user_index[memory.user_id].discard(memory_id)
            if not self._user_index[memory.user_id]:
                del self._user_index[memory.user_id]

        # Agent index
        if memory.agent_id and memory.agent_id in self._agent_index:
            self._agent_index[memory.agent_id].discard(memory_id)
            if not self._agent_index[memory.agent_id]:
                del self._agent_index[memory.agent_id]

        # Topic index
        for topic in memory.topics:
            if topic in self._topic_index:
                self._topic_index[topic].discard(memory_id)
                if not self._topic_index[topic]:
                    del self._topic_index[topic]

        # Type index
        if memory.memory_type and memory.memory_type in self._type_index:
            self._type_index[memory.memory_type].discard(memory_id)
            if not self._type_index[memory.memory_type]:
                del self._type_index[memory.memory_type]

    def _match_pattern(self, pattern: str) -> set[str]:
        """Match memory IDs against a pattern.

        Pattern format: {dimension}:{value}:{suffix}

        Examples:
        - user:123:* - All memories for user 123
        - user:123:preference - User 123's preferences
        - topic:orders:* - All memories with topic "orders"
        - *:*:* - All memories (use clear() instead)

        Args:
            pattern: Glob-style pattern

        Returns:
            Set of matching memory IDs
        """
        parts = pattern.split(":", 2)

        if len(parts) < 2:
            # Invalid pattern, return empty
            return set()

        dimension = parts[0]
        value = parts[1]
        suffix = parts[2] if len(parts) > 2 else "*"

        # Get candidates from index
        candidates: set[str] = set()

        if dimension in {"user", "*"}:
            if value == "*":
                for ids in self._user_index.values():
                    candidates.update(ids)
            elif value in self._user_index:
                candidates.update(self._user_index[value])

        if dimension in {"agent", "*"}:
            if value == "*":
                for ids in self._agent_index.values():
                    candidates.update(ids)
            elif value in self._agent_index:
                candidates.update(self._agent_index[value])

        if dimension in {"topic", "*"}:
            if value == "*":
                for ids in self._topic_index.values():
                    candidates.update(ids)
            elif value in self._topic_index:
                candidates.update(self._topic_index[value])

        if dimension in {"type", "*"}:
            if value == "*":
                for ids in self._type_index.values():
                    candidates.update(ids)
            elif value in self._type_index:
                candidates.update(self._type_index[value])

        # Filter by suffix if not wildcard
        if suffix != "*":
            filtered = set()
            for memory_id in candidates:
                entry = self._cache.get(memory_id)
                if entry:
                    memory = entry.memory
                    # Match suffix against memory_type
                    if fnmatch.fnmatch(memory.memory_type, suffix):
                        filtered.add(memory_id)
            candidates = filtered

        return candidates

    def _emit_event(
        self,
        event_type: CacheEventType,
        key: str,
        data: dict[str, Any],
    ) -> None:
        """Emit a cache event to the callback.

        Args:
            event_type: Type of event
            key: Memory ID or pattern
            data: Additional event data
        """
        if self.on_event:
            try:
                self.on_event(event_type, key, data)
            except Exception:
                # Don't let callback errors break cache operations
                pass

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, memory_id: str) -> bool:
        """Check if memory is in cache."""
        entry = self._cache.get(memory_id)
        return bool(entry and entry.age_seconds <= self.ttl_seconds)
