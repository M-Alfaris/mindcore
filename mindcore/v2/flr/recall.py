"""FLR - Fast Learning Recall.

A protocol for rapid retrieval, inference-time memory access, and short-term
contextual recall among AI agents or between agent cores.

FLR handles:
- Short-term memory (active context)
- Fast retrieval from long-term storage (CLST)
- Attention routing and scoring
- Reinforcement signals
- Cross-agent attention routing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from collections import OrderedDict
import hashlib

if TYPE_CHECKING:
    from ..storage.base import BaseStorage


@dataclass
class Memory:
    """A memory unit in the system."""

    memory_id: str
    content: str
    memory_type: str
    user_id: str
    agent_id: str | None = None

    # Metadata
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    importance: float = 0.5
    entities: list[str] = field(default_factory=list)
    access_level: str = "private"

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime | None = None
    expires_at: datetime | None = None

    # FLR-specific
    reinforcement_score: float = 0.0  # Accumulated reinforcement signals
    access_count: int = 0
    embedding: list[float] | None = None

    # Versioning
    vocabulary_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "topics": self.topics,
            "categories": self.categories,
            "sentiment": self.sentiment,
            "importance": self.importance,
            "entities": self.entities,
            "access_level": self.access_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reinforcement_score": self.reinforcement_score,
            "access_count": self.access_count,
            "vocabulary_version": self.vocabulary_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create from dictionary."""
        # Parse datetime fields
        for dt_field in ["created_at", "last_accessed", "expires_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        # Remove embedding if present but None
        embedding = data.pop("embedding", None)

        memory = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        memory.embedding = embedding
        return memory


@dataclass
class RecallResult:
    """Result from FLR query."""

    memories: list[Memory]
    scores: list[float]  # Relevance scores for each memory
    query_latency_ms: float
    sources: list[str]  # Where memories came from: "cache", "storage", "cross_agent"

    # Attention hints for the agent
    attention_focus: list[str]  # Top topics to focus on
    suggested_memory_types: list[str]  # Relevant memory types

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "scores": self.scores,
            "query_latency_ms": self.query_latency_ms,
            "sources": self.sources,
            "attention_focus": self.attention_focus,
            "suggested_memory_types": self.suggested_memory_types,
        }


@dataclass
class ContextWindow:
    """Active context window for inference-time updates."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    working_memories: list[Memory] = field(default_factory=list)
    attention_hints: list[str] = field(default_factory=list)
    session_id: str | None = None
    max_messages: int = 50

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        """Add a message to the context window."""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Trim if over limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self):
        """Clear the context window."""
        self.messages = []
        self.working_memories = []
        self.attention_hints = []


class FLR:
    """Fast Learning Recall - Hot path memory access.

    Handles rapid retrieval, scoring, and attention routing for AI agents.

    Example:
        flr = FLR(storage=storage)

        # Query for relevant memories
        result = flr.query(
            query="What's my order status?",
            user_id="user123",
            agent_id="support_bot",
            attention_hints=["orders", "shipping"],
        )

        # Reinforce useful memories
        flr.reinforce(memory_id, signal=+1.0)

        # Update active context
        flr.update_context(messages=[...])
    """

    def __init__(
        self,
        storage: BaseStorage,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 300,
        embedding_fn: callable | None = None,
    ):
        """Initialize FLR.

        Args:
            storage: Storage backend (connects to CLST)
            cache_size: Max memories in hot cache
            cache_ttl_seconds: Cache TTL
            embedding_fn: Optional function to generate embeddings
        """
        self.storage = storage
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl_seconds
        self.embedding_fn = embedding_fn

        # Hot cache (LRU)
        self._cache: OrderedDict[str, tuple[Memory, float]] = OrderedDict()

        # Active context windows by session
        self._contexts: dict[str, ContextWindow] = {}

        # Reinforcement scores (in-memory, periodically flushed to storage)
        self._reinforcement_buffer: dict[str, float] = {}

    def query(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        include_cross_agent: bool = False,
        min_score: float = 0.0,
    ) -> RecallResult:
        """Query for relevant memories.

        Fast retrieval with scoring based on:
        - Semantic similarity (if embeddings available)
        - Topic/category match
        - Recency
        - Reinforcement score
        - Importance

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier (for access control)
            attention_hints: Topics/categories to prioritize
            memory_types: Filter by memory types
            limit: Max memories to return
            include_cross_agent: Include memories from other agents
            min_score: Minimum relevance score

        Returns:
            RecallResult with scored memories
        """
        start_time = time.time()
        attention_hints = attention_hints or []
        memory_types = memory_types or []
        sources = []

        # 1. Check hot cache first
        cached_memories = self._query_cache(
            query, user_id, agent_id, attention_hints, memory_types
        )
        if cached_memories:
            sources.append("cache")

        # 2. Query storage (CLST)
        storage_memories = self._query_storage(
            query, user_id, agent_id, attention_hints, memory_types,
            limit=limit * 2,  # Get more for scoring
            include_cross_agent=include_cross_agent,
        )
        if storage_memories:
            sources.append("storage")

        # 3. Combine and deduplicate
        all_memories = self._deduplicate(cached_memories + storage_memories)

        # 4. Score memories
        scored = self._score_memories(all_memories, query, attention_hints)

        # 5. Filter by min_score and limit
        filtered = [(m, s) for m, s in scored if s >= min_score]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:limit]

        # 6. Update cache with accessed memories
        for memory, score in filtered:
            self._cache_memory(memory)
            memory.last_accessed = datetime.now(timezone.utc)
            memory.access_count += 1

        # 7. Extract attention focus
        attention_focus = self._extract_attention_focus(
            [m for m, _ in filtered], attention_hints
        )

        # 8. Suggest memory types
        suggested_types = self._suggest_memory_types(query, [m for m, _ in filtered])

        latency = (time.time() - start_time) * 1000

        return RecallResult(
            memories=[m for m, _ in filtered],
            scores=[s for _, s in filtered],
            query_latency_ms=latency,
            sources=sources,
            attention_focus=attention_focus,
            suggested_memory_types=suggested_types,
        )

    def reinforce(self, memory_id: str, signal: float) -> None:
        """Reinforce a memory with a learning signal.

        Positive signals increase future recall probability.
        Negative signals decrease it.

        Args:
            memory_id: Memory to reinforce
            signal: Reinforcement signal (-1.0 to +1.0)
        """
        signal = max(-1.0, min(1.0, signal))  # Clamp to [-1, 1]

        # Buffer reinforcement (batched writes to storage)
        if memory_id in self._reinforcement_buffer:
            self._reinforcement_buffer[memory_id] += signal
        else:
            self._reinforcement_buffer[memory_id] = signal

        # Update cache if present
        if memory_id in self._cache:
            memory, timestamp = self._cache[memory_id]
            memory.reinforcement_score += signal
            self._cache[memory_id] = (memory, timestamp)

    def promote(self, memory_id: str) -> bool:
        """Promote a working memory to long-term storage.

        Args:
            memory_id: Memory to promote

        Returns:
            True if promoted successfully
        """
        # Find in working memories across all contexts
        for context in self._contexts.values():
            for memory in context.working_memories:
                if memory.memory_id == memory_id:
                    # Change type from working to appropriate long-term type
                    if memory.memory_type == "working":
                        memory.memory_type = "episodic"  # Default promotion type
                    # Store in CLST
                    self.storage.store(memory)
                    context.working_memories.remove(memory)
                    return True
        return False

    def update_context(
        self,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        working_memories: list[Memory] | None = None,
        attention_hints: list[str] | None = None,
    ) -> ContextWindow:
        """Update active context window for a session.

        Args:
            session_id: Session identifier
            messages: New messages to add
            working_memories: Working memories to add
            attention_hints: Attention hints to set

        Returns:
            Updated ContextWindow
        """
        if session_id not in self._contexts:
            self._contexts[session_id] = ContextWindow(session_id=session_id)

        context = self._contexts[session_id]

        if messages:
            for msg in messages:
                context.add_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    metadata=msg.get("metadata"),
                )

        if working_memories:
            context.working_memories.extend(working_memories)

        if attention_hints:
            context.attention_hints = attention_hints

        return context

    def get_context(self, session_id: str) -> ContextWindow | None:
        """Get active context window for a session."""
        return self._contexts.get(session_id)

    def clear_context(self, session_id: str) -> None:
        """Clear context window for a session."""
        if session_id in self._contexts:
            self._contexts[session_id].clear()

    def flush_reinforcements(self) -> int:
        """Flush buffered reinforcement signals to storage.

        Returns:
            Number of memories updated
        """
        if not self._reinforcement_buffer:
            return 0

        count = 0
        for memory_id, signal in self._reinforcement_buffer.items():
            try:
                self.storage.update_reinforcement(memory_id, signal)
                count += 1
            except Exception:
                pass  # Log error in production

        self._reinforcement_buffer.clear()
        return count

    def _query_cache(
        self,
        query: str,
        user_id: str,
        agent_id: str | None,
        attention_hints: list[str],
        memory_types: list[str],
    ) -> list[Memory]:
        """Query hot cache."""
        now = time.time()
        results = []

        # Clean expired entries
        expired = [
            mid for mid, (_, ts) in self._cache.items()
            if now - ts > self.cache_ttl
        ]
        for mid in expired:
            del self._cache[mid]

        # Search cache
        for memory_id, (memory, _) in self._cache.items():
            # Access control
            if memory.user_id != user_id:
                if memory.access_level == "private":
                    continue
                if memory.access_level == "team" and memory.agent_id != agent_id:
                    continue  # TODO: Check team membership

            # Type filter
            if memory_types and memory.memory_type not in memory_types:
                continue

            # Basic relevance check
            if self._is_relevant(memory, query, attention_hints):
                results.append(memory)

        return results

    def _query_storage(
        self,
        query: str,
        user_id: str,
        agent_id: str | None,
        attention_hints: list[str],
        memory_types: list[str],
        limit: int,
        include_cross_agent: bool,
    ) -> list[Memory]:
        """Query storage backend (CLST)."""
        try:
            return self.storage.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id if not include_cross_agent else None,
                topics=attention_hints,
                memory_types=memory_types,
                limit=limit,
            )
        except Exception:
            return []

    def _score_memories(
        self,
        memories: list[Memory],
        query: str,
        attention_hints: list[str],
    ) -> list[tuple[Memory, float]]:
        """Score memories by relevance."""
        scored = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for memory in memories:
            score = 0.0

            # 1. Content similarity (simple word overlap)
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)
            score += overlap * 0.1  # Up to ~0.5 for good overlap

            # 2. Topic match with attention hints
            if attention_hints:
                topic_matches = len(set(memory.topics) & set(attention_hints))
                score += topic_matches * 0.2  # Strong boost for topic match

            # 3. Recency (decay over time)
            if memory.created_at:
                age_hours = (datetime.now(timezone.utc) - memory.created_at).total_seconds() / 3600
                recency_score = max(0, 1 - (age_hours / 168))  # Decay over 1 week
                score += recency_score * 0.15

            # 4. Reinforcement score
            score += memory.reinforcement_score * 0.2

            # 5. Importance
            score += memory.importance * 0.15

            # 6. Access count (popularity)
            popularity = min(1.0, memory.access_count / 100)
            score += popularity * 0.1

            # Normalize to 0-1
            score = min(1.0, max(0.0, score))

            scored.append((memory, score))

        return scored

    def _is_relevant(
        self,
        memory: Memory,
        query: str,
        attention_hints: list[str],
    ) -> bool:
        """Quick relevance check for cache filtering."""
        query_lower = query.lower()

        # Check topic match
        if attention_hints and set(memory.topics) & set(attention_hints):
            return True

        # Check content contains query words
        for word in query_lower.split():
            if len(word) > 3 and word in memory.content.lower():
                return True

        # Check entities
        for entity in memory.entities:
            if entity.lower() in query_lower:
                return True

        return False

    def _deduplicate(self, memories: list[Memory]) -> list[Memory]:
        """Remove duplicate memories."""
        seen = set()
        unique = []
        for memory in memories:
            if memory.memory_id not in seen:
                seen.add(memory.memory_id)
                unique.append(memory)
        return unique

    def _cache_memory(self, memory: Memory) -> None:
        """Add memory to hot cache."""
        # LRU eviction
        while len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)

        self._cache[memory.memory_id] = (memory, time.time())
        self._cache.move_to_end(memory.memory_id)

    def _extract_attention_focus(
        self,
        memories: list[Memory],
        hints: list[str],
    ) -> list[str]:
        """Extract top topics to focus on."""
        topic_counts: dict[str, int] = {}

        for memory in memories:
            for topic in memory.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Boost hinted topics
        for hint in hints:
            if hint in topic_counts:
                topic_counts[hint] *= 2

        # Sort by count and return top 5
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_topics[:5]]

    def _suggest_memory_types(
        self,
        query: str,
        memories: list[Memory],
    ) -> list[str]:
        """Suggest relevant memory types based on query."""
        query_lower = query.lower()
        suggestions = set()

        # Keyword-based suggestions
        if any(w in query_lower for w in ["how", "steps", "process", "workflow"]):
            suggestions.add("procedural")
        if any(w in query_lower for w in ["prefer", "like", "want", "setting"]):
            suggestions.add("preference")
        if any(w in query_lower for w in ["who", "where", "what is"]):
            suggestions.add("entity")
        if any(w in query_lower for w in ["last time", "before", "remember when"]):
            suggestions.add("episodic")
        if any(w in query_lower for w in ["fact", "know", "information"]):
            suggestions.add("semantic")

        # Add types from retrieved memories
        for memory in memories[:5]:
            suggestions.add(memory.memory_type)

        return list(suggestions)[:5]

    def get_stats(self) -> dict[str, Any]:
        """Get FLR statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_max": self.cache_size,
            "active_contexts": len(self._contexts),
            "pending_reinforcements": len(self._reinforcement_buffer),
        }
