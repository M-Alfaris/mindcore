"""FLR - Fast Learning Recall.

A protocol for rapid retrieval, inference-time memory access, and short-term
contextual recall among AI agents or between agent cores.

FLR handles:
- Short-term memory (active context)
- Fast retrieval from long-term storage (CLST)
- Attention routing and scoring
- Reinforcement signals (naive and robust modes)
- Cross-agent attention routing

Reinforcement Modes:
- Legacy (naive): Simple bounded accumulation with diminishing returns
- Robust: Temporal decay, multi-signal types, exploration bonus, trend tracking

Example (robust reinforcement):
    from mindcore.v2.flr import FLR
    from mindcore.v2.flr.reinforcement import SignalType, SignalSource

    flr = FLR(storage=storage, use_robust_reinforcement=True)

    # Apply detailed signal
    flr.reinforce_robust(
        memory_id="mem_123",
        signal_value=0.8,
        signal_type=SignalType.RELEVANCE,
        source=SignalSource.USER_EXPLICIT,
        context_similarity=0.9,
    )
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .reinforcement import (
    RobustReinforcement,
    ReinforcementSignal,
    SignalType,
    SignalSource,
    create_feedback_signal,
)

from .metadata_feedback import (
    MetadataFeedbackTracker,
    MetadataSignal,
)


if TYPE_CHECKING:
    from mindcore.v2.cross_agent.registry import AgentRegistry
    from mindcore.v2.storage.base import BaseStorage


# Constants for reinforcement score bounds
REINFORCEMENT_SCORE_MIN = -1.0
REINFORCEMENT_SCORE_MAX = 1.0


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

    # Session/Thread context (for hierarchical retrieval)
    session_id: str | None = None
    thread_id: str | None = None  # For multi-thread conversations within a session
    message_index: int = 0  # Order within session/thread for event series

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime | None = None
    expires_at: datetime | None = None

    # FLR-specific (legacy naive reinforcement)
    reinforcement_score: float = 0.0  # Accumulated reinforcement signals (bounded to [-1, 1])
    access_count: int = 0
    embedding: list[float] | None = None

    # Robust reinforcement (optional, stores full signal history)
    robust_reinforcement: RobustReinforcement | None = None

    # Versioning
    vocabulary_version: str = "1.0.0"

    def apply_reinforcement(self, signal: float, use_robust: bool = False) -> float:
        """Apply a reinforcement signal with bounds checking.

        Args:
            signal: Reinforcement signal to apply (will be clamped to [-1, 1])
            use_robust: If True, use robust reinforcement with temporal decay

        Returns:
            The new reinforcement score after applying the signal
        """
        if use_robust:
            return self.apply_robust_reinforcement(signal)

        # Legacy naive implementation
        # Clamp signal to valid range
        clamped_signal = max(-1.0, min(1.0, signal))

        # Apply signal with exponential decay toward bounds
        # This prevents score from getting stuck at bounds
        if clamped_signal > 0:
            # Positive signal: diminishing returns as we approach max
            headroom = REINFORCEMENT_SCORE_MAX - self.reinforcement_score
            effective_signal = clamped_signal * (headroom / 2.0)  # Scale by available headroom
        else:
            # Negative signal: diminishing returns as we approach min
            headroom = self.reinforcement_score - REINFORCEMENT_SCORE_MIN
            effective_signal = clamped_signal * (headroom / 2.0)

        self.reinforcement_score += effective_signal

        # Ensure bounds are respected (safety clamp)
        self.reinforcement_score = max(
            REINFORCEMENT_SCORE_MIN, min(REINFORCEMENT_SCORE_MAX, self.reinforcement_score)
        )

        return self.reinforcement_score

    def apply_robust_reinforcement(
        self,
        signal_value: float,
        signal_type: SignalType = SignalType.RELEVANCE,
        source: SignalSource = SignalSource.LLM_EVALUATION,
        context_similarity: float = 1.0,
        query_id: str | None = None,
    ) -> float:
        """Apply a robust reinforcement signal with full tracking.

        Args:
            signal_value: Signal value (-1 to 1)
            signal_type: Type of reinforcement signal
            source: Source of the signal
            context_similarity: How similar the retrieval context was
            query_id: Associated query ID

        Returns:
            The new aggregated reinforcement score
        """
        # Initialize robust reinforcement if not present
        if self.robust_reinforcement is None:
            self.robust_reinforcement = RobustReinforcement()
            # Migrate legacy score as initial signal if non-zero
            if self.reinforcement_score != 0.0:
                self.robust_reinforcement.apply_simple_signal(
                    value=self.reinforcement_score,
                    signal_type=SignalType.RELEVANCE,
                    source=SignalSource.AUTOMATED_METRIC,
                )

        signal = ReinforcementSignal(
            signal_type=signal_type,
            value=signal_value,
            source=source,
            context_similarity=context_similarity,
            query_id=query_id,
            session_id=self.session_id,
        )

        new_score = self.robust_reinforcement.apply_signal(signal)

        # Keep legacy score in sync for backward compatibility
        self.reinforcement_score = new_score

        return new_score

    def get_effective_reinforcement_score(
        self,
        use_robust: bool = False,
        exploration_factor: float = 0.1,
        total_retrievals: int = 1000,
    ) -> float:
        """Get effective reinforcement score for ranking.

        Args:
            use_robust: Use robust reinforcement with exploration bonus
            exploration_factor: Weight for exploration (0-1)
            total_retrievals: Total retrievals for UCB calculation

        Returns:
            Effective score for ranking
        """
        if use_robust and self.robust_reinforcement is not None:
            return self.robust_reinforcement.get_effective_score(
                exploration_factor=exploration_factor,
                total_retrievals=total_retrievals,
            )
        return self.reinforcement_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
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
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "message_index": self.message_index,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reinforcement_score": self.reinforcement_score,
            "access_count": self.access_count,
            "vocabulary_version": self.vocabulary_version,
        }

        # Include robust reinforcement if present
        if self.robust_reinforcement is not None:
            data["robust_reinforcement"] = self.robust_reinforcement.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create from dictionary."""
        # Make a copy to avoid modifying the original
        data = dict(data)

        # Parse datetime fields
        for dt_field in ["created_at", "last_accessed", "expires_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        # Remove embedding if present but None
        embedding = data.pop("embedding", None)

        # Extract robust reinforcement data
        robust_data = data.pop("robust_reinforcement", None)

        memory = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        memory.embedding = embedding

        # Restore robust reinforcement if present
        if robust_data is not None:
            memory.robust_reinforcement = RobustReinforcement.from_dict(robust_data)

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

    # Query context for feedback (used by metadata effectiveness tracking)
    query_topics: list[str] = field(default_factory=list)
    query_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "scores": self.scores,
            "query_latency_ms": self.query_latency_ms,
            "sources": self.sources,
            "attention_focus": self.attention_focus,
            "suggested_memory_types": self.suggested_memory_types,
            "query_topics": self.query_topics,
            "query_categories": self.query_categories,
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
        self.messages.append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Trim if over limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

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
        agent_registry: AgentRegistry | None = None,
        use_robust_reinforcement: bool = False,
        exploration_factor: float = 0.1,
        decay_half_life_hours: float = 168.0,
    ):
        """Initialize FLR.

        Args:
            storage: Storage backend (connects to CLST)
            cache_size: Max memories in hot cache
            cache_ttl_seconds: Cache TTL
            embedding_fn: Optional function to generate embeddings
            agent_registry: Optional agent registry for team-based access control
            use_robust_reinforcement: Use robust reinforcement with temporal decay
            exploration_factor: Weight for exploration bonus (0-1, only for robust mode)
            decay_half_life_hours: Reinforcement half-life in hours (only for robust mode)
        """
        self.storage = storage
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl_seconds
        self.embedding_fn = embedding_fn
        self.agent_registry = agent_registry

        # Reinforcement configuration
        self.use_robust_reinforcement = use_robust_reinforcement
        self.exploration_factor = exploration_factor
        self.decay_half_life_hours = decay_half_life_hours

        # Hot cache (LRU)
        self._cache: OrderedDict[str, tuple[Memory, float]] = OrderedDict()

        # Active context windows by session
        self._contexts: dict[str, ContextWindow] = {}

        # Reinforcement scores (in-memory, periodically flushed to storage)
        self._reinforcement_buffer: dict[str, float] = {}

        # Robust reinforcement buffer (stores full signals)
        self._robust_reinforcement_buffer: dict[str, list[ReinforcementSignal]] = {}

        # Total retrieval count for UCB exploration calculation
        self._total_retrievals: int = 0

        # Metadata effectiveness tracking (for improving LLM assignments)
        self._metadata_tracker = MetadataFeedbackTracker()

        # Last query context (for feedback correlation)
        self._last_query_context: dict[str, list[str]] = {}

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

        # When agent_registry is configured, search cross-agent for team/shared access
        # We'll filter by access control after retrieval
        search_cross_agent = include_cross_agent or (self.agent_registry is not None)

        # 1. Check hot cache first
        cached_memories = self._query_cache(query, user_id, agent_id, attention_hints, memory_types)
        if cached_memories:
            sources.append("cache")

        # 2. Query storage (CLST)
        storage_memories = self._query_storage(
            query,
            user_id,
            agent_id,
            attention_hints,
            memory_types,
            limit=limit * 2,  # Get more for scoring
            include_cross_agent=search_cross_agent,
        )
        if storage_memories:
            sources.append("storage")

        # 3. Combine and deduplicate
        all_memories = self._deduplicate(cached_memories + storage_memories)

        # 4. Filter by access control when agent_registry is configured
        if self.agent_registry and agent_id:
            all_memories = self._filter_by_access(all_memories, agent_id)

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
            # Track access for robust reinforcement
            if self.use_robust_reinforcement and memory.robust_reinforcement:
                memory.robust_reinforcement.record_access()

        # Track total retrievals for UCB exploration calculation
        self._total_retrievals += len(filtered)

        # 7. Extract attention focus
        attention_focus = self._extract_attention_focus([m for m, _ in filtered], attention_hints)

        # 8. Suggest memory types
        suggested_types = self._suggest_memory_types(query, [m for m, _ in filtered])

        latency = (time.time() - start_time) * 1000

        # Store query context for metadata feedback correlation
        self._last_query_context = {
            "topics": attention_hints or [],
            "categories": [],  # Can be extended if categories are used in queries
            "memory_ids": [m.memory_id for m, _ in filtered],
        }

        return RecallResult(
            memories=[m for m, _ in filtered],
            scores=[s for _, s in filtered],
            query_latency_ms=latency,
            sources=sources,
            attention_focus=attention_focus,
            suggested_memory_types=suggested_types,
            query_topics=attention_hints or [],
            query_categories=[],
        )

    def reinforce(self, memory_id: str, signal: float) -> float:
        """Reinforce a memory with a learning signal.

        Positive signals increase future recall probability.
        Negative signals decrease it. Uses bounded reinforcement with
        diminishing returns as scores approach limits.

        Args:
            memory_id: Memory to reinforce
            signal: Reinforcement signal (-1.0 to +1.0)

        Returns:
            The new reinforcement score

        Raises:
            ValueError: If signal is not a valid number
        """
        if not isinstance(signal, int | float):
            raise TypeError(f"Signal must be a number, got {type(signal).__name__}")

        # Clamp signal to valid range
        clamped_signal = max(-1.0, min(1.0, float(signal)))

        # Buffer reinforcement (batched writes to storage)
        if memory_id in self._reinforcement_buffer:
            # Buffer stores raw signals, bounds applied on flush
            self._reinforcement_buffer[memory_id] += clamped_signal
        else:
            self._reinforcement_buffer[memory_id] = clamped_signal

        # Clamp buffer to prevent unbounded accumulation
        self._reinforcement_buffer[memory_id] = max(
            -1.0, min(1.0, self._reinforcement_buffer[memory_id])
        )

        # Update cache if present using bounded method
        new_score = 0.0
        if memory_id in self._cache:
            memory, timestamp = self._cache[memory_id]
            new_score = memory.apply_reinforcement(clamped_signal)
            self._cache[memory_id] = (memory, timestamp)

        # Persist to storage immediately for consistency
        try:
            self.storage.update_reinforcement(memory_id, clamped_signal)
            # Clear from buffer since we persisted
            self._reinforcement_buffer.pop(memory_id, None)
            # If we got the score from storage, fetch it
            if new_score == 0.0:
                memory = self.storage.get(memory_id)
                if memory:
                    new_score = memory.reinforcement_score
        except Exception:
            # Keep in buffer for later flush if storage fails
            pass

        return new_score

    def reinforce_robust(
        self,
        memory_id: str,
        signal_value: float,
        signal_type: SignalType = SignalType.RELEVANCE,
        source: SignalSource = SignalSource.LLM_EVALUATION,
        context_similarity: float = 1.0,
        query_id: str | None = None,
        session_id: str | None = None,
    ) -> float:
        """Apply robust reinforcement signal with full tracking.

        This method provides detailed reinforcement with:
        - Signal type classification (relevance, usefulness, correctness, etc.)
        - Source weighting (user explicit, LLM evaluation, automated, etc.)
        - Context similarity weighting
        - Temporal decay (automatic via RobustReinforcement)

        Args:
            memory_id: Memory to reinforce
            signal_value: Signal value (-1.0 to +1.0)
            signal_type: Type of reinforcement signal
            source: Source of the signal
            context_similarity: How similar the retrieval context was (0-1)
            query_id: Associated query ID (for tracking)
            session_id: Associated session ID (for tracking)

        Returns:
            The new aggregated reinforcement score

        Example:
            # User gave explicit positive feedback
            flr.reinforce_robust(
                memory_id="mem_123",
                signal_value=0.9,
                signal_type=SignalType.USEFULNESS,
                source=SignalSource.USER_EXPLICIT,
                context_similarity=0.85,
            )
        """
        # Create the signal
        signal = ReinforcementSignal(
            signal_type=signal_type,
            value=max(-1.0, min(1.0, signal_value)),
            source=source,
            context_similarity=context_similarity,
            query_id=query_id,
            session_id=session_id,
        )

        # Buffer for batch persistence
        if memory_id not in self._robust_reinforcement_buffer:
            self._robust_reinforcement_buffer[memory_id] = []
        self._robust_reinforcement_buffer[memory_id].append(signal)

        # Update cache if present
        new_score = 0.0
        if memory_id in self._cache:
            memory, timestamp = self._cache[memory_id]

            # Initialize robust reinforcement if needed
            if memory.robust_reinforcement is None:
                memory.robust_reinforcement = RobustReinforcement(
                    decay_half_life_hours=self.decay_half_life_hours
                )

            new_score = memory.robust_reinforcement.apply_signal(signal)
            memory.reinforcement_score = new_score  # Keep legacy in sync
            self._cache[memory_id] = (memory, timestamp)

        # Also update via legacy method for storage persistence
        try:
            self.storage.update_reinforcement(memory_id, signal_value)
            # Clear buffer on success
            self._robust_reinforcement_buffer.pop(memory_id, None)

            if new_score == 0.0:
                memory = self.storage.get(memory_id)
                if memory:
                    new_score = memory.reinforcement_score
        except Exception:
            # Keep in buffer for later flush
            pass

        return new_score

    def reinforce_from_feedback(
        self,
        memory_id: str,
        feedback_value: float,
        is_user_feedback: bool = False,
        context_similarity: float = 1.0,
        query_id: str | None = None,
        session_id: str | None = None,
    ) -> float:
        """Convenience method to reinforce from simple feedback.

        Automatically determines signal type and source based on the feedback.

        Args:
            memory_id: Memory to reinforce
            feedback_value: Feedback value (-1 to 1)
            is_user_feedback: Whether this is direct user feedback
            context_similarity: How similar the retrieval context was
            query_id: Associated query ID
            session_id: Associated session ID

        Returns:
            New reinforcement score
        """
        if not self.use_robust_reinforcement:
            # Fall back to simple reinforcement
            return self.reinforce(memory_id, feedback_value)

        signal = create_feedback_signal(
            value=feedback_value,
            is_user_feedback=is_user_feedback,
            context_similarity=context_similarity,
            query_id=query_id,
            session_id=session_id,
        )

        return self.reinforce_robust(
            memory_id=memory_id,
            signal_value=signal.value,
            signal_type=signal.signal_type,
            source=signal.source,
            context_similarity=signal.context_similarity,
            query_id=query_id,
            session_id=session_id,
        )

    def reinforce_with_metadata_feedback(
        self,
        memory_id: str,
        signal: float,
        is_user_feedback: bool = False,
        session_id: str | None = None,
    ) -> tuple[float, MetadataSignal | None]:
        """Reinforce memory AND track metadata effectiveness.

        This is the recommended method for providing feedback. It:
        1. Updates the memory's reinforcement score (propagates to CLST)
        2. Tracks which metadata assignments led to this outcome
        3. Enables future improvement of LLM metadata assignments

        Args:
            memory_id: Memory to reinforce
            signal: Feedback signal (-1 to +1)
            is_user_feedback: True if this is explicit user feedback
            session_id: Current session ID

        Returns:
            Tuple of (new_score, metadata_signal or None)

        Example:
            # User found the retrieved memory helpful
            score, meta = flr.reinforce_with_metadata_feedback(
                memory_id="mem_123",
                signal=0.9,
                is_user_feedback=True,
            )
            # This tells us: the LLM's topic/category assignments were good
        """
        # Get the memory to access its metadata
        memory = None
        if memory_id in self._cache:
            memory, _ = self._cache[memory_id]
        else:
            try:
                memory = self.storage.get(memory_id)
            except Exception:
                pass

        # Apply reinforcement (propagates to CLST storage)
        new_score = self.reinforce(memory_id, signal)

        # Track metadata effectiveness if we have the memory
        metadata_signal = None
        if memory:
            query_topics = self._last_query_context.get("topics", [])
            query_categories = self._last_query_context.get("categories", [])

            metadata_signal = self._metadata_tracker.record_retrieval_feedback(
                memory_id=memory_id,
                assigned_topics=memory.topics,
                assigned_categories=memory.categories,
                query_topics=query_topics,
                query_categories=query_categories,
                signal=signal,
                assigned_intent=None,  # Could extract from memory if stored
                assigned_type=memory.memory_type,
                session_id=session_id,
            )

        return new_score, metadata_signal

    def get_metadata_effectiveness_report(self) -> dict:
        """Get report on metadata assignment effectiveness.

        Use this to understand which LLM-assigned metadata values
        lead to successful retrievals.

        Returns:
            Report with effectiveness scores by metadata type

        Example:
            report = flr.get_metadata_effectiveness_report()
            # {
            #     "topics": {"refund": {"effectiveness_score": 0.85}, ...},
            #     "categories": {...},
            #     "summary": {"total_signals": 150, ...}
            # }
        """
        return self._metadata_tracker.get_effectiveness_report()

    def get_metadata_feedback_for_extractor(self) -> dict:
        """Get structured feedback for improving MetadataExtractor.

        This returns data that can be injected into the LLM prompt
        to improve future metadata assignments.

        Returns:
            Feedback structure for prompt injection

        Example:
            feedback = flr.get_metadata_feedback_for_extractor()
            # Use in MetadataExtractor prompt:
            # "High-quality topics: 'refund', 'billing'"
            # "Low-quality topics: 'general', 'misc'"
        """
        return self._metadata_tracker.get_feedback_for_extractor()

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

        # Also check storage for working memories not in a context
        try:
            memory = self.storage.get(memory_id)
            if memory and memory.memory_type == "working":
                # Promote by updating the memory type
                memory.memory_type = "episodic"
                self.storage.update(memory)
                return True
        except Exception:
            pass

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
        """Clear and remove context window for a session."""
        if session_id in self._contexts:
            self._contexts[session_id].clear()
            del self._contexts[session_id]

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
        expired = [mid for mid, (_, ts) in self._cache.items() if now - ts > self.cache_ttl]
        for mid in expired:
            del self._cache[mid]

        # Search cache
        for memory, _ in self._cache.values():
            # Access control
            # Note: For full team-based access control, use CrossAgentLayer which
            # has proper team registration and membership checking. FLR only does
            # basic agent-level filtering.
            if memory.user_id != user_id:
                if memory.access_level == "private":
                    continue
                if memory.access_level == "team" and memory.agent_id != agent_id:
                    # Check team membership via registry
                    if not self._check_team_access(agent_id, memory.agent_id):
                        continue

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
        """Score memories by relevance.

        Uses either legacy or robust reinforcement scoring based on configuration.
        Robust mode includes:
        - Temporal decay of reinforcement
        - UCB-like exploration bonus
        - Multi-signal type weighting
        """
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
                # Ensure created_at is timezone-aware for comparison
                created_at = memory.created_at
                if created_at.tzinfo is None:
                    # Assume UTC if no timezone info
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                recency_score = max(0, 1 - (age_hours / 168))  # Decay over 1 week
                score += recency_score * 0.15

            # 4. Reinforcement score (robust or legacy)
            if self.use_robust_reinforcement:
                # Use effective score with exploration bonus
                reinforcement_score = memory.get_effective_reinforcement_score(
                    use_robust=True,
                    exploration_factor=self.exploration_factor,
                    total_retrievals=max(1, self._total_retrievals),
                )
            else:
                # Legacy simple score
                reinforcement_score = memory.reinforcement_score
            score += reinforcement_score * 0.2

            # 5. Importance
            score += memory.importance * 0.15

            # 6. Access count (popularity) - reduced weight in robust mode
            # as exploration bonus already accounts for this
            popularity = min(1.0, memory.access_count / 100)
            if self.use_robust_reinforcement:
                score += popularity * 0.05  # Lower weight, exploration handles it
            else:
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
        return any(entity.lower() in query_lower for entity in memory.entities)

    def _check_team_access(
        self,
        requesting_agent_id: str | None,
        memory_agent_id: str | None,
    ) -> bool:
        """Check if requesting agent shares a team with memory owner.

        Args:
            requesting_agent_id: Agent requesting access
            memory_agent_id: Agent that owns the memory

        Returns:
            True if agents share at least one team, False otherwise
        """
        if not requesting_agent_id or not memory_agent_id:
            return False

        if not self.agent_registry:
            # No registry configured - deny team access for safety
            return False

        requester = self.agent_registry.get_agent(requesting_agent_id)
        owner = self.agent_registry.get_agent(memory_agent_id)

        if not requester or not owner:
            return False

        # Use the Agent.shares_team_with method from registry
        return requester.shares_team_with(owner)

    def _filter_by_access(
        self,
        memories: list[Memory],
        requesting_agent_id: str,
    ) -> list[Memory]:
        """Filter memories by access control.

        Access levels:
        - private: Only the owning agent can access
        - team: Agents in the same team can access
        - shared: All agents can access
        - global: Public access (no filtering)

        Args:
            memories: List of memories to filter
            requesting_agent_id: Agent requesting access

        Returns:
            Filtered list of accessible memories
        """
        accessible = []
        for memory in memories:
            access_level = getattr(memory, "access_level", "private")

            if access_level in ("global", "shared"):
                # Global/shared access - everyone can see
                accessible.append(memory)
            elif access_level == "team":
                # Team access - check if agents share a team
                if memory.agent_id == requesting_agent_id or self._check_team_access(
                    requesting_agent_id, memory.agent_id
                ):
                    accessible.append(memory)
            # Private access - only owning agent
            elif memory.agent_id == requesting_agent_id or memory.agent_id is None:
                accessible.append(memory)

        return accessible

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
        stats = {
            "cache_size": len(self._cache),
            "cache_max": self.cache_size,
            "active_contexts": len(self._contexts),
            "pending_reinforcements": len(self._reinforcement_buffer),
            "total_retrievals": self._total_retrievals,
        }

        # Add robust reinforcement stats if enabled
        if self.use_robust_reinforcement:
            stats["robust_reinforcement"] = {
                "enabled": True,
                "exploration_factor": self.exploration_factor,
                "decay_half_life_hours": self.decay_half_life_hours,
                "pending_robust_signals": sum(
                    len(signals)
                    for signals in self._robust_reinforcement_buffer.values()
                ),
            }
        else:
            stats["robust_reinforcement"] = {"enabled": False}

        return stats
