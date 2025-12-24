"""ContextGateway - Unified context assembly for MindCore.

The ContextGateway is the single entry point for building LLM context.
It orchestrates FLR (hot path), CLST (cold path), and SVL (data sources)
to produce a unified, controlled context for the main LLM.

Key Features:
- Hierarchical retrieval: Sessions → Memories (reduces search space)
- Weighted metadata matching: No embeddings needed for most queries
- SVL data source integration: Auto-fetches tables/APIs based on topics
- FLR hot cache: Fast path for recent/frequent memories
- Unified context format: Single API for all context needs

Example:
    gateway = ContextGateway(
        storage=postgres_storage,
        svl=shared_vocabulary_layer,
        flr_cache_size=1000,
    )

    # Get unified context for a user query
    context = gateway.build_context(
        query="What about my order #12345?",
        user_id="user_123",
        session_id="session_abc",
        attention_hints=["orders", "shipping"],
    )

    # Context contains:
    # - Relevant memories (hierarchically retrieved)
    # - SVL data source results (auto-fetched from topics)
    # - Session context (topic weights, importance stats)
    # - Formatted for LLM consumption
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mindcore.v2.clst.aggregates import SessionAggregate
    from mindcore.v2.flr import Memory
    from mindcore.v2.storage.base import BaseStorage
    from mindcore.v2.svl import SharedVocabularyLayer


@dataclass
class QueryMetadata:
    """SVL-compliant metadata for a query.

    Tracks query characteristics for traceability and analysis.
    """

    query_id: str
    query_text: str
    session_id: str | None
    user_id: str

    # SVL-compliant fields (populated from query analysis)
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    message_type: str = "query"
    message_intent: str | None = None
    urgency: str = "medium"
    confidence: str = "high"

    # Context retrieval info
    attention_hints: list[str] = field(default_factory=list)
    sessions_searched: int = 0
    memories_retrieved: int = 0
    sources_fetched: int = 0

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize query metadata."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "topics": self.topics,
            "categories": self.categories,
            "message_type": self.message_type,
            "message_intent": self.message_intent,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "attention_hints": self.attention_hints,
            "sessions_searched": self.sessions_searched,
            "memories_retrieved": self.memories_retrieved,
            "sources_fetched": self.sources_fetched,
            "created_at": self.created_at.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class ResponseMetadata:
    """SVL-compliant metadata for a response.

    Tracks response characteristics for traceability and analysis.
    """

    response_id: str
    query_id: str  # Links to originating query
    session_id: str | None
    user_id: str

    # SVL-compliant fields
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    message_type: str = "response"
    message_intent: str | None = None
    sentiment: str = "neutral"
    confidence: str = "high"

    # Memory operations performed
    memories_stored: int = 0
    memory_ids: list[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize response metadata."""
        return {
            "response_id": self.response_id,
            "query_id": self.query_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "topics": self.topics,
            "categories": self.categories,
            "message_type": self.message_type,
            "message_intent": self.message_intent,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "memories_stored": self.memories_stored,
            "memory_ids": self.memory_ids,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ContextResult:
    """Result from context building.

    Contains all context needed for LLM response generation,
    plus SVL-compliant query metadata for traceability.
    """

    # Retrieved memories (ordered by relevance/time)
    memories: list[Memory] = field(default_factory=list)

    # Session context for continuity
    current_session: SessionAggregate | None = None
    related_sessions: list[SessionAggregate] = field(default_factory=list)

    # SVL data source results
    source_data: dict[str, list[Any]] = field(default_factory=dict)

    # Topics/categories extracted or matched
    matched_topics: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)

    # SVL-compliant query metadata (for traceability)
    query_metadata: QueryMetadata | None = None

    # Statistics
    total_memories_searched: int = 0
    sessions_searched: int = 0
    sources_fetched: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False

    def to_llm_context(self, max_memories: int = 20, include_source_data: bool = True) -> str:
        """Format context for LLM consumption.

        Returns structured text suitable for injection into LLM prompt.
        """
        parts = []

        # Current session summary
        if self.current_session:
            parts.append(self._format_session_summary(self.current_session))

        # Memories grouped by topic
        if self.memories:
            parts.append(self._format_memories(self.memories[:max_memories]))

        # Source data (tables, APIs, etc.)
        if include_source_data and self.source_data:
            parts.append(self._format_source_data())

        return "\n\n".join(parts)

    def _format_session_summary(self, session: SessionAggregate) -> str:
        """Format session summary for context."""
        top_topics = session.get_top_topics(5)
        topics_str = ", ".join(f"{t[0]} ({t[1]:.0%})" for t in top_topics)

        return f"""## Current Session Context
- Topics discussed: {topics_str}
- Messages in session: {session.message_count}
- Session importance: {session.importance_avg:.2f}
- Dominant sentiment: {session.dominant_sentiment or 'neutral'}"""

    def _format_memories(self, memories: list[Memory]) -> str:
        """Format memories for LLM context."""
        if not memories:
            return ""

        lines = ["## Relevant Memories"]

        # Group by session for narrative flow
        current_session = None
        for memory in memories:
            if memory.session_id != current_session:
                current_session = memory.session_id
                if current_session:
                    lines.append(f"\n### Session: {current_session[:8]}...")

            importance_marker = "⭐" if memory.importance > 0.7 else ""
            lines.append(
                f"- [{memory.memory_type}] {memory.content} {importance_marker}"
            )

        return "\n".join(lines)

    def _format_source_data(self) -> str:
        """Format source data for LLM context."""
        lines = ["## Related Data"]

        for source_name, results in self.source_data.items():
            lines.append(f"\n### {source_name}")
            for result in results[:5]:  # Limit per source
                if isinstance(result, dict):
                    # Format dict results
                    items = [f"{k}: {v}" for k, v in list(result.items())[:5]]
                    lines.append(f"  - {', '.join(items)}")
                else:
                    lines.append(f"  - {result}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context result."""
        return {
            "memories": [m.to_dict() for m in self.memories],
            "current_session": self.current_session.to_dict() if self.current_session else None,
            "related_sessions": [s.to_dict() for s in self.related_sessions],
            "source_data": self.source_data,
            "matched_topics": self.matched_topics,
            "matched_categories": self.matched_categories,
            "query_metadata": self.query_metadata.to_dict() if self.query_metadata else None,
            "stats": {
                "total_memories_searched": self.total_memories_searched,
                "sessions_searched": self.sessions_searched,
                "sources_fetched": self.sources_fetched,
                "latency_ms": self.latency_ms,
                "from_cache": self.from_cache,
            },
        }


class ContextGateway:
    """Unified context gateway for FLR, CLST, and SVL.

    This is the single entry point for building LLM context. It coordinates:
    - FLR: Hot path for recent/cached memories
    - CLST: Cold path with hierarchical session-based retrieval
    - SVL: Data source mapping and auto-fetching

    The gateway uses weighted metadata matching to reduce embedding usage
    and provides fast, relevant context retrieval.
    """

    def __init__(
        self,
        storage: BaseStorage,
        svl: SharedVocabularyLayer | None = None,
        flr_cache_size: int = 1000,
        flr_cache_ttl_seconds: int = 300,
        default_session_limit: int = 5,
        default_memory_limit: int = 50,
        track_queries: bool = False,
    ):
        """Initialize ContextGateway.

        Args:
            storage: Storage backend (PostgreSQL recommended)
            svl: SharedVocabularyLayer for data source mapping
            flr_cache_size: Size of FLR LRU cache
            flr_cache_ttl_seconds: TTL for cached memories
            default_session_limit: Default number of sessions to search
            default_memory_limit: Default number of memories to return
            track_queries: Store queries/responses as working memories for traceability
        """
        self._storage = storage
        self._svl = svl
        self._flr_cache_size = flr_cache_size
        self._flr_cache_ttl = flr_cache_ttl_seconds
        self._default_session_limit = default_session_limit
        self._default_memory_limit = default_memory_limit
        self._track_queries = track_queries

        # LRU cache for hot path
        self._hot_cache: dict[str, tuple[list[Any], float]] = {}

    def build_context(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float = 0.3,
        min_topic_weight: float = 0.1,
        session_limit: int | None = None,
        memory_limit: int | None = None,
        include_source_data: bool = True,
        use_cache: bool = True,
    ) -> ContextResult:
        """Build unified context for LLM.

        This is the main entry point. It:
        1. Checks FLR hot cache
        2. Queries relevant sessions by weighted topics
        3. Retrieves memories from those sessions
        4. Fetches SVL data sources for matched topics
        5. Assembles unified context

        Args:
            query: User query text
            user_id: User identifier
            session_id: Current session (for context)
            agent_id: Agent identifier
            attention_hints: Topics to focus on
            category_hints: Categories to focus on
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            min_topic_weight: Minimum topic weight for session matching
            session_limit: Max sessions to search
            memory_limit: Max memories to return
            include_source_data: Whether to fetch SVL sources
            use_cache: Whether to check FLR cache

        Returns:
            ContextResult with all context data
        """
        start_time = time.time()

        session_limit = session_limit or self._default_session_limit
        memory_limit = memory_limit or self._default_memory_limit

        result = ContextResult()

        # Step 1: Check FLR hot cache
        cache_key = self._build_cache_key(user_id, session_id, attention_hints)
        if use_cache and cache_key in self._hot_cache:
            cached_memories, cache_time = self._hot_cache[cache_key]
            if time.time() - cache_time < self._flr_cache_ttl:
                result.memories = cached_memories
                result.from_cache = True
                result.latency_ms = (time.time() - start_time) * 1000
                return result

        # Step 2: Get current session context if provided
        if session_id:
            result.current_session = self._storage.get_session_aggregate(session_id)

            # Merge current session topics with attention hints
            if result.current_session and not attention_hints:
                top_topics = result.current_session.get_top_topics(5)
                attention_hints = [t[0] for t in top_topics]

        # Step 3: Query relevant sessions by weighted metadata
        if attention_hints or category_hints:
            related_sessions = self._storage.query_sessions(
                user_id=user_id,
                topic_hints=attention_hints,
                category_hints=category_hints,
                min_importance_avg=min_importance,
                min_topic_weight=min_topic_weight,
                agent_ids=[agent_id] if agent_id else None,
                limit=session_limit,
            )
            result.related_sessions = related_sessions
            result.sessions_searched = len(related_sessions)

            # Extract session IDs for memory query
            session_ids = [s.session_id for s in related_sessions]

            # Include current session if not already in list
            if session_id and session_id not in session_ids:
                session_ids.insert(0, session_id)

            # Step 4: Query memories from relevant sessions
            if session_ids:
                result.memories = self._storage.query_memories_by_sessions(
                    session_ids=session_ids,
                    min_importance=min_importance,
                    memory_types=memory_types,
                    limit=memory_limit,
                    order_by_message_index=True,
                )
                result.total_memories_searched = len(result.memories)

        # Fallback: Direct memory search if no sessions found
        if not result.memories:
            result.memories = self._storage.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                topics=attention_hints,
                categories=category_hints,
                memory_types=memory_types,
                min_importance=min_importance,
                limit=memory_limit,
            )
            result.total_memories_searched = len(result.memories)

        # Step 5: Collect matched topics/categories
        result.matched_topics = self._collect_topics(result.memories)
        result.matched_categories = self._collect_categories(result.memories)

        # Step 6: Fetch SVL data sources for matched topics
        if include_source_data and self._svl and result.matched_topics:
            result.source_data = self._fetch_source_data(
                topics=result.matched_topics,
                user_id=user_id,
                session_id=session_id,
                query=query,
            )
            result.sources_fetched = len(result.source_data)

        # Update cache
        if use_cache:
            self._hot_cache[cache_key] = (result.memories, time.time())
            self._prune_cache()

        result.latency_ms = (time.time() - start_time) * 1000

        # Step 7: Create SVL-compliant query metadata for traceability
        result.query_metadata = QueryMetadata(
            query_id=f"qry_{uuid.uuid4().hex[:12]}",
            query_text=query,
            session_id=session_id,
            user_id=user_id,
            topics=result.matched_topics,
            categories=result.matched_categories,
            message_type="query",
            message_intent=self._infer_intent(query) if query else None,
            attention_hints=attention_hints or [],
            sessions_searched=result.sessions_searched,
            memories_retrieved=result.total_memories_searched,
            sources_fetched=result.sources_fetched,
            latency_ms=result.latency_ms,
        )

        # Store query as memory for traceability if session is active
        if session_id and self._track_queries:
            self._store_query_memory(result.query_metadata)

        return result

    def record_response(
        self,
        query_metadata: QueryMetadata,
        response_text: str,
        memories_to_store: list[Any] | None = None,
        sentiment: str = "neutral",
        confidence: str = "high",
    ) -> ResponseMetadata:
        """Record a response with SVL-compliant metadata.

        Creates a traceable record of the response and stores any
        memories generated by the LLM.

        Args:
            query_metadata: Metadata from the originating query
            response_text: The LLM's response text
            memories_to_store: Memories generated by the LLM to store
            sentiment: Response sentiment
            confidence: Response confidence level

        Returns:
            ResponseMetadata for the recorded response
        """
        response_id = f"rsp_{uuid.uuid4().hex[:12]}"
        memory_ids = []

        # Store any memories from the LLM response
        if memories_to_store:
            for memory in memories_to_store:
                if query_metadata.session_id:
                    memory.session_id = query_metadata.session_id
                memory_id = self._storage.store(memory)
                memory_ids.append(memory_id)

        # Create response metadata
        response_metadata = ResponseMetadata(
            response_id=response_id,
            query_id=query_metadata.query_id,
            session_id=query_metadata.session_id,
            user_id=query_metadata.user_id,
            topics=query_metadata.topics,
            categories=query_metadata.categories,
            message_type="response",
            sentiment=sentiment,
            confidence=confidence,
            memories_stored=len(memory_ids),
            memory_ids=memory_ids,
        )

        # Store response as memory for traceability if session is active
        if query_metadata.session_id and self._track_queries:
            self._store_response_memory(response_metadata, response_text)

        return response_metadata

    def _store_query_memory(self, query_metadata: QueryMetadata) -> None:
        """Store query as a working memory for session traceability."""
        from mindcore.v2.flr import Memory

        memory = Memory(
            memory_id=f"mem_{query_metadata.query_id}",
            content=f"[Query] {query_metadata.query_text}",
            memory_type="working",
            user_id=query_metadata.user_id,
            session_id=query_metadata.session_id,
            topics=query_metadata.topics,
            categories=query_metadata.categories,
            importance=0.3,  # Queries are less important than content
            access_level="private",
        )
        self._storage.store(memory)

    def _store_response_memory(self, response_metadata: ResponseMetadata, response_text: str) -> None:
        """Store response as a working memory for session traceability."""
        from mindcore.v2.flr import Memory

        # Truncate long responses for storage
        content = response_text[:500] + "..." if len(response_text) > 500 else response_text

        memory = Memory(
            memory_id=f"mem_{response_metadata.response_id}",
            content=f"[Response] {content}",
            memory_type="working",
            user_id=response_metadata.user_id,
            session_id=response_metadata.session_id,
            topics=response_metadata.topics,
            categories=response_metadata.categories,
            sentiment=response_metadata.sentiment,
            importance=0.3,
            access_level="private",
        )
        self._storage.store(memory)

    def _infer_intent(self, query: str) -> str | None:
        """Infer basic intent from query text (simple heuristics)."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["what", "who", "where", "when", "how", "why", "?"]):
            return "ask_question"
        if any(w in query_lower for w in ["please", "can you", "could you", "would you"]):
            return "request_action"
        if any(w in query_lower for w in ["thanks", "thank you", "appreciate"]):
            return "thanks"
        if any(w in query_lower for w in ["hi", "hello", "hey"]):
            return "greeting"
        if any(w in query_lower for w in ["bye", "goodbye", "see you"]):
            return "farewell"

        return "provide_info"

    def store_memory(
        self,
        memory: Any,  # Memory type
        session_id: str | None = None,
    ) -> str:
        """Store a memory and update session aggregate.

        This is the unified store method that:
        1. Assigns session_id and message_index if provided
        2. Stores the memory
        3. Updates session aggregate automatically

        Args:
            memory: Memory to store
            session_id: Session to associate with

        Returns:
            Memory ID
        """
        if session_id:
            memory.session_id = session_id

        # Storage automatically updates session aggregate
        return self._storage.store(memory)

    def promote_to_session(
        self,
        memory_ids: list[str],
        session_id: str,
        user_id: str,
    ) -> int:
        """Promote standalone memories to a session.

        Used when memories were created without session context
        and need to be grouped later.

        Args:
            memory_ids: Memories to promote
            session_id: Target session
            user_id: User identifier

        Returns:
            Number of memories promoted
        """
        promoted = 0
        for memory_id in memory_ids:
            memory = self._storage.get(memory_id)
            if memory and memory.user_id == user_id:
                memory.session_id = session_id
                memory.message_index = self._storage.get_next_message_index(session_id)
                self._storage.update(memory)
                self._storage.update_session_aggregate_from_memory(session_id, memory)
                promoted += 1
        return promoted

    def get_session_summary(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session summary for context injection.

        Returns a dict suitable for LLM context with:
        - Top topics and their weights
        - Importance/confidence statistics
        - Message count and time span
        """
        aggregate = self._storage.get_session_aggregate(session_id)
        if not aggregate:
            return None

        return {
            "session_id": session_id,
            "top_topics": aggregate.get_top_topics(5),
            "top_categories": aggregate.get_top_categories(3),
            "importance": {
                "min": aggregate.importance_min,
                "max": aggregate.importance_max,
                "avg": aggregate.importance_avg,
            },
            "message_count": aggregate.message_count,
            "dominant_sentiment": aggregate.dominant_sentiment,
            "started_at": aggregate.started_at.isoformat() if aggregate.started_at else None,
            "last_activity": aggregate.last_activity_at.isoformat() if aggregate.last_activity_at else None,
        }

    def invalidate_cache(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Invalidate cached entries.

        Args:
            user_id: Invalidate all caches for this user
            session_id: Invalidate caches containing this session

        Returns:
            Number of entries invalidated
        """
        if user_id is None and session_id is None:
            count = len(self._hot_cache)
            self._hot_cache.clear()
            return count

        to_remove = []
        for key in self._hot_cache:
            if user_id and key.startswith(f"{user_id}:"):
                to_remove.append(key)
            elif session_id and f":{session_id}:" in key:
                to_remove.append(key)

        for key in to_remove:
            del self._hot_cache[key]

        return len(to_remove)

    def _build_cache_key(
        self,
        user_id: str,
        session_id: str | None,
        attention_hints: list[str] | None,
    ) -> str:
        """Build cache key for FLR hot cache."""
        hints_str = ",".join(sorted(attention_hints)) if attention_hints else ""
        return f"{user_id}:{session_id or 'none'}:{hints_str}"

    def _prune_cache(self) -> None:
        """Prune cache to max size using LRU."""
        if len(self._hot_cache) <= self._flr_cache_size:
            return

        # Sort by timestamp (oldest first)
        sorted_keys = sorted(
            self._hot_cache.keys(),
            key=lambda k: self._hot_cache[k][1]
        )

        # Remove oldest entries
        to_remove = len(self._hot_cache) - self._flr_cache_size
        for key in sorted_keys[:to_remove]:
            del self._hot_cache[key]

    def _collect_topics(self, memories: list[Any]) -> list[str]:
        """Collect unique topics from memories."""
        topics = set()
        for memory in memories:
            topics.update(memory.topics)
        return list(topics)

    def _collect_categories(self, memories: list[Any]) -> list[str]:
        """Collect unique categories from memories."""
        categories = set()
        for memory in memories:
            categories.update(memory.categories)
        return list(categories)

    def _fetch_source_data(
        self,
        topics: list[str],
        user_id: str,
        session_id: str | None,
        query: str,
    ) -> dict[str, list[Any]]:
        """Fetch data from SVL-mapped sources."""
        if not self._svl:
            return {}

        context = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
        }

        try:
            results = self._svl.fetch_for_topics(topics, context)

            # Flatten results into source_name -> data mapping
            source_data = {}
            for topic, fetch_results in results.items():
                for fetch_result in fetch_results:
                    source_name = fetch_result.source_name
                    if source_name not in source_data:
                        source_data[source_name] = []
                    if fetch_result.data:
                        source_data[source_name].extend(
                            fetch_result.data if isinstance(fetch_result.data, list)
                            else [fetch_result.data]
                        )

            return source_data
        except Exception:
            # Don't fail context building on source fetch errors
            return {}

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return {
            "cache_size": len(self._hot_cache),
            "cache_max_size": self._flr_cache_size,
            "cache_ttl_seconds": self._flr_cache_ttl,
            "default_session_limit": self._default_session_limit,
            "default_memory_limit": self._default_memory_limit,
            "svl_enabled": self._svl is not None,
        }
