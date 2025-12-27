"""SVL Pipeline - Complete data flow orchestration through SVL.

This module provides the complete wired implementation that makes SVL
the mandatory kernel for ALL data flows in MindCore.

The pipeline implements:
1. INBOUND: Query validation and context decision
2. HOT PATH: FLR-only for simple queries (no CLST needed)
3. FULL PATH: FLR + CLST + External Sources for complex queries
4. OUTBOUND: Response validation and sanitization

Architecture:
    User Query
        ↓
    SVL Gate (inbound) → Validate query structure
        ↓
    Context Decision → Does this query need historical context?
        ↓
    ┌─────────────────────────────────────────────────────────┐
    │ HOT PATH (needs_clst=False)  │  FULL PATH (needs_clst=True)  │
    │   - FLR cache only           │   - FLR cache               │
    │   - Fast response            │   - CLST storage            │
    │                              │   - External sources         │
    │                              │     (fetch_for_topics)       │
    └─────────────────────────────────────────────────────────┘
        ↓
    SVL Gate (outbound) → Validate + sanitize response
        ↓
    Return to LLM

Example:
    from mindcore.v2.svl import SVLPipeline

    # Create complete pipeline
    pipeline = SVLPipeline(
        storage="sqlite:///memory.db",
        llm_call=my_llm_function,
    )

    # Store with full validation
    result = pipeline.store(
        llm_output={"content": "...", "memory_type": "preference"},
        user_id="user123",
    )

    # Query with automatic hot-path optimization
    result = pipeline.query(
        query="What are my preferences?",
        user_id="user123",
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .enforced_metadata import ContextDecision, HistoricalContextNeeded, MetadataExtractor
from .gate import GateDecision, GatePolicy, GateResult, RetryConfig, SVLGate

if TYPE_CHECKING:
    from mindcore.v2.flr import Memory
    from mindcore.v2.storage.base import BaseStorage

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Result Types
# =============================================================================


@dataclass
class QueryResult:
    """Result of a pipeline query operation."""

    success: bool
    memories: list[dict[str, Any]] = field(default_factory=list)
    external_data: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    # Flow information
    path_used: str = "full"  # "hot" or "full"
    context_decision: ContextDecision | None = None

    # Timing
    total_time_ms: float = 0.0
    gate_time_ms: float = 0.0
    flr_time_ms: float = 0.0
    clst_time_ms: float = 0.0
    external_fetch_time_ms: float = 0.0

    # Attention hints for LLM
    attention_focus: list[str] = field(default_factory=list)
    suggested_memory_types: list[str] = field(default_factory=list)

    # Warnings/errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_count": len(self.memories),
            "external_data_count": len(self.external_data),
            "path_used": self.path_used,
            "context_decision": self.context_decision.to_dict() if self.context_decision else None,
            "timing": {
                "total_ms": self.total_time_ms,
                "gate_ms": self.gate_time_ms,
                "flr_ms": self.flr_time_ms,
                "clst_ms": self.clst_time_ms,
                "external_ms": self.external_fetch_time_ms,
            },
            "attention_focus": self.attention_focus,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class StoreResult:
    """Result of a pipeline store operation."""

    success: bool
    memory_id: str | None = None
    gate_result: GateResult | None = None

    # Processing info
    canonicalized: bool = False
    retry_count: int = 0
    quality_score: float = 1.0

    # Error information
    error_message: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "canonicalized": self.canonicalized,
            "retry_count": self.retry_count,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "errors": self.errors,
        }


# =============================================================================
# SVL Pipeline - The Complete Orchestrator
# =============================================================================


class SVLPipeline:
    """Complete SVL-first data flow pipeline.

    This class orchestrates ALL data flows through SVL, implementing:

    1. INBOUND FLOW (Store):
       LLM Output → SVL Gate (validate + canonicalize) → CLST

    2. QUERY FLOW (Recall):
       User Query → Context Decision → Hot/Full Path → SVL Gate (outbound)

    3. HOT PATH OPTIMIZATION:
       When ContextDecision.needs_clst() is False, skip CLST entirely
       for faster responses on simple queries.

    4. EXTERNAL DATA INTEGRATION:
       When context is needed, fetch from external sources
       (databases, APIs, MCP servers) based on suggested topics.

    Example:
        pipeline = SVLPipeline(
            storage="sqlite:///memory.db",
            llm_call=my_llm_function,
        )

        # Store with mandatory SVL validation
        result = pipeline.store(
            llm_output={"content": "User prefers dark mode", "memory_type": "preference"},
            user_id="user123",
        )

        # Query with automatic hot-path optimization
        result = pipeline.query(
            query="What are my display preferences?",
            user_id="user123",
        )

        # Hot path is used when no historical context needed
        # Full path is used when CLST + external data needed
    """

    def __init__(
        self,
        storage: str | Any = "sqlite:///mindcore.db",
        vocabulary: Any = None,
        gate_policy: GatePolicy | None = None,
        retry_config: RetryConfig | None = None,
        llm_call: Callable[[str], str] | None = None,
        enable_hot_path: bool = True,
        enable_external_sources: bool = True,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize the SVL Pipeline.

        Args:
            storage: Storage backend or connection string
            vocabulary: SharedVocabularyLayer (uses default if None)
            gate_policy: SVL Gate policy configuration
            retry_config: Retry strategy configuration
            llm_call: LLM function for context decisions and retries
            enable_hot_path: Enable hot-path optimization (skip CLST when not needed)
            enable_external_sources: Enable external data source fetching
            cache_size: FLR cache size
            cache_ttl_seconds: FLR cache TTL
        """
        from mindcore.v2.flr import FLR
        from mindcore.v2.storage import SQLiteStorage
        from mindcore.v2.svl import DEFAULT_SVL, SharedVocabularyLayer

        # Initialize storage
        if isinstance(storage, str):
            if storage.startswith("sqlite:///"):
                db_path = storage[10:]
                self._storage = SQLiteStorage(db_path)
            elif storage.startswith(("postgresql://", "postgres://")):
                from mindcore.v2.storage.postgres import PostgresStorage

                self._storage = PostgresStorage(storage)
            else:
                self._storage = SQLiteStorage(storage)
        else:
            self._storage = storage

        # Initialize vocabulary (SVL)
        self._vocabulary = vocabulary or DEFAULT_SVL

        # Initialize SVL Gate
        self._gate = SVLGate(
            svl=self._vocabulary,
            policy=gate_policy or GatePolicy(),
            retry_config=retry_config or RetryConfig(),
        )

        # Initialize FLR (cache layer)
        self._flr = FLR(
            storage=self._storage,
            cache_size=cache_size,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        # Initialize metadata extractor for context decisions
        self._metadata_extractor = MetadataExtractor(svl=self._vocabulary)

        # Configuration
        self._llm_call = llm_call
        self._enable_hot_path = enable_hot_path
        self._enable_external_sources = enable_external_sources

        # Statistics
        self._stats = {
            "total_queries": 0,
            "hot_path_queries": 0,
            "full_path_queries": 0,
            "external_fetches": 0,
            "stores": 0,
            "store_failures": 0,
        }

    # =========================================================================
    # STORE FLOW (Inbound)
    # =========================================================================

    def store(
        self,
        llm_output: dict[str, Any] | str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> StoreResult:
        """Store a memory through the SVL Gate.

        This is the ONLY way to store memories. The data flow is:
        1. Parse LLM output
        2. SVL Gate: Canonicalize + Validate
        3. Store in CLST

        Args:
            llm_output: LLM output (dict or JSON string)
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier

        Returns:
            StoreResult with success status and memory ID
        """
        self._stats["stores"] += 1

        # Process through SVL Gate
        gate_result = self._gate.process_inbound(
            llm_output=llm_output,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_call=self._llm_call,
        )

        if not gate_result.success:
            self._stats["store_failures"] += 1
            return StoreResult(
                success=False,
                gate_result=gate_result,
                error_message="; ".join(e.message for e in gate_result.errors),
                errors=[e.message for e in gate_result.errors],
            )

        # Store the validated/canonicalized memory
        try:
            from mindcore.v2.flr import Memory

            memory = Memory.from_dict(gate_result.memory)
            memory_id = self._storage.store(memory)

            return StoreResult(
                success=True,
                memory_id=memory_id,
                gate_result=gate_result,
                canonicalized=gate_result.canonicalized,
                retry_count=gate_result.retry_count,
                quality_score=gate_result.quality_score,
            )

        except Exception as e:
            logger.error("Storage error after gate validation: %s", e)
            self._stats["store_failures"] += 1
            return StoreResult(
                success=False,
                gate_result=gate_result,
                error_message=f"Storage error: {e}",
                errors=[str(e)],
            )

    # =========================================================================
    # QUERY FLOW (with Hot Path Optimization)
    # =========================================================================

    def query(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
        force_full_path: bool = False,
    ) -> QueryResult:
        """Query memories with automatic hot-path optimization.

        The data flow is:
        1. Get ContextDecision from LLM (if available)
        2. If needs_clst=False AND hot_path enabled: use cache only
        3. If needs_clst=True OR full path forced: query CLST + external sources
        4. SVL Gate: Validate + sanitize outbound data

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            attention_hints: Topics to prioritize
            memory_types: Filter by memory types
            limit: Maximum memories to return
            force_full_path: Force full CLST query (disable hot path)

        Returns:
            QueryResult with memories and metadata
        """
        start_time = time.time()
        self._stats["total_queries"] += 1

        result = QueryResult(success=True)

        # Step 1: Get context decision
        context_decision = self._get_context_decision(
            query=query,
            user_id=user_id,
            session_id=session_id,
        )
        result.context_decision = context_decision

        # Step 2: Determine path
        use_hot_path = (
            self._enable_hot_path
            and not force_full_path
            and context_decision is not None
            and not context_decision.needs_clst()
        )

        if use_hot_path:
            # HOT PATH: Cache only
            result.path_used = "hot"
            self._stats["hot_path_queries"] += 1
            result = self._execute_hot_path(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                attention_hints=attention_hints or (
                    context_decision.suggested_topics if context_decision else []
                ),
                memory_types=memory_types,
                limit=limit,
                result=result,
            )
        else:
            # FULL PATH: Cache + CLST + External Sources
            result.path_used = "full"
            self._stats["full_path_queries"] += 1
            result = self._execute_full_path(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                attention_hints=attention_hints or (
                    context_decision.suggested_topics if context_decision else []
                ),
                memory_types=memory_types,
                limit=limit,
                context_decision=context_decision,
                result=result,
            )

        # Step 3: Outbound validation
        gate_start = time.time()
        validated_memories = []
        for memory in result.memories:
            gate_result = self._gate.process_outbound(memory)
            if gate_result.success:
                validated_memories.append(gate_result.memory)
            else:
                result.warnings.append(
                    f"Memory {memory.get('memory_id', 'unknown')} failed outbound validation"
                )

        result.memories = validated_memories
        result.gate_time_ms += (time.time() - gate_start) * 1000

        # Final timing
        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    def _get_context_decision(
        self,
        query: str,
        user_id: str,
        session_id: str | None,
    ) -> ContextDecision | None:
        """Get context decision from LLM or use default.

        Args:
            query: User query
            user_id: User ID
            session_id: Session ID

        Returns:
            ContextDecision or None if LLM not available
        """
        if not self._llm_call:
            # No LLM available - default to needing context
            return ContextDecision(
                historical_context_needed=HistoricalContextNeeded.TRUE,
                suggested_topics=[],
                suggested_categories=[],
                reasoning="LLM not available - defaulting to full context",
                urgency="medium",
            )

        try:
            # Get context decision prompt
            prompt = self._metadata_extractor.get_context_decision_prompt(
                user_message=query,
                session_id=session_id or "",
                user_id=user_id,
            )

            # Call LLM
            response = self._llm_call(prompt)

            # Parse response
            decision_data = self._parse_json(response)
            return ContextDecision.from_dict(decision_data)

        except Exception as e:
            logger.warning("Failed to get context decision: %s", e)
            # Default to needing context on error
            return ContextDecision(
                historical_context_needed=HistoricalContextNeeded.TRUE,
                reasoning=f"Error getting decision: {e}",
            )

    def _execute_hot_path(
        self,
        query: str,
        user_id: str,
        agent_id: str | None,
        attention_hints: list[str],
        memory_types: list[str] | None,
        limit: int,
        result: QueryResult,
    ) -> QueryResult:
        """Execute hot path - FLR cache only.

        This is the fast path for queries that don't need historical context.

        Args:
            query: Search query
            user_id: User ID
            agent_id: Agent ID
            attention_hints: Topics to prioritize
            memory_types: Filter by types
            limit: Max results
            result: QueryResult to populate

        Returns:
            Updated QueryResult
        """
        flr_start = time.time()

        try:
            # Query FLR cache only (don't hit storage)
            # We access the internal cache directly for true hot-path
            cached_memories = self._flr._query_cache(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                attention_hints=attention_hints,
                memory_types=memory_types or [],
            )

            # Score memories
            scored = self._flr._score_memories(
                memories=cached_memories,
                query=query,
                attention_hints=attention_hints,
            )

            # Filter and limit
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:limit]

            # Convert to dict for result
            result.memories = [m.to_dict() for m, _ in scored]
            result.scores = [s for _, s in scored]

            # Extract attention focus
            result.attention_focus = self._flr._extract_attention_focus(
                memories=[m for m, _ in scored],
                hints=attention_hints,
            )

        except Exception as e:
            logger.error("Hot path query failed: %s", e)
            result.errors.append(f"Hot path error: {e}")
            result.success = False

        result.flr_time_ms = (time.time() - flr_start) * 1000
        return result

    def _execute_full_path(
        self,
        query: str,
        user_id: str,
        agent_id: str | None,
        session_id: str | None,
        attention_hints: list[str],
        memory_types: list[str] | None,
        limit: int,
        context_decision: ContextDecision | None,
        result: QueryResult,
    ) -> QueryResult:
        """Execute full path - FLR + CLST + External Sources.

        This is the complete path for queries needing historical context.

        Args:
            query: Search query
            user_id: User ID
            agent_id: Agent ID
            session_id: Session ID
            attention_hints: Topics to prioritize
            memory_types: Filter by types
            limit: Max results
            context_decision: Context decision from LLM
            result: QueryResult to populate

        Returns:
            Updated QueryResult
        """
        # Step 1: Query FLR (includes cache + CLST)
        flr_start = time.time()
        try:
            flr_result = self._flr.query(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                attention_hints=attention_hints,
                memory_types=memory_types,
                limit=limit,
            )

            result.memories = [m.to_dict() for m in flr_result.memories]
            result.scores = flr_result.scores
            result.attention_focus = flr_result.attention_focus
            result.suggested_memory_types = flr_result.suggested_memory_types

        except Exception as e:
            logger.error("FLR query failed: %s", e)
            result.errors.append(f"FLR error: {e}")

        result.flr_time_ms = (time.time() - flr_start) * 1000

        # Step 2: Fetch external data if enabled
        if self._enable_external_sources and context_decision:
            external_start = time.time()
            result = self._fetch_external_sources(
                context_decision=context_decision,
                user_id=user_id,
                session_id=session_id,
                query=query,
                result=result,
            )
            result.external_fetch_time_ms = (time.time() - external_start) * 1000

        return result

    def _fetch_external_sources(
        self,
        context_decision: ContextDecision,
        user_id: str,
        session_id: str | None,
        query: str,
        result: QueryResult,
    ) -> QueryResult:
        """Fetch data from external sources based on context decision.

        Args:
            context_decision: Context decision with suggested topics
            user_id: User ID
            session_id: Session ID
            query: Original query
            result: QueryResult to populate

        Returns:
            Updated QueryResult with external data
        """
        if not context_decision.suggested_topics:
            return result

        self._stats["external_fetches"] += 1

        try:
            # Build context for external source fetching
            context = {
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
            }

            # Fetch for suggested topics
            external_results = self._vocabulary.fetch_for_topics(
                topics=context_decision.suggested_topics,
                context=context,
            )

            # Add to result
            for fetch_result in external_results:
                if fetch_result.success and fetch_result.data:
                    result.external_data.append({
                        "source": fetch_result.source_name,
                        "topic": fetch_result.topic,
                        "data": fetch_result.data,
                        "cached": fetch_result.cached,
                    })

        except Exception as e:
            logger.warning("External source fetch failed: %s", e)
            result.warnings.append(f"External fetch warning: {e}")

        return result

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from: {text[:200]}...")

    # =========================================================================
    # STATISTICS & MANAGEMENT
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        total_queries = self._stats["total_queries"]
        return {
            "total_queries": total_queries,
            "hot_path_queries": self._stats["hot_path_queries"],
            "full_path_queries": self._stats["full_path_queries"],
            "hot_path_ratio": (
                self._stats["hot_path_queries"] / total_queries
                if total_queries > 0
                else 0
            ),
            "external_fetches": self._stats["external_fetches"],
            "stores": self._stats["stores"],
            "store_failures": self._stats["store_failures"],
            "store_success_rate": (
                (self._stats["stores"] - self._stats["store_failures"])
                / self._stats["stores"]
                if self._stats["stores"] > 0
                else 0
            ),
            "gate_stats": self._gate.get_stats(),
            "flr_stats": self._flr.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_queries": 0,
            "hot_path_queries": 0,
            "full_path_queries": 0,
            "external_fetches": 0,
            "stores": 0,
            "store_failures": 0,
        }
        self._gate.reset_stats()

    def close(self) -> None:
        """Close all connections."""
        self._flr.flush_reinforcements()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def create_pipeline(
    storage: str = "sqlite:///mindcore.db",
    llm_call: Callable[[str], str] | None = None,
    enable_hot_path: bool = True,
    enable_external_sources: bool = True,
) -> SVLPipeline:
    """Create a configured SVL Pipeline.

    This is the recommended way to create a MindCore instance with
    full SVL Gate enforcement.

    Args:
        storage: Storage connection string
        llm_call: LLM function for context decisions
        enable_hot_path: Enable hot-path optimization
        enable_external_sources: Enable external data source fetching

    Returns:
        Configured SVLPipeline
    """
    return SVLPipeline(
        storage=storage,
        llm_call=llm_call,
        enable_hot_path=enable_hot_path,
        enable_external_sources=enable_external_sources,
    )
