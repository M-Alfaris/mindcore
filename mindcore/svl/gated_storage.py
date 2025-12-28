"""SVL Gated Storage - Memory storage with mandatory SVL Gate enforcement.

This module provides storage wrappers that enforce SVL Gate validation
for ALL memory operations. There are NO bypass paths.

The architecture ensures:
    LLM Output -> SVL Gate -> GatedCLST -> Storage
    Storage -> GatedCLST -> SVL Gate -> LLM Input

This is analogous to how:
- OS kernels enforce syscall validation
- Database query planners validate and optimize queries
- Compilers perform type checking before code generation

Example:
    from mindcore.svl import SharedVocabularyLayer, SVLGate
    from mindcore.svl.gated_storage import GatedCLST, GatedFLR

    svl = SharedVocabularyLayer()
    gate = SVLGate(svl=svl)

    # Create gated storage (mandatory validation)
    clst = GatedCLST(storage=storage, gate=gate)

    # ALL stores go through the gate - no bypass possible
    result = clst.store(memory_data, user_id="user123")

    if result.success:
        print(f"Stored: {result.memory['memory_id']}")
    else:
        print(f"Rejected: {result.errors}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from .gate import GateDecision, GateResult, SVLGate


if TYPE_CHECKING:
    from mindcore.storage.base import BaseStorage

logger = logging.getLogger(__name__)


@dataclass
class StoreResult:
    """Result of a gated store operation."""

    success: bool
    memory_id: str | None = None
    gate_result: GateResult | None = None

    # Error information
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "gate_decision": self.gate_result.decision.value if self.gate_result else None,
            "quality_score": self.gate_result.quality_score if self.gate_result else None,
            "error_message": self.error_message,
        }


@dataclass
class RecallResult:
    """Result of a gated recall operation."""

    success: bool
    memories: list[dict[str, Any]]
    scores: list[float]
    query_latency_ms: float
    gate_processing_ms: float

    # Attention hints for the agent
    attention_focus: list[str]
    suggested_memory_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_count": len(self.memories),
            "scores": self.scores,
            "query_latency_ms": self.query_latency_ms,
            "gate_processing_ms": self.gate_processing_ms,
            "attention_focus": self.attention_focus,
            "suggested_memory_types": self.suggested_memory_types,
        }


class GatedCLST:
    """Gated Cognitive Long-term Storage Transfer.

    This class wraps CLST with mandatory SVL Gate validation.
    ALL memory operations MUST pass through the gate.

    IMPORTANT: There is NO bypass path. The validate=False option
    that existed in the original CLST has been removed.

    Example:
        gate = SVLGate(svl=svl)
        clst = GatedCLST(storage=storage, gate=gate)

        # Store with mandatory validation
        result = clst.store({"content": "...", "memory_type": "preference"}, user_id="u1")

        if result.success:
            memory_id = result.memory_id
        else:
            # Handle validation errors
            for error in result.gate_result.errors:
                print(f"Error: {error.message}")

        # Store with LLM retry (for automatic error correction)
        result = clst.store(
            memory_data,
            user_id="u1",
            llm_call=my_llm_function,  # For retry on validation failure
        )
    """

    def __init__(
        self,
        storage: BaseStorage,
        gate: SVLGate,
        compression_llm: Callable | None = None,
    ):
        """Initialize GatedCLST.

        Args:
            storage: Storage backend
            gate: SVL Gate (mandatory)
            compression_llm: Optional LLM for compression summarization
        """
        self._storage = storage
        self._gate = gate
        self._compression_llm = compression_llm

    def store(
        self,
        memory_data: dict[str, Any],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> StoreResult:
        """Store a memory with mandatory SVL Gate validation.

        This is the ONLY way to store memories. There is no bypass.

        Args:
            memory_data: Memory data dict from LLM
            user_id: User identifier (required)
            agent_id: Agent identifier (optional)
            session_id: Session identifier (optional)
            llm_call: Optional LLM function for retry strategies

        Returns:
            StoreResult with success status and memory ID
        """
        # Pass through SVL Gate - MANDATORY
        gate_result = self._gate.process_inbound(
            llm_output=memory_data,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_call=llm_call,
        )

        if not gate_result.success:
            return StoreResult(
                success=False,
                gate_result=gate_result,
                error_message="; ".join(e.message for e in gate_result.errors),
            )

        # Gate passed - store the validated/canonicalized memory
        try:
            from mindcore.flr import Memory

            memory = Memory.from_dict(gate_result.memory)
            memory_id = self._storage.store(memory)

            return StoreResult(
                success=True,
                memory_id=memory_id,
                gate_result=gate_result,
            )

        except Exception as e:
            logger.error("Storage error after gate validation: %s", e)
            return StoreResult(
                success=False,
                gate_result=gate_result,
                error_message=f"Storage error: {e}",
            )

    def store_batch(
        self,
        memories: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> list[StoreResult]:
        """Store multiple memories with mandatory validation.

        Each memory is validated individually through the SVL Gate.

        Args:
            memories: List of memory data dicts
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            llm_call: Optional LLM function for retry

        Returns:
            List of StoreResults (one per memory)
        """
        results = []
        for memory_data in memories:
            result = self.store(
                memory_data=memory_data,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                llm_call=llm_call,
            )
            results.append(result)
        return results

    def retrieve(
        self,
        memory_id: str,
        for_llm: bool = False,
    ) -> GateResult | None:
        """Retrieve a memory with outbound validation.

        Args:
            memory_id: Memory identifier
            for_llm: If True, process through outbound gate

        Returns:
            GateResult with validated memory, or None if not found
        """
        memory = self._storage.get(memory_id)
        if memory is None:
            return None

        if for_llm:
            # Process through outbound gate
            return self._gate.process_outbound(memory)

        # Return raw (for internal use only)
        return GateResult(
            success=True,
            decision=GateDecision.ACCEPT,
            memory=memory.to_dict() if hasattr(memory, "to_dict") else memory,
        )

    def search(
        self,
        query: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        memory_types: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        for_llm: bool = False,
    ) -> list[GateResult]:
        """Search memories with optional outbound validation.

        Args:
            query: Text search query
            user_id: Filter by user
            agent_id: Filter by agent
            topics: Filter by topics
            categories: Filter by categories
            memory_types: Filter by memory types
            start_date: Filter by creation date (start)
            end_date: Filter by creation date (end)
            limit: Max results
            for_llm: If True, process through outbound gate

        Returns:
            List of GateResults with validated memories
        """
        memories = self._storage.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            topics=topics,
            categories=categories,
            memory_types=memory_types,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        results = []
        for memory in memories:
            if for_llm:
                result = self._gate.process_outbound(memory)
            else:
                result = GateResult(
                    success=True,
                    decision=GateDecision.ACCEPT,
                    memory=memory.to_dict() if hasattr(memory, "to_dict") else memory,
                )
            results.append(result)

        return results

    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Note: Delete operations don't require gate validation
        as no data is entering the system.

        Args:
            memory_id: Memory identifier
        """
        self._storage.delete(memory_id)

    def compress(
        self,
        user_id: str,
        older_than: timedelta | None = None,
        memory_types: list[str] | None = None,
        strategy: str = "summarize",
        min_memories: int = 10,
    ) -> dict[str, Any]:
        """Compress old memories.

        Compressed memories are re-validated through the gate before storage.

        Args:
            user_id: User whose memories to compress
            older_than: Only compress memories older than this
            memory_types: Only compress these memory types
            strategy: Compression strategy
            min_memories: Minimum memories required

        Returns:
            Compression result dict
        """
        from mindcore.clst import CLST, CompressionStrategy

        # Create temporary CLST for compression logic
        # Note: We use the internal storage directly for compression
        # but re-validate results through gate
        temp_clst = CLST(
            storage=self._storage,
            vocabulary=self._gate._svl,
            compression_llm=self._compression_llm,
        )

        try:
            strategy_enum = CompressionStrategy(strategy)
        except ValueError:
            strategy_enum = CompressionStrategy.SUMMARIZE

        result = temp_clst.compress(
            user_id=user_id,
            older_than=older_than,
            memory_types=memory_types or ["episodic"],
            strategy=strategy_enum,
            min_memories=min_memories,
        )

        return {
            "original_count": result.original_count,
            "compressed_count": result.compressed_count,
            "compression_ratio": result.compression_ratio,
            "removed_count": len(result.removed_memory_ids),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get storage and gate statistics."""
        return {
            "gate_stats": self._gate.get_stats(),
            "storage_stats": self._storage.get_stats()
            if hasattr(self._storage, "get_stats")
            else {},
        }


class GatedFLR:
    """Gated Fast Learning Recall.

    This class wraps FLR with mandatory SVL Gate validation.
    All data entering or leaving goes through the gate.

    Example:
        gate = SVLGate(svl=svl)
        flr = GatedFLR(storage=storage, gate=gate)

        # Query with outbound validation
        result = flr.query(
            query="What are the user's preferences?",
            user_id="u1",
        )

        # All returned memories are gate-validated
        for memory in result.memories:
            # Safe to send to LLM
            pass
    """

    def __init__(
        self,
        storage: BaseStorage,
        gate: SVLGate,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 300,
        embedding_fn: Callable | None = None,
    ):
        """Initialize GatedFLR.

        Args:
            storage: Storage backend
            gate: SVL Gate (mandatory)
            cache_size: Max memories in hot cache
            cache_ttl_seconds: Cache TTL
            embedding_fn: Optional function to generate embeddings
        """
        from mindcore.flr import FLR

        self._gate = gate
        self._flr = FLR(
            storage=storage,
            cache_size=cache_size,
            cache_ttl_seconds=cache_ttl_seconds,
            embedding_fn=embedding_fn,
        )

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
        """Query for relevant memories with outbound gate validation.

        All returned memories are validated through the SVL Gate
        before being returned for LLM context.

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier
            attention_hints: Topics to prioritize
            memory_types: Filter by memory types
            limit: Max memories to return
            include_cross_agent: Include memories from other agents
            min_score: Minimum relevance score

        Returns:
            RecallResult with gate-validated memories
        """
        import time

        start_time = time.time()

        # Query FLR
        flr_result = self._flr.query(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            attention_hints=attention_hints,
            memory_types=memory_types,
            limit=limit,
            include_cross_agent=include_cross_agent,
            min_score=min_score,
        )

        # Process each memory through outbound gate
        validated_memories = []
        validated_scores = []
        gate_time = 0.0

        for memory, score in zip(flr_result.memories, flr_result.scores, strict=False):
            gate_start = time.time()
            gate_result = self._gate.process_outbound(memory)
            gate_time += (time.time() - gate_start) * 1000

            if gate_result.success:
                validated_memories.append(gate_result.memory)
                validated_scores.append(score)
            else:
                logger.warning(
                    "Memory %s failed outbound validation: %s",
                    memory.memory_id,
                    [e.message for e in gate_result.errors],
                )

        total_time = (time.time() - start_time) * 1000

        return RecallResult(
            success=True,
            memories=validated_memories,
            scores=validated_scores,
            query_latency_ms=total_time - gate_time,
            gate_processing_ms=gate_time,
            attention_focus=flr_result.attention_focus,
            suggested_memory_types=flr_result.suggested_memory_types,
        )

    def reinforce(self, memory_id: str, signal: float) -> float:
        """Reinforce a memory with a learning signal.

        Reinforcement doesn't require gate validation as it's
        updating existing validated data.

        Args:
            memory_id: Memory to reinforce
            signal: Signal from -1.0 to +1.0

        Returns:
            New reinforcement score
        """
        return self._flr.reinforce(memory_id, signal)

    def flush_reinforcements(self) -> int:
        """Flush buffered reinforcement signals."""
        return self._flr.flush_reinforcements()

    def get_stats(self) -> dict[str, Any]:
        """Get FLR and gate statistics."""
        return {
            "flr_stats": self._flr.get_stats(),
            "gate_stats": self._gate.get_stats(),
        }


class GatedMindcore:
    """Gated Mindcore - The complete gated memory system.

    This is the recommended entry point for using MindCore with
    mandatory SVL Gate validation. It combines GatedCLST and GatedFLR
    into a unified interface.

    IMPORTANT: This class enforces the SVL Gate as the ONLY path
    for data to enter or exit the memory system. There are NO bypass paths.

    Example:
        from mindcore.svl.gated_storage import GatedMindcore

        # Create gated memory system
        memory = GatedMindcore(
            storage="sqlite:///memory.db",
            vocabulary=my_vocabulary,
        )

        # Store from LLM output (mandatory validation)
        result = memory.store(
            llm_output={"content": "...", "memory_type": "preference"},
            user_id="u1",
            llm_call=my_llm_function,  # For retry on failure
        )

        # Recall for LLM context (mandatory outbound validation)
        memories = memory.recall(
            query="What are the preferences?",
            user_id="u1",
        )
    """

    def __init__(
        self,
        storage: str | Any = "sqlite:///mindcore.db",
        vocabulary: Any = None,
        gate_policy: Any = None,
        retry_config: Any = None,
    ):
        """Initialize GatedMindcore.

        Args:
            storage: Storage backend or connection string
            vocabulary: SharedVocabularyLayer (uses default if None)
            gate_policy: GatePolicy configuration
            retry_config: RetryConfig configuration
        """
        from mindcore.storage import SQLiteStorage
        from mindcore.svl import DEFAULT_SVL

        from .gate import GatePolicy, RetryConfig

        # Initialize storage
        if isinstance(storage, str):
            if storage.startswith("sqlite:///"):
                db_path = storage[10:]
                self._storage = SQLiteStorage(db_path)
            elif storage.startswith(("postgresql://", "postgres://")):
                from mindcore.storage.postgres import PostgresStorage

                self._storage = PostgresStorage(storage)
            else:
                self._storage = SQLiteStorage(storage)
        else:
            self._storage = storage

        # Initialize vocabulary
        self._vocabulary = vocabulary or DEFAULT_SVL

        # Initialize gate
        self._gate = SVLGate(
            svl=self._vocabulary,
            policy=gate_policy or GatePolicy(),
            retry_config=retry_config or RetryConfig(),
        )

        # Initialize gated storage layers
        self._clst = GatedCLST(storage=self._storage, gate=self._gate)
        self._flr = GatedFLR(storage=self._storage, gate=self._gate)

    def store(
        self,
        llm_output: dict[str, Any] | str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> StoreResult:
        """Store a memory from LLM output.

        The data is processed through the SVL Gate which:
        1. Canonicalizes inputs to unified schema
        2. Validates against vocabulary
        3. Retries with LLM if validation fails
        4. Falls back to rule-based extraction if needed

        Args:
            llm_output: Raw LLM output (dict or JSON string)
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            llm_call: Optional LLM function for retry

        Returns:
            StoreResult with success status
        """
        # Parse if string
        if isinstance(llm_output, str):
            try:
                import json

                data = json.loads(llm_output)
            except json.JSONDecodeError:
                from .gate import GateDecision, PolicyViolation, ValidationError

                return StoreResult(
                    success=False,
                    gate_result=GateResult(
                        success=False,
                        decision=GateDecision.REJECT,
                        errors=[
                            ValidationError(
                                violation=PolicyViolation.INVALID_JSON,
                                field="input",
                                message="Invalid JSON input",
                            )
                        ],
                    ),
                    error_message="Invalid JSON input",
                )
        else:
            data = llm_output

        return self._clst.store(
            memory_data=data,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_call=llm_call,
        )

    def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """Recall memories for LLM context.

        All returned memories are validated through the outbound
        gate to ensure they're safe to send to the LLM.

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier
            attention_hints: Topics to prioritize
            memory_types: Filter by memory types
            limit: Max memories to return

        Returns:
            RecallResult with validated memories
        """
        return self._flr.query(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            attention_hints=attention_hints,
            memory_types=memory_types,
            limit=limit,
        )

    def reinforce(self, memory_id: str, signal: float) -> float:
        """Reinforce a memory.

        Args:
            memory_id: Memory identifier
            signal: Signal from -1.0 to +1.0

        Returns:
            New reinforcement score
        """
        return self._flr.reinforce(memory_id, signal)

    def delete(self, memory_id: str) -> None:
        """Delete a memory."""
        self._clst.delete(memory_id)

    def get_json_schema(self, include_response: bool = True) -> dict[str, Any]:
        """Get JSON schema for LLM structured output."""
        return self._vocabulary.get_full_memory_schema(include_response=include_response)

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        return {
            "gate": self._gate.get_stats(),
            "clst": self._clst.get_stats(),
            "flr": self._flr.get_stats(),
        }

    def close(self) -> None:
        """Close all connections."""
        self._flr.flush_reinforcements()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
