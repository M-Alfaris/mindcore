"""CrewAI integration for Mindcore.

Provides memory adapters that integrate Mindcore with CrewAI crews and agents.
Updated for CrewAI 2025 patterns.

This module provides two integration approaches:

1. MindcoreRAGStorage - Implements the RAGStorage interface for use as a
   custom storage backend with CrewAI's memory system. This is the recommended
   approach for CrewAI 2025+.

2. MindcoreCrewMemory - Legacy memory interface for backwards compatibility.

Example (Modern - RAGStorage backend):
    from crewai import Crew, Agent
    from crewai.memory import ShortTermMemory, EntityMemory
    from mindcore.integrations import MindcoreRAGStorage

    # Create storage backends
    stm_storage = MindcoreRAGStorage(
        storage="postgresql://localhost/mindcore",
        storage_type="short_term",
    )
    entity_storage = MindcoreRAGStorage(
        storage="postgresql://localhost/mindcore",
        storage_type="entity",
    )

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=True,
        short_term_memory=ShortTermMemory(storage=stm_storage),
        entity_memory=EntityMemory(storage=entity_storage),
    )

Example (Legacy):
    from mindcore.integrations import MindcoreCrewMemory

    memory = MindcoreCrewMemory(
        storage="postgresql://localhost/mindcore",
    )

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=memory,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mindcore import Mindcore


if TYPE_CHECKING:
    from mindcore.flr import Memory


@dataclass
class MindcoreRAGStorage:
    """CrewAI 2025 compatible RAG storage backend backed by Mindcore.

    Implements the RAGStorage interface for use with CrewAI's memory system.
    Can be used as a custom storage backend for ShortTermMemory, EntityMemory,
    and other CrewAI memory types.

    This is the recommended integration for CrewAI 2025+.

    Benefits:
    - PostgreSQL-first with deterministic scoring
    - Semantic search for relevant context retrieval
    - Cross-crew memory sharing
    - Full audit trail
    - No dependency on ChromaDB

    Example:
        from crewai import Crew
        from crewai.memory import ShortTermMemory

        storage = MindcoreRAGStorage(
            storage="postgresql://localhost/mindcore",
            storage_type="short_term",
        )

        crew = Crew(
            agents=[...],
            memory=True,
            short_term_memory=ShortTermMemory(storage=storage),
        )

    Attributes:
        type: Storage type identifier (short_term, entity, long_term)
        allow_reset: Whether reset() is allowed
    """

    storage: str = "sqlite:///mindcore.db"
    storage_type: str = "short_term"  # Renamed from 'type' to avoid builtin shadow
    allow_reset: bool = True
    embedder_config: dict[str, Any] | None = None
    crew: Any | None = None  # CrewAI Crew instance
    crew_id: str = "default_crew"
    _mindcore: Mindcore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize Mindcore connection."""
        self._mindcore = Mindcore(storage=self.storage)

    @property
    def type(self) -> str:
        """Return storage type (RAGStorage interface)."""
        return self.storage_type

    @property
    def mindcore(self) -> Mindcore:
        """Get Mindcore instance."""
        if self._mindcore is None:
            self._mindcore = Mindcore(storage=self.storage)
        return self._mindcore

    def save(
        self,
        value: str,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> str:
        """Save a value to storage (RAGStorage interface).

        Args:
            value: Content to store
            metadata: Optional metadata dict
            agent: Agent name storing the value

        Returns:
            Memory ID
        """
        metadata = metadata or {}

        # Extract topics from metadata if available
        topics = metadata.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]

        # Add storage type as topic
        topics.append(self.storage_type)

        # Map metadata to Mindcore fields
        importance = metadata.get("importance", 0.5)
        memory_type = metadata.get("type", "episodic")

        return self.mindcore.store(
            content=value,
            memory_type=memory_type,
            user_id=self.crew_id,
            topics=topics,
            importance=importance,
            agent_id=agent,
            categories=metadata.get("categories", []),
            entities=metadata.get("entities", []),
        )

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search storage for relevant entries (RAGStorage interface).

        Args:
            query: Search query
            limit: Maximum results to return
            filter: Optional filter dict (e.g., {"agent": "researcher"})
            score_threshold: Minimum relevance score (0.0 - 1.0)

        Returns:
            List of matching entries with content and metadata
        """
        # Perform semantic search
        result = self.mindcore.recall(
            query=query,
            user_id=self.crew_id,
            agent_id=filter.get("agent") if filter else None,
            limit=limit * 2,  # Fetch extra for filtering
        )

        memories = result.memories if hasattr(result, "memories") else []

        # Filter by storage type
        memories = [m for m in memories if self.storage_type in m.topics]

        # Filter by score threshold
        if score_threshold > 0:
            memories = [m for m in memories if m.importance >= score_threshold]

        # Apply limit
        memories = memories[:limit]

        # Format for CrewAI
        results = []
        for mem in memories:
            results.append(
                {
                    "content": mem.content,
                    "score": mem.importance,
                    "metadata": {
                        "memory_id": mem.memory_id,
                        "topics": mem.topics,
                        "importance": mem.importance,
                        "created_at": (mem.created_at.isoformat() if mem.created_at else None),
                        "agent": mem.agent_id,
                    },
                }
            )

        return results

    def reset(self) -> None:
        """Reset/clear all entries in this storage (RAGStorage interface).

        Note: This only clears entries of the current storage_type.
        """
        if not self.allow_reset:
            raise ValueError("Reset is not allowed for this storage")

        memories = self.mindcore.search(
            user_id=self.crew_id,
            topics=[self.storage_type],
            limit=10000,
        )

        for mem in memories:
            try:
                self.mindcore.delete(mem.memory_id)
            except Exception:
                pass


class MindcoreCrewMemory:
    """Legacy CrewAI memory interface backed by Mindcore.

    This class provides backwards compatibility with older CrewAI
    memory patterns. For new implementations, prefer MindcoreRAGStorage
    as a custom storage backend.

    Example:
        memory = MindcoreCrewMemory(
            storage="postgresql://localhost/mindcore",
        )

        crew = Crew(
            agents=[...],
            tasks=[...],
            memory=memory,
        )
    """

    def __init__(
        self,
        storage: str = "sqlite:///mindcore.db",
        crew_id: str = "default_crew",
        enable_cross_crew: bool = False,
    ):
        """Initialize Mindcore memory for CrewAI.

        Args:
            storage: Mindcore storage connection string
            crew_id: Identifier for this crew (used as user_id)
            enable_cross_crew: Enable memory sharing across crews
        """
        self._mindcore = Mindcore(
            storage=storage,
            enable_multi_agent=enable_cross_crew,
        )
        self._crew_id = crew_id
        self._enable_cross_crew = enable_cross_crew

    def save(
        self,
        value: str,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> str:
        """Save a memory (CrewAI interface).

        Args:
            value: Content to store
            metadata: Optional metadata dict
            agent: Agent name storing the memory

        Returns:
            Memory ID
        """
        metadata = metadata or {}

        # Extract topics from metadata if available
        topics = metadata.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]

        # Map CrewAI metadata to Mindcore fields
        importance = metadata.get("importance", 0.5)
        memory_type = metadata.get("type", "episodic")

        return self._mindcore.store(
            content=value,
            memory_type=memory_type,
            user_id=self._crew_id,
            topics=topics,
            importance=importance,
            agent_id=agent,
            categories=metadata.get("categories", []),
            entities=metadata.get("entities", []),
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        agent: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search memories (CrewAI interface).

        Args:
            query: Search query
            limit: Maximum results
            agent: Filter by agent (optional)
            score_threshold: Minimum relevance score

        Returns:
            List of matching memories with content and metadata
        """
        result = self._mindcore.recall(
            query=query,
            user_id=self._crew_id,
            agent_id=agent,
            limit=limit,
        )

        memories = result.memories if hasattr(result, "memories") else []

        # Filter by score threshold if specified
        if score_threshold > 0:
            memories = [m for m in memories if m.importance >= score_threshold]

        # Format for CrewAI
        formatted = []
        for mem in memories:
            formatted.append(
                {
                    "content": mem.content,
                    "metadata": {
                        "memory_id": mem.memory_id,
                        "topics": mem.topics,
                        "importance": mem.importance,
                        "created_at": (mem.created_at.isoformat() if mem.created_at else None),
                        "agent": mem.agent_id,
                    },
                }
            )

        return formatted

    def reset(self) -> None:
        """Reset/clear all memories (CrewAI interface).

        Note: This deletes ALL memories for the crew.
        Use with caution in production.
        """
        memories = self._mindcore.search(
            user_id=self._crew_id,
            limit=10000,
        )
        for mem in memories:
            try:
                self._mindcore.delete(mem.memory_id)
            except Exception:
                pass

    # Additional CrewAI-specific methods

    def kickoff_memory(
        self,
        task_description: str,
        expected_output: str,
        agent: str,
    ) -> str:
        """Store task kickoff context.

        Args:
            task_description: What the task is about
            expected_output: Expected result
            agent: Agent assigned to task

        Returns:
            Memory ID
        """
        content = f"Task: {task_description}\nExpected: {expected_output}"
        return self._mindcore.store(
            content=content,
            memory_type="working",
            user_id=self._crew_id,
            topics=["task", "kickoff"],
            importance=0.7,
            agent_id=agent,
        )

    def task_output_memory(
        self,
        task_description: str,
        output: str,
        agent: str,
    ) -> str:
        """Store task output/result.

        Args:
            task_description: What the task was
            output: Task result
            agent: Agent that completed task

        Returns:
            Memory ID
        """
        content = f"Completed: {task_description}\nResult: {output}"
        return self._mindcore.store(
            content=content,
            memory_type="episodic",
            user_id=self._crew_id,
            topics=["task", "output"],
            importance=0.8,
            agent_id=agent,
        )

    def agent_context(
        self,
        agent: str,
        limit: int = 5,
    ) -> list[Memory]:
        """Get recent context for an agent.

        Args:
            agent: Agent name
            limit: Number of memories

        Returns:
            Recent memories for the agent
        """
        memories = self._mindcore.search(
            user_id=self._crew_id,
            limit=limit,
        )

        # Filter by agent if needed
        if agent:
            memories = [m for m in memories if m.agent_id == agent]

        return memories

    def share_across_crews(
        self,
        memory_id: str,
        target_crew_id: str,
    ) -> str | None:
        """Share a memory with another crew.

        Requires enable_cross_crew=True.

        Args:
            memory_id: Memory to share
            target_crew_id: Target crew identifier

        Returns:
            New memory ID in target crew, or None if failed
        """
        if not self._enable_cross_crew:
            raise ValueError("Cross-crew sharing not enabled")

        # Get original memory
        memory = self._mindcore.get(memory_id)
        if not memory:
            return None

        # Store copy in target crew
        return self._mindcore.store(
            content=memory.content,
            memory_type=memory.memory_type,
            user_id=target_crew_id,
            topics=memory.topics,
            importance=memory.importance,
            categories=memory.categories,
        )
