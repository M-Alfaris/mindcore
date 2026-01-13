"""CrewAI integration for Mindcore.

Provides memory adapters that integrate Mindcore with CrewAI crews and agents.

Example:
    from crewai import Crew, Agent, Task
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

from typing import Any

from mindcore import Mindcore


class MindcoreCrewMemory:
    """CrewAI-compatible memory backed by Mindcore.

    Implements the CrewAI memory interface while delegating all
    storage and retrieval to Mindcore's FLR/CLST protocols.

    Benefits over CrewAI's built-in memory:
    - PostgreSQL-first with deterministic scoring
    - Cross-crew memory sharing via CLST
    - Session aggregates for hierarchical retrieval
    - Full audit trail for compliance

    Example:
        memory = MindcoreCrewMemory(
            storage="postgresql://localhost/mindcore",
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
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
            memories = [m for m in memories if m.get("importance", 0) >= score_threshold]

        # Format for CrewAI
        formatted = []
        for mem in memories:
            formatted.append(
                {
                    "content": mem.get("content", ""),
                    "metadata": {
                        "memory_id": mem.get("memory_id"),
                        "topics": mem.get("topics", []),
                        "importance": mem.get("importance", 0.5),
                        "created_at": mem.get("created_at"),
                        "agent": mem.get("agent_id"),
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
                self._mindcore.delete(mem["memory_id"])
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
    ) -> list[dict[str, Any]]:
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
            memories = [m for m in memories if m.get("agent_id") == agent]

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
            content=memory["content"],
            memory_type=memory.get("memory_type", "episodic"),
            user_id=target_crew_id,
            topics=memory.get("topics", []),
            importance=memory.get("importance", 0.5),
            categories=memory.get("categories", []),
        )
