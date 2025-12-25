"""Base storage interface for Mindcore v2.

Storage backends follow the "fail hard" philosophy:
- Operations raise exceptions on failure rather than returning False/None
- MemoryNotFoundError is raised when a memory doesn't exist
- StorageError is raised for connection/database issues
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import datetime

    from mindcore.v2.clst.aggregates import SessionAggregate
    from mindcore.v2.flr import Memory


class BaseStorage(ABC):
    """Abstract base class for memory storage backends.

    All storage backends (SQLite, PostgreSQL, etc.) must implement this interface.
    This ensures FLR and CLST can work with any storage backend.

    Error Handling:
        All methods that can fail raise exceptions rather than returning
        False/None. This provides predictable behavior and clear error messages.

        - MemoryNotFoundError: When a memory_id doesn't exist
        - StorageError: For database/connection issues
        - ValueError: For invalid parameters
    """

    @abstractmethod
    def store(self, memory: Memory) -> str:
        """Store a memory.

        Args:
            memory: Memory to store

        Returns:
            Memory ID
        """

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory or None if not found
        """

    @abstractmethod
    def update(self, memory: Memory) -> None:
        """Update an existing memory.

        Args:
            memory: Memory with updated fields

        Raises:
            MemoryNotFoundError: If memory doesn't exist
            StorageError: If update fails
        """

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Args:
            memory_id: Memory identifier

        Raises:
            MemoryNotFoundError: If memory doesn't exist
            StorageError: If deletion fails
        """

    @abstractmethod
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
        min_importance: float | None = None,
        access_levels: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories with filters.

        Args:
            query: Text search query
            user_id: Filter by user
            agent_id: Filter by agent
            topics: Filter by topics (OR match)
            categories: Filter by categories (OR match)
            memory_types: Filter by memory types
            start_date: Filter by creation date (start)
            end_date: Filter by creation date (end)
            min_importance: Minimum importance score
            access_levels: Filter by access levels
            limit: Max results
            offset: Offset for pagination

        Returns:
            List of matching memories
        """

    @abstractmethod
    def search_by_version(
        self,
        version: str,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by vocabulary version.

        Args:
            version: Vocabulary version
            user_id: Optional user filter
            limit: Max results
            offset: Offset for pagination

        Returns:
            List of memories with matching version
        """

    @abstractmethod
    def update_reinforcement(self, memory_id: str, signal: float) -> None:
        """Update reinforcement score for a memory.

        The reinforcement score is bounded to [-1.0, 1.0] to prevent
        unbounded accumulation. Implementations should clamp the final
        score to this range.

        Args:
            memory_id: Memory identifier
            signal: Reinforcement signal to add (will be clamped)

        Raises:
            MemoryNotFoundError: If memory doesn't exist
            ValueError: If signal is not a valid number
        """

    @abstractmethod
    def store_transfer(self, transfer_id: str, data: list[dict]) -> None:
        """Store transfer data for cross-instance transfers.

        Args:
            transfer_id: Transfer identifier
            data: Serialized memory data
        """

    @abstractmethod
    def get_transfer(self, transfer_id: str) -> list[dict] | None:
        """Retrieve transfer data.

        Args:
            transfer_id: Transfer identifier

        Returns:
            Serialized memory data or None
        """

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with storage stats
        """

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""

    # ==========================================================================
    # Session Aggregate Methods (for hierarchical retrieval)
    # ==========================================================================

    def store_session_aggregate(self, aggregate: SessionAggregate) -> str:
        """Store or update a session aggregate.

        Args:
            aggregate: SessionAggregate to store

        Returns:
            Session ID
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")

    def get_session_aggregate(self, session_id: str) -> SessionAggregate | None:
        """Retrieve a session aggregate by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionAggregate or None if not found
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")

    def query_sessions(
        self,
        user_id: str,
        topic_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        min_importance_avg: float | None = None,
        min_topic_weight: float = 0.0,
        agent_ids: list[str] | None = None,
        access_levels: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SessionAggregate]:
        """Query sessions by weighted metadata.

        This is the primary method for hierarchical retrieval - find relevant
        sessions first, then query memories from those sessions.

        Args:
            user_id: Filter by user
            topic_hints: Topics to match (scored by weight)
            category_hints: Categories to match (scored by weight)
            min_importance_avg: Minimum average importance
            min_topic_weight: Minimum weight for topic matches
            agent_ids: Filter by agents
            access_levels: Filter by access levels
            start_date: Filter by session start date
            end_date: Filter by session end date
            limit: Max results
            offset: Offset for pagination

        Returns:
            List of SessionAggregates ordered by relevance
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")

    def query_memories_by_sessions(
        self,
        session_ids: list[str],
        min_importance: float | None = None,
        min_confidence: float | None = None,
        memory_types: list[str] | None = None,
        limit: int = 100,
        order_by_message_index: bool = True,
    ) -> list[Memory]:
        """Query memories from specific sessions.

        Used after query_sessions() to get actual memories from relevant sessions.

        Args:
            session_ids: Sessions to query from
            min_importance: Minimum importance filter
            min_confidence: Minimum confidence filter
            memory_types: Filter by memory types
            limit: Max results
            order_by_message_index: Preserve event order within sessions

        Returns:
            List of memories ordered by session and message_index
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")

    def get_next_message_index(self, session_id: str) -> int:
        """Get the next message index for a session.

        Used to maintain event ordering within sessions.

        Args:
            session_id: Session identifier

        Returns:
            Next available message index
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")

    def update_session_aggregate_from_memory(
        self,
        session_id: str,
        memory: Memory,
    ) -> None:
        """Update session aggregate incrementally from a new memory.

        Called automatically when storing memories with session_id.

        Args:
            session_id: Session to update
            memory: New memory that was added
        """
        raise NotImplementedError("Session aggregates not supported by this storage backend")
