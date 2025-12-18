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
