"""Base storage interface for Mindcore v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..flr import Memory


class BaseStorage(ABC):
    """Abstract base class for memory storage backends.

    All storage backends (SQLite, PostgreSQL, etc.) must implement this interface.
    This ensures FLR and CLST can work with any storage backend.
    """

    @abstractmethod
    def store(self, memory: Memory) -> str:
        """Store a memory.

        Args:
            memory: Memory to store

        Returns:
            Memory ID
        """
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory or None if not found
        """
        pass

    @abstractmethod
    def update(self, memory: Memory) -> bool:
        """Update an existing memory.

        Args:
            memory: Memory with updated fields

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        pass

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
        pass

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
        pass

    @abstractmethod
    def update_reinforcement(self, memory_id: str, signal: float) -> bool:
        """Update reinforcement score for a memory.

        Args:
            memory_id: Memory identifier
            signal: Reinforcement signal to add

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    def store_transfer(self, transfer_id: str, data: list[dict]) -> None:
        """Store transfer data for cross-instance transfers.

        Args:
            transfer_id: Transfer identifier
            data: Serialized memory data
        """
        pass

    @abstractmethod
    def get_transfer(self, transfer_id: str) -> list[dict] | None:
        """Retrieve transfer data.

        Args:
            transfer_id: Transfer identifier

        Returns:
            Serialized memory data or None
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with storage stats
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        pass
