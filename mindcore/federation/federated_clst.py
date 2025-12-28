"""Federated Cold Long-term Storage (CLST).

Provides shared CLST across multiple agents with access control
and namespace-based isolation.

Features:
- Namespace-scoped storage and retrieval
- Access control enforcement on all operations
- Cross-namespace visibility with proper permissions
- Aggregated reinforcement from multiple agents

Example:
    # Create federated CLST
    clst = FederatedCLST(
        org_id="acme-corp",
        storage=pinecone_storage,  # Underlying vector store
    )

    # Store with namespace and access control
    clst.store(
        memory=memory,
        namespace=team_namespace,
        access_level=AccessLevel.TEAM,
        agent_scope=agent_scope,
    )

    # Search with namespace filtering
    results = clst.search(
        query="billing issues",
        requester=agent_scope,
        namespaces=[team_namespace],
        include_ancestors=True,  # Also search department/org level
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .access_control import (
    AccessLevel,
    AccessPolicy,
    AccessScope,
    MemoryACL,
)
from .namespace import MemoryNamespace


class StorageBackend(Protocol):
    """Protocol for underlying vector storage."""

    def store(
        self,
        memory_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store a vector with metadata."""
        ...

    def search(
        self,
        query_vector: list[float],
        filter: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        ...

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID."""
        ...

    def update_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Update metadata for a memory."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        ...


@dataclass
class NamespacedQuery:
    """Query configuration for namespace-aware search.

    Attributes:
        query_vector: Embedding vector for semantic search
        namespaces: Namespaces to search in
        requester: Scope of the requesting agent
        include_ancestors: Include parent namespaces in search
        include_descendants: Include child namespaces (requires permission)
        access_levels: Filter by access levels
        limit: Maximum results to return
        min_score: Minimum similarity score threshold
    """

    query_vector: list[float]
    namespaces: list[MemoryNamespace]
    requester: AccessScope
    include_ancestors: bool = True
    include_descendants: bool = False
    access_levels: list[AccessLevel] | None = None
    limit: int = 10
    min_score: float = 0.0
    time_range: tuple[datetime, datetime] | None = None


@dataclass
class FederatedMemory:
    """Memory with federation metadata.

    Extends base memory with namespace and ACL information.
    """

    memory_id: str
    content: str
    vector: list[float]
    namespace: MemoryNamespace
    acl: MemoryACL
    metadata: dict[str, Any] = field(default_factory=dict)

    # Aggregated reinforcement from multiple agents
    aggregated_reinforcement: float = 0.0
    reinforcement_sources: dict[str, float] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to storage format."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "namespace": self.namespace.to_dict(),
            "acl": self.acl.to_dict(),
            "metadata": self.metadata,
            "aggregated_reinforcement": self.aggregated_reinforcement,
            "reinforcement_sources": self.reinforcement_sources,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class FederatedCLST:
    """Federated Cold Long-term Storage.

    Provides organization-wide memory storage with:
    - Namespace-based isolation
    - Access control enforcement
    - Cross-agent reinforcement aggregation

    Attributes:
        org_id: Organization identifier
        storage: Underlying vector storage backend
        acl_cache: Cache of ACLs (optional, for performance)
    """

    org_id: str
    storage: StorageBackend
    acl_store: dict[str, MemoryACL] = field(default_factory=dict)

    def store(
        self,
        memory_id: str,
        content: str,
        vector: list[float],
        namespace: MemoryNamespace,
        owner_agent_id: str,
        owner_scope: AccessScope,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        policy: AccessPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FederatedMemory:
        """Store a memory with namespace and access control.

        Args:
            memory_id: Unique memory identifier
            content: Memory content
            vector: Embedding vector
            namespace: Storage namespace
            owner_agent_id: ID of the creating agent
            owner_scope: Scope of the owner
            access_level: Default access level
            policy: Custom access policy (overrides access_level)
            metadata: Additional metadata

        Returns:
            Stored FederatedMemory
        """
        # Validate namespace belongs to org
        if namespace.org_id != self.org_id:
            raise ValueError(
                f"Namespace org {namespace.org_id} doesn't match CLST org {self.org_id}"
            )

        # Create ACL
        if policy is None:
            policy = AccessPolicy(access_level=access_level)

        acl = MemoryACL(
            memory_id=memory_id,
            owner_agent_id=owner_agent_id,
            owner_scope=owner_scope,
            policy=policy,
        )

        # Create federated memory
        memory = FederatedMemory(
            memory_id=memory_id,
            content=content,
            vector=vector,
            namespace=namespace,
            acl=acl,
            metadata=metadata or {},
        )

        # Store in backend with full metadata
        storage_metadata = memory.to_storage_dict()
        storage_metadata["vector"] = None  # Don't duplicate vector in metadata
        self.storage.store(
            memory_id=memory_id,
            vector=vector,
            metadata=storage_metadata,
        )

        # Cache ACL
        self.acl_store[memory_id] = acl

        return memory

    def search(
        self,
        query: NamespacedQuery,
    ) -> list[FederatedMemory]:
        """Search for memories with namespace and access control.

        Args:
            query: Namespaced query configuration

        Returns:
            List of accessible memories matching the query
        """
        # Build namespace filter
        namespace_paths = set()
        for ns in query.namespaces:
            namespace_paths.add(ns.path)
            if query.include_ancestors:
                for ancestor in ns.get_ancestors():
                    namespace_paths.add(ancestor.path)

        # Build storage filter
        storage_filter: dict[str, Any] = {
            "org_id": self.org_id,
            "namespace_path": {"$in": list(namespace_paths)},
        }

        if query.access_levels:
            storage_filter["access_level"] = {"$in": [level.value for level in query.access_levels]}

        if query.time_range:
            storage_filter["created_at"] = {
                "$gte": query.time_range[0].isoformat(),
                "$lte": query.time_range[1].isoformat(),
            }

        # Search storage
        results = self.storage.search(
            query_vector=query.query_vector,
            filter=storage_filter,
            limit=query.limit * 2,  # Over-fetch for ACL filtering
        )

        # Filter by ACL and build memories
        accessible_memories: list[FederatedMemory] = []

        for result in results:
            # Check similarity threshold
            score = result.get("score", 1.0)
            if score < query.min_score:
                continue

            # Get or reconstruct ACL
            memory_id = result["memory_id"]
            acl = self._get_acl(memory_id, result)

            # Check access
            if not acl.can_read(query.requester):
                continue

            # Record access
            acl.record_access(query.requester.agent_id or "unknown")

            # Build federated memory
            memory = FederatedMemory(
                memory_id=memory_id,
                content=result.get("content", ""),
                vector=result.get("vector", []),
                namespace=MemoryNamespace.from_dict(
                    result.get("namespace", {"org_id": self.org_id})
                ),
                acl=acl,
                metadata=result.get("metadata", {}),
                aggregated_reinforcement=result.get("aggregated_reinforcement", 0.0),
                reinforcement_sources=result.get("reinforcement_sources", {}),
            )

            accessible_memories.append(memory)

            if len(accessible_memories) >= query.limit:
                break

        return accessible_memories

    def get(
        self,
        memory_id: str,
        requester: AccessScope,
    ) -> FederatedMemory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory identifier
            requester: Scope of the requesting agent

        Returns:
            Memory if accessible, None otherwise
        """
        result = self.storage.get(memory_id)
        if result is None:
            return None

        acl = self._get_acl(memory_id, result)
        if not acl.can_read(requester):
            return None

        acl.record_access(requester.agent_id or "unknown")

        return FederatedMemory(
            memory_id=memory_id,
            content=result.get("content", ""),
            vector=result.get("vector", []),
            namespace=MemoryNamespace.from_dict(result.get("namespace", {"org_id": self.org_id})),
            acl=acl,
            metadata=result.get("metadata", {}),
            aggregated_reinforcement=result.get("aggregated_reinforcement", 0.0),
            reinforcement_sources=result.get("reinforcement_sources", {}),
        )

    def apply_reinforcement(
        self,
        memory_id: str,
        agent_id: str,
        signal: float,
        requester: AccessScope,
    ) -> float | None:
        """Apply reinforcement signal from an agent.

        Signals are aggregated across all agents with appropriate weighting.

        Args:
            memory_id: Memory to reinforce
            agent_id: ID of the reinforcing agent
            signal: Reinforcement signal (-1 to 1)
            requester: Scope of the requesting agent

        Returns:
            New aggregated reinforcement score, or None if not accessible
        """
        result = self.storage.get(memory_id)
        if result is None:
            return None

        acl = self._get_acl(memory_id, result)
        if not acl.can_read(requester):
            return None

        # Get current reinforcement state
        sources = result.get("reinforcement_sources", {})
        sources[agent_id] = signal

        # Aggregate signals (simple average for now, can be weighted)
        aggregated = sum(sources.values()) / len(sources) if sources else 0.0

        # Update storage
        self.storage.update_metadata(
            memory_id=memory_id,
            metadata={
                "reinforcement_sources": sources,
                "aggregated_reinforcement": aggregated,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return aggregated

    def update_access(
        self,
        memory_id: str,
        new_policy: AccessPolicy,
        requester: AccessScope,
    ) -> bool:
        """Update access policy for a memory.

        Only the owner or agents with write access can update.

        Args:
            memory_id: Memory to update
            new_policy: New access policy
            requester: Scope of the requesting agent

        Returns:
            True if updated successfully
        """
        result = self.storage.get(memory_id)
        if result is None:
            return False

        acl = self._get_acl(memory_id, result)
        if not acl.can_write(requester):
            return False

        # Update ACL
        acl.policy = new_policy
        self.acl_store[memory_id] = acl

        # Update storage
        self.storage.update_metadata(
            memory_id=memory_id,
            metadata={
                "acl": acl.to_dict(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return True

    def delete(
        self,
        memory_id: str,
        requester: AccessScope,
    ) -> bool:
        """Delete a memory.

        Only the owner or agents with write access can delete.

        Args:
            memory_id: Memory to delete
            requester: Scope of the requesting agent

        Returns:
            True if deleted successfully
        """
        result = self.storage.get(memory_id)
        if result is None:
            return False

        acl = self._get_acl(memory_id, result)
        if not acl.can_write(requester):
            return False

        # Remove from storage and cache
        self.storage.delete(memory_id)
        self.acl_store.pop(memory_id, None)

        return True

    def get_namespace_stats(
        self,
        namespace: MemoryNamespace,
        requester: AccessScope,
    ) -> dict[str, Any]:
        """Get statistics for a namespace.

        Args:
            namespace: Namespace to analyze
            requester: Scope of the requesting agent

        Returns:
            Statistics including count, access distribution, etc.
        """
        # This would typically query an aggregation endpoint
        # Simplified implementation
        return {
            "namespace": namespace.path,
            "total_memories": 0,  # Would be computed
            "by_access_level": {},
            "by_agent": {},
            "avg_reinforcement": 0.0,
        }

    def _get_acl(
        self,
        memory_id: str,
        result: dict[str, Any],
    ) -> MemoryACL:
        """Get ACL from cache or reconstruct from storage."""
        if memory_id in self.acl_store:
            return self.acl_store[memory_id]

        acl_data = result.get("acl")
        if acl_data:
            acl = MemoryACL.from_dict(acl_data)
            self.acl_store[memory_id] = acl
            return acl

        # Fallback: create default ACL
        owner_scope = AccessScope(
            org_id=self.org_id,
            agent_id=result.get("owner_agent_id", "unknown"),
        )
        return MemoryACL(
            memory_id=memory_id,
            owner_agent_id=result.get("owner_agent_id", "unknown"),
            owner_scope=owner_scope,
            policy=AccessPolicy(access_level=AccessLevel.PRIVATE),
        )
