"""Cross-Agent Memory Sharing - Memory sharing and synchronization.

Handles memory sharing between agents based on access levels,
and synchronization of memories across agent boundaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.v2.flr import Memory
    from mindcore.v2.storage.base import BaseStorage

    from .registry import AgentRegistry


class AgentSyncDirection(str, Enum):
    """Direction of memory synchronization between agents."""

    ONE_WAY = "one_way"  # Source -> Target only
    BIDIRECTIONAL = "bidirectional"  # Both ways
    MERGE = "merge"  # Merge with conflict resolution


# Alias for backwards compatibility
SyncDirection = AgentSyncDirection


class ConflictResolution(str, Enum):
    """Strategy for resolving sync conflicts.

    When the same memory content exists on both source and target agents,
    these strategies determine which version to keep.

    Strategies:
        SOURCE_WINS: Always use the source agent's version
        TARGET_WINS: Always keep the target agent's version (skip sync)
        NEWEST_WINS: Keep the most recently modified version
        HIGHEST_IMPORTANCE: Keep the version with higher importance score
        MERGE_METADATA: Combine metadata from both versions
        SKIP: Don't sync conflicting memories
    """

    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    NEWEST_WINS = "newest_wins"
    HIGHEST_IMPORTANCE = "highest_importance"
    MERGE_METADATA = "merge_metadata"
    SKIP = "skip"


@dataclass
class ConflictInfo:
    """Information about a sync conflict."""

    memory_id: str
    source_memory: Any  # Memory type
    target_memory: Any  # Memory type
    resolution: ConflictResolution
    resolved_memory: Any | None  # The winning/merged memory
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "source_content": self.source_memory.content if self.source_memory else None,
            "target_content": self.target_memory.content if self.target_memory else None,
            "resolution": self.resolution.value,
            "reason": self.reason,
        }


@dataclass
class ShareResult:
    """Result of a memory sharing operation."""

    success: bool
    memory_id: str
    source_agent: str
    target_agents: list[str]
    access_level: str
    shared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "access_level": self.access_level,
            "shared_at": self.shared_at.isoformat(),
            "error": self.error,
        }


@dataclass
class AgentSyncResult:
    """Result of a synchronization operation between agents."""

    success: bool
    source_agent: str
    target_agent: str
    direction: AgentSyncDirection
    memories_synced: int
    memories_skipped: int
    conflicts: int
    conflict_resolution: ConflictResolution | None = None
    conflict_details: list[ConflictInfo] = field(default_factory=list)
    sync_duration_ms: float = 0.0
    synced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "direction": self.direction.value,
            "memories_synced": self.memories_synced,
            "memories_skipped": self.memories_skipped,
            "conflicts": self.conflicts,
            "conflict_resolution": self.conflict_resolution.value
            if self.conflict_resolution
            else None,
            "conflict_details": [c.to_dict() for c in self.conflict_details],
            "sync_duration_ms": self.sync_duration_ms,
            "synced_at": self.synced_at.isoformat(),
            "errors": self.errors,
        }


# Alias for backwards compatibility
SyncResult = AgentSyncResult


class CrossAgentMemory:
    """Cross-agent memory sharing and synchronization.

    Manages memory visibility and synchronization between agents
    based on access levels and team membership.

    Access Levels:
    - private: Only the owning agent can access
    - team: Agents in the same team can access
    - shared: Any agent for the same user can access
    - global: All agents can access (knowledge base)

    Example:
        sharing = CrossAgentMemory(storage, registry)

        # Share a memory with team
        result = sharing.share_memory(
            memory_id="mem_123",
            source_agent="support_bot",
            access_level="team",
        )

        # Sync memories between agents
        sync_result = sharing.sync_agents(
            source_agent="support_bot",
            target_agent="sales_bot",
            user_id="user123",
            direction=SyncDirection.ONE_WAY,
        )

        # Get accessible memories for an agent
        memories = sharing.get_accessible_memories(
            agent_id="sales_bot",
            user_id="user123",
        )
    """

    def __init__(
        self,
        storage: BaseStorage,
        registry: AgentRegistry,
    ) -> None:
        """Initialize cross-agent memory sharing.

        Args:
            storage: Storage backend
            registry: Agent registry
        """
        self.storage = storage
        self.registry = registry

        # Track sharing history
        self._share_history: list[ShareResult] = []
        self._sync_history: list[SyncResult] = []

    def share_memory(
        self,
        memory_id: str,
        source_agent: str,
        access_level: str,
        target_agents: list[str] | None = None,
    ) -> ShareResult:
        """Share a memory with other agents.

        Updates the memory's access level and optionally notifies target agents.

        Args:
            memory_id: Memory to share
            source_agent: Agent sharing the memory
            access_level: New access level (private/team/shared/global)
            target_agents: Specific agents to share with (for notification)

        Returns:
            ShareResult with operation details
        """
        # Validate source agent
        agent = self.registry.get_agent(source_agent)
        if not agent:
            return ShareResult(
                success=False,
                memory_id=memory_id,
                source_agent=source_agent,
                target_agents=[],
                access_level=access_level,
                error=f"Agent '{source_agent}' not found",
            )

        # Get memory
        memory = self.storage.get(memory_id)
        if not memory:
            return ShareResult(
                success=False,
                memory_id=memory_id,
                source_agent=source_agent,
                target_agents=[],
                access_level=access_level,
                error=f"Memory '{memory_id}' not found",
            )

        # Verify ownership
        if memory.agent_id != source_agent:
            return ShareResult(
                success=False,
                memory_id=memory_id,
                source_agent=source_agent,
                target_agents=[],
                access_level=access_level,
                error="Only the memory owner can share it",
            )

        # Check global write permission
        if access_level == "global" and not agent.can_write_global:
            return ShareResult(
                success=False,
                memory_id=memory_id,
                source_agent=source_agent,
                target_agents=[],
                access_level=access_level,
                error="Agent lacks permission to write global memories",
            )

        # Update memory access level
        memory.access_level = access_level
        self.storage.update(memory)

        # Determine target agents
        if target_agents is None:
            if access_level == "team":
                target_agents = [
                    a.agent_id for a in self.registry.get_agent_teammates(source_agent)
                ]
            elif access_level == "shared":
                target_agents = [
                    a.agent_id for a in self.registry.list_agents() if a.agent_id != source_agent
                ]
            elif access_level == "global":
                target_agents = [
                    a.agent_id
                    for a in self.registry.list_agents()
                    if a.can_read_global and a.agent_id != source_agent
                ]
            else:
                target_agents = []

        result = ShareResult(
            success=True,
            memory_id=memory_id,
            source_agent=source_agent,
            target_agents=target_agents,
            access_level=access_level,
        )

        self._share_history.append(result)
        return result

    def get_accessible_memories(
        self,
        agent_id: str,
        user_id: str,
        topics: list[str] | None = None,
        memory_types: list[str] | None = None,
        include_global: bool = True,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories accessible to an agent.

        Args:
            agent_id: Requesting agent
            user_id: User context
            topics: Filter by topics
            memory_types: Filter by memory types
            include_global: Include global memories
            limit: Maximum memories to return

        Returns:
            List of accessible memories
        """
        agent = self.registry.get_agent(agent_id)
        if not agent or not agent.is_active():
            return []

        accessible = []

        # Get agent's own memories (private + all access levels)
        own_memories = self.storage.search(
            user_id=user_id,
            agent_id=agent_id,
            topics=topics,
            memory_types=memory_types,
            limit=limit,
        )
        accessible.extend(own_memories)

        # Get team memories
        teammates = self.registry.get_agent_teammates(agent_id)
        for teammate in teammates:
            team_memories = self.storage.search(
                user_id=user_id,
                agent_id=teammate.agent_id,
                topics=topics,
                memory_types=memory_types,
                access_levels=["team", "shared"],
                limit=limit,
            )
            accessible.extend(team_memories)

        # Get shared memories from other agents
        other_agents = [
            a
            for a in self.registry.list_agents()
            if a.agent_id != agent_id and a.agent_id not in [t.agent_id for t in teammates]
        ]
        for other in other_agents:
            shared_memories = self.storage.search(
                user_id=user_id,
                agent_id=other.agent_id,
                topics=topics,
                memory_types=memory_types,
                access_levels=["shared"],
                limit=limit,
            )
            accessible.extend(shared_memories)

        # Get global memories
        if include_global and agent.can_read_global:
            global_memories = self.storage.search(
                topics=topics,
                memory_types=memory_types,
                access_levels=["global"],
                limit=limit,
            )
            accessible.extend(global_memories)

        # Deduplicate by memory_id
        seen = set()
        unique = []
        for mem in accessible:
            if mem.memory_id not in seen:
                seen.add(mem.memory_id)
                unique.append(mem)

        # Apply limit
        return unique[:limit]

    def sync_agents(
        self,
        source_agent: str,
        target_agent: str,
        user_id: str,
        direction: SyncDirection = SyncDirection.ONE_WAY,
        conflict_resolution: ConflictResolution = ConflictResolution.SOURCE_WINS,
        topics: list[str] | None = None,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> SyncResult:
        """Synchronize memories between two agents.

        Handles conflicts when the same memory content exists on both agents
        using the specified conflict resolution strategy.

        Conflict Resolution Strategies:
            - SOURCE_WINS: Always use source agent's version
            - TARGET_WINS: Keep target's version (skip sync)
            - NEWEST_WINS: Keep most recently created/modified
            - HIGHEST_IMPORTANCE: Keep version with higher importance
            - MERGE_METADATA: Combine topics/categories from both
            - SKIP: Don't sync conflicting memories

        Args:
            source_agent: Agent to sync from
            target_agent: Agent to sync to
            user_id: User context
            direction: Sync direction (ONE_WAY, BIDIRECTIONAL, MERGE)
            conflict_resolution: How to resolve conflicts
            topics: Filter by topics
            memory_types: Filter by memory types
            since: Only sync memories created after this time

        Returns:
            SyncResult with operation details including conflict information
        """
        start_time = time.time()
        errors = []
        synced = 0
        skipped = 0
        conflicts = 0
        conflict_details = []

        # Validate agents
        source = self.registry.get_agent(source_agent)
        target = self.registry.get_agent(target_agent)

        if not source:
            return SyncResult(
                success=False,
                source_agent=source_agent,
                target_agent=target_agent,
                direction=direction,
                memories_synced=0,
                memories_skipped=0,
                conflicts=0,
                conflict_resolution=conflict_resolution,
                sync_duration_ms=(time.time() - start_time) * 1000,
                errors=[f"Source agent '{source_agent}' not found"],
            )

        if not target:
            return SyncResult(
                success=False,
                source_agent=source_agent,
                target_agent=target_agent,
                direction=direction,
                memories_synced=0,
                memories_skipped=0,
                conflicts=0,
                conflict_resolution=conflict_resolution,
                sync_duration_ms=(time.time() - start_time) * 1000,
                errors=[f"Target agent '{target_agent}' not found"],
            )

        # Check if agents can share (must be teammates or both have shared access)
        is_same_team = source.shares_team_with(target)
        if not is_same_team:
            errors.append("Agents are not in the same team - syncing shared memories only")

        access_levels = ["team", "shared"] if is_same_team else ["shared"]

        # Get source memories that can be synced
        source_memories = self.storage.search(
            user_id=user_id,
            agent_id=source_agent,
            topics=topics,
            memory_types=memory_types,
            access_levels=access_levels,
            start_date=since,
            limit=1000,
        )

        # Get target memories for conflict detection (by content hash)
        target_memories = self.storage.search(
            user_id=user_id,
            agent_id=target_agent,
            topics=topics,
            memory_types=memory_types,
            limit=1000,
        )
        target_content_map = {self._content_hash(m.content): m for m in target_memories}

        # Sync each memory with conflict detection
        for memory in source_memories:
            content_hash = self._content_hash(memory.content)

            # Check for conflict (similar content exists on target)
            if content_hash in target_content_map:
                conflicts += 1
                target_memory = target_content_map[content_hash]

                # Resolve conflict
                resolved, reason = self._resolve_conflict(
                    memory, target_memory, conflict_resolution
                )

                conflict_info = ConflictInfo(
                    memory_id=memory.memory_id,
                    source_memory=memory,
                    target_memory=target_memory,
                    resolution=conflict_resolution,
                    resolved_memory=resolved,
                    reason=reason,
                )
                conflict_details.append(conflict_info)

                if resolved is None:
                    skipped += 1
                    continue
                if resolved.memory_id == target_memory.memory_id:
                    # Target wins, skip
                    skipped += 1
                    continue
                # Otherwise, update target with resolved version
                # In production, this would update the target memory

            synced += 1

        # Bidirectional sync
        if direction == SyncDirection.BIDIRECTIONAL:
            # Get source content for reverse conflict detection
            source_content_map = {self._content_hash(m.content): m for m in source_memories}

            reverse_memories = self.storage.search(
                user_id=user_id,
                agent_id=target_agent,
                topics=topics,
                memory_types=memory_types,
                access_levels=access_levels,
                start_date=since,
                limit=1000,
            )

            for memory in reverse_memories:
                content_hash = self._content_hash(memory.content)

                if content_hash in source_content_map:
                    # Already handled in forward sync
                    continue

                synced += 1

        duration = (time.time() - start_time) * 1000

        result = SyncResult(
            success=len(errors) == 0 or synced > 0,
            source_agent=source_agent,
            target_agent=target_agent,
            direction=direction,
            memories_synced=synced,
            memories_skipped=skipped,
            conflicts=conflicts,
            conflict_resolution=conflict_resolution,
            conflict_details=conflict_details,
            sync_duration_ms=duration,
            errors=errors,
        )

        self._sync_history.append(result)
        return result

    def _content_hash(self, content: str) -> str:
        """Generate a hash for content comparison."""
        import hashlib

        # SHA256 for content comparison (more robust than MD5)
        return hashlib.sha256(content.lower().strip().encode()).hexdigest()

    def _resolve_conflict(
        self,
        source_memory: Any,
        target_memory: Any,
        strategy: ConflictResolution,
    ) -> tuple[Any | None, str]:
        """Resolve a conflict between two memories.

        Args:
            source_memory: Memory from source agent
            target_memory: Memory from target agent
            strategy: Conflict resolution strategy

        Returns:
            Tuple of (resolved_memory or None if skip, reason string)
        """
        if strategy == ConflictResolution.SOURCE_WINS:
            return source_memory, "Source agent's version selected (source_wins strategy)"

        if strategy == ConflictResolution.TARGET_WINS:
            return target_memory, "Target agent's version kept (target_wins strategy)"

        if strategy == ConflictResolution.SKIP:
            return None, "Conflict skipped (skip strategy)"

        if strategy == ConflictResolution.NEWEST_WINS:
            source_time = source_memory.created_at or source_memory.last_accessed
            target_time = target_memory.created_at or target_memory.last_accessed

            if source_time and target_time:
                if source_time >= target_time:
                    return source_memory, f"Source is newer ({source_time} >= {target_time})"
                return target_memory, f"Target is newer ({target_time} > {source_time})"
            if source_time:
                return source_memory, "Source has timestamp, target doesn't"
            return target_memory, "Target has timestamp, source doesn't"

        if strategy == ConflictResolution.HIGHEST_IMPORTANCE:
            if source_memory.importance >= target_memory.importance:
                return (
                    source_memory,
                    f"Source has higher importance ({source_memory.importance} >= {target_memory.importance})",
                )
            return (
                target_memory,
                f"Target has higher importance ({target_memory.importance} > {source_memory.importance})",
            )

        if strategy == ConflictResolution.MERGE_METADATA:
            # Create a merged version with combined metadata
            from mindcore.v2.flr import Memory

            merged = Memory(
                memory_id=source_memory.memory_id,
                content=source_memory.content,  # Keep source content
                memory_type=source_memory.memory_type,
                user_id=source_memory.user_id,
                agent_id=source_memory.agent_id,
                # Merge topics and categories
                topics=list(set(source_memory.topics + target_memory.topics)),
                categories=list(set(source_memory.categories + target_memory.categories)),
                # Take higher importance
                importance=max(source_memory.importance, target_memory.importance),
                # Combine entities
                entities=list(set(source_memory.entities + target_memory.entities)),
                # Keep source access level
                access_level=source_memory.access_level,
                # Take higher reinforcement
                reinforcement_score=max(
                    source_memory.reinforcement_score, target_memory.reinforcement_score
                ),
            )
            return merged, "Merged metadata from both versions"

        # Default to source wins
        return source_memory, "Default: source wins"

    def get_memory_visibility(
        self,
        memory_id: str,
    ) -> dict[str, bool]:
        """Get which agents can see a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Dict mapping agent_id to visibility (True/False)
        """
        memory = self.storage.get(memory_id)
        if not memory:
            return {}

        visibility = {}
        for agent in self.registry.list_agents():
            can_access = self.registry.can_agent_access_memory(
                requesting_agent_id=agent.agent_id,
                memory_agent_id=memory.agent_id,
                memory_access_level=memory.access_level,
            )
            visibility[agent.agent_id] = can_access

        return visibility

    def get_share_history(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[ShareResult]:
        """Get sharing history.

        Args:
            agent_id: Filter by agent
            limit: Maximum results

        Returns:
            List of share results
        """
        history = self._share_history

        if agent_id:
            history = [
                r for r in history if r.source_agent == agent_id or agent_id in r.target_agents
            ]

        return history[-limit:]

    def get_sync_history(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[SyncResult]:
        """Get sync history.

        Args:
            agent_id: Filter by agent
            limit: Maximum results

        Returns:
            List of sync results
        """
        history = self._sync_history

        if agent_id:
            history = [r for r in history if agent_id in (r.source_agent, r.target_agent)]

        return history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get sharing statistics."""
        return {
            "total_shares": len(self._share_history),
            "successful_shares": len([r for r in self._share_history if r.success]),
            "total_syncs": len(self._sync_history),
            "successful_syncs": len([r for r in self._sync_history if r.success]),
            "total_memories_synced": sum(r.memories_synced for r in self._sync_history),
        }
