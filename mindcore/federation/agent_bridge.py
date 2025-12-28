"""Agent Memory Bridge.

Connects an individual agent's FLR to the federated memory system.

The bridge provides:
- Local FLR for fast, session-scoped memory
- Connection to shared CLST with access control
- Signal propagation to cross-agent aggregator
- Feedback flow to federated SVL

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     AgentMemoryBridge                        │
    │  ┌─────────┐                                                │
    │  │  FLR    │ ◄── Local hot memory (agent-owned)             │
    │  │ (Local) │                                                │
    │  └────┬────┘                                                │
    │       │                                                      │
    │       ▼                                                      │
    │  ┌─────────┐    ┌─────────────┐    ┌──────────────────┐    │
    │  │ Signal  │───►│  Federated  │───►│ Cross-Agent      │    │
    │  │Propagate│    │    CLST     │    │ Signal Aggregator│    │
    │  └─────────┘    └─────────────┘    └──────────────────┘    │
    │       │                                                      │
    │       ▼                                                      │
    │  ┌─────────────┐                                            │
    │  │ Federated   │◄── Shared vocabulary + feedback            │
    │  │    SVL      │                                            │
    │  └─────────────┘                                            │
    └─────────────────────────────────────────────────────────────┘

Example:
    # Create agent bridge
    bridge = AgentMemoryBridge(
        agent_id="support-agent-001",
        agent_type="support-bot",
        namespace=team_namespace,
        federated_clst=clst,
        federated_svl=svl,
        signal_aggregator=aggregator,
    )

    # Store locally (FLR) and optionally persist to CLST
    bridge.remember(
        content="Customer prefers email",
        user_id="customer-123",
        persist=True,  # Also store in CLST
        access_level=AccessLevel.TEAM,
    )

    # Recall with combined local + federated search
    memories = bridge.recall(
        query="customer contact preferences",
        include_federated=True,
    )

    # Reinforce propagates to aggregator
    bridge.reinforce("memory-id", signal=0.8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .access_control import AccessLevel, AccessPolicy, AccessScope
from .namespace import MemoryNamespace


if TYPE_CHECKING:
    from .federated_clst import FederatedCLST
    from .federated_svl import FederatedSVL
    from .signal_aggregator import CrossAgentSignalAggregator


@dataclass
class AgentIdentity:
    """Identity information for an agent.

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type/class of agent (e.g., "support-bot", "sales-agent")
        display_name: Human-readable name
        capabilities: Set of capabilities the agent has
        groups: Ad-hoc group memberships
        created_at: When agent was registered
    """

    agent_id: str
    agent_type: str
    display_name: str | None = None
    capabilities: set[str] = field(default_factory=set)
    groups: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_scope(self, namespace: MemoryNamespace) -> AccessScope:
        """Create AccessScope from this identity and namespace."""
        return AccessScope(
            org_id=namespace.org_id,
            department=namespace.department,
            team=namespace.team,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            groups=self.groups,
        )


@dataclass
class LocalMemory:
    """Memory stored in local FLR.

    Lightweight structure for hot memory.
    """

    memory_id: str
    content: str
    user_id: str | None = None
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reinforcement_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    persisted_to_clst: bool = False


@dataclass
class AgentMemoryBridge:
    """Bridge connecting an agent to the federated memory system.

    Provides unified interface for:
    - Local FLR operations
    - Federated CLST access
    - Signal propagation
    - SVL feedback

    Attributes:
        identity: Agent identity information
        namespace: Agent's namespace context
        federated_clst: Shared cold storage (optional)
        federated_svl: Shared vocabulary (optional)
        signal_aggregator: Cross-agent signal aggregation (optional)
        local_memories: Local FLR cache
        default_access_level: Default access for persisted memories
    """

    identity: AgentIdentity
    namespace: MemoryNamespace
    federated_clst: FederatedCLST | None = None
    federated_svl: FederatedSVL | None = None
    signal_aggregator: CrossAgentSignalAggregator | None = None
    local_memories: dict[str, LocalMemory] = field(default_factory=dict)
    default_access_level: AccessLevel = AccessLevel.TEAM

    # Configuration
    auto_persist_threshold: float = 0.7  # Auto-persist if reinforcement > this
    max_local_memories: int = 1000
    propagate_signals: bool = True

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        self._scope = self.identity.to_scope(self.namespace)

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.identity.agent_id

    @property
    def scope(self) -> AccessScope:
        """Get agent's access scope."""
        return self._scope

    def remember(
        self,
        content: str,
        user_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = False,
        access_level: AccessLevel | None = None,
        access_policy: AccessPolicy | None = None,
        vector: list[float] | None = None,
    ) -> str:
        """Store a memory in local FLR, optionally persisting to CLST.

        Args:
            content: Memory content
            user_id: Associated user ID
            topics: Memory topics (extracted via SVL)
            categories: Memory categories
            metadata: Additional metadata
            persist: Whether to also store in federated CLST
            access_level: Access level for CLST storage
            access_policy: Custom access policy
            vector: Embedding vector (required for CLST)

        Returns:
            Memory ID
        """
        memory_id = str(uuid4())

        # Create local memory
        local = LocalMemory(
            memory_id=memory_id,
            content=content,
            user_id=user_id,
            topics=topics or [],
            categories=categories or [],
            metadata=metadata or {},
        )

        # Store locally
        self.local_memories[memory_id] = local

        # Evict old memories if needed
        self._evict_if_needed()

        # Persist to CLST if requested
        if persist and self.federated_clst and vector:
            self.federated_clst.store(
                memory_id=memory_id,
                content=content,
                vector=vector,
                namespace=self.namespace,
                owner_agent_id=self.agent_id,
                owner_scope=self.scope,
                access_level=access_level or self.default_access_level,
                policy=access_policy,
                metadata={
                    "user_id": user_id,
                    "topics": topics or [],
                    "categories": categories or [],
                    **(metadata or {}),
                },
            )
            local.persisted_to_clst = True

        # Record SVL feedback if available
        if self.federated_svl:
            for topic in topics or []:
                self.federated_svl.record_feedback(
                    topic=topic,
                    was_effective=True,  # Assumed effective at creation
                    agent_id=self.agent_id,
                    namespace=self.namespace,
                )

        return memory_id

    def recall(
        self,
        query: str | None = None,
        query_vector: list[float] | None = None,
        topics: list[str] | None = None,
        user_id: str | None = None,
        include_federated: bool = True,
        include_ancestors: bool = True,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[LocalMemory | Any]:
        """Recall memories from local FLR and optionally federated CLST.

        Args:
            query: Text query (for local search)
            query_vector: Embedding vector (for CLST search)
            topics: Filter by topics
            user_id: Filter by user ID
            include_federated: Include CLST search
            include_ancestors: Search ancestor namespaces
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of memories (LocalMemory or FederatedMemory)
        """
        results: list[Any] = []

        # Search local FLR first
        local_results = self._search_local(
            query=query,
            topics=topics,
            user_id=user_id,
            limit=limit,
        )
        results.extend(local_results)

        # Search federated CLST
        if include_federated and self.federated_clst and query_vector:
            from .federated_clst import NamespacedQuery

            federated_query = NamespacedQuery(
                query_vector=query_vector,
                namespaces=[self.namespace],
                requester=self.scope,
                include_ancestors=include_ancestors,
                limit=limit,
                min_score=min_score,
            )

            federated_results = self.federated_clst.search(federated_query)

            # Deduplicate (local takes precedence)
            local_ids = {m.memory_id for m in local_results}
            for fm in federated_results:
                if fm.memory_id not in local_ids:
                    results.append(fm)

        # Sort by relevance/recency and limit
        results = results[:limit]

        # Track access
        for mem in results:
            if isinstance(mem, LocalMemory):
                mem.access_count += 1

        return results

    def reinforce(
        self,
        memory_id: str,
        signal: float,
        context: dict[str, Any] | None = None,
        was_effective: bool | None = None,
    ) -> float:
        """Apply reinforcement signal to a memory.

        Signal is applied locally and propagated to federation.

        Args:
            memory_id: Memory to reinforce
            signal: Reinforcement value (-1 to 1)
            context: Optional context about the signal
            was_effective: Whether retrieval was effective (for SVL feedback)

        Returns:
            New reinforcement score
        """
        score = signal

        # Apply to local memory
        if memory_id in self.local_memories:
            local = self.local_memories[memory_id]
            local.reinforcement_score = max(
                -1.0, min(1.0, local.reinforcement_score + signal * 0.5)
            )
            score = local.reinforcement_score

            # Auto-persist if threshold reached
            if (
                not local.persisted_to_clst
                and local.reinforcement_score > self.auto_persist_threshold
            ):
                # Would need vector to persist - flag for later
                pass

            # SVL feedback
            if was_effective is not None and self.federated_svl:
                for topic in local.topics:
                    self.federated_svl.record_feedback(
                        topic=topic,
                        was_effective=was_effective,
                        agent_id=self.agent_id,
                        namespace=self.namespace,
                    )
                for category in local.categories:
                    self.federated_svl.record_feedback(
                        category=category,
                        was_effective=was_effective,
                        agent_id=self.agent_id,
                        namespace=self.namespace,
                    )

        # Propagate to federated CLST
        if self.federated_clst:
            clst_score = self.federated_clst.apply_reinforcement(
                memory_id=memory_id,
                agent_id=self.agent_id,
                signal=signal,
                requester=self.scope,
            )
            if clst_score is not None:
                score = clst_score

        # Propagate to cross-agent aggregator
        if self.propagate_signals and self.signal_aggregator:
            score = self.signal_aggregator.add_signal(
                memory_id=memory_id,
                agent_id=self.agent_id,
                value=signal,
                agent_scope=self.scope,
                context=context,
            )

        return score

    def get_vocabulary(self) -> dict[str, Any]:
        """Get vocabulary from federated SVL."""
        if self.federated_svl:
            return self.federated_svl.get_vocabulary()
        return {}

    def get_feedback_for_extraction(self) -> dict[str, Any]:
        """Get aggregated feedback for metadata extraction.

        Returns personalized feedback combining agent's history
        with team/department/org patterns.
        """
        if self.federated_svl:
            return self.federated_svl.get_feedback_for_extractor(
                agent_id=self.agent_id,
                namespace=self.namespace,
            )
        return {}

    def get_cross_agent_score(self, memory_id: str) -> float:
        """Get cross-agent aggregated score for a memory."""
        if self.signal_aggregator:
            return self.signal_aggregator.get_aggregated_score(
                memory_id=memory_id,
                reference_scope=self.scope,
            )
        return 0.0

    def share_memory(
        self,
        memory_id: str,
        access_level: AccessLevel,
        access_policy: AccessPolicy | None = None,
        vector: list[float] | None = None,
    ) -> bool:
        """Share a local memory to federated CLST.

        Args:
            memory_id: Local memory to share
            access_level: Access level for sharing
            access_policy: Custom access policy
            vector: Embedding vector (required)

        Returns:
            True if shared successfully
        """
        if memory_id not in self.local_memories:
            return False

        if not self.federated_clst or not vector:
            return False

        local = self.local_memories[memory_id]

        self.federated_clst.store(
            memory_id=memory_id,
            content=local.content,
            vector=vector,
            namespace=self.namespace,
            owner_agent_id=self.agent_id,
            owner_scope=self.scope,
            access_level=access_level,
            policy=access_policy,
            metadata={
                "user_id": local.user_id,
                "topics": local.topics,
                "categories": local.categories,
                **local.metadata,
            },
        )

        local.persisted_to_clst = True
        return True

    def get_local_stats(self) -> dict[str, Any]:
        """Get statistics about local memory."""
        return {
            "total_memories": len(self.local_memories),
            "persisted_count": sum(1 for m in self.local_memories.values() if m.persisted_to_clst),
            "avg_reinforcement": (
                sum(m.reinforcement_score for m in self.local_memories.values())
                / len(self.local_memories)
                if self.local_memories
                else 0
            ),
            "total_accesses": sum(m.access_count for m in self.local_memories.values()),
        }

    def _search_local(
        self,
        query: str | None = None,
        topics: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[LocalMemory]:
        """Search local FLR."""
        results = []

        for memory in self.local_memories.values():
            # Filter by user_id
            if user_id and memory.user_id != user_id:
                continue

            # Filter by topics
            if topics:
                if not any(t in memory.topics for t in topics):
                    continue

            # Simple text match for query
            if query:
                if query.lower() not in memory.content.lower():
                    continue

            results.append(memory)

        # Sort by reinforcement + recency
        results.sort(
            key=lambda m: (m.reinforcement_score, m.created_at),
            reverse=True,
        )

        return results[:limit]

    def _evict_if_needed(self) -> None:
        """Evict old memories if over limit."""
        if len(self.local_memories) <= self.max_local_memories:
            return

        # Sort by score + recency, evict lowest
        sorted_memories = sorted(
            self.local_memories.items(),
            key=lambda x: (x[1].reinforcement_score, x[1].created_at),
        )

        # Remove 10% of lowest scoring
        to_remove = max(1, len(sorted_memories) // 10)
        for memory_id, _ in sorted_memories[:to_remove]:
            del self.local_memories[memory_id]
