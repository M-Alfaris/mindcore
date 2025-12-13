"""Cross-Agent Layer - Unified interface for cross-agent memory operations.

Combines agent registry, memory sharing, and attention routing
into a single cohesive interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .registry import Agent, AgentRegistry, AgentStatus, Team
from .sharing import CrossAgentMemory, ShareResult, SyncDirection, SyncResult
from .routing import AttentionRouter, RouteResult, RoutingStrategy

if TYPE_CHECKING:
    from ..flr import Memory
    from ..storage.base import BaseStorage


class CrossAgentLayer:
    """Unified cross-agent memory layer.

    Provides a single interface for:
    - Agent registration and management
    - Memory sharing between agents
    - Cross-agent memory synchronization
    - Intelligent query routing

    Example:
        from mindcore.v2.cross_agent import CrossAgentLayer

        # Initialize with storage
        layer = CrossAgentLayer(storage)

        # Register agents
        layer.register_agent(
            agent_id="support_bot",
            name="Support Agent",
            capabilities=["customer_support", "billing"],
            teams=["customer_service"],
        )

        layer.register_agent(
            agent_id="sales_bot",
            name="Sales Agent",
            capabilities=["sales", "product_info"],
            teams=["customer_service"],
        )

        # Create team
        layer.create_team(
            team_id="customer_service",
            name="Customer Service Team",
            shared_topics=["billing", "orders", "products"],
        )

        # Store memory with sharing
        layer.store_memory(
            memory=memory,
            agent_id="support_bot",
            access_level="team",
        )

        # Query across agents
        result = layer.query(
            query="billing questions",
            user_id="user123",
            requesting_agent="sales_bot",
        )

        # Sync between agents
        sync_result = layer.sync(
            source_agent="support_bot",
            target_agent="sales_bot",
            user_id="user123",
        )
    """

    def __init__(self, storage: BaseStorage) -> None:
        """Initialize cross-agent layer.

        Args:
            storage: Storage backend for memories
        """
        self.storage = storage
        self.registry = AgentRegistry()
        self.sharing = CrossAgentMemory(storage, self.registry)
        self.router = AttentionRouter(storage, self.registry)

    # === Agent Management ===

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        capabilities: list[str] | None = None,
        specializations: list[str] | None = None,
        teams: list[str] | None = None,
        can_read_global: bool = True,
        can_write_global: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Register a new agent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable name
            description: Agent description
            capabilities: List of capabilities (e.g., "customer_support", "billing")
            specializations: Specialized topics/domains
            teams: Teams the agent belongs to
            can_read_global: Can read global knowledge base
            can_write_global: Can write to global knowledge base
            metadata: Additional metadata

        Returns:
            Registered Agent

        Raises:
            ValueError: If agent_id already exists
        """
        return self.registry.register_agent(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
            specializations=specializations,
            teams=teams,
            can_read_global=can_read_global,
            can_write_global=can_write_global,
            metadata=metadata,
        )

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        return self.registry.get_agent(agent_id)

    def list_agents(
        self,
        status: AgentStatus | None = None,
        team: str | None = None,
        capability: str | None = None,
    ) -> list[Agent]:
        """List agents with optional filters."""
        return self.registry.list_agents(status=status, team=team, capability=capability)

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        return self.registry.unregister_agent(agent_id)

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Set agent status (active, inactive, suspended, maintenance)."""
        return self.registry.set_agent_status(agent_id, status)

    # === Team Management ===

    def create_team(
        self,
        team_id: str,
        name: str,
        description: str = "",
        shared_topics: list[str] | None = None,
        shared_memory_types: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Team:
        """Create a new team.

        Args:
            team_id: Unique team identifier
            name: Human-readable name
            description: Team description
            shared_topics: Topics shared within team
            shared_memory_types: Memory types shared within team
            metadata: Additional metadata

        Returns:
            Created Team
        """
        return self.registry.create_team(
            team_id=team_id,
            name=name,
            description=description,
            shared_topics=shared_topics,
            shared_memory_types=shared_memory_types,
            metadata=metadata,
        )

    def get_team(self, team_id: str) -> Team | None:
        """Get team by ID."""
        return self.registry.get_team(team_id)

    def list_teams(self) -> list[Team]:
        """List all teams."""
        return self.registry.list_teams()

    def delete_team(self, team_id: str) -> bool:
        """Delete a team."""
        return self.registry.delete_team(team_id)

    def add_agent_to_team(self, agent_id: str, team_id: str) -> bool:
        """Add an agent to a team."""
        return self.registry.add_agent_to_team(agent_id, team_id)

    def remove_agent_from_team(self, agent_id: str, team_id: str) -> bool:
        """Remove an agent from a team."""
        return self.registry.remove_agent_from_team(agent_id, team_id)

    def get_team_members(self, team_id: str) -> list[Agent]:
        """Get all agents in a team."""
        return self.registry.get_team_members(team_id)

    def get_teammates(self, agent_id: str) -> list[Agent]:
        """Get all teammates of an agent."""
        return self.registry.get_agent_teammates(agent_id)

    # === Memory Operations ===

    def store_memory(
        self,
        memory: Memory,
        agent_id: str,
        access_level: str = "private",
    ) -> str:
        """Store a memory with agent ownership.

        Args:
            memory: Memory to store
            agent_id: Agent storing the memory
            access_level: Access level (private/team/shared/global)

        Returns:
            Memory ID

        Raises:
            ValueError: If agent not found or lacks permission
        """
        agent = self.registry.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        if access_level == "global" and not agent.can_write_global:
            raise ValueError(f"Agent '{agent_id}' lacks permission to write global memories")

        # Set memory ownership
        memory.agent_id = agent_id
        memory.access_level = access_level

        # Store memory
        memory_id = self.storage.store(memory)

        # Update agent stats
        agent.memories_created += 1
        self.registry.record_agent_activity(agent_id)

        return memory_id

    def share_memory(
        self,
        memory_id: str,
        source_agent: str,
        access_level: str,
        target_agents: list[str] | None = None,
    ) -> ShareResult:
        """Share a memory with other agents.

        Args:
            memory_id: Memory to share
            source_agent: Agent sharing the memory
            access_level: New access level
            target_agents: Specific agents to notify

        Returns:
            ShareResult with operation details
        """
        return self.sharing.share_memory(
            memory_id=memory_id,
            source_agent=source_agent,
            access_level=access_level,
            target_agents=target_agents,
        )

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
        return self.sharing.get_accessible_memories(
            agent_id=agent_id,
            user_id=user_id,
            topics=topics,
            memory_types=memory_types,
            include_global=include_global,
            limit=limit,
        )

    # === Sync Operations ===

    def sync(
        self,
        source_agent: str,
        target_agent: str,
        user_id: str,
        direction: SyncDirection = SyncDirection.ONE_WAY,
        topics: list[str] | None = None,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> SyncResult:
        """Synchronize memories between agents.

        Args:
            source_agent: Agent to sync from
            target_agent: Agent to sync to
            user_id: User context
            direction: Sync direction
            topics: Filter by topics
            memory_types: Filter by memory types
            since: Only sync memories created after this time

        Returns:
            SyncResult with operation details
        """
        return self.sharing.sync_agents(
            source_agent=source_agent,
            target_agent=target_agent,
            user_id=user_id,
            direction=direction,
            topics=topics,
            memory_types=memory_types,
            since=since,
        )

    # === Query Operations ===

    def query(
        self,
        query: str,
        user_id: str,
        requesting_agent: str | None = None,
        strategy: RoutingStrategy = RoutingStrategy.BEST_MATCH,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        max_agents: int = 5,
        max_memories_per_agent: int = 10,
    ) -> RouteResult:
        """Query memories across agents.

        Uses intelligent routing to find the most relevant agent memories.

        Args:
            query: Search query
            user_id: User context
            requesting_agent: Agent making the request
            strategy: Routing strategy
            attention_hints: Topics/capabilities to prioritize
            memory_types: Filter by memory types
            max_agents: Maximum agents to query
            max_memories_per_agent: Maximum memories per agent

        Returns:
            RouteResult with memories from selected agents
        """
        # Record activity
        if requesting_agent:
            agent = self.registry.get_agent(requesting_agent)
            if agent:
                agent.queries_handled += 1
                self.registry.record_agent_activity(requesting_agent)

        return self.router.route(
            query=query,
            user_id=user_id,
            requesting_agent=requesting_agent,
            strategy=strategy,
            attention_hints=attention_hints,
            memory_types=memory_types,
            max_agents=max_agents,
            max_memories_per_agent=max_memories_per_agent,
        )

    def rank_agents(
        self,
        query: str,
        attention_hints: list[str] | None = None,
        requesting_agent: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank agents by relevance to a query.

        Args:
            query: Search query
            attention_hints: Topics/capabilities to prioritize
            requesting_agent: Agent making the request

        Returns:
            List of agent rankings with scores
        """
        scores = self.router.rank_agents(
            query=query,
            attention_hints=attention_hints,
            requesting_agent=requesting_agent,
        )

        return [
            {
                "agent_id": s.agent_id,
                "score": s.score,
                "reasons": s.reasons,
                "breakdown": {
                    "capability_score": s.capability_score,
                    "specialization_score": s.specialization_score,
                    "team_score": s.team_score,
                    "history_score": s.history_score,
                },
            }
            for s in scores
        ]

    def suggest_agents(
        self,
        attention_hints: list[str],
        limit: int = 3,
    ) -> list[Agent]:
        """Suggest agents based on needed capabilities.

        Args:
            attention_hints: Topics/capabilities needed
            limit: Maximum agents to suggest

        Returns:
            List of suggested agents
        """
        return self.router.suggest_agents(attention_hints, limit)

    # === Access Control ===

    def can_access(
        self,
        requesting_agent: str,
        memory_agent_id: str | None,
        access_level: str,
    ) -> bool:
        """Check if an agent can access a memory.

        Args:
            requesting_agent: Agent requesting access
            memory_agent_id: Agent that owns the memory
            access_level: Memory's access level

        Returns:
            True if access is allowed
        """
        return self.registry.can_agent_access_memory(
            requesting_agent_id=requesting_agent,
            memory_agent_id=memory_agent_id,
            memory_access_level=access_level,
        )

    def get_memory_visibility(self, memory_id: str) -> dict[str, bool]:
        """Get which agents can see a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Dict mapping agent_id to visibility
        """
        return self.sharing.get_memory_visibility(memory_id)

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cross-agent statistics."""
        return {
            "registry": self.registry.get_stats(),
            "sharing": self.sharing.get_stats(),
            "routing": self.router.get_routing_stats(),
        }

    def get_agent_stats(self, agent_id: str) -> dict[str, Any] | None:
        """Get statistics for a specific agent.

        Args:
            agent_id: Agent to query

        Returns:
            Agent statistics or None if not found
        """
        agent = self.registry.get_agent(agent_id)
        if not agent:
            return None

        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "status": agent.status.value,
            "memories_created": agent.memories_created,
            "memories_accessed": agent.memories_accessed,
            "queries_handled": agent.queries_handled,
            "teams": agent.teams,
            "capabilities": agent.capabilities,
            "specializations": agent.specializations,
            "last_active": agent.last_active.isoformat() if agent.last_active else None,
        }
