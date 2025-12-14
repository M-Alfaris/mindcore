"""Agent Registry - Agent registration and management.

Manages agent identities, capabilities, teams, and status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    """Agent operational status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    MAINTENANCE = "maintenance"


class AgentCapability(str, Enum):
    """Standard agent capabilities."""

    # Customer-facing
    CUSTOMER_SUPPORT = "customer_support"
    SALES = "sales"
    ONBOARDING = "onboarding"

    # Technical
    TECHNICAL_SUPPORT = "technical_support"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"

    # Data
    DATA_ANALYSIS = "data_analysis"
    REPORTING = "reporting"

    # Content
    CONTENT_CREATION = "content_creation"
    TRANSLATION = "translation"

    # Administrative
    SCHEDULING = "scheduling"
    BILLING = "billing"

    # Generic
    GENERAL = "general"
    SPECIALIZED = "specialized"


@dataclass
class Agent:
    """Registered agent in the system."""

    agent_id: str
    name: str
    description: str = ""

    # Capabilities
    capabilities: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)

    # Teams and access
    teams: list[str] = field(default_factory=list)
    can_read_global: bool = True
    can_write_global: bool = False

    # Status
    status: AgentStatus = AgentStatus.ACTIVE

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Statistics
    memories_created: int = 0
    memories_accessed: int = 0
    queries_handled: int = 0

    def is_active(self) -> bool:
        """Check if agent is active."""
        return self.status == AgentStatus.ACTIVE

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a capability."""
        return capability in self.capabilities

    def is_in_team(self, team: str) -> bool:
        """Check if agent is in a team."""
        return team in self.teams

    def shares_team_with(self, other: Agent) -> bool:
        """Check if agents share any team."""
        return bool(set(self.teams) & set(other.teams))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "specializations": self.specializations,
            "teams": self.teams,
            "can_read_global": self.can_read_global,
            "can_write_global": self.can_write_global,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "metadata": self.metadata,
            "memories_created": self.memories_created,
            "memories_accessed": self.memories_accessed,
            "queries_handled": self.queries_handled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Create from dictionary."""
        data = data.copy()

        # Parse enums
        if "status" in data and isinstance(data["status"], str):
            data["status"] = AgentStatus(data["status"])

        # Parse datetimes
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_active" in data and isinstance(data["last_active"], str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Team:
    """Team of agents with shared access."""

    team_id: str
    name: str
    description: str = ""

    # Members
    member_agent_ids: list[str] = field(default_factory=list)

    # Access control
    shared_topics: list[str] = field(default_factory=list)
    shared_memory_types: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_member(self, agent_id: str) -> None:
        """Add an agent to the team."""
        if agent_id not in self.member_agent_ids:
            self.member_agent_ids.append(agent_id)

    def remove_member(self, agent_id: str) -> bool:
        """Remove an agent from the team."""
        if agent_id in self.member_agent_ids:
            self.member_agent_ids.remove(agent_id)
            return True
        return False

    def has_member(self, agent_id: str) -> bool:
        """Check if agent is a member."""
        return agent_id in self.member_agent_ids

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "name": self.name,
            "description": self.description,
            "member_agent_ids": self.member_agent_ids,
            "shared_topics": self.shared_topics,
            "shared_memory_types": self.shared_memory_types,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class AgentRegistry:
    """Registry for managing agents and teams.

    Example:
        registry = AgentRegistry()

        # Register agents
        registry.register_agent(
            agent_id="support_bot",
            name="Support Agent",
            capabilities=["customer_support"],
            teams=["support_team"],
        )

        # Create teams
        registry.create_team(
            team_id="support_team",
            name="Support Team",
            shared_topics=["billing", "orders"],
        )

        # Query agents
        agents = registry.find_agents_by_capability("customer_support")
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._agents: dict[str, Agent] = {}
        self._teams: dict[str, Team] = {}

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
            capabilities: List of capabilities
            specializations: Specialized topics/domains
            teams: Teams the agent belongs to
            can_read_global: Can read global memories
            can_write_global: Can write global memories
            metadata: Additional metadata

        Returns:
            Registered Agent

        Raises:
            ValueError: If agent_id already exists
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already registered")

        agent = Agent(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities or [],
            specializations=specializations or [],
            teams=teams or [],
            can_read_global=can_read_global,
            can_write_global=can_write_global,
            metadata=metadata or {},
        )

        self._agents[agent_id] = agent

        # Add to teams
        for team_id in agent.teams:
            if team_id in self._teams:
                self._teams[team_id].add_member(agent_id)

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def update_agent(
        self,
        agent_id: str,
        **updates: Any,
    ) -> Agent | None:
        """Update agent properties.

        Args:
            agent_id: Agent to update
            **updates: Fields to update

        Returns:
            Updated agent or None if not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent to remove

        Returns:
            True if removed, False if not found
        """
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]

        # Remove from teams
        for team_id in agent.teams:
            if team_id in self._teams:
                self._teams[team_id].remove_member(agent_id)

        del self._agents[agent_id]
        return True

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Set agent status.

        Args:
            agent_id: Agent to update
            status: New status

        Returns:
            True if updated, False if not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.status = status
        return True

    def record_agent_activity(self, agent_id: str) -> None:
        """Record agent activity (updates last_active)."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)

    def list_agents(
        self,
        status: AgentStatus | None = None,
        team: str | None = None,
        capability: str | None = None,
    ) -> list[Agent]:
        """List agents with optional filters.

        Args:
            status: Filter by status
            team: Filter by team membership
            capability: Filter by capability

        Returns:
            List of matching agents
        """
        agents = list(self._agents.values())

        if status:
            agents = [a for a in agents if a.status == status]
        if team:
            agents = [a for a in agents if a.is_in_team(team)]
        if capability:
            agents = [a for a in agents if a.has_capability(capability)]

        return agents

    def find_agents_by_capability(self, capability: str) -> list[Agent]:
        """Find agents with a specific capability."""
        return [a for a in self._agents.values() if a.has_capability(capability)]

    def find_agents_by_specialization(self, specialization: str) -> list[Agent]:
        """Find agents with a specific specialization."""
        return [
            a for a in self._agents.values()
            if specialization in a.specializations
        ]

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

        Raises:
            ValueError: If team_id already exists
        """
        if team_id in self._teams:
            raise ValueError(f"Team '{team_id}' already exists")

        team = Team(
            team_id=team_id,
            name=name,
            description=description,
            shared_topics=shared_topics or [],
            shared_memory_types=shared_memory_types or [],
            metadata=metadata or {},
        )

        # Add existing agents that claim this team
        for agent in self._agents.values():
            if team_id in agent.teams:
                team.add_member(agent.agent_id)

        self._teams[team_id] = team
        return team

    def get_team(self, team_id: str) -> Team | None:
        """Get team by ID."""
        return self._teams.get(team_id)

    def delete_team(self, team_id: str) -> bool:
        """Delete a team.

        Args:
            team_id: Team to delete

        Returns:
            True if deleted, False if not found
        """
        if team_id not in self._teams:
            return False

        del self._teams[team_id]
        return True

    def add_agent_to_team(self, agent_id: str, team_id: str) -> bool:
        """Add an agent to a team.

        Args:
            agent_id: Agent to add
            team_id: Target team

        Returns:
            True if added, False if agent or team not found
        """
        agent = self._agents.get(agent_id)
        team = self._teams.get(team_id)

        if not agent or not team:
            return False

        if team_id not in agent.teams:
            agent.teams.append(team_id)
        team.add_member(agent_id)

        return True

    def remove_agent_from_team(self, agent_id: str, team_id: str) -> bool:
        """Remove an agent from a team.

        Args:
            agent_id: Agent to remove
            team_id: Target team

        Returns:
            True if removed, False if not found
        """
        agent = self._agents.get(agent_id)
        team = self._teams.get(team_id)

        if not agent or not team:
            return False

        if team_id in agent.teams:
            agent.teams.remove(team_id)
        team.remove_member(agent_id)

        return True

    def get_team_members(self, team_id: str) -> list[Agent]:
        """Get all agents in a team.

        Args:
            team_id: Team to query

        Returns:
            List of agents in the team
        """
        team = self._teams.get(team_id)
        if not team:
            return []

        return [
            self._agents[aid]
            for aid in team.member_agent_ids
            if aid in self._agents
        ]

    def get_agent_teammates(self, agent_id: str) -> list[Agent]:
        """Get all teammates of an agent.

        Args:
            agent_id: Agent to query

        Returns:
            List of agents in the same teams (excluding self)
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return []

        teammates = set()
        for team_id in agent.teams:
            team = self._teams.get(team_id)
            if team:
                teammates.update(team.member_agent_ids)

        teammates.discard(agent_id)
        return [self._agents[aid] for aid in teammates if aid in self._agents]

    def list_teams(self) -> list[Team]:
        """List all teams."""
        return list(self._teams.values())

    # === Access Control Helpers ===

    def can_agent_access_memory(
        self,
        requesting_agent_id: str,
        memory_agent_id: str | None,
        memory_access_level: str,
    ) -> bool:
        """Check if an agent can access a memory.

        Args:
            requesting_agent_id: Agent requesting access
            memory_agent_id: Agent that owns the memory
            memory_access_level: Memory's access level (private/team/shared/global)

        Returns:
            True if access is allowed
        """
        requester = self._agents.get(requesting_agent_id)
        if not requester or not requester.is_active():
            return False

        # Global memories
        if memory_access_level == "global":
            return requester.can_read_global

        # Shared memories (any agent for same user)
        if memory_access_level == "shared":
            return True

        # Own memories
        if memory_agent_id == requesting_agent_id:
            return True

        # Team memories
        if memory_access_level == "team" and memory_agent_id:
            owner = self._agents.get(memory_agent_id)
            if owner and requester.shares_team_with(owner):
                return True

        # Private memories - only owner
        if memory_access_level == "private":
            return memory_agent_id == requesting_agent_id

        return False

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        agents = list(self._agents.values())

        return {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.is_active()]),
            "total_teams": len(self._teams),
            "agents_by_status": {
                status.value: len([a for a in agents if a.status == status])
                for status in AgentStatus
            },
            "total_memories_created": sum(a.memories_created for a in agents),
            "total_queries_handled": sum(a.queries_handled for a in agents),
        }
