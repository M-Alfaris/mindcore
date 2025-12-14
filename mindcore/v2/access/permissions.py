"""Multi-agent access control for Mindcore v2.

Handles agent registration, permissions, and memory access control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum


class Permission(str, Enum):
    """Memory operation permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


@dataclass
class AgentProfile:
    """Profile for a registered agent."""

    agent_id: str
    name: str
    description: str = ""

    # Team/group memberships
    teams: list[str] = field(default_factory=list)

    # Permissions by access level
    permissions: dict[str, list[Permission]] = field(default_factory=dict)
    # Example: {"private": [READ, WRITE], "shared": [READ], "global": [READ]}

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime | None = None
    is_active: bool = True

    # Default access level for new memories
    default_access_level: str = "private"

    def __post_init__(self):
        """Set default permissions if not provided."""
        if not self.permissions:
            self.permissions = {
                "private": [Permission.READ, Permission.WRITE, Permission.DELETE],
                "team": [Permission.READ, Permission.WRITE],
                "shared": [Permission.READ],
                "global": [Permission.READ],
            }

    def has_permission(
        self,
        access_level: str,
        permission: Permission,
    ) -> bool:
        """Check if agent has a specific permission for an access level."""
        level_permissions = self.permissions.get(access_level, [])
        return permission in level_permissions or Permission.ADMIN in level_permissions

    def can_read(self, access_level: str) -> bool:
        """Check if agent can read memories at access level."""
        return self.has_permission(access_level, Permission.READ)

    def can_write(self, access_level: str) -> bool:
        """Check if agent can write memories at access level."""
        return self.has_permission(access_level, Permission.WRITE)

    def can_delete(self, access_level: str) -> bool:
        """Check if agent can delete memories at access level."""
        return self.has_permission(access_level, Permission.DELETE)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "teams": self.teams,
            "permissions": {
                level: [p.value for p in perms]
                for level, perms in self.permissions.items()
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "is_active": self.is_active,
            "default_access_level": self.default_access_level,
        }


@dataclass
class AccessDecision:
    """Result of an access control decision."""

    allowed: bool
    reason: str
    agent_id: str
    memory_id: str | None = None
    access_level: str | None = None
    permission: Permission | None = None


class AccessController:
    """Multi-agent access control manager.

    Handles:
    - Agent registration and management
    - Permission checks for memory operations
    - Team-based access control
    - Cross-agent memory sharing

    Example:
        acl = AccessController()

        # Register agents
        acl.register_agent("support_bot", "Support Agent", teams=["support"])
        acl.register_agent("sales_bot", "Sales Agent", teams=["sales"])

        # Check access
        decision = acl.can_access(
            agent_id="support_bot",
            memory_access_level="team",
            memory_agent_id="sales_bot",
            permission=Permission.READ,
        )
    """

    def __init__(self):
        """Initialize access controller."""
        self._agents: dict[str, AgentProfile] = {}
        self._teams: dict[str, set[str]] = {}  # team -> set of agent_ids

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        teams: list[str] | None = None,
        permissions: dict[str, list[Permission]] | None = None,
        default_access_level: str = "private",
    ) -> AgentProfile:
        """Register a new agent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable name
            description: Agent description
            teams: Team memberships
            permissions: Custom permissions by access level
            default_access_level: Default access level for new memories

        Returns:
            AgentProfile for the registered agent
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")

        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            description=description,
            teams=teams or [],
            permissions=permissions or {},
            default_access_level=default_access_level,
        )

        self._agents[agent_id] = profile

        # Update team memberships
        for team in profile.teams:
            if team not in self._teams:
                self._teams[team] = set()
            self._teams[team].add(agent_id)

        return profile

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistered, False if not found
        """
        if agent_id not in self._agents:
            return False

        profile = self._agents[agent_id]

        # Remove from teams
        for team in profile.teams:
            if team in self._teams:
                self._teams[team].discard(agent_id)

        del self._agents[agent_id]
        return True

    def get_agent(self, agent_id: str) -> AgentProfile | None:
        """Get agent profile."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[AgentProfile]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_team_members(self, team: str) -> list[str]:
        """Get all agent IDs in a team."""
        return list(self._teams.get(team, set()))

    def add_agent_to_team(self, agent_id: str, team: str) -> bool:
        """Add an agent to a team."""
        if agent_id not in self._agents:
            return False

        self._agents[agent_id].teams.append(team)

        if team not in self._teams:
            self._teams[team] = set()
        self._teams[team].add(agent_id)

        return True

    def remove_agent_from_team(self, agent_id: str, team: str) -> bool:
        """Remove an agent from a team."""
        if agent_id not in self._agents:
            return False

        if team in self._agents[agent_id].teams:
            self._agents[agent_id].teams.remove(team)

        if team in self._teams:
            self._teams[team].discard(agent_id)

        return True

    def can_access(
        self,
        agent_id: str,
        memory_access_level: str,
        memory_agent_id: str | None,
        memory_teams: list[str] | None = None,
        permission: Permission = Permission.READ,
        memory_id: str | None = None,
    ) -> AccessDecision:
        """Check if an agent can access a memory.

        Args:
            agent_id: Agent requesting access
            memory_access_level: Access level of the memory
            memory_agent_id: Agent that owns the memory
            memory_teams: Teams the memory is shared with
            permission: Permission being requested
            memory_id: Optional memory ID for logging

        Returns:
            AccessDecision with result and reason
        """
        # Get agent profile
        profile = self._agents.get(agent_id)
        if not profile:
            return AccessDecision(
                allowed=False,
                reason=f"Agent {agent_id} not registered",
                agent_id=agent_id,
                memory_id=memory_id,
                access_level=memory_access_level,
                permission=permission,
            )

        if not profile.is_active:
            return AccessDecision(
                allowed=False,
                reason=f"Agent {agent_id} is inactive",
                agent_id=agent_id,
                memory_id=memory_id,
                access_level=memory_access_level,
                permission=permission,
            )

        # Check permission for access level
        if not profile.has_permission(memory_access_level, permission):
            return AccessDecision(
                allowed=False,
                reason=f"Agent lacks {permission.value} permission for {memory_access_level} memories",
                agent_id=agent_id,
                memory_id=memory_id,
                access_level=memory_access_level,
                permission=permission,
            )

        # Access level specific checks
        if memory_access_level == "private":
            # Private: only owner can access
            if memory_agent_id != agent_id:
                return AccessDecision(
                    allowed=False,
                    reason="Cannot access private memory of another agent",
                    agent_id=agent_id,
                    memory_id=memory_id,
                    access_level=memory_access_level,
                    permission=permission,
                )

        elif memory_access_level == "team":
            # Team: must be in a common team
            if memory_agent_id != agent_id:
                owner_profile = self._agents.get(memory_agent_id)
                if owner_profile:
                    common_teams = set(profile.teams) & set(owner_profile.teams)
                    if not common_teams:
                        # Check if memory is explicitly shared with agent's teams
                        if memory_teams:
                            shared_teams = set(profile.teams) & set(memory_teams)
                            if not shared_teams:
                                return AccessDecision(
                                    allowed=False,
                                    reason="No common team with memory owner",
                                    agent_id=agent_id,
                                    memory_id=memory_id,
                                    access_level=memory_access_level,
                                    permission=permission,
                                )
                        else:
                            return AccessDecision(
                                allowed=False,
                                reason="No common team with memory owner",
                                agent_id=agent_id,
                                memory_id=memory_id,
                                access_level=memory_access_level,
                                permission=permission,
                            )

        # Shared and global: accessible by all with permission
        # (already checked permission above)

        # Update last active
        profile.last_active = datetime.now(timezone.utc)

        return AccessDecision(
            allowed=True,
            reason="Access granted",
            agent_id=agent_id,
            memory_id=memory_id,
            access_level=memory_access_level,
            permission=permission,
        )

    def filter_accessible_memories(
        self,
        agent_id: str,
        memories: list[Any],
        permission: Permission = Permission.READ,
    ) -> list[Any]:
        """Filter a list of memories to only those accessible by agent.

        Args:
            agent_id: Agent requesting access
            memories: List of Memory objects
            permission: Required permission

        Returns:
            Filtered list of accessible memories
        """
        accessible = []

        for memory in memories:
            decision = self.can_access(
                agent_id=agent_id,
                memory_access_level=memory.access_level,
                memory_agent_id=memory.agent_id,
                permission=permission,
                memory_id=memory.memory_id,
            )
            if decision.allowed:
                accessible.append(memory)

        return accessible

    def get_default_access_level(self, agent_id: str) -> str:
        """Get default access level for an agent's new memories."""
        profile = self._agents.get(agent_id)
        return profile.default_access_level if profile else "private"

    def get_stats(self) -> dict[str, Any]:
        """Get access control statistics."""
        return {
            "total_agents": len(self._agents),
            "active_agents": sum(1 for a in self._agents.values() if a.is_active),
            "total_teams": len(self._teams),
            "agents_by_team": {
                team: len(members) for team, members in self._teams.items()
            },
        }
