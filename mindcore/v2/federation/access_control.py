"""Access Control for Federated Memory.

Provides fine-grained access control for memories across agents,
teams, departments, and organizations.

Access Levels (hierarchical):
- PRIVATE: Only the creating agent
- AGENT_TYPE: Same type of agent (e.g., all "support-bot" instances)
- TEAM: Same team
- DEPARTMENT: Same department
- AD_HOC_GROUP: Custom group membership
- ORGANIZATION: All agents in org
- PUBLIC: Cross-organization (rare)

Example:
    # Create ACL for a sensitive memory
    acl = MemoryACL(
        owner_agent_id="agent-001",
        access_level=AccessLevel.TEAM,
        namespace=namespace,
        allowed_groups=["vip-handlers"],  # Additional ad-hoc access
        denied_agents=["agent-003"],  # Explicit deny
    )

    # Check access
    if acl.can_read(requesting_agent):
        memory = clst.get(memory_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .namespace import MemoryNamespace


class AccessLevel(IntEnum):
    """Memory access levels (ordered by scope, lower = more restricted)."""

    PRIVATE = 0  # Only the creating agent
    AGENT_TYPE = 10  # Same agent type (e.g., all "support-bot")
    TEAM = 20  # Same team
    DEPARTMENT = 30  # Same department
    AD_HOC_GROUP = 40  # Custom group membership
    ORGANIZATION = 50  # All agents in org
    PUBLIC = 100  # Cross-organization (use carefully)


@dataclass
class AccessScope:
    """Defines the scope of access for a memory or query.

    Attributes:
        org_id: Organization identifier (required)
        department: Department within org (optional)
        team: Team within department (optional)
        agent_type: Type/class of agent (optional)
        agent_id: Specific agent identifier (optional)
        groups: Ad-hoc group memberships (optional)
    """

    org_id: str
    department: str | None = None
    team: str | None = None
    agent_type: str | None = None
    agent_id: str | None = None
    groups: set[str] = field(default_factory=set)

    def matches_level(self, level: AccessLevel, target: AccessScope) -> bool:
        """Check if this scope can access target at given level.

        Args:
            level: Required access level
            target: The scope to check access against

        Returns:
            True if access is granted
        """
        # Must be same org (except PUBLIC)
        if level != AccessLevel.PUBLIC and self.org_id != target.org_id:
            return False

        if level == AccessLevel.PUBLIC:
            return True

        if level == AccessLevel.ORGANIZATION:
            return self.org_id == target.org_id

        if level == AccessLevel.DEPARTMENT:
            return (
                self.org_id == target.org_id
                and self.department == target.department
            )

        if level == AccessLevel.TEAM:
            return (
                self.org_id == target.org_id
                and self.department == target.department
                and self.team == target.team
            )

        if level == AccessLevel.AGENT_TYPE:
            return (
                self.org_id == target.org_id
                and self.agent_type == target.agent_type
            )

        if level == AccessLevel.AD_HOC_GROUP:
            # Check if any group overlaps
            return bool(self.groups & target.groups)

        if level == AccessLevel.PRIVATE:
            return self.agent_id == target.agent_id

        return False

    def to_filter_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage filtering."""
        return {
            "org_id": self.org_id,
            "department": self.department,
            "team": self.team,
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "groups": list(self.groups),
        }

    @classmethod
    def from_namespace(
        cls,
        namespace: MemoryNamespace,
        agent_id: str | None = None,
        agent_type: str | None = None,
        groups: set[str] | None = None,
    ) -> AccessScope:
        """Create AccessScope from a MemoryNamespace."""
        return cls(
            org_id=namespace.org_id,
            department=namespace.department,
            team=namespace.team,
            agent_id=agent_id,
            agent_type=agent_type,
            groups=groups or set(),
        )


@dataclass
class AccessPolicy:
    """Policy defining access rules for a memory.

    Supports both allow-list and deny-list patterns.
    Deny always takes precedence over allow.
    """

    # Primary access level
    access_level: AccessLevel = AccessLevel.PRIVATE

    # Explicit allow/deny lists
    allowed_agents: set[str] = field(default_factory=set)
    denied_agents: set[str] = field(default_factory=set)
    allowed_groups: set[str] = field(default_factory=set)
    denied_groups: set[str] = field(default_factory=set)
    allowed_departments: set[str] = field(default_factory=set)
    denied_departments: set[str] = field(default_factory=set)

    # Time-based access
    expires_at: datetime | None = None
    available_from: datetime | None = None

    # Capability requirements
    required_capabilities: set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if access has expired."""
        if self.expires_at is None:
            return False
        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        expires = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return now > expires

    def is_available(self) -> bool:
        """Check if access is currently available."""
        if self.available_from is None:
            return True
        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        available = self.available_from if self.available_from.tzinfo else self.available_from.replace(tzinfo=timezone.utc)
        return now >= available


@dataclass
class MemoryACL:
    """Access Control List for a specific memory.

    Combines ownership, policy, and audit information.

    Example:
        acl = MemoryACL(
            memory_id="mem-123",
            owner_agent_id="agent-001",
            owner_scope=agent_scope,
            policy=AccessPolicy(
                access_level=AccessLevel.TEAM,
                allowed_groups={"vip-handlers"},
            ),
        )

        # Check access
        can_access = acl.can_read(requesting_scope)
        can_modify = acl.can_write(requesting_scope)
    """

    memory_id: str
    owner_agent_id: str
    owner_scope: AccessScope
    policy: AccessPolicy = field(default_factory=AccessPolicy)

    # Audit trail
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_by: str | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0

    def can_read(
        self,
        requester: AccessScope,
        capabilities: set[str] | None = None,
    ) -> bool:
        """Check if requester can read this memory.

        Args:
            requester: Scope of the requesting agent
            capabilities: Capabilities the requester has

        Returns:
            True if read access is granted
        """
        # Check time-based restrictions
        if self.policy.is_expired() or not self.policy.is_available():
            return False

        # Check explicit deny (takes precedence)
        if requester.agent_id and requester.agent_id in self.policy.denied_agents:
            return False
        if requester.groups & self.policy.denied_groups:
            return False
        if requester.department and requester.department in self.policy.denied_departments:
            return False

        # Check capability requirements
        if self.policy.required_capabilities:
            if not capabilities or not (capabilities >= self.policy.required_capabilities):
                return False

        # Check explicit allow (overrides level)
        if requester.agent_id and requester.agent_id in self.policy.allowed_agents:
            return True
        if requester.groups & self.policy.allowed_groups:
            return True
        if requester.department and requester.department in self.policy.allowed_departments:
            return True

        # Check access level
        return requester.matches_level(self.policy.access_level, self.owner_scope)

    def can_write(
        self,
        requester: AccessScope,
        capabilities: set[str] | None = None,
    ) -> bool:
        """Check if requester can modify this memory.

        Write access is more restrictive than read:
        - PRIVATE: Only owner
        - TEAM and above: Requires explicit allow or same scope

        Args:
            requester: Scope of the requesting agent
            capabilities: Capabilities the requester has

        Returns:
            True if write access is granted
        """
        # Must have read access first
        if not self.can_read(requester, capabilities):
            return False

        # Owner always has write access
        if requester.agent_id == self.owner_agent_id:
            return True

        # Private memories: owner only
        if self.policy.access_level == AccessLevel.PRIVATE:
            return False

        # Check explicit allow for write
        if requester.agent_id and requester.agent_id in self.policy.allowed_agents:
            return True

        # For team/department level, require same scope
        if self.policy.access_level in (AccessLevel.TEAM, AccessLevel.DEPARTMENT):
            return requester.matches_level(self.policy.access_level, self.owner_scope)

        # Organization level: any agent in org can write
        if self.policy.access_level >= AccessLevel.ORGANIZATION:
            return requester.org_id == self.owner_scope.org_id

        return False

    def record_access(self, agent_id: str) -> None:
        """Record an access event for audit."""
        self.last_accessed_by = agent_id
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize ACL for storage."""
        return {
            "memory_id": self.memory_id,
            "owner_agent_id": self.owner_agent_id,
            "owner_scope": self.owner_scope.to_filter_dict(),
            "policy": {
                "access_level": self.policy.access_level.value,
                "allowed_agents": list(self.policy.allowed_agents),
                "denied_agents": list(self.policy.denied_agents),
                "allowed_groups": list(self.policy.allowed_groups),
                "denied_groups": list(self.policy.denied_groups),
                "allowed_departments": list(self.policy.allowed_departments),
                "denied_departments": list(self.policy.denied_departments),
                "expires_at": self.policy.expires_at.isoformat() if self.policy.expires_at else None,
                "available_from": self.policy.available_from.isoformat() if self.policy.available_from else None,
                "required_capabilities": list(self.policy.required_capabilities),
            },
            "created_at": self.created_at.isoformat(),
            "last_accessed_by": self.last_accessed_by,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryACL:
        """Deserialize ACL from storage."""
        owner_scope_data = data["owner_scope"]
        owner_scope = AccessScope(
            org_id=owner_scope_data["org_id"],
            department=owner_scope_data.get("department"),
            team=owner_scope_data.get("team"),
            agent_type=owner_scope_data.get("agent_type"),
            agent_id=owner_scope_data.get("agent_id"),
            groups=set(owner_scope_data.get("groups", [])),
        )

        policy_data = data["policy"]
        policy = AccessPolicy(
            access_level=AccessLevel(policy_data["access_level"]),
            allowed_agents=set(policy_data.get("allowed_agents", [])),
            denied_agents=set(policy_data.get("denied_agents", [])),
            allowed_groups=set(policy_data.get("allowed_groups", [])),
            denied_groups=set(policy_data.get("denied_groups", [])),
            allowed_departments=set(policy_data.get("allowed_departments", [])),
            denied_departments=set(policy_data.get("denied_departments", [])),
            expires_at=datetime.fromisoformat(policy_data["expires_at"]) if policy_data.get("expires_at") else None,
            available_from=datetime.fromisoformat(policy_data["available_from"]) if policy_data.get("available_from") else None,
            required_capabilities=set(policy_data.get("required_capabilities", [])),
        )

        return cls(
            memory_id=data["memory_id"],
            owner_agent_id=data["owner_agent_id"],
            owner_scope=owner_scope,
            policy=policy,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_by=data.get("last_accessed_by"),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]) if data.get("last_accessed_at") else None,
            access_count=data.get("access_count", 0),
        )


def create_acl_for_level(
    memory_id: str,
    owner_agent_id: str,
    owner_scope: AccessScope,
    level: AccessLevel,
    **kwargs: Any,
) -> MemoryACL:
    """Convenience function to create ACL with a specific level.

    Args:
        memory_id: ID of the memory
        owner_agent_id: ID of the creating agent
        owner_scope: Scope of the owner
        level: Access level to set
        **kwargs: Additional policy parameters

    Returns:
        Configured MemoryACL
    """
    policy = AccessPolicy(access_level=level, **kwargs)
    return MemoryACL(
        memory_id=memory_id,
        owner_agent_id=owner_agent_id,
        owner_scope=owner_scope,
        policy=policy,
    )
