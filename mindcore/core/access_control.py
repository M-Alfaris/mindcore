"""
Multi-Agent Access Control and Privacy System
==============================================

Provides fine-grained access control for knowledge sharing between
multiple agents with private and shared layers.

Architecture:
------------
Knowledge is organized into three visibility tiers:

1. **Private** - Only accessible by the owner agent
   - Personal conversation history
   - Agent-specific learned preferences
   - Sensitive user data

2. **Shared** - Accessible by agents in a sharing group
   - Team knowledge bases
   - Shared customer context
   - Cross-agent handoff information

3. **Public** - Accessible by all authorized agents
   - Company-wide knowledge
   - Product documentation
   - General FAQs

Access Control:
--------------
- Each piece of knowledge has an `owner_id` (the creating agent)
- `visibility` determines the base access level
- `acl` (Access Control List) provides fine-grained permissions
- Agents must be registered and authenticated

Example:
    >>> from mindcore.core.access_control import (
    ...     AccessControlManager,
    ...     KnowledgeVisibility,
    ...     AgentRegistration
    ... )
    >>>
    >>> acm = AccessControlManager(db)
    >>>
    >>> # Register an agent
    >>> agent = acm.register_agent(
    ...     agent_id="support-agent-1",
    ...     name="Customer Support Agent",
    ...     owner_id="org-123",
    ...     groups=["support-team"]
    ... )
    >>>
    >>> # Check access
    >>> can_read = acm.can_access(
    ...     agent_id="support-agent-1",
    ...     resource_id="msg-456",
    ...     permission="read"
    ... )
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import hashlib
import secrets

from ..utils.logger import get_logger
from .schemas import KnowledgeVisibility  # Single source of truth

logger = get_logger(__name__)


class Permission(str, Enum):
    """Permission types for access control."""
    READ = "read"  # Can read/retrieve the resource
    WRITE = "write"  # Can modify the resource
    DELETE = "delete"  # Can delete the resource
    SHARE = "share"  # Can share with other agents
    ADMIN = "admin"  # Full control including permission management


@dataclass
class AgentRegistration:
    """
    Registered agent identity.

    Each agent must be registered to access the knowledge system.
    """
    agent_id: str  # Unique identifier for this agent
    name: str  # Human-readable name
    owner_id: str  # Organization or user that owns this agent

    # Group memberships for shared access
    groups: List[str] = field(default_factory=list)

    # Agent capabilities/roles
    roles: List[str] = field(default_factory=list)
    # e.g., ["support", "sales", "admin"]

    # API key hash for authentication
    api_key_hash: Optional[str] = None

    # Status
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "groups": self.groups,
            "roles": self.roles,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "metadata": self.metadata
        }


@dataclass
class AccessControlEntry:
    """
    Single entry in an Access Control List (ACL).

    Grants specific permissions to a principal (agent or group).
    """
    principal_id: str  # Agent ID or group name (prefixed with "group:")
    principal_type: str  # "agent" or "group"
    permissions: Set[Permission] = field(default_factory=set)
    granted_by: Optional[str] = None  # Agent who granted this access
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # Optional expiration

    def __post_init__(self):
        if self.granted_at is None:
            self.granted_at = datetime.now(timezone.utc)
        # Convert string permissions to enum
        if self.permissions and isinstance(list(self.permissions)[0], str):
            self.permissions = {Permission(p) for p in self.permissions}

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_permission(self, permission: Permission) -> bool:
        if self.is_expired():
            return False
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "principal_type": self.principal_type,
            "permissions": [p.value for p in self.permissions],
            "granted_by": self.granted_by,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class KnowledgeAccessPolicy:
    """
    Access policy attached to a knowledge resource.

    Combines visibility with fine-grained ACL.
    """
    resource_id: str  # ID of the resource (message_id, document_id, etc.)
    resource_type: str  # "message", "document", "summary", "vector"
    owner_id: str  # Agent that created this resource
    owner_org: str  # Organization the owner belongs to

    # Base visibility
    visibility: KnowledgeVisibility = KnowledgeVisibility.PRIVATE

    # Fine-grained access control
    acl: List[AccessControlEntry] = field(default_factory=list)

    # Sharing groups (for SHARED visibility)
    sharing_groups: List[str] = field(default_factory=list)

    # Audit
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.modified_at is None:
            self.modified_at = self.created_at

    def add_acl_entry(
        self,
        principal_id: str,
        principal_type: str,
        permissions: Set[Permission],
        granted_by: str,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Add an ACL entry for a principal."""
        # Remove existing entry for this principal
        self.acl = [e for e in self.acl if e.principal_id != principal_id]

        entry = AccessControlEntry(
            principal_id=principal_id,
            principal_type=principal_type,
            permissions=permissions,
            granted_by=granted_by,
            expires_at=expires_at
        )
        self.acl.append(entry)
        self.modified_at = datetime.now(timezone.utc)

    def remove_acl_entry(self, principal_id: str) -> bool:
        """Remove ACL entry for a principal."""
        original_len = len(self.acl)
        self.acl = [e for e in self.acl if e.principal_id != principal_id]
        if len(self.acl) < original_len:
            self.modified_at = datetime.now(timezone.utc)
            return True
        return False

    def get_acl_entry(self, principal_id: str) -> Optional[AccessControlEntry]:
        """Get ACL entry for a principal."""
        for entry in self.acl:
            if entry.principal_id == principal_id:
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "owner_id": self.owner_id,
            "owner_org": self.owner_org,
            "visibility": self.visibility.value,
            "acl": [e.to_dict() for e in self.acl],
            "sharing_groups": self.sharing_groups,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None
        }


@dataclass
class AccessAuditLog:
    """Audit log entry for access attempts."""
    log_id: str
    timestamp: datetime
    agent_id: str
    resource_id: str
    resource_type: str
    action: str  # "read", "write", "delete", "share"
    success: bool
    denial_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "action": self.action,
            "success": self.success,
            "denial_reason": self.denial_reason,
            "metadata": self.metadata
        }


class AccessControlManager:
    """
    Manages access control for multi-agent knowledge sharing.

    Handles:
    - Agent registration and authentication
    - Permission checking
    - Knowledge visibility enforcement
    - Audit logging
    """

    def __init__(
        self,
        database: Any = None,
        enable_audit_logging: bool = True,
        default_visibility: KnowledgeVisibility = KnowledgeVisibility.PRIVATE
    ):
        """
        Initialize access control manager.

        Args:
            database: Database manager for persistence
            enable_audit_logging: Whether to log access attempts
            default_visibility: Default visibility for new resources
        """
        self._db = database
        self._audit_enabled = enable_audit_logging
        self._default_visibility = default_visibility

        # In-memory caches (for performance)
        self._agents: Dict[str, AgentRegistration] = {}
        self._policies: Dict[str, KnowledgeAccessPolicy] = {}

        logger.info("AccessControlManager initialized")

    # =========================================================================
    # Agent Management
    # =========================================================================

    def register_agent(
        self,
        agent_id: str,
        name: str,
        owner_id: str,
        groups: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[AgentRegistration, str]:
        """
        Register a new agent.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            owner_id: Organization or user that owns this agent
            groups: Group memberships for shared access
            roles: Agent capabilities/roles
            metadata: Additional metadata

        Returns:
            Tuple of (AgentRegistration, api_key)
        """
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        api_key_hash = self._hash_api_key(api_key)

        agent = AgentRegistration(
            agent_id=agent_id,
            name=name,
            owner_id=owner_id,
            groups=groups or [],
            roles=roles or [],
            api_key_hash=api_key_hash,
            metadata=metadata or {}
        )

        self._agents[agent_id] = agent

        # Persist to database
        if self._db:
            self._persist_agent(agent)

        logger.info(f"Registered agent: {agent_id} (owner: {owner_id})")
        return agent, api_key

    def authenticate_agent(self, agent_id: str, api_key: str) -> Optional[AgentRegistration]:
        """
        Authenticate an agent by API key.

        Args:
            agent_id: Agent identifier
            api_key: API key to verify

        Returns:
            AgentRegistration if authenticated, None otherwise
        """
        agent = self._agents.get(agent_id)
        if not agent:
            # Try loading from database
            agent = self._load_agent(agent_id)
            if agent:
                self._agents[agent_id] = agent

        if not agent or not agent.is_active:
            return None

        # Verify API key
        if not agent.api_key_hash:
            return None

        if self._hash_api_key(api_key) != agent.api_key_hash:
            return None

        # Update last active
        agent.last_active_at = datetime.now(timezone.utc)

        return agent

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID."""
        if agent_id in self._agents:
            return self._agents[agent_id]

        agent = self._load_agent(agent_id)
        if agent:
            self._agents[agent_id] = agent
        return agent

    def add_agent_to_group(self, agent_id: str, group: str) -> bool:
        """Add an agent to a sharing group."""
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        if group not in agent.groups:
            agent.groups.append(group)
            if self._db:
                self._persist_agent(agent)
            return True
        return False

    def remove_agent_from_group(self, agent_id: str, group: str) -> bool:
        """Remove an agent from a sharing group."""
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        if group in agent.groups:
            agent.groups.remove(group)
            if self._db:
                self._persist_agent(agent)
            return True
        return False

    # =========================================================================
    # Policy Management
    # =========================================================================

    def create_policy(
        self,
        resource_id: str,
        resource_type: str,
        owner_id: str,
        owner_org: str,
        visibility: Optional[KnowledgeVisibility] = None,
        sharing_groups: Optional[List[str]] = None
    ) -> KnowledgeAccessPolicy:
        """
        Create an access policy for a resource.

        Args:
            resource_id: ID of the resource
            resource_type: Type of resource ("message", "document", etc.)
            owner_id: Agent that owns this resource
            owner_org: Organization the owner belongs to
            visibility: Visibility level (defaults to manager default)
            sharing_groups: Groups that can access (for SHARED visibility)

        Returns:
            Created policy
        """
        policy = KnowledgeAccessPolicy(
            resource_id=resource_id,
            resource_type=resource_type,
            owner_id=owner_id,
            owner_org=owner_org,
            visibility=visibility or self._default_visibility,
            sharing_groups=sharing_groups or []
        )

        self._policies[resource_id] = policy

        if self._db:
            self._persist_policy(policy)

        return policy

    def get_policy(self, resource_id: str) -> Optional[KnowledgeAccessPolicy]:
        """Get policy for a resource."""
        if resource_id in self._policies:
            return self._policies[resource_id]

        policy = self._load_policy(resource_id)
        if policy:
            self._policies[resource_id] = policy
        return policy

    def update_visibility(
        self,
        resource_id: str,
        visibility: KnowledgeVisibility,
        requesting_agent_id: str
    ) -> bool:
        """
        Update visibility of a resource.

        Only owner or agents with ADMIN permission can change visibility.
        """
        policy = self.get_policy(resource_id)
        if not policy:
            return False

        # Check permission
        if not self._can_modify_policy(requesting_agent_id, policy):
            self._log_access(
                requesting_agent_id, resource_id, policy.resource_type,
                "update_visibility", False, "Permission denied"
            )
            return False

        policy.visibility = visibility
        policy.modified_at = datetime.now(timezone.utc)

        if self._db:
            self._persist_policy(policy)

        return True

    def grant_access(
        self,
        resource_id: str,
        principal_id: str,
        principal_type: str,
        permissions: Set[Permission],
        granting_agent_id: str,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Grant access to a resource.

        Args:
            resource_id: Resource to grant access to
            principal_id: Agent or group to grant access to
            principal_type: "agent" or "group"
            permissions: Set of permissions to grant
            granting_agent_id: Agent granting the access
            expires_at: Optional expiration time

        Returns:
            True if access was granted
        """
        policy = self.get_policy(resource_id)
        if not policy:
            return False

        # Check if granting agent has SHARE or ADMIN permission
        if not self.can_access(granting_agent_id, resource_id, Permission.SHARE):
            self._log_access(
                granting_agent_id, resource_id, policy.resource_type,
                "grant_access", False, "No share permission"
            )
            return False

        policy.add_acl_entry(
            principal_id=principal_id,
            principal_type=principal_type,
            permissions=permissions,
            granted_by=granting_agent_id,
            expires_at=expires_at
        )

        if self._db:
            self._persist_policy(policy)

        logger.info(
            f"Access granted: {principal_id} -> {resource_id} "
            f"({[p.value for p in permissions]})"
        )
        return True

    def revoke_access(
        self,
        resource_id: str,
        principal_id: str,
        revoking_agent_id: str
    ) -> bool:
        """Revoke access from a principal."""
        policy = self.get_policy(resource_id)
        if not policy:
            return False

        # Check if revoking agent has ADMIN permission or is the owner
        if not self._can_modify_policy(revoking_agent_id, policy):
            return False

        if policy.remove_acl_entry(principal_id):
            if self._db:
                self._persist_policy(policy)
            return True
        return False

    # =========================================================================
    # Access Checking
    # =========================================================================

    def can_access(
        self,
        agent_id: str,
        resource_id: str,
        permission: Permission = Permission.READ
    ) -> bool:
        """
        Check if an agent can access a resource with given permission.

        Args:
            agent_id: Agent requesting access
            resource_id: Resource to access
            permission: Required permission

        Returns:
            True if access is allowed
        """
        agent = self.get_agent(agent_id)
        policy = self.get_policy(resource_id)

        if not agent or not policy:
            self._log_access(
                agent_id, resource_id, "unknown",
                permission.value, False, "Agent or policy not found"
            )
            return False

        if not agent.is_active:
            self._log_access(
                agent_id, resource_id, policy.resource_type,
                permission.value, False, "Agent inactive"
            )
            return False

        # Check if owner (owners have all permissions)
        if policy.owner_id == agent_id:
            self._log_access(
                agent_id, resource_id, policy.resource_type,
                permission.value, True
            )
            return True

        # Check visibility-based access
        allowed = self._check_visibility_access(agent, policy, permission)

        if allowed:
            self._log_access(
                agent_id, resource_id, policy.resource_type,
                permission.value, True
            )
        else:
            self._log_access(
                agent_id, resource_id, policy.resource_type,
                permission.value, False, "Access denied by policy"
            )

        return allowed

    def _check_visibility_access(
        self,
        agent: AgentRegistration,
        policy: KnowledgeAccessPolicy,
        permission: Permission
    ) -> bool:
        """Check access based on visibility and ACL."""

        # PRIVATE: Only owner can access (already checked above)
        if policy.visibility == KnowledgeVisibility.PRIVATE:
            # Check explicit ACL
            return self._check_acl_access(agent, policy, permission)

        # SHARED: Check if agent is in sharing groups
        if policy.visibility == KnowledgeVisibility.SHARED:
            # Same organization gets read access
            if (agent.owner_id == policy.owner_org and
                    permission == Permission.READ):
                return True

            # Check sharing groups
            agent_groups = set(agent.groups)
            sharing_groups = set(policy.sharing_groups)
            if agent_groups & sharing_groups:
                # In a sharing group - check if permission is allowed
                if permission == Permission.READ:
                    return True
                # Check ACL for write/delete/share
                return self._check_acl_access(agent, policy, permission)

            # Check explicit ACL
            return self._check_acl_access(agent, policy, permission)

        # PUBLIC: All authorized agents can read
        if policy.visibility == KnowledgeVisibility.PUBLIC:
            if permission == Permission.READ:
                return True
            # For other permissions, check ACL
            return self._check_acl_access(agent, policy, permission)

        return False

    def _check_acl_access(
        self,
        agent: AgentRegistration,
        policy: KnowledgeAccessPolicy,
        permission: Permission
    ) -> bool:
        """Check ACL for specific permission."""
        # Check direct agent ACL
        entry = policy.get_acl_entry(agent.agent_id)
        if entry and entry.has_permission(permission):
            return True

        # Check group ACLs
        for group in agent.groups:
            entry = policy.get_acl_entry(f"group:{group}")
            if entry and entry.has_permission(permission):
                return True

        return False

    def _can_modify_policy(self, agent_id: str, policy: KnowledgeAccessPolicy) -> bool:
        """Check if agent can modify a policy."""
        # Owner can always modify
        if policy.owner_id == agent_id:
            return True

        # Check for ADMIN permission
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        return self._check_acl_access(agent, policy, Permission.ADMIN)

    # =========================================================================
    # Knowledge Sharing
    # =========================================================================

    def share_with_agent(
        self,
        resource_id: str,
        target_agent_id: str,
        sharing_agent_id: str,
        permissions: Optional[Set[Permission]] = None
    ) -> bool:
        """
        Share a resource with another agent.

        Args:
            resource_id: Resource to share
            target_agent_id: Agent to share with
            sharing_agent_id: Agent doing the sharing
            permissions: Permissions to grant (defaults to READ)

        Returns:
            True if sharing succeeded
        """
        if permissions is None:
            permissions = {Permission.READ}

        return self.grant_access(
            resource_id=resource_id,
            principal_id=target_agent_id,
            principal_type="agent",
            permissions=permissions,
            granting_agent_id=sharing_agent_id
        )

    def share_with_group(
        self,
        resource_id: str,
        group_name: str,
        sharing_agent_id: str,
        permissions: Optional[Set[Permission]] = None
    ) -> bool:
        """
        Share a resource with a group.

        Args:
            resource_id: Resource to share
            group_name: Group to share with
            sharing_agent_id: Agent doing the sharing
            permissions: Permissions to grant (defaults to READ)

        Returns:
            True if sharing succeeded
        """
        if permissions is None:
            permissions = {Permission.READ}

        return self.grant_access(
            resource_id=resource_id,
            principal_id=f"group:{group_name}",
            principal_type="group",
            permissions=permissions,
            granting_agent_id=sharing_agent_id
        )

    def get_accessible_resources(
        self,
        agent_id: str,
        resource_type: Optional[str] = None,
        permission: Permission = Permission.READ
    ) -> List[str]:
        """
        Get all resources an agent can access.

        Args:
            agent_id: Agent to check
            resource_type: Optional filter by resource type
            permission: Required permission level

        Returns:
            List of accessible resource IDs
        """
        accessible = []

        for resource_id, policy in self._policies.items():
            if resource_type and policy.resource_type != resource_type:
                continue

            if self.can_access(agent_id, resource_id, permission):
                accessible.append(resource_id)

        return accessible

    # =========================================================================
    # Audit Logging
    # =========================================================================

    def _log_access(
        self,
        agent_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        success: bool,
        denial_reason: Optional[str] = None
    ) -> None:
        """Log an access attempt."""
        if not self._audit_enabled:
            return

        log_entry = AccessAuditLog(
            log_id=secrets.token_hex(16),
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            success=success,
            denial_reason=denial_reason
        )

        if self._db:
            self._persist_audit_log(log_entry)

        if not success:
            logger.warning(
                f"Access denied: agent={agent_id} resource={resource_id} "
                f"action={action} reason={denial_reason}"
            )

    def get_audit_logs(
        self,
        agent_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AccessAuditLog]:
        """Get audit logs with optional filters."""
        # Would query database in real implementation
        return []

    # =========================================================================
    # Persistence Helpers
    # =========================================================================

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _persist_agent(self, agent: AgentRegistration) -> None:
        """Persist agent to database."""
        if not self._db:
            return

        try:
            # Use JSON storage for agent data
            agent_data = agent.to_dict()
            agent_data['api_key_hash'] = agent.api_key_hash

            if hasattr(self._db, 'execute'):
                # SQLite/PostgreSQL with raw execute
                self._db.execute("""
                    INSERT OR REPLACE INTO agents (agent_id, data, created_at)
                    VALUES (?, ?, ?)
                """, (agent.agent_id, str(agent_data), agent.created_at))
            elif hasattr(self._db, 'upsert_json'):
                # If db manager has JSON helper
                self._db.upsert_json('agents', agent.agent_id, agent_data)
            else:
                logger.warning("Database doesn't support agent persistence")
        except Exception as e:
            logger.error(f"Failed to persist agent {agent.agent_id}: {e}")

    def _load_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Load agent from database."""
        if not self._db:
            return None

        try:
            if hasattr(self._db, 'execute'):
                result = self._db.execute(
                    "SELECT data FROM agents WHERE agent_id = ?",
                    (agent_id,)
                )
                if result:
                    import ast
                    data = ast.literal_eval(result[0][0])
                    return AgentRegistration(
                        agent_id=data['agent_id'],
                        name=data['name'],
                        owner_id=data['owner_id'],
                        groups=data.get('groups', []),
                        roles=data.get('roles', []),
                        api_key_hash=data.get('api_key_hash'),
                        is_active=data.get('is_active', True),
                        metadata=data.get('metadata', {})
                    )
            elif hasattr(self._db, 'get_json'):
                data = self._db.get_json('agents', agent_id)
                if data:
                    return AgentRegistration(**data)
        except Exception as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")

        return None

    def _persist_policy(self, policy: KnowledgeAccessPolicy) -> None:
        """Persist policy to database."""
        if not self._db:
            return

        try:
            policy_data = policy.to_dict()

            if hasattr(self._db, 'execute'):
                self._db.execute("""
                    INSERT OR REPLACE INTO access_policies (resource_id, data, modified_at)
                    VALUES (?, ?, ?)
                """, (policy.resource_id, str(policy_data), policy.modified_at))
            elif hasattr(self._db, 'upsert_json'):
                self._db.upsert_json('access_policies', policy.resource_id, policy_data)
        except Exception as e:
            logger.error(f"Failed to persist policy {policy.resource_id}: {e}")

    def _load_policy(self, resource_id: str) -> Optional[KnowledgeAccessPolicy]:
        """Load policy from database."""
        if not self._db:
            return None

        try:
            if hasattr(self._db, 'execute'):
                result = self._db.execute(
                    "SELECT data FROM access_policies WHERE resource_id = ?",
                    (resource_id,)
                )
                if result:
                    import ast
                    data = ast.literal_eval(result[0][0])
                    return KnowledgeAccessPolicy(
                        resource_id=data['resource_id'],
                        resource_type=data['resource_type'],
                        owner_id=data['owner_id'],
                        owner_org=data['owner_org'],
                        visibility=KnowledgeVisibility(data['visibility']),
                        sharing_groups=data.get('sharing_groups', []),
                        acl=[AccessControlEntry(**e) for e in data.get('acl', [])]
                    )
            elif hasattr(self._db, 'get_json'):
                data = self._db.get_json('access_policies', resource_id)
                if data:
                    data['visibility'] = KnowledgeVisibility(data['visibility'])
                    data['acl'] = [AccessControlEntry(**e) for e in data.get('acl', [])]
                    return KnowledgeAccessPolicy(**data)
        except Exception as e:
            logger.error(f"Failed to load policy {resource_id}: {e}")

        return None

    def _persist_audit_log(self, log: AccessAuditLog) -> None:
        """Persist audit log to database."""
        if not self._db:
            return

        try:
            if hasattr(self._db, 'execute'):
                self._db.execute("""
                    INSERT INTO audit_logs (log_id, timestamp, agent_id, resource_id, action, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (log.log_id, log.timestamp, log.agent_id, log.resource_id, log.action, log.success))
        except Exception as e:
            logger.debug(f"Failed to persist audit log: {e}")  # Debug level - non-critical


# =============================================================================
# Casbin Integration (Optional - requires: pip install casbin)
# =============================================================================

class CasbinAccessControlManager(AccessControlManager):
    """
    Enhanced access control using Casbin for RBAC/ABAC.

    Casbin provides a powerful and flexible authorization mechanism that supports:
    - Role-Based Access Control (RBAC)
    - Attribute-Based Access Control (ABAC)
    - ACL (Access Control Lists)

    This class extends the base AccessControlManager with Casbin-powered
    policy enforcement while maintaining backwards compatibility.

    Install: pip install casbin>=1.50.0

    Example:
        >>> from mindcore.core.access_control import CasbinAccessControlManager
        >>>
        >>> # Use with built-in RBAC model
        >>> acm = CasbinAccessControlManager.with_rbac()
        >>>
        >>> # Or with custom model
        >>> acm = CasbinAccessControlManager(
        ...     model_path="./config/model.conf",
        ...     policy_path="./config/policy.csv"
        ... )

    Models:
        - RBAC: Role-based with hierarchical roles
        - RBAC with domains: Multi-tenant role separation
        - ABAC: Attribute-based for fine-grained control

    See: https://casbin.org/docs/overview
    """

    def __init__(
        self,
        database: Any = None,
        model_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        model_text: Optional[str] = None,
        enable_audit_logging: bool = True,
        default_visibility: KnowledgeVisibility = KnowledgeVisibility.PRIVATE
    ):
        """
        Initialize Casbin-powered access control.

        Args:
            database: Database manager for persistence
            model_path: Path to Casbin model file (.conf)
            policy_path: Path to Casbin policy file (.csv)
            model_text: Model definition as string (alternative to model_path)
            enable_audit_logging: Whether to log access attempts
            default_visibility: Default visibility for new resources
        """
        super().__init__(database, enable_audit_logging, default_visibility)

        try:
            import casbin
        except ImportError:
            raise ImportError(
                "Casbin not installed. Run: pip install casbin>=1.50.0"
            )

        self._casbin = casbin
        self._enforcer = None

        if model_path and policy_path:
            self._enforcer = casbin.Enforcer(model_path, policy_path)
        elif model_text:
            # Create enforcer from string model
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
                f.write(model_text)
                temp_model_path = f.name
            self._enforcer = casbin.Enforcer(temp_model_path)

        logger.info("CasbinAccessControlManager initialized")

    @classmethod
    def with_rbac(
        cls,
        database: Any = None,
        enable_audit_logging: bool = True
    ) -> "CasbinAccessControlManager":
        """
        Create manager with built-in RBAC model.

        RBAC Model:
        - Supports role hierarchy (e.g., admin inherits from manager)
        - Policies: agent, resource, action
        - Role assignments: agent, role
        """
        rbac_model = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
"""
        return cls(
            database=database,
            model_text=rbac_model,
            enable_audit_logging=enable_audit_logging
        )

    @classmethod
    def with_rbac_domains(
        cls,
        database: Any = None,
        enable_audit_logging: bool = True
    ) -> "CasbinAccessControlManager":
        """
        Create manager with RBAC + domains (multi-tenant).

        Supports different role sets per organization/tenant.
        """
        rbac_domains_model = """
[request_definition]
r = sub, dom, obj, act

[policy_definition]
p = sub, dom, obj, act

[role_definition]
g = _, _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub, r.dom) && r.dom == p.dom && r.obj == p.obj && r.act == p.act
"""
        return cls(
            database=database,
            model_text=rbac_domains_model,
            enable_audit_logging=enable_audit_logging
        )

    def add_policy(self, *args) -> bool:
        """
        Add a policy rule.

        For RBAC: add_policy(agent_id, resource_id, action)
        For RBAC domains: add_policy(agent_id, domain, resource_id, action)
        """
        if not self._enforcer:
            return False
        return self._enforcer.add_policy(*args)

    def remove_policy(self, *args) -> bool:
        """Remove a policy rule."""
        if not self._enforcer:
            return False
        return self._enforcer.remove_policy(*args)

    def add_role_for_agent(self, agent_id: str, role: str, domain: Optional[str] = None) -> bool:
        """
        Assign a role to an agent.

        Args:
            agent_id: Agent identifier
            role: Role name (e.g., "admin", "support", "viewer")
            domain: Optional domain/tenant for multi-tenant RBAC
        """
        if not self._enforcer:
            return False
        if domain:
            return self._enforcer.add_grouping_policy(agent_id, role, domain)
        return self._enforcer.add_grouping_policy(agent_id, role)

    def remove_role_from_agent(self, agent_id: str, role: str, domain: Optional[str] = None) -> bool:
        """Remove a role from an agent."""
        if not self._enforcer:
            return False
        if domain:
            return self._enforcer.remove_grouping_policy(agent_id, role, domain)
        return self._enforcer.remove_grouping_policy(agent_id, role)

    def get_roles_for_agent(self, agent_id: str, domain: Optional[str] = None) -> List[str]:
        """Get all roles for an agent."""
        if not self._enforcer:
            return []
        if domain:
            return self._enforcer.get_roles_for_user_in_domain(agent_id, domain)
        return self._enforcer.get_roles_for_user(agent_id)

    def get_agents_for_role(self, role: str, domain: Optional[str] = None) -> List[str]:
        """Get all agents with a specific role."""
        if not self._enforcer:
            return []
        if domain:
            return self._enforcer.get_users_for_role_in_domain(role, domain)
        return self._enforcer.get_users_for_role(role)

    def enforce(self, *args) -> bool:
        """
        Check if a request should be allowed.

        For RBAC: enforce(agent_id, resource_id, action)
        For RBAC domains: enforce(agent_id, domain, resource_id, action)

        Returns:
            True if access is allowed by Casbin policy
        """
        if not self._enforcer:
            return False
        return self._enforcer.enforce(*args)

    def can_access(
        self,
        agent_id: str,
        resource_id: str,
        permission: Permission = Permission.READ,
        domain: Optional[str] = None
    ) -> bool:
        """
        Check access using both Casbin and parent's visibility-based checks.

        First checks Casbin policies, then falls back to visibility-based
        access control from the parent class.
        """
        # Try Casbin first if enforcer is configured
        if self._enforcer:
            if domain:
                if self.enforce(agent_id, domain, resource_id, permission.value):
                    self._log_access(
                        agent_id, resource_id, "resource",
                        permission.value, True
                    )
                    return True
            else:
                if self.enforce(agent_id, resource_id, permission.value):
                    self._log_access(
                        agent_id, resource_id, "resource",
                        permission.value, True
                    )
                    return True

        # Fall back to parent's visibility-based checks
        return super().can_access(agent_id, resource_id, permission)

    def save_policy(self) -> bool:
        """Save current policies to the adapter (file or database)."""
        if not self._enforcer:
            return False
        return self._enforcer.save_policy()

    def load_policy(self) -> None:
        """Reload policies from the adapter."""
        if self._enforcer:
            self._enforcer.load_policy()
