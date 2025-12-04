"""Tests for AccessControlManager - Multi-agent access control system."""

from datetime import datetime, timedelta, timezone

import pytest

from mindcore.core.access_control import (
    AccessControlEntry,
    AccessControlManager,
    AccessAuditLog,
    AgentRegistration,
    KnowledgeAccessPolicy,
    Permission,
)
from mindcore.core.schemas import KnowledgeVisibility


class TestAgentRegistration:
    """Test AgentRegistration dataclass."""

    def test_agent_registration_creation(self):
        """Test creating an agent registration."""
        agent = AgentRegistration(
            agent_id="agent_001",
            name="Support Agent",
            owner_id="org_123",
            groups=["support-team"],
            roles=["support"],
        )

        assert agent.agent_id == "agent_001"
        assert agent.name == "Support Agent"
        assert agent.owner_id == "org_123"
        assert "support-team" in agent.groups
        assert "support" in agent.roles
        assert agent.is_active is True
        assert agent.created_at is not None

    def test_agent_registration_defaults(self):
        """Test agent registration with defaults."""
        agent = AgentRegistration(
            agent_id="agent_002",
            name="Test Agent",
            owner_id="org_123",
        )

        assert agent.groups == []
        assert agent.roles == []
        assert agent.api_key_hash is None
        assert agent.metadata == {}

    def test_agent_to_dict(self):
        """Test agent serialization."""
        agent = AgentRegistration(
            agent_id="agent_003",
            name="Test Agent",
            owner_id="org_123",
            groups=["team-a"],
        )

        data = agent.to_dict()

        assert data["agent_id"] == "agent_003"
        assert data["name"] == "Test Agent"
        assert data["groups"] == ["team-a"]
        assert "created_at" in data


class TestAccessControlEntry:
    """Test AccessControlEntry dataclass."""

    def test_acl_entry_creation(self):
        """Test creating an ACL entry."""
        entry = AccessControlEntry(
            principal_id="agent_001",
            principal_type="agent",
            permissions={Permission.READ, Permission.WRITE},
            granted_by="admin_001",
        )

        assert entry.principal_id == "agent_001"
        assert entry.principal_type == "agent"
        assert Permission.READ in entry.permissions
        assert Permission.WRITE in entry.permissions
        assert entry.granted_at is not None
        assert entry.expires_at is None

    def test_acl_entry_with_expiration(self):
        """Test ACL entry with expiration."""
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        entry = AccessControlEntry(
            principal_id="agent_001",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="admin_001",
            expires_at=expires,
        )

        assert entry.expires_at == expires
        assert entry.is_expired() is False

    def test_acl_entry_expired(self):
        """Test expired ACL entry."""
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        entry = AccessControlEntry(
            principal_id="agent_001",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="admin_001",
            expires_at=expired_time,
        )

        assert entry.is_expired() is True
        assert entry.has_permission(Permission.READ) is False

    def test_acl_entry_has_permission(self):
        """Test permission checking."""
        entry = AccessControlEntry(
            principal_id="agent_001",
            principal_type="agent",
            permissions={Permission.READ, Permission.WRITE},
            granted_by="admin_001",
        )

        assert entry.has_permission(Permission.READ) is True
        assert entry.has_permission(Permission.WRITE) is True
        assert entry.has_permission(Permission.DELETE) is False

    def test_acl_entry_admin_has_all_permissions(self):
        """Test that ADMIN permission grants all access."""
        entry = AccessControlEntry(
            principal_id="admin_001",
            principal_type="agent",
            permissions={Permission.ADMIN},
            granted_by="system",
        )

        assert entry.has_permission(Permission.READ) is True
        assert entry.has_permission(Permission.WRITE) is True
        assert entry.has_permission(Permission.DELETE) is True
        assert entry.has_permission(Permission.SHARE) is True

    def test_acl_entry_to_dict(self):
        """Test ACL entry serialization."""
        entry = AccessControlEntry(
            principal_id="agent_001",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="admin_001",
        )

        data = entry.to_dict()

        assert data["principal_id"] == "agent_001"
        assert "read" in data["permissions"]
        assert "granted_at" in data


class TestKnowledgeAccessPolicy:
    """Test KnowledgeAccessPolicy dataclass."""

    def test_policy_creation(self):
        """Test creating an access policy."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
            sharing_groups=["support-team"],
        )

        assert policy.resource_id == "msg_001"
        assert policy.resource_type == "message"
        assert policy.visibility == KnowledgeVisibility.SHARED
        assert "support-team" in policy.sharing_groups
        assert policy.created_at is not None

    def test_policy_add_acl_entry(self):
        """Test adding ACL entry to policy."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        assert len(policy.acl) == 1
        assert policy.acl[0].principal_id == "agent_002"

    def test_policy_add_acl_entry_replaces_existing(self):
        """Test that adding ACL entry for same principal replaces existing."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        # Add first entry
        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        # Add second entry for same principal
        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ, Permission.WRITE},
            granted_by="agent_001",
        )

        assert len(policy.acl) == 1
        assert Permission.WRITE in policy.acl[0].permissions

    def test_policy_remove_acl_entry(self):
        """Test removing ACL entry from policy."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        result = policy.remove_acl_entry("agent_002")

        assert result is True
        assert len(policy.acl) == 0

    def test_policy_get_acl_entry(self):
        """Test getting ACL entry from policy."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        entry = policy.get_acl_entry("agent_002")
        assert entry is not None
        assert entry.principal_id == "agent_002"

        missing = policy.get_acl_entry("agent_999")
        assert missing is None

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = KnowledgeAccessPolicy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
        )

        data = policy.to_dict()

        assert data["resource_id"] == "msg_001"
        assert data["visibility"] == "shared"
        assert "created_at" in data


class TestAccessControlManager:
    """Test AccessControlManager class."""

    @pytest.fixture
    def acm(self):
        """Create AccessControlManager without database."""
        return AccessControlManager(database=None, enable_audit_logging=True)

    @pytest.fixture
    def registered_agent(self, acm):
        """Register and return a test agent."""
        agent, api_key = acm.register_agent(
            agent_id="agent_001",
            name="Test Agent",
            owner_id="org_123",
            groups=["team-a"],
            roles=["support"],
        )
        return agent, api_key

    def test_manager_initialization(self, acm):
        """Test manager initializes correctly."""
        assert acm._audit_enabled is True
        assert acm._default_visibility == KnowledgeVisibility.PRIVATE

    def test_register_agent(self, acm):
        """Test agent registration."""
        agent, api_key = acm.register_agent(
            agent_id="agent_new",
            name="New Agent",
            owner_id="org_123",
            groups=["team-a"],
        )

        assert agent.agent_id == "agent_new"
        assert agent.name == "New Agent"
        assert api_key is not None
        assert len(api_key) > 0
        assert agent.api_key_hash is not None

    def test_authenticate_agent_success(self, acm, registered_agent):
        """Test successful agent authentication."""
        agent, api_key = registered_agent

        authenticated = acm.authenticate_agent("agent_001", api_key)

        assert authenticated is not None
        assert authenticated.agent_id == "agent_001"
        assert authenticated.last_active_at is not None

    def test_authenticate_agent_wrong_key(self, acm, registered_agent):
        """Test authentication with wrong API key."""
        agent, _ = registered_agent

        authenticated = acm.authenticate_agent("agent_001", "wrong_key")

        assert authenticated is None

    def test_authenticate_agent_not_found(self, acm):
        """Test authentication for non-existent agent."""
        authenticated = acm.authenticate_agent("nonexistent", "any_key")
        assert authenticated is None

    def test_get_agent(self, acm, registered_agent):
        """Test getting agent by ID."""
        agent, _ = registered_agent

        retrieved = acm.get_agent("agent_001")

        assert retrieved is not None
        assert retrieved.agent_id == "agent_001"

    def test_add_agent_to_group(self, acm, registered_agent):
        """Test adding agent to group."""
        agent, _ = registered_agent

        result = acm.add_agent_to_group("agent_001", "team-b")

        assert result is True
        assert "team-b" in acm.get_agent("agent_001").groups

    def test_add_agent_to_group_already_member(self, acm, registered_agent):
        """Test adding agent to group they're already in."""
        agent, _ = registered_agent

        result = acm.add_agent_to_group("agent_001", "team-a")

        assert result is False  # Already a member

    def test_remove_agent_from_group(self, acm, registered_agent):
        """Test removing agent from group."""
        agent, _ = registered_agent

        result = acm.remove_agent_from_group("agent_001", "team-a")

        assert result is True
        assert "team-a" not in acm.get_agent("agent_001").groups

    def test_create_policy(self, acm, registered_agent):
        """Test creating access policy."""
        agent, _ = registered_agent

        policy = acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
            sharing_groups=["team-a"],
        )

        assert policy.resource_id == "msg_001"
        assert policy.visibility == KnowledgeVisibility.SHARED

    def test_get_policy(self, acm, registered_agent):
        """Test getting policy by resource ID."""
        agent, _ = registered_agent

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        policy = acm.get_policy("msg_001")

        assert policy is not None
        assert policy.resource_id == "msg_001"

    def test_can_access_owner(self, acm, registered_agent):
        """Test owner always has access."""
        agent, _ = registered_agent

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        assert acm.can_access("agent_001", "msg_001", Permission.READ) is True
        assert acm.can_access("agent_001", "msg_001", Permission.WRITE) is True
        assert acm.can_access("agent_001", "msg_001", Permission.DELETE) is True

    def test_can_access_private_denied(self, acm):
        """Test private visibility denies non-owner access."""
        # Register two agents
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_123")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is False

    def test_can_access_shared_same_group(self, acm):
        """Test shared visibility allows group access."""
        acm.register_agent("agent_001", "Agent 1", "org_123", groups=["team-a"])
        acm.register_agent("agent_002", "Agent 2", "org_123", groups=["team-a"])

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
            sharing_groups=["team-a"],
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True

    def test_can_access_shared_different_group(self, acm):
        """Test shared visibility denies different group access from different org."""
        # Agents must be in DIFFERENT organizations for group-based denial to apply
        # Same org gets read access regardless of groups (see _check_visibility_access)
        acm.register_agent("agent_001", "Agent 1", "org_123", groups=["team-a"])
        acm.register_agent("agent_002", "Agent 2", "org_456", groups=["team-b"])  # Different org

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
            sharing_groups=["team-a"],
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is False

    def test_can_access_public(self, acm):
        """Test public visibility allows read access."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PUBLIC,
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True
        # Write access still denied for non-owner
        assert acm.can_access("agent_002", "msg_001", Permission.WRITE) is False

    def test_can_access_via_acl(self, acm):
        """Test access via explicit ACL entry."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        policy = acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        # Grant explicit access via ACL
        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ, Permission.WRITE},
            granted_by="agent_001",
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True
        assert acm.can_access("agent_002", "msg_001", Permission.WRITE) is True
        assert acm.can_access("agent_002", "msg_001", Permission.DELETE) is False

    def test_can_access_via_group_acl(self, acm):
        """Test access via group ACL entry."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456", groups=["external-team"])

        policy = acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        # Grant access to group
        policy.add_acl_entry(
            principal_id="group:external-team",
            principal_type="group",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True

    def test_grant_access(self, acm):
        """Test granting access to a resource."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        # Owner grants access
        result = acm.grant_access(
            resource_id="msg_001",
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granting_agent_id="agent_001",
        )

        assert result is True
        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True

    def test_grant_access_without_permission(self, acm):
        """Test granting access fails without SHARE permission."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")
        acm.register_agent("agent_003", "Agent 3", "org_789")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        # agent_002 tries to grant access (not owner, no SHARE permission)
        result = acm.grant_access(
            resource_id="msg_001",
            principal_id="agent_003",
            principal_type="agent",
            permissions={Permission.READ},
            granting_agent_id="agent_002",
        )

        assert result is False

    def test_revoke_access(self, acm):
        """Test revoking access from a resource."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        policy = acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        # Grant then revoke
        policy.add_acl_entry(
            principal_id="agent_002",
            principal_type="agent",
            permissions={Permission.READ},
            granted_by="agent_001",
        )

        result = acm.revoke_access("msg_001", "agent_002", "agent_001")

        assert result is True
        assert acm.can_access("agent_002", "msg_001", Permission.READ) is False

    def test_share_with_agent(self, acm):
        """Test sharing resource with another agent."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        result = acm.share_with_agent(
            resource_id="msg_001",
            target_agent_id="agent_002",
            sharing_agent_id="agent_001",
        )

        assert result is True
        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True

    def test_share_with_group(self, acm):
        """Test sharing resource with a group."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456", groups=["partners"])

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        result = acm.share_with_group(
            resource_id="msg_001",
            group_name="partners",
            sharing_agent_id="agent_001",
        )

        assert result is True
        assert acm.can_access("agent_002", "msg_001", Permission.READ) is True

    def test_update_visibility(self, acm):
        """Test updating resource visibility."""
        acm.register_agent("agent_001", "Agent 1", "org_123")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        result = acm.update_visibility(
            resource_id="msg_001",
            visibility=KnowledgeVisibility.PUBLIC,
            requesting_agent_id="agent_001",
        )

        assert result is True
        assert acm.get_policy("msg_001").visibility == KnowledgeVisibility.PUBLIC

    def test_update_visibility_denied(self, acm):
        """Test visibility update denied for non-owner."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_456")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        result = acm.update_visibility(
            resource_id="msg_001",
            visibility=KnowledgeVisibility.PUBLIC,
            requesting_agent_id="agent_002",
        )

        assert result is False

    def test_get_accessible_resources(self, acm):
        """Test getting all accessible resources for an agent."""
        acm.register_agent("agent_001", "Agent 1", "org_123")
        acm.register_agent("agent_002", "Agent 2", "org_123", groups=["team-a"])

        # Create multiple resources
        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.SHARED,
            sharing_groups=["team-a"],
        )

        acm.create_policy(
            resource_id="msg_002",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
            visibility=KnowledgeVisibility.PRIVATE,
        )

        accessible = acm.get_accessible_resources("agent_002", resource_type="message")

        assert "msg_001" in accessible
        assert "msg_002" not in accessible

    def test_inactive_agent_denied(self, acm):
        """Test inactive agent is denied access."""
        agent, _ = acm.register_agent("agent_001", "Agent 1", "org_123")

        acm.create_policy(
            resource_id="msg_001",
            resource_type="message",
            owner_id="agent_001",
            owner_org="org_123",
        )

        # Deactivate agent
        agent.is_active = False

        assert acm.can_access("agent_001", "msg_001", Permission.READ) is False


class TestAccessAuditLog:
    """Test AccessAuditLog dataclass."""

    def test_audit_log_creation(self):
        """Test creating an audit log entry."""
        log = AccessAuditLog(
            log_id="log_001",
            timestamp=datetime.now(timezone.utc),
            agent_id="agent_001",
            resource_id="msg_001",
            resource_type="message",
            action="read",
            success=True,
        )

        assert log.log_id == "log_001"
        assert log.success is True
        assert log.denial_reason is None

    def test_audit_log_with_denial(self):
        """Test audit log with denial reason."""
        log = AccessAuditLog(
            log_id="log_002",
            timestamp=datetime.now(timezone.utc),
            agent_id="agent_001",
            resource_id="msg_001",
            resource_type="message",
            action="write",
            success=False,
            denial_reason="Permission denied",
        )

        assert log.success is False
        assert log.denial_reason == "Permission denied"

    def test_audit_log_to_dict(self):
        """Test audit log serialization."""
        log = AccessAuditLog(
            log_id="log_003",
            timestamp=datetime.now(timezone.utc),
            agent_id="agent_001",
            resource_id="msg_001",
            resource_type="message",
            action="read",
            success=True,
        )

        data = log.to_dict()

        assert data["log_id"] == "log_003"
        assert data["success"] is True
        assert "timestamp" in data
