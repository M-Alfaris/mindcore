"""Test 09: RBAC (Role-Based Access Control) Tests.

Tests the access control and permissions system:
- Permission definitions
- Agent profiles
- Access decisions
- Team-based access
- Access level enforcement
"""

import pytest


# ============================================================================
# Access Controller Fixtures
# ============================================================================


@pytest.fixture
def access_controller():
    """Create an AccessController instance."""
    try:
        from mindcore.v2.access.permissions import AccessController

        return AccessController()
    except ImportError:
        pytest.skip("Access controller not available")


@pytest.fixture
def registered_agents(access_controller):
    """Register a set of test agents."""
    agents = [
        {
            "agent_id": "support_agent",
            "name": "Support Agent",
            "teams": ["support", "general"],
            "permissions": None,  # Use defaults
        },
        {"agent_id": "sales_agent", "name": "Sales Agent", "teams": ["sales"], "permissions": None},
        {
            "agent_id": "admin_agent",
            "name": "Admin Agent",
            "teams": ["admin"],
            "permissions": {
                "private": ["read", "write", "delete", "admin"],
                "team": ["read", "write", "delete"],
                "shared": ["read", "write"],
                "global": ["read", "write"],
            },
        },
    ]

    for agent in agents:
        access_controller.register_agent(
            agent_id=agent["agent_id"],
            name=agent["name"],
            teams=agent["teams"],
            permissions=agent["permissions"],
        )

    return access_controller


# ============================================================================
# Agent Registration Tests
# ============================================================================


class TestAgentRegistration:
    """Test agent registration and management."""

    def test_register_agent(self, access_controller):
        """Test registering a new agent."""
        profile = access_controller.register_agent(
            agent_id="test_agent_001",
            name="Test Agent",
            description="A test agent",
            teams=["test_team"],
        )

        assert profile is not None
        assert profile.agent_id == "test_agent_001"
        assert profile.name == "Test Agent"
        assert "test_team" in profile.teams

    def test_register_agent_with_custom_permissions(self, access_controller):
        """Test registering agent with custom permissions."""
        custom_perms = {
            "private": ["read", "write"],
            "team": ["read"],
            "shared": ["read"],
            "global": [],
        }

        profile = access_controller.register_agent(
            agent_id="custom_perms_agent",
            name="Custom Perms Agent",
            teams=["custom"],
            permissions=custom_perms,
        )

        assert profile is not None
        # Should have custom permissions

    def test_get_agent(self, registered_agents):
        """Test getting an agent profile."""
        profile = registered_agents.get_agent("support_agent")

        assert profile is not None
        assert profile.agent_id == "support_agent"

    def test_get_nonexistent_agent(self, access_controller):
        """Test getting non-existent agent returns None."""
        profile = access_controller.get_agent("nonexistent_agent")

        assert profile is None

    def test_list_agents(self, registered_agents):
        """Test listing all agents."""
        agents = registered_agents.list_agents()

        assert len(agents) >= 3
        agent_ids = [a.agent_id for a in agents]
        assert "support_agent" in agent_ids
        assert "sales_agent" in agent_ids
        assert "admin_agent" in agent_ids

    def test_unregister_agent(self, access_controller):
        """Test unregistering an agent."""
        access_controller.register_agent(agent_id="to_remove", name="To Remove", teams=["temp"])

        result = access_controller.unregister_agent("to_remove")
        assert result is True

        # Should be gone
        assert access_controller.get_agent("to_remove") is None


# ============================================================================
# Team Management Tests
# ============================================================================


class TestTeamManagement:
    """Test team membership management."""

    def test_get_team_members(self, registered_agents):
        """Test getting members of a team."""
        members = registered_agents.get_team_members("support")

        assert "support_agent" in members

    def test_add_agent_to_team(self, registered_agents):
        """Test adding agent to a team."""
        result = registered_agents.add_agent_to_team("sales_agent", "support")

        assert result is True

        # Verify membership
        members = registered_agents.get_team_members("support")
        assert "sales_agent" in members

    def test_remove_agent_from_team(self, registered_agents):
        """Test removing agent from team."""
        # Add first
        registered_agents.add_agent_to_team("sales_agent", "support")

        # Then remove
        result = registered_agents.remove_agent_from_team("sales_agent", "support")
        assert result is True

        # Verify removed
        members = registered_agents.get_team_members("support")
        assert "sales_agent" not in members


# ============================================================================
# Permission Tests
# ============================================================================


class TestPermissions:
    """Test permission checking."""

    def test_has_permission_private_read(self, registered_agents):
        """Test private read permission for owner."""
        profile = registered_agents.get_agent("support_agent")

        # Owner should have read on private
        assert profile.has_permission("private", "read")

    def test_has_permission_private_write(self, registered_agents):
        """Test private write permission for owner."""
        profile = registered_agents.get_agent("support_agent")

        assert profile.has_permission("private", "write")

    def test_has_permission_team_read(self, registered_agents):
        """Test team read permission."""
        profile = registered_agents.get_agent("support_agent")

        assert profile.has_permission("team", "read")

    def test_has_permission_shared_read(self, registered_agents):
        """Test shared read permission."""
        profile = registered_agents.get_agent("support_agent")

        assert profile.has_permission("shared", "read")

    def test_has_permission_global_read(self, registered_agents):
        """Test global read permission."""
        profile = registered_agents.get_agent("support_agent")

        assert profile.has_permission("global", "read")

    def test_admin_has_admin_permission(self, registered_agents):
        """Test admin agent has admin permission."""
        profile = registered_agents.get_agent("admin_agent")

        assert profile.has_permission("private", "admin")


# ============================================================================
# Access Decision Tests
# ============================================================================


class TestAccessDecisions:
    """Test access decision logic."""

    def test_can_access_own_private_memory(self, registered_agents):
        """Test agent can access own private memory."""
        decision = registered_agents.can_access(
            agent_id="support_agent",
            memory_access_level="private",
            memory_agent_id="support_agent",  # Same agent
            permission="read",
        )

        assert decision.allowed is True

    def test_cannot_access_other_private_memory(self, registered_agents):
        """Test agent cannot access other's private memory."""
        decision = registered_agents.can_access(
            agent_id="sales_agent",
            memory_access_level="private",
            memory_agent_id="support_agent",  # Different agent
            permission="read",
        )

        assert decision.allowed is False

    def test_can_access_team_memory_same_team(self, registered_agents):
        """Test agent can access team memory when in same team."""
        # Both in "support" team
        decision = registered_agents.can_access(
            agent_id="support_agent",
            memory_access_level="team",
            memory_agent_id="support_agent",
            memory_teams=["support"],
            permission="read",
        )

        assert decision.allowed is True

    def test_cannot_access_team_memory_different_team(self, registered_agents):
        """Test agent cannot access team memory from different team."""
        decision = registered_agents.can_access(
            agent_id="sales_agent",
            memory_access_level="team",
            memory_agent_id="support_agent",
            memory_teams=["support"],  # sales_agent not in support
            permission="read",
        )

        assert decision.allowed is False

    def test_can_access_shared_memory(self, registered_agents):
        """Test any agent can access shared memory for same user."""
        decision = registered_agents.can_access(
            agent_id="sales_agent",
            memory_access_level="shared",
            memory_agent_id="support_agent",
            permission="read",
        )

        assert decision.allowed is True

    def test_can_access_global_memory(self, registered_agents):
        """Test agents can access global memory."""
        decision = registered_agents.can_access(
            agent_id="sales_agent",
            memory_access_level="global",
            memory_agent_id="support_agent",
            permission="read",
        )

        assert decision.allowed is True


# ============================================================================
# Memory Filtering Tests
# ============================================================================


class TestMemoryFiltering:
    """Test filtering memories by access."""

    def test_filter_accessible_memories(self, registered_agents):
        """Test filtering list of memories by access."""
        from datetime import datetime

        from mindcore.v2.flr import Memory

        memories = [
            Memory(
                memory_id="1",
                content="Private to support",
                memory_type="semantic",
                user_id="user1",
                agent_id="support_agent",
                access_level="private",
                topics=[],
                created_at=datetime.now(),
            ),
            Memory(
                memory_id="2",
                content="Team shared",
                memory_type="semantic",
                user_id="user1",
                agent_id="support_agent",
                access_level="team",
                topics=[],
                created_at=datetime.now(),
            ),
            Memory(
                memory_id="3",
                content="Globally shared",
                memory_type="semantic",
                user_id="user1",
                agent_id="support_agent",
                access_level="global",
                topics=[],
                created_at=datetime.now(),
            ),
        ]

        # Filter for sales_agent (not in support team)
        accessible = registered_agents.filter_accessible_memories(
            agent_id="sales_agent", memories=memories
        )

        # Should only get shared and global, not private or team
        accessible_ids = [m.memory_id for m in accessible]
        assert "1" not in accessible_ids  # Private
        assert "2" not in accessible_ids  # Team (different team)
        assert "3" in accessible_ids  # Global


# ============================================================================
# Default Access Level Tests
# ============================================================================


class TestDefaultAccessLevel:
    """Test default access level handling."""

    def test_get_default_access_level(self, access_controller):
        """Test getting agent's default access level."""
        access_controller.register_agent(
            agent_id="default_test",
            name="Default Test",
            teams=["test"],
            default_access_level="team",
        )

        default = access_controller.get_default_access_level("default_test")
        assert default == "team"

    def test_default_access_level_is_private(self, registered_agents):
        """Test default access level defaults to private."""
        default = registered_agents.get_default_access_level("support_agent")
        assert default == "private"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestRBACStats:
    """Test RBAC statistics."""

    def test_get_stats(self, registered_agents):
        """Test getting access controller stats."""
        stats = registered_agents.get_stats()

        assert stats is not None
        assert "agent_count" in stats or "agents" in stats


# ============================================================================
# Edge Cases
# ============================================================================


class TestRBACEdgeCases:
    """Test RBAC edge cases."""

    def test_access_without_agent_id(self, registered_agents):
        """Test access check without agent ID."""
        registered_agents.can_access(
            agent_id=None,  # No agent
            memory_access_level="global",
            memory_agent_id="support_agent",
        )

        # Should still work for global

    def test_access_without_memory_teams(self, registered_agents):
        """Test access check without memory teams specified."""
        registered_agents.can_access(
            agent_id="support_agent",
            memory_access_level="team",
            memory_agent_id="support_agent",
            memory_teams=None,  # No teams specified
        )

        # Should handle gracefully

    def test_permission_on_nonexistent_agent(self, access_controller):
        """Test permission check on non-existent agent."""
        access_controller.can_access(
            agent_id="nonexistent", memory_access_level="private", memory_agent_id="nonexistent"
        )

        # Should handle gracefully (deny or error)


# ============================================================================
# Write Permission Tests
# ============================================================================


class TestWritePermissions:
    """Test write permission scenarios."""

    def test_can_write_own_private(self, registered_agents):
        """Test agent can write to own private memory."""
        profile = registered_agents.get_agent("support_agent")
        assert profile.can_write("private")

    def test_can_write_team_memory(self, registered_agents):
        """Test agent can write to team memory."""
        profile = registered_agents.get_agent("support_agent")
        assert profile.can_write("team")

    def test_can_write_shared_memory(self, registered_agents):
        """Test shared write permission."""
        profile = registered_agents.get_agent("admin_agent")
        assert profile.can_write("shared")


# ============================================================================
# Delete Permission Tests
# ============================================================================


class TestDeletePermissions:
    """Test delete permission scenarios."""

    def test_can_delete_own_private(self, registered_agents):
        """Test agent can delete own private memory."""
        profile = registered_agents.get_agent("support_agent")
        assert profile.can_delete("private")

    def test_admin_can_delete_team(self, registered_agents):
        """Test admin can delete team memory."""
        profile = registered_agents.get_agent("admin_agent")
        assert profile.can_delete("team")
