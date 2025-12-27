"""Tests for Access Permissions - Multi-agent access control.

Tests cover:
- AgentProfile: permissions, team membership
- AccessController: registration, access decisions
- Team-based access control
- Permission filtering
"""

import pytest

from mindcore.access.permissions import (
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
)
from mindcore.flr import Memory


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def controller():
    """Create AccessController instance."""
    return AccessController()


@pytest.fixture
def support_agent(controller):
    """Create support agent profile."""
    return controller.register_agent(
        agent_id="support_bot",
        name="Support Agent",
        description="Handles customer support",
        teams=["support", "customer_service"],
    )


@pytest.fixture
def sales_agent(controller):
    """Create sales agent profile."""
    return controller.register_agent(
        agent_id="sales_bot",
        name="Sales Agent",
        description="Handles sales inquiries",
        teams=["sales"],
    )


# =============================================================================
# Permission Enum Tests
# =============================================================================


class TestPermissionEnum:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test Permission enum values."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.SHARE.value == "share"
        assert Permission.ADMIN.value == "admin"


# =============================================================================
# AgentProfile Tests
# =============================================================================


class TestAgentProfile:
    """Tests for AgentProfile dataclass."""

    def test_create_profile(self):
        """Test creating agent profile."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test Agent",
            description="Test description",
        )

        assert profile.agent_id == "agent_1"
        assert profile.name == "Test Agent"
        assert profile.is_active is True

    def test_default_permissions(self):
        """Test default permissions are set."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test Agent",
        )

        # Should have default permissions
        assert Permission.READ in profile.permissions.get("private", [])
        assert Permission.WRITE in profile.permissions.get("private", [])
        assert Permission.READ in profile.permissions.get("shared", [])

    def test_custom_permissions(self):
        """Test custom permissions override defaults."""
        custom_perms = {
            "private": [Permission.READ],
            "shared": [Permission.READ, Permission.WRITE],
        }
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test Agent",
            permissions=custom_perms,
        )

        assert profile.permissions == custom_perms

    def test_has_permission(self):
        """Test has_permission method."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test Agent",
        )

        assert profile.has_permission("private", Permission.READ) is True
        assert profile.has_permission("private", Permission.WRITE) is True
        assert profile.has_permission("shared", Permission.READ) is True

    def test_has_permission_with_admin(self):
        """Test admin permission grants all access."""
        profile = AgentProfile(
            agent_id="admin_agent",
            name="Admin",
            permissions={"private": [Permission.ADMIN]},
        )

        # Admin should have all permissions
        assert profile.has_permission("private", Permission.READ) is True
        assert profile.has_permission("private", Permission.WRITE) is True
        assert profile.has_permission("private", Permission.DELETE) is True

    def test_can_read(self):
        """Test can_read shortcut."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test",
        )

        assert profile.can_read("private") is True
        assert profile.can_read("shared") is True

    def test_can_write(self):
        """Test can_write shortcut."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test",
        )

        assert profile.can_write("private") is True
        assert profile.can_write("team") is True

    def test_can_delete(self):
        """Test can_delete shortcut."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test",
        )

        assert profile.can_delete("private") is True
        # Default: no delete on shared
        assert profile.can_delete("shared") is False

    def test_shares_team_with(self):
        """Test team membership checking."""
        agent1 = AgentProfile(
            agent_id="agent_1",
            name="Agent 1",
            teams=["support", "sales"],
        )
        agent2 = AgentProfile(
            agent_id="agent_2",
            name="Agent 2",
            teams=["support", "billing"],
        )
        agent3 = AgentProfile(
            agent_id="agent_3",
            name="Agent 3",
            teams=["engineering"],
        )

        assert agent1.shares_team_with(agent2) is True  # Both in "support"
        assert agent1.shares_team_with(agent3) is False  # No common team
        assert agent2.shares_team_with(agent3) is False

    def test_shares_team_empty_teams(self):
        """Test shares_team_with with empty teams."""
        agent1 = AgentProfile(agent_id="a1", name="A1", teams=[])
        agent2 = AgentProfile(agent_id="a2", name="A2", teams=["support"])

        assert agent1.shares_team_with(agent2) is False

    def test_to_dict(self):
        """Test serialization to dict."""
        profile = AgentProfile(
            agent_id="agent_1",
            name="Test Agent",
            teams=["support"],
        )

        data = profile.to_dict()

        assert data["agent_id"] == "agent_1"
        assert data["name"] == "Test Agent"
        assert data["teams"] == ["support"]
        assert "permissions" in data


# =============================================================================
# AccessController Registration Tests
# =============================================================================


class TestAccessControllerRegistration:
    """Tests for AccessController agent registration."""

    def test_register_agent(self, controller):
        """Test registering an agent."""
        profile = controller.register_agent(
            agent_id="new_agent",
            name="New Agent",
            description="A new agent",
        )

        assert profile.agent_id == "new_agent"
        assert controller.get_agent("new_agent") is not None

    def test_register_duplicate_raises(self, controller, support_agent):
        """Test registering duplicate agent raises error."""
        with pytest.raises(ValueError, match="already registered"):
            controller.register_agent(
                agent_id="support_bot",
                name="Duplicate",
            )

    def test_register_with_teams(self, controller):
        """Test registering agent with teams."""
        profile = controller.register_agent(
            agent_id="team_agent",
            name="Team Agent",
            teams=["team_a", "team_b"],
        )

        assert profile.teams == ["team_a", "team_b"]
        assert "team_agent" in controller.get_team_members("team_a")
        assert "team_agent" in controller.get_team_members("team_b")

    def test_unregister_agent(self, controller, support_agent):
        """Test unregistering an agent."""
        result = controller.unregister_agent("support_bot")

        assert result is True
        assert controller.get_agent("support_bot") is None

    def test_unregister_nonexistent(self, controller):
        """Test unregistering nonexistent agent."""
        result = controller.unregister_agent("nonexistent")
        assert result is False

    def test_unregister_removes_from_teams(self, controller):
        """Test unregistering removes agent from teams."""
        controller.register_agent(
            agent_id="team_agent",
            name="Team Agent",
            teams=["team_a"],
        )

        assert "team_agent" in controller.get_team_members("team_a")

        controller.unregister_agent("team_agent")

        assert "team_agent" not in controller.get_team_members("team_a")

    def test_get_agent(self, controller, support_agent):
        """Test getting agent profile."""
        profile = controller.get_agent("support_bot")

        assert profile is not None
        assert profile.agent_id == "support_bot"

    def test_get_agent_nonexistent(self, controller):
        """Test getting nonexistent agent."""
        profile = controller.get_agent("nonexistent")
        assert profile is None

    def test_list_agents(self, controller, support_agent, sales_agent):
        """Test listing all agents."""
        agents = controller.list_agents()

        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "support_bot" in agent_ids
        assert "sales_bot" in agent_ids


# =============================================================================
# Team Management Tests
# =============================================================================


class TestTeamManagement:
    """Tests for team management."""

    def test_get_team_members(self, controller, support_agent):
        """Test getting team members."""
        members = controller.get_team_members("support")

        assert "support_bot" in members

    def test_get_team_members_empty(self, controller):
        """Test getting members of nonexistent team."""
        members = controller.get_team_members("nonexistent_team")
        assert members == []

    def test_add_agent_to_team(self, controller, support_agent):
        """Test adding agent to team."""
        result = controller.add_agent_to_team("support_bot", "new_team")

        assert result is True
        assert "new_team" in support_agent.teams
        assert "support_bot" in controller.get_team_members("new_team")

    def test_add_nonexistent_agent_to_team(self, controller):
        """Test adding nonexistent agent to team."""
        result = controller.add_agent_to_team("nonexistent", "team")
        assert result is False

    def test_remove_agent_from_team(self, controller, support_agent):
        """Test removing agent from team."""
        result = controller.remove_agent_from_team("support_bot", "support")

        assert result is True
        assert "support" not in support_agent.teams
        assert "support_bot" not in controller.get_team_members("support")

    def test_remove_nonexistent_agent_from_team(self, controller):
        """Test removing nonexistent agent from team."""
        result = controller.remove_agent_from_team("nonexistent", "team")
        assert result is False


# =============================================================================
# Access Decision Tests
# =============================================================================


class TestAccessDecisions:
    """Tests for access control decisions."""

    def test_access_own_private_memory(self, controller, support_agent):
        """Test agent can access own private memory."""
        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="private",
            memory_agent_id="support_bot",
            permission=Permission.READ,
        )

        assert decision.allowed is True
        assert decision.reason == "Access granted"

    def test_cannot_access_other_private(self, controller, support_agent, sales_agent):
        """Test agent cannot access other's private memory."""
        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="private",
            memory_agent_id="sales_bot",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "private memory of another agent" in decision.reason

    def test_access_team_memory_same_team(self, controller):
        """Test team access with common team."""
        controller.register_agent("agent_a", "Agent A", teams=["team_x"])
        controller.register_agent("agent_b", "Agent B", teams=["team_x"])

        decision = controller.can_access(
            agent_id="agent_a",
            memory_access_level="team",
            memory_agent_id="agent_b",
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_team_memory_different_team(self, controller, support_agent, sales_agent):
        """Test team access denied without common team."""
        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="team",
            memory_agent_id="sales_bot",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "No common team" in decision.reason

    def test_access_shared_memory(self, controller, support_agent, sales_agent):
        """Test any agent can access shared memory."""
        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="shared",
            memory_agent_id="sales_bot",
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_global_memory(self, controller, support_agent):
        """Test agent can access global memory."""
        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="global",
            memory_agent_id=None,
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_unregistered_agent_denied(self, controller):
        """Test unregistered agent is denied."""
        decision = controller.can_access(
            agent_id="unregistered",
            memory_access_level="shared",
            memory_agent_id="someone",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "not registered" in decision.reason

    def test_inactive_agent_denied(self, controller, support_agent):
        """Test inactive agent is denied."""
        support_agent.is_active = False

        decision = controller.can_access(
            agent_id="support_bot",
            memory_access_level="shared",
            memory_agent_id="someone",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "inactive" in decision.reason

    def test_lacking_permission_denied(self, controller):
        """Test agent lacking permission is denied."""
        controller.register_agent(
            agent_id="limited",
            name="Limited Agent",
            permissions={"private": [Permission.READ]},  # No write
        )

        decision = controller.can_access(
            agent_id="limited",
            memory_access_level="private",
            memory_agent_id="limited",
            permission=Permission.WRITE,
        )

        assert decision.allowed is False
        assert "lacks" in decision.reason

    def test_access_with_memory_teams(self, controller, support_agent, sales_agent):
        """Test team access with memory_teams parameter."""
        # Sales agent has team "sales"
        # Memory is shared with "sales" team
        decision = controller.can_access(
            agent_id="sales_bot",
            memory_access_level="team",
            memory_agent_id="support_bot",  # Different agent
            memory_teams=["sales"],  # But shared with sales team
            permission=Permission.READ,
        )

        assert decision.allowed is True


# =============================================================================
# AccessDecision Dataclass Tests
# =============================================================================


class TestAccessDecision:
    """Tests for AccessDecision dataclass."""

    def test_access_decision_fields(self):
        """Test AccessDecision fields."""
        decision = AccessDecision(
            allowed=True,
            reason="Access granted",
            agent_id="agent_1",
            memory_id="mem_1",
            access_level="shared",
            permission=Permission.READ,
        )

        assert decision.allowed is True
        assert decision.reason == "Access granted"
        assert decision.memory_id == "mem_1"


# =============================================================================
# Filter Accessible Memories Tests
# =============================================================================


class TestFilterAccessibleMemories:
    """Tests for filter_accessible_memories method."""

    def test_filter_own_memories(self, controller, support_agent):
        """Test filtering keeps own memories."""
        memories = [
            Memory(
                memory_id="m1",
                content="Own memory",
                memory_type="fact",
                user_id="user_1",
                agent_id="support_bot",
                access_level="private",
            ),
        ]

        filtered = controller.filter_accessible_memories(
            agent_id="support_bot",
            memories=memories,
        )

        assert len(filtered) == 1

    def test_filter_removes_inaccessible(self, controller, support_agent):
        """Test filtering removes inaccessible memories."""
        memories = [
            Memory(
                memory_id="m1",
                content="Own memory",
                memory_type="fact",
                user_id="user_1",
                agent_id="support_bot",
                access_level="private",
            ),
            Memory(
                memory_id="m2",
                content="Other's private",
                memory_type="fact",
                user_id="user_1",
                agent_id="other_agent",
                access_level="private",
            ),
        ]

        filtered = controller.filter_accessible_memories(
            agent_id="support_bot",
            memories=memories,
        )

        assert len(filtered) == 1
        assert filtered[0].memory_id == "m1"

    def test_filter_keeps_shared(self, controller, support_agent):
        """Test filtering keeps shared memories."""
        memories = [
            Memory(
                memory_id="m1",
                content="Shared memory",
                memory_type="fact",
                user_id="user_1",
                agent_id="other_agent",
                access_level="shared",
            ),
        ]

        filtered = controller.filter_accessible_memories(
            agent_id="support_bot",
            memories=memories,
        )

        assert len(filtered) == 1


# =============================================================================
# Utility Methods Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_default_access_level(self, controller, support_agent):
        """Test getting default access level."""
        level = controller.get_default_access_level("support_bot")
        assert level == "private"

    def test_get_default_access_level_custom(self, controller):
        """Test custom default access level."""
        controller.register_agent(
            agent_id="shared_agent",
            name="Shared Agent",
            default_access_level="shared",
        )

        level = controller.get_default_access_level("shared_agent")
        assert level == "shared"

    def test_get_default_access_level_unknown(self, controller):
        """Test default access level for unknown agent."""
        level = controller.get_default_access_level("unknown")
        assert level == "private"

    def test_get_stats(self, controller, support_agent, sales_agent):
        """Test getting statistics."""
        stats = controller.get_stats()

        assert stats["total_agents"] == 2
        assert stats["agent_count"] == 2  # Alias
        assert stats["active_agents"] == 2
        assert stats["total_teams"] >= 2
        assert "agents_by_team" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
