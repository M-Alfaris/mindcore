"""Comprehensive tests for access control module."""

import pytest
from datetime import datetime, timezone

from mindcore.v2.access import (
    AccessController,
    AccessDecision,
    AgentProfile,
    Permission,
)
from mindcore.v2.flr import Memory


class TestAgentProfile:
    """Test AgentProfile dataclass."""

    def test_create_basic_profile(self):
        """Test creating a basic agent profile."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
        )

        assert profile.agent_id == "agent_123"
        assert profile.name == "Test Agent"
        assert profile.is_active is True
        assert profile.teams == []

    def test_create_profile_with_teams(self):
        """Test creating profile with team memberships."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
            teams=["support", "sales"],
        )

        assert "support" in profile.teams
        assert "sales" in profile.teams

    def test_default_permissions(self):
        """Test that default permissions are set."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
        )

        # Should have default permissions
        assert Permission.READ in profile.permissions.get("private", [])
        assert Permission.WRITE in profile.permissions.get("private", [])
        assert Permission.DELETE in profile.permissions.get("private", [])
        assert Permission.READ in profile.permissions.get("global", [])

    def test_has_permission(self):
        """Test permission checking."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
        )

        assert profile.has_permission("private", Permission.READ) is True
        assert profile.has_permission("private", Permission.WRITE) is True
        assert profile.has_permission("global", Permission.READ) is True
        assert profile.has_permission("global", Permission.WRITE) is False

    def test_admin_bypasses_permission_check(self):
        """Test that admin permission grants all access."""
        profile = AgentProfile(
            agent_id="admin_agent",
            name="Admin Agent",
            permissions={"private": [Permission.ADMIN]},
        )

        # Admin should have all permissions
        assert profile.has_permission("private", Permission.READ) is True
        assert profile.has_permission("private", Permission.WRITE) is True
        assert profile.has_permission("private", Permission.DELETE) is True

    def test_can_read_write_delete(self):
        """Test convenience permission methods."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
        )

        assert profile.can_read("private") is True
        assert profile.can_write("private") is True
        assert profile.can_delete("private") is True
        assert profile.can_write("global") is False

    def test_to_dict(self):
        """Test converting profile to dictionary."""
        profile = AgentProfile(
            agent_id="agent_123",
            name="Test Agent",
            description="A test agent",
            teams=["team1"],
        )

        result = profile.to_dict()

        assert isinstance(result, dict)
        assert result["agent_id"] == "agent_123"
        assert result["name"] == "Test Agent"
        assert "teams" in result
        assert "permissions" in result


class TestAccessDecision:
    """Test AccessDecision dataclass."""

    def test_allowed_decision(self):
        """Test creating an allowed decision."""
        decision = AccessDecision(
            allowed=True,
            reason="Access granted",
            agent_id="agent_123",
        )

        assert decision.allowed is True
        assert decision.reason == "Access granted"

    def test_denied_decision(self):
        """Test creating a denied decision."""
        decision = AccessDecision(
            allowed=False,
            reason="No permission",
            agent_id="agent_123",
            memory_id="mem_456",
            access_level="private",
            permission=Permission.WRITE,
        )

        assert decision.allowed is False
        assert decision.memory_id == "mem_456"


class TestAccessController:
    """Test AccessController class."""

    @pytest.fixture
    def controller(self):
        """Create a fresh access controller."""
        return AccessController()

    def test_register_agent(self, controller):
        """Test registering an agent."""
        profile = controller.register_agent(
            agent_id="agent_123",
            name="Test Agent",
            description="A test agent",
        )

        assert profile.agent_id == "agent_123"
        assert profile.name == "Test Agent"

    def test_register_duplicate_agent(self, controller):
        """Test that duplicate registration raises error."""
        controller.register_agent("agent_123", "Agent 1")

        with pytest.raises(ValueError):
            controller.register_agent("agent_123", "Agent 1 Duplicate")

    def test_register_agent_with_teams(self, controller):
        """Test registering agent with teams."""
        profile = controller.register_agent(
            agent_id="agent_123",
            name="Test Agent",
            teams=["support", "sales"],
        )

        assert "support" in profile.teams
        assert "sales" in profile.teams

        # Check team mappings
        assert "agent_123" in controller.get_team_members("support")
        assert "agent_123" in controller.get_team_members("sales")

    def test_unregister_agent(self, controller):
        """Test unregistering an agent."""
        controller.register_agent("agent_123", "Test Agent", teams=["team1"])

        result = controller.unregister_agent("agent_123")

        assert result is True
        assert controller.get_agent("agent_123") is None
        assert "agent_123" not in controller.get_team_members("team1")

    def test_unregister_nonexistent_agent(self, controller):
        """Test unregistering non-existent agent."""
        result = controller.unregister_agent("nonexistent")
        assert result is False

    def test_get_agent(self, controller):
        """Test getting agent profile."""
        controller.register_agent("agent_123", "Test Agent")

        profile = controller.get_agent("agent_123")

        assert profile is not None
        assert profile.agent_id == "agent_123"

    def test_get_nonexistent_agent(self, controller):
        """Test getting non-existent agent."""
        profile = controller.get_agent("nonexistent")
        assert profile is None

    def test_list_agents(self, controller):
        """Test listing all agents."""
        controller.register_agent("agent_1", "Agent 1")
        controller.register_agent("agent_2", "Agent 2")

        agents = controller.list_agents()

        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "agent_1" in agent_ids
        assert "agent_2" in agent_ids

    def test_get_team_members(self, controller):
        """Test getting team members."""
        controller.register_agent("agent_1", "Agent 1", teams=["support"])
        controller.register_agent("agent_2", "Agent 2", teams=["support"])
        controller.register_agent("agent_3", "Agent 3", teams=["sales"])

        support_members = controller.get_team_members("support")

        assert len(support_members) == 2
        assert "agent_1" in support_members
        assert "agent_2" in support_members
        assert "agent_3" not in support_members

    def test_add_agent_to_team(self, controller):
        """Test adding agent to team."""
        controller.register_agent("agent_1", "Agent 1")

        result = controller.add_agent_to_team("agent_1", "new_team")

        assert result is True
        assert "new_team" in controller.get_agent("agent_1").teams
        assert "agent_1" in controller.get_team_members("new_team")

    def test_add_nonexistent_agent_to_team(self, controller):
        """Test adding non-existent agent to team."""
        result = controller.add_agent_to_team("nonexistent", "team1")
        assert result is False

    def test_remove_agent_from_team(self, controller):
        """Test removing agent from team."""
        controller.register_agent("agent_1", "Agent 1", teams=["team1"])

        result = controller.remove_agent_from_team("agent_1", "team1")

        assert result is True
        assert "team1" not in controller.get_agent("agent_1").teams
        assert "agent_1" not in controller.get_team_members("team1")


class TestAccessControlDecisions:
    """Test access control decision making."""

    @pytest.fixture
    def controller(self):
        """Create controller with registered agents."""
        ac = AccessController()
        ac.register_agent("agent_1", "Agent 1", teams=["support"])
        ac.register_agent("agent_2", "Agent 2", teams=["support"])
        ac.register_agent("agent_3", "Agent 3", teams=["sales"])
        return ac

    def test_access_own_private_memory(self, controller):
        """Test accessing own private memory."""
        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="private",
            memory_agent_id="agent_1",
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_other_private_memory(self, controller):
        """Test accessing another agent's private memory."""
        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="private",
            memory_agent_id="agent_2",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "private" in decision.reason.lower()

    def test_access_team_memory_same_team(self, controller):
        """Test accessing team memory from same team."""
        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="team",
            memory_agent_id="agent_2",  # Same team
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_team_memory_different_team(self, controller):
        """Test accessing team memory from different team."""
        decision = controller.can_access(
            agent_id="agent_1",  # support team
            memory_access_level="team",
            memory_agent_id="agent_3",  # sales team
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "team" in decision.reason.lower()

    def test_access_shared_memory(self, controller):
        """Test accessing shared memory."""
        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="shared",
            memory_agent_id="agent_3",
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_global_memory(self, controller):
        """Test accessing global memory."""
        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="global",
            memory_agent_id="agent_3",
            permission=Permission.READ,
        )

        assert decision.allowed is True

    def test_access_unregistered_agent(self, controller):
        """Test access check for unregistered agent."""
        decision = controller.can_access(
            agent_id="unknown_agent",
            memory_access_level="shared",
            memory_agent_id="agent_1",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "not registered" in decision.reason.lower()

    def test_inactive_agent_denied(self, controller):
        """Test that inactive agents are denied."""
        profile = controller.get_agent("agent_1")
        profile.is_active = False

        decision = controller.can_access(
            agent_id="agent_1",
            memory_access_level="shared",
            memory_agent_id="agent_2",
            permission=Permission.READ,
        )

        assert decision.allowed is False
        assert "inactive" in decision.reason.lower()

    def test_access_updates_last_active(self, controller):
        """Test that access check updates last_active timestamp."""
        before = datetime.now(timezone.utc)

        controller.can_access(
            agent_id="agent_1",
            memory_access_level="shared",
            memory_agent_id="agent_2",
            permission=Permission.READ,
        )

        profile = controller.get_agent("agent_1")
        assert profile.last_active is not None
        assert profile.last_active >= before


class TestFilterAccessibleMemories:
    """Test filtering memories by access."""

    @pytest.fixture
    def controller(self):
        """Create controller with agents."""
        ac = AccessController()
        ac.register_agent("agent_1", "Agent 1", teams=["support"])
        ac.register_agent("agent_2", "Agent 2", teams=["support"])
        ac.register_agent("agent_3", "Agent 3", teams=["sales"])
        return ac

    def test_filter_memories(self, controller):
        """Test filtering list of memories."""
        memories = [
            Memory(
                memory_id="mem_1",
                content="Private memory",
                memory_type="episodic",
                user_id="user_1",
                agent_id="agent_1",
                access_level="private",
            ),
            Memory(
                memory_id="mem_2",
                content="Team memory",
                memory_type="episodic",
                user_id="user_1",
                agent_id="agent_2",
                access_level="team",
            ),
            Memory(
                memory_id="mem_3",
                content="Other team memory",
                memory_type="episodic",
                user_id="user_1",
                agent_id="agent_3",
                access_level="team",
            ),
            Memory(
                memory_id="mem_4",
                content="Shared memory",
                memory_type="episodic",
                user_id="user_1",
                agent_id="agent_3",
                access_level="shared",
            ),
        ]

        accessible = controller.filter_accessible_memories(
            agent_id="agent_1",
            memories=memories,
            permission=Permission.READ,
        )

        # Should have: own private, team memory from same team, shared
        assert len(accessible) == 3
        memory_ids = [m.memory_id for m in accessible]
        assert "mem_1" in memory_ids  # Own private
        assert "mem_2" in memory_ids  # Team from same team
        assert "mem_3" not in memory_ids  # Team from different team
        assert "mem_4" in memory_ids  # Shared


class TestAccessControllerStats:
    """Test access controller statistics."""

    def test_get_stats(self):
        """Test getting access control stats."""
        controller = AccessController()
        controller.register_agent("agent_1", "Agent 1", teams=["support"])
        controller.register_agent("agent_2", "Agent 2", teams=["support", "sales"])
        controller.register_agent("agent_3", "Agent 3", teams=["sales"])

        # Deactivate one
        controller.get_agent("agent_3").is_active = False

        stats = controller.get_stats()

        assert stats["total_agents"] == 3
        assert stats["active_agents"] == 2
        assert stats["total_teams"] == 2
        assert stats["agents_by_team"]["support"] == 2
        assert stats["agents_by_team"]["sales"] == 2

    def test_get_default_access_level(self):
        """Test getting default access level."""
        controller = AccessController()
        controller.register_agent(
            "agent_1",
            "Agent 1",
            default_access_level="team",
        )

        assert controller.get_default_access_level("agent_1") == "team"
        assert controller.get_default_access_level("nonexistent") == "private"
