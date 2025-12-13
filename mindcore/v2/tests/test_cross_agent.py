"""Tests for cross-agent memory layer."""

import os
import tempfile

import pytest

from mindcore.v2 import (
    CrossAgentLayer,
    AgentStatus,
    RoutingStrategy,
    SQLiteStorage,
    Memory,
)
from mindcore.v2.cross_agent import SyncDirection


class TestAgentRegistry:
    """Test agent registration and management."""

    @pytest.fixture
    def layer(self):
        """Create a CrossAgentLayer with temp storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        cal = CrossAgentLayer(storage)
        yield cal
        storage.close()
        os.unlink(db_path)

    def test_register_agent(self, layer):
        """Test registering an agent."""
        agent = layer.register_agent(
            agent_id="support_bot",
            name="Support Agent",
            description="Handles customer support",
            capabilities=["customer_support", "billing"],
            teams=["customer_service"],
        )

        assert agent.agent_id == "support_bot"
        assert agent.name == "Support Agent"
        assert "customer_support" in agent.capabilities
        assert "customer_service" in agent.teams

    def test_get_agent(self, layer):
        """Test retrieving an agent."""
        layer.register_agent(
            agent_id="test_agent",
            name="Test Agent",
        )

        agent = layer.get_agent("test_agent")
        assert agent is not None
        assert agent.agent_id == "test_agent"

        # Non-existent agent
        assert layer.get_agent("nonexistent") is None

    def test_list_agents(self, layer):
        """Test listing agents with filters."""
        layer.register_agent(
            agent_id="agent1",
            name="Agent 1",
            capabilities=["support"],
            teams=["team_a"],
        )
        layer.register_agent(
            agent_id="agent2",
            name="Agent 2",
            capabilities=["sales"],
            teams=["team_a"],
        )
        layer.register_agent(
            agent_id="agent3",
            name="Agent 3",
            capabilities=["support"],
            teams=["team_b"],
        )

        # All agents
        all_agents = layer.list_agents()
        assert len(all_agents) == 3

        # Filter by team
        team_a = layer.list_agents(team="team_a")
        assert len(team_a) == 2

        # Filter by capability
        support = layer.list_agents(capability="support")
        assert len(support) == 2

    def test_unregister_agent(self, layer):
        """Test unregistering an agent."""
        layer.register_agent(agent_id="temp", name="Temp")

        assert layer.get_agent("temp") is not None
        assert layer.unregister_agent("temp") is True
        assert layer.get_agent("temp") is None

    def test_agent_status(self, layer):
        """Test agent status changes."""
        layer.register_agent(agent_id="agent", name="Agent")

        agent = layer.get_agent("agent")
        assert agent.status == AgentStatus.ACTIVE

        layer.set_agent_status("agent", AgentStatus.MAINTENANCE)
        agent = layer.get_agent("agent")
        assert agent.status == AgentStatus.MAINTENANCE


class TestTeamManagement:
    """Test team management."""

    @pytest.fixture
    def layer(self):
        """Create a CrossAgentLayer with temp storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        cal = CrossAgentLayer(storage)
        yield cal
        storage.close()
        os.unlink(db_path)

    def test_create_team(self, layer):
        """Test creating a team."""
        team = layer.create_team(
            team_id="support_team",
            name="Support Team",
            shared_topics=["billing", "orders"],
        )

        assert team.team_id == "support_team"
        assert "billing" in team.shared_topics

    def test_add_agent_to_team(self, layer):
        """Test adding agents to teams."""
        layer.create_team(team_id="team1", name="Team 1")
        layer.register_agent(agent_id="agent1", name="Agent 1")

        assert layer.add_agent_to_team("agent1", "team1") is True

        members = layer.get_team_members("team1")
        assert len(members) == 1
        assert members[0].agent_id == "agent1"

    def test_get_teammates(self, layer):
        """Test getting teammates."""
        layer.create_team(team_id="team1", name="Team 1")
        layer.register_agent(agent_id="agent1", name="Agent 1", teams=["team1"])
        layer.register_agent(agent_id="agent2", name="Agent 2", teams=["team1"])
        layer.register_agent(agent_id="agent3", name="Agent 3", teams=["other"])

        teammates = layer.get_teammates("agent1")
        assert len(teammates) == 1
        assert teammates[0].agent_id == "agent2"


class TestMemorySharing:
    """Test memory sharing between agents."""

    @pytest.fixture
    def layer(self):
        """Create a CrossAgentLayer with temp storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        cal = CrossAgentLayer(storage)

        # Set up agents
        cal.register_agent(
            agent_id="support",
            name="Support",
            teams=["customer_service"],
        )
        cal.register_agent(
            agent_id="sales",
            name="Sales",
            teams=["customer_service"],
        )
        cal.register_agent(
            agent_id="other",
            name="Other",
            teams=["other_team"],
        )
        cal.create_team(team_id="customer_service", name="Customer Service")

        yield cal
        storage.close()
        os.unlink(db_path)

    def test_store_memory_with_agent(self, layer):
        """Test storing memory with agent ownership."""
        memory = Memory(
            memory_id="",
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
        )

        memory_id = layer.store_memory(
            memory=memory,
            agent_id="support",
            access_level="team",
        )

        assert memory_id is not None

        # Verify ownership
        stored = layer.storage.get(memory_id)
        assert stored.agent_id == "support"
        assert stored.access_level == "team"

    def test_share_memory(self, layer):
        """Test sharing a memory."""
        memory = Memory(
            memory_id="",
            content="Shared info",
            memory_type="semantic",
            user_id="user123",
        )

        memory_id = layer.store_memory(
            memory=memory,
            agent_id="support",
            access_level="private",
        )

        # Share with team
        result = layer.share_memory(
            memory_id=memory_id,
            source_agent="support",
            access_level="team",
        )

        assert result.success is True
        assert result.access_level == "team"

    def test_access_control(self, layer):
        """Test memory access control."""
        # Support can access its own memories
        assert layer.can_access("support", "support", "private") is True

        # Sales can access team memories from support
        assert layer.can_access("sales", "support", "team") is True

        # Other cannot access team memories
        assert layer.can_access("other", "support", "team") is False

        # Everyone can access shared memories
        assert layer.can_access("other", "support", "shared") is True


class TestQueryRouting:
    """Test cross-agent query routing."""

    @pytest.fixture
    def layer(self):
        """Create a CrossAgentLayer with temp storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        cal = CrossAgentLayer(storage)

        # Set up specialized agents
        cal.register_agent(
            agent_id="billing_bot",
            name="Billing Bot",
            capabilities=["billing", "payments"],
            specializations=["invoices", "refunds"],
        )
        cal.register_agent(
            agent_id="product_bot",
            name="Product Bot",
            capabilities=["product_info", "catalog"],
            specializations=["features", "pricing"],
        )

        yield cal
        storage.close()
        os.unlink(db_path)

    def test_route_by_capability(self, layer):
        """Test routing based on capability matching."""
        result = layer.query(
            query="billing question",
            user_id="user123",
            strategy=RoutingStrategy.CAPABILITY_MATCH,
            attention_hints=["billing"],
        )

        assert "billing_bot" in result.selected_agents

    def test_rank_agents(self, layer):
        """Test agent ranking."""
        rankings = layer.rank_agents(
            query="payment issue",
            attention_hints=["billing", "payments"],
        )

        assert len(rankings) > 0
        # Billing bot should rank higher for billing queries
        billing_rank = next(
            (r for r in rankings if r["agent_id"] == "billing_bot"),
            None
        )
        product_rank = next(
            (r for r in rankings if r["agent_id"] == "product_bot"),
            None
        )

        assert billing_rank is not None
        assert billing_rank["score"] > product_rank["score"]

    def test_suggest_agents(self, layer):
        """Test agent suggestions."""
        suggestions = layer.suggest_agents(
            attention_hints=["billing"],
            limit=2,
        )

        assert len(suggestions) >= 1
        assert suggestions[0].agent_id == "billing_bot"


class TestCrossAgentStats:
    """Test cross-agent statistics."""

    @pytest.fixture
    def layer(self):
        """Create a CrossAgentLayer with temp storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage = SQLiteStorage(db_path)
        cal = CrossAgentLayer(storage)

        cal.register_agent(agent_id="agent1", name="Agent 1")
        cal.register_agent(agent_id="agent2", name="Agent 2")

        yield cal
        storage.close()
        os.unlink(db_path)

    def test_get_stats(self, layer):
        """Test getting overall stats."""
        stats = layer.get_stats()

        assert "registry" in stats
        assert "sharing" in stats
        assert "routing" in stats

        assert stats["registry"]["total_agents"] == 2
        assert stats["registry"]["active_agents"] == 2

    def test_get_agent_stats(self, layer):
        """Test getting agent-specific stats."""
        stats = layer.get_agent_stats("agent1")

        assert stats is not None
        assert stats["agent_id"] == "agent1"
        assert stats["memories_created"] == 0
        assert stats["queries_handled"] == 0

        # Non-existent agent
        assert layer.get_agent_stats("nonexistent") is None
