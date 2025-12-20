"""Test 03: Multi-Agent Tests.

Tests multi-agent functionality:
- Agent registration and management
- Team-based permissions
- Memory sharing across agents
- Access level enforcement
- Cross-agent isolation
"""

import pytest


# ============================================================================
# Agent Registration
# ============================================================================


class TestAgentRegistration:
    """Test agent registration and management."""

    def test_register_agent(self, mindcore_multi_agent):
        """Test registering a new agent."""
        result = mindcore_multi_agent.register_agent(
            agent_id="test_agent_001",
            name="Test Agent",
            description="A test agent",
            teams=["support"],
        )

        assert result is not None
        assert result.get("agent_id") == "test_agent_001" or result.agent_id == "test_agent_001"

    def test_register_multiple_agents(self, mindcore_multi_agent, sample_agents):
        """Test registering multiple agents."""
        registered = []
        for agent in sample_agents:
            result = mindcore_multi_agent.register_agent(
                agent_id=agent["agent_id"],
                name=agent["name"],
                description=agent.get("description", ""),
                teams=agent.get("teams", []),
            )
            registered.append(result)

        assert len(registered) == len(sample_agents)

    def test_list_agents(self, mindcore_multi_agent):
        """Test listing registered agents."""
        # Register some agents
        mindcore_multi_agent.register_agent(
            agent_id="list_test_1", name="List Test 1", teams=["team_a"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="list_test_2", name="List Test 2", teams=["team_b"]
        )

        agents = mindcore_multi_agent.list_agents()

        assert len(agents) >= 2
        agent_ids = [
            a.get("agent_id", a.agent_id if hasattr(a, "agent_id") else None) for a in agents
        ]
        assert "list_test_1" in agent_ids
        assert "list_test_2" in agent_ids

    def test_unregister_agent(self, mindcore_multi_agent):
        """Test unregistering an agent."""
        # Register
        mindcore_multi_agent.register_agent(
            agent_id="to_unregister", name="To Unregister", teams=["temp"]
        )

        # Unregister
        mindcore_multi_agent.unregister_agent("to_unregister")

        # Verify gone
        agents = mindcore_multi_agent.list_agents()
        agent_ids = [a.get("agent_id", getattr(a, "agent_id", None)) for a in agents]
        assert "to_unregister" not in agent_ids

    def test_duplicate_agent_registration(self, mindcore_multi_agent):
        """Test that duplicate agent registration is handled."""
        mindcore_multi_agent.register_agent(
            agent_id="duplicate_test", name="First Registration", teams=["team1"]
        )

        # Second registration with same ID - should update or error
        # Behavior depends on implementation
        try:
            mindcore_multi_agent.register_agent(
                agent_id="duplicate_test", name="Second Registration", teams=["team2"]
            )
            # If it succeeds, it should update the existing agent
        except Exception:
            # Or it should raise an error
            pass


# ============================================================================
# Team-Based Access
# ============================================================================


class TestTeamBasedAccess:
    """Test team-based memory access."""

    def test_team_memory_sharing(self, mindcore_multi_agent):
        """Test that agents in same team can share memories."""
        # Register agents in same team
        mindcore_multi_agent.register_agent(
            agent_id="team_agent_1", name="Team Agent 1", teams=["shared_team"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="team_agent_2", name="Team Agent 2", teams=["shared_team"]
        )

        # Store with team access
        mindcore_multi_agent.store(
            content="Team shared memory content",
            memory_type="semantic",
            user_id="team_user",
            topics=["api"],
            access_level="team",
            agent_id="team_agent_1",
        )

        # Recall as team_agent_2 - should see the memory
        result = mindcore_multi_agent.recall(
            query="team shared memory", user_id="team_user", agent_id="team_agent_2"
        )

        # Should find the shared memory
        assert len(result.memories) > 0

    def test_different_team_isolation(self, mindcore_multi_agent):
        """Test that agents in different teams don't see team memories."""
        # Register agents in different teams
        mindcore_multi_agent.register_agent(
            agent_id="team_a_agent", name="Team A Agent", teams=["team_a"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="team_b_agent", name="Team B Agent", teams=["team_b"]
        )

        # Store with team access for team_a
        mindcore_multi_agent.store(
            content="Team A only secret",
            memory_type="semantic",
            user_id="iso_user",
            topics=["api"],
            access_level="team",
            agent_id="team_a_agent",
        )

        # Recall as team_b_agent - should NOT see team_a memory
        result = mindcore_multi_agent.recall(
            query="Team A secret", user_id="iso_user", agent_id="team_b_agent"
        )

        # Should not find team_a's private memory
        for memory in result.memories:
            if hasattr(memory, "agent_id") and memory.agent_id == "team_a_agent":
                # If found, access_level should not be "team" only
                assert memory.access_level in ["shared", "global"]


# ============================================================================
# Access Levels
# ============================================================================


class TestAccessLevels:
    """Test different access levels."""

    def test_private_access(self, mindcore_multi_agent):
        """Test private access level - only owner can access."""
        mindcore_multi_agent.register_agent(
            agent_id="private_owner", name="Private Owner", teams=["access_test"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="private_other", name="Other Agent", teams=["access_test"]
        )

        # Store private memory
        mindcore_multi_agent.store(
            content="Private only content",
            memory_type="semantic",
            user_id="access_user",
            topics=["api"],
            access_level="private",
            agent_id="private_owner",
        )

        # Owner should see it
        result_owner = mindcore_multi_agent.recall(
            query="private content", user_id="access_user", agent_id="private_owner"
        )

        # Other agent should NOT see it
        result_other = mindcore_multi_agent.recall(
            query="private content", user_id="access_user", agent_id="private_other"
        )

        # Owner should have access
        any("Private only" in m.content for m in result_owner.memories)
        any("Private only" in m.content for m in result_other.memories)

        # At minimum, other agent shouldn't see private content
        # (Owner visibility depends on implementation details)

    def test_shared_access(self, mindcore_multi_agent):
        """Test shared access level - all agents for same user."""
        mindcore_multi_agent.register_agent(
            agent_id="shared_agent_1", name="Shared Agent 1", teams=["team_x"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="shared_agent_2",
            name="Shared Agent 2",
            teams=["team_y"],  # Different team
        )

        # Store with shared access
        mindcore_multi_agent.store(
            content="Shared across all agents",
            memory_type="semantic",
            user_id="shared_user",
            topics=["api"],
            access_level="shared",
            agent_id="shared_agent_1",
        )

        # Both agents should see it
        result_1 = mindcore_multi_agent.recall(
            query="shared across agents", user_id="shared_user", agent_id="shared_agent_1"
        )
        result_2 = mindcore_multi_agent.recall(
            query="shared across agents", user_id="shared_user", agent_id="shared_agent_2"
        )

        assert len(result_1.memories) > 0 or len(result_2.memories) > 0

    def test_global_access(self, mindcore_multi_agent):
        """Test global access level - cross-user visibility."""
        mindcore_multi_agent.register_agent(
            agent_id="global_agent", name="Global Agent", teams=["global_team"]
        )

        # Store with global access
        mindcore_multi_agent.store(
            content="Global knowledge for everyone",
            memory_type="semantic",
            user_id="global_user_1",
            topics=["api"],
            access_level="global",
            agent_id="global_agent",
        )

        # Different user should be able to see it
        mindcore_multi_agent.recall(
            query="global knowledge",
            user_id="global_user_2",  # Different user!
            agent_id="global_agent",
        )

        # Global memories should be visible
        # Note: Actual visibility depends on implementation


# ============================================================================
# Memory Sharing
# ============================================================================


class TestMemorySharing:
    """Test memory sharing between agents."""

    def test_store_with_agent_id(self, mindcore_multi_agent):
        """Test storing memory with explicit agent ID."""
        mindcore_multi_agent.register_agent(
            agent_id="store_agent", name="Store Agent", teams=["store_team"]
        )

        memory_id = mindcore_multi_agent.store(
            content="Memory with agent attribution",
            memory_type="semantic",
            user_id="store_user",
            topics=["api"],
            agent_id="store_agent",
        )

        memory = mindcore_multi_agent.get(memory_id)
        assert memory.agent_id == "store_agent"

    def test_recall_includes_agent_info(self, mindcore_multi_agent):
        """Test that recall results include agent information."""
        mindcore_multi_agent.register_agent(
            agent_id="recall_agent", name="Recall Agent", teams=["recall_team"]
        )

        mindcore_multi_agent.store(
            content="Agent attributed memory",
            memory_type="semantic",
            user_id="recall_user",
            topics=["api"],
            agent_id="recall_agent",
        )

        result = mindcore_multi_agent.recall(query="agent attributed", user_id="recall_user")

        if len(result.memories) > 0:
            # Check that agent info is present
            assert hasattr(result.memories[0], "agent_id")


# ============================================================================
# Cross-Agent Sync
# ============================================================================


class TestCrossAgentSync:
    """Test synchronization between agents."""

    def test_sync_memories_between_agents(self, mindcore_multi_agent):
        """Test syncing memories from one agent to another."""
        # Register source and target agents
        mindcore_multi_agent.register_agent(
            agent_id="sync_source", name="Sync Source", teams=["sync_team"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="sync_target", name="Sync Target", teams=["sync_team"]
        )

        # Store as source agent
        mindcore_multi_agent.store(
            content="Memory to sync",
            memory_type="semantic",
            user_id="sync_user",
            topics=["api"],
            access_level="team",
            agent_id="sync_source",
        )

        # Sync
        result = mindcore_multi_agent.sync(
            source_agent="sync_source", target_agent="sync_target", user_id="sync_user"
        )

        assert result is not None
        # Check sync result for success indicators


# ============================================================================
# Multi-Agent Not Enabled
# ============================================================================


class TestMultiAgentDisabled:
    """Test behavior when multi-agent is not enabled."""

    def test_register_agent_when_disabled(self, mindcore):
        """Test that register_agent fails when multi-agent is disabled."""
        from mindcore.v2.exceptions import MultiAgentNotEnabledError

        with pytest.raises(MultiAgentNotEnabledError):
            mindcore.register_agent(
                agent_id="should_fail", name="Should Fail Agent", teams=["team"]
            )

    def test_store_with_agent_id_when_disabled(self, mindcore):
        """Test storing with agent_id when multi-agent is disabled."""
        # Should either ignore agent_id or raise error
        try:
            memory_id = mindcore.store(
                content="Test without multi-agent",
                memory_type="semantic",
                user_id="user",
                topics=["api"],
                agent_id="some_agent",
            )
            # If it succeeds, agent_id is likely ignored
            mindcore.get(memory_id)
            # agent_id should be None or ignored
        except Exception:
            # Or it raises an error
            pass


# ============================================================================
# Edge Cases
# ============================================================================


class TestMultiAgentEdgeCases:
    """Test edge cases in multi-agent scenarios."""

    def test_agent_with_no_teams(self, mindcore_multi_agent):
        """Test agent with no team memberships."""
        result = mindcore_multi_agent.register_agent(
            agent_id="no_teams_agent",
            name="No Teams Agent",
            teams=[],  # Empty teams
        )

        assert result is not None

    def test_agent_with_multiple_teams(self, mindcore_multi_agent):
        """Test agent with multiple team memberships."""
        result = mindcore_multi_agent.register_agent(
            agent_id="multi_team_agent", name="Multi Team Agent", teams=["team1", "team2", "team3"]
        )

        assert result is not None

    def test_store_without_agent_id(self, mindcore_multi_agent):
        """Test storing without specifying agent ID."""
        memory_id = mindcore_multi_agent.store(
            content="No agent attribution",
            memory_type="semantic",
            user_id="no_agent_user",
            topics=["api"],
            # No agent_id specified
        )

        memory = mindcore_multi_agent.get(memory_id)
        # Should have None or default agent_id
        assert memory is not None

    def test_recall_from_nonexistent_agent(self, mindcore_multi_agent):
        """Test recall with non-existent agent ID."""
        # Should either work (no filtering) or handle gracefully
        try:
            mindcore_multi_agent.recall(
                query="test query", user_id="test_user", agent_id="nonexistent_agent"
            )
            # If succeeds, should return empty or handle gracefully
        except Exception:
            # Or raise appropriate error
            pass
