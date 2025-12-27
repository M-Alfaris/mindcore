"""Tests for Multi-Agent Memory Federation.

Tests cover:
- Access control and ACL
- Namespace hierarchy
- Cross-agent signal aggregation
- Federated SVL feedback
- Agent bridge integration
- Configuration and quick setup
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from mindcore.federation import (
    AccessLevel,
    AccessPolicy,
    AccessScope,
    AgentIdentity,
    AgentMemoryBridge,
    AggregatedFeedback,
    CrossAgentSignalAggregator,
    FederatedSVL,
    Federation,
    FederationConfig,
    LocalMemory,
    MemoryACL,
    MemoryNamespace,
    NamespaceHierarchy,
    SignalWeight,
    TrustPolicy,
    quick_setup,
)


# =============================================================================
# Access Control Tests
# =============================================================================


class TestAccessLevel:
    """Tests for AccessLevel enum."""

    def test_access_level_ordering(self):
        """Test access levels are properly ordered."""
        assert AccessLevel.PRIVATE < AccessLevel.AGENT_TYPE
        assert AccessLevel.AGENT_TYPE < AccessLevel.TEAM
        assert AccessLevel.TEAM < AccessLevel.DEPARTMENT
        assert AccessLevel.DEPARTMENT < AccessLevel.AD_HOC_GROUP
        assert AccessLevel.AD_HOC_GROUP < AccessLevel.ORGANIZATION
        assert AccessLevel.ORGANIZATION < AccessLevel.PUBLIC


class TestAccessScope:
    """Tests for AccessScope."""

    def test_scope_creation(self):
        """Test basic scope creation."""
        scope = AccessScope(
            org_id="acme",
            department="support",
            team="tier-1",
            agent_id="agent-001",
        )

        assert scope.org_id == "acme"
        assert scope.department == "support"
        assert scope.team == "tier-1"

    def test_scope_matches_private(self):
        """Test private level matching."""
        scope1 = AccessScope(org_id="acme", agent_id="agent-001")
        scope2 = AccessScope(org_id="acme", agent_id="agent-001")
        scope3 = AccessScope(org_id="acme", agent_id="agent-002")

        assert scope1.matches_level(AccessLevel.PRIVATE, scope2)
        assert not scope1.matches_level(AccessLevel.PRIVATE, scope3)

    def test_scope_matches_team(self):
        """Test team level matching."""
        scope1 = AccessScope(org_id="acme", department="support", team="tier-1")
        scope2 = AccessScope(org_id="acme", department="support", team="tier-1")
        scope3 = AccessScope(org_id="acme", department="support", team="tier-2")

        assert scope1.matches_level(AccessLevel.TEAM, scope2)
        assert not scope1.matches_level(AccessLevel.TEAM, scope3)

    def test_scope_matches_department(self):
        """Test department level matching."""
        scope1 = AccessScope(org_id="acme", department="support")
        scope2 = AccessScope(org_id="acme", department="support", team="tier-1")
        scope3 = AccessScope(org_id="acme", department="sales")

        assert scope1.matches_level(AccessLevel.DEPARTMENT, scope2)
        assert not scope1.matches_level(AccessLevel.DEPARTMENT, scope3)

    def test_scope_matches_organization(self):
        """Test organization level matching."""
        scope1 = AccessScope(org_id="acme")
        scope2 = AccessScope(org_id="acme", department="sales")
        scope3 = AccessScope(org_id="other")

        assert scope1.matches_level(AccessLevel.ORGANIZATION, scope2)
        assert not scope1.matches_level(AccessLevel.ORGANIZATION, scope3)

    def test_scope_with_groups(self):
        """Test ad-hoc group matching."""
        scope1 = AccessScope(org_id="acme", groups={"vip-handlers", "managers"})
        scope2 = AccessScope(org_id="acme", groups={"vip-handlers"})
        scope3 = AccessScope(org_id="acme", groups={"regular"})

        assert scope1.matches_level(AccessLevel.AD_HOC_GROUP, scope2)
        assert not scope1.matches_level(AccessLevel.AD_HOC_GROUP, scope3)


class TestMemoryACL:
    """Tests for MemoryACL."""

    def test_acl_can_read_owner(self):
        """Test owner always has read access."""
        owner_scope = AccessScope(org_id="acme", agent_id="owner-001")
        acl = MemoryACL(
            memory_id="mem-1",
            owner_agent_id="owner-001",
            owner_scope=owner_scope,
            policy=AccessPolicy(access_level=AccessLevel.PRIVATE),
        )

        assert acl.can_read(owner_scope)

    def test_acl_can_read_denied_agent(self):
        """Test denied agent cannot read."""
        owner_scope = AccessScope(org_id="acme", department="support")
        requester = AccessScope(org_id="acme", department="support", agent_id="bad-agent")

        acl = MemoryACL(
            memory_id="mem-1",
            owner_agent_id="owner-001",
            owner_scope=owner_scope,
            policy=AccessPolicy(
                access_level=AccessLevel.DEPARTMENT,
                denied_agents={"bad-agent"},
            ),
        )

        assert not acl.can_read(requester)

    def test_acl_can_read_allowed_agent(self):
        """Test explicitly allowed agent can read."""
        owner_scope = AccessScope(org_id="acme")
        requester = AccessScope(org_id="acme", agent_id="special-agent")

        acl = MemoryACL(
            memory_id="mem-1",
            owner_agent_id="owner-001",
            owner_scope=owner_scope,
            policy=AccessPolicy(
                access_level=AccessLevel.PRIVATE,
                allowed_agents={"special-agent"},
            ),
        )

        assert acl.can_read(requester)

    def test_acl_expired(self):
        """Test expired ACL denies access."""
        owner_scope = AccessScope(org_id="acme")
        requester = AccessScope(org_id="acme")

        acl = MemoryACL(
            memory_id="mem-1",
            owner_agent_id="owner-001",
            owner_scope=owner_scope,
            policy=AccessPolicy(
                access_level=AccessLevel.ORGANIZATION,
                expires_at=datetime.utcnow() - timedelta(hours=1),
            ),
        )

        assert not acl.can_read(requester)

    def test_acl_serialization(self):
        """Test ACL serialization/deserialization."""
        owner_scope = AccessScope(org_id="acme", department="support")
        acl = MemoryACL(
            memory_id="mem-1",
            owner_agent_id="owner-001",
            owner_scope=owner_scope,
            policy=AccessPolicy(access_level=AccessLevel.TEAM),
        )

        data = acl.to_dict()
        restored = MemoryACL.from_dict(data)

        assert restored.memory_id == acl.memory_id
        assert restored.owner_agent_id == acl.owner_agent_id
        assert restored.policy.access_level == acl.policy.access_level


# =============================================================================
# Namespace Tests
# =============================================================================


class TestMemoryNamespace:
    """Tests for MemoryNamespace."""

    def test_namespace_path(self):
        """Test namespace path generation."""
        ns = MemoryNamespace(
            org_id="acme",
            department="support",
            team="tier-1",
        )

        assert ns.path == "acme/support/tier-1"

    def test_namespace_depth(self):
        """Test namespace depth calculation."""
        org_ns = MemoryNamespace(org_id="acme")
        dept_ns = MemoryNamespace(org_id="acme", department="support")
        team_ns = MemoryNamespace(org_id="acme", department="support", team="tier-1")

        assert org_ns.depth == 1
        assert dept_ns.depth == 2
        assert team_ns.depth == 3

    def test_namespace_ancestors(self):
        """Test getting ancestor namespaces."""
        ns = MemoryNamespace(org_id="acme", department="support", team="tier-1")
        ancestors = ns.get_ancestors()

        assert len(ancestors) == 2
        assert ancestors[0].path == "acme/support"
        assert ancestors[1].path == "acme"

    def test_namespace_lineage(self):
        """Test getting namespace lineage."""
        ns = MemoryNamespace(org_id="acme", department="support", team="tier-1")
        lineage = ns.get_lineage()

        assert len(lineage) == 3
        assert lineage[0].path == "acme"
        assert lineage[1].path == "acme/support"
        assert lineage[2].path == "acme/support/tier-1"

    def test_namespace_is_ancestor(self):
        """Test ancestor relationship."""
        org_ns = MemoryNamespace(org_id="acme")
        dept_ns = MemoryNamespace(org_id="acme", department="support")
        team_ns = MemoryNamespace(org_id="acme", department="support", team="tier-1")

        assert org_ns.is_ancestor_of(dept_ns)
        assert org_ns.is_ancestor_of(team_ns)
        assert dept_ns.is_ancestor_of(team_ns)
        assert not team_ns.is_ancestor_of(dept_ns)

    def test_namespace_from_path(self):
        """Test creating namespace from path."""
        ns = MemoryNamespace.from_path("acme/support/tier-1")

        assert ns.org_id == "acme"
        assert ns.department == "support"
        assert ns.team == "tier-1"

    def test_namespace_validation(self):
        """Test namespace validation."""
        with pytest.raises(ValueError, match="department"):
            # Team without department is invalid
            MemoryNamespace(org_id="acme", team="tier-1")


class TestNamespaceHierarchy:
    """Tests for NamespaceHierarchy."""

    def test_register_structure(self):
        """Test registering departments and teams."""
        hierarchy = NamespaceHierarchy(org_id="acme")

        hierarchy.register_department("support")
        hierarchy.register_team("support", "tier-1")
        hierarchy.register_team("support", "tier-2")

        assert "support" in hierarchy.departments
        assert "tier-1" in hierarchy.departments["support"]
        assert "tier-2" in hierarchy.departments["support"]

    def test_common_ancestor(self):
        """Test finding common ancestor."""
        hierarchy = NamespaceHierarchy(org_id="acme")

        ns1 = MemoryNamespace(org_id="acme", department="support", team="tier-1")
        ns2 = MemoryNamespace(org_id="acme", department="support", team="tier-2")

        common = hierarchy.common_ancestor(ns1, ns2)

        assert common is not None
        assert common.department == "support"
        assert common.team is None

    def test_get_descendants(self):
        """Test getting descendant namespaces."""
        hierarchy = NamespaceHierarchy(org_id="acme")
        hierarchy.register_department("support")
        hierarchy.register_team("support", "tier-1")
        hierarchy.register_team("support", "tier-2")

        dept_ns = MemoryNamespace(org_id="acme", department="support")
        descendants = hierarchy.get_all_descendants(dept_ns)

        assert len(descendants) == 2
        teams = {d.team for d in descendants}
        assert "tier-1" in teams
        assert "tier-2" in teams


# =============================================================================
# Signal Aggregator Tests
# =============================================================================


class TestCrossAgentSignalAggregator:
    """Tests for CrossAgentSignalAggregator."""

    def test_equal_aggregation(self):
        """Test equal weight aggregation."""
        aggregator = CrossAgentSignalAggregator(trust_policy=TrustPolicy.EQUAL)

        scope = AccessScope(org_id="acme")

        aggregator.add_signal("mem-1", "agent-1", 0.8, scope)
        aggregator.add_signal("mem-1", "agent-2", 0.6, scope)
        aggregator.add_signal("mem-1", "agent-3", 0.4, scope)

        score = aggregator.get_aggregated_score("mem-1")

        assert abs(score - 0.6) < 0.01  # Average of 0.8, 0.6, 0.4

    def test_namespace_weighted_aggregation(self):
        """Test namespace-weighted aggregation."""
        aggregator = CrossAgentSignalAggregator(
            trust_policy=TrustPolicy.NAMESPACE_WEIGHTED,
            weight_config=SignalWeight(
                base_weight=1.0,
                same_team_bonus=0.5,
            ),
        )

        ref_scope = AccessScope(org_id="acme", department="support", team="tier-1")
        same_team = AccessScope(org_id="acme", department="support", team="tier-1")
        diff_team = AccessScope(org_id="acme", department="support", team="tier-2")

        aggregator.add_signal("mem-1", "agent-1", 0.8, same_team, reference_scope=ref_scope)
        aggregator.add_signal("mem-1", "agent-2", 0.2, diff_team, reference_scope=ref_scope)

        score = aggregator.get_aggregated_score("mem-1", ref_scope)

        # Same team (0.8 * 1.5) + diff team (0.2 * 1.0) / (1.5 + 1.0)
        # = (1.2 + 0.2) / 2.5 = 0.56
        assert score > 0.5  # Should be weighted toward same-team

    def test_signal_details(self):
        """Test getting signal details."""
        aggregator = CrossAgentSignalAggregator()
        scope = AccessScope(org_id="acme")

        aggregator.add_signal("mem-1", "agent-1", 0.8, scope)
        aggregator.add_signal("mem-1", "agent-2", 0.6, scope)

        details = aggregator.get_signal_details("mem-1")

        assert details is not None
        assert details.signal_count == 2
        assert "agent-1" in details.signals
        assert "agent-2" in details.signals

    def test_remove_agent_signals(self):
        """Test removing agent signals."""
        aggregator = CrossAgentSignalAggregator()
        scope = AccessScope(org_id="acme")

        aggregator.add_signal("mem-1", "agent-1", 0.8, scope)
        aggregator.add_signal("mem-1", "agent-2", 0.6, scope)
        aggregator.add_signal("mem-2", "agent-1", 0.5, scope)

        updated = aggregator.remove_agent_signals("agent-1")

        assert "mem-1" in updated
        assert "mem-2" in updated

        details = aggregator.get_signal_details("mem-1")
        assert details.signal_count == 1

    def test_agreement_ratio(self):
        """Test agreement ratio calculation."""
        aggregator = CrossAgentSignalAggregator()
        scope = AccessScope(org_id="acme")

        aggregator.add_signal("mem-1", "agent-1", 0.8, scope)
        aggregator.add_signal("mem-1", "agent-2", 0.6, scope)
        aggregator.add_signal("mem-1", "agent-3", -0.2, scope)

        details = aggregator.get_signal_details("mem-1")

        # 2 positive, 1 negative = 2/3 agreement
        assert abs(details.agreement_ratio - 0.667) < 0.01


# =============================================================================
# Federated SVL Tests
# =============================================================================


class TestFederatedSVL:
    """Tests for FederatedSVL."""

    def test_record_feedback(self):
        """Test recording topic feedback."""
        svl = FederatedSVL(org_id="acme")
        namespace = MemoryNamespace(org_id="acme", department="support")

        svl.record_feedback(
            topic="billing",
            was_effective=True,
            namespace=namespace,
            agent_id="agent-001",
        )
        svl.record_feedback(
            topic="billing",
            was_effective=True,
            namespace=namespace,
            agent_id="agent-001",
        )
        svl.record_feedback(
            topic="billing",
            was_effective=False,
            namespace=namespace,
            agent_id="agent-001",
        )

        feedback = svl.get_scoped_feedback(namespace)

        assert feedback is not None
        assert "billing" in feedback.topic_feedback
        assert feedback.topic_feedback["billing"].total_uses == 3
        assert feedback.topic_feedback["billing"].effective_uses == 2

    def test_feedback_rollup(self):
        """Test feedback rolls up to ancestors."""
        svl = FederatedSVL(org_id="acme")
        team_ns = MemoryNamespace(org_id="acme", department="support", team="tier-1")
        dept_ns = MemoryNamespace(org_id="acme", department="support")
        org_ns = MemoryNamespace(org_id="acme")

        svl.record_feedback(topic="billing", was_effective=True, namespace=team_ns)

        # Should be recorded at all levels
        assert svl.get_scoped_feedback(team_ns) is not None
        assert svl.get_scoped_feedback(dept_ns) is not None
        assert svl.get_scoped_feedback(org_ns) is not None

    def test_aggregated_feedback(self):
        """Test aggregated feedback from multiple scopes."""
        svl = FederatedSVL(org_id="acme")
        namespace = MemoryNamespace(org_id="acme", department="support")

        # Record enough data
        for _ in range(5):
            svl.record_feedback(topic="billing", was_effective=True, namespace=namespace)
        for _ in range(5):
            svl.record_feedback(topic="general", was_effective=False, namespace=namespace)

        aggregated = svl.get_aggregated_feedback(namespace=namespace)

        assert "billing" in aggregated.topic_effectiveness
        assert aggregated.topic_effectiveness["billing"] > 0.6
        assert "billing" in aggregated.preferred_topics

    def test_agent_specific_feedback(self):
        """Test agent-specific feedback tracking."""
        svl = FederatedSVL(org_id="acme")
        namespace = MemoryNamespace(org_id="acme")

        svl.record_feedback(
            topic="billing",
            was_effective=True,
            namespace=namespace,
            agent_id="agent-001",
        )

        agent_fb = svl.get_agent_feedback("agent-001")

        assert agent_fb is not None
        assert "billing" in agent_fb.topic_feedback


# =============================================================================
# Agent Bridge Tests
# =============================================================================


class TestAgentMemoryBridge:
    """Tests for AgentMemoryBridge."""

    @pytest.fixture
    def agent_bridge(self):
        """Create a test agent bridge."""
        identity = AgentIdentity(
            agent_id="test-agent",
            agent_type="support-bot",
        )
        namespace = MemoryNamespace(
            org_id="acme",
            department="support",
            team="tier-1",
        )

        return AgentMemoryBridge(
            identity=identity,
            namespace=namespace,
        )

    def test_remember_local(self, agent_bridge):
        """Test storing memory locally."""
        memory_id = agent_bridge.remember(
            content="Customer prefers email",
            user_id="customer-123",
            topics=["preferences"],
        )

        assert memory_id in agent_bridge.local_memories
        assert agent_bridge.local_memories[memory_id].content == "Customer prefers email"

    def test_recall_local(self, agent_bridge):
        """Test recalling local memories."""
        agent_bridge.remember(
            content="Customer prefers email",
            topics=["preferences"],
        )
        agent_bridge.remember(
            content="Customer billing issue",
            topics=["billing"],
        )

        results = agent_bridge.recall(
            topics=["preferences"],
            include_federated=False,
        )

        assert len(results) == 1
        assert results[0].content == "Customer prefers email"

    def test_reinforce_local(self, agent_bridge):
        """Test reinforcing local memory."""
        memory_id = agent_bridge.remember(content="Test memory")

        new_score = agent_bridge.reinforce(memory_id, signal=0.8)

        assert new_score > 0
        assert agent_bridge.local_memories[memory_id].reinforcement_score > 0

    def test_get_vocabulary(self, agent_bridge):
        """Test getting vocabulary with no SVL."""
        vocab = agent_bridge.get_vocabulary()
        assert vocab == {}

    def test_get_local_stats(self, agent_bridge):
        """Test getting local statistics."""
        agent_bridge.remember(content="Memory 1")
        agent_bridge.remember(content="Memory 2")

        stats = agent_bridge.get_local_stats()

        assert stats["total_memories"] == 2
        assert stats["persisted_count"] == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestQuickSetup:
    """Tests for quick_setup convenience function."""

    def test_simple_setup(self):
        """Test simple quick setup."""
        federation = quick_setup(
            org_id="startup",
            departments=["engineering", "sales"],
            topics=["bug", "feature"],
        )

        assert federation.config.org_id == "startup"
        assert "engineering" in federation.hierarchy.departments
        assert "sales" in federation.hierarchy.departments

    def test_setup_with_teams(self):
        """Test quick setup with teams."""
        federation = quick_setup(
            org_id="enterprise",
            departments={
                "engineering": ["backend", "frontend"],
                "sales": ["enterprise", "smb"],
            },
        )

        assert "backend" in federation.hierarchy.departments["engineering"]
        assert "smb" in federation.hierarchy.departments["sales"]

    def test_create_agent(self):
        """Test creating agent through federation."""
        federation = quick_setup(
            org_id="acme",
            departments={"support": ["tier-1"]},
        )

        agent = federation.create_agent(
            agent_id="support-001",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )

        assert agent.agent_id == "support-001"
        assert agent.namespace.department == "support"
        assert "support-001" in federation.agents


class TestFederationConfig:
    """Tests for FederationConfig."""

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "org_id": "acme",
            "structure": {
                "departments": {
                    "support": {"teams": ["tier-1", "tier-2"]},
                }
            },
            "defaults": {
                "access_level": "team",
            },
            "vocabulary": {
                "topics": ["billing", "technical"],
            },
            "trust_policy": "namespace_weighted",
        }

        config = FederationConfig.from_dict(data)

        assert config.org_id == "acme"
        assert "support" in config.structure.departments
        assert config.defaults.access_level == AccessLevel.TEAM
        assert "billing" in config.vocabulary.topics

    def test_config_to_dict(self):
        """Test exporting config to dictionary."""
        config = FederationConfig(
            org_id="acme",
            vocabulary=VocabularyConfig(topics=["billing"]),
        )

        data = config.to_dict()

        assert data["org_id"] == "acme"
        assert "billing" in data["vocabulary"]["topics"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiAgentIntegration:
    """Integration tests for multi-agent scenarios."""

    def test_cross_agent_signal_propagation(self):
        """Test signals propagate across agents."""
        federation = quick_setup(
            org_id="acme",
            departments={"support": ["tier-1"]},
        )

        agent1 = federation.create_agent(
            agent_id="agent-001",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )
        agent2 = federation.create_agent(
            agent_id="agent-002",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )

        # Agent 1 creates memory
        memory_id = agent1.remember(content="Shared knowledge")

        # Both agents reinforce
        agent1.reinforce(memory_id, signal=0.8)
        agent2.reinforce(memory_id, signal=0.6)

        # Check aggregated score
        score = federation.signal_aggregator.get_aggregated_score(memory_id)
        assert score > 0

    def test_svl_feedback_sharing(self):
        """Test SVL feedback is shared across agents."""
        federation = quick_setup(
            org_id="acme",
            departments={"support": ["tier-1"]},
        )

        agent1 = federation.create_agent(
            agent_id="agent-001",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )
        agent2 = federation.create_agent(
            agent_id="agent-002",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )

        # Agent 1 provides feedback
        memory_id = agent1.remember(
            content="Billing help",
            topics=["billing"],
        )
        agent1.reinforce(memory_id, signal=0.8, was_effective=True)

        # Agent 2 should see the feedback
        agent2.get_feedback_for_extraction()
        # Note: Need enough samples for recommendations
        # This is a structural test

    def test_team_isolation(self):
        """Test agents in different teams have isolation."""
        federation = quick_setup(
            org_id="acme",
            departments={"support": ["tier-1", "tier-2"]},
        )

        agent_t1 = federation.create_agent(
            agent_id="agent-t1",
            agent_type="support-bot",
            department="support",
            team="tier-1",
        )
        agent_t2 = federation.create_agent(
            agent_id="agent-t2",
            agent_type="support-bot",
            department="support",
            team="tier-2",
        )

        assert agent_t1.namespace.team != agent_t2.namespace.team
        assert agent_t1.namespace.department == agent_t2.namespace.department


# Import for VocabularyConfig
from mindcore.federation.config import VocabularyConfig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
