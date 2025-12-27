"""Extended tests for Federation - FederatedCLST and related components.

Tests cover:
- FederatedCLST: store, search, get, delete, reinforce
- NamespacedQuery configuration
- FederatedMemory dataclass
- Access control in federated context
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from mindcore.v2.federation.access_control import (
    AccessLevel,
    AccessPolicy,
    AccessScope,
    MemoryACL,
)
from mindcore.v2.federation.federated_clst import (
    FederatedCLST,
    FederatedMemory,
    NamespacedQuery,
)
from mindcore.v2.federation.namespace import MemoryNamespace


# =============================================================================
# Mock Storage Backend
# =============================================================================


class MockStorageBackend:
    """Mock storage backend for testing."""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def store(self, memory_id: str, vector: list[float], metadata: dict) -> None:
        self._data[memory_id] = {
            "memory_id": memory_id,
            "vector": vector,
            **metadata,
        }

    def search(
        self,
        query_vector: list[float],
        filter: dict | None = None,
        limit: int = 10,
    ) -> list[dict]:
        results = []
        for data in self._data.values():
            # Simple filter matching - be more lenient for tests
            if filter:
                match = True
                org_id = filter.get("org_id")
                if org_id:
                    ns = data.get("namespace", {})
                    data_org = ns.get("org_id", "") if isinstance(ns, dict) else ""
                    if data_org != org_id:
                        match = False

                # For namespace_path, check if any path matches
                ns_filter = filter.get("namespace_path")
                if ns_filter and isinstance(ns_filter, dict) and "$in" in ns_filter:
                    ns = data.get("namespace", {})
                    data_path = ns.get("path", "") if isinstance(ns, dict) else ""
                    # Check if data_path matches or is a parent of any filter path
                    paths = ns_filter["$in"]
                    path_match = False
                    for p in paths:
                        if data_path in p or p in data_path or data_path == "" or p == "":
                            path_match = True
                            break
                    if not path_match:
                        match = False

                if not match:
                    continue

            results.append({**data, "score": 0.9})

        return results[:limit]

    def get(self, memory_id: str) -> dict | None:
        return self._data.get(memory_id)

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        if memory_id in self._data:
            self._data[memory_id].update(metadata)

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._data:
            del self._data[memory_id]
            return True
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage():
    """Create mock storage backend."""
    return MockStorageBackend()


@pytest.fixture
def org_namespace():
    """Create organization namespace."""
    return MemoryNamespace(org_id="acme-corp")


@pytest.fixture
def team_namespace():
    """Create team namespace."""
    return MemoryNamespace(
        org_id="acme-corp",
        department="engineering",
        team="backend",
    )


@pytest.fixture
def owner_scope():
    """Create owner access scope."""
    return AccessScope(
        org_id="acme-corp",
        department="engineering",
        team="backend",
        agent_id="agent_1",
    )


@pytest.fixture
def other_scope():
    """Create other agent's access scope."""
    return AccessScope(
        org_id="acme-corp",
        department="engineering",
        team="frontend",
        agent_id="agent_2",
    )


@pytest.fixture
def federated_clst(storage):
    """Create FederatedCLST instance."""
    return FederatedCLST(
        org_id="acme-corp",
        storage=storage,
    )


# =============================================================================
# NamespacedQuery Tests
# =============================================================================


class TestNamespacedQuery:
    """Tests for NamespacedQuery dataclass."""

    def test_create_query(self, team_namespace, owner_scope):
        """Test creating a namespaced query."""
        query = NamespacedQuery(
            query_vector=[0.1, 0.2, 0.3],
            namespaces=[team_namespace],
            requester=owner_scope,
            include_ancestors=True,
            limit=10,
        )

        assert query.query_vector == [0.1, 0.2, 0.3]
        assert len(query.namespaces) == 1
        assert query.include_ancestors is True
        assert query.limit == 10

    def test_query_with_time_range(self, team_namespace, owner_scope):
        """Test query with time range filter."""
        now = datetime.now(timezone.utc)
        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
            time_range=(now - timedelta(days=7), now),
        )

        assert query.time_range is not None
        assert query.time_range[0] < query.time_range[1]

    def test_query_with_access_levels(self, team_namespace, owner_scope):
        """Test query with access level filter."""
        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
            access_levels=[AccessLevel.TEAM, AccessLevel.PUBLIC],
        )

        assert len(query.access_levels) == 2
        assert AccessLevel.TEAM in query.access_levels

    def test_query_defaults(self, team_namespace, owner_scope):
        """Test query default values."""
        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
        )

        assert query.include_ancestors is True
        assert query.include_descendants is False
        assert query.limit == 10
        assert query.min_score == 0.0


# =============================================================================
# FederatedMemory Tests
# =============================================================================


class TestFederatedMemory:
    """Tests for FederatedMemory dataclass."""

    def test_create_memory(self, team_namespace, owner_scope):
        """Test creating a federated memory."""
        acl = MemoryACL(
            memory_id="mem_1",
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            policy=AccessPolicy(access_level=AccessLevel.TEAM),
        )

        memory = FederatedMemory(
            memory_id="mem_1",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            namespace=team_namespace,
            acl=acl,
        )

        assert memory.memory_id == "mem_1"
        assert memory.content == "Test content"
        assert memory.aggregated_reinforcement == 0.0

    def test_to_storage_dict(self, team_namespace, owner_scope):
        """Test serialization to storage dict."""
        acl = MemoryACL(
            memory_id="mem_1",
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            policy=AccessPolicy(access_level=AccessLevel.TEAM),
        )

        memory = FederatedMemory(
            memory_id="mem_1",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            namespace=team_namespace,
            acl=acl,
            metadata={"key": "value"},
        )

        data = memory.to_storage_dict()

        assert data["memory_id"] == "mem_1"
        assert data["content"] == "Test content"
        assert "namespace" in data
        assert "acl" in data
        assert "created_at" in data

    def test_reinforcement_sources(self, team_namespace, owner_scope):
        """Test reinforcement sources tracking."""
        acl = MemoryACL(
            memory_id="mem_1",
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        memory = FederatedMemory(
            memory_id="mem_1",
            content="Test",
            vector=[0.1],
            namespace=team_namespace,
            acl=acl,
            reinforcement_sources={"agent_1": 0.5, "agent_2": 0.8},
            aggregated_reinforcement=0.65,
        )

        assert memory.reinforcement_sources["agent_1"] == 0.5
        assert memory.aggregated_reinforcement == 0.65


# =============================================================================
# FederatedCLST Store Tests
# =============================================================================


class TestFederatedCLSTStore:
    """Tests for FederatedCLST.store method."""

    def test_store_memory(self, federated_clst, team_namespace, owner_scope):
        """Test storing a federated memory."""
        memory = federated_clst.store(
            memory_id="mem_1",
            content="Important information",
            vector=[0.1, 0.2, 0.3],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.TEAM,
        )

        assert memory.memory_id == "mem_1"
        assert memory.content == "Important information"
        assert memory.namespace.team == "backend"

    def test_store_with_policy(self, federated_clst, team_namespace, owner_scope):
        """Test storing with custom access policy."""
        policy = AccessPolicy(
            access_level=AccessLevel.TEAM,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        memory = federated_clst.store(
            memory_id="mem_2",
            content="Temporary info",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            policy=policy,
        )

        assert memory.acl.policy.expires_at is not None

    def test_store_wrong_org_raises(self, federated_clst, owner_scope):
        """Test storing with wrong org raises error."""
        wrong_namespace = MemoryNamespace(org_id="other-corp")

        with pytest.raises(ValueError, match="doesn't match"):
            federated_clst.store(
                memory_id="mem_3",
                content="Test",
                vector=[0.1],
                namespace=wrong_namespace,
                owner_agent_id="agent_1",
                owner_scope=owner_scope,
            )

    def test_store_caches_acl(self, federated_clst, team_namespace, owner_scope):
        """Test that store caches ACL."""
        federated_clst.store(
            memory_id="mem_4",
            content="Test",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        assert "mem_4" in federated_clst.acl_store


# =============================================================================
# FederatedCLST Search Tests
# =============================================================================


class TestFederatedCLSTSearch:
    """Tests for FederatedCLST.search method."""

    def test_search_own_memories(self, federated_clst, team_namespace, owner_scope):
        """Test searching for own memories."""
        # Store a memory first
        federated_clst.store(
            memory_id="search_1",
            content="Searchable content",
            vector=[0.1, 0.2],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.PRIVATE,
        )

        query = NamespacedQuery(
            query_vector=[0.1, 0.2],
            namespaces=[team_namespace],
            requester=owner_scope,
        )

        results = federated_clst.search(query)

        assert len(results) >= 1

    def test_search_with_ancestor_namespaces(self, federated_clst, team_namespace, owner_scope):
        """Test search includes ancestor namespaces."""
        # Store at org level
        org_namespace = MemoryNamespace(org_id="acme-corp")
        federated_clst.store(
            memory_id="org_mem",
            content="Org-wide knowledge",
            vector=[0.1],
            namespace=org_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.PUBLIC,
        )

        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
            include_ancestors=True,
        )

        results = federated_clst.search(query)

        # Should find org-level memory
        assert len(results) >= 1

    def test_search_respects_min_score(self, federated_clst, team_namespace, owner_scope):
        """Test search respects minimum score threshold."""
        federated_clst.store(
            memory_id="low_score",
            content="Test",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
            min_score=0.99,  # Very high threshold
        )

        results = federated_clst.search(query)

        # High threshold should filter out results
        assert len(results) == 0

    def test_search_respects_limit(self, federated_clst, team_namespace, owner_scope):
        """Test search respects result limit."""
        # Store multiple memories
        for i in range(5):
            federated_clst.store(
                memory_id=f"limit_mem_{i}",
                content=f"Content {i}",
                vector=[0.1],
                namespace=team_namespace,
                owner_agent_id="agent_1",
                owner_scope=owner_scope,
            )

        query = NamespacedQuery(
            query_vector=[0.1],
            namespaces=[team_namespace],
            requester=owner_scope,
            limit=2,
        )

        results = federated_clst.search(query)

        assert len(results) <= 2


# =============================================================================
# FederatedCLST Get Tests
# =============================================================================


class TestFederatedCLSTGet:
    """Tests for FederatedCLST.get method."""

    def test_get_own_memory(self, federated_clst, team_namespace, owner_scope):
        """Test getting own memory."""
        federated_clst.store(
            memory_id="get_test",
            content="Get this",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        memory = federated_clst.get("get_test", owner_scope)

        assert memory is not None
        assert memory.content == "Get this"

    def test_get_nonexistent(self, federated_clst, owner_scope):
        """Test getting nonexistent memory."""
        memory = federated_clst.get("nonexistent", owner_scope)

        assert memory is None

    def test_get_inaccessible(self, federated_clst, team_namespace, owner_scope, other_scope):
        """Test getting inaccessible memory."""
        federated_clst.store(
            memory_id="private_mem",
            content="Private",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.PRIVATE,
        )

        memory = federated_clst.get("private_mem", other_scope)

        assert memory is None


# =============================================================================
# FederatedCLST Reinforcement Tests
# =============================================================================


class TestFederatedCLSTReinforcement:
    """Tests for FederatedCLST.apply_reinforcement method."""

    def test_apply_reinforcement(self, federated_clst, team_namespace, owner_scope):
        """Test applying reinforcement signal."""
        federated_clst.store(
            memory_id="reinforce_test",
            content="Reinforce me",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        result = federated_clst.apply_reinforcement(
            memory_id="reinforce_test",
            agent_id="agent_1",
            signal=0.8,
            requester=owner_scope,
        )

        assert result == 0.8

    def test_apply_multiple_reinforcements(self, federated_clst, team_namespace, owner_scope):
        """Test applying multiple reinforcement signals."""
        federated_clst.store(
            memory_id="multi_reinforce",
            content="Multiple signals",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.TEAM,
        )

        # First agent reinforces
        federated_clst.apply_reinforcement(
            memory_id="multi_reinforce",
            agent_id="agent_1",
            signal=0.6,
            requester=owner_scope,
        )

        # Second agent reinforces (from same scope for simplicity)
        result = federated_clst.apply_reinforcement(
            memory_id="multi_reinforce",
            agent_id="agent_2",
            signal=0.4,
            requester=owner_scope,
        )

        # Average of 0.6 and 0.4 = 0.5
        assert result == 0.5

    def test_apply_reinforcement_nonexistent(self, federated_clst, owner_scope):
        """Test applying reinforcement to nonexistent memory."""
        result = federated_clst.apply_reinforcement(
            memory_id="nonexistent",
            agent_id="agent_1",
            signal=0.5,
            requester=owner_scope,
        )

        assert result is None


# =============================================================================
# FederatedCLST Update Access Tests
# =============================================================================


class TestFederatedCLSTUpdateAccess:
    """Tests for FederatedCLST.update_access method."""

    def test_update_access_policy(self, federated_clst, team_namespace, owner_scope):
        """Test updating access policy."""
        federated_clst.store(
            memory_id="access_test",
            content="Change access",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
            access_level=AccessLevel.PRIVATE,
        )

        new_policy = AccessPolicy(access_level=AccessLevel.TEAM)
        result = federated_clst.update_access(
            memory_id="access_test",
            new_policy=new_policy,
            requester=owner_scope,
        )

        assert result is True

    def test_update_access_nonexistent(self, federated_clst, owner_scope):
        """Test updating access for nonexistent memory."""
        new_policy = AccessPolicy(access_level=AccessLevel.PUBLIC)
        result = federated_clst.update_access(
            memory_id="nonexistent",
            new_policy=new_policy,
            requester=owner_scope,
        )

        assert result is False


# =============================================================================
# FederatedCLST Delete Tests
# =============================================================================


class TestFederatedCLSTDelete:
    """Tests for FederatedCLST.delete method."""

    def test_delete_own_memory(self, federated_clst, team_namespace, owner_scope):
        """Test deleting own memory."""
        federated_clst.store(
            memory_id="delete_test",
            content="Delete me",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        result = federated_clst.delete("delete_test", owner_scope)

        assert result is True
        assert federated_clst.get("delete_test", owner_scope) is None

    def test_delete_nonexistent(self, federated_clst, owner_scope):
        """Test deleting nonexistent memory."""
        result = federated_clst.delete("nonexistent", owner_scope)

        assert result is False

    def test_delete_removes_from_acl_cache(self, federated_clst, team_namespace, owner_scope):
        """Test that delete removes from ACL cache."""
        federated_clst.store(
            memory_id="cache_test",
            content="Cached",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        assert "cache_test" in federated_clst.acl_store

        federated_clst.delete("cache_test", owner_scope)

        assert "cache_test" not in federated_clst.acl_store


# =============================================================================
# FederatedCLST Stats Tests
# =============================================================================


class TestFederatedCLSTStats:
    """Tests for FederatedCLST.get_namespace_stats method."""

    def test_get_namespace_stats(self, federated_clst, team_namespace, owner_scope):
        """Test getting namespace statistics."""
        stats = federated_clst.get_namespace_stats(team_namespace, owner_scope)

        assert "namespace" in stats
        assert "total_memories" in stats
        assert "by_access_level" in stats
        assert "avg_reinforcement" in stats


# =============================================================================
# ACL Helper Tests
# =============================================================================


class TestACLHelper:
    """Tests for _get_acl helper method."""

    def test_get_acl_from_cache(self, federated_clst, team_namespace, owner_scope):
        """Test getting ACL from cache."""
        federated_clst.store(
            memory_id="acl_cache_test",
            content="Test",
            vector=[0.1],
            namespace=team_namespace,
            owner_agent_id="agent_1",
            owner_scope=owner_scope,
        )

        # ACL should be cached
        acl = federated_clst._get_acl(
            "acl_cache_test",
            {"acl": federated_clst.acl_store["acl_cache_test"].to_dict()},
        )

        assert acl.memory_id == "acl_cache_test"

    def test_get_acl_reconstruct(self, federated_clst):
        """Test reconstructing ACL from storage data."""
        now = datetime.now(timezone.utc)
        result = {
            "acl": {
                "memory_id": "reconstructed",
                "owner_agent_id": "agent_x",
                "owner_scope": {"org_id": "acme-corp"},
                "policy": {"access_level": 0},  # PRIVATE = 0
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
        }

        acl = federated_clst._get_acl("reconstructed", result)

        assert acl.memory_id == "reconstructed"
        assert acl.owner_agent_id == "agent_x"

    def test_get_acl_fallback(self, federated_clst):
        """Test fallback ACL creation when no ACL data."""
        result = {"owner_agent_id": "unknown_agent"}

        acl = federated_clst._get_acl("fallback_test", result)

        assert acl.memory_id == "fallback_test"
        assert acl.policy.access_level == AccessLevel.PRIVATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
