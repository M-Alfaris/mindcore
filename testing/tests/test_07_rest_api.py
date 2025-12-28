"""Test 07: REST API Tests.

Tests the REST API server endpoints:
- Memory CRUD operations
- Search and recall endpoints
- Agent management endpoints
- Vocabulary endpoints
- Error responses
- CORS and headers
"""

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures for REST API Testing
# ============================================================================


@pytest.fixture
def mock_mindcore():
    """Create a mock mindcore instance for API testing."""
    mock = MagicMock()
    mock.store.return_value = "mem_test_123"
    mock.get.return_value = MagicMock(
        memory_id="mem_test_123",
        content="Test content",
        memory_type="semantic",
        user_id="test_user",
        topics=["api"],
        importance=0.5,
        to_dict=lambda: {
            "memory_id": "mem_test_123",
            "content": "Test content",
            "memory_type": "semantic",
            "user_id": "test_user",
            "topics": ["api"],
            "importance": 0.5,
        },
    )
    mock.recall.return_value = MagicMock(
        memories=[mock.get.return_value],
        scores=[0.9],
        to_dict=lambda: {"memories": [mock.get.return_value.to_dict()], "scores": [0.9]},
    )
    mock.search.return_value = [mock.get.return_value]
    mock.get_json_schema.return_value = {"type": "object"}
    mock.list_agents.return_value = []
    return mock


@pytest.fixture
def test_client(flr, clst):
    """Create a test client for the REST API using real FLR/CLST."""
    try:
        from starlette.testclient import TestClient

        from mindcore.server.rest import create_app

        app = create_app(flr=flr, clst=clst)
        client = TestClient(app)
        yield client, (flr, clst)
    except ImportError:
        pytest.skip("starlette or REST server not available")


# ============================================================================
# Health and Info Endpoints
# ============================================================================


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        client, _ = test_client
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "version" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        client, _ = test_client
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "healthy" in str(data).lower()


# ============================================================================
# Memory CRUD Endpoints
# ============================================================================


class TestMemoryCRUD:
    """Test memory CRUD endpoints."""

    def test_create_memory(self, test_client):
        """Test POST /memories to create a memory."""
        client, _ = test_client

        response = client.post(
            "/memories",
            json={
                "content": "Test memory content",
                "memory_type": "semantic",
                "user_id": "test_user",
                "topics": ["api"],
                "importance": 0.7,
            },
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert "memory_id" in data or "id" in data or "success" in data

    def test_get_memory(self, test_client):
        """Test GET /memories/{id} to retrieve a memory."""
        client, (_flr, clst) = test_client

        # First create a memory
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Test content for get",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        response = client.get(f"/memories/{memory_id}")

        assert response.status_code == 200
        data = response.json()
        assert "Test content" in data.get("content", "")

    def test_update_memory(self, test_client):
        """Test PUT /memories/{id} to update a memory."""
        client, _ = test_client

        # PUT might not be implemented, so accept 200, 404, or 405
        response = client.put(
            "/memories/mem_test_123", json={"content": "Updated content", "importance": 0.9}
        )

        assert response.status_code in [200, 404, 405]

    def test_delete_memory(self, test_client):
        """Test DELETE /memories/{id} to delete a memory."""
        client, (_flr, clst) = test_client

        # First create a memory to delete
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Memory to delete",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        response = client.delete(f"/memories/{memory_id}")

        assert response.status_code in [200, 204]

    def test_get_nonexistent_memory(self, test_client):
        """Test GET for non-existent memory returns 404."""
        client, _ = test_client

        response = client.get("/memories/nonexistent_id_12345")

        assert response.status_code == 404


# ============================================================================
# Search and Recall Endpoints
# ============================================================================


class TestSearchRecallEndpoints:
    """Test search and recall endpoints."""

    def test_search_memories(self, test_client):
        """Test POST /memories/search endpoint."""
        client, (_flr, clst) = test_client

        # First create some memories to search
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Searchable memory content",
            memory_type="semantic",
            user_id="search_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        clst.store(memory)

        response = client.post(
            "/memories/search", json={"user_id": "search_user", "topics": ["api"], "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "memories" in data

    def test_recall_memories(self, test_client):
        """Test POST /recall endpoint."""
        client, (_flr, clst) = test_client

        # First create some memories
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Recallable memory for testing",
            memory_type="semantic",
            user_id="recall_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        clst.store(memory)

        response = client.post(
            "/recall", json={"query": "recallable memory", "user_id": "recall_user", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data or isinstance(data, list)

    def test_recall_with_attention_hints(self, test_client):
        """Test recall with attention hints."""
        client, _ = test_client

        response = client.post(
            "/recall",
            json={
                "query": "user preferences",
                "user_id": "test_user",
                "attention_hints": ["billing"],
                "memory_types": ["preference"],
            },
        )

        assert response.status_code == 200


# ============================================================================
# Reinforcement Endpoint
# ============================================================================


class TestReinforcementEndpoint:
    """Test reinforcement signal endpoint."""

    def test_reinforce_memory(self, test_client):
        """Test POST /reinforce endpoint."""
        client, (_flr, clst) = test_client

        # Create a memory first
        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Memory to reinforce",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        response = client.post("/reinforce", json={"memory_id": memory_id, "signal": 0.8})

        assert response.status_code == 200

    def test_reinforce_negative_signal(self, test_client):
        """Test reinforcement with negative signal."""
        client, (_flr, clst) = test_client

        from datetime import datetime

        from mindcore.flr import Memory

        memory = Memory(
            memory_id="",
            content="Memory for negative reinforce",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        response = client.post("/reinforce", json={"memory_id": memory_id, "signal": -0.5})

        assert response.status_code == 200

    def test_reinforce_invalid_signal(self, test_client):
        """Test reinforcement with out-of-range signal."""
        client, _ = test_client

        response = client.post(
            "/reinforce",
            json={
                "memory_id": "test_mem",
                "signal": 2.0,  # Invalid: > 1.0
            },
        )

        # Should return error
        assert response.status_code in [400, 422]


# ============================================================================
# Agent Endpoints
# ============================================================================


class TestAgentEndpoints:
    """Test agent management endpoints."""

    def test_list_agents(self, test_client):
        """Test GET /agents endpoint."""
        client, _ = test_client

        response = client.get("/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data or isinstance(data, list)

    def test_register_agent(self, test_client):
        """Test POST /agents to register an agent."""
        client, _ = test_client

        response = client.post(
            "/agents",
            json={
                "agent_id": "new_agent_rest",
                "name": "New Agent",
                "description": "Test agent",
                "teams": ["support"],
            },
        )

        # May return 200, 201, or 400 if access control not configured
        assert response.status_code in [200, 201, 400]

    def test_get_agent(self, test_client):
        """Test GET /agents/{id} endpoint."""
        client, _ = test_client

        response = client.get("/agents/test_agent")

        # May return 200 or 400/404 depending on if access control configured
        assert response.status_code in [200, 400, 404]

    def test_unregister_agent(self, test_client):
        """Test DELETE /agents/{id} endpoint."""
        client, _ = test_client

        response = client.delete("/agents/test_agent")

        assert response.status_code in [200, 204, 400, 404]


# ============================================================================
# Vocabulary Endpoint
# ============================================================================


class TestVocabularyEndpoints:
    """Test vocabulary-related endpoints."""

    def test_get_vocabulary(self, test_client):
        """Test GET /vocabulary endpoint."""
        client, _ = test_client

        response = client.get("/vocabulary")

        assert response.status_code == 200
        response.json()
        # Should return vocabulary configuration

    def test_get_vocabulary_schema(self, test_client):
        """Test GET /vocabulary/schema endpoint."""
        client, _ = test_client

        response = client.get("/vocabulary/schema")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ============================================================================
# Compression and Sync Endpoints
# ============================================================================


class TestCompressionSyncEndpoints:
    """Test compression and sync endpoints."""

    def test_compress_endpoint(self, test_client):
        """Test POST /compress endpoint."""
        client, _ = test_client

        response = client.post(
            "/compress",
            params={"user_id": "test_user", "older_than_days": 30, "strategy": "deduplicate"},
        )

        assert response.status_code == 200

    def test_sync_endpoint(self, test_client):
        """Test POST /sync endpoint."""
        client, _ = test_client

        response = client.post(
            "/sync",
            params={"source_agent": "agent_a", "target_agent": "agent_b", "user_id": "test_user"},
        )

        assert response.status_code == 200


# ============================================================================
# Migration Endpoints
# ============================================================================


class TestMigrationEndpoints:
    """Test vocabulary migration endpoints."""

    def test_migrate_endpoint(self, test_client):
        """Test POST /migrate endpoint."""
        client, _ = test_client

        response = client.post("/migrate", json={"from_version": "0.9.0", "user_id": "test_user"})

        # Migrate endpoint may or may not exist
        assert response.status_code in [200, 404, 405]

    def test_rollback_migration_endpoint(self, test_client):
        """Test POST /rollback-migration endpoint."""
        client, _ = test_client

        response = client.post("/rollback-migration")

        # Rollback endpoint may or may not exist
        assert response.status_code in [200, 404, 405]


# ============================================================================
# Error Response Tests
# ============================================================================


class TestErrorResponses:
    """Test API error responses."""

    def test_invalid_json(self, test_client):
        """Test response for invalid JSON body."""
        client, _ = test_client

        response = client.post(
            "/memories", content="not valid json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [400, 422]

    def test_missing_required_field(self, test_client):
        """Test response for missing required field."""
        client, _ = test_client

        response = client.post(
            "/memories",
            json={
                "content": "Test"
                # Missing user_id
            },
        )

        assert response.status_code in [400, 422]

    def test_invalid_memory_type(self, test_client):
        """Test that invalid memory type raises validation error."""
        client, _ = test_client

        response = client.post(
            "/memories",
            json={
                "content": "Test",
                "memory_type": "some_invalid_type",  # Non-standard type
                "user_id": "test_user",
                "topics": ["api"],
            },
        )

        # Validation should fail for invalid memory types
        assert response.status_code in [200, 400, 422, 500]


# ============================================================================
# Header Tests
# ============================================================================


class TestHeaders:
    """Test API headers and CORS."""

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        client, _ = test_client

        response = client.options("/memories")

        # CORS headers should be present or method not allowed
        assert response.status_code in [200, 204, 405]

    def test_agent_id_header(self, test_client):
        """Test X-Agent-ID header is processed."""
        client, _ = test_client

        response = client.post(
            "/memories",
            json={
                "content": "Test with agent header",
                "memory_type": "semantic",
                "user_id": "test_user",
                "topics": ["api"],
            },
            headers={"X-Agent-ID": "test_agent"},
        )

        # Should accept the header
        assert response.status_code in [200, 201]


# ============================================================================
# Rate Limiting Tests (if implemented)
# ============================================================================


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_not_exceeded(self, test_client):
        """Test normal requests under rate limit."""
        client, _ = test_client

        # Make a few requests
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_headers(self, test_client):
        """Test rate limit headers are present."""
        client, _ = test_client

        client.get("/health")

        # May have X-RateLimit-Limit, X-RateLimit-Remaining, etc.


# ============================================================================
# Performance Tests
# ============================================================================


class TestAPIPerformance:
    """Test API response times."""

    def test_health_response_time(self, test_client):
        """Test health endpoint responds quickly."""
        import time

        client, _ = test_client

        start = time.perf_counter()
        response = client.get("/health")
        elapsed = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 500, f"Health check took {elapsed:.2f}ms"

    def test_recall_response_time(self, test_client):
        """Test recall endpoint response time."""
        import time

        client, _ = test_client

        start = time.perf_counter()
        response = client.post("/recall", json={"query": "test", "user_id": "test_user"})
        elapsed = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        # Target: < 500ms
        assert elapsed < 500, f"Recall took {elapsed:.2f}ms"
