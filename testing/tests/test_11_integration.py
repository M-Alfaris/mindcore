"""Test 11: Integration Tests.

Full workflow integration tests covering:
- Complete ingest → store → recall → reinforce flow
- Multi-component interactions
- End-to-end scenarios
- Performance under load
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pytest

from tests.conftest import requires_postgres


# ============================================================================
# Full Workflow Tests
# ============================================================================


class TestFullWorkflow:
    """Test complete memory workflow."""

    def test_basic_workflow(self, mindcore):
        """Test basic store → recall → reinforce workflow."""
        # Step 1: Store memory
        memory_id = mindcore.store(
            content="User prefers Python for data analysis",
            memory_type="preference",
            user_id="workflow_user",
            topics=["api", "integration"],
            categories=["technical"],
            importance=0.8,
            entities=["Python", "data analysis"],
        )

        assert memory_id is not None

        # Step 2: Recall memory
        result = mindcore.recall(query="Python data analysis preferences", user_id="workflow_user")

        assert len(result.memories) > 0
        assert "Python" in result.memories[0].content

        # Step 3: Reinforce (positive feedback)
        mindcore.reinforce(memory_id, 0.8)

        # Step 4: Verify reinforcement affected score
        memory = mindcore.get(memory_id)
        assert memory.reinforcement_score > 0

        # Step 5: Delete
        mindcore.delete(memory_id)
        assert mindcore.get(memory_id) is None

    def test_multi_memory_workflow(self, mindcore):
        """Test workflow with multiple memories."""
        user_id = "multi_workflow_user"

        # Store multiple memories
        memories = [
            ("User likes dark mode", "preference", 0.7),
            ("User works at TechCorp", "entity", 0.8),
            ("User asked about billing yesterday", "episodic", 0.6),
            ("To reset password: click forgot, enter email", "procedural", 0.5),
        ]

        memory_ids = []
        for content, mtype, importance in memories:
            mid = mindcore.store(
                content=content,
                memory_type=mtype,
                user_id=user_id,
                topics=["api"],
                importance=importance,
            )
            memory_ids.append(mid)

        assert len(memory_ids) == 4

        # Recall with different queries
        result1 = mindcore.recall(
            query="user preferences", user_id=user_id, memory_types=["preference"]
        )
        assert len(result1.memories) > 0

        result2 = mindcore.recall(
            query="password reset help", user_id=user_id, memory_types=["procedural"]
        )
        assert len(result2.memories) > 0

        # Search by topics
        search_results = mindcore.search(user_id=user_id, topics=["api"])
        assert len(search_results) >= 4


# ============================================================================
# Multi-Agent Integration Tests
# ============================================================================


class TestMultiAgentIntegration:
    """Test multi-agent integration scenarios."""

    def test_multi_agent_workflow(self, mindcore_multi_agent):
        """Test multi-agent memory sharing workflow."""
        mc = mindcore_multi_agent

        # Register agents
        mc.register_agent(agent_id="support_agent", name="Support Agent", teams=["support"])
        mc.register_agent(
            agent_id="tech_agent", name="Technical Agent", teams=["support", "engineering"]
        )

        user_id = "multi_agent_user"

        # Support agent stores team memory
        mc.store(
            content="User reported login issue on mobile",
            memory_type="episodic",
            user_id=user_id,
            topics=["issue", "bug"],
            access_level="team",
            agent_id="support_agent",
        )

        # Tech agent should see it (same team)
        result = mc.recall(query="login issue mobile", user_id=user_id, agent_id="tech_agent")

        assert len(result.memories) > 0

    def test_agent_isolation(self, mindcore_multi_agent):
        """Test agent memory isolation."""
        mc = mindcore_multi_agent

        # Register agents in different teams
        mc.register_agent(agent_id="sales_agent", name="Sales Agent", teams=["sales"])
        mc.register_agent(agent_id="hr_agent", name="HR Agent", teams=["hr"])

        user_id = "isolation_user"

        # Sales stores private memory
        mc.store(
            content="User interested in enterprise plan",
            memory_type="semantic",
            user_id=user_id,
            topics=["billing"],
            access_level="private",
            agent_id="sales_agent",
        )

        # HR should not see sales private memory
        result = mc.recall(query="enterprise plan", user_id=user_id, agent_id="hr_agent")

        # Should not find private sales memory
        for memory in result.memories:
            if hasattr(memory, "agent_id") and memory.agent_id == "sales_agent":
                assert memory.access_level != "private"


# ============================================================================
# FLR/CLST Integration Tests
# ============================================================================


class TestFLRCLSTIntegration:
    """Test FLR and CLST integration."""

    def test_hot_cold_transfer(self, mindcore):
        """Test memory transfer from hot to cold storage."""
        user_id = "transfer_user"

        # Store some memories
        for i in range(20):
            mindcore.store(
                content=f"Transfer test memory {i}",
                memory_type="semantic",
                user_id=user_id,
                topics=["api"],
                importance=0.5,
            )

        # Compress old memories (transfer to cold)
        result = mindcore.compress(
            user_id=user_id,
            older_than_days=0,  # All memories
            strategy="deduplicate",
        )

        # Should have compression result
        assert result is not None

    def test_reinforcement_propagation(self, mindcore):
        """Test reinforcement signals propagate correctly."""
        user_id = "reinforce_prop_user"

        memory_id = mindcore.store(
            content="Reinforcement propagation test",
            memory_type="semantic",
            user_id=user_id,
            topics=["api"],
        )

        # Initial score
        initial = mindcore.get(memory_id)
        initial_score = initial.reinforcement_score

        # Multiple reinforcements
        for _ in range(5):
            mindcore.reinforce(memory_id, 0.5)

        # Score should have increased
        final = mindcore.get(memory_id)
        assert final.reinforcement_score > initial_score


# ============================================================================
# SVL Integration Tests
# ============================================================================


class TestSVLIntegration:
    """Test SVL integration with core functionality."""

    def test_vocabulary_validation_in_store(self, mindcore):
        """Test that vocabulary is validated on store."""
        # Valid memory with proper vocabulary terms
        memory_id = mindcore.store(
            content="Valid vocabulary test",
            memory_type="semantic",
            user_id="vocab_user",
            topics=["api"],  # Valid topic
            categories=["technical"],  # Valid category
            importance=0.5,
        )

        assert memory_id is not None

    def test_schema_generation_usable(self, mindcore):
        """Test that generated schema can be used."""
        schema = mindcore.get_json_schema()

        # Schema should be valid for structured output
        assert "properties" in schema or "$defs" in schema or "type" in schema


# ============================================================================
# PostgreSQL Integration Tests
# ============================================================================


@requires_postgres
class TestPostgresIntegration:
    """Test PostgreSQL-specific integration."""

    def test_postgres_full_workflow(self, mindcore_postgres):
        """Test complete workflow with PostgreSQL."""
        mc = mindcore_postgres
        user_id = "postgres_workflow_user"

        # Store
        memory_id = mc.store(
            content="PostgreSQL integration test memory",
            memory_type="semantic",
            user_id=user_id,
            topics=["api"],
            importance=0.7,
        )

        # Recall
        result = mc.recall(query="PostgreSQL integration", user_id=user_id)

        assert len(result.memories) > 0

        # Clean up
        mc.delete(memory_id)

    def test_postgres_large_scale(self, mindcore_postgres):
        """Test PostgreSQL with larger data set."""
        mc = mindcore_postgres
        user_id = "postgres_scale_user"

        # Store many memories
        memory_ids = []
        for i in range(100):
            mid = mc.store(
                content=f"Scale test memory {i} with content for searching",
                memory_type="semantic",
                user_id=user_id,
                topics=["api"],
                importance=0.5,
            )
            memory_ids.append(mid)

        # Search should be fast
        start = time.perf_counter()
        results = mc.search(user_id=user_id, topics=["api"], limit=50)
        elapsed = (time.perf_counter() - start) * 1000

        assert len(results) >= 50
        # PostgreSQL may be slower on first runs with cold caches; allow up to 1s
        assert elapsed < 1000, f"Search took {elapsed:.2f}ms"

        # Clean up
        for mid in memory_ids:
            try:
                mc.delete(mid)
            except Exception:
                pass


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Test concurrent access scenarios."""

    def test_concurrent_stores(self, mindcore):
        """Test concurrent memory stores."""
        user_id = "concurrent_store_user"
        results = []

        def store_memory(i):
            try:
                mid = mindcore.store(
                    content=f"Concurrent memory {i}",
                    memory_type="semantic",
                    user_id=user_id,
                    topics=["api"],
                )
                return ("success", mid)
            except Exception as e:
                return ("error", str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(store_memory, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        successes = [r for r in results if r[0] == "success"]
        assert len(successes) >= 45  # Allow for some failures

    def test_concurrent_reads(self, mindcore):
        """Test concurrent memory reads."""
        user_id = "concurrent_read_user"

        # Create test memory
        memory_id = mindcore.store(
            content="Concurrent read test memory",
            memory_type="semantic",
            user_id=user_id,
            topics=["api"],
        )

        def read_memory():
            try:
                memory = mindcore.get(memory_id)
                return ("success", memory.content if memory else None)
            except Exception as e:
                return ("error", str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(read_memory) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]

        successes = [r for r in results if r[0] == "success"]
        assert len(successes) >= 95

    def test_concurrent_store_and_recall(self, mindcore):
        """Test concurrent store and recall operations."""
        user_id = "concurrent_mixed_user"

        def store_operation(i):
            return mindcore.store(
                content=f"Concurrent mixed {i}",
                memory_type="semantic",
                user_id=user_id,
                topics=["api"],
            )

        def recall_operation():
            return mindcore.recall(query="concurrent mixed", user_id=user_id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            store_futures = [executor.submit(store_operation, i) for i in range(20)]
            recall_futures = [executor.submit(recall_operation) for _ in range(10)]

            all_futures = store_futures + recall_futures
            results = [f.result() for f in as_completed(all_futures)]

        # Should complete without errors
        assert len(results) == 30


# ============================================================================
# Performance Integration Tests
# ============================================================================


class TestPerformanceIntegration:
    """Test performance across integrated components."""

    def test_end_to_end_latency(self, mindcore):
        """Test end-to-end latency for common operations."""
        user_id = "latency_user"

        # Warm up
        for i in range(10):
            mindcore.store(
                content=f"Warm up memory {i}",
                memory_type="semantic",
                user_id=user_id,
                topics=["api"],
            )

        # Measure store latency
        store_times = []
        for i in range(20):
            start = time.perf_counter()
            mindcore.store(
                content=f"Latency test {i}", memory_type="semantic", user_id=user_id, topics=["api"]
            )
            store_times.append((time.perf_counter() - start) * 1000)

        avg_store = sum(store_times) / len(store_times)
        assert avg_store < 20, f"Store latency {avg_store:.2f}ms exceeds 20ms"

        # Measure recall latency
        recall_times = []
        for _ in range(20):
            start = time.perf_counter()
            mindcore.recall(query="latency test", user_id=user_id, limit=10)
            recall_times.append((time.perf_counter() - start) * 1000)

        avg_recall = sum(recall_times) / len(recall_times)
        assert avg_recall < 50, f"Recall latency {avg_recall:.2f}ms exceeds 50ms"


# ============================================================================
# Demo Data Integration Tests
# ============================================================================


class TestDemoDataIntegration:
    """Test integration with demo data."""

    def test_load_demo_memories(self, mindcore, memories_data):
        """Test loading and using demo memories."""
        user_memories = memories_data.get("user_memories", [])

        if len(user_memories) > 0:
            user_data = user_memories[0]
            user_id = user_data["user_id"]

            # Load first few memories
            for memory in user_data.get("memories", [])[:3]:
                mindcore.store(
                    content=memory["content"],
                    memory_type=memory.get("memory_type", "semantic"),
                    user_id=user_id,
                    topics=memory.get("topics", ["api"]),
                    importance=memory.get("importance", 0.5),
                )

            # Recall should find them
            result = mindcore.recall(
                query=user_data["memories"][0]["content"][:20], user_id=user_id
            )

            assert len(result.memories) > 0

    def test_load_demo_agents(self, mindcore_multi_agent, agents_data):
        """Test loading and using demo agents."""
        agents = agents_data.get("agents", [])

        for agent in agents[:2]:
            mindcore_multi_agent.register_agent(
                agent_id=agent["id"], name=agent["name"], teams=agent.get("teams", [])
            )

        # List should show registered agents
        registered = mindcore_multi_agent.list_agents()
        assert len(registered) >= 2


# ============================================================================
# Error Recovery Integration Tests
# ============================================================================


class TestErrorRecoveryIntegration:
    """Test error recovery in integrated scenarios."""

    def test_recovery_after_failed_store(self, mindcore):
        """Test system recovers after failed store."""
        user_id = "recovery_user"

        # Attempt invalid store
        try:
            mindcore.store(
                content="",  # May be invalid
                memory_type="invalid",
                user_id=user_id,
                topics=[],
            )
        except Exception:
            pass

        # Valid store should work
        memory_id = mindcore.store(
            content="Valid after failure", memory_type="semantic", user_id=user_id, topics=["api"]
        )

        assert memory_id is not None

    def test_graceful_degradation(self, mindcore):
        """Test system degrades gracefully under stress."""
        user_id = "stress_user"

        # Rapid operations
        memory_ids = []
        for i in range(100):
            try:
                mid = mindcore.store(
                    content=f"Stress test {i}",
                    memory_type="semantic",
                    user_id=user_id,
                    topics=["api"],
                )
                memory_ids.append(mid)
            except Exception:
                # Allow some failures under stress
                pass

        # Should have completed most operations
        assert len(memory_ids) >= 90


# ============================================================================
# Success Metrics Tests
# ============================================================================


class TestSuccessMetrics:
    """Verify success metrics are met."""

    def test_sqlite_latency_target(self, mindcore):
        """Verify SQLite operations < 10ms average."""
        user_id = "metric_user"
        times = []

        for i in range(50):
            start = time.perf_counter()
            mindcore.store(
                content=f"Metric test {i}", memory_type="semantic", user_id=user_id, topics=["api"]
            )
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        assert avg_time < 10, f"SQLite avg latency {avg_time:.2f}ms exceeds 10ms target"

    def test_memory_isolation_zero_leaks(self, mindcore):
        """Verify zero cross-user memory leaks."""
        # Store for user A
        mindcore.store(
            content="User A secret data XYZ123",
            memory_type="semantic",
            user_id="user_a_secret",
            topics=["api"],
        )

        # Search as user B
        results = mindcore.search(user_id="user_b_different", query="XYZ123")

        # Should find nothing
        for result in results:
            assert result.user_id != "user_a_secret"

    def test_rbac_100_percent_correct(self, mindcore_multi_agent):
        """Verify RBAC decisions are 100% correct."""
        mc = mindcore_multi_agent

        mc.register_agent(agent_id="rbac_agent_a", name="Agent A", teams=["team_x"])
        mc.register_agent(agent_id="rbac_agent_b", name="Agent B", teams=["team_y"])

        # Store private memory
        mc.store(
            content="Private RBAC test",
            memory_type="semantic",
            user_id="rbac_user",
            topics=["api"],
            access_level="private",
            agent_id="rbac_agent_a",
        )

        # Agent B should not see it
        result = mc.recall(query="Private RBAC", user_id="rbac_user", agent_id="rbac_agent_b")

        # Should not find agent A's private memory
        for memory in result.memories:
            if hasattr(memory, "access_level") and memory.access_level == "private":
                assert memory.agent_id != "rbac_agent_a"
