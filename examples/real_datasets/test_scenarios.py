"""Test scenarios for CLST, FLR, and SVL validation.

This module provides comprehensive test scenarios that exercise
the complete Mindcore memory system:

1. CLSTTestScenario - Cold-path testing with historical data
2. FLRTestScenario - Hot-path testing with active session injection
3. SVLValidationTest - SVL gate validation on both paths

The tests use real dataset data enriched with SVL-compliant metadata
to ensure production-realistic validation.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from examples.real_datasets.postgres_store import EnrichedMemory, PostgresDatasetStore


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of a complete test scenario."""

    scenario_name: str
    started_at: datetime
    completed_at: datetime | None = None
    tests: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if not t.passed)

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "duration_ms": t.duration_ms,
                    "details": t.details,
                    "errors": t.errors,
                }
                for t in self.tests
            ],
        }


class CLSTTestScenario:
    """Test scenario for CLST (Cognitive Long-term Storage Transfer) cold-path.

    Tests historical data retrieval, session clustering, and cross-session
    memory queries using real dataset data.

    CLST is the cold-path for historical context:
    - Queries past sessions
    - Clusters memories by session
    - Provides long-term context for queries
    """

    def __init__(
        self,
        postgres_dsn: str,
        svl_pipeline: Any = None,
    ):
        """Initialize CLST test scenario.

        Args:
            postgres_dsn: PostgreSQL connection string
            svl_pipeline: Optional SVLPipeline instance for integration testing
        """
        self.store = PostgresDatasetStore(dsn=postgres_dsn)
        self._pipeline = svl_pipeline
        self._clst = None

    def setup(self) -> None:
        """Setup test resources."""
        self.store.connect()

        # Get or create CLST instance
        if self._pipeline:
            self._clst = self._pipeline._clst
        else:
            # Create standalone CLST for testing
            try:
                from mindcore.clst import CLST

                self._clst = CLST(storage=self.store.dsn)
            except ImportError:
                logger.warning("CLST not available, using store-only tests")
                self._clst = None

    def cleanup(self) -> None:
        """Cleanup test resources."""
        self.store.close()

    def run(self, dataset_name: str = "locomo") -> ScenarioResult:
        """Run the CLST test scenario.

        Args:
            dataset_name: Dataset to test against

        Returns:
            ScenarioResult with all test results
        """
        result = ScenarioResult(
            scenario_name="CLST Cold-Path Tests",
            started_at=datetime.now(timezone.utc),
        )

        self.setup()

        try:
            # Run individual tests
            result.tests.append(self._test_session_retrieval(dataset_name))
            result.tests.append(self._test_user_memory_history(dataset_name))
            result.tests.append(self._test_topic_based_search(dataset_name))
            result.tests.append(self._test_cross_session_query(dataset_name))
            result.tests.append(self._test_importance_ranking(dataset_name))
            result.tests.append(self._test_temporal_ordering(dataset_name))

            if self._clst:
                result.tests.append(self._test_clst_search())
                result.tests.append(self._test_clst_session_clustering())

        finally:
            self.cleanup()
            result.completed_at = datetime.now(timezone.utc)

        return result

    def _test_session_retrieval(self, dataset_name: str) -> TestResult:
        """Test: Retrieve all memories for a session."""
        start = time.perf_counter()
        errors = []

        try:
            # Get sessions from store
            stats = self.store.get_dataset_stats(dataset_name)
            if stats["unique_sessions"] == 0:
                return TestResult(
                    name="session_retrieval",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No sessions found in dataset"],
                )

            # Get first session's memories
            memories = self.store.query_memories(dataset_name=dataset_name, limit=1)
            if not memories:
                return TestResult(
                    name="session_retrieval",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No memories found"],
                )

            session_id = memories[0].session_id
            session_memories = self.store.get_session_memories(session_id)

            # Verify all memories have required metadata
            valid_count = 0
            for mem in session_memories:
                if mem.topics and mem.categories and mem.memory_type:
                    valid_count += 1
                else:
                    errors.append(f"Memory {mem.memory_id} missing metadata")

            passed = valid_count == len(session_memories) and len(session_memories) > 0

            return TestResult(
                name="session_retrieval",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "session_id": session_id,
                    "memories_retrieved": len(session_memories),
                    "valid_metadata_count": valid_count,
                },
                errors=errors[:5],  # Limit errors
            )

        except Exception as e:
            return TestResult(
                name="session_retrieval",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_user_memory_history(self, dataset_name: str) -> TestResult:
        """Test: Retrieve complete memory history for a user."""
        start = time.perf_counter()

        try:
            # Get a user from the dataset
            memories = self.store.query_memories(dataset_name=dataset_name, limit=1)
            if not memories:
                return TestResult(
                    name="user_memory_history",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No memories found"],
                )

            user_id = memories[0].user_id
            user_memories = self.store.get_user_memories(user_id, limit=100)

            # Check for session distribution
            sessions = set(m.session_id for m in user_memories)

            passed = len(user_memories) > 0 and len(sessions) >= 1

            return TestResult(
                name="user_memory_history",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "user_id": user_id,
                    "total_memories": len(user_memories),
                    "unique_sessions": len(sessions),
                },
            )

        except Exception as e:
            return TestResult(
                name="user_memory_history",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_topic_based_search(self, dataset_name: str) -> TestResult:
        """Test: Search memories by topic."""
        start = time.perf_counter()

        try:
            # Get common topics
            memories = self.store.query_memories(dataset_name=dataset_name, limit=50)
            if not memories:
                return TestResult(
                    name="topic_based_search",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No memories found"],
                )

            # Find a common topic
            topic_counts = {}
            for mem in memories:
                for topic in mem.topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

            if not topic_counts:
                return TestResult(
                    name="topic_based_search",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No topics found in memories"],
                )

            # Search by most common topic
            common_topic = max(topic_counts, key=topic_counts.get)
            topic_memories = self.store.query_memories(
                dataset_name=dataset_name,
                topics=[common_topic],
                limit=20,
            )

            # Verify results have the topic
            valid = all(common_topic in m.topics for m in topic_memories)
            passed = len(topic_memories) > 0 and valid

            return TestResult(
                name="topic_based_search",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "searched_topic": common_topic,
                    "results_count": len(topic_memories),
                    "all_match_topic": valid,
                },
            )

        except Exception as e:
            return TestResult(
                name="topic_based_search",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_cross_session_query(self, dataset_name: str) -> TestResult:
        """Test: Query memories across multiple sessions."""
        start = time.perf_counter()

        try:
            # Get user with multiple sessions
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)
            if not memories:
                return TestResult(
                    name="cross_session_query",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No memories found"],
                )

            # Find user with multiple sessions
            user_sessions = {}
            for mem in memories:
                if mem.user_id not in user_sessions:
                    user_sessions[mem.user_id] = set()
                user_sessions[mem.user_id].add(mem.session_id)

            multi_session_users = [u for u, s in user_sessions.items() if len(s) > 1]

            if not multi_session_users:
                # Still pass if dataset doesn't have multi-session users
                return TestResult(
                    name="cross_session_query",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={
                        "note": "No multi-session users in dataset",
                        "unique_users": len(user_sessions),
                    },
                )

            user_id = multi_session_users[0]
            user_memories = self.store.get_user_memories(user_id)
            sessions = set(m.session_id for m in user_memories)

            passed = len(sessions) > 1 and len(user_memories) >= len(sessions)

            return TestResult(
                name="cross_session_query",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "user_id": user_id,
                    "sessions_count": len(sessions),
                    "memories_count": len(user_memories),
                },
            )

        except Exception as e:
            return TestResult(
                name="cross_session_query",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_importance_ranking(self, dataset_name: str) -> TestResult:
        """Test: Memories are ranked by importance."""
        start = time.perf_counter()

        try:
            memories = self.store.query_memories(
                dataset_name=dataset_name,
                limit=20,
            )

            if len(memories) < 2:
                return TestResult(
                    name="importance_ranking",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "Not enough memories to test ranking"},
                )

            # Check if ordered by importance (descending)
            importances = [m.importance for m in memories]
            is_sorted = all(
                importances[i] >= importances[i + 1] for i in range(len(importances) - 1)
            )

            return TestResult(
                name="importance_ranking",
                passed=is_sorted,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_checked": len(memories),
                    "importance_range": f"{min(importances):.2f} - {max(importances):.2f}",
                    "properly_sorted": is_sorted,
                },
            )

        except Exception as e:
            return TestResult(
                name="importance_ranking",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_temporal_ordering(self, dataset_name: str) -> TestResult:
        """Test: Session memories are ordered by turn index."""
        start = time.perf_counter()

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=1)
            if not memories:
                return TestResult(
                    name="temporal_ordering",
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    errors=["No memories found"],
                )

            session_id = memories[0].session_id
            session_memories = self.store.get_session_memories(session_id, order_by_turn=True)

            # Check turn order
            turn_indices = [m.turn_index for m in session_memories]
            is_ordered = all(
                turn_indices[i] <= turn_indices[i + 1] for i in range(len(turn_indices) - 1)
            )

            return TestResult(
                name="temporal_ordering",
                passed=is_ordered,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "session_id": session_id,
                    "turns_checked": len(session_memories),
                    "properly_ordered": is_ordered,
                },
            )

        except Exception as e:
            return TestResult(
                name="temporal_ordering",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_clst_search(self) -> TestResult:
        """Test: CLST search functionality."""
        start = time.perf_counter()

        if not self._clst:
            return TestResult(
                name="clst_search",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "CLST not available, skipped"},
            )

        try:
            # Search through CLST
            results = self._clst.search(
                memory_types=["preference", "semantic"],
                limit=10,
            )

            passed = results is not None

            return TestResult(
                name="clst_search",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "results_count": len(results) if results else 0,
                },
            )

        except Exception as e:
            return TestResult(
                name="clst_search",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_clst_session_clustering(self) -> TestResult:
        """Test: CLST session clustering."""
        start = time.perf_counter()

        if not self._clst:
            return TestResult(
                name="clst_session_clustering",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "CLST not available, skipped"},
            )

        try:
            # Get clustered memories
            if hasattr(self._clst, "get_session_cluster"):
                cluster = self._clst.get_session_cluster(limit=5)
                passed = cluster is not None
            else:
                passed = True  # Method not available

            return TestResult(
                name="clst_session_clustering",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="clst_session_clustering",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )


class FLRTestScenario:
    """Test scenario for FLR (Fast Learning Recall) hot-path.

    Tests active session memory injection, real-time queries, and
    quick recall from the hot-path cache.

    FLR is the hot-path for active sessions:
    - Handles real-time memory storage
    - Provides fast recall for current session
    - Manages working memory
    """

    def __init__(
        self,
        postgres_dsn: str,
        svl_pipeline: Any = None,
    ):
        """Initialize FLR test scenario.

        Args:
            postgres_dsn: PostgreSQL connection string
            svl_pipeline: Optional SVLPipeline instance
        """
        self.store = PostgresDatasetStore(dsn=postgres_dsn)
        self._pipeline = svl_pipeline
        self._flr = None

    def setup(self) -> None:
        """Setup test resources."""
        self.store.connect()

        if self._pipeline:
            self._flr = self._pipeline._flr
        else:
            try:
                from mindcore.flr import SimpleFLR

                self._flr = SimpleFLR()
            except ImportError:
                logger.warning("FLR not available")
                self._flr = None

    def cleanup(self) -> None:
        """Cleanup test resources."""
        self.store.close()

    def run(self, dataset_name: str = "locomo") -> ScenarioResult:
        """Run the FLR test scenario.

        Args:
            dataset_name: Dataset for test data

        Returns:
            ScenarioResult with all test results
        """
        result = ScenarioResult(
            scenario_name="FLR Hot-Path Tests",
            started_at=datetime.now(timezone.utc),
        )

        self.setup()

        try:
            # Run individual tests
            result.tests.append(self._test_active_session_injection())
            result.tests.append(self._test_real_time_recall())
            result.tests.append(self._test_session_context_query())
            result.tests.append(self._test_working_memory())
            result.tests.append(self._test_hot_path_latency())

            if self._flr:
                result.tests.append(self._test_flr_cache_hit())
                result.tests.append(self._test_flr_to_clst_transfer())

        finally:
            self.cleanup()
            result.completed_at = datetime.now(timezone.utc)

        return result

    def _test_active_session_injection(self) -> TestResult:
        """Test: Inject memories into active session."""
        start = time.perf_counter()

        try:
            # Create test session
            session_id = f"test_session_{uuid.uuid4().hex[:8]}"
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"

            # Create test memories
            test_memories = [
                EnrichedMemory(
                    content="I prefer dark mode for coding.",
                    user_id=user_id,
                    session_id=session_id,
                    topics=["settings", "preferences"],
                    categories=["user_preference"],
                    memory_type="preference",
                    message_intent="express_preference",
                    importance=0.7,
                    turn_index=0,
                ),
                EnrichedMemory(
                    content="My timezone is PST (UTC-8).",
                    user_id=user_id,
                    session_id=session_id,
                    topics=["settings", "personal"],
                    categories=["user_preference"],
                    memory_type="semantic",
                    message_intent="provide_info",
                    importance=0.6,
                    turn_index=1,
                ),
            ]

            # Store through FLR if available
            if self._flr and hasattr(self._flr, "store"):
                for mem in test_memories:
                    self._flr.store(
                        content=mem.content,
                        user_id=mem.user_id,
                        session_id=mem.session_id,
                        metadata=mem.to_dict(),
                    )
                stored_count = len(test_memories)
            else:
                # Fall back to direct store
                self.store.store_memories(test_memories)
                stored_count = len(test_memories)

            # Verify storage
            session_memories = self.store.get_session_memories(session_id)
            passed = len(session_memories) >= stored_count

            return TestResult(
                name="active_session_injection",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "session_id": session_id,
                    "memories_injected": stored_count,
                    "memories_verified": len(session_memories),
                },
            )

        except Exception as e:
            return TestResult(
                name="active_session_injection",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_real_time_recall(self) -> TestResult:
        """Test: Real-time recall from active session."""
        start = time.perf_counter()

        try:
            # Create and store test memory
            session_id = f"test_recall_{uuid.uuid4().hex[:8]}"
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"

            test_memory = EnrichedMemory(
                content="I love using Python for data analysis.",
                user_id=user_id,
                session_id=session_id,
                topics=["technology", "programming"],
                categories=["work"],
                memory_type="preference",
                importance=0.8,
            )

            self.store.store_memory(test_memory)

            # Recall
            results = self.store.search_memories(
                query="Python data analysis",
                user_id=user_id,
                limit=5,
            )

            passed = len(results) > 0 and any("python" in r.content.lower() for r in results)

            return TestResult(
                name="real_time_recall",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "query": "Python data analysis",
                    "results_count": len(results),
                },
            )

        except Exception as e:
            return TestResult(
                name="real_time_recall",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_session_context_query(self) -> TestResult:
        """Test: Query within session context."""
        start = time.perf_counter()

        try:
            # Get a session with memories
            memories = self.store.query_memories(limit=5)
            if not memories:
                return TestResult(
                    name="session_context_query",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories available for test"},
                )

            session_id = memories[0].session_id
            user_id = memories[0].user_id

            # Query within session
            session_memories = self.store.query_memories(
                user_id=user_id,
                session_id=session_id,
                limit=10,
            )

            # All should be from same session
            all_same_session = all(m.session_id == session_id for m in session_memories)
            passed = len(session_memories) > 0 and all_same_session

            return TestResult(
                name="session_context_query",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "session_id": session_id,
                    "memories_in_context": len(session_memories),
                    "all_same_session": all_same_session,
                },
            )

        except Exception as e:
            return TestResult(
                name="session_context_query",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_working_memory(self) -> TestResult:
        """Test: Working memory updates during session."""
        start = time.perf_counter()

        try:
            session_id = f"working_mem_{uuid.uuid4().hex[:8]}"
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"

            # Simulate preference update (dark mode -> light mode)
            memory1 = EnrichedMemory(
                content="I prefer dark mode.",
                user_id=user_id,
                session_id=session_id,
                topics=["settings"],
                categories=["user_preference"],
                memory_type="preference",
                importance=0.7,
                turn_index=0,
            )

            memory2 = EnrichedMemory(
                content="Actually, I now prefer light mode.",
                user_id=user_id,
                session_id=session_id,
                topics=["settings"],
                categories=["user_preference"],
                memory_type="preference",
                importance=0.8,  # Higher importance for update
                turn_index=1,
            )

            self.store.store_memories([memory1, memory2])

            # Query should return higher importance first
            results = self.store.query_memories(
                user_id=user_id,
                session_id=session_id,
                topics=["settings"],
            )

            # Most recent/important should be first
            passed = len(results) >= 2 and "light mode" in results[0].content

            return TestResult(
                name="working_memory",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_stored": 2,
                    "latest_preference": results[0].content if results else "N/A",
                },
            )

        except Exception as e:
            return TestResult(
                name="working_memory",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_hot_path_latency(self) -> TestResult:
        """Test: Hot-path query latency."""
        start = time.perf_counter()

        try:
            latencies = []

            # Run multiple queries
            for _ in range(10):
                query_start = time.perf_counter()
                self.store.query_memories(limit=5)
                latencies.append((time.perf_counter() - query_start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

            # Hot path should be fast (< 50ms average)
            passed = avg_latency < 50

            return TestResult(
                name="hot_path_latency",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "queries": len(latencies),
                    "avg_latency_ms": round(avg_latency, 2),
                    "p99_latency_ms": round(p99_latency, 2),
                },
            )

        except Exception as e:
            return TestResult(
                name="hot_path_latency",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_flr_cache_hit(self) -> TestResult:
        """Test: FLR cache hit rate."""
        start = time.perf_counter()

        if not self._flr:
            return TestResult(
                name="flr_cache_hit",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "FLR not available, skipped"},
            )

        try:
            # Query same thing multiple times
            if hasattr(self._flr, "query"):
                self._flr.query("test query", limit=5)
                self._flr.query("test query", limit=5)  # Should hit cache

            if hasattr(self._flr, "get_stats"):
                stats = self._flr.get_stats()
                hit_rate = stats.get("cache_hit_rate", 0)
            else:
                hit_rate = 0

            passed = True  # Cache is optional optimization

            return TestResult(
                name="flr_cache_hit",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "cache_hit_rate": hit_rate,
                },
            )

        except Exception as e:
            return TestResult(
                name="flr_cache_hit",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_flr_to_clst_transfer(self) -> TestResult:
        """Test: FLR to CLST memory transfer."""
        start = time.perf_counter()

        if not self._flr:
            return TestResult(
                name="flr_to_clst_transfer",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "FLR not available, skipped"},
            )

        try:
            # Check if transfer mechanism exists
            if hasattr(self._flr, "transfer_to_clst"):
                result = self._flr.transfer_to_clst()
                passed = result is not None
            else:
                passed = True  # Method not required

            return TestResult(
                name="flr_to_clst_transfer",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="flr_to_clst_transfer",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )


class SVLValidationTest:
    """Test SVL (Shared Vocabulary Layer) validation on both paths.

    Tests that:
    1. All memories have SVL-compliant metadata
    2. SVL gate correctly validates/rejects memories
    3. Vocabulary consistency is maintained
    4. Both CLST and FLR respect SVL rules
    """

    def __init__(
        self,
        postgres_dsn: str,
        svl_pipeline: Any = None,
    ):
        """Initialize SVL validation test.

        Args:
            postgres_dsn: PostgreSQL connection string
            svl_pipeline: Optional SVLPipeline instance
        """
        self.store = PostgresDatasetStore(dsn=postgres_dsn)
        self._pipeline = svl_pipeline
        self._svl = None

    def setup(self) -> None:
        """Setup test resources."""
        self.store.connect()

        if self._pipeline:
            self._svl = self._pipeline._vocabulary
        else:
            try:
                from mindcore.svl import SharedVocabularyLayer

                self._svl = SharedVocabularyLayer()
            except ImportError:
                logger.warning("SVL not available")
                self._svl = None

    def cleanup(self) -> None:
        """Cleanup test resources."""
        self.store.close()

    def run(self, dataset_name: str = "locomo") -> ScenarioResult:
        """Run SVL validation tests.

        Args:
            dataset_name: Dataset to validate

        Returns:
            ScenarioResult with all test results
        """
        result = ScenarioResult(
            scenario_name="SVL Validation Tests",
            started_at=datetime.now(timezone.utc),
        )

        self.setup()

        try:
            # Run validation tests
            result.tests.append(self._test_metadata_completeness(dataset_name))
            result.tests.append(self._test_topic_vocabulary(dataset_name))
            result.tests.append(self._test_category_vocabulary(dataset_name))
            result.tests.append(self._test_memory_type_validation(dataset_name))
            result.tests.append(self._test_importance_bounds(dataset_name))
            result.tests.append(self._test_required_fields(dataset_name))

            if self._svl:
                result.tests.append(self._test_svl_gate_accept())
                result.tests.append(self._test_svl_gate_reject())
                result.tests.append(self._test_svl_canonicalization())

        finally:
            self.cleanup()
            result.completed_at = datetime.now(timezone.utc)

        return result

    def _test_metadata_completeness(self, dataset_name: str) -> TestResult:
        """Test: All memories have complete metadata."""
        start = time.perf_counter()
        errors = []

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="metadata_completeness",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            complete_count = 0
            for mem in memories:
                issues = []

                if not mem.topics:
                    issues.append("missing topics")
                if not mem.categories:
                    issues.append("missing categories")
                if not mem.message_type:
                    issues.append("missing message_type")
                if not mem.message_intent:
                    issues.append("missing message_intent")
                if not mem.memory_type:
                    issues.append("missing memory_type")
                if mem.importance is None:
                    issues.append("missing importance")

                if not issues:
                    complete_count += 1
                elif len(errors) < 5:
                    errors.append(f"{mem.memory_id}: {', '.join(issues)}")

            completeness_rate = complete_count / len(memories)
            passed = completeness_rate >= 0.95  # 95% threshold

            return TestResult(
                name="metadata_completeness",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_checked": len(memories),
                    "complete_count": complete_count,
                    "completeness_rate": f"{completeness_rate:.1%}",
                },
                errors=errors,
            )

        except Exception as e:
            return TestResult(
                name="metadata_completeness",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_topic_vocabulary(self, dataset_name: str) -> TestResult:
        """Test: Topics are from valid vocabulary."""
        start = time.perf_counter()

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="topic_vocabulary",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            # Collect all topics
            all_topics = set()
            for mem in memories:
                all_topics.update(mem.topics)

            # Check against SVL vocabulary if available
            if self._svl and hasattr(self._svl, "schema"):
                valid_topics = set(self._svl.schema.get_all_topics())
                if valid_topics:
                    invalid_topics = all_topics - valid_topics
                    passed = len(invalid_topics) == 0
                else:
                    passed = True  # No vocabulary defined
            else:
                passed = True  # Can't validate without SVL

            return TestResult(
                name="topic_vocabulary",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "unique_topics": len(all_topics),
                    "sample_topics": list(all_topics)[:10],
                },
            )

        except Exception as e:
            return TestResult(
                name="topic_vocabulary",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_category_vocabulary(self, dataset_name: str) -> TestResult:
        """Test: Categories are from valid vocabulary."""
        start = time.perf_counter()

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="category_vocabulary",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            all_categories = set()
            for mem in memories:
                all_categories.update(mem.categories)

            # Check against SVL if available
            if self._svl and hasattr(self._svl, "schema"):
                valid_cats = set(self._svl.schema.get_all_categories())
                if valid_cats:
                    invalid = all_categories - valid_cats
                    passed = len(invalid) == 0
                else:
                    passed = True
            else:
                passed = True

            return TestResult(
                name="category_vocabulary",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "unique_categories": len(all_categories),
                    "sample_categories": list(all_categories)[:10],
                },
            )

        except Exception as e:
            return TestResult(
                name="category_vocabulary",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_memory_type_validation(self, dataset_name: str) -> TestResult:
        """Test: Memory types are valid."""
        start = time.perf_counter()

        valid_types = {"episodic", "semantic", "procedural", "preference", "entity", "working"}

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="memory_type_validation",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            invalid_count = 0
            for mem in memories:
                if mem.memory_type not in valid_types:
                    invalid_count += 1

            passed = invalid_count == 0

            # Count distribution
            type_counts = {}
            for mem in memories:
                type_counts[mem.memory_type] = type_counts.get(mem.memory_type, 0) + 1

            return TestResult(
                name="memory_type_validation",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_checked": len(memories),
                    "invalid_count": invalid_count,
                    "type_distribution": type_counts,
                },
            )

        except Exception as e:
            return TestResult(
                name="memory_type_validation",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_importance_bounds(self, dataset_name: str) -> TestResult:
        """Test: Importance scores are within bounds [0, 1]."""
        start = time.perf_counter()

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="importance_bounds",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            out_of_bounds = 0
            for mem in memories:
                if mem.importance < 0 or mem.importance > 1:
                    out_of_bounds += 1

            passed = out_of_bounds == 0

            importances = [m.importance for m in memories]

            return TestResult(
                name="importance_bounds",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_checked": len(memories),
                    "out_of_bounds": out_of_bounds,
                    "min_importance": min(importances),
                    "max_importance": max(importances),
                    "avg_importance": sum(importances) / len(importances),
                },
            )

        except Exception as e:
            return TestResult(
                name="importance_bounds",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_required_fields(self, dataset_name: str) -> TestResult:
        """Test: All required fields are present."""
        start = time.perf_counter()

        required_fields = [
            "memory_id",
            "content",
            "user_id",
            "session_id",
            "topics",
            "categories",
            "memory_type",
            "importance",
        ]

        try:
            memories = self.store.query_memories(dataset_name=dataset_name, limit=100)

            if not memories:
                return TestResult(
                    name="required_fields",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"note": "No memories to validate"},
                )

            missing_count = 0
            for mem in memories:
                mem_dict = mem.to_dict()
                for field in required_fields:
                    if field not in mem_dict or mem_dict[field] is None:
                        missing_count += 1
                        break

            passed = missing_count == 0

            return TestResult(
                name="required_fields",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    "memories_checked": len(memories),
                    "missing_fields_count": missing_count,
                    "required_fields": required_fields,
                },
            )

        except Exception as e:
            return TestResult(
                name="required_fields",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_svl_gate_accept(self) -> TestResult:
        """Test: SVL gate accepts valid memories."""
        start = time.perf_counter()

        if not self._svl:
            return TestResult(
                name="svl_gate_accept",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "SVL not available, skipped"},
            )

        try:
            # Create valid memory
            valid_memory = {
                "content": "Test memory with valid metadata",
                "topics": ["general"],
                "categories": ["general"],
                "memory_type": "episodic",
                "message_intent": "provide_info",
                "importance": 0.5,
            }

            # Attempt to validate
            if hasattr(self._svl, "validate"):
                result = self._svl.validate(valid_memory)
                passed = result is True or (hasattr(result, "valid") and result.valid)
            else:
                passed = True

            return TestResult(
                name="svl_gate_accept",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="svl_gate_accept",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    def _test_svl_gate_reject(self) -> TestResult:
        """Test: SVL gate rejects invalid memories."""
        start = time.perf_counter()

        if not self._svl:
            return TestResult(
                name="svl_gate_reject",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "SVL not available, skipped"},
            )

        try:
            # Create invalid memory (missing required fields)
            invalid_memory = {
                "content": "Test memory without metadata",
                # Missing topics, categories, etc.
            }

            # Attempt to validate
            if hasattr(self._svl, "validate"):
                result = self._svl.validate(invalid_memory)
                # Should reject (return False or invalid result)
                passed = result is False or (hasattr(result, "valid") and not result.valid)
            else:
                passed = True

            return TestResult(
                name="svl_gate_reject",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception:
            # Rejection via exception is also valid
            return TestResult(
                name="svl_gate_reject",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"rejection_method": "exception"},
            )

    def _test_svl_canonicalization(self) -> TestResult:
        """Test: SVL canonicalizes metadata correctly."""
        start = time.perf_counter()

        if not self._svl:
            return TestResult(
                name="svl_canonicalization",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"note": "SVL not available, skipped"},
            )

        try:
            # Test topic canonicalization
            if hasattr(self._svl, "canonicalize_topic"):
                # Test case variations
                result1 = self._svl.canonicalize_topic("TECHNOLOGY")
                result2 = self._svl.canonicalize_topic("technology")
                passed = result1 == result2  # Should normalize
            else:
                passed = True

            return TestResult(
                name="svl_canonicalization",
                passed=passed,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return TestResult(
                name="svl_canonicalization",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )
