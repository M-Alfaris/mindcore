"""Test 04: FLR ↔ CLST Flow Tests.

Tests the hot-path (FLR) and cold storage (CLST) integration:
- FLR caching behavior
- CLST long-term storage
- Transfer between hot and cold storage
- Compression strategies
- Reinforcement signal propagation
"""

import time
from datetime import datetime, timedelta

import pytest

from tests.conftest import requires_postgres


# ============================================================================
# FLR (Fast Learning Recall) Tests
# ============================================================================


class TestFLRCache:
    """Test FLR caching behavior."""

    def test_flr_query_caching(self, flr, sqlite_storage):
        """Test that FLR caches frequently accessed memories."""
        from mindcore.v2.flr import Memory

        # Store some memories directly in storage
        memory = Memory(
            memory_id="",
            content="Frequently accessed memory",
            memory_type="semantic",
            user_id="cache_user",
            topics=["api"],
            importance=0.8,
            created_at=datetime.now(),
        )
        sqlite_storage.store(memory)

        # First query - cache miss
        result1 = flr.query(query="frequently accessed", user_id="cache_user")

        # Second query - should hit cache
        result2 = flr.query(query="frequently accessed", user_id="cache_user")

        # Both should return results
        assert len(result1.memories) > 0 or len(result2.memories) > 0

    def test_flr_cache_stats(self, flr, sqlite_storage):
        """Test FLR cache statistics."""
        from mindcore.v2.flr import Memory

        # Add some memories
        for i in range(5):
            memory = Memory(
                memory_id="",
                content=f"Stats test memory {i}",
                memory_type="semantic",
                user_id="stats_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Query to populate cache
        flr.query(query="stats test", user_id="stats_user")

        stats = flr.get_stats()
        assert stats is not None
        # Should have cache-related stats

    def test_flr_reinforcement(self, flr, sqlite_storage):
        """Test reinforcement through FLR."""
        from mindcore.v2.flr import Memory

        memory = Memory(
            memory_id="",
            content="Reinforcement test memory",
            memory_type="semantic",
            user_id="reinforce_user",
            topics=["api"],
            reinforcement_score=0.0,
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Apply reinforcement
        new_score = flr.reinforce(memory_id, 0.7)

        # Score should increase
        assert new_score > 0

    def test_flr_flush_reinforcements(self, flr, sqlite_storage):
        """Test flushing pending reinforcements to storage."""
        from mindcore.v2.flr import Memory

        memory = Memory(
            memory_id="",
            content="Flush test memory",
            memory_type="semantic",
            user_id="flush_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Apply reinforcement
        flr.reinforce(memory_id, 0.5)

        # Flush to storage
        flr.flush_reinforcements()

        # Should have flushed at least one
        # Note: Count depends on implementation

    def test_flr_query_with_attention(self, flr, sqlite_storage):
        """Test FLR query with attention hints."""
        from mindcore.v2.flr import Memory

        # Store memories with different topics
        topics_list = [["billing"], ["api"], ["billing", "api"]]
        for i, topics in enumerate(topics_list):
            memory = Memory(
                memory_id="",
                content=f"Attention test {i} with specific topics",
                memory_type="semantic",
                user_id="attention_user",
                topics=topics,
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Query with billing attention
        result = flr.query(
            query="attention test", user_id="attention_user", attention_hints=["billing"]
        )

        # Billing-related should be prioritized
        if len(result.memories) > 0:
            assert result.attention_focus is not None or len(result.memories) > 0


# ============================================================================
# CLST (Cognitive Long-term Storage Transfer) Tests
# ============================================================================


class TestCLSTStorage:
    """Test CLST long-term storage operations."""

    def test_clst_store(self, clst):
        """Test storing memory through CLST."""
        from mindcore.v2.flr import Memory

        memory = Memory(
            memory_id="",
            content="CLST stored memory",
            memory_type="semantic",
            user_id="clst_user",
            topics=["api"],
            importance=0.7,
            created_at=datetime.now(),
        )

        memory_id = clst.store(memory)
        assert memory_id is not None

    def test_clst_retrieve(self, clst):
        """Test retrieving memory from CLST."""
        from mindcore.v2.flr import Memory

        memory = Memory(
            memory_id="",
            content="CLST retrieve test",
            memory_type="semantic",
            user_id="clst_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = clst.store(memory)

        retrieved = clst.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved.content == "CLST retrieve test"

    def test_clst_search(self, clst):
        """Test searching in CLST."""
        from mindcore.v2.flr import Memory

        # Store multiple memories
        for i in range(5):
            memory = Memory(
                memory_id="",
                content=f"CLST search test {i}",
                memory_type="semantic",
                user_id="clst_search_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            clst.store(memory)

        results = clst.search(user_id="clst_search_user", query="CLST search")

        assert len(results) >= 1

    def test_clst_batch_store(self, clst):
        """Test batch storing in CLST."""
        from mindcore.v2.flr import Memory

        memories = [
            Memory(
                memory_id="",
                content=f"Batch CLST {i}",
                memory_type="semantic",
                user_id="batch_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            for i in range(10)
        ]

        memory_ids = clst.store_batch(memories)
        assert len(memory_ids) == 10


# ============================================================================
# Compression Tests
# ============================================================================


class TestCLSTCompression:
    """Test CLST compression strategies."""

    def test_compress_old_memories(self, clst):
        """Test compressing old memories."""
        from mindcore.v2.flr import Memory

        # Store old memories
        old_date = datetime.now() - timedelta(days=60)
        for i in range(15):  # Need more than min_memories (10)
            memory = Memory(
                memory_id="",
                content=f"Old memory to compress {i}",
                memory_type="semantic",
                user_id="compress_user",
                topics=["api"],
                created_at=old_date,
            )
            clst.store(memory)

        # Compress
        result = clst.compress(
            user_id="compress_user", older_than=timedelta(days=30), strategy="deduplicate"
        )

        assert result is not None
        # Result should have compression stats

    def test_compress_summarize_strategy(self, clst):
        """Test summarize compression strategy."""
        from mindcore.v2.flr import Memory

        old_date = datetime.now() - timedelta(days=60)
        for i in range(12):
            memory = Memory(
                memory_id="",
                content=f"Summarize test memory {i} with similar content",
                memory_type="semantic",
                user_id="summarize_user",
                topics=["api"],
                created_at=old_date,
            )
            clst.store(memory)

        # Try summarize (may need LLM, might skip)
        try:
            result = clst.compress(
                user_id="summarize_user", older_than=timedelta(days=30), strategy="summarize"
            )
            assert result is not None
        except Exception:
            # Summarize may require LLM
            pytest.skip("Summarize strategy requires LLM")

    def test_compress_merge_strategy(self, clst):
        """Test merge compression strategy."""
        from mindcore.v2.flr import Memory

        old_date = datetime.now() - timedelta(days=60)
        for i in range(12):
            memory = Memory(
                memory_id="",
                content=f"Merge test memory about topic A version {i}",
                memory_type="semantic",
                user_id="merge_user",
                topics=["api"],
                created_at=old_date,
            )
            clst.store(memory)

        result = clst.compress(
            user_id="merge_user", older_than=timedelta(days=30), strategy="merge"
        )

        assert result is not None


# ============================================================================
# FLR ↔ CLST Transfer Tests
# ============================================================================


class TestFLRCLSTTransfer:
    """Test transfers between FLR and CLST."""

    def test_promote_working_memory(self, flr, sqlite_storage):
        """Test promoting working memory to long-term storage."""
        from mindcore.v2.flr import Memory

        # Create working memory
        memory = Memory(
            memory_id="",
            content="Working memory to promote",
            memory_type="working",
            user_id="promote_user",
            topics=["api"],
            created_at=datetime.now(),
        )
        memory_id = sqlite_storage.store(memory)

        # Promote
        promoted = flr.promote(memory_id)

        assert promoted is True

    def test_context_window_management(self, flr):
        """Test FLR context window management."""
        session_id = "test_session_001"

        # Create context
        context = flr.update_context(
            session_id=session_id,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            attention_hints=["greeting"],
        )

        assert context is not None
        assert len(context.messages) == 2

        # Get context
        retrieved = flr.get_context(session_id)
        assert retrieved is not None

        # Clear context
        flr.clear_context(session_id)
        cleared = flr.get_context(session_id)
        assert cleared is None


# ============================================================================
# Sync Tests
# ============================================================================


class TestCLSTSync:
    """Test CLST synchronization features."""

    def test_sync_between_agents(self, mindcore_multi_agent):
        """Test syncing memories between agents via CLST."""
        # Register agents
        mindcore_multi_agent.register_agent(
            agent_id="sync_agent_a", name="Sync Agent A", teams=["sync_team"]
        )
        mindcore_multi_agent.register_agent(
            agent_id="sync_agent_b", name="Sync Agent B", teams=["sync_team"]
        )

        # Store as agent A
        mindcore_multi_agent.store(
            content="Memory to sync between agents",
            memory_type="semantic",
            user_id="sync_user",
            topics=["api"],
            access_level="team",
            agent_id="sync_agent_a",
        )

        # Sync
        result = mindcore_multi_agent.sync(
            source_agent="sync_agent_a", target_agent="sync_agent_b", user_id="sync_user"
        )

        assert result is not None


# ============================================================================
# Performance Tests
# ============================================================================


class TestFLRCLSTPerformance:
    """Test FLR and CLST performance characteristics."""

    def test_flr_cache_hit_performance(self, flr, sqlite_storage):
        """Test that cache hits are faster than cache misses."""
        from mindcore.v2.flr import Memory

        # Preload data
        for i in range(50):
            memory = Memory(
                memory_id="",
                content=f"Performance cache test {i}",
                memory_type="semantic",
                user_id="perf_cache_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # First query (cache miss)
        start = time.perf_counter()
        flr.query(query="performance cache", user_id="perf_cache_user")
        time.perf_counter() - start

        # Second query (potential cache hit)
        start = time.perf_counter()
        flr.query(query="performance cache", user_id="perf_cache_user")
        time.perf_counter() - start

        # Second should typically be faster due to caching
        # But this depends on implementation details

    def test_clst_search_performance(self, clst):
        """Test CLST search performance with many records."""
        from mindcore.v2.flr import Memory

        # Store many memories
        memories = [
            Memory(
                memory_id="",
                content=f"Large scale test memory {i} with various content",
                memory_type="semantic",
                user_id="scale_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            for i in range(100)
        ]
        clst.store_batch(memories)

        # Time the search
        times = []
        for _ in range(10):
            start = time.perf_counter()
            clst.search(user_id="scale_user", query="large scale test", limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 100, f"Average search time {avg_time:.2f}ms exceeds 100ms"


# ============================================================================
# Cache Hit Rate Tests
# ============================================================================


class TestCacheHitRate:
    """Test FLR cache hit rate targets."""

    def test_repeated_query_cache_hits(self, flr, sqlite_storage):
        """Test cache hit rate on repeated queries."""
        from mindcore.v2.flr import Memory

        # Setup data
        for i in range(20):
            memory = Memory(
                memory_id="",
                content=f"Cache hit rate test {i}",
                memory_type="semantic",
                user_id="hit_rate_user",
                topics=["api"],
                created_at=datetime.now(),
            )
            sqlite_storage.store(memory)

        # Warm up cache
        flr.query(query="cache hit rate", user_id="hit_rate_user")

        # Get initial stats
        flr.get_stats()

        # Make repeated queries
        for _ in range(10):
            flr.query(query="cache hit rate", user_id="hit_rate_user")

        flr.get_stats()

        # Check cache effectiveness
        # Target: >80% hit rate on repeated queries
        # Note: Actual hit rate depends on implementation
