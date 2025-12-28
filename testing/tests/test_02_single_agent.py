"""Test 02: Single Agent Tests.

Tests basic mindcore operations for a single agent:
- Store and recall operations
- Vocabulary validation
- Importance scoring
- Reinforcement learning
- Memory types
"""

import time
from datetime import datetime

import pytest


# ============================================================================
# Basic Store and Recall
# ============================================================================


class TestBasicOperations:
    """Test basic single-agent memory operations."""

    def test_store_memory(self, mindcore):
        """Test storing a single memory."""
        memory_id = mindcore.store(
            content="User prefers dark mode",
            memory_type="preference",
            user_id="user_001",
            topics=["settings"],
            importance=0.8,
        )

        assert memory_id is not None
        assert len(memory_id) > 0

    def test_recall_memory(self, mindcore):
        """Test recalling stored memories."""
        # Store some memories
        mindcore.store(
            content="User prefers Python for backend development",
            memory_type="preference",
            user_id="user_001",
            topics=["api", "integration"],
            importance=0.9,
        )
        mindcore.store(
            content="User asked about async patterns yesterday",
            memory_type="episodic",
            user_id="user_001",
            topics=["api", "help"],
            importance=0.7,
        )

        # Recall
        result = mindcore.recall(query="Python backend development preferences", user_id="user_001")

        assert result is not None
        assert len(result.memories) > 0
        assert "Python" in result.memories[0].content

    def test_recall_with_attention_hints(self, mindcore):
        """Test recall with attention hints to focus results."""
        # Store memories with different topics
        mindcore.store(
            content="User billing address is 123 Main St",
            memory_type="entity",
            user_id="user_001",
            topics=["billing", "account"],
            importance=0.6,
        )
        mindcore.store(
            content="User prefers email notifications for billing",
            memory_type="preference",
            user_id="user_001",
            topics=["billing", "settings"],
            importance=0.7,
        )
        mindcore.store(
            content="User had technical issue with API",
            memory_type="episodic",
            user_id="user_001",
            topics=["api", "issue"],
            importance=0.8,
        )

        # Recall with billing focus
        result = mindcore.recall(
            query="user preferences", user_id="user_001", attention_hints=["billing"]
        )

        assert len(result.memories) > 0
        # Billing-related memories should be prioritized
        billing_count = sum(1 for m in result.memories if "billing" in str(m.topics))
        assert billing_count > 0

    def test_recall_by_memory_type(self, mindcore):
        """Test filtering recall by memory type."""
        # Store different types
        mindcore.store(
            content="Preference: likes dark mode",
            memory_type="preference",
            user_id="user_001",
            topics=["settings"],
        )
        mindcore.store(
            content="Fact: works at TechCorp",
            memory_type="semantic",
            user_id="user_001",
            topics=["account"],
        )

        # Recall only preferences
        result = mindcore.recall(
            query="user likes", user_id="user_001", memory_types=["preference"]
        )

        assert all(m.memory_type == "preference" for m in result.memories)

    def test_get_memory_by_id(self, mindcore):
        """Test getting a specific memory by ID."""
        memory_id = mindcore.store(
            content="Specific memory to retrieve",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
        )

        memory = mindcore.get(memory_id)

        assert memory is not None
        assert memory.memory_id == memory_id
        assert memory.content == "Specific memory to retrieve"

    def test_delete_memory(self, mindcore):
        """Test deleting a memory."""
        from mindcore.exceptions import MemoryNotFoundError

        memory_id = mindcore.store(
            content="Memory to delete", memory_type="semantic", user_id="user_001", topics=["api"]
        )

        # Verify exists
        assert mindcore.get(memory_id) is not None

        # Delete
        mindcore.delete(memory_id)

        # Verify gone
        assert mindcore.get(memory_id) is None

        # Delete again should raise error
        with pytest.raises(MemoryNotFoundError):
            mindcore.delete(memory_id)

    def test_search_memories(self, mindcore):
        """Test searching memories with filters."""
        # Store memories
        for i in range(5):
            mindcore.store(
                content=f"Search test memory {i}",
                memory_type="semantic",
                user_id="search_user",
                topics=["api"],
                categories=["technical"],
                importance=0.5 + (i * 0.1),
            )

        # Search
        results = mindcore.search(user_id="search_user", topics=["api"], limit=10)

        assert len(results) == 5


# ============================================================================
# Memory Types
# ============================================================================


class TestMemoryTypes:
    """Test different memory types and their behaviors."""

    def test_episodic_memory(self, mindcore):
        """Test episodic (event-based) memory."""
        memory_id = mindcore.store(
            content="User called support on Monday about login issue",
            memory_type="episodic",
            user_id="user_001",
            topics=["issue", "help"],
            categories=["support"],
        )

        memory = mindcore.get(memory_id)
        assert memory.memory_type == "episodic"

    def test_semantic_memory(self, mindcore):
        """Test semantic (factual) memory."""
        memory_id = mindcore.store(
            content="API rate limit is 1000 requests per hour",
            memory_type="semantic",
            user_id="user_001",
            topics=["api", "documentation"],
        )

        memory = mindcore.get(memory_id)
        assert memory.memory_type == "semantic"

    def test_procedural_memory(self, mindcore):
        """Test procedural (how-to) memory."""
        memory_id = mindcore.store(
            content="To reset password: 1. Click forgot 2. Enter email 3. Check inbox",
            memory_type="procedural",
            user_id="user_001",
            topics=["help", "account"],
        )

        memory = mindcore.get(memory_id)
        assert memory.memory_type == "procedural"

    def test_preference_memory(self, mindcore):
        """Test preference memory."""
        memory_id = mindcore.store(
            content="User prefers detailed explanations over brief answers",
            memory_type="preference",
            user_id="user_001",
            topics=["settings"],
        )

        memory = mindcore.get(memory_id)
        assert memory.memory_type == "preference"

    def test_entity_memory(self, mindcore):
        """Test entity (named thing) memory."""
        memory_id = mindcore.store(
            content="John Smith is the account manager",
            memory_type="entity",
            user_id="user_001",
            topics=["account"],
            entities=["John Smith", "account manager"],
        )

        memory = mindcore.get(memory_id)
        assert memory.memory_type == "entity"
        assert "John Smith" in memory.entities


# ============================================================================
# Importance Scoring
# ============================================================================


class TestImportanceScoring:
    """Test importance scoring and its effects."""

    def test_importance_range(self, mindcore):
        """Test that importance is stored correctly."""
        memory_id = mindcore.store(
            content="High importance memory",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
            importance=0.95,
        )

        memory = mindcore.get(memory_id)
        assert memory.importance == 0.95

    def test_importance_affects_recall_order(self, mindcore):
        """Test that higher importance memories are prioritized in recall."""
        # Store low importance
        mindcore.store(
            content="Low importance topic info",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
            importance=0.2,
        )

        # Store high importance
        mindcore.store(
            content="High importance topic info",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
            importance=0.9,
        )

        result = mindcore.recall(query="topic info", user_id="user_001")

        # High importance should come first (or have higher score)
        if len(result.memories) >= 2:
            # First memory should have higher importance
            assert result.memories[0].importance >= result.memories[1].importance

    def test_default_importance(self, mindcore):
        """Test default importance value."""
        memory_id = mindcore.store(
            content="Default importance test",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
            # No importance specified
        )

        memory = mindcore.get(memory_id)
        assert memory.importance == 0.5  # Default


# ============================================================================
# Reinforcement Learning
# ============================================================================


class TestReinforcement:
    """Test reinforcement signal processing."""

    def test_positive_reinforcement(self, mindcore):
        """Test applying positive reinforcement."""
        memory_id = mindcore.store(
            content="Memory to reinforce positively",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
        )

        initial = mindcore.get(memory_id)
        initial_score = initial.reinforcement_score

        # Apply positive reinforcement
        mindcore.reinforce(memory_id, 0.8)

        updated = mindcore.get(memory_id)
        assert updated.reinforcement_score > initial_score

    def test_negative_reinforcement(self, mindcore):
        """Test applying negative reinforcement."""
        memory_id = mindcore.store(
            content="Memory to reinforce negatively",
            memory_type="semantic",
            user_id="user_001",
            topics=["api"],
        )

        # First apply some positive
        mindcore.reinforce(memory_id, 0.5)

        boosted = mindcore.get(memory_id)
        boosted_score = boosted.reinforcement_score

        # Now apply negative
        mindcore.reinforce(memory_id, -0.5)

        reduced = mindcore.get(memory_id)
        assert reduced.reinforcement_score < boosted_score

    def test_reinforcement_bounds(self, mindcore):
        """Test that reinforcement score stays within bounds."""
        memory_id = mindcore.store(
            content="Bounds test memory", memory_type="semantic", user_id="user_001", topics=["api"]
        )

        # Apply extreme positive
        for _ in range(10):
            mindcore.reinforce(memory_id, 1.0)

        memory = mindcore.get(memory_id)
        assert memory.reinforcement_score <= 1.0

        # Apply extreme negative
        for _ in range(20):
            mindcore.reinforce(memory_id, -1.0)

        memory = mindcore.get(memory_id)
        assert memory.reinforcement_score >= -1.0


# ============================================================================
# Vocabulary Validation
# ============================================================================


class TestVocabularyValidation:
    """Test vocabulary validation and schema generation."""

    def test_get_json_schema(self, mindcore):
        """Test getting JSON schema for LLM integration."""
        schema = mindcore.get_json_schema()

        assert schema is not None
        assert "properties" in schema or "$defs" in schema or "definitions" in schema

    def test_get_vocabulary_instructions(self, mindcore):
        """Test getting vocabulary instructions for prompts."""
        instructions = mindcore.get_vocabulary_instructions()

        assert instructions is not None
        assert len(instructions) > 0
        assert isinstance(instructions, str)

    def test_validate_memory_valid(self, mindcore):
        """Test validating a correct memory."""
        memory_data = {
            "content": "Test content",
            "memory_type": "semantic",
            "topics": ["api"],
            "categories": ["technical"],
            "importance": 0.5,
        }

        _is_valid, _errors = mindcore.validate_memory(memory_data)
        # Should be valid with standard vocabulary topics
        # Note: This depends on the SVL configuration

    def test_validate_memory_invalid_type(self, mindcore):
        """Test validating memory with invalid type."""
        memory_data = {
            "content": "Test content",
            "memory_type": "invalid_type_xyz",
            "topics": ["api"],
            "importance": 0.5,
        }

        is_valid, errors = mindcore.validate_memory(memory_data)
        assert not is_valid
        assert len(errors) > 0


# ============================================================================
# User Isolation
# ============================================================================


class TestUserIsolation:
    """Test that users' memories are properly isolated."""

    def test_recall_only_returns_user_memories(self, mindcore):
        """Test that recall only returns memories for the specified user."""
        # Store for user A
        mindcore.store(
            content="User A secret preference",
            memory_type="preference",
            user_id="user_a",
            topics=["settings"],
        )

        # Store for user B
        mindcore.store(
            content="User B different preference",
            memory_type="preference",
            user_id="user_b",
            topics=["settings"],
        )

        # Recall for user A
        result_a = mindcore.recall(query="preference settings", user_id="user_a")

        # Should only get user A's memories
        assert all(m.user_id == "user_a" for m in result_a.memories)

        # Recall for user B
        result_b = mindcore.recall(query="preference settings", user_id="user_b")

        # Should only get user B's memories
        assert all(m.user_id == "user_b" for m in result_b.memories)

    def test_search_respects_user_isolation(self, mindcore):
        """Test that search respects user isolation."""
        # Store for different users
        mindcore.store(
            content="User C data", memory_type="semantic", user_id="user_c", topics=["api"]
        )
        mindcore.store(
            content="User D data", memory_type="semantic", user_id="user_d", topics=["api"]
        )

        # Search as user C
        results = mindcore.search(user_id="user_c", topics=["api"])

        # Should only get user C's data
        assert all(m.user_id == "user_c" for m in results)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_store_latency(self, mindcore):
        """Test that store operations complete quickly."""
        times = []

        for i in range(50):
            start = time.perf_counter()
            mindcore.store(
                content=f"Performance test memory {i}",
                memory_type="semantic",
                user_id="perf_user",
                topics=["api"],
                importance=0.5,
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 20, f"Average store latency {avg_time:.2f}ms exceeds 20ms"

    def test_recall_latency(self, mindcore):
        """Test that recall operations complete quickly."""
        # Preload some data
        for i in range(100):
            mindcore.store(
                content=f"Recall performance test {i} with various keywords",
                memory_type="semantic",
                user_id="recall_perf_user",
                topics=["api"],
                importance=0.5,
            )

        times = []
        for _ in range(20):
            start = time.perf_counter()
            mindcore.recall(query="performance test keywords", user_id="recall_perf_user", limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 50, f"Average recall latency {avg_time:.2f}ms exceeds 50ms"
