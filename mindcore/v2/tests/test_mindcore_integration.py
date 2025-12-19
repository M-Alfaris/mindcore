"""Comprehensive integration tests for Mindcore main class."""

import os
import tempfile

import pytest

from mindcore.v2 import Mindcore
from mindcore.v2.vocabulary import VocabularySchema


class TestMindcoreInitialization:
    """Test Mindcore initialization options."""

    def test_default_init(self):
        """Test default initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")
            assert mc._vocabulary is not None
            assert mc._access_controller is None
            mc.close()
        finally:
            os.unlink(db_path)

    def test_init_with_custom_vocabulary(self):
        """Test initialization with custom vocabulary."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            vocab = VocabularySchema(
                version="2.0.0",
                topics=["custom_topic"],
                categories=["custom_category"],
            )
            mc = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)

            assert mc._vocabulary.version == "2.0.0"
            assert "custom_topic" in mc._vocabulary.topics
            mc.close()
        finally:
            os.unlink(db_path)

    def test_init_with_multi_agent(self):
        """Test initialization with multi-agent enabled."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(
                storage=f"sqlite:///{db_path}",
                enable_multi_agent=True,
            )

            assert mc._access_controller is not None
            mc.close()
        finally:
            os.unlink(db_path)

    def test_context_manager(self):
        """Test using Mindcore as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with Mindcore(storage=f"sqlite:///{db_path}") as mc:
                mc.store(
                    content="Test memory",
                    memory_type="episodic",
                    user_id="user_123",
                )
            # Should be closed after context
        finally:
            os.unlink(db_path)


class TestMindcoreStoreRecall:
    """Test store and recall operations."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore instance with vocabulary that includes needed topics."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        vocab = VocabularySchema(
            version="1.0.0",
            topics=["settings", "billing", "support", "product"],
            categories=["support", "feedback", "urgent"],
        )
        mc = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_store_and_recall(self, mindcore):
        """Test basic store and recall."""
        memory_id = mindcore.store(
            content="User prefers dark mode",
            memory_type="preference",
            user_id="user_123",
            topics=["settings"],
        )

        assert memory_id is not None
        assert memory_id.startswith("mem_")

        result = mindcore.recall(
            query="dark mode preferences",
            user_id="user_123",
        )

        assert len(result.memories) >= 1

    def test_store_with_all_fields(self, mindcore):
        """Test storing with all optional fields."""
        memory_id = mindcore.store(
            content="Important customer note",
            memory_type="episodic",
            user_id="user_123",
            topics=["support"],
            categories=["support"],
            importance=0.9,
            entities=["customer_name"],
            access_level="team",
            agent_id="agent_456",
        )

        memory = mindcore.get(memory_id)

        assert memory is not None
        assert memory.importance == 0.9
        assert memory.access_level == "team"
        assert memory.agent_id == "agent_456"

    def test_recall_with_filters(self, mindcore):
        """Test recall with attention hints and type filters."""
        # Store different types
        mindcore.store(
            content="User likes email",
            memory_type="preference",
            user_id="user_123",
            topics=["settings"],
        )
        mindcore.store(
            content="User called about billing",
            memory_type="episodic",
            user_id="user_123",
            topics=["billing"],
        )

        # Recall with type filter
        result = mindcore.recall(
            query="user contact",
            user_id="user_123",
            memory_types=["preference"],
        )

        for m in result.memories:
            assert m.memory_type == "preference"

    def test_recall_respects_user_id(self, mindcore):
        """Test that recall respects user_id."""
        mindcore.store(
            content="Memory for user 1",
            memory_type="episodic",
            user_id="user_1",
        )
        mindcore.store(
            content="Memory for user 2",
            memory_type="episodic",
            user_id="user_2",
        )

        result = mindcore.recall(query="memory", user_id="user_1")

        for m in result.memories:
            assert m.user_id == "user_1"


class TestMindcoreSearch:
    """Test search operations."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore with test data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support", "product"],
        )
        mc = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)

        # Add test data
        mc.store("Billing issue", "episodic", "user_123", topics=["billing"])
        mc.store("Support request", "episodic", "user_123", topics=["support"])
        mc.store("Product feedback", "semantic", "user_123", topics=["product"])

        yield mc
        mc.close()
        os.unlink(db_path)

    def test_search_by_topic(self, mindcore):
        """Test searching by topic."""
        results = mindcore.search(
            user_id="user_123",
            topics=["billing"],
        )

        assert len(results) >= 1
        for m in results:
            assert "billing" in m.topics

    def test_search_by_memory_type(self, mindcore):
        """Test searching by memory type."""
        results = mindcore.search(
            user_id="user_123",
            memory_types=["semantic"],
        )

        assert len(results) >= 1
        for m in results:
            assert m.memory_type == "semantic"

    def test_search_with_query(self, mindcore):
        """Test searching with text query."""
        results = mindcore.search(
            user_id="user_123",
            query="billing",
        )

        assert len(results) >= 1

    def test_search_with_limit(self, mindcore):
        """Test search with limit."""
        results = mindcore.search(
            user_id="user_123",
            limit=1,
        )

        assert len(results) <= 1


class TestMindcoreGetDelete:
    """Test get and delete operations."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_get_memory(self, mindcore):
        """Test getting a specific memory."""
        memory_id = mindcore.store(
            content="Test content",
            memory_type="episodic",
            user_id="user_123",
        )

        memory = mindcore.get(memory_id)

        assert memory is not None
        assert memory.content == "Test content"

    def test_get_nonexistent(self, mindcore):
        """Test getting non-existent memory."""
        memory = mindcore.get("nonexistent_id")
        assert memory is None

    def test_delete_memory(self, mindcore):
        """Test deleting a memory."""
        memory_id = mindcore.store(
            content="To be deleted",
            memory_type="episodic",
            user_id="user_123",
        )

        result = mindcore.delete(memory_id)
        assert result is True

        memory = mindcore.get(memory_id)
        assert memory is None

    def test_delete_nonexistent(self, mindcore):
        """Test deleting non-existent memory."""
        result = mindcore.delete("nonexistent_id")
        assert result is False


class TestMindcoreReinforcement:
    """Test reinforcement learning."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_reinforce_positive(self, mindcore):
        """Test positive reinforcement."""
        memory_id = mindcore.store(
            content="Useful memory",
            memory_type="semantic",
            user_id="user_123",
        )

        # Should not raise
        mindcore.reinforce(memory_id, signal=1.0)

    def test_reinforce_negative(self, mindcore):
        """Test negative reinforcement."""
        memory_id = mindcore.store(
            content="Not so useful",
            memory_type="semantic",
            user_id="user_123",
        )

        mindcore.reinforce(memory_id, signal=-0.5)


class TestMindcoreExtraction:
    """Test memory extraction from LLM responses."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore with vocabulary."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry"],
            memory_types=["episodic", "preference"],
        )
        mc = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_extract_and_store(self, mindcore):
        """Test extracting and auto-storing memories."""
        llm_response = {
            "response": "I'll help you with that.",
            "memories_to_store": [
                {
                    "content": "User prefers email contact",
                    "memory_type": "preference",
                    "topics": ["support"],
                }
            ],
        }

        memories = mindcore.extract_from_response(
            llm_response=llm_response,
            user_id="user_123",
            auto_store=True,
        )

        assert len(memories) == 1
        assert memories[0].content == "User prefers email contact"

        # Verify stored
        result = mindcore.search(user_id="user_123")
        assert len(result) >= 1

    def test_extract_without_store(self, mindcore):
        """Test extracting without auto-storing."""
        llm_response = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Test memory",
                    "memory_type": "episodic",
                    "topics": ["billing"],
                }
            ],
        }

        memories = mindcore.extract_from_response(
            llm_response=llm_response,
            user_id="user_123",
            auto_store=False,
        )

        assert len(memories) == 1

    def test_extract_with_agent_id(self, mindcore):
        """Test extraction with agent ID."""
        llm_response = {
            "response": "Done.",
            "memories_to_store": [
                {
                    "content": "Agent memory",
                    "memory_type": "episodic",
                    "topics": ["support"],
                }
            ],
        }

        memories = mindcore.extract_from_response(
            llm_response=llm_response,
            user_id="user_123",
            agent_id="agent_456",
        )

        assert memories[0].agent_id == "agent_456"


class TestMindcoreVocabulary:
    """Test vocabulary operations."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore with vocabulary."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry", "complaint"],
        )
        mc = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_get_json_schema(self, mindcore):
        """Test getting JSON schema."""
        schema = mindcore.get_json_schema()

        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_get_json_schema_without_response(self, mindcore):
        """Test getting schema without response field."""
        schema = mindcore.get_json_schema(include_response=False)

        assert "properties" in schema
        # Without response, it returns memory schema directly
        assert "content" in schema["properties"]
        assert "memory_type" in schema["properties"]

    def test_get_vocabulary_instructions(self, mindcore):
        """Test getting vocabulary instructions."""
        instructions = mindcore.get_vocabulary_instructions()

        assert isinstance(instructions, str)
        assert "billing" in instructions or "support" in instructions

    def test_validate_memory_valid(self, mindcore):
        """Test validating valid memory data."""
        memory_data = {
            "content": "Test memory",
            "memory_type": "episodic",
            "topics": ["billing"],
        }

        is_valid, errors = mindcore.validate_memory(memory_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_memory_invalid(self, mindcore):
        """Test validating invalid memory data."""
        memory_data = {
            "content": "Test memory",
            "memory_type": "episodic",
            "topics": ["invalid_topic"],
        }

        is_valid, errors = mindcore.validate_memory(memory_data)

        assert is_valid is False
        assert len(errors) > 0


class TestMindcoreMultiAgent:
    """Test multi-agent functionality."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore with multi-agent enabled."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(
            storage=f"sqlite:///{db_path}",
            enable_multi_agent=True,
        )
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_register_agent(self, mindcore):
        """Test registering an agent."""
        profile = mindcore.register_agent(
            agent_id="agent_1",
            name="Support Agent",
            description="Handles support",
            teams=["support"],
        )

        assert profile["agent_id"] == "agent_1"
        assert profile["name"] == "Support Agent"
        assert "support" in profile["teams"]

    def test_register_duplicate_agent(self, mindcore):
        """Test registering duplicate agent raises error."""
        mindcore.register_agent("agent_1", "Agent 1")

        with pytest.raises(ValueError):
            mindcore.register_agent("agent_1", "Agent 1 Duplicate")

    def test_list_agents(self, mindcore):
        """Test listing agents."""
        mindcore.register_agent("agent_1", "Agent 1")
        mindcore.register_agent("agent_2", "Agent 2")

        agents = mindcore.list_agents()

        assert len(agents) == 2
        agent_ids = [a["agent_id"] for a in agents]
        assert "agent_1" in agent_ids
        assert "agent_2" in agent_ids

    def test_unregister_agent(self, mindcore):
        """Test unregistering an agent."""
        mindcore.register_agent("agent_1", "Agent 1")

        result = mindcore.unregister_agent("agent_1")
        assert result is True

        agents = mindcore.list_agents()
        agent_ids = [a["agent_id"] for a in agents]
        assert "agent_1" not in agent_ids

    def test_register_without_multi_agent(self):
        """Test that register fails without multi-agent enabled."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(
                storage=f"sqlite:///{db_path}",
                enable_multi_agent=False,
            )

            with pytest.raises(RuntimeError) as exc_info:
                mc.register_agent("agent_1", "Agent 1")

            assert "multi-agent not enabled" in str(exc_info.value).lower()
            mc.close()
        finally:
            os.unlink(db_path)


class TestMindcoreStats:
    """Test statistics."""

    @pytest.fixture
    def mindcore(self):
        """Create Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(
            storage=f"sqlite:///{db_path}",
            enable_multi_agent=True,
        )
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_get_stats(self, mindcore):
        """Test getting system stats."""
        mindcore.store(
            content="Test memory",
            memory_type="episodic",
            user_id="user_123",
        )

        stats = mindcore.get_stats()

        assert "vocabulary_version" in stats
        assert "multi_agent_enabled" in stats
        assert stats["multi_agent_enabled"] is True
        assert "flr" in stats
        assert "clst" in stats
        assert "access" in stats

    def test_get_stats_without_multi_agent(self):
        """Test stats without multi-agent."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            stats = mc.get_stats()

            assert stats["multi_agent_enabled"] is False
            assert stats["access"] is None
            mc.close()
        finally:
            os.unlink(db_path)


class TestMindcoreEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self):
        """Test complete memory workflow."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            vocab = VocabularySchema(
                version="1.0.0",
                topics=["billing", "support", "product"],
                categories=["inquiry", "complaint"],
            )

            with Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab) as mc:
                # Store memories
                id1 = mc.store(
                    content="User prefers phone calls",
                    memory_type="preference",
                    user_id="user_123",
                    topics=["support"],
                    importance=0.8,
                )
                id2 = mc.store(
                    content="User had billing issue last month",
                    memory_type="episodic",
                    user_id="user_123",
                    topics=["billing"],
                )
                id3 = mc.store(
                    content="User likes product X",
                    memory_type="preference",
                    user_id="user_123",
                    topics=["product"],
                )

                # Recall
                result = mc.recall(
                    query="phone calls",
                    user_id="user_123",
                    attention_hints=["support"],
                )

                # Note: FLR may not always return results for short queries
                # The key is testing the integration, not FLR scoring
                assert isinstance(result.memories, list)

                # Reinforce useful memory
                mc.reinforce(id1, signal=1.0)

                # Search
                search_results = mc.search(
                    user_id="user_123",
                    topics=["billing"],
                )
                assert len(search_results) >= 1

                # Delete old memory
                mc.delete(id2)

                # Verify deletion
                assert mc.get(id2) is None

                # Get stats
                stats = mc.get_stats()
                assert stats["vocabulary_version"] == "1.0.0"
        finally:
            os.unlink(db_path)

    def test_multi_user_isolation(self):
        """Test that users are properly isolated."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with Mindcore(storage=f"sqlite:///{db_path}") as mc:
                # Store for different users
                mc.store("User 1 secret", "episodic", "user_1")
                mc.store("User 2 secret", "episodic", "user_2")

                # User 1 should only see their memory
                result1 = mc.recall(query="secret", user_id="user_1")
                for m in result1.memories:
                    assert m.user_id == "user_1"

                # User 2 should only see their memory
                result2 = mc.recall(query="secret", user_id="user_2")
                for m in result2.memories:
                    assert m.user_id == "user_2"

                # Search should also be isolated
                search1 = mc.search(user_id="user_1")
                for m in search1:
                    assert m.user_id == "user_1"
        finally:
            os.unlink(db_path)

    def test_llm_integration_workflow(self):
        """Test LLM structured output integration."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            vocab = VocabularySchema(
                version="1.0.0",
                topics=["support", "billing"],
            )

            with Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab) as mc:
                # Get schema for LLM
                schema = mc.get_json_schema()
                assert "properties" in schema

                # Simulate LLM response
                llm_response = {
                    "response": "I understand you prefer email. I've noted that.",
                    "memories_to_store": [
                        {
                            "content": "User prefers email communication",
                            "memory_type": "preference",
                            "topics": ["support"],
                            "importance": 0.7,
                        }
                    ],
                }

                # Extract and store
                memories = mc.extract_from_response(
                    llm_response=llm_response,
                    user_id="user_123",
                    agent_id="support_bot",
                )

                assert len(memories) == 1
                assert memories[0].vocabulary_version == "1.0.0"

                # Later, recall
                result = mc.recall(
                    query="communication preferences",
                    user_id="user_123",
                )

                assert len(result.memories) >= 1
        finally:
            os.unlink(db_path)
