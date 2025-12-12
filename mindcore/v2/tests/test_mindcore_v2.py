"""Tests for Mindcore v2 architecture."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from mindcore.v2 import (
    Mindcore,
    VocabularySchema,
    Memory,
    MemoryType,
    AccessLevel,
)


class TestMindcoreBasics:
    """Test basic Mindcore operations."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance with temp database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_store_and_recall(self, memory):
        """Test basic store and recall."""
        # Store a memory
        memory_id = memory.store(
            content="User prefers dark mode",
            memory_type="preference",
            user_id="user123",
            topics=["settings"],
            importance=0.8,
        )

        assert memory_id is not None

        # Recall
        result = memory.recall(
            query="What are the user's preferences?",
            user_id="user123",
        )

        assert len(result.memories) > 0
        assert "dark mode" in result.memories[0].content

    def test_store_and_get(self, memory):
        """Test store and get by ID."""
        memory_id = memory.store(
            content="Test memory",
            memory_type="episodic",
            user_id="user123",
        )

        retrieved = memory.get(memory_id)

        assert retrieved is not None
        assert retrieved.content == "Test memory"
        assert retrieved.memory_type == "episodic"

    def test_delete(self, memory):
        """Test delete memory."""
        memory_id = memory.store(
            content="To be deleted",
            memory_type="episodic",
            user_id="user123",
        )

        # Verify it exists
        assert memory.get(memory_id) is not None

        # Delete
        result = memory.delete(memory_id)
        assert result is True

        # Verify it's gone
        assert memory.get(memory_id) is None

    def test_search_with_filters(self, memory):
        """Test search with various filters."""
        # Store memories with different topics
        memory.store("About billing", "semantic", "user123", topics=["billing"])
        memory.store("About orders", "semantic", "user123", topics=["orders"])
        memory.store("About support", "semantic", "user123", topics=["support"])

        # Search by topic
        results = memory.search(user_id="user123", topics=["billing"])
        assert len(results) >= 1
        assert all("billing" in m.topics for m in results)

    def test_reinforce(self, memory):
        """Test memory reinforcement."""
        memory_id = memory.store(
            content="Important memory",
            memory_type="semantic",
            user_id="user123",
        )

        # Reinforce positively
        memory.reinforce(memory_id, signal=0.5)

        # Check that it still works (reinforcement is internal)
        retrieved = memory.get(memory_id)
        assert retrieved is not None


class TestVocabulary:
    """Test vocabulary schema functionality."""

    def test_vocabulary_creation(self):
        """Test creating a vocabulary schema."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support", "orders"],
            categories=["inquiry", "complaint"],
        )

        assert vocab.version == "1.0.0"
        assert len(vocab.topics) == 3
        assert len(vocab.categories) == 2

    def test_json_schema_export(self):
        """Test JSON schema export."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
        )

        schema = vocab.to_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "memories_to_store" in schema["properties"]

    def test_validation(self):
        """Test memory validation against vocabulary."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry"],
        )

        # Valid memory
        valid_memory = {
            "content": "Test",
            "memory_type": "semantic",
            "topics": ["billing"],
        }
        is_valid, errors = vocab.validate(valid_memory)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid topic
        invalid_memory = {
            "content": "Test",
            "memory_type": "semantic",
            "topics": ["invalid_topic"],
        }
        is_valid, errors = vocab.validate(invalid_memory)
        assert is_valid is False
        assert len(errors) > 0

    def test_pydantic_export(self):
        """Test Pydantic model generation."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
        )

        code = vocab.to_pydantic()
        assert "class Memory(BaseModel)" in code
        assert "content: str" in code

    def test_typescript_export(self):
        """Test TypeScript type generation."""
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["billing", "support"],
        )

        ts = vocab.to_typescript()
        assert "interface Memory" in ts
        assert "content: string" in ts


class TestFLR:
    """Test FLR (Fast Learning Recall) functionality."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_recall_with_attention_hints(self, memory):
        """Test recall with attention hints."""
        # Store memories
        memory.store("Billing issue resolved", "episodic", "user123", topics=["billing"])
        memory.store("Support ticket opened", "episodic", "user123", topics=["support"])
        memory.store("Order placed", "episodic", "user123", topics=["orders"])

        # Recall with billing attention
        result = memory.recall(
            query="Tell me about recent activity",
            user_id="user123",
            attention_hints=["billing"],
        )

        # Billing-related should be prioritized
        assert len(result.memories) > 0
        assert "billing" in result.attention_focus or any(
            "billing" in m.topics for m in result.memories
        )

    def test_recall_returns_scores(self, memory):
        """Test that recall returns relevance scores."""
        memory.store("Test memory", "semantic", "user123", topics=["test"])

        result = memory.recall(query="test", user_id="user123")

        assert len(result.scores) == len(result.memories)
        assert all(0 <= s <= 1 for s in result.scores)


class TestCLST:
    """Test CLST (Cognitive Long-term Storage Transfer) functionality."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_compress_old_memories(self, memory):
        """Test memory compression."""
        # Store some memories
        for i in range(15):
            memory.store(
                content=f"Memory {i}",
                memory_type="episodic",
                user_id="user123",
                topics=["test"],
            )

        # Compress (with 0 days to compress all)
        result = memory.compress(
            user_id="user123",
            older_than_days=0,
            strategy="deduplicate",
        )

        assert "original_count" in result
        assert "compressed_count" in result


class TestMultiAgent:
    """Test multi-agent functionality."""

    @pytest.fixture
    def memory(self):
        """Create a multi-agent Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(
            storage=f"sqlite:///{db_path}",
            enable_multi_agent=True,
        )
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_register_agent(self, memory):
        """Test agent registration."""
        profile = memory.register_agent(
            agent_id="support_bot",
            name="Support Agent",
            description="Handles support queries",
            teams=["support"],
        )

        assert profile["agent_id"] == "support_bot"
        assert profile["name"] == "Support Agent"
        assert "support" in profile["teams"]

    def test_list_agents(self, memory):
        """Test listing agents."""
        memory.register_agent("agent1", "Agent 1")
        memory.register_agent("agent2", "Agent 2")

        agents = memory.list_agents()
        assert len(agents) == 2

    def test_unregister_agent(self, memory):
        """Test unregistering an agent."""
        memory.register_agent("temp_agent", "Temporary")

        result = memory.unregister_agent("temp_agent")
        assert result is True

        agents = memory.list_agents()
        assert all(a["agent_id"] != "temp_agent" for a in agents)


class TestExtraction:
    """Test memory extraction functionality."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_extract_from_structured_output(self, memory):
        """Test extracting from LLM structured output."""
        llm_response = {
            "response": "I'll help you with that.",
            "memories_to_store": [
                {
                    "content": "User prefers email communication",
                    "memory_type": "preference",
                    "topics": ["communication"],
                    "importance": 0.8,
                }
            ],
        }

        memories = memory.extract_from_response(
            llm_response=llm_response,
            user_id="user123",
            auto_store=True,
        )

        assert len(memories) == 1
        assert memories[0].memory_type == "preference"

    def test_auto_extract_from_messages(self, memory):
        """Test auto-extraction from conversation."""
        messages = [
            {"role": "user", "content": "I prefer to be contacted by email, not phone."},
            {"role": "assistant", "content": "Noted, I'll remember your preference."},
            {"role": "user", "content": "My order number is #12345."},
        ]

        memories = memory.auto_extract(
            messages=messages,
            user_id="user123",
            auto_store=True,
        )

        # Should extract preference and entity
        assert len(memories) >= 1


class TestStats:
    """Test statistics functionality."""

    def test_get_stats(self):
        """Test getting system stats."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            stats = mc.get_stats()

            assert "vocabulary_version" in stats
            assert "multi_agent_enabled" in stats
            assert "flr" in stats
            assert "clst" in stats

            mc.close()
        finally:
            os.unlink(db_path)


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test using Mindcore as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with Mindcore(storage=f"sqlite:///{db_path}") as memory:
                memory.store("Test", "semantic", "user123")
                result = memory.recall("Test", "user123")
                assert len(result.memories) > 0
        finally:
            os.unlink(db_path)
