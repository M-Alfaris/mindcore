"""Tests for Mindcore v2 architecture."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from mindcore import (
    AccessLevel,
    Memory,
    MemoryType,
    Mindcore,
    VocabularySchema,
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

        # Recall using query words that match the stored content
        result = memory.recall(
            query="dark mode user prefers",
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

        # Delete (now raises on failure instead of returning False)
        memory.delete(memory_id)

        # Verify it's gone
        assert memory.get(memory_id) is None

    def test_search_with_filters(self, memory):
        """Test search with various filters."""
        # Store memories with different topics (using valid vocabulary topics)
        memory.store("About billing", "semantic", "user123", topics=["billing"])
        memory.store("About orders", "semantic", "user123", topics=["order"])
        memory.store("About products", "semantic", "user123", topics=["product"])

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
    """Test vocabulary schema functionality using SharedVocabularyLayer."""

    def test_vocabulary_creation(self):
        """Test creating a vocabulary schema using SVLSchema."""
        from mindcore.svl import SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support", "orders"],
            categories=["inquiry", "complaint"],
        )
        vocab = SharedVocabularyLayer(schema=schema)

        assert vocab.schema.version == "1.0.0"
        assert len(vocab.schema.topics) == 3
        assert len(vocab.schema.categories) == 2

    def test_json_schema_export(self):
        """Test JSON schema export."""
        from mindcore.svl import SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support"],
        )
        vocab = SharedVocabularyLayer(schema=schema)

        json_schema = vocab.get_full_memory_schema()

        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "memories_to_store" in json_schema["properties"]

    def test_validation(self):
        """Test memory validation against vocabulary."""
        from mindcore.svl import SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support"],
            categories=["inquiry"],
        )
        vocab = SharedVocabularyLayer(schema=schema)

        # Valid memory
        valid_memory = {
            "content": "Test",
            "memory_type": "semantic",
            "topics": ["billing"],
        }
        is_valid, errors = vocab.validate_memory(valid_memory)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid topic
        invalid_memory = {
            "content": "Test",
            "memory_type": "semantic",
            "topics": ["invalid_topic"],
        }
        is_valid, errors = vocab.validate_memory(invalid_memory)
        assert is_valid is False
        assert len(errors) > 0

    def test_pydantic_export(self):
        """Test Pydantic model generation."""
        from mindcore.svl import SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support"],
        )
        vocab = SharedVocabularyLayer(schema=schema)

        code = vocab.to_pydantic()
        assert "class Memory(BaseModel)" in code
        assert "content: str" in code

    def test_typescript_export(self):
        """Test TypeScript type generation."""
        from mindcore.svl import SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support"],
        )
        vocab = SharedVocabularyLayer(schema=schema)

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
        # Store memories (using valid vocabulary topics)
        memory.store("Billing issue resolved", "episodic", "user123", topics=["billing"])
        memory.store("Product question asked", "episodic", "user123", topics=["product"])
        memory.store("Order placed", "episodic", "user123", topics=["order"])

        # Recall with billing topic filter - query matches stored content
        result = memory.recall(
            query="billing issue",
            user_id="user123",
            attention_hints=["billing"],
        )

        # Billing-related should be returned
        assert len(result.memories) > 0
        assert any("billing" in m.topics for m in result.memories)

    def test_recall_returns_scores(self, memory):
        """Test that recall returns relevance scores."""
        memory.store("Test memory", "semantic", "user123", topics=["product"])

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
                topics=["product"],
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

        # Unregister (now raises on failure instead of returning False)
        memory.unregister_agent("temp_agent")

        agents = memory.list_agents()
        assert all(a["agent_id"] != "temp_agent" for a in agents)


class TestDirectStructuredOutput:
    """Test direct memory storage from structured LLM output."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_store_from_structured_output(self, memory):
        """Test storing directly from LLM structured output."""
        # Simulated LLM response with structured output
        llm_response = {
            "response": "I'll help you with that.",
            "memories_to_store": [
                {
                    "content": "User prefers email communication",
                    "memory_type": "preference",
                    "topics": ["settings"],
                    "importance": 0.8,
                }
            ],
        }

        # Store directly - no extraction layer needed
        stored_ids = []
        for mem in llm_response["memories_to_store"]:
            memory_id = memory.store(
                content=mem["content"],
                memory_type=mem["memory_type"],
                user_id="user123",
                topics=mem.get("topics", []),
                importance=mem.get("importance", 0.5),
            )
            stored_ids.append(memory_id)

        assert len(stored_ids) == 1

        # Verify it was stored correctly
        retrieved = memory.get(stored_ids[0])
        assert retrieved is not None
        assert retrieved.memory_type == "preference"
        assert retrieved.importance == 0.8

    def test_store_multiple_memories(self, memory):
        """Test storing multiple memories from structured output."""
        llm_response = {
            "response": "Here's what I learned about you.",
            "memories_to_store": [
                {
                    "content": "Prefers dark mode",
                    "memory_type": "preference",
                    "topics": ["settings"],
                },
                {"content": "Uses Python daily", "memory_type": "semantic", "topics": ["api"]},
                {
                    "content": "Works at tech company",
                    "memory_type": "entity",
                    "topics": ["account"],
                },
            ],
        }

        for mem in llm_response["memories_to_store"]:
            memory.store(
                content=mem["content"],
                memory_type=mem["memory_type"],
                user_id="user123",
                topics=mem.get("topics", []),
            )

        # Recall should find all stored memories
        result = memory.recall(query="user preferences settings", user_id="user123")
        assert len(result.memories) >= 1

    def test_get_json_schema_for_llm(self, memory):
        """Test getting JSON schema to configure LLM structured output."""
        schema = memory.get_json_schema()

        # Schema should define the expected output format
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "memories_to_store" in schema["properties"]
        assert "response" in schema["properties"]


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


# === NEW TESTS FOR IMPLEMENTED IMPROVEMENTS ===


class TestReinforcementBounds:
    """Test reinforcement score bounds checking (Improvement #1)."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_reinforcement_score_bounds_positive(self, memory):
        """Test that reinforcement scores stay within bounds for positive signals."""
        memory_id = memory.store(
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
        )

        # Apply many positive reinforcements
        for _ in range(100):
            memory.reinforce(memory_id, signal=1.0)

        retrieved = memory.get(memory_id)
        assert retrieved.reinforcement_score <= 1.0
        assert retrieved.reinforcement_score >= -1.0

    def test_reinforcement_score_bounds_negative(self, memory):
        """Test that reinforcement scores stay within bounds for negative signals."""
        memory_id = memory.store(
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
        )

        # Apply many negative reinforcements
        for _ in range(100):
            memory.reinforce(memory_id, signal=-1.0)

        retrieved = memory.get(memory_id)
        assert retrieved.reinforcement_score >= -1.0
        assert retrieved.reinforcement_score <= 1.0

    def test_reinforcement_signal_clamping(self, memory):
        """Test that out-of-range signals are clamped."""
        memory_id = memory.store(
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
        )

        # Apply out-of-range signals
        memory.reinforce(memory_id, signal=10.0)  # Should be clamped to 1.0
        memory.reinforce(memory_id, signal=-10.0)  # Should be clamped to -1.0

        retrieved = memory.get(memory_id)
        assert -1.0 <= retrieved.reinforcement_score <= 1.0

    def test_reinforcement_invalid_signal_raises(self, memory):
        """Test that invalid signal types raise ValueError."""
        memory_id = memory.store(
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
        )

        with pytest.raises(TypeError, match="Signal must be a number"):
            memory.reinforce(memory_id, signal="not a number")

    def test_memory_apply_reinforcement_diminishing_returns(self):
        """Test that Memory.apply_reinforcement has diminishing returns near bounds."""
        mem = Memory(
            memory_id="test",
            content="Test",
            memory_type="semantic",
            user_id="user123",
            reinforcement_score=0.9,  # Already high
        )

        # Apply positive signal - should have diminishing effect
        old_score = mem.reinforcement_score
        mem.apply_reinforcement(0.5)
        change = mem.reinforcement_score - old_score

        # The change should be less than 0.5 due to diminishing returns
        assert change < 0.5
        assert mem.reinforcement_score <= 1.0


class TestVocabularyMigrationRollback:
    """Test vocabulary migration with rollback capability using SVL."""

    def test_migration_creates_checkpoints(self):
        """Test that migration creates rollback checkpoints."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing", "support"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    renames={"old_billing": "billing", "old_support": "support"},
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        memory_data = {
            "memory_id": "test_123",
            "content": "Test",
            "topics": ["old_billing"],
            "vocabulary_version": "1.0.0",
        }

        migrated, checkpoint = svl.migrate_memory(memory_data, "1.0.0", create_checkpoint=True)

        assert checkpoint is not None
        assert checkpoint.from_version == "1.0.0"
        assert checkpoint.to_version == "2.0.0"
        assert checkpoint.original_data["topics"] == ["old_billing"]
        assert "billing" in migrated["topics"]

    def test_migration_rollback_restores_original(self):
        """Test that rollback restores original data."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing", "support"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    renames={"old_billing": "billing"},
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        original_data = {
            "memory_id": "test_123",
            "content": "Test",
            "topics": ["old_billing"],
            "vocabulary_version": "1.0.0",
        }

        # Migrate
        migrated, checkpoint = svl.migrate_memory(original_data, "1.0.0", create_checkpoint=True)
        assert "billing" in migrated["topics"]

        # Rollback
        restored = svl.rollback_memory(migrated, checkpoint)
        assert restored["topics"] == ["old_billing"]
        assert restored["vocabulary_version"] == "1.0.0"

    def test_migration_without_checkpoint_uses_metadata(self):
        """Test rollback using embedded metadata when no checkpoint provided."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    renames={"old_billing": "billing"},
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        original_data = {
            "memory_id": "test_123",
            "content": "Test",
            "topics": ["old_billing"],
            "vocabulary_version": "1.0.0",
        }

        # Migrate with checkpoint creation (embeds metadata)
        migrated, _ = svl.migrate_memory(original_data, "1.0.0", create_checkpoint=True)

        # Should have migration metadata embedded
        assert "_migration_metadata" in migrated

        # Rollback without explicit checkpoint
        restored = svl.rollback_memory(migrated, checkpoint=None)
        assert restored["topics"] == ["old_billing"]

    def test_migration_no_checkpoint_raises_on_rollback(self):
        """Test that rollback fails without checkpoint or metadata."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    renames={"old_billing": "billing"},
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        # Memory without migration metadata
        migrated_data = {
            "memory_id": "test_123",
            "content": "Test",
            "topics": ["billing"],
            "vocabulary_version": "2.0.0",
        }

        with pytest.raises(ValueError, match="Cannot rollback"):
            svl.rollback_memory(migrated_data, checkpoint=None)

    def test_get_migration_path(self):
        """Test getting migration path between versions."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        path = svl.get_migration_path("1.0.0")
        assert path == ["1.0.0", "2.0.0"]

        # Same version returns single element
        path = svl.get_migration_path("2.0.0")
        assert path == ["2.0.0"]

        # Unknown version raises
        with pytest.raises(ValueError, match="No migration path"):
            svl.get_migration_path("0.5.0")


class TestExceptionHandling:
    """Test standardized exception handling (Improvement #3)."""

    @pytest.fixture
    def memory(self):
        """Create a Mindcore instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mc = Mindcore(storage=f"sqlite:///{db_path}")
        yield mc
        mc.close()
        os.unlink(db_path)

    def test_delete_nonexistent_raises_exception(self, memory):
        """Test that deleting nonexistent memory raises MemoryNotFoundError."""
        from mindcore.exceptions import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError) as exc_info:
            memory.delete("nonexistent_id")

        assert "nonexistent_id" in str(exc_info.value)
        assert exc_info.value.memory_id == "nonexistent_id"

    def test_memory_not_found_has_details(self):
        """Test that MemoryNotFoundError includes details."""
        from mindcore.exceptions import MemoryNotFoundError

        error = MemoryNotFoundError("test_id")
        assert error.memory_id == "test_id"
        assert "test_id" in error.details["memory_id"]
        assert "test_id" in str(error)

    def test_multi_agent_not_enabled_raises(self):
        """Test that multi-agent operations raise when not enabled."""
        from mindcore.exceptions import MultiAgentNotEnabledError

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}", enable_multi_agent=False)

            with pytest.raises(MultiAgentNotEnabledError):
                mc.list_agents()

            with pytest.raises(MultiAgentNotEnabledError):
                mc.register_agent("test", "Test")

            mc.close()
        finally:
            os.unlink(db_path)

    def test_storage_error_includes_backend(self):
        """Test that StorageError includes backend info."""
        from mindcore.exceptions import StorageConnectionError

        error = StorageConnectionError("Connection failed", backend="postgresql")
        assert error.backend == "postgresql"
        assert "postgresql" in error.details["backend"]

    def test_validation_error_includes_errors(self):
        """Test that VocabularyValidationError includes error list."""
        from mindcore.exceptions import VocabularyValidationError

        errors = ["Invalid topic: foo", "Invalid category: bar"]
        error = VocabularyValidationError(errors)

        assert error.errors == errors
        assert "validation_errors" in error.details
        assert len(error.details["validation_errors"]) == 2

    def test_exception_hierarchy(self):
        """Test that exception hierarchy is correct."""
        from mindcore.exceptions import (
            AccessError,
            ConfigurationError,
            MemoryNotFoundError,
            MindcoreError,
            StorageError,
            ValidationError,
        )

        # All should inherit from MindcoreError
        assert issubclass(StorageError, MindcoreError)
        assert issubclass(MemoryNotFoundError, StorageError)
        assert issubclass(ValidationError, MindcoreError)
        assert issubclass(AccessError, MindcoreError)
        assert issubclass(ConfigurationError, MindcoreError)


class TestCrossAgentConflictResolution:
    """Test cross-agent sync conflict resolution (Improvement #4)."""

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

    def test_conflict_resolution_enum_values(self):
        """Test that ConflictResolution enum has all strategies."""
        from mindcore.cross_agent.sharing import ConflictResolution

        assert hasattr(ConflictResolution, "SOURCE_WINS")
        assert hasattr(ConflictResolution, "TARGET_WINS")
        assert hasattr(ConflictResolution, "NEWEST_WINS")
        assert hasattr(ConflictResolution, "HIGHEST_IMPORTANCE")
        assert hasattr(ConflictResolution, "MERGE_METADATA")
        assert hasattr(ConflictResolution, "SKIP")

    def test_conflict_info_to_dict(self):
        """Test ConflictInfo serialization."""
        from mindcore.cross_agent.sharing import ConflictInfo, ConflictResolution

        info = ConflictInfo(
            memory_id="test_123",
            source_memory=Memory(
                memory_id="src",
                content="Source content",
                memory_type="semantic",
                user_id="user1",
            ),
            target_memory=Memory(
                memory_id="tgt",
                content="Target content",
                memory_type="semantic",
                user_id="user1",
            ),
            resolution=ConflictResolution.SOURCE_WINS,
            resolved_memory=None,
            reason="Test reason",
        )

        data = info.to_dict()
        assert data["memory_id"] == "test_123"
        assert data["resolution"] == "source_wins"
        assert data["reason"] == "Test reason"

    def test_sync_result_includes_conflict_details(self):
        """Test that SyncResult includes conflict information."""
        from mindcore.cross_agent.sharing import ConflictResolution, SyncDirection, SyncResult

        result = SyncResult(
            success=True,
            source_agent="agent1",
            target_agent="agent2",
            direction=SyncDirection.ONE_WAY,
            memories_synced=5,
            memories_skipped=2,
            conflicts=2,
            conflict_resolution=ConflictResolution.NEWEST_WINS,
        )

        data = result.to_dict()
        assert data["conflicts"] == 2
        assert data["conflict_resolution"] == "newest_wins"
        assert "conflict_details" in data

    def test_cross_agent_layer_sync_with_conflict_resolution(self, memory):
        """Test CrossAgentLayer sync with conflict resolution parameters."""
        from mindcore.cross_agent import (
            ConflictResolution,
            CrossAgentLayer,
            SyncDirection,
        )

        # Create a CrossAgentLayer for testing sync
        layer = CrossAgentLayer(storage=memory._storage)
        layer.register_agent("agent1", "Agent 1", teams=["team1"])
        layer.register_agent("agent2", "Agent 2", teams=["team1"])

        # Store some shared memories via the layer
        from mindcore import Memory

        mem = Memory(
            memory_id="test_sync_mem",
            content="Shared knowledge",
            memory_type="semantic",
            user_id="user123",
            agent_id="agent1",
            access_level="team",
        )
        memory._storage.store(mem)

        # Sync with conflict resolution
        result = layer.sync(
            source_agent="agent1",
            target_agent="agent2",
            user_id="user123",
            direction=SyncDirection.ONE_WAY,
            conflict_resolution=ConflictResolution.SOURCE_WINS,
        )

        assert result is not None
        assert result.conflict_resolution == ConflictResolution.SOURCE_WINS
        assert hasattr(result, "conflict_details")


class TestConnectionPoolManagement:
    """Test connection pool management (Improvement #5)."""

    def test_sqlite_storage_has_pool_config(self):
        """Test SQLiteStorage accepts pool configuration."""
        from mindcore.storage import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(
                db_path=db_path,
                max_connections=5,
                connection_timeout=10.0,
            )

            assert storage.max_connections == 5
            assert storage.connection_timeout == 10.0

            storage.close()
        finally:
            os.unlink(db_path)

    def test_sqlite_storage_stats_include_pool_info(self):
        """Test that SQLite stats include connection pool info."""
        from mindcore.storage import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path=db_path, max_connections=10)

            stats = storage.get_stats()

            assert "connection_pool" in stats
            assert stats["connection_pool"]["max_connections"] == 10
            assert "active_connections" in stats["connection_pool"]
            assert "available_connections" in stats["connection_pool"]

            storage.close()
        finally:
            os.unlink(db_path)

    def test_sqlite_update_nonexistent_raises(self):
        """Test that SQLite update raises MemoryNotFoundError."""
        from mindcore.exceptions import MemoryNotFoundError
        from mindcore.storage import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path=db_path)

            mem = Memory(
                memory_id="nonexistent",
                content="Test",
                memory_type="semantic",
                user_id="user123",
            )

            with pytest.raises(MemoryNotFoundError):
                storage.update(mem)

            storage.close()
        finally:
            os.unlink(db_path)

    def test_sqlite_update_reinforcement_nonexistent_raises(self):
        """Test that update_reinforcement raises MemoryNotFoundError."""
        from mindcore.exceptions import MemoryNotFoundError
        from mindcore.storage import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path=db_path)

            with pytest.raises(MemoryNotFoundError):
                storage.update_reinforcement("nonexistent", 0.5)

            storage.close()
        finally:
            os.unlink(db_path)

    def test_sqlite_reinforcement_bounds_in_storage(self):
        """Test that SQLite storage enforces reinforcement bounds."""
        from mindcore.storage import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path=db_path)

            mem = Memory(
                memory_id="test_123",
                content="Test",
                memory_type="semantic",
                user_id="user123",
                reinforcement_score=0.0,
            )
            storage.store(mem)

            # Apply large positive signal
            storage.update_reinforcement("test_123", 100.0)
            retrieved = storage.get("test_123")
            assert retrieved.reinforcement_score <= 1.0

            # Apply large negative signal
            storage.update_reinforcement("test_123", -200.0)
            retrieved = storage.get("test_123")
            assert retrieved.reinforcement_score >= -1.0

            storage.close()
        finally:
            os.unlink(db_path)


class TestCLSTMigrationWithRollback:
    """Test CLST migration with rollback capability."""

    @pytest.fixture
    def memory_with_migration(self):
        """Create a Mindcore instance with custom vocabulary for migration."""
        from mindcore.svl import Migration, SharedVocabularyLayer, SVLSchema

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        schema = SVLSchema(
            version="2.0.0",
            topics=["billing", "support", "orders"],
            migrations={
                "1.0.0": Migration(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    renames={"old_billing": "billing"},
                    added_fields={"new_field": "default"},
                )
            },
        )
        svl = SharedVocabularyLayer(schema=schema)

        mc = Mindcore(
            storage=f"sqlite:///{db_path}",
            vocabulary=svl,
        )
        yield mc, db_path
        mc.close()
        os.unlink(db_path)

    def test_clst_migrate_returns_checkpoint_info(self, memory_with_migration):
        """Test that CLST migrate returns checkpoint information."""
        mc, _ = memory_with_migration

        # Store a memory with old version
        memory_id = mc.store(
            content="Old billing memory",
            memory_type="semantic",
            user_id="user123",
            topics=["billing"],
        )

        # Manually set vocabulary version to old
        mem = mc.get(memory_id)
        mem.vocabulary_version = "1.0.0"
        mc._storage.update(mem)

        # Migrate
        result = mc.migrate_vocabulary(from_version="1.0.0", create_checkpoints=True)

        assert "can_rollback" in result
        assert "checkpoint_count" in result

    def test_mindcore_rollback_vocabulary_migration(self, memory_with_migration):
        """Test Mindcore.rollback_vocabulary_migration."""
        mc, _ = memory_with_migration

        # Store memory with old version
        memory_id = mc.store(
            content="Test memory",
            memory_type="semantic",
            user_id="user123",
            topics=["billing"],
        )

        # Set to old version
        mem = mc.get(memory_id)
        mem.vocabulary_version = "1.0.0"
        mc._storage.update(mem)

        # Migrate
        mc.migrate_vocabulary(from_version="1.0.0", create_checkpoints=True)

        # Rollback
        result = mc.rollback_vocabulary_migration()

        assert "memories_rolled_back" in result
        assert "from_version" in result
        assert "to_version" in result

    def test_rollback_without_migration_raises(self, memory_with_migration):
        """Test that rollback without prior migration raises error."""
        mc, _ = memory_with_migration

        with pytest.raises(ValueError, match="No migration to rollback"):
            mc.rollback_vocabulary_migration()


class TestMigrationCheckpoint:
    """Test MigrationCheckpoint dataclass."""

    def test_checkpoint_serialization(self):
        """Test checkpoint to_dict and from_dict."""
        from mindcore.svl import MigrationCheckpoint

        checkpoint = MigrationCheckpoint(
            checkpoint_id="cp_123",
            from_version="1.0.0",
            to_version="2.0.0",
            memory_id="mem_456",
            original_data={"topics": ["old"]},
            migrated_data={"topics": ["new"]},
        )

        data = checkpoint.to_dict()
        assert data["checkpoint_id"] == "cp_123"
        assert data["from_version"] == "1.0.0"

        restored = MigrationCheckpoint.from_dict(data)
        assert restored.checkpoint_id == "cp_123"
        assert restored.memory_id == "mem_456"
        assert restored.original_data == {"topics": ["old"]}


class TestMigrationRollbackTopics:
    """Test Migration.rollback_topics functionality."""

    def test_rollback_topics_with_original(self):
        """Test rollback uses original topics when available."""
        from mindcore.svl import Migration

        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            renames={"old": "new"},
        )

        current = ["new", "other"]
        original = ["old", "other"]

        result = migration.rollback_topics(current, original)
        assert result == original

    def test_rollback_topics_reverses_renames(self):
        """Test rollback reverses renames when no original."""
        from mindcore.svl import Migration

        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
            renames={"old_topic": "new_topic"},
        )

        current = ["new_topic", "unchanged"]
        result = migration.rollback_topics(current, original_topics=[])

        assert "old_topic" in result
        assert "unchanged" in result

    def test_migration_can_rollback(self):
        """Test Migration.can_rollback method."""
        from mindcore.svl import Migration

        migration = Migration(
            from_version="1.0.0",
            to_version="2.0.0",
        )

        assert migration.can_rollback() is True
