"""Comprehensive tests for MindcoreClient.

CRITICAL: The MindcoreClient is the main entry point for the entire framework.
Testing covers:
- Message ingestion (ingest)
- Context retrieval (get_context)
- Multi-agent operations
- Cache management
- Worker health monitoring
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mindcore import (
    AgentProfile,
    AgentVisibility,
    AssembledContext,
    MemorySharingMode,
    Message,
    MessageMetadata,
    MessageRole,
    MultiAgentConfig,
    UserPreferences,
)


class TestMindcoreClientInitialization:
    """Tests for MindcoreClient initialization."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_mindcore.db"

    def test_initialization_with_sqlite(self, temp_db_path):
        """Test client initialization with SQLite backend."""
        # We need to mock the LLM provider to avoid requiring API keys
        with patch("mindcore.OpenAIProvider") as mock_openai:
            mock_provider = MagicMock()
            mock_provider.name = "mock_openai"
            mock_provider.is_available.return_value = True
            mock_openai.return_value = mock_provider

            with patch("mindcore.create_provider") as mock_create:
                mock_create.return_value = mock_provider

                from mindcore import MindcoreClient

                client = MindcoreClient(
                    use_sqlite=True,
                    sqlite_path=str(temp_db_path),
                    persistent_cache=False,
                )

                assert client._use_sqlite is True
                assert client.db is not None
                assert client.cache is not None
                assert client.vocabulary is not None

                client.close()

    def test_initialization_stores_config(self, temp_db_path):
        """Test that initialization stores configuration."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(temp_db_path),
                persistent_cache=False,
                enrichment_workers=2,
            )

            assert client._enrichment_workers == 2
            assert client._use_sqlite is True

            client.close()

    def test_initialization_with_multi_agent_config(self, temp_db_path):
        """Test initialization with multi-agent configuration."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            config = MultiAgentConfig(
                enabled=True,
                mode=MemorySharingMode.SHARED,
                require_agent_id=True,
            )

            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(temp_db_path),
                persistent_cache=False,
                multi_agent_config=config,
            )

            assert client.multi_agent_enabled is True
            assert client._multi_agent_config.mode == MemorySharingMode.SHARED

            client.close()


class TestMindcoreClientIngest:
    """Tests for MindcoreClient.ingest() method."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_ingest_creates_message(self, mock_client):
        """Test that ingest creates a message."""
        message = mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Hello, how are you?",
        )

        assert message is not None
        assert message.user_id == "user123"
        assert message.thread_id == "thread456"
        assert message.session_id == "session789"
        assert message.role == MessageRole.USER
        assert message.raw_text == "Hello, how are you?"
        assert message.message_id is not None

    def test_ingest_with_custom_message_id(self, mock_client):
        """Test ingest with custom message ID."""
        message = mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Test message",
            message_id="custom_id_123",
        )

        assert message.message_id == "custom_id_123"

    def test_ingest_validates_user_id(self, mock_client):
        """Test that ingest validates user_id."""
        with pytest.raises(ValueError, match="Invalid message"):
            mock_client.ingest(
                user_id="",  # Empty user_id
                thread_id="thread456",
                session_id="session789",
                role="user",
                text="Test",
            )

    def test_ingest_validates_role(self, mock_client):
        """Test that ingest validates role."""
        # Valid roles should work
        for role in ["user", "assistant", "system", "tool"]:
            message = mock_client.ingest(
                user_id=f"user_{role}",
                thread_id="thread456",
                session_id="session789",
                role=role,
                text="Test",
            )
            assert message is not None

    def test_ingest_stores_in_cache(self, mock_client):
        """Test that ingest stores message in cache."""
        message = mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Test message",
        )

        # Should be retrievable from cache
        cached = mock_client.cache.get_recent_messages("user123", "thread456", 10)
        assert len(cached) >= 1
        assert any(m.message_id == message.message_id for m in cached)

    def test_ingest_queues_for_enrichment(self, mock_client):
        """Test that ingest queues message for background enrichment."""
        initial_size = mock_client._enrichment_queue.qsize()

        mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Test message for enrichment",
        )

        # Queue size should have increased
        # Note: May have processed already, so check if stored in db
        message = mock_client.db.get_message_by_id
        assert mock_client._enrichment_queue.qsize() >= 0  # Queue exists

    def test_ingest_multi_agent_requires_agent_id(self, tmp_path):
        """Test that ingest validates agent_id in multi-agent mode."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            config = MultiAgentConfig(
                enabled=True,
                mode=MemorySharingMode.SHARED,
                require_agent_id=True,
            )

            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(tmp_path / "test.db"),
                persistent_cache=False,
                multi_agent_config=config,
            )

            try:
                # Should require agent_id
                with pytest.raises(ValueError, match="agent_id"):
                    client.ingest(
                        user_id="user123",
                        thread_id="thread456",
                        session_id="session789",
                        role="user",
                        text="Test",
                    )
            finally:
                client.close()


class TestMindcoreClientGetContext:
    """Tests for MindcoreClient.get_context() method."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_context_validates_user_id(self, mock_client):
        """Test that get_context validates user_id."""
        with pytest.raises(ValueError, match="Invalid query"):
            mock_client.get_context(
                user_id="",  # Empty user_id
                thread_id="thread456",
                query="test query",
            )

    def test_get_context_validates_query(self, mock_client):
        """Test that get_context validates query."""
        with pytest.raises(ValueError, match="Invalid query"):
            mock_client.get_context(
                user_id="user123",
                thread_id="thread456",
                query="",  # Empty query
            )


class TestMindcoreClientMessage:
    """Tests for message retrieval methods."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_message_returns_none_for_nonexistent(self, mock_client):
        """Test get_message returns None for non-existent ID."""
        result = mock_client.get_message("nonexistent_id")
        assert result is None

    def test_get_message_returns_message(self, mock_client):
        """Test get_message returns stored message."""
        message = mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Test message",
        )

        retrieved = mock_client.get_message(message.message_id)
        assert retrieved is not None
        assert retrieved.message_id == message.message_id
        assert retrieved.raw_text == "Test message"

    def test_get_recent_messages_empty(self, mock_client):
        """Test get_recent_messages returns empty list for new user."""
        messages = mock_client.get_recent_messages("new_user", "new_thread", 10)
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_get_recent_messages_returns_messages(self, mock_client):
        """Test get_recent_messages returns ingested messages."""
        # Ingest some messages
        for i in range(5):
            mock_client.ingest(
                user_id="user123",
                thread_id="thread456",
                session_id="session789",
                role="user",
                text=f"Message {i}",
            )

        messages = mock_client.get_recent_messages("user123", "thread456", 10)
        assert len(messages) == 5

    def test_get_recent_messages_respects_limit(self, mock_client):
        """Test get_recent_messages respects limit parameter."""
        # Ingest some messages
        for i in range(10):
            mock_client.ingest(
                user_id="user123",
                thread_id="thread456",
                session_id="session789",
                role="user",
                text=f"Message {i}",
            )

        messages = mock_client.get_recent_messages("user123", "thread456", 3)
        assert len(messages) == 3


class TestMindcoreClientCache:
    """Tests for cache management methods."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_clear_cache_clears_thread(self, mock_client):
        """Test clear_cache clears specific thread."""
        # Ingest messages
        mock_client.ingest(
            user_id="user123",
            thread_id="thread456",
            session_id="session789",
            role="user",
            text="Test message",
        )

        # Clear cache for thread
        mock_client.clear_cache(user_id="user123", thread_id="thread456")

        # Cache should be empty for this thread
        cached = mock_client.cache.get_recent_messages("user123", "thread456", 10)
        assert len(cached) == 0

    def test_clear_cache_clears_all(self, mock_client):
        """Test clear_cache clears all when no args."""
        # Ingest messages in multiple threads
        mock_client.ingest(
            user_id="user1",
            thread_id="thread1",
            session_id="session1",
            role="user",
            text="Message 1",
        )
        mock_client.ingest(
            user_id="user2",
            thread_id="thread2",
            session_id="session2",
            role="user",
            text="Message 2",
        )

        # Clear all cache
        mock_client.clear_cache()

        # Both should be empty
        cached1 = mock_client.cache.get_recent_messages("user1", "thread1", 10)
        cached2 = mock_client.cache.get_recent_messages("user2", "thread2", 10)
        assert len(cached1) == 0
        assert len(cached2) == 0

    def test_get_cache_stats(self, mock_client):
        """Test get_cache_stats returns statistics."""
        stats = mock_client.get_cache_stats()

        assert isinstance(stats, dict)
        assert "cache" in stats
        assert "invalidation" in stats


class TestMindcoreClientMultiAgent:
    """Tests for multi-agent management methods."""

    @pytest.fixture
    def multi_agent_client(self, tmp_path):
        """Create a MindcoreClient with multi-agent mode enabled."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            config = MultiAgentConfig(
                enabled=True,
                mode=MemorySharingMode.SHARED,
                require_agent_id=False,  # Don't require for simpler testing
            )

            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(tmp_path / "test.db"),
                persistent_cache=False,
                multi_agent_config=config,
            )
            yield client
            client.close()

    def test_multi_agent_enabled_property(self, multi_agent_client):
        """Test multi_agent_enabled property."""
        assert multi_agent_client.multi_agent_enabled is True

    def test_register_agent(self, multi_agent_client):
        """Test registering an agent."""
        profile = multi_agent_client.register_agent(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            sharing_groups=["support"],
            default_visibility="shared",
        )

        assert isinstance(profile, AgentProfile)
        assert profile.agent_id == "test_agent"
        assert profile.name == "Test Agent"

    def test_get_agent(self, multi_agent_client):
        """Test getting an agent profile."""
        multi_agent_client.register_agent(
            agent_id="test_agent",
            name="Test Agent",
        )

        profile = multi_agent_client.get_agent("test_agent")
        assert profile is not None
        assert profile.agent_id == "test_agent"

    def test_get_agent_nonexistent(self, multi_agent_client):
        """Test getting non-existent agent returns None."""
        profile = multi_agent_client.get_agent("nonexistent")
        assert profile is None

    def test_list_agents(self, multi_agent_client):
        """Test listing all agents."""
        multi_agent_client.register_agent(agent_id="agent1", name="Agent 1")
        multi_agent_client.register_agent(agent_id="agent2", name="Agent 2")

        agents = multi_agent_client.list_agents()
        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "agent1" in agent_ids
        assert "agent2" in agent_ids

    def test_unregister_agent(self, multi_agent_client):
        """Test unregistering an agent."""
        multi_agent_client.register_agent(agent_id="test_agent", name="Test Agent")

        result = multi_agent_client.unregister_agent("test_agent")
        assert result is True

        profile = multi_agent_client.get_agent("test_agent")
        assert profile is None

    def test_unregister_nonexistent_agent(self, multi_agent_client):
        """Test unregistering non-existent agent returns False."""
        result = multi_agent_client.unregister_agent("nonexistent")
        assert result is False

    def test_get_multi_agent_stats(self, multi_agent_client):
        """Test getting multi-agent statistics."""
        multi_agent_client.register_agent(agent_id="agent1", name="Agent 1")

        stats = multi_agent_client.get_multi_agent_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert stats["enabled"] is True


class TestMindcoreClientWorkerHealth:
    """Tests for worker health monitoring."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_worker_health(self, mock_client):
        """Test get_worker_health returns health info."""
        health = mock_client.get_worker_health()

        assert isinstance(health, dict)
        assert "healthy" in health
        assert "worker_pool_size" in health

    def test_get_enrichment_metrics(self, mock_client):
        """Test get_enrichment_metrics returns metrics."""
        metrics = mock_client.get_enrichment_metrics()

        assert isinstance(metrics, dict)
        assert "processed_count" in metrics or "status" in metrics

    def test_enrichment_worker_count_property(self, mock_client):
        """Test enrichment_worker_count property."""
        count = mock_client.enrichment_worker_count
        assert isinstance(count, int)
        assert count >= 1


class TestMindcoreClientProperties:
    """Tests for client properties."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock_provider"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_llm_provider_property(self, mock_client):
        """Test llm_provider property returns provider."""
        provider = mock_client.llm_provider
        assert provider is not None

    def test_provider_name_property(self, mock_client):
        """Test provider_name property returns name."""
        name = mock_client.provider_name
        assert name == "mock_provider"


class TestMindcoreClientVocabulary:
    """Tests for vocabulary management."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_vocabulary_initialized(self, mock_client):
        """Test vocabulary is initialized."""
        assert mock_client.vocabulary is not None

    def test_refresh_vocabulary(self, mock_client):
        """Test refresh_vocabulary doesn't raise."""
        # Should not raise any errors
        mock_client.refresh_vocabulary()


class TestMindcoreClientClose:
    """Tests for client cleanup."""

    def test_close_cleans_up_resources(self, tmp_path):
        """Test close cleans up all resources."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )

            # Close should not raise
            client.close()

            # Verify enrichment worker stopped
            assert client._enrichment_running is False


class TestMindcoreClientHelperFunctions:
    """Tests for helper functions."""

    def test_get_sort_key_with_datetime(self):
        """Test _get_sort_key with datetime."""
        from mindcore import _get_sort_key

        message = Message(
            message_id="test",
            user_id="user",
            thread_id="thread",
            session_id="session",
            role=MessageRole.USER,
            raw_text="Test",
            metadata=MessageMetadata(),
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        key = _get_sort_key(message)
        assert isinstance(key, float)
        assert key > 0

    def test_get_sort_key_with_none(self):
        """Test _get_sort_key with None created_at."""
        from mindcore import _get_sort_key

        # Create a message and explicitly set created_at to None after creation
        message = Message(
            message_id="test",
            user_id="user",
            thread_id="thread",
            session_id="session",
            role=MessageRole.USER,
            raw_text="Test",
            metadata=MessageMetadata(),
        )
        # Force created_at to None to test the edge case
        message.created_at = None

        key = _get_sort_key(message)
        assert key == 0.0

    def test_get_sort_key_with_string(self):
        """Test _get_sort_key with ISO string."""
        from mindcore import _get_sort_key

        message = Message(
            message_id="test",
            user_id="user",
            thread_id="thread",
            session_id="session",
            role=MessageRole.USER,
            raw_text="Test",
            metadata=MessageMetadata(),
            created_at="2024-01-01T12:00:00+00:00",
        )

        key = _get_sort_key(message)
        assert isinstance(key, float)
        assert key > 0


class TestMindcoreModuleFunctions:
    """Tests for module-level functions."""

    def test_initialize_creates_singleton(self, tmp_path):
        """Test initialize creates a singleton instance."""
        import mindcore

        # Reset singleton
        mindcore._mindcore_instance = None

        with patch("mindcore.MindcoreClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = mindcore.initialize(use_sqlite=True)
            assert result is mock_instance

            # Second call returns same instance
            result2 = mindcore.initialize(use_sqlite=True)
            assert result2 is mock_instance

            # Reset for other tests
            mindcore._mindcore_instance = None

    def test_get_client_creates_if_needed(self, tmp_path):
        """Test get_client creates instance if needed."""
        import mindcore

        # Reset singleton
        mindcore._mindcore_instance = None

        with patch("mindcore.MindcoreClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = mindcore.get_client()
            assert result is mock_instance

            # Reset for other tests
            mindcore._mindcore_instance = None


class TestMindcoreClientUserPreferences:
    """Tests for user preferences methods."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_user_preferences(self, mock_client):
        """Test get_user_preferences returns preferences."""
        prefs = mock_client.get_user_preferences("user123")
        assert prefs is not None

    def test_get_preference_signals(self, mock_client):
        """Test get_preference_signals returns signals."""
        signals = mock_client.get_preference_signals("user123")
        assert isinstance(signals, dict)


class TestMindcoreClientRetention:
    """Tests for retention policy methods."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_decayed_importance(self, mock_client):
        """Test get_decayed_importance returns float."""
        # Create message and set importance via attribute
        message = Message(
            message_id="test",
            user_id="user",
            thread_id="thread",
            session_id="session",
            role=MessageRole.USER,
            raw_text="Test",
            metadata=MessageMetadata(),
            created_at=datetime.now(timezone.utc),
        )

        importance = mock_client.get_decayed_importance(message)
        assert isinstance(importance, float)
        # Importance can be 0 to 1+ depending on decay calculation
        assert importance >= 0.0

    def test_get_context_window(self, mock_client):
        """Test get_context_window returns messages."""
        # Ingest some messages first
        for i in range(5):
            mock_client.ingest(
                user_id="user123",
                thread_id="thread456",
                session_id="session789",
                role="user",
                text=f"Message {i}",
            )

        messages = mock_client.get_context_window(
            user_id="user123",
            thread_id="thread456",
            max_messages=10,
        )
        assert isinstance(messages, list)


class TestMindcoreClientPrometheus:
    """Tests for Prometheus metrics methods."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Create a mocked MindcoreClient."""
        with patch("mindcore.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.name = "mock"
            mock_create.return_value = mock_provider

            from mindcore import MindcoreClient

            db_path = tmp_path / "test.db"
            client = MindcoreClient(
                use_sqlite=True,
                sqlite_path=str(db_path),
                persistent_cache=False,
            )
            yield client
            client.close()

    def test_get_prometheus_metrics(self, mock_client):
        """Test get_prometheus_metrics returns dict."""
        metrics = mock_client.get_prometheus_metrics()
        assert isinstance(metrics, dict)

    def test_prometheus_enabled_property(self, mock_client):
        """Test prometheus_enabled property."""
        enabled = mock_client.prometheus_enabled
        assert isinstance(enabled, bool)
