"""Tests for Context Lake."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from mindcore.context_lake.lake import (
    ContextLake,
    ContextLakeConfig,
    ContextQuery,
    ContextResult,
    ContextSource,
)
from mindcore.context_lake.knowledge_base import (
    Document,
    KnowledgeBase,
    KnowledgeBaseConfig,
    SearchResult,
)
from mindcore.context_lake.api_listener import (
    APIListener,
    APIListenerConfig,
    APIListenerRegistry,
    EventHandler,
    WebhookListener,
    PollingAPIListener,
)
from mindcore.core.schemas import Message, MessageMetadata, MessageRole


class TestContextQuery:
    """Tests for ContextQuery."""

    def test_default_values(self):
        """Test default query values."""
        query = ContextQuery(
            user_id="user123",
            query="What about my order?",
        )

        assert query.user_id == "user123"
        assert query.time_range == "recent"
        assert query.include_messages is True
        assert query.include_knowledge_base is True
        assert query.max_messages == 20

    def test_time_range_recent(self):
        """Test recent time range."""
        query = ContextQuery(
            user_id="user123",
            query="test",
            time_range="recent",
        )

        start, end = query.get_time_range()
        assert start is not None
        assert end is not None
        assert (end - start).total_seconds() <= 86400 + 1  # ~24 hours

    def test_time_range_all(self):
        """Test all time range."""
        query = ContextQuery(
            user_id="user123",
            query="test",
            time_range="all",
        )

        start, end = query.get_time_range()
        assert start is None
        assert end is None

    def test_custom_time_range(self):
        """Test custom time range."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        query = ContextQuery(
            user_id="user123",
            query="test",
            start_date=start,
            end_date=end,
        )

        result_start, result_end = query.get_time_range()
        assert result_start == start
        assert result_end == end


class TestContextResult:
    """Tests for ContextResult."""

    def test_default_values(self):
        """Test default result values."""
        result = ContextResult()

        assert result.messages == []
        assert result.knowledge_results == []
        assert result.total_items == 0

    def test_to_dict(self):
        """Test result serialization."""
        result = ContextResult(
            sources_queried=[ContextSource.MESSAGES, ContextSource.KNOWLEDGE_BASE],
            sources_with_data=[ContextSource.MESSAGES],
            total_items=5,
        )

        data = result.to_dict()

        assert data["total_items"] == 5
        assert "messages" in data["sources_queried"]
        assert "knowledge_base" in data["sources_queried"]


class TestDocument:
    """Tests for Document."""

    def test_auto_generate_id(self):
        """Test automatic ID generation."""
        doc = Document(
            content="Test content",
            source="test_source",
        )

        assert doc.doc_id is not None
        assert len(doc.doc_id) == 16

    def test_to_dict(self):
        """Test document serialization."""
        doc = Document(
            content="Test content",
            source="test_source",
            topics=["topic1", "topic2"],
            categories=["cat1"],
        )

        data = doc.to_dict()

        assert data["content"] == "Test content"
        assert data["source"] == "test_source"
        assert data["topics"] == ["topic1", "topic2"]

    def test_from_dict(self):
        """Test document deserialization."""
        data = {
            "doc_id": "test123",
            "content": "Test content",
            "source": "test_source",
            "topics": ["topic1"],
            "created_at": "2024-01-01T00:00:00+00:00",
        }

        doc = Document.from_dict(data)

        assert doc.doc_id == "test123"
        assert doc.content == "Test content"
        assert doc.topics == ["topic1"]


class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.add = MagicMock()
        store.query = MagicMock(return_value=[])
        store.delete = MagicMock()
        return store

    def test_add_document(self, mock_vector_store):
        """Test adding a document."""
        kb = KnowledgeBase(
            vector_store=mock_vector_store,
            config=KnowledgeBaseConfig(validate_vocabulary=False),
        )

        doc = Document(
            content="Short content",
            source="test",
            topics=["topic1"],
        )

        doc_id = kb.add(doc)

        assert doc_id is not None
        mock_vector_store.add.assert_called_once()

    def test_add_long_document_chunks(self, mock_vector_store):
        """Test that long documents are chunked."""
        kb = KnowledgeBase(
            vector_store=mock_vector_store,
            config=KnowledgeBaseConfig(
                chunk_size=100,
                chunk_overlap=20,
                validate_vocabulary=False,
            ),
        )

        long_content = "x" * 500  # 500 chars

        doc = Document(
            content=long_content,
            source="test",
        )

        kb.add(doc)

        # Should have multiple add calls for chunks
        assert mock_vector_store.add.call_count > 1

    def test_search(self, mock_vector_store):
        """Test searching the knowledge base."""
        mock_vector_store.query.return_value = [
            ({"content": "Result 1", "doc_id": "d1", "source": "test", "topics": []}, 0.2),
            ({"content": "Result 2", "doc_id": "d2", "source": "test", "topics": []}, 0.3),
        ]

        kb = KnowledgeBase(
            vector_store=mock_vector_store,
            config=KnowledgeBaseConfig(validate_vocabulary=False),
        )

        results = kb.search("test query", limit=5)

        assert len(results) == 2
        assert results[0].similarity > results[1].similarity


class TestAPIListenerConfig:
    """Tests for APIListenerConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = APIListenerConfig(
            name="test_listener",
            topics=["topic1"],
        )

        assert config.name == "test_listener"
        assert config.poll_interval == 60
        assert config.cache_ttl == 300
        assert config.enabled is True


class TestEventHandler:
    """Tests for EventHandler."""

    def test_register_and_process(self):
        """Test registering and processing events."""
        handler = EventHandler("test")

        processed = []

        def on_order_created(data):
            processed.append(data)
            return {"processed": True}

        handler.register("order.created", on_order_created)

        results = handler.process("order.created", {"order_id": "123"})

        assert len(processed) == 1
        assert processed[0]["order_id"] == "123"
        assert len(results) == 1
        assert results[0]["processed"] is True

    def test_unregistered_event(self):
        """Test processing unregistered event type."""
        handler = EventHandler("test")

        results = handler.process("unknown.event", {"data": "test"})

        assert results == []


class TestWebhookListener:
    """Tests for WebhookListener."""

    def test_process_webhook(self):
        """Test processing a webhook event."""
        config = APIListenerConfig(
            name="test_webhook",
            topics=["orders"],
        )

        listener = WebhookListener(config)

        event_data = {
            "type": "order.created",
            "user_id": "user123",
            "data": {"order_id": "ord_456"},
        }

        result = listener.process_webhook(event_data)

        assert result is not None
        assert result["type"] == "order.created"

    def test_fetch_buffered_events(self):
        """Test fetching buffered events."""
        config = APIListenerConfig(
            name="test_webhook",
            topics=["orders"],
        )

        listener = WebhookListener(config)

        # Process some events
        listener.process_webhook({
            "type": "order.created",
            "user_id": "user123",
        })

        data = listener.fetch("user123")

        assert data is not None
        assert data["event_count"] == 1


class TestAPIListenerRegistry:
    """Tests for APIListenerRegistry."""

    def test_register_listener(self):
        """Test registering a listener."""
        registry = APIListenerRegistry()

        config = APIListenerConfig(
            name="test_listener",
            topics=["orders", "billing"],
        )
        listener = WebhookListener(config)

        registry.register(listener)

        assert registry.get("test_listener") is not None

    def test_get_listeners_for_topics(self):
        """Test getting listeners by topic."""
        registry = APIListenerRegistry()

        config1 = APIListenerConfig(name="orders_listener", topics=["orders"])
        config2 = APIListenerConfig(name="billing_listener", topics=["billing"])

        registry.register(WebhookListener(config1))
        registry.register(WebhookListener(config2))

        listeners = registry.get_listeners_for_topics(["orders"])

        assert len(listeners) == 1
        assert listeners[0].name == "orders_listener"

    def test_get_listeners_for_multiple_topics(self):
        """Test getting listeners for multiple topics."""
        registry = APIListenerRegistry()

        config1 = APIListenerConfig(name="orders_listener", topics=["orders"])
        config2 = APIListenerConfig(name="billing_listener", topics=["billing"])

        registry.register(WebhookListener(config1))
        registry.register(WebhookListener(config2))

        listeners = registry.get_listeners_for_topics(["orders", "billing"])

        assert len(listeners) == 2


class TestContextLake:
    """Tests for ContextLake."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database manager."""
        db = MagicMock()
        db.search_messages = MagicMock(return_value=[])
        db.get_thread_summaries = MagicMock(return_value=[])
        db.get_user_preferences = MagicMock(return_value=None)
        return db

    @pytest.fixture
    def mock_kb(self):
        """Create mock knowledge base."""
        kb = MagicMock()
        kb.search = MagicMock(return_value=[])
        return kb

    def test_query_messages_only(self, mock_db):
        """Test querying only messages."""
        lake = ContextLake(db_manager=mock_db)

        query = ContextQuery(
            user_id="user123",
            query="test query",
            include_knowledge_base=False,
            include_external_data=False,
            include_api_data=False,
        )

        result = lake.query(query)

        assert ContextSource.MESSAGES in result.sources_queried
        mock_db.search_messages.assert_called()

    def test_query_with_knowledge_base(self, mock_db, mock_kb):
        """Test querying with knowledge base."""
        lake = ContextLake(
            db_manager=mock_db,
            knowledge_base=mock_kb,
        )

        query = ContextQuery(
            user_id="user123",
            query="test query",
        )

        result = lake.query(query)

        assert ContextSource.KNOWLEDGE_BASE in result.sources_queried
        mock_kb.search.assert_called()

    def test_aggregated_topics(self, mock_db):
        """Test topic aggregation from results."""
        msg = Message(
            message_id="msg1",
            user_id="user123",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="Test",
            metadata=MessageMetadata(topics=["orders", "billing"]),
        )

        mock_db.search_messages.return_value = [msg]

        lake = ContextLake(db_manager=mock_db)

        query = ContextQuery(
            user_id="user123",
            query="test",
        )

        result = lake.query(query)

        assert "orders" in result.aggregated_topics
        assert "billing" in result.aggregated_topics

    def test_get_condensed_context(self, mock_db):
        """Test generating condensed context."""
        msg = Message(
            message_id="msg1",
            user_id="user123",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="I need help with my order",
            metadata=MessageMetadata(),
        )

        result = ContextResult(
            messages=[msg],
            sources_with_data=[ContextSource.MESSAGES],
        )

        lake = ContextLake(db_manager=mock_db)

        condensed = lake.get_condensed_context(result, "test query")

        assert "Recent Messages" in condensed
        assert "order" in condensed

    def test_get_status(self, mock_db, mock_kb):
        """Test getting lake status."""
        lake = ContextLake(
            db_manager=mock_db,
            knowledge_base=mock_kb,
        )

        status = lake.get_status()

        assert status["sources_available"]["messages"] is True
        assert status["sources_available"]["knowledge_base"] is True
        assert status["sources_available"]["connectors"] is False
