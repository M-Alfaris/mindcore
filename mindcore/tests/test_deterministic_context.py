"""Tests for DeterministicContextAgent."""

import pytest

from mindcore.agents.deterministic_context_agent import (
    ContextToolCallbacks,
    DeterministicContextAgent,
    FetchedData,
    QueryIntent,
)
from mindcore.core.schemas import AssembledContext, Message, MessageMetadata, MessageRole


class TestQueryIntent:
    """Tests for QueryIntent dataclass."""

    def test_default_values(self):
        """Test default QueryIntent values."""
        intent = QueryIntent()

        assert intent.topics == []
        assert intent.categories == []
        assert intent.time_range == "recent"
        assert intent.needs_external_data is False
        assert intent.needs_user_preferences is False
        assert intent.needs_summaries is False
        assert intent.confidence == 0.0

    def test_custom_values(self):
        """Test QueryIntent with custom values."""
        intent = QueryIntent(
            topics=["orders", "billing"],
            categories=["support"],
            time_range="historical",
            needs_external_data=True,
            extracted_entities={"order_id": "12345"},
            confidence=0.8,
        )

        assert intent.topics == ["orders", "billing"]
        assert intent.needs_external_data is True
        assert intent.extracted_entities["order_id"] == "12345"


class TestFetchedData:
    """Tests for FetchedData dataclass."""

    def test_default_values(self):
        """Test default FetchedData values."""
        data = FetchedData()

        assert data.recent_messages == []
        assert data.historical_messages == []
        assert data.summaries == []
        assert data.user_preferences is None
        assert data.external_data == []
        assert data.fetch_errors == []

    def test_with_messages(self):
        """Test FetchedData with messages."""
        msg = Message(
            message_id="msg1",
            user_id="user1",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="Hello",
            metadata=MessageMetadata(),
        )

        data = FetchedData(
            recent_messages=[msg],
            fetch_latency_ms=50.0,
        )

        assert len(data.recent_messages) == 1
        assert data.fetch_latency_ms == 50.0


class TestDeterministicContextAgentFastPath:
    """Tests for fast path pattern matching."""

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks."""
        def get_recent(user_id, thread_id, limit):
            return []

        def search_history(user_id, thread_id, topics, categories, intent, limit):
            return []

        def get_metadata(user_id, thread_id):
            return {}

        return ContextToolCallbacks(
            get_recent_messages=get_recent,
            search_history=search_history,
            get_session_metadata=get_metadata,
        )

    def test_order_query_fast_path(self, mock_callbacks):
        """Test fast path for order queries."""
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.vocabulary = MagicMock()
        agent.vocabulary.validate_topics = lambda x: x
        agent.callbacks = mock_callbacks

        # Test order pattern
        intent = agent._try_fast_path("Where is my order #12345?")
        assert intent is not None
        assert "orders" in intent.topics or "tracking" in intent.topics
        assert intent.needs_external_data is True

    def test_billing_query_fast_path(self, mock_callbacks):
        """Test fast path for billing queries."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.vocabulary = MagicMock()
        agent.vocabulary.validate_topics = lambda x: x
        agent.callbacks = mock_callbacks

        intent = agent._try_fast_path("I have a question about my billing")
        assert intent is not None
        assert intent.needs_external_data is True

    def test_historical_query_fast_path(self, mock_callbacks):
        """Test fast path for historical queries."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.vocabulary = MagicMock()
        agent.vocabulary.validate_topics = lambda x: x
        agent.callbacks = mock_callbacks

        intent = agent._try_fast_path("What did we discuss last time?")
        assert intent is not None
        assert intent.time_range == "historical"
        assert intent.needs_summaries is True

    def test_no_fast_path_match(self, mock_callbacks):
        """Test when no fast path matches."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.vocabulary = MagicMock()
        agent.vocabulary.validate_topics = lambda x: x
        agent.callbacks = mock_callbacks

        intent = agent._try_fast_path("What is the meaning of life?")
        assert intent is None


class TestEntityExtraction:
    """Tests for fast entity extraction."""

    def test_extract_order_id(self):
        """Test order ID extraction."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)

        entities = agent._extract_entities_fast("My order #ABC-12345 hasn't arrived")
        assert "order_id" in entities
        assert entities["order_id"] == "ABC-12345"

    def test_extract_invoice_id(self):
        """Test invoice ID extraction."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)

        entities = agent._extract_entities_fast("I need invoice INV-98765")
        assert "invoice_id" in entities
        assert entities["invoice_id"] == "INV-98765"

    def test_extract_date(self):
        """Test date extraction."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)

        entities = agent._extract_entities_fast("I ordered on 2024-03-15")
        assert "date" in entities
        assert entities["date"] == "2024-03-15"

    def test_no_entities(self):
        """Test when no entities are found."""
        from unittest.mock import MagicMock

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)

        entities = agent._extract_entities_fast("Hello, how are you?")
        assert entities == {}


class TestParallelFetching:
    """Tests for parallel data fetching."""

    def test_fetch_data_parallel(self):
        """Test parallel data fetching."""
        from unittest.mock import MagicMock

        # Create mock message
        msg = Message(
            message_id="msg1",
            user_id="user1",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="Hello",
            metadata=MessageMetadata(),
        )

        callbacks = ContextToolCallbacks(
            get_recent_messages=lambda u, t, l: [msg],
            search_history=lambda u, t, topics, cats, intent, l: [],
            get_session_metadata=lambda u, t: {"topics": ["general"]},
        )

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.callbacks = callbacks
        agent.parallel_workers = 2

        intent = QueryIntent(topics=["general"], time_range="recent")

        data = agent._fetch_data_parallel(
            user_id="user1",
            thread_id="thread1",
            intent=intent,
        )

        assert len(data.recent_messages) == 1
        assert data.session_metadata.get("topics") == ["general"]
        assert data.fetch_latency_ms > 0

    def test_fetch_with_errors(self):
        """Test parallel fetching handles errors gracefully."""
        def failing_fetch(user_id, thread_id, limit):
            raise Exception("Fetch failed")

        callbacks = ContextToolCallbacks(
            get_recent_messages=failing_fetch,
            search_history=lambda u, t, topics, cats, intent, l: [],
            get_session_metadata=lambda u, t: {},
        )

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        agent.callbacks = callbacks
        agent.parallel_workers = 2

        intent = QueryIntent(topics=["general"])

        data = agent._fetch_data_parallel(
            user_id="user1",
            thread_id="thread1",
            intent=intent,
        )

        # Should have error recorded but not crash
        assert len(data.fetch_errors) > 0
        assert data.recent_messages == []


class TestContextGeneration:
    """Tests for context generation."""

    def test_format_fetched_data(self):
        """Test formatting fetched data for LLM."""
        msg = Message(
            message_id="msg1",
            user_id="user1",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="Hello, I need help with my order",
            metadata=MessageMetadata(topics=["orders"]),
        )

        data = FetchedData(
            recent_messages=[msg],
            session_metadata={"topics": ["orders"], "message_count": 5},
        )

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        formatted = agent._format_fetched_data(data)

        assert "Recent Messages" in formatted
        assert "Hello, I need help with my order" in formatted
        assert "Session Info" in formatted

    def test_fallback_context(self):
        """Test fallback context creation."""
        msg = Message(
            message_id="msg1",
            user_id="user1",
            thread_id="thread1",
            session_id="session1",
            role=MessageRole.USER,
            raw_text="Hello",
            metadata=MessageMetadata(),
        )

        data = FetchedData(recent_messages=[msg])

        agent = DeterministicContextAgent.__new__(DeterministicContextAgent)
        context = agent._create_fallback_context("test query", data)

        assert isinstance(context, AssembledContext)
        assert context.metadata["context_source"] == "fallback"
        assert context.metadata["confidence"] == "low"
