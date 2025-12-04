"""Comprehensive tests for Mindcore core schemas.

These are CRITICAL data models used throughout the entire system.
Every module depends on Message, MessageMetadata, and related classes.
"""

from datetime import datetime, timezone

import pytest

from mindcore.core.schemas import (
    AssembledContext,
    ContextRequest,
    IngestRequest,
    KnowledgeVisibility,
    Message,
    MessageMetadata,
    MessageRole,
    MetadataSchema,
    ThreadSummary,
    UserPreferences,
)


class TestMetadataSchema:
    """Tests for MetadataSchema - predefined metadata for enrichment."""

    def test_default_topics(self):
        """Test default topics are populated."""
        schema = MetadataSchema()
        assert len(schema.topics) > 0
        assert "greeting" in schema.topics
        assert "billing" in schema.topics
        assert "api" in schema.topics

    def test_default_categories(self):
        """Test default categories are populated."""
        schema = MetadataSchema()
        assert len(schema.categories) > 0
        assert "support" in schema.categories
        assert "billing" in schema.categories
        assert "general" in schema.categories

    def test_default_intents(self):
        """Test default intents are populated."""
        schema = MetadataSchema()
        assert len(schema.intents) > 0
        assert "ask_question" in schema.intents
        assert "request_action" in schema.intents

    def test_default_sentiments(self):
        """Test default sentiments are populated."""
        schema = MetadataSchema()
        assert schema.sentiments == ["positive", "negative", "neutral", "mixed"]

    def test_to_prompt_list(self):
        """Test formatting schema for LLM prompts."""
        schema = MetadataSchema()
        prompt = schema.to_prompt_list()

        assert "Available Topics:" in prompt
        assert "Available Categories:" in prompt
        assert "Available Intents:" in prompt
        assert "Available Sentiments:" in prompt
        assert "greeting" in prompt

    def test_validate_topics_filters_invalid(self):
        """Test topic validation filters out invalid topics."""
        schema = MetadataSchema()

        topics = ["greeting", "invalid_topic", "billing", "another_invalid"]
        valid = schema.validate_topics(topics)

        assert "greeting" in valid
        assert "billing" in valid
        assert "invalid_topic" not in valid
        assert "another_invalid" not in valid

    def test_validate_topics_case_insensitive(self):
        """Test topic validation is case-insensitive."""
        schema = MetadataSchema()

        topics = ["GREETING", "Billing", "API"]
        valid = schema.validate_topics(topics)

        # Should match case-insensitively
        assert len(valid) >= 1

    def test_validate_categories_filters_invalid(self):
        """Test category validation filters out invalid categories."""
        schema = MetadataSchema()

        categories = ["support", "invalid_cat", "billing"]
        valid = schema.validate_categories(categories)

        assert "support" in valid
        assert "billing" in valid
        assert "invalid_cat" not in valid

    def test_validate_intent_valid(self):
        """Test intent validation for valid intents."""
        schema = MetadataSchema()

        assert schema.validate_intent("ask_question") == "ask_question"
        assert schema.validate_intent("request_action") == "request_action"

    def test_validate_intent_case_insensitive(self):
        """Test intent validation is case-insensitive."""
        schema = MetadataSchema()

        assert schema.validate_intent("ASK_QUESTION") == "ask_question"
        assert schema.validate_intent("Request_Action") == "request_action"

    def test_validate_intent_invalid_returns_none(self):
        """Test intent validation returns None for invalid intents."""
        schema = MetadataSchema()

        assert schema.validate_intent("invalid_intent") is None
        assert schema.validate_intent("") is None
        assert schema.validate_intent(None) is None


class TestMessageRole:
    """Tests for MessageRole enumeration."""

    def test_role_values(self):
        """Test message role values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.TOOL.value == "tool"

    def test_role_is_string_enum(self):
        """Test MessageRole is a string enum."""
        assert isinstance(MessageRole.USER, str)
        assert MessageRole.USER == "user"


class TestMessageMetadata:
    """Tests for MessageMetadata dataclass."""

    def test_default_values(self):
        """Test default metadata values."""
        metadata = MessageMetadata()

        assert metadata.topics == []
        assert metadata.categories == []
        assert metadata.importance == 0.5
        assert metadata.sentiment == {}
        assert metadata.intent is None
        assert metadata.tags == []
        assert metadata.entities == []
        assert metadata.key_phrases == []
        assert metadata.enrichment_failed is False
        assert metadata.enrichment_error is None

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = MessageMetadata(
            topics=["billing", "orders"],
            categories=["support"],
            importance=0.8,
            intent="ask_question",
        )

        result = metadata.to_dict()

        assert result["topics"] == ["billing", "orders"]
        assert result["categories"] == ["support"]
        assert result["importance"] == 0.8
        assert result["intent"] == "ask_question"

    def test_is_enriched_true_with_topics(self):
        """Test is_enriched returns True when topics present."""
        metadata = MessageMetadata(topics=["billing"])
        assert metadata.is_enriched is True

    def test_is_enriched_true_with_categories(self):
        """Test is_enriched returns True when categories present."""
        metadata = MessageMetadata(categories=["support"])
        assert metadata.is_enriched is True

    def test_is_enriched_true_with_intent(self):
        """Test is_enriched returns True when intent present."""
        metadata = MessageMetadata(intent="ask_question")
        assert metadata.is_enriched is True

    def test_is_enriched_true_with_tags(self):
        """Test is_enriched returns True when tags present."""
        metadata = MessageMetadata(tags=["important"])
        assert metadata.is_enriched is True

    def test_is_enriched_true_with_entities(self):
        """Test is_enriched returns True when entities present."""
        metadata = MessageMetadata(entities=["order_123"])
        assert metadata.is_enriched is True

    def test_is_enriched_false_when_empty(self):
        """Test is_enriched returns False when no enrichment."""
        metadata = MessageMetadata()
        assert metadata.is_enriched is False

    def test_is_enriched_false_when_failed(self):
        """Test is_enriched returns False when enrichment failed."""
        metadata = MessageMetadata(
            topics=["billing"],  # Has topics but...
            enrichment_failed=True,  # ...enrichment failed
        )
        assert metadata.is_enriched is False


class TestKnowledgeVisibility:
    """Tests for KnowledgeVisibility enum."""

    def test_visibility_values(self):
        """Test knowledge visibility values."""
        assert KnowledgeVisibility.PRIVATE.value == "private"
        assert KnowledgeVisibility.SHARED.value == "shared"
        assert KnowledgeVisibility.PUBLIC.value == "public"


class TestMessage:
    """Tests for Message dataclass - the core data structure."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Hello, how can I help you?",
        )

        assert msg.message_id == "msg_001"
        assert msg.user_id == "user_123"
        assert msg.thread_id == "thread_456"
        assert msg.session_id == "session_789"
        assert msg.role == MessageRole.USER
        assert msg.raw_text == "Hello, how can I help you?"

    def test_message_default_metadata(self):
        """Test message has default empty metadata."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Test",
        )

        assert isinstance(msg.metadata, MessageMetadata)
        assert msg.metadata.topics == []

    def test_message_auto_timestamp(self):
        """Test message gets automatic timestamp if not provided."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Test",
        )

        assert msg.created_at is not None
        assert isinstance(msg.created_at, datetime)

    def test_message_role_from_string(self):
        """Test message converts string role to enum."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role="user",  # String, not enum
            raw_text="Test",
        )

        assert msg.role == MessageRole.USER
        assert isinstance(msg.role, MessageRole)

    def test_message_metadata_from_dict(self):
        """Test message converts dict metadata to MessageMetadata."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Test",
            metadata={"topics": ["billing"], "importance": 0.9},
        )

        assert isinstance(msg.metadata, MessageMetadata)
        assert msg.metadata.topics == ["billing"]
        assert msg.metadata.importance == 0.9

    def test_message_multi_agent_defaults(self):
        """Test message has default multi-agent values."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Test",
        )

        assert msg.agent_id is None
        assert msg.visibility == "private"
        assert msg.sharing_groups == []

    def test_message_with_multi_agent_fields(self):
        """Test message with multi-agent configuration."""
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Test",
            agent_id="support_bot",
            visibility="shared",
            sharing_groups=["support", "sales"],
        )

        assert msg.agent_id == "support_bot"
        assert msg.visibility == "shared"
        assert msg.sharing_groups == ["support", "sales"]

    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        now = datetime.now(timezone.utc)
        msg = Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Hello!",
            metadata=MessageMetadata(topics=["greeting"]),
            created_at=now,
            agent_id="bot_1",
            visibility="public",
            sharing_groups=["team_a"],
        )

        result = msg.to_dict()

        assert result["message_id"] == "msg_001"
        assert result["user_id"] == "user_123"
        assert result["role"] == "user"
        assert result["raw_text"] == "Hello!"
        assert result["metadata"]["topics"] == ["greeting"]
        assert result["created_at"] == now.isoformat()
        assert result["agent_id"] == "bot_1"
        assert result["visibility"] == "public"
        assert result["sharing_groups"] == ["team_a"]


class TestAssembledContext:
    """Tests for AssembledContext dataclass."""

    def test_assembled_context_creation(self):
        """Test creating assembled context."""
        ctx = AssembledContext(
            assembled_context="User asked about billing. Previous messages show concern about charges.",
            key_points=["billing inquiry", "charge dispute"],
            relevant_message_ids=["msg_001", "msg_002"],
            metadata={"confidence": 0.95},
        )

        assert "billing" in ctx.assembled_context
        assert len(ctx.key_points) == 2
        assert len(ctx.relevant_message_ids) == 2
        assert ctx.metadata["confidence"] == 0.95

    def test_assembled_context_to_dict(self):
        """Test assembled context serialization."""
        ctx = AssembledContext(
            assembled_context="Context here",
            key_points=["point1"],
            relevant_message_ids=["msg_001"],
            metadata={"source": "test"},
        )

        result = ctx.to_dict()

        assert result["assembled_context"] == "Context here"
        assert result["key_points"] == ["point1"]
        assert result["relevant_message_ids"] == ["msg_001"]
        assert result["metadata"]["source"] == "test"


class TestContextRequest:
    """Tests for ContextRequest dataclass."""

    def test_context_request_creation(self):
        """Test creating context request."""
        req = ContextRequest(
            user_id="user_123",
            thread_id="thread_456",
            query="What about my order?",
        )

        assert req.user_id == "user_123"
        assert req.thread_id == "thread_456"
        assert req.query == "What about my order?"
        assert req.max_messages == 50  # Default
        assert req.include_metadata is True  # Default

    def test_context_request_custom_limits(self):
        """Test context request with custom limits."""
        req = ContextRequest(
            user_id="user_123",
            thread_id="thread_456",
            query="Query",
            max_messages=100,
            include_metadata=False,
        )

        assert req.max_messages == 100
        assert req.include_metadata is False

    def test_context_request_to_dict(self):
        """Test context request serialization."""
        req = ContextRequest(
            user_id="user_123",
            thread_id="thread_456",
            query="Test query",
        )

        result = req.to_dict()

        assert result["user_id"] == "user_123"
        assert result["thread_id"] == "thread_456"
        assert result["query"] == "Test query"


class TestIngestRequest:
    """Tests for IngestRequest dataclass."""

    def test_ingest_request_creation(self):
        """Test creating ingest request."""
        req = IngestRequest(
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role="user",
            text="Hello world!",
        )

        assert req.user_id == "user_123"
        assert req.thread_id == "thread_456"
        assert req.session_id == "session_789"
        assert req.role == "user"
        assert req.text == "Hello world!"
        assert req.message_id is None  # Optional

    def test_ingest_request_with_message_id(self):
        """Test ingest request with custom message ID."""
        req = IngestRequest(
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role="assistant",
            text="Response",
            message_id="custom_msg_001",
        )

        assert req.message_id == "custom_msg_001"

    def test_ingest_request_to_dict(self):
        """Test ingest request serialization."""
        req = IngestRequest(
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role="user",
            text="Test",
        )

        result = req.to_dict()

        assert result["user_id"] == "user_123"
        assert result["text"] == "Test"


class TestThreadSummary:
    """Tests for ThreadSummary dataclass."""

    def test_thread_summary_creation(self):
        """Test creating thread summary."""
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
            summary="User inquired about order status and delivery.",
            key_facts=["Order #12345", "Expected delivery: March 15"],
            topics=["orders", "delivery"],
            categories=["support"],
            overall_sentiment="neutral",
            message_count=15,
        )

        assert summary.summary_id == "sum_001"
        assert "order status" in summary.summary
        assert len(summary.key_facts) == 2
        assert "orders" in summary.topics

    def test_thread_summary_auto_timestamp(self):
        """Test thread summary gets automatic summarized_at timestamp."""
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
        )

        assert summary.summarized_at is not None
        assert isinstance(summary.summarized_at, datetime)

    def test_thread_summary_defaults(self):
        """Test thread summary default values."""
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
        )

        assert summary.session_id is None
        assert summary.summary == ""
        assert summary.key_facts == []
        assert summary.topics == []
        assert summary.categories == []
        assert summary.overall_sentiment == "neutral"
        assert summary.message_count == 0
        assert summary.entities == {}
        assert summary.messages_deleted is False

    def test_thread_summary_to_dict(self):
        """Test thread summary serialization."""
        now = datetime.now(timezone.utc)
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
            summary="Test summary",
            key_facts=["fact1"],
            topics=["topic1"],
            summarized_at=now,
            first_message_at=now,
            last_message_at=now,
        )

        result = summary.to_dict()

        assert result["summary_id"] == "sum_001"
        assert result["summary"] == "Test summary"
        assert result["key_facts"] == ["fact1"]
        assert result["topics"] == ["topic1"]
        assert result["summarized_at"] == now.isoformat()

    def test_thread_summary_to_context_string(self):
        """Test formatting thread summary for AI context."""
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
            summary="User discussed order issues.",
            key_facts=["Order delayed", "Refund requested"],
            topics=["orders", "refund"],
        )

        context = summary.to_context_string()

        assert "Summary: User discussed order issues." in context
        assert "Key facts:" in context
        assert "Order delayed" in context
        assert "Topics discussed:" in context
        assert "orders" in context

    def test_thread_summary_to_context_string_minimal(self):
        """Test context string with minimal data."""
        summary = ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
            summary="Basic summary",
        )

        context = summary.to_context_string()

        assert "Summary: Basic summary" in context
        # No key facts or topics lines when empty


class TestUserPreferences:
    """Tests for UserPreferences dataclass."""

    def test_user_preferences_creation(self):
        """Test creating user preferences."""
        prefs = UserPreferences(user_id="user_123")

        assert prefs.user_id == "user_123"
        assert prefs.language == "en"
        assert prefs.timezone == "UTC"
        assert prefs.communication_style == "balanced"

    def test_user_preferences_defaults(self):
        """Test user preferences default values."""
        prefs = UserPreferences(user_id="user_123")

        assert prefs.interests == []
        assert prefs.goals == []
        assert prefs.preferred_name is None
        assert prefs.custom_context == {}
        assert prefs.notification_topics == []

    def test_user_preferences_auto_timestamps(self):
        """Test user preferences gets automatic timestamps."""
        prefs = UserPreferences(user_id="user_123")

        assert prefs.created_at is not None
        assert prefs.updated_at is not None
        assert prefs.created_at == prefs.updated_at

    def test_user_preferences_amendable_fields(self):
        """Test AMENDABLE_FIELDS class variable."""
        expected = {
            "language",
            "timezone",
            "communication_style",
            "interests",
            "goals",
            "preferred_name",
            "custom_context",
            "notification_topics",
        }
        assert UserPreferences.AMENDABLE_FIELDS == expected

    def test_update_amendable_field(self):
        """Test updating an amendable field."""
        prefs = UserPreferences(user_id="user_123")
        original_updated = prefs.updated_at

        result = prefs.update("language", "es")

        assert result is True
        assert prefs.language == "es"
        assert prefs.updated_at > original_updated

    def test_update_non_amendable_field_fails(self):
        """Test updating a non-amendable field returns False."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.update("user_id", "new_user")

        assert result is False
        assert prefs.user_id == "user_123"  # Unchanged

    def test_add_to_list_interests(self):
        """Test adding to interests list."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.add_to_list("interests", "AI")

        assert result is True
        assert "AI" in prefs.interests

    def test_add_to_list_goals(self):
        """Test adding to goals list."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.add_to_list("goals", "learn programming")

        assert result is True
        assert "learn programming" in prefs.goals

    def test_add_to_list_notification_topics(self):
        """Test adding to notification_topics list."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.add_to_list("notification_topics", "orders")

        assert result is True
        assert "orders" in prefs.notification_topics

    def test_add_to_list_invalid_field_fails(self):
        """Test adding to non-list field returns False."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.add_to_list("language", "value")

        assert result is False

    def test_add_to_list_no_duplicates(self):
        """Test adding duplicate value doesn't create duplicates."""
        prefs = UserPreferences(user_id="user_123", interests=["AI"])

        prefs.add_to_list("interests", "AI")

        assert prefs.interests.count("AI") == 1

    def test_remove_from_list(self):
        """Test removing from a list field."""
        prefs = UserPreferences(user_id="user_123", interests=["AI", "ML"])

        result = prefs.remove_from_list("interests", "AI")

        assert result is True
        assert "AI" not in prefs.interests
        assert "ML" in prefs.interests

    def test_remove_from_list_nonexistent_value(self):
        """Test removing nonexistent value returns False."""
        prefs = UserPreferences(user_id="user_123", interests=["AI"])

        result = prefs.remove_from_list("interests", "ML")

        assert result is False

    def test_remove_from_list_invalid_field_fails(self):
        """Test removing from non-list field returns False."""
        prefs = UserPreferences(user_id="user_123")

        result = prefs.remove_from_list("language", "en")

        assert result is False

    def test_to_dict(self):
        """Test user preferences serialization."""
        prefs = UserPreferences(
            user_id="user_123",
            language="es",
            preferred_name="John",
            interests=["AI"],
        )

        result = prefs.to_dict()

        assert result["user_id"] == "user_123"
        assert result["language"] == "es"
        assert result["preferred_name"] == "John"
        assert result["interests"] == ["AI"]
        assert "created_at" in result
        assert "updated_at" in result

    def test_to_context_string_full(self):
        """Test formatting preferences for AI context."""
        prefs = UserPreferences(
            user_id="user_123",
            preferred_name="John",
            language="es",
            communication_style="formal",
            interests=["AI", "ML"],
            goals=["learn programming"],
            custom_context={"role": "developer"},
        )

        context = prefs.to_context_string()

        assert "User prefers to be called: John" in context
        assert "Preferred language: es" in context
        assert "Communication style: formal" in context
        assert "Interests: AI, ML" in context
        assert "Goals: learn programming" in context
        assert "role: developer" in context

    def test_to_context_string_minimal(self):
        """Test context string with default preferences is empty."""
        prefs = UserPreferences(user_id="user_123")

        context = prefs.to_context_string()

        # Default English, balanced style = no special context needed
        assert context == ""

    def test_to_context_string_partial(self):
        """Test context string with some preferences."""
        prefs = UserPreferences(
            user_id="user_123",
            preferred_name="Jane",
            # Keep defaults for language and style
        )

        context = prefs.to_context_string()

        assert "User prefers to be called: Jane" in context
        assert "Preferred language:" not in context  # Default English
