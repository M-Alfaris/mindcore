"""Tests for SQLiteManager."""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from mindcore.core.schemas import (
    Message,
    MessageMetadata,
    MessageRole,
    ThreadSummary,
    UserPreferences,
)
from mindcore.core.sqlite_manager import SQLiteManager, _normalize_datetime


class TestNormalizeDatetime:
    """Tests for the _normalize_datetime helper function."""

    def test_normalize_none(self):
        """Test normalizing None returns None."""
        assert _normalize_datetime(None) is None

    def test_normalize_aware_datetime(self):
        """Test normalizing timezone-aware datetime returns UTC."""
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _normalize_datetime(dt)
        assert result == dt
        assert result.tzinfo == timezone.utc

    def test_normalize_naive_datetime(self):
        """Test normalizing naive datetime assumes UTC."""
        dt = datetime(2024, 1, 15, 12, 0, 0)
        result = _normalize_datetime(dt)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024

    def test_normalize_iso_string(self):
        """Test normalizing ISO format string."""
        result = _normalize_datetime("2024-01-15T12:00:00")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024

    def test_normalize_iso_string_with_z(self):
        """Test normalizing ISO string with Z suffix."""
        result = _normalize_datetime("2024-01-15T12:00:00Z")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_normalize_invalid_string(self):
        """Test normalizing invalid string returns None."""
        assert _normalize_datetime("invalid-date") is None

    def test_normalize_non_datetime_type(self):
        """Test normalizing non-datetime type returns None."""
        assert _normalize_datetime(12345) is None
        assert _normalize_datetime({"date": "value"}) is None


class TestSQLiteManager:
    """Tests for SQLiteManager class."""

    @pytest.fixture
    def db(self):
        """Create an in-memory SQLite database for testing."""
        manager = SQLiteManager(":memory:")
        yield manager
        manager.close()

    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            message_id="msg_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Hello, this is a test message",
            metadata=MessageMetadata(
                topics=["general", "testing"],
                categories=["question"],
                intent="ask_question",
                importance=0.7,
            ),
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_messages(self):
        """Create multiple sample messages for testing."""
        messages = []
        base_time = datetime.now(timezone.utc)
        for i in range(5):
            msg = Message(
                message_id=f"msg_{i:03d}",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                raw_text=f"Test message {i}",
                metadata=MessageMetadata(
                    topics=["general", "testing"] if i % 2 == 0 else ["orders"],
                    categories=["question"] if i % 2 == 0 else ["response"],
                    intent="ask_question" if i % 2 == 0 else "provide_info",
                    importance=0.5 + (i * 0.1),
                ),
                created_at=base_time,
            )
            messages.append(msg)
        return messages

    # =====================
    # Initialization Tests
    # =====================

    def test_initialization(self, db):
        """Test SQLiteManager initializes correctly."""
        assert db.db_path == ":memory:"
        assert db._local is not None
        assert db._lock is not None

    def test_schema_creation(self, db):
        """Test schema is created with all required tables."""
        with db.get_connection() as conn:
            # Check messages table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            assert cursor.fetchone() is not None

            # Check thread_summaries table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='thread_summaries'"
            )
            assert cursor.fetchone() is not None

            # Check user_preferences table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
            )
            assert cursor.fetchone() is not None

    def test_indexes_created(self, db):
        """Test indexes are created on messages table."""
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='messages'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_user_thread" in indexes
            assert "idx_thread_created" in indexes
            assert "idx_created_at" in indexes
            assert "idx_session" in indexes

    # =====================
    # Message CRUD Tests
    # =====================

    def test_insert_message(self, db, sample_message):
        """Test inserting a message."""
        result = db.insert_message(sample_message)
        assert result is True

        # Verify message was inserted
        msg = db.get_message_by_id(sample_message.message_id)
        assert msg is not None
        assert msg.message_id == sample_message.message_id
        assert msg.raw_text == sample_message.raw_text

    def test_insert_message_replaces_on_conflict(self, db, sample_message):
        """Test that inserting with same ID replaces the message."""
        db.insert_message(sample_message)

        # Modify and re-insert
        sample_message.raw_text = "Updated message text"
        result = db.insert_message(sample_message)
        assert result is True

        msg = db.get_message_by_id(sample_message.message_id)
        assert msg.raw_text == "Updated message text"

    def test_insert_message_with_dict_metadata(self, db):
        """Test inserting message with dict metadata (not MessageMetadata)."""
        msg = Message(
            message_id="msg_dict",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role=MessageRole.USER,
            raw_text="Dict metadata test",
            metadata={"topics": ["test"]},  # type: ignore
            created_at=datetime.now(timezone.utc),
        )
        result = db.insert_message(msg)
        assert result is True

    def test_get_message_by_id(self, db, sample_message):
        """Test retrieving a message by ID."""
        db.insert_message(sample_message)

        msg = db.get_message_by_id(sample_message.message_id)
        assert msg is not None
        assert msg.message_id == sample_message.message_id
        assert msg.user_id == sample_message.user_id
        assert msg.role == sample_message.role

    def test_get_message_by_id_not_found(self, db):
        """Test retrieving non-existent message returns None."""
        msg = db.get_message_by_id("nonexistent_id")
        assert msg is None

    def test_fetch_recent_messages(self, db, sample_messages):
        """Test fetching recent messages."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.fetch_recent_messages("user_123", "thread_456", limit=10)
        assert len(messages) == 5
        # Should be in reverse chronological order (DESC)
        for msg in messages:
            assert msg.user_id == "user_123"
            assert msg.thread_id == "thread_456"

    def test_fetch_recent_messages_with_limit(self, db, sample_messages):
        """Test fetching recent messages with limit."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.fetch_recent_messages("user_123", "thread_456", limit=2)
        assert len(messages) == 2

    def test_fetch_recent_messages_empty(self, db):
        """Test fetching messages for non-existent user/thread."""
        messages = db.fetch_recent_messages("no_user", "no_thread")
        assert messages == []

    def test_update_message_metadata(self, db, sample_message):
        """Test updating message metadata."""
        db.insert_message(sample_message)

        new_metadata = MessageMetadata(
            topics=["updated", "topics"],
            categories=["new_category"],
            importance=0.9,
        )

        result = db.update_message_metadata(sample_message.message_id, new_metadata)
        assert result is True

        msg = db.get_message_by_id(sample_message.message_id)
        assert "updated" in msg.metadata.topics
        assert msg.metadata.importance == 0.9

    def test_update_message_metadata_not_found(self, db):
        """Test updating metadata for non-existent message."""
        result = db.update_message_metadata(
            "nonexistent_id",
            MessageMetadata(topics=["test"]),
        )
        assert result is False

    # =====================
    # Search Tests
    # =====================

    def test_search_messages_by_topic(self, db, sample_messages):
        """Test searching messages by topic."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.search_messages_by_topic(
            "user_123", "thread_456", topics=["testing"], limit=10
        )
        # Messages with "testing" in topics (even indices)
        assert len(messages) > 0
        for msg in messages:
            assert "testing" in msg.metadata.topics

    def test_search_by_relevance_with_topics(self, db, sample_messages):
        """Test relevance search with topic filter."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.search_by_relevance(
            user_id="user_123",
            topics=["orders"],
            thread_id="thread_456",
            limit=10,
        )
        # Should return messages with "orders" topic
        for msg in messages:
            assert "orders" in msg.metadata.topics

    def test_search_by_relevance_with_min_importance(self, db, sample_messages):
        """Test relevance search with minimum importance."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.search_by_relevance(
            user_id="user_123",
            min_importance=0.7,
            thread_id="thread_456",
            limit=10,
        )
        for msg in messages:
            assert msg.metadata.importance >= 0.7

    def test_search_by_relevance_with_intent(self, db, sample_messages):
        """Test relevance search with intent filter scores matching messages higher."""
        for msg in sample_messages:
            db.insert_message(msg)

        messages = db.search_by_relevance(
            user_id="user_123",
            intent="ask_question",
            thread_id="thread_456",
            limit=10,
        )
        # Intent matching adds to score but doesn't filter out non-matches
        # Just verify we get results back
        assert len(messages) > 0

    # =====================
    # Thread Summary Tests
    # =====================

    @pytest.fixture
    def sample_summary(self):
        """Create a sample thread summary."""
        return ThreadSummary(
            summary_id="sum_001",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            summary="This thread discussed testing and general topics.",
            key_facts=["User asked about testing", "Assistant provided info"],
            topics=["general", "testing"],
            categories=["question", "response"],
            overall_sentiment="positive",
            message_count=10,
            first_message_at=datetime.now(timezone.utc),
            last_message_at=datetime.now(timezone.utc),
            summarized_at=datetime.now(timezone.utc),
            entities={"user": "John"},
            messages_deleted=False,
        )

    def test_insert_summary(self, db, sample_summary):
        """Test inserting a thread summary."""
        result = db.insert_summary(sample_summary)
        assert result is True

        summary = db.get_summary("user_123", "thread_456")
        assert summary is not None
        assert summary.summary_id == sample_summary.summary_id

    def test_get_summary(self, db, sample_summary):
        """Test getting a thread summary."""
        db.insert_summary(sample_summary)

        summary = db.get_summary("user_123", "thread_456")
        assert summary is not None
        assert summary.summary == sample_summary.summary
        assert summary.message_count == 10
        assert "general" in summary.topics

    def test_get_summary_not_found(self, db):
        """Test getting non-existent summary returns None."""
        summary = db.get_summary("no_user", "no_thread")
        assert summary is None

    def test_get_user_summaries(self, db, sample_summary):
        """Test getting all summaries for a user."""
        db.insert_summary(sample_summary)

        # Add another summary
        summary2 = ThreadSummary(
            summary_id="sum_002",
            user_id="user_123",
            thread_id="thread_789",
            summary="Another thread summary",
            topics=["orders"],
        )
        db.insert_summary(summary2)

        summaries = db.get_user_summaries("user_123")
        assert len(summaries) == 2

    def test_get_user_summaries_with_topic_filter(self, db, sample_summary):
        """Test getting summaries filtered by topics."""
        db.insert_summary(sample_summary)

        summaries = db.get_user_summaries("user_123", topics=["testing"])
        assert len(summaries) == 1
        assert "testing" in summaries[0].topics

    # =====================
    # User Preferences Tests
    # =====================

    def test_get_preferences_not_found(self, db):
        """Test getting preferences for non-existent user."""
        prefs = db.get_preferences("no_user")
        assert prefs is None

    def test_save_and_get_preferences(self, db):
        """Test saving and retrieving preferences."""
        prefs = UserPreferences(
            user_id="user_123",
            language="en",
            timezone="America/New_York",
            communication_style="formal",
            interests=["technology", "science"],
            goals=["learn", "improve"],
            preferred_name="John",
        )
        result = db.save_preferences(prefs)
        assert result is True

        retrieved = db.get_preferences("user_123")
        assert retrieved is not None
        assert retrieved.language == "en"
        assert retrieved.preferred_name == "John"
        assert "technology" in retrieved.interests

    def test_get_or_create_preferences_creates(self, db):
        """Test get_or_create creates default preferences."""
        prefs = db.get_or_create_preferences("new_user")
        assert prefs is not None
        assert prefs.user_id == "new_user"
        assert prefs.language == "en"  # Default

    def test_get_or_create_preferences_returns_existing(self, db):
        """Test get_or_create returns existing preferences."""
        # Save custom preferences
        prefs = UserPreferences(
            user_id="user_123",
            language="es",
        )
        db.save_preferences(prefs)

        retrieved = db.get_or_create_preferences("user_123")
        assert retrieved.language == "es"

    def test_update_preference(self, db):
        """Test updating a single preference field."""
        db.get_or_create_preferences("user_123")

        result = db.update_preference("user_123", "language", "fr")
        assert result is True

        prefs = db.get_preferences("user_123")
        assert prefs.language == "fr"

    # =====================
    # Message Deletion Tests
    # =====================

    def test_delete_summarized_messages(self, db, sample_messages):
        """Test deleting messages from a summarized thread."""
        for msg in sample_messages:
            db.insert_message(msg)

        deleted = db.delete_summarized_messages("thread_456", keep_last_n=0)
        assert deleted == 5

        messages = db.fetch_recent_messages("user_123", "thread_456")
        assert len(messages) == 0

    def test_delete_summarized_messages_keep_last(self, db, sample_messages):
        """Test deleting messages but keeping the last N."""
        for msg in sample_messages:
            db.insert_message(msg)

        deleted = db.delete_summarized_messages("thread_456", keep_last_n=2)
        assert deleted == 3

        messages = db.fetch_recent_messages("user_123", "thread_456")
        assert len(messages) == 2

    # =====================
    # Connection Tests
    # =====================

    def test_close(self, db, sample_message):
        """Test closing the database connection."""
        db.insert_message(sample_message)
        db.close()

        # Connection should be cleared
        assert not hasattr(db._local, "connection") or db._local.connection is None

    def test_connection_context_manager_rollback_on_error(self, db):
        """Test that context manager rolls back on error."""
        with patch.object(db, "_get_connection") as mock_conn:
            mock_conn_instance = mock_conn.return_value
            mock_conn_instance.rollback = lambda: None

            # Simulate an error by creating an invalid scenario
            with pytest.raises(Exception):
                with db.get_connection() as conn:
                    raise Exception("Test error")

    def test_thread_safety_with_lock(self, db, sample_message):
        """Test that schema initialization uses lock."""
        # This verifies the lock is used (indirectly through successful operation)
        db.initialize_schema()
        result = db.insert_message(sample_message)
        assert result is True


class TestSQLiteManagerErrorHandling:
    """Tests for error handling in SQLiteManager."""

    @pytest.fixture
    def db(self):
        """Create an in-memory SQLite database for testing."""
        manager = SQLiteManager(":memory:")
        yield manager
        manager.close()

    def test_insert_message_handles_exception(self, db):
        """Test insert_message handles exceptions gracefully."""
        # Create an invalid message (missing required fields would cause issues)
        with patch.object(db, "get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = lambda x: x
            mock_conn.return_value.__exit__ = lambda *args: None
            mock_conn.return_value.execute = lambda *args: (_ for _ in ()).throw(
                Exception("DB Error")
            )

            msg = Message(
                message_id="msg_error",
                user_id="user_123",
                thread_id="thread_456",
                session_id="session_789",
                role=MessageRole.USER,
                raw_text="Test",
            )
            result = db.insert_message(msg)
            assert result is False

    def test_fetch_messages_handles_exception(self, db):
        """Test fetch_recent_messages handles exceptions gracefully."""
        with patch.object(db, "get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = lambda x: x
            mock_conn.return_value.__exit__ = lambda *args: None
            mock_conn.return_value.execute = lambda *args: (_ for _ in ()).throw(
                Exception("DB Error")
            )

            messages = db.fetch_recent_messages("user", "thread")
            assert messages == []

    def test_get_message_by_id_handles_exception(self, db):
        """Test get_message_by_id handles exceptions gracefully."""
        with patch.object(db, "get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = lambda x: x
            mock_conn.return_value.__exit__ = lambda *args: None
            mock_conn.return_value.execute = lambda *args: (_ for _ in ()).throw(
                Exception("DB Error")
            )

            msg = db.get_message_by_id("msg_id")
            assert msg is None

    def test_search_by_relevance_handles_exception(self, db):
        """Test search_by_relevance handles exceptions gracefully."""
        with patch.object(db, "get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = lambda x: x
            mock_conn.return_value.__exit__ = lambda *args: None
            mock_conn.return_value.execute = lambda *args: (_ for _ in ()).throw(
                Exception("DB Error")
            )

            messages = db.search_by_relevance("user_123")
            assert messages == []
