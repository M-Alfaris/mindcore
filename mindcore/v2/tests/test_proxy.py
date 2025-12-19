"""Comprehensive tests for proxy module."""

import pytest
from datetime import datetime, timezone

from mindcore.v2.proxy import (
    Session,
    SessionManager,
    SessionState,
    ReasoningExtractor,
    ExtractedReasoning,
    ReasoningType,
    InjectionStrategy,
)


class TestSessionState:
    """Test SessionState enum."""

    def test_session_state_values(self):
        """Test session state enum values."""
        assert SessionState.PENDING.value == "pending"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.PAUSED.value == "paused"
        assert SessionState.COMPLETED.value == "completed"
        assert SessionState.FAILED.value == "failed"


class TestSession:
    """Test Session class."""

    def test_create_session(self):
        """Test creating a session."""
        session = Session(
            session_id="session_123",
            user_id="user_456",
        )

        assert session.session_id == "session_123"
        assert session.user_id == "user_456"
        assert session.state == SessionState.PENDING
        assert session.messages == []

    def test_add_message(self):
        """Test adding messages to session."""
        session = Session(session_id="session_123", user_id="user_456")

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"
        assert session.messages[1]["role"] == "assistant"
        assert session.updated_at is not None

    def test_add_message_with_metadata(self):
        """Test adding message with extra metadata."""
        session = Session(session_id="session_123", user_id="user_456")

        session.add_message("user", "Hello", tool_calls=["read_file"])

        assert session.messages[0]["tool_calls"] == ["read_file"]

    def test_activate_session(self):
        """Test activating a session."""
        session = Session(session_id="session_123", user_id="user_456")

        session.activate()

        assert session.state == SessionState.ACTIVE
        assert session.updated_at is not None

    def test_complete_session(self):
        """Test completing a session."""
        session = Session(session_id="session_123", user_id="user_456")
        session.activate()

        session.complete()

        assert session.state == SessionState.COMPLETED

    def test_fail_session(self):
        """Test failing a session."""
        session = Session(session_id="session_123", user_id="user_456")

        session.fail(reason="Connection lost")

        assert session.state == SessionState.FAILED
        assert session.metadata["failure_reason"] == "Connection lost"

    def test_to_dict(self):
        """Test converting session to dictionary."""
        session = Session(
            session_id="session_123",
            user_id="user_456",
            metadata={"key": "value"},
        )
        session.add_message("user", "Hello")

        result = session.to_dict()

        assert isinstance(result, dict)
        assert result["session_id"] == "session_123"
        assert result["user_id"] == "user_456"
        assert result["state"] == "pending"
        assert len(result["messages"]) == 1


class TestSessionManager:
    """Test SessionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a session manager."""
        return SessionManager()

    def test_create_session(self, manager):
        """Test creating a session."""
        session = manager.create("session_123", "user_456")

        assert session.session_id == "session_123"
        assert session.user_id == "user_456"

    def test_create_duplicate_session(self, manager):
        """Test that creating duplicate session raises error."""
        manager.create("session_123", "user_456")

        with pytest.raises(ValueError):
            manager.create("session_123", "user_456")

    def test_get_session(self, manager):
        """Test getting a session."""
        manager.create("session_123", "user_456")

        session = manager.get("session_123")

        assert session is not None
        assert session.session_id == "session_123"

    def test_get_nonexistent_session(self, manager):
        """Test getting non-existent session."""
        session = manager.get("nonexistent")
        assert session is None

    def test_get_or_create_existing(self, manager):
        """Test get_or_create with existing session."""
        original = manager.create("session_123", "user_456")

        result = manager.get_or_create("session_123", "user_999")

        assert result is original

    def test_get_or_create_new(self, manager):
        """Test get_or_create with new session."""
        session = manager.get_or_create("session_123", "user_456")

        assert session.session_id == "session_123"
        assert session.user_id == "user_456"

    def test_list_sessions(self, manager):
        """Test listing sessions."""
        manager.create("session_1", "user_1")
        manager.create("session_2", "user_1")
        manager.create("session_3", "user_2")

        all_sessions = manager.list_sessions()
        user_1_sessions = manager.list_sessions(user_id="user_1")

        assert len(all_sessions) == 3
        assert len(user_1_sessions) == 2

    def test_list_sessions_by_state(self, manager):
        """Test listing sessions by state."""
        s1 = manager.create("session_1", "user_1")
        s2 = manager.create("session_2", "user_1")
        s1.activate()

        active = manager.list_sessions(state=SessionState.ACTIVE)
        pending = manager.list_sessions(state=SessionState.PENDING)

        assert len(active) == 1
        assert len(pending) == 1

    def test_delete_session(self, manager):
        """Test deleting a session."""
        manager.create("session_123", "user_456")

        result = manager.delete("session_123")

        assert result is True
        assert manager.get("session_123") is None

    def test_delete_nonexistent_session(self, manager):
        """Test deleting non-existent session."""
        result = manager.delete("nonexistent")
        assert result is False

    def test_cleanup_completed(self, manager):
        """Test cleaning up old completed sessions."""
        s1 = manager.create("session_1", "user_1")
        s2 = manager.create("session_2", "user_1")
        s3 = manager.create("session_3", "user_1")

        s1.complete()
        s2.fail()
        # s3 stays pending

        # Cleanup with 0 max age (should clean all completed/failed)
        removed = manager.cleanup_completed(max_age_seconds=0)

        assert removed == 2
        assert manager.get("session_1") is None
        assert manager.get("session_2") is None
        assert manager.get("session_3") is not None


class TestReasoningType:
    """Test ReasoningType enum."""

    def test_reasoning_type_values(self):
        """Test reasoning type values."""
        assert ReasoningType.DECISION.value == "decision"
        assert ReasoningType.LEARNING.value == "learning"
        assert ReasoningType.PREFERENCE.value == "preference"
        assert ReasoningType.INSIGHT.value == "insight"
        assert ReasoningType.ERROR_RECOVERY.value == "error_recovery"
        assert ReasoningType.PATTERN.value == "pattern"


class TestExtractedReasoning:
    """Test ExtractedReasoning class."""

    def test_create_reasoning(self):
        """Test creating extracted reasoning."""
        reasoning = ExtractedReasoning(
            reasoning_type=ReasoningType.DECISION,
            content="I decided to use Python",
            confidence=0.9,
        )

        assert reasoning.reasoning_type == ReasoningType.DECISION
        assert reasoning.content == "I decided to use Python"
        assert reasoning.confidence == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        reasoning = ExtractedReasoning(
            reasoning_type=ReasoningType.LEARNING,
            content="I learned that X works better",
            topics=["python", "testing"],
        )

        result = reasoning.to_dict()

        assert result["reasoning_type"] == "learning"
        assert result["content"] == "I learned that X works better"
        assert "python" in result["topics"]

    def test_to_memory_dict(self):
        """Test converting to memory dictionary."""
        reasoning = ExtractedReasoning(
            reasoning_type=ReasoningType.PREFERENCE,
            content="User prefers dark mode",
            confidence=0.85,
            topics=["settings"],
        )

        result = reasoning.to_memory_dict("user_123")

        assert result["content"] == "User prefers dark mode"
        assert result["memory_type"] == "preference"
        assert result["user_id"] == "user_123"
        assert result["importance"] == 0.85


class TestReasoningExtractor:
    """Test ReasoningExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor."""
        return ReasoningExtractor(min_confidence=0.5)

    def test_extract_decision(self, extractor):
        """Test extracting decisions."""
        content = "I'll use the pytest framework for testing. It's the most suitable choice."

        results = extractor.extract_from_message(content, role="assistant")

        assert len(results) > 0
        assert any(r.reasoning_type == ReasoningType.DECISION for r in results)

    def test_extract_learning(self, extractor):
        """Test extracting learnings."""
        content = "I learned that the config file uses YAML format instead of JSON."

        results = extractor.extract_from_message(content, role="assistant")

        assert len(results) > 0
        assert any(r.reasoning_type == ReasoningType.LEARNING for r in results)

    def test_extract_preference(self, extractor):
        """Test extracting preferences."""
        content = "The user prefers to receive notifications via email."

        results = extractor.extract_from_message(content, role="assistant")

        assert len(results) > 0
        assert any(r.reasoning_type == ReasoningType.PREFERENCE for r in results)

    def test_extract_insight(self, extractor):
        """Test extracting insights."""
        content = "I noticed that the database queries are slow during peak hours."

        results = extractor.extract_from_message(content, role="assistant")

        assert len(results) > 0
        assert any(r.reasoning_type == ReasoningType.INSIGHT for r in results)

    def test_extract_error_recovery(self, extractor):
        """Test extracting error recovery."""
        content = "The error was caused by missing permissions. Fixed by adding admin role."

        results = extractor.extract_from_message(content, role="assistant")

        assert len(results) > 0
        assert any(r.reasoning_type == ReasoningType.ERROR_RECOVERY for r in results)

    def test_ignores_user_messages(self, extractor):
        """Test that user messages are ignored."""
        content = "I decided to use Python"

        results = extractor.extract_from_message(content, role="user")

        assert len(results) == 0

    def test_extract_with_context(self, extractor):
        """Test extraction with context."""
        content = "Based on the previous discussion, I'll use SQLite."
        context = "We talked about database options"

        results = extractor.extract_from_message(
            content,
            role="assistant",
            context=context,
        )

        if results:
            assert results[0].context == context

    def test_extract_from_session(self, extractor):
        """Test extracting from entire session."""
        messages = [
            {"role": "user", "content": "Help me with testing"},
            {"role": "assistant", "content": "I'll use pytest for the tests."},
            {"role": "user", "content": "Good idea"},
            {"role": "assistant", "content": "I learned that fixtures help a lot."},
        ]

        results = extractor.extract_from_session(messages)

        # Should extract from assistant messages
        assert len(results) >= 1

    def test_extract_topics(self, extractor):
        """Test topic extraction."""
        content = "I'll use Python with pytest for database testing."

        results = extractor.extract_from_message(content, role="assistant")

        if results:
            topics = results[0].topics
            # Should extract tech keywords
            assert any(t in ["python", "database", "test"] for t in topics)

    def test_min_confidence_filter(self):
        """Test minimum confidence filtering."""
        # High confidence threshold
        extractor = ReasoningExtractor(min_confidence=0.99)

        content = "I'll use X."  # Short, low confidence

        results = extractor.extract_from_message(content, role="assistant")

        # Should filter out low confidence results
        assert all(r.confidence >= 0.99 for r in results)


class TestInjectionStrategy:
    """Test InjectionStrategy enum."""

    def test_strategy_values(self):
        """Test injection strategy values."""
        assert InjectionStrategy.PREPEND.value == "prepend"
        assert InjectionStrategy.APPEND.value == "append"
        assert InjectionStrategy.INTERLEAVE.value == "interleave"
        assert InjectionStrategy.SYSTEM.value == "system"
