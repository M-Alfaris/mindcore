"""Shared test fixtures for Mindcore tests.

This module provides centralized test fixtures following 2025 best practices:
- Lazy fixture creation for efficiency
- Proper scoping (function, class, module, session)
- Factory fixtures for flexible test data
- Clear fixture dependencies
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mindcore.core.schemas import Message, MessageMetadata, MessageRole
from mindcore.llm import LLMResponse


if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_config() -> dict:
    """Provide session-scoped test configuration.

    This fixture is created once per test session and shared across all tests.
    Use for configuration that's expensive to create and doesn't change.
    """
    return {
        "test_user_id": "user_123",
        "test_thread_id": "thread_456",
        "test_session_id": "session_789",
        "default_model": "mock-model",
        "max_tokens": 1000,
    }


# =============================================================================
# LLM Provider Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider() -> Mock:
    """Create a mock LLM provider.

    Returns a mock that implements the BaseLLMProvider interface.
    """
    provider = Mock()
    provider.name = "mock"
    provider.is_available.return_value = True
    provider.close = Mock()
    provider.generate.return_value = LLMResponse(
        content='{"result": "success"}',
        model="mock",
        provider="mock",
        tokens_used=10,
    )
    return provider


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create a standard mock LLM response."""
    return LLMResponse(
        content='{"result": "success"}',
        model="mock",
        provider="mock",
        tokens_used=10,
        latency_ms=50.0,
    )


@pytest.fixture
def llm_response_factory() -> Callable[..., LLMResponse]:
    """Factory fixture for creating custom LLM responses.

    Usage:
        def test_something(llm_response_factory):
            response = llm_response_factory(content="custom", tokens_used=100)
    """

    def _create_response(
        content: str = '{"result": "success"}',
        model: str = "mock",
        provider: str = "mock",
        tokens_used: int | None = 10,
        latency_ms: float | None = 50.0,
        **metadata,
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    return _create_response


# =============================================================================
# Message Fixtures
# =============================================================================


@pytest.fixture
def sample_message(test_config: dict) -> Message:
    """Create a single sample message."""
    return Message(
        message_id="msg_test_001",
        user_id=test_config["test_user_id"],
        thread_id=test_config["test_thread_id"],
        session_id=test_config["test_session_id"],
        role=MessageRole.USER,
        raw_text="Test message content",
        metadata=MessageMetadata(topics=["general"], importance=0.5),
    )


@pytest.fixture
def sample_messages(test_config: dict) -> list[Message]:
    """Create a list of sample messages for testing."""
    messages = []
    for i in range(5):
        msg = Message(
            message_id=f"msg_{i}",
            user_id=test_config["test_user_id"],
            thread_id=test_config["test_thread_id"],
            session_id=test_config["test_session_id"],
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            raw_text=f"Test message content {i}",
            metadata=MessageMetadata(
                topics=["general", "testing"],
                importance=0.5 + (i * 0.1),
            ),
        )
        messages.append(msg)
    return messages


@pytest.fixture
def message_factory(test_config: dict) -> Callable[..., Message]:
    """Factory fixture for creating custom messages.

    Usage:
        def test_something(message_factory):
            msg = message_factory(raw_text="custom message", role=MessageRole.ASSISTANT)
    """

    def _create_message(
        message_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        session_id: str | None = None,
        role: MessageRole = MessageRole.USER,
        raw_text: str = "Test message",
        topics: list[str] | None = None,
        importance: float = 0.5,
    ) -> Message:
        import uuid

        return Message(
            message_id=message_id or f"msg_{uuid.uuid4().hex[:8]}",
            user_id=user_id or test_config["test_user_id"],
            thread_id=thread_id or test_config["test_thread_id"],
            session_id=session_id or test_config["test_session_id"],
            role=role,
            raw_text=raw_text,
            metadata=MessageMetadata(
                topics=topics or ["general"],
                importance=importance,
            ),
        )

    return _create_message


@pytest.fixture
def sample_message_dict(test_config: dict) -> dict:
    """Create a sample message dictionary for ingestion."""
    return {
        "user_id": test_config["test_user_id"],
        "thread_id": test_config["test_thread_id"],
        "session_id": test_config["test_session_id"],
        "role": "user",
        "text": "What are best practices for AI agents?",
    }


# =============================================================================
# Vocabulary Fixtures
# =============================================================================


@pytest.fixture
def mock_vocabulary() -> Mock:
    """Create a mock VocabularyManager."""
    vocab = Mock()
    vocab.get_topics.return_value = ["general", "orders", "billing", "support", "technical"]
    vocab.get_categories.return_value = ["general", "question", "request", "complaint"]
    vocab.get_intents.return_value = ["ask_question", "request_action", "provide_info"]
    vocab.validate_topics.return_value = ["general"]
    vocab.validate_categories.return_value = ["general"]
    vocab.resolve_intent.return_value = "ask_question"
    vocab.is_valid_sentiment.return_value = True
    vocab.to_prompt_list.return_value = "Available Topics: general, orders, billing"
    return vocab


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def mock_db_connection() -> Mock:
    """Create a mock database connection."""
    conn = Mock()
    conn.execute = Mock()
    conn.fetchone = Mock(return_value=None)
    conn.fetchall = Mock(return_value=[])
    conn.commit = Mock()
    conn.rollback = Mock()
    conn.close = Mock()
    return conn


@pytest.fixture
def mock_cursor() -> Mock:
    """Create a mock database cursor."""
    cursor = Mock()
    cursor.execute = Mock()
    cursor.fetchone = Mock(return_value=None)
    cursor.fetchall = Mock(return_value=[])
    cursor.close = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    return cursor


# =============================================================================
# VectorStore Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_function() -> Mock:
    """Create a mock embedding function."""
    embed_fn = Mock()
    embed_fn.dimension = 3
    embed_fn.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    embed_fn.embed_query.return_value = [0.1, 0.2, 0.3]
    return embed_fn


@pytest.fixture
def sample_documents() -> list[dict]:
    """Create sample documents for vector store testing."""
    return [
        {"page_content": "First document about AI", "metadata": {"source": "test", "idx": 0}},
        {"page_content": "Second document about ML", "metadata": {"source": "test", "idx": 1}},
        {"page_content": "Third document about NLP", "metadata": {"source": "test", "idx": 2}},
    ]


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
def event_loop_policy():
    """Provide event loop policy for async tests.

    Note: With pytest-asyncio's auto mode and function scope,
    this is typically handled automatically.
    """
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests.

    This runs automatically before each test to ensure test isolation.
    """
    # Reset singletons if they exist in the codebase
    yield
    # Cleanup after test if needed


# =============================================================================
# Markers Registration (optional, also in pyproject.toml)
# =============================================================================


def pytest_configure(config):
    """Register custom markers programmatically.

    These are also defined in pyproject.toml for consistency.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests requiring external services")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
