"""Tests for Database Manager."""

from unittest.mock import MagicMock, patch

import pytest

from mindcore.core import DatabaseManager, Message, MessageMetadata


# Check if psycopg v3 is available for mocking
try:
    import psycopg
    from psycopg_pool import ConnectionPool

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

# Skip all tests in this module if psycopg is not available
pytestmark = pytest.mark.skipif(
    not HAS_PSYCOPG, reason="psycopg v3 not installed. Install with: pip install 'psycopg[binary,pool]'"
)


class TestDatabaseManager:
    """Test cases for DatabaseManager."""

    @pytest.fixture
    def db_config(self):
        """Database configuration for testing."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "mindcore_test",
            "user": "postgres",
            "password": "postgres",
        }

    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return Message(
            message_id="msg_test_123",
            user_id="user_123",
            thread_id="thread_456",
            session_id="session_789",
            role="user",
            raw_text="Test message content",
            metadata=MessageMetadata(topics=["testing", "database"], importance=0.7),
        )

    @pytest.fixture
    def mock_db(self, db_config):
        """Create a DatabaseManager with mocked connection pool."""
        with patch("mindcore.core.db_manager.ConnectionPool") as mock_pool:
            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance
            db = DatabaseManager(db_config)
            db._mock_pool = mock_pool
            db._mock_pool_instance = mock_pool_instance
            yield db

    def test_db_config_loading(self, mock_db, db_config):
        """Test database configuration loading."""
        assert mock_db.config["host"] == "localhost"
        assert mock_db.config["port"] == 5432

    def test_connection_pool_initialization(self, mock_db):
        """Test connection pool is created."""
        mock_db._mock_pool.assert_called_once()

    def test_schema_initialization(self, mock_db):
        """Test schema initialization SQL execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(mock_db, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            mock_db.initialize_schema()

            # Should execute schema SQL
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_insert_message(self, mock_db, sample_message):
        """Test message insertion."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(mock_db, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            result = mock_db.insert_message(sample_message)

            assert result is True
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_fetch_recent_messages(self, mock_db):
        """Test fetching recent messages."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock database rows
        mock_cursor.fetchall.return_value = [
            {
                "message_id": "msg_1",
                "user_id": "user_123",
                "thread_id": "thread_456",
                "session_id": "session_789",
                "role": "user",
                "raw_text": "Test message",
                "metadata": {"topics": ["test"]},
                "created_at": "2024-01-01 00:00:00",
            }
        ]

        with patch.object(mock_db, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            messages = mock_db.fetch_recent_messages("user_123", "thread_456", limit=10)

            assert isinstance(messages, list)
            mock_cursor.execute.assert_called_once()
