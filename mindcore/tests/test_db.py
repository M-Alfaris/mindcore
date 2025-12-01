"""
Tests for Database Manager.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from mindcore.core import DatabaseManager, Message, MessageMetadata

# Check if psycopg2 is available for mocking
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Skip all tests in this module if psycopg2 is not available
pytestmark = pytest.mark.skipif(
    not HAS_PSYCOPG2,
    reason="psycopg2 not installed. Install with: pip install mindcore[postgres]"
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
            "password": "postgres"
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
            metadata=MessageMetadata(
                topics=["testing", "database"],
                importance=0.7
            )
        )

    def test_db_config_loading(self, db_config):
        """Test database configuration loading."""
        with patch('psycopg2.pool.ThreadedConnectionPool'):
            db = DatabaseManager(db_config)
            assert db.config["host"] == "localhost"
            assert db.config["port"] == 5432

    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_connection_pool_initialization(self, mock_pool, db_config):
        """Test connection pool is created."""
        db = DatabaseManager(db_config)
        mock_pool.assert_called_once()

    def test_schema_initialization(self, db_config):
        """Test schema initialization SQL execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch('psycopg2.pool.ThreadedConnectionPool'):
            db = DatabaseManager(db_config)

            with patch.object(db, 'get_connection') as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn

                db.initialize_schema()

                # Should execute schema SQL
                mock_cursor.execute.assert_called_once()
                mock_conn.commit.assert_called_once()

    def test_insert_message(self, db_config, sample_message):
        """Test message insertion."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch('psycopg2.pool.ThreadedConnectionPool'):
            db = DatabaseManager(db_config)

            with patch.object(db, 'get_connection') as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn

                result = db.insert_message(sample_message)

                assert result is True
                mock_cursor.execute.assert_called_once()
                mock_conn.commit.assert_called_once()

    def test_fetch_recent_messages(self, db_config):
        """Test fetching recent messages."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock database rows
        mock_cursor.fetchall.return_value = [
            {
                'message_id': 'msg_1',
                'user_id': 'user_123',
                'thread_id': 'thread_456',
                'session_id': 'session_789',
                'role': 'user',
                'raw_text': 'Test message',
                'metadata': {'topics': ['test']},
                'created_at': '2024-01-01 00:00:00'
            }
        ]

        with patch('psycopg2.pool.ThreadedConnectionPool'):
            db = DatabaseManager(db_config)

            with patch.object(db, 'get_connection') as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn

                messages = db.fetch_recent_messages("user_123", "thread_456", limit=10)

                assert isinstance(messages, list)
                mock_cursor.execute.assert_called_once()
