"""
SQLite database manager for local development.

Provides a lightweight alternative to PostgreSQL for:
- Local development and testing
- Quick prototyping
- Environments where PostgreSQL isn't available
"""
import json
import sqlite3
import threading
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path

from .schemas import Message, MessageMetadata, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteManager:
    """
    SQLite database manager for local development.

    Thread-safe implementation using connection-per-thread pattern.

    Usage:
        from mindcore.core import SQLiteManager

        # Use in-memory database
        db = SQLiteManager(":memory:")

        # Or file-based
        db = SQLiteManager("mindcore.db")
    """

    def __init__(self, db_path: str = "mindcore.db"):
        """
        Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file.
                    Use ":memory:" for in-memory database.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize schema on first connection
        self.initialize_schema()
        logger.info(f"SQLite manager initialized with database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            SQLite connection.
        """
        conn = self._get_connection()
        try:
            yield conn
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            conn.rollback()
            raise
        finally:
            pass  # Keep connection open for reuse

    def initialize_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_user_thread
            ON messages(user_id, thread_id);

        CREATE INDEX IF NOT EXISTS idx_thread_created
            ON messages(thread_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_created_at
            ON messages(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_session
            ON messages(session_id);
        """

        with self._lock:
            conn = self._get_connection()
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("SQLite schema initialized")

    def insert_message(self, message: Message) -> bool:
        """
        Insert a message into the database.

        Args:
            message: Message object to insert.

        Returns:
            True if successful, False otherwise.
        """
        insert_sql = """
        INSERT OR REPLACE INTO messages (
            message_id, user_id, thread_id, session_id,
            role, raw_text, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                metadata_json = json.dumps(
                    message.metadata.to_dict() if isinstance(message.metadata, MessageMetadata)
                    else message.metadata
                )
                conn.execute(insert_sql, (
                    message.message_id,
                    message.user_id,
                    message.thread_id,
                    message.session_id,
                    message.role.value,
                    message.raw_text,
                    metadata_json,
                    message.created_at.isoformat() if message.created_at else None,
                ))
                conn.commit()
                logger.debug(f"Message {message.message_id} inserted successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            return False

    def fetch_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        limit: int = 50
    ) -> List[Message]:
        """
        Fetch recent messages for a user and thread.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            limit: Maximum number of messages to fetch.

        Returns:
            List of Message objects.
        """
        select_sql = """
        SELECT * FROM messages
        WHERE user_id = ? AND thread_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(select_sql, (user_id, thread_id, limit))
                rows = cursor.fetchall()

                messages = []
                for row in rows:
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
                    message = Message(
                        message_id=row['message_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        role=MessageRole(row['role']),
                        raw_text=row['raw_text'],
                        metadata=MessageMetadata(**metadata_dict) if metadata_dict else MessageMetadata(),
                        created_at=row['created_at']
                    )
                    messages.append(message)

                logger.debug(f"Fetched {len(messages)} messages for {user_id}/{thread_id}")
                return messages
        except Exception as e:
            logger.error(f"Failed to fetch messages: {e}")
            return []

    def search_messages_by_topic(
        self,
        user_id: str,
        thread_id: str,
        topics: List[str],
        limit: int = 20
    ) -> List[Message]:
        """
        Search messages by topics.

        Note: SQLite doesn't have native JSON array search, so this uses LIKE.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            topics: List of topics to search for.
            limit: Maximum number of messages.

        Returns:
            List of Message objects.
        """
        # Build LIKE conditions for each topic
        topic_conditions = " OR ".join(["metadata LIKE ?" for _ in topics])
        topic_params = [f'%"{topic}"%' for topic in topics]

        search_sql = f"""
        SELECT * FROM messages
        WHERE user_id = ?
            AND thread_id = ?
            AND ({topic_conditions})
        ORDER BY created_at DESC
        LIMIT ?
        """

        try:
            with self.get_connection() as conn:
                params = [user_id, thread_id] + topic_params + [limit]
                cursor = conn.execute(search_sql, params)
                rows = cursor.fetchall()

                messages = []
                for row in rows:
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
                    message = Message(
                        message_id=row['message_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        role=MessageRole(row['role']),
                        raw_text=row['raw_text'],
                        metadata=MessageMetadata(**metadata_dict) if metadata_dict else MessageMetadata(),
                        created_at=row['created_at']
                    )
                    messages.append(message)

                return messages
        except Exception as e:
            logger.error(f"Failed to search messages by topic: {e}")
            return []

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get a single message by ID.

        Args:
            message_id: Message identifier.

        Returns:
            Message object or None.
        """
        select_sql = "SELECT * FROM messages WHERE message_id = ?"

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(select_sql, (message_id,))
                row = cursor.fetchone()

                if row:
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
                    return Message(
                        message_id=row['message_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        role=MessageRole(row['role']),
                        raw_text=row['raw_text'],
                        metadata=MessageMetadata(**metadata_dict) if metadata_dict else MessageMetadata(),
                        created_at=row['created_at']
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get message by ID: {e}")
            return None

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.info("SQLite connection closed")
