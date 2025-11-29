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
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path

from .schemas import Message, MessageMetadata, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_datetime(dt: Any) -> Optional[datetime]:
    """
    Normalize datetime to be timezone-aware (UTC).

    SQLite stores timestamps as text (ISO format strings), so this function
    handles both string parsing and timezone normalization.

    Args:
        dt: A datetime object (naive or aware), ISO string, or None.

    Returns:
        Timezone-aware datetime in UTC, or None if input is None/invalid.
    """
    if dt is None:
        return None

    # Handle string datetimes (SQLite stores as text)
    if isinstance(dt, str):
        try:
            # Handle various ISO formats
            dt_str = dt.replace('Z', '+00:00')
            # Handle SQLite's default format (no timezone)
            if '+' not in dt_str and 'T' in dt_str:
                dt = datetime.fromisoformat(dt_str)
            else:
                dt = datetime.fromisoformat(dt_str)
        except (ValueError, AttributeError):
            return None

    if not isinstance(dt, datetime):
        return None

    # If naive, assume UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    # Already timezone-aware, convert to UTC
    return dt.astimezone(timezone.utc)


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
                        created_at=_normalize_datetime(row['created_at'])
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
                        created_at=_normalize_datetime(row['created_at'])
                    )
                    messages.append(message)

                return messages
        except Exception as e:
            logger.error(f"Failed to search messages by topic: {e}")
            return []

    def search_by_relevance(
        self,
        user_id: str,
        topics: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        intent: Optional[str] = None,
        min_importance: float = 0.0,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Message]:
        """
        Search messages by relevance using metadata matching and scoring.

        SQLite version uses JSON functions for matching. Scores results by:
        - Topic overlap (3x weight)
        - Category match (2x weight)
        - Intent match (1.5x weight)
        - Importance score (1x weight)
        - Recency (0.5x weight)

        Args:
            user_id: User identifier.
            topics: List of topics to match.
            categories: List of categories to match.
            intent: Intent to match (exact match).
            min_importance: Minimum importance score (0.0-1.0).
            thread_id: Optional thread filter.
            session_id: Optional session filter.
            limit: Maximum number of messages.

        Returns:
            List of Message objects sorted by relevance score.
        """
        # Fetch all matching messages and score in Python (SQLite JSON support is limited)
        conditions = ["user_id = ?"]
        params: List[Any] = [user_id]

        if thread_id:
            conditions.append("thread_id = ?")
            params.append(thread_id)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where_clause = " AND ".join(conditions)

        # Fetch more than limit to allow for scoring/filtering
        fetch_limit = limit * 5
        params.append(fetch_limit)

        search_sql = f"""
        SELECT * FROM messages
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(search_sql, params)
                rows = cursor.fetchall()

                scored_messages = []
                now = datetime.now(timezone.utc)

                for row in rows:
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}

                    # Calculate relevance score
                    score = 0.0

                    # Topic matching (3x weight)
                    if topics:
                        msg_topics = metadata_dict.get('topics', [])
                        topic_matches = len(set(topics) & set(msg_topics))
                        if topic_matches == 0:
                            continue  # Skip if no topic match when topics are specified
                        score += topic_matches * 3.0

                    # Category matching (2x weight)
                    if categories:
                        msg_categories = metadata_dict.get('categories', [])
                        category_matches = len(set(categories) & set(msg_categories))
                        score += category_matches * 2.0

                    # Intent matching (1.5x weight)
                    if intent and metadata_dict.get('intent') == intent:
                        score += 1.5

                    # Importance score (1x weight)
                    importance = metadata_dict.get('importance', 0.5)
                    if importance < min_importance:
                        continue  # Skip if below importance threshold
                    score += importance

                    # Recency score (0.5x weight)
                    created_at = _normalize_datetime(row['created_at'])
                    if created_at:
                        age_seconds = (now - created_at).total_seconds()
                        recency = max(0, 1 - (age_seconds / 604800))  # 7 days
                        score += recency * 0.5

                    message = Message(
                        message_id=row['message_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        role=MessageRole(row['role']),
                        raw_text=row['raw_text'],
                        metadata=MessageMetadata(**metadata_dict) if metadata_dict else MessageMetadata(),
                        created_at=created_at
                    )
                    scored_messages.append((score, message))

                # Sort by score descending and return top results
                scored_messages.sort(key=lambda x: x[0], reverse=True)
                messages = [msg for _, msg in scored_messages[:limit]]

                logger.debug(f"Found {len(messages)} relevant messages for user {user_id}")
                return messages

        except Exception as e:
            logger.error(f"Failed to search by relevance: {e}")
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
                        created_at=_normalize_datetime(row['created_at'])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get message by ID: {e}")
            return None

    def update_message_metadata(
        self,
        message_id: str,
        metadata: MessageMetadata
    ) -> bool:
        """
        Update the metadata of an existing message.

        Used by background enrichment to update messages that were
        initially stored with empty metadata.

        Args:
            message_id: Message identifier.
            metadata: New metadata to store.

        Returns:
            True if successful, False otherwise.
        """
        update_sql = """
        UPDATE messages SET metadata = ? WHERE message_id = ?
        """

        try:
            with self.get_connection() as conn:
                metadata_json = json.dumps(
                    metadata.to_dict() if isinstance(metadata, MessageMetadata)
                    else metadata
                )
                cursor = conn.execute(update_sql, (metadata_json, message_id))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.debug(f"Updated metadata for message {message_id}")
                    return True
                else:
                    logger.warning(f"Message {message_id} not found for metadata update")
                    return False
        except Exception as e:
            logger.error(f"Failed to update message metadata: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.info("SQLite connection closed")
