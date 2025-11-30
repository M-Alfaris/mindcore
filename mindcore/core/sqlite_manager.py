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

from .schemas import Message, MessageMetadata, MessageRole, ThreadSummary, UserPreferences
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

        -- Thread summaries table
        CREATE TABLE IF NOT EXISTS thread_summaries (
            summary_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            session_id TEXT,
            summary TEXT NOT NULL,
            key_facts TEXT DEFAULT '[]',
            topics TEXT DEFAULT '[]',
            categories TEXT DEFAULT '[]',
            overall_sentiment TEXT DEFAULT 'neutral',
            message_count INTEGER DEFAULT 0,
            first_message_at TIMESTAMP,
            last_message_at TIMESTAMP,
            summarized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            entities TEXT DEFAULT '{}',
            messages_deleted BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_summary_user_thread
            ON thread_summaries(user_id, thread_id);

        CREATE INDEX IF NOT EXISTS idx_summary_user
            ON thread_summaries(user_id);

        -- User preferences table
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            language TEXT DEFAULT 'en',
            timezone TEXT DEFAULT 'UTC',
            communication_style TEXT DEFAULT 'balanced',
            interests TEXT DEFAULT '[]',
            goals TEXT DEFAULT '[]',
            preferred_name TEXT,
            custom_context TEXT DEFAULT '{}',
            notification_topics TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
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

    # =====================
    # Thread Summary Methods
    # =====================

    def insert_summary(self, summary: ThreadSummary) -> bool:
        """
        Insert or update a thread summary.

        Args:
            summary: ThreadSummary object to insert.

        Returns:
            True if successful, False otherwise.
        """
        insert_sql = """
        INSERT OR REPLACE INTO thread_summaries (
            summary_id, user_id, thread_id, session_id,
            summary, key_facts, topics, categories, overall_sentiment,
            message_count, first_message_at, last_message_at, summarized_at,
            entities, messages_deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                conn.execute(insert_sql, (
                    summary.summary_id,
                    summary.user_id,
                    summary.thread_id,
                    summary.session_id,
                    summary.summary,
                    json.dumps(summary.key_facts),
                    json.dumps(summary.topics),
                    json.dumps(summary.categories),
                    summary.overall_sentiment,
                    summary.message_count,
                    summary.first_message_at.isoformat() if summary.first_message_at else None,
                    summary.last_message_at.isoformat() if summary.last_message_at else None,
                    summary.summarized_at.isoformat() if summary.summarized_at else None,
                    json.dumps(summary.entities),
                    summary.messages_deleted
                ))
                conn.commit()
                logger.debug(f"Summary {summary.summary_id} inserted for thread {summary.thread_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to insert summary: {e}")
            return False

    def get_summary(self, user_id: str, thread_id: str) -> Optional[ThreadSummary]:
        """
        Get summary for a specific thread.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.

        Returns:
            ThreadSummary object or None.
        """
        select_sql = """
        SELECT * FROM thread_summaries
        WHERE user_id = ? AND thread_id = ?
        ORDER BY summarized_at DESC
        LIMIT 1
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(select_sql, (user_id, thread_id))
                row = cursor.fetchone()

                if row:
                    return ThreadSummary(
                        summary_id=row['summary_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        summary=row['summary'],
                        key_facts=json.loads(row['key_facts']) if row['key_facts'] else [],
                        topics=json.loads(row['topics']) if row['topics'] else [],
                        categories=json.loads(row['categories']) if row['categories'] else [],
                        overall_sentiment=row['overall_sentiment'] or 'neutral',
                        message_count=row['message_count'] or 0,
                        first_message_at=_normalize_datetime(row['first_message_at']),
                        last_message_at=_normalize_datetime(row['last_message_at']),
                        summarized_at=_normalize_datetime(row['summarized_at']),
                        entities=json.loads(row['entities']) if row['entities'] else {},
                        messages_deleted=bool(row['messages_deleted'])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return None

    def get_user_summaries(
        self,
        user_id: str,
        topics: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[ThreadSummary]:
        """
        Get summaries for a user, optionally filtered by topics.

        Args:
            user_id: User identifier.
            topics: Optional topics to filter by.
            limit: Maximum number of summaries.

        Returns:
            List of ThreadSummary objects.
        """
        if topics:
            # Build LIKE conditions for topic matching
            topic_conditions = " OR ".join(["topics LIKE ?" for _ in topics])
            topic_params = [f'%"{topic}"%' for topic in topics]

            select_sql = f"""
            SELECT * FROM thread_summaries
            WHERE user_id = ? AND ({topic_conditions})
            ORDER BY summarized_at DESC
            LIMIT ?
            """
            params = [user_id] + topic_params + [limit]
        else:
            select_sql = """
            SELECT * FROM thread_summaries
            WHERE user_id = ?
            ORDER BY summarized_at DESC
            LIMIT ?
            """
            params = [user_id, limit]

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(select_sql, params)
                rows = cursor.fetchall()

                summaries = []
                for row in rows:
                    summaries.append(ThreadSummary(
                        summary_id=row['summary_id'],
                        user_id=row['user_id'],
                        thread_id=row['thread_id'],
                        session_id=row['session_id'],
                        summary=row['summary'],
                        key_facts=json.loads(row['key_facts']) if row['key_facts'] else [],
                        topics=json.loads(row['topics']) if row['topics'] else [],
                        categories=json.loads(row['categories']) if row['categories'] else [],
                        overall_sentiment=row['overall_sentiment'] or 'neutral',
                        message_count=row['message_count'] or 0,
                        first_message_at=_normalize_datetime(row['first_message_at']),
                        last_message_at=_normalize_datetime(row['last_message_at']),
                        summarized_at=_normalize_datetime(row['summarized_at']),
                        entities=json.loads(row['entities']) if row['entities'] else {},
                        messages_deleted=bool(row['messages_deleted'])
                    ))
                return summaries
        except Exception as e:
            logger.error(f"Failed to get user summaries: {e}")
            return []

    def delete_summarized_messages(
        self,
        thread_id: str,
        keep_last_n: int = 0
    ) -> int:
        """
        Delete raw messages for a summarized thread.

        Args:
            thread_id: Thread identifier.
            keep_last_n: Number of recent messages to keep.

        Returns:
            Number of messages deleted.
        """
        try:
            with self.get_connection() as conn:
                if keep_last_n > 0:
                    # Get IDs of messages to keep
                    keep_sql = """
                    SELECT message_id FROM messages
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """
                    cursor = conn.execute(keep_sql, (thread_id, keep_last_n))
                    keep_ids = [row['message_id'] for row in cursor.fetchall()]

                    if keep_ids:
                        placeholders = ','.join('?' * len(keep_ids))
                        delete_sql = f"""
                        DELETE FROM messages
                        WHERE thread_id = ? AND message_id NOT IN ({placeholders})
                        """
                        cursor = conn.execute(delete_sql, [thread_id] + keep_ids)
                    else:
                        # No messages to keep, delete all
                        delete_sql = "DELETE FROM messages WHERE thread_id = ?"
                        cursor = conn.execute(delete_sql, (thread_id,))
                else:
                    delete_sql = "DELETE FROM messages WHERE thread_id = ?"
                    cursor = conn.execute(delete_sql, (thread_id,))

                conn.commit()
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} messages from thread {thread_id}")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete summarized messages: {e}")
            return 0

    def get_threads_for_summarization(
        self,
        max_age_days: int = 7,
        min_messages: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get threads that are candidates for summarization.

        Args:
            max_age_days: Only summarize threads older than this.
            min_messages: Minimum messages required for summarization.

        Returns:
            List of dicts with thread_id, user_id, message_count.
        """
        select_sql = """
        SELECT
            thread_id,
            user_id,
            COUNT(*) as message_count,
            MIN(created_at) as first_message,
            MAX(created_at) as last_message
        FROM messages
        WHERE created_at < datetime('now', ?)
        GROUP BY thread_id, user_id
        HAVING COUNT(*) >= ?
        ORDER BY last_message DESC
        """

        try:
            with self.get_connection() as conn:
                age_param = f'-{max_age_days} days'
                cursor = conn.execute(select_sql, (age_param, min_messages))
                rows = cursor.fetchall()

                threads = []
                for row in rows:
                    # Check if already summarized
                    existing = self.get_summary(row['user_id'], row['thread_id'])
                    if existing is None:
                        threads.append({
                            'thread_id': row['thread_id'],
                            'user_id': row['user_id'],
                            'message_count': row['message_count'],
                            'first_message': row['first_message'],
                            'last_message': row['last_message']
                        })
                return threads
        except Exception as e:
            logger.error(f"Failed to get threads for summarization: {e}")
            return []

    # =========================
    # User Preferences Methods
    # =========================

    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences.

        Args:
            user_id: User identifier.

        Returns:
            UserPreferences object or None.
        """
        select_sql = "SELECT * FROM user_preferences WHERE user_id = ?"

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(select_sql, (user_id,))
                row = cursor.fetchone()

                if row:
                    return UserPreferences(
                        user_id=row['user_id'],
                        language=row['language'] or 'en',
                        timezone=row['timezone'] or 'UTC',
                        communication_style=row['communication_style'] or 'balanced',
                        interests=json.loads(row['interests']) if row['interests'] else [],
                        goals=json.loads(row['goals']) if row['goals'] else [],
                        preferred_name=row['preferred_name'],
                        custom_context=json.loads(row['custom_context']) if row['custom_context'] else {},
                        notification_topics=json.loads(row['notification_topics']) if row['notification_topics'] else [],
                        created_at=_normalize_datetime(row['created_at']),
                        updated_at=_normalize_datetime(row['updated_at'])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return None

    def get_or_create_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences, creating defaults if not exists.

        Args:
            user_id: User identifier.

        Returns:
            UserPreferences object (existing or newly created).
        """
        existing = self.get_preferences(user_id)
        if existing:
            return existing

        # Create default preferences
        default_prefs = UserPreferences(user_id=user_id)
        self.save_preferences(default_prefs)
        return default_prefs

    def save_preferences(self, preferences: UserPreferences) -> bool:
        """
        Save user preferences.

        Args:
            preferences: UserPreferences object to save.

        Returns:
            True if successful, False otherwise.
        """
        insert_sql = """
        INSERT OR REPLACE INTO user_preferences (
            user_id, language, timezone, communication_style,
            interests, goals, preferred_name, custom_context,
            notification_topics, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                conn.execute(insert_sql, (
                    preferences.user_id,
                    preferences.language,
                    preferences.timezone,
                    preferences.communication_style,
                    json.dumps(preferences.interests),
                    json.dumps(preferences.goals),
                    preferences.preferred_name,
                    json.dumps(preferences.custom_context),
                    json.dumps(preferences.notification_topics),
                    preferences.created_at.isoformat() if preferences.created_at else None,
                    preferences.updated_at.isoformat() if preferences.updated_at else None
                ))
                conn.commit()
                logger.debug(f"Saved preferences for user {preferences.user_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
            return False

    def update_preference(
        self,
        user_id: str,
        field: str,
        value: Any
    ) -> bool:
        """
        Update a single preference field.

        Args:
            user_id: User identifier.
            field: Field name to update.
            value: New value.

        Returns:
            True if successful, False otherwise.
        """
        prefs = self.get_or_create_preferences(user_id)
        if prefs.update(field, value):
            return self.save_preferences(prefs)
        return False
