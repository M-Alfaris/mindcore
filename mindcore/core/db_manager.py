"""
Database manager for PostgreSQL persistence.
"""
import json
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool, PoolError
from contextlib import contextmanager

from .schemas import Message, MessageMetadata, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""

    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize database manager.

        Args:
            db_config: Database configuration dictionary.

        Raises:
            DatabaseConnectionError: If connection pool cannot be created.
        """
        self.config = db_config
        self.pool: Optional[ThreadedConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=self.config.get("min_connections", 1),
                maxconn=self.config.get("max_connections", 10),
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 5432),
                database=self.config.get("database", "mindcore"),
                user=self.config.get("user", "postgres"),
                password=self.config.get("password", "postgres"),
            )
            logger.info("Database connection pool initialized")
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Cannot connect to database: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}") from e

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            Database connection from pool.

        Raises:
            DatabaseConnectionError: If connection cannot be obtained.
        """
        if self.pool is None:
            raise DatabaseConnectionError("Database pool is not initialized")

        conn = None
        try:
            conn = self.pool.getconn()
            if conn is None:
                raise DatabaseConnectionError("Failed to obtain connection from pool")

            # Test if connection is still valid
            try:
                conn.cursor().execute("SELECT 1")
            except psycopg2.OperationalError:
                # Connection is stale, try to get a fresh one
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()
                if conn is None:
                    raise DatabaseConnectionError("Failed to obtain fresh connection")

            yield conn

        except PoolError as e:
            logger.error(f"Connection pool error: {e}")
            raise DatabaseConnectionError(f"Connection pool error: {e}") from e
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise DatabaseConnectionError(f"Database error: {e}") from e
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")

    def initialize_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            message_id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            thread_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            raw_text TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_user_thread
            ON messages(user_id, thread_id);

        CREATE INDEX IF NOT EXISTS idx_thread_created
            ON messages(thread_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_metadata_topics
            ON messages USING GIN ((metadata->'topics'));

        CREATE INDEX IF NOT EXISTS idx_created_at
            ON messages(created_at DESC);
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()
                logger.info("Database schema initialized")

    def insert_message(self, message: Message) -> bool:
        """
        Insert a message into the database.

        Args:
            message: Message object to insert.

        Returns:
            True if successful, False otherwise.
        """
        insert_sql = """
        INSERT INTO messages (
            message_id, user_id, thread_id, session_id,
            role, raw_text, metadata, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (message_id) DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            metadata = EXCLUDED.metadata;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        message.message_id,
                        message.user_id,
                        message.thread_id,
                        message.session_id,
                        message.role.value,
                        message.raw_text,
                        Json(message.metadata.to_dict() if isinstance(message.metadata, MessageMetadata) else message.metadata),
                        message.created_at,
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
        WHERE user_id = %s AND thread_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(select_sql, (user_id, thread_id, limit))
                    rows = cursor.fetchall()

                    messages = []
                    for row in rows:
                        message = Message(
                            message_id=row['message_id'],
                            user_id=row['user_id'],
                            thread_id=row['thread_id'],
                            session_id=row['session_id'],
                            role=MessageRole(row['role']),
                            raw_text=row['raw_text'],
                            metadata=MessageMetadata(**row['metadata']) if row['metadata'] else MessageMetadata(),
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

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            topics: List of topics to search for.
            limit: Maximum number of messages.

        Returns:
            List of Message objects.
        """
        search_sql = """
        SELECT * FROM messages
        WHERE user_id = %s
            AND thread_id = %s
            AND metadata->'topics' ?| %s
        ORDER BY created_at DESC
        LIMIT %s;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(search_sql, (user_id, thread_id, topics, limit))
                    rows = cursor.fetchall()

                    messages = []
                    for row in rows:
                        message = Message(
                            message_id=row['message_id'],
                            user_id=row['user_id'],
                            thread_id=row['thread_id'],
                            session_id=row['session_id'],
                            role=MessageRole(row['role']),
                            raw_text=row['raw_text'],
                            metadata=MessageMetadata(**row['metadata']) if row['metadata'] else MessageMetadata(),
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
        select_sql = "SELECT * FROM messages WHERE message_id = %s;"

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(select_sql, (message_id,))
                    row = cursor.fetchone()

                    if row:
                        return Message(
                            message_id=row['message_id'],
                            user_id=row['user_id'],
                            thread_id=row['thread_id'],
                            session_id=row['session_id'],
                            role=MessageRole(row['role']),
                            raw_text=row['raw_text'],
                            metadata=MessageMetadata(**row['metadata']) if row['metadata'] else MessageMetadata(),
                            created_at=row['created_at']
                        )
                    return None
        except Exception as e:
            logger.error(f"Failed to get message by ID: {e}")
            return None

    def close(self) -> None:
        """Close all database connections."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")
