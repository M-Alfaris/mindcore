"""
Database manager for PostgreSQL persistence.

Supports both direct PostgreSQL connections and PgBouncer pooled connections.
When using PgBouncer, set MINDCORE_DB_USE_PGBOUNCER=true for optimal settings.
"""
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from .schemas import Message, MessageMetadata, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try psycopg v3 first (modern, async-capable), fall back to psycopg2
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    _PSYCOPG_VERSION = 3
    logger.debug("Using psycopg v3")
except ImportError:
    try:
        import psycopg2 as psycopg
        from psycopg2.extras import RealDictCursor
        from psycopg2.pool import ThreadedConnectionPool
        _PSYCOPG_VERSION = 2
        logger.debug("Using psycopg2 (legacy)")
    except ImportError:
        raise ImportError(
            "No PostgreSQL driver found. Install with:\n"
            "  pip install 'psycopg[binary]'  # Recommended (v3)\n"
            "  pip install psycopg2-binary    # Legacy (v2)"
        )


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


def _normalize_datetime(dt: Any) -> Optional[datetime]:
    """
    Normalize datetime to be timezone-aware (UTC).

    PostgreSQL returns naive datetimes, but Mindcore creates messages with
    timezone-aware datetimes. This function ensures consistency by converting
    all datetimes to UTC-aware.

    Args:
        dt: A datetime object (naive or aware), string, or None.

    Returns:
        Timezone-aware datetime in UTC, or None if input is None/invalid.
    """
    if dt is None:
        return None

    # Handle string datetimes
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    if not isinstance(dt, datetime):
        return None

    # If naive, assume UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    # Already timezone-aware, convert to UTC
    return dt.astimezone(timezone.utc)


def get_db_config_from_env() -> Dict[str, Any]:
    """
    Get database configuration from environment variables.

    Environment variables:
        MINDCORE_DB_HOST: Database host (default: localhost)
        MINDCORE_DB_PORT: Database port (default: 5432, or 6432 for PgBouncer)
        MINDCORE_DB_NAME: Database name (default: mindcore)
        MINDCORE_DB_USER: Database user (default: postgres)
        MINDCORE_DB_PASSWORD: Database password (default: postgres)
        MINDCORE_DB_MIN_CONNECTIONS: Minimum pool connections (default: 1)
        MINDCORE_DB_MAX_CONNECTIONS: Maximum pool connections (default: 10)
        MINDCORE_DB_USE_PGBOUNCER: Set to 'true' for PgBouncer mode

    Returns:
        Database configuration dictionary
    """
    use_pgbouncer = os.getenv("MINDCORE_DB_USE_PGBOUNCER", "").lower() == "true"

    # When using PgBouncer, use smaller app-level pool since PgBouncer handles pooling
    default_min = 1 if use_pgbouncer else 2
    default_max = 5 if use_pgbouncer else 10

    return {
        "host": os.getenv("MINDCORE_DB_HOST", "localhost"),
        "port": int(os.getenv("MINDCORE_DB_PORT", "5432")),
        "database": os.getenv("MINDCORE_DB_NAME", "mindcore"),
        "user": os.getenv("MINDCORE_DB_USER", "postgres"),
        "password": os.getenv("MINDCORE_DB_PASSWORD", "postgres"),
        "min_connections": int(os.getenv("MINDCORE_DB_MIN_CONNECTIONS", str(default_min))),
        "max_connections": int(os.getenv("MINDCORE_DB_MAX_CONNECTIONS", str(default_max))),
        "use_pgbouncer": use_pgbouncer,
    }


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations.

    Supports both direct PostgreSQL and PgBouncer connections.
    When using PgBouncer (transaction mode), the app uses a minimal
    connection pool since PgBouncer handles the actual pooling.

    Configuration via environment variables:
        MINDCORE_DB_HOST, MINDCORE_DB_PORT, MINDCORE_DB_NAME,
        MINDCORE_DB_USER, MINDCORE_DB_PASSWORD,
        MINDCORE_DB_MIN_CONNECTIONS, MINDCORE_DB_MAX_CONNECTIONS,
        MINDCORE_DB_USE_PGBOUNCER

    Or pass db_config dict directly.
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager.

        Args:
            db_config: Database configuration dictionary. If None, reads from
                      environment variables.

        Raises:
            DatabaseConnectionError: If connection pool cannot be created.
        """
        self.config = db_config or get_db_config_from_env()
        self._pool = None
        self._use_pgbouncer = self.config.get("use_pgbouncer", False)
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        try:
            if _PSYCOPG_VERSION == 3:
                # psycopg v3 with ConnectionPool
                conninfo = (
                    f"host={self.config.get('host', 'localhost')} "
                    f"port={self.config.get('port', 5432)} "
                    f"dbname={self.config.get('database', 'mindcore')} "
                    f"user={self.config.get('user', 'postgres')} "
                    f"password={self.config.get('password', 'postgres')}"
                )

                self._pool = ConnectionPool(
                    conninfo=conninfo,
                    min_size=self.config.get("min_connections", 1),
                    max_size=self.config.get("max_connections", 10),
                    # PgBouncer compatibility: don't use prepared statements in transaction mode
                    kwargs={"prepare_threshold": None} if self._use_pgbouncer else {},
                )
                logger.info(
                    f"Database pool initialized (psycopg3, "
                    f"pgbouncer={self._use_pgbouncer}, "
                    f"min={self.config.get('min_connections', 1)}, "
                    f"max={self.config.get('max_connections', 10)})"
                )
            else:
                # psycopg2 with ThreadedConnectionPool
                self._pool = ThreadedConnectionPool(
                    minconn=self.config.get("min_connections", 1),
                    maxconn=self.config.get("max_connections", 10),
                    host=self.config.get("host", "localhost"),
                    port=self.config.get("port", 5432),
                    database=self.config.get("database", "mindcore"),
                    user=self.config.get("user", "postgres"),
                    password=self.config.get("password", "postgres"),
                )
                logger.info(
                    f"Database pool initialized (psycopg2, "
                    f"min={self.config.get('min_connections', 1)}, "
                    f"max={self.config.get('max_connections', 10)})"
                )

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
        if self._pool is None:
            raise DatabaseConnectionError("Database pool is not initialized")

        conn = None
        try:
            if _PSYCOPG_VERSION == 3:
                # psycopg v3
                with self._pool.connection() as conn:
                    yield conn
                return  # Connection returned to pool automatically
            else:
                # psycopg2
                conn = self._pool.getconn()
                if conn is None:
                    raise DatabaseConnectionError("Failed to obtain connection from pool")

                # Test if connection is still valid
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                except psycopg.OperationalError:
                    # Connection is stale, try to get a fresh one
                    self._pool.putconn(conn, close=True)
                    conn = self._pool.getconn()
                    if conn is None:
                        raise DatabaseConnectionError("Failed to obtain fresh connection")

                yield conn

        except Exception as e:
            logger.error(f"Database error: {e}")
            if _PSYCOPG_VERSION == 2 and conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise DatabaseConnectionError(f"Database error: {e}") from e
        finally:
            if _PSYCOPG_VERSION == 2 and conn:
                try:
                    self._pool.putconn(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")

    def _get_cursor(self, conn):
        """Get a cursor with dict row factory."""
        if _PSYCOPG_VERSION == 3:
            return conn.cursor(row_factory=dict_row)
        else:
            return conn.cursor(cursor_factory=RealDictCursor)

    def _execute(self, conn, sql: str, params: tuple = None):
        """Execute SQL and return cursor."""
        with self._get_cursor(conn) as cursor:
            cursor.execute(sql, params)
            return cursor

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
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            -- Multi-agent support
            agent_id VARCHAR(255),
            visibility VARCHAR(50) DEFAULT 'private',
            sharing_groups TEXT[] DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_user_thread
            ON messages(user_id, thread_id);

        CREATE INDEX IF NOT EXISTS idx_thread_created
            ON messages(thread_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_metadata_topics
            ON messages USING GIN ((metadata->'topics'));

        CREATE INDEX IF NOT EXISTS idx_metadata_categories
            ON messages USING GIN ((metadata->'categories'));

        CREATE INDEX IF NOT EXISTS idx_metadata_importance
            ON messages (((metadata->>'importance')::float));

        CREATE INDEX IF NOT EXISTS idx_metadata_intent
            ON messages ((metadata->>'intent'));

        CREATE INDEX IF NOT EXISTS idx_session
            ON messages(session_id);

        CREATE INDEX IF NOT EXISTS idx_created_at
            ON messages(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_agent
            ON messages(agent_id) WHERE agent_id IS NOT NULL;
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
            role, raw_text, metadata, created_at,
            agent_id, visibility, sharing_groups
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (message_id) DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            metadata = EXCLUDED.metadata;
        """

        try:
            metadata_dict = (
                message.metadata.to_dict()
                if isinstance(message.metadata, MessageMetadata)
                else message.metadata
            )

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if _PSYCOPG_VERSION == 3:
                        from psycopg.types.json import Jsonb
                        cursor.execute(insert_sql, (
                            message.message_id,
                            message.user_id,
                            message.thread_id,
                            message.session_id,
                            message.role.value,
                            message.raw_text,
                            Jsonb(metadata_dict),
                            message.created_at,
                            getattr(message, 'agent_id', None),
                            getattr(message, 'visibility', 'private'),
                            getattr(message, 'sharing_groups', []),
                        ))
                    else:
                        from psycopg2.extras import Json
                        cursor.execute(insert_sql, (
                            message.message_id,
                            message.user_id,
                            message.thread_id,
                            message.session_id,
                            message.role.value,
                            message.raw_text,
                            Json(metadata_dict),
                            message.created_at,
                            getattr(message, 'agent_id', None),
                            getattr(message, 'visibility', 'private'),
                            getattr(message, 'sharing_groups', []),
                        ))
                    conn.commit()
                    logger.debug(f"Message {message.message_id} inserted successfully")
                    return True
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            return False

    def update_message_metadata(self, message_id: str, metadata: MessageMetadata) -> bool:
        """
        Update message metadata.

        Args:
            message_id: Message identifier.
            metadata: New metadata to set.

        Returns:
            True if successful, False otherwise.
        """
        update_sql = "UPDATE messages SET metadata = %s WHERE message_id = %s;"

        try:
            metadata_dict = (
                metadata.to_dict()
                if isinstance(metadata, MessageMetadata)
                else metadata
            )

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if _PSYCOPG_VERSION == 3:
                        from psycopg.types.json import Jsonb
                        cursor.execute(update_sql, (Jsonb(metadata_dict), message_id))
                    else:
                        from psycopg2.extras import Json
                        cursor.execute(update_sql, (Json(metadata_dict), message_id))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update message metadata: {e}")
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
                with self._get_cursor(conn) as cursor:
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
                            created_at=_normalize_datetime(row['created_at']),
                            agent_id=row.get('agent_id'),
                            visibility=row.get('visibility', 'private'),
                            sharing_groups=row.get('sharing_groups', []),
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
                with self._get_cursor(conn) as cursor:
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

        Uses GIN indexes for fast topic/category matching. Scores results by:
        - Topic overlap (3x weight)
        - Category match (2x weight)
        - Intent match (1.5x weight)
        - Importance score (1x weight)
        - Recency (0.5x weight)

        Args:
            user_id: User identifier.
            topics: List of topics to match (uses GIN index).
            categories: List of categories to match (uses GIN index).
            intent: Intent to match (exact match).
            min_importance: Minimum importance score (0.0-1.0).
            thread_id: Optional thread filter.
            session_id: Optional session filter.
            limit: Maximum number of messages.

        Returns:
            List of Message objects sorted by relevance score.
        """
        # Build dynamic query with relevance scoring
        conditions = ["user_id = %s"]
        params: List[Any] = [user_id]

        # Optional filters
        if thread_id:
            conditions.append("thread_id = %s")
            params.append(thread_id)

        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)

        # Importance filter
        if min_importance > 0:
            conditions.append("COALESCE((metadata->>'importance')::float, 0.5) >= %s")
            params.append(min_importance)

        # Build relevance score calculation
        score_parts = []

        # Topic matching score (3x weight)
        if topics:
            conditions.append("metadata->'topics' ?| %s")
            params.append(topics)
            score_parts.append("""
                (SELECT COUNT(*) FROM jsonb_array_elements_text(COALESCE(metadata->'topics', '[]'::jsonb)) t
                 WHERE t = ANY(%s)) * 3.0
            """)
            params.append(topics)

        # Category matching score (2x weight)
        if categories:
            conditions.append("metadata->'categories' ?| %s")
            params.append(categories)
            score_parts.append("""
                (SELECT COUNT(*) FROM jsonb_array_elements_text(COALESCE(metadata->'categories', '[]'::jsonb)) c
                 WHERE c = ANY(%s)) * 2.0
            """)
            params.append(categories)

        # Intent matching score (1.5x weight)
        if intent:
            score_parts.append("""
                CASE WHEN metadata->>'intent' = %s THEN 1.5 ELSE 0 END
            """)
            params.append(intent)

        # Importance score (1x weight)
        score_parts.append("COALESCE((metadata->>'importance')::float, 0.5)")

        # Recency score (0.5x weight, normalized to 0-1 based on last 7 days)
        score_parts.append("""
            LEAST(1.0, EXTRACT(EPOCH FROM (NOW() - created_at)) / 604800) * -0.5 + 0.5
        """)

        # Combine scores
        if score_parts:
            relevance_score = " + ".join(score_parts)
        else:
            relevance_score = "1.0"

        where_clause = " AND ".join(conditions)
        params.append(limit)

        search_sql = f"""
        SELECT *, ({relevance_score}) as relevance_score
        FROM messages
        WHERE {where_clause}
        ORDER BY relevance_score DESC, created_at DESC
        LIMIT %s;
        """

        try:
            with self.get_connection() as conn:
                with self._get_cursor(conn) as cursor:
                    cursor.execute(search_sql, params)
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
                            created_at=_normalize_datetime(row['created_at'])
                        )
                        messages.append(message)

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
        select_sql = "SELECT * FROM messages WHERE message_id = %s;"

        try:
            with self.get_connection() as conn:
                with self._get_cursor(conn) as cursor:
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
                            created_at=_normalize_datetime(row['created_at'])
                        )
                    return None
        except Exception as e:
            logger.error(f"Failed to get message by ID: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Check database connectivity and return status.

        Returns:
            Dict with status, latency, and pool info.
        """
        import time
        start = time.time()

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()

            latency_ms = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "driver": f"psycopg{_PSYCOPG_VERSION}",
                "pgbouncer": self._use_pgbouncer,
                "pool_min": self.config.get("min_connections", 1),
                "pool_max": self.config.get("max_connections", 10),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "driver": f"psycopg{_PSYCOPG_VERSION}",
            }

    def close(self) -> None:
        """Close all database connections."""
        if self._pool:
            if _PSYCOPG_VERSION == 3:
                self._pool.close()
            else:
                self._pool.closeall()
            logger.info("Database connection pool closed")
