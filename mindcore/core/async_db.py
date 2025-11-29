"""
Async database managers for high-performance applications.

Provides async alternatives to sync database managers:
- AsyncSQLiteManager: Uses aiosqlite for non-blocking SQLite operations
- AsyncDatabaseManager: Uses asyncpg for non-blocking PostgreSQL operations

Usage:
    from mindcore.core.async_db import AsyncSQLiteManager

    async with AsyncSQLiteManager("mindcore.db") as db:
        await db.insert_message(message)
        messages = await db.fetch_recent_messages(user_id, thread_id)
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .schemas import Message, MessageMetadata, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AsyncSQLiteManager:
    """
    Async SQLite database manager using aiosqlite.

    Provides non-blocking database operations for asyncio applications.

    Usage:
        async with AsyncSQLiteManager("mindcore.db") as db:
            await db.insert_message(message)
            messages = await db.fetch_recent_messages(user_id, thread_id)
    """

    def __init__(self, db_path: str = "mindcore.db"):
        """
        Initialize async SQLite manager.

        Args:
            db_path: Path to SQLite database file.
                    Use ":memory:" for in-memory database.
        """
        self.db_path = db_path
        self._connection = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for async SQLite support. "
                "Install it with: pip install aiosqlite"
            )

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys = ON")

        if not self._initialized:
            await self.initialize_schema()
            self._initialized = True

        logger.info(f"Async SQLite connected to: {self.db_path}")

    async def initialize_schema(self) -> None:
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
        await self._connection.executescript(schema_sql)
        await self._connection.commit()
        logger.info("Async SQLite schema initialized")

    async def insert_message(self, message: Message) -> bool:
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
            metadata_json = json.dumps(
                message.metadata.to_dict() if isinstance(message.metadata, MessageMetadata)
                else message.metadata
            )
            await self._connection.execute(insert_sql, (
                message.message_id,
                message.user_id,
                message.thread_id,
                message.session_id,
                message.role.value,
                message.raw_text,
                metadata_json,
                message.created_at.isoformat() if message.created_at else None,
            ))
            await self._connection.commit()
            logger.debug(f"Message {message.message_id} inserted (async)")
            return True
        except Exception as e:
            logger.error(f"Failed to insert message (async): {e}")
            return False

    async def fetch_recent_messages(
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
            cursor = await self._connection.execute(select_sql, (user_id, thread_id, limit))
            rows = await cursor.fetchall()

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

            logger.debug(f"Fetched {len(messages)} messages (async) for {user_id}/{thread_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to fetch messages (async): {e}")
            return []

    async def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get a single message by ID.

        Args:
            message_id: Message identifier.

        Returns:
            Message object or None.
        """
        select_sql = "SELECT * FROM messages WHERE message_id = ?"

        try:
            cursor = await self._connection.execute(select_sql, (message_id,))
            row = await cursor.fetchone()

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
            logger.error(f"Failed to get message by ID (async): {e}")
            return None

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Async SQLite connection closed")


class AsyncDatabaseManager:
    """
    Async PostgreSQL database manager using asyncpg.

    Provides non-blocking database operations with connection pooling.

    Usage:
        async with AsyncDatabaseManager(db_config) as db:
            await db.insert_message(message)
            messages = await db.fetch_recent_messages(user_id, thread_id)
    """

    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize async database manager.

        Args:
            db_config: Database configuration dictionary.
        """
        self.config = db_config
        self._pool = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Create connection pool and initialize schema."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for async PostgreSQL support. "
                "Install it with: pip install asyncpg"
            )

        self._pool = await asyncpg.create_pool(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            database=self.config.get("database", "mindcore"),
            user=self.config.get("user", "postgres"),
            password=self.config.get("password", "postgres"),
            min_size=self.config.get("min_connections", 1),
            max_size=self.config.get("max_connections", 10),
        )

        await self.initialize_schema()
        logger.info("Async PostgreSQL pool initialized")

    async def initialize_schema(self) -> None:
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

        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)
        logger.info("Async PostgreSQL schema initialized")

    async def insert_message(self, message: Message) -> bool:
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
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (message_id) DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            metadata = EXCLUDED.metadata
        """

        try:
            metadata_json = json.dumps(
                message.metadata.to_dict() if isinstance(message.metadata, MessageMetadata)
                else message.metadata
            )
            async with self._pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    message.message_id,
                    message.user_id,
                    message.thread_id,
                    message.session_id,
                    message.role.value,
                    message.raw_text,
                    metadata_json,
                    message.created_at,
                )
            logger.debug(f"Message {message.message_id} inserted (async)")
            return True
        except Exception as e:
            logger.error(f"Failed to insert message (async): {e}")
            return False

    async def fetch_recent_messages(
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
        WHERE user_id = $1 AND thread_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(select_sql, user_id, thread_id, limit)

            messages = []
            for row in rows:
                metadata_raw = row['metadata']
                metadata_dict = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
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

            logger.debug(f"Fetched {len(messages)} messages (async) for {user_id}/{thread_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to fetch messages (async): {e}")
            return []

    async def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get a single message by ID.

        Args:
            message_id: Message identifier.

        Returns:
            Message object or None.
        """
        select_sql = "SELECT * FROM messages WHERE message_id = $1"

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(select_sql, message_id)

            if row:
                metadata_raw = row['metadata']
                metadata_dict = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
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
            logger.error(f"Failed to get message by ID (async): {e}")
            return None

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Async PostgreSQL pool closed")
