"""SQLite storage backend for Mindcore v2."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..flr import Memory
from .base import BaseStorage


class SQLiteStorage(BaseStorage):
    """SQLite storage backend.

    Thread-safe SQLite storage with full-text search support.

    Example:
        storage = SQLiteStorage("mindcore.db")
        memory_id = storage.store(memory)
        results = storage.search(query="order", topics=["billing"])
    """

    def __init__(self, db_path: str = "mindcore.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
        return self._local.connection

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                topics TEXT,
                categories TEXT,
                sentiment TEXT DEFAULT 'neutral',
                importance REAL DEFAULT 0.5,
                entities TEXT,
                access_level TEXT DEFAULT 'private',
                created_at TEXT,
                last_accessed TEXT,
                expires_at TEXT,
                reinforcement_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                vocabulary_version TEXT DEFAULT '1.0.0',
                embedding BLOB
            )
        """)

        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id
            ON memories(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_agent_id
            ON memories(agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created
            ON memories(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_version
            ON memories(vocabulary_version)
        """)

        # Full-text search table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id,
                content,
                topics,
                entities,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(memory_id, content, topics, entities)
                VALUES (new.memory_id, new.content, new.topics, new.entities);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, memory_id, content, topics, entities)
                VALUES ('delete', old.memory_id, old.content, old.topics, old.entities);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, memory_id, content, topics, entities)
                VALUES ('delete', old.memory_id, old.content, old.topics, old.entities);
                INSERT INTO memories_fts(memory_id, content, topics, entities)
                VALUES (new.memory_id, new.content, new.topics, new.entities);
            END
        """)

        # Transfers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                transfer_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    def store(self, memory: Memory) -> str:
        """Store a memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate ID if not set
        if not memory.memory_id:
            memory.memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        # Set created_at if not set
        if not memory.created_at:
            memory.created_at = datetime.now(timezone.utc)

        cursor.execute("""
            INSERT OR REPLACE INTO memories (
                memory_id, content, memory_type, user_id, agent_id,
                topics, categories, sentiment, importance, entities,
                access_level, created_at, last_accessed, expires_at,
                reinforcement_score, access_count, vocabulary_version, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id,
            memory.content,
            memory.memory_type,
            memory.user_id,
            memory.agent_id,
            json.dumps(memory.topics),
            json.dumps(memory.categories),
            memory.sentiment,
            memory.importance,
            json.dumps(memory.entities),
            memory.access_level,
            memory.created_at.isoformat() if memory.created_at else None,
            memory.last_accessed.isoformat() if memory.last_accessed else None,
            memory.expires_at.isoformat() if memory.expires_at else None,
            memory.reinforcement_score,
            memory.access_count,
            memory.vocabulary_version,
            json.dumps(memory.embedding) if memory.embedding else None,
        ))

        conn.commit()
        return memory.memory_id

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE memories SET
                content = ?,
                memory_type = ?,
                topics = ?,
                categories = ?,
                sentiment = ?,
                importance = ?,
                entities = ?,
                access_level = ?,
                last_accessed = ?,
                reinforcement_score = ?,
                access_count = ?,
                vocabulary_version = ?,
                embedding = ?
            WHERE memory_id = ?
        """, (
            memory.content,
            memory.memory_type,
            json.dumps(memory.topics),
            json.dumps(memory.categories),
            memory.sentiment,
            memory.importance,
            json.dumps(memory.entities),
            memory.access_level,
            memory.last_accessed.isoformat() if memory.last_accessed else None,
            memory.reinforcement_score,
            memory.access_count,
            memory.vocabulary_version,
            json.dumps(memory.embedding) if memory.embedding else None,
            memory.memory_id,
        ))

        conn.commit()
        return cursor.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        conn.commit()

        return cursor.rowcount > 0

    def search(
        self,
        query: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        memory_types: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_importance: float | None = None,
        access_levels: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories with filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query
        conditions = []
        params = []

        if query:
            # Use FTS for text search
            conditions.append("""
                memory_id IN (
                    SELECT memory_id FROM memories_fts
                    WHERE memories_fts MATCH ?
                )
            """)
            # Escape special FTS characters
            safe_query = query.replace('"', '""')
            params.append(f'"{safe_query}"')

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)

        if topics:
            # Check if any topic matches (JSON array contains)
            topic_conditions = []
            for topic in topics:
                topic_conditions.append("topics LIKE ?")
                params.append(f'%"{topic}"%')
            conditions.append(f"({' OR '.join(topic_conditions)})")

        if categories:
            cat_conditions = []
            for cat in categories:
                cat_conditions.append("categories LIKE ?")
                params.append(f'%"{cat}"%')
            conditions.append(f"({' OR '.join(cat_conditions)})")

        if memory_types:
            placeholders = ",".join(["?" for _ in memory_types])
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        if min_importance is not None:
            conditions.append("importance >= ?")
            params.append(min_importance)

        if access_levels:
            placeholders = ",".join(["?" for _ in access_levels])
            conditions.append(f"access_level IN ({placeholders})")
            params.extend(access_levels)

        # Filter expired memories
        conditions.append("(expires_at IS NULL OR expires_at > ?)")
        params.append(datetime.now(timezone.utc).isoformat())

        # Build final query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def search_by_version(
        self,
        version: str,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by vocabulary version."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute("""
                SELECT * FROM memories
                WHERE vocabulary_version = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (version, user_id, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM memories
                WHERE vocabulary_version = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (version, limit, offset))

        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def update_reinforcement(self, memory_id: str, signal: float) -> bool:
        """Update reinforcement score."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE memories
            SET reinforcement_score = reinforcement_score + ?
            WHERE memory_id = ?
        """, (signal, memory_id))

        conn.commit()
        return cursor.rowcount > 0

    def store_transfer(self, transfer_id: str, data: list[dict]) -> None:
        """Store transfer data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transfers (transfer_id, data)
            VALUES (?, ?)
        """, (transfer_id, json.dumps(data)))

        conn.commit()

    def get_transfer(self, transfer_id: str) -> list[dict] | None:
        """Retrieve transfer data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data FROM transfers WHERE transfer_id = ?",
            (transfer_id,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        return json.loads(row["data"])

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        total = cursor.fetchone()["count"]

        # By memory type
        cursor.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """)
        by_type = {row["memory_type"]: row["count"] for row in cursor.fetchall()}

        # By user
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id) as count FROM memories
        """)
        unique_users = cursor.fetchone()["count"]

        # By agent
        cursor.execute("""
            SELECT COUNT(DISTINCT agent_id) as count
            FROM memories
            WHERE agent_id IS NOT NULL
        """)
        unique_agents = cursor.fetchone()["count"]

        # Database size
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

        return {
            "total_memories": total,
            "by_memory_type": by_type,
            "unique_users": unique_users,
            "unique_agents": unique_agents,
            "database_size_bytes": db_size,
        }

    def close(self) -> None:
        """Close storage connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        # Parse datetime fields
        created_at = None
        if row["created_at"]:
            created_at = datetime.fromisoformat(row["created_at"])

        last_accessed = None
        if row["last_accessed"]:
            last_accessed = datetime.fromisoformat(row["last_accessed"])

        expires_at = None
        if row["expires_at"]:
            expires_at = datetime.fromisoformat(row["expires_at"])

        # Parse JSON fields
        topics = json.loads(row["topics"]) if row["topics"] else []
        categories = json.loads(row["categories"]) if row["categories"] else []
        entities = json.loads(row["entities"]) if row["entities"] else []
        embedding = json.loads(row["embedding"]) if row["embedding"] else None

        return Memory(
            memory_id=row["memory_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            topics=topics,
            categories=categories,
            sentiment=row["sentiment"],
            importance=row["importance"],
            entities=entities,
            access_level=row["access_level"],
            created_at=created_at,
            last_accessed=last_accessed,
            expires_at=expires_at,
            reinforcement_score=row["reinforcement_score"],
            access_count=row["access_count"],
            vocabulary_version=row["vocabulary_version"],
            embedding=embedding,
        )
