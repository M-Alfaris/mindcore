"""PostgreSQL storage backend for Mindcore v2.

Primary production storage backend with full-text search and JSON support.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..flr import Memory
from .base import BaseStorage


class PostgresStorage(BaseStorage):
    """PostgreSQL storage backend.

    Production-ready storage with:
    - Full-text search via tsvector
    - JSONB for flexible metadata
    - Connection pooling
    - Proper indexing

    Example:
        storage = PostgresStorage("postgresql://user:pass@localhost/mindcore")
        memory_id = storage.store(memory)
        results = storage.search(query="order", topics=["billing"])
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            max_overflow: Max overflow connections

        Raises:
            ImportError: If psycopg is not installed
        """
        try:
            import psycopg
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "psycopg v3 required for PostgreSQL. Install with:\n"
                "  pip install 'psycopg[binary,pool]'"
            )

        self._pool = ConnectionPool(
            connection_string,
            min_size=pool_size,
            max_size=pool_size + max_overflow,
        )
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Main memories table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_id TEXT,
                        topics JSONB DEFAULT '[]'::jsonb,
                        categories JSONB DEFAULT '[]'::jsonb,
                        sentiment TEXT DEFAULT 'neutral',
                        importance REAL DEFAULT 0.5,
                        entities JSONB DEFAULT '[]'::jsonb,
                        access_level TEXT DEFAULT 'private',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_accessed TIMESTAMPTZ,
                        expires_at TIMESTAMPTZ,
                        reinforcement_score REAL DEFAULT 0.0,
                        access_count INTEGER DEFAULT 0,
                        vocabulary_version TEXT DEFAULT '1.0.0',
                        embedding JSONB,
                        search_vector tsvector GENERATED ALWAYS AS (
                            setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
                            setweight(to_tsvector('english', coalesce(topics::text, '')), 'B') ||
                            setweight(to_tsvector('english', coalesce(entities::text, '')), 'C')
                        ) STORED
                    )
                """)

                # Indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_user_id
                    ON memories(user_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_agent_id
                    ON memories(agent_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(memory_type)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_created
                    ON memories(created_at DESC)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_version
                    ON memories(vocabulary_version)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_topics
                    ON memories USING GIN(topics)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_categories
                    ON memories USING GIN(categories)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_search
                    ON memories USING GIN(search_vector)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_access_level
                    ON memories(access_level)
                """)

                # Transfers table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS transfers (
                        transfer_id TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                conn.commit()

    def store(self, memory: Memory) -> str:
        """Store a memory."""
        if not memory.memory_id:
            memory.memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        if not memory.created_at:
            memory.created_at = datetime.now(timezone.utc)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (
                        memory_id, content, memory_type, user_id, agent_id,
                        topics, categories, sentiment, importance, entities,
                        access_level, created_at, last_accessed, expires_at,
                        reinforcement_score, access_count, vocabulary_version, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        memory_type = EXCLUDED.memory_type,
                        topics = EXCLUDED.topics,
                        categories = EXCLUDED.categories,
                        sentiment = EXCLUDED.sentiment,
                        importance = EXCLUDED.importance,
                        entities = EXCLUDED.entities,
                        access_level = EXCLUDED.access_level,
                        last_accessed = EXCLUDED.last_accessed,
                        reinforcement_score = EXCLUDED.reinforcement_score,
                        access_count = EXCLUDED.access_count,
                        vocabulary_version = EXCLUDED.vocabulary_version,
                        embedding = EXCLUDED.embedding
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
                    memory.created_at,
                    memory.last_accessed,
                    memory.expires_at,
                    memory.reinforcement_score,
                    memory.access_count,
                    memory.vocabulary_version,
                    json.dumps(memory.embedding) if memory.embedding else None,
                ))
                conn.commit()

        return memory.memory_id

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM memories WHERE memory_id = %s",
                    (memory_id,)
                )
                row = cur.fetchone()

                if not row:
                    return None

                return self._row_to_memory(row, cur.description)

    def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE memories SET
                        content = %s,
                        memory_type = %s,
                        topics = %s,
                        categories = %s,
                        sentiment = %s,
                        importance = %s,
                        entities = %s,
                        access_level = %s,
                        last_accessed = %s,
                        reinforcement_score = %s,
                        access_count = %s,
                        vocabulary_version = %s,
                        embedding = %s
                    WHERE memory_id = %s
                """, (
                    memory.content,
                    memory.memory_type,
                    json.dumps(memory.topics),
                    json.dumps(memory.categories),
                    memory.sentiment,
                    memory.importance,
                    json.dumps(memory.entities),
                    memory.access_level,
                    memory.last_accessed,
                    memory.reinforcement_score,
                    memory.access_count,
                    memory.vocabulary_version,
                    json.dumps(memory.embedding) if memory.embedding else None,
                    memory.memory_id,
                ))
                conn.commit()
                return cur.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memories WHERE memory_id = %s",
                    (memory_id,)
                )
                conn.commit()
                return cur.rowcount > 0

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
        conditions = []
        params = []

        # Full-text search
        if query:
            conditions.append("search_vector @@ plainto_tsquery('english', %s)")
            params.append(query)

        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)

        if agent_id:
            conditions.append("agent_id = %s")
            params.append(agent_id)

        # JSONB array contains any of the topics
        if topics:
            conditions.append("topics ?| %s")
            params.append(topics)

        if categories:
            conditions.append("categories ?| %s")
            params.append(categories)

        if memory_types:
            conditions.append("memory_type = ANY(%s)")
            params.append(memory_types)

        if start_date:
            conditions.append("created_at >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("created_at <= %s")
            params.append(end_date)

        if min_importance is not None:
            conditions.append("importance >= %s")
            params.append(min_importance)

        if access_levels:
            conditions.append("access_level = ANY(%s)")
            params.append(access_levels)

        # Filter expired memories
        conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        # Add ranking for full-text search
        order_by = "created_at DESC"
        if query:
            order_by = f"ts_rank(search_vector, plainto_tsquery('english', %s)) DESC, created_at DESC"
            params.append(query)

        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_memory(row, cur.description) for row in rows]

    def search_by_version(
        self,
        version: str,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by vocabulary version."""
        conditions = ["vocabulary_version = %s"]
        params = [version]

        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_memory(row, cur.description) for row in rows]

    def update_reinforcement(self, memory_id: str, signal: float) -> bool:
        """Update reinforcement score."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE memories
                    SET reinforcement_score = reinforcement_score + %s
                    WHERE memory_id = %s
                """, (signal, memory_id))
                conn.commit()
                return cur.rowcount > 0

    def store_transfer(self, transfer_id: str, data: list[dict]) -> None:
        """Store transfer data."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transfers (transfer_id, data)
                    VALUES (%s, %s)
                """, (transfer_id, json.dumps(data)))
                conn.commit()

    def get_transfer(self, transfer_id: str) -> list[dict] | None:
        """Retrieve transfer data."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM transfers WHERE transfer_id = %s",
                    (transfer_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                return row[0]  # JSONB is automatically parsed

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Total memories
                cur.execute("SELECT COUNT(*) FROM memories")
                total = cur.fetchone()[0]

                # By memory type
                cur.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories
                    GROUP BY memory_type
                """)
                by_type = {row[0]: row[1] for row in cur.fetchall()}

                # Unique users
                cur.execute("SELECT COUNT(DISTINCT user_id) FROM memories")
                unique_users = cur.fetchone()[0]

                # Unique agents
                cur.execute("""
                    SELECT COUNT(DISTINCT agent_id)
                    FROM memories
                    WHERE agent_id IS NOT NULL
                """)
                unique_agents = cur.fetchone()[0]

                # Database size
                cur.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size('memories'))
                """)
                db_size = cur.fetchone()[0]

                return {
                    "total_memories": total,
                    "by_memory_type": by_type,
                    "unique_users": unique_users,
                    "unique_agents": unique_agents,
                    "database_size": db_size,
                    "pool_size": self._pool.get_stats(),
                }

    def close(self) -> None:
        """Close connection pool."""
        self._pool.close()

    def _row_to_memory(self, row: tuple, description: Any) -> Memory:
        """Convert database row to Memory object."""
        # Build dict from row and description
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row))

        # Parse datetime fields (psycopg returns datetime directly)
        created_at = data.get("created_at")
        last_accessed = data.get("last_accessed")
        expires_at = data.get("expires_at")

        # JSONB fields are automatically parsed by psycopg
        topics = data.get("topics", [])
        categories = data.get("categories", [])
        entities = data.get("entities", [])
        embedding = data.get("embedding")

        return Memory(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            user_id=data["user_id"],
            agent_id=data.get("agent_id"),
            topics=topics if isinstance(topics, list) else [],
            categories=categories if isinstance(categories, list) else [],
            sentiment=data.get("sentiment", "neutral"),
            importance=data.get("importance", 0.5),
            entities=entities if isinstance(entities, list) else [],
            access_level=data.get("access_level", "private"),
            created_at=created_at,
            last_accessed=last_accessed,
            expires_at=expires_at,
            reinforcement_score=data.get("reinforcement_score", 0.0),
            access_count=data.get("access_count", 0),
            vocabulary_version=data.get("vocabulary_version", "1.0.0"),
            embedding=embedding,
        )
