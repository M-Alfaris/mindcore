"""PostgreSQL storage for enriched dataset memories.

Provides a PostgreSQL backend for storing dataset memories with
full SVL-compliant metadata. Uses the same schema as Mindcore's
production storage for realistic benchmarking.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class EnrichedMemory:
    """A memory with full SVL-compliant metadata."""

    # Content
    content: str
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")

    # Required identifiers
    user_id: str = ""
    session_id: str = ""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    agent_id: str | None = None
    thread_id: str | None = None

    # SVL-enforced classifications
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    message_type: str = "statement"
    message_intent: str = "provide_info"

    # Scores
    importance: float = 0.5
    confidence: float = 0.8
    urgency: str = "medium"

    # Additional SVL fields
    sentiment: str = "neutral"
    emotional_classification: str = "neutral"
    temporal_qualifier: str | None = None
    domain_label: str | None = None

    # Memory classification
    memory_type: str = "episodic"
    access_level: str = "private"

    # Source tracking
    dataset_name: str = ""
    turn_index: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "topics": self.topics,
            "categories": self.categories,
            "entities": self.entities,
            "message_type": self.message_type,
            "message_intent": self.message_intent,
            "importance": self.importance,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "emotional_classification": self.emotional_classification,
            "temporal_qualifier": self.temporal_qualifier,
            "domain_label": self.domain_label,
            "memory_type": self.memory_type,
            "access_level": self.access_level,
            "dataset_name": self.dataset_name,
            "turn_index": self.turn_index,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichedMemory:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            memory_id=data.get("memory_id", f"mem_{uuid.uuid4().hex[:12]}"),
            content=data.get("content", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            message_id=data.get("message_id", f"msg_{uuid.uuid4().hex[:12]}"),
            agent_id=data.get("agent_id"),
            thread_id=data.get("thread_id"),
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            entities=data.get("entities", []),
            message_type=data.get("message_type", "statement"),
            message_intent=data.get("message_intent", "provide_info"),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.8),
            urgency=data.get("urgency", "medium"),
            sentiment=data.get("sentiment", "neutral"),
            emotional_classification=data.get("emotional_classification", "neutral"),
            temporal_qualifier=data.get("temporal_qualifier"),
            domain_label=data.get("domain_label"),
            memory_type=data.get("memory_type", "episodic"),
            access_level=data.get("access_level", "private"),
            dataset_name=data.get("dataset_name", ""),
            turn_index=data.get("turn_index", 0),
            created_at=created_at,
        )


class PostgresDatasetStore:
    """PostgreSQL storage for enriched dataset memories.

    Schema mirrors Mindcore's production storage for realistic
    benchmarking with full SVL metadata support.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/mindcore_datasets",
        schema_name: str = "datasets",
    ):
        """Initialize PostgreSQL store.

        Args:
            dsn: PostgreSQL connection string
            schema_name: Schema name for dataset tables
        """
        self.dsn = dsn
        self.schema_name = schema_name
        self._conn = None
        self._ensure_psycopg()

    def _ensure_psycopg(self) -> None:
        """Ensure psycopg is available."""
        try:
            import psycopg

            self._psycopg = psycopg
        except ImportError:
            try:
                import psycopg2 as psycopg

                self._psycopg = psycopg
            except ImportError:
                raise ImportError(
                    "psycopg3 or psycopg2 required. "
                    "Install with: pip install psycopg[binary] or pip install psycopg2-binary"
                )

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        if self._conn is None:
            self._conn = self._psycopg.connect(self.dsn)
            logger.info(f"Connected to PostgreSQL: {self.dsn}")

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def create_schema(self) -> None:
        """Create database schema for dataset storage."""
        self.connect()

        with self._conn.cursor() as cur:
            # Create schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")

            # Create memories table with full SVL metadata
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,

                    -- Identifiers
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    agent_id TEXT,
                    thread_id TEXT,

                    -- SVL classifications (stored as JSONB for flexibility)
                    topics JSONB DEFAULT '[]',
                    categories JSONB DEFAULT '[]',
                    entities JSONB DEFAULT '[]',

                    -- Message classification
                    message_type TEXT DEFAULT 'statement',
                    message_intent TEXT DEFAULT 'provide_info',

                    -- Scores
                    importance FLOAT DEFAULT 0.5,
                    confidence FLOAT DEFAULT 0.8,
                    urgency TEXT DEFAULT 'medium',

                    -- Additional SVL fields
                    sentiment TEXT DEFAULT 'neutral',
                    emotional_classification TEXT DEFAULT 'neutral',
                    temporal_qualifier TEXT,
                    domain_label TEXT,

                    -- Memory classification
                    memory_type TEXT DEFAULT 'episodic',
                    access_level TEXT DEFAULT 'private',

                    -- Source tracking
                    dataset_name TEXT,
                    turn_index INTEGER DEFAULT 0,

                    -- Timestamps
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Create indexes for efficient querying
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                ON {self.schema_name}.memories(user_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_session_id
                ON {self.schema_name}.memories(session_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_memory_type
                ON {self.schema_name}.memories(memory_type)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_dataset
                ON {self.schema_name}.memories(dataset_name)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON {self.schema_name}.memories(importance DESC)
            """)

            # GIN indexes for JSONB array search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_topics
                ON {self.schema_name}.memories USING GIN(topics)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_categories
                ON {self.schema_name}.memories USING GIN(categories)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_entities
                ON {self.schema_name}.memories USING GIN(entities)
            """)

            # Full-text search index
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_content_fts
                ON {self.schema_name}.memories
                USING GIN(to_tsvector('english', content))
            """)

            # Create sessions table for session metadata
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    dataset_name TEXT,
                    persona JSONB DEFAULT '[]',
                    domain TEXT,
                    total_turns INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON {self.schema_name}.sessions(user_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_sessions_dataset
                ON {self.schema_name}.sessions(dataset_name)
            """)

            self._conn.commit()
            logger.info(f"Created schema: {self.schema_name}")

    def drop_schema(self, cascade: bool = True) -> None:
        """Drop the dataset schema.

        Args:
            cascade: Drop all objects in schema
        """
        self.connect()

        with self._conn.cursor() as cur:
            cascade_sql = "CASCADE" if cascade else ""
            cur.execute(f"DROP SCHEMA IF EXISTS {self.schema_name} {cascade_sql}")
            self._conn.commit()
            logger.info(f"Dropped schema: {self.schema_name}")

    def store_memory(self, memory: EnrichedMemory) -> str:
        """Store a single memory.

        Args:
            memory: EnrichedMemory to store

        Returns:
            Memory ID
        """
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema_name}.memories (
                    memory_id, content, user_id, session_id, message_id,
                    agent_id, thread_id, topics, categories, entities,
                    message_type, message_intent, importance, confidence, urgency,
                    sentiment, emotional_classification, temporal_qualifier, domain_label,
                    memory_type, access_level, dataset_name, turn_index, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    updated_at = NOW()
            """,
                (
                    memory.memory_id,
                    memory.content,
                    memory.user_id,
                    memory.session_id,
                    memory.message_id,
                    memory.agent_id,
                    memory.thread_id,
                    json.dumps(memory.topics),
                    json.dumps(memory.categories),
                    json.dumps(memory.entities),
                    memory.message_type,
                    memory.message_intent,
                    memory.importance,
                    memory.confidence,
                    memory.urgency,
                    memory.sentiment,
                    memory.emotional_classification,
                    memory.temporal_qualifier,
                    memory.domain_label,
                    memory.memory_type,
                    memory.access_level,
                    memory.dataset_name,
                    memory.turn_index,
                    memory.created_at,
                ),
            )
            self._conn.commit()

        return memory.memory_id

    def store_memories(self, memories: list[EnrichedMemory]) -> list[str]:
        """Store multiple memories efficiently.

        Args:
            memories: List of EnrichedMemory to store

        Returns:
            List of memory IDs
        """
        if not memories:
            return []

        self.connect()

        memory_ids = []
        with self._conn.cursor() as cur:
            for memory in memories:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema_name}.memories (
                        memory_id, content, user_id, session_id, message_id,
                        agent_id, thread_id, topics, categories, entities,
                        message_type, message_intent, importance, confidence, urgency,
                        sentiment, emotional_classification, temporal_qualifier, domain_label,
                        memory_type, access_level, dataset_name, turn_index, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        updated_at = NOW()
                """,
                    (
                        memory.memory_id,
                        memory.content,
                        memory.user_id,
                        memory.session_id,
                        memory.message_id,
                        memory.agent_id,
                        memory.thread_id,
                        json.dumps(memory.topics),
                        json.dumps(memory.categories),
                        json.dumps(memory.entities),
                        memory.message_type,
                        memory.message_intent,
                        memory.importance,
                        memory.confidence,
                        memory.urgency,
                        memory.sentiment,
                        memory.emotional_classification,
                        memory.temporal_qualifier,
                        memory.domain_label,
                        memory.memory_type,
                        memory.access_level,
                        memory.dataset_name,
                        memory.turn_index,
                        memory.created_at,
                    ),
                )
                memory_ids.append(memory.memory_id)

            self._conn.commit()

        logger.info(f"Stored {len(memories)} memories")
        return memory_ids

    def store_session(
        self,
        session_id: str,
        user_id: str,
        dataset_name: str,
        persona: list[str] | None = None,
        domain: str | None = None,
        total_turns: int = 0,
        metadata: dict | None = None,
    ) -> str:
        """Store session metadata.

        Args:
            session_id: Session ID
            user_id: User ID
            dataset_name: Source dataset name
            persona: Persona traits (for Persona-Chat)
            domain: Domain (for MultiWOZ)
            total_turns: Number of turns
            metadata: Additional metadata

        Returns:
            Session ID
        """
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema_name}.sessions (
                    session_id, user_id, dataset_name, persona, domain,
                    total_turns, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    total_turns = EXCLUDED.total_turns,
                    metadata = EXCLUDED.metadata
            """,
                (
                    session_id,
                    user_id,
                    dataset_name,
                    json.dumps(persona or []),
                    domain,
                    total_turns,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

        return session_id

    def get_memory(self, memory_id: str) -> EnrichedMemory | None:
        """Get a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            EnrichedMemory or None
        """
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self.schema_name}.memories
                WHERE memory_id = %s
            """,
                (memory_id,),
            )

            row = cur.fetchone()
            if not row:
                return None

            return self._row_to_memory(row, cur.description)

    def query_memories(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        dataset_name: str | None = None,
        memory_type: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        min_importance: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EnrichedMemory]:
        """Query memories with filters.

        Args:
            user_id: Filter by user
            session_id: Filter by session
            dataset_name: Filter by dataset
            memory_type: Filter by memory type
            topics: Filter by topics (any match)
            categories: Filter by categories (any match)
            min_importance: Minimum importance score
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of matching memories
        """
        self.connect()

        conditions = []
        params = []

        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)

        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)

        if dataset_name:
            conditions.append("dataset_name = %s")
            params.append(dataset_name)

        if memory_type:
            conditions.append("memory_type = %s")
            params.append(memory_type)

        if topics:
            conditions.append("topics ?| %s")
            params.append(topics)

        if categories:
            conditions.append("categories ?| %s")
            params.append(categories)

        if min_importance is not None:
            conditions.append("importance >= %s")
            params.append(min_importance)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self.schema_name}.memories
                WHERE {where_clause}
                ORDER BY importance DESC, created_at DESC
                LIMIT %s OFFSET %s
            """,
                params + [limit, offset],
            )

            rows = cur.fetchall()
            return [self._row_to_memory(row, cur.description) for row in rows]

    def search_memories(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[EnrichedMemory]:
        """Full-text search memories.

        Args:
            query: Search query
            user_id: Optional user filter
            limit: Maximum results

        Returns:
            List of matching memories
        """
        self.connect()

        with self._conn.cursor() as cur:
            if user_id:
                cur.execute(
                    f"""
                    SELECT *, ts_rank(to_tsvector('english', content), query) as rank
                    FROM {self.schema_name}.memories,
                         plainto_tsquery('english', %s) query
                    WHERE to_tsvector('english', content) @@ query
                      AND user_id = %s
                    ORDER BY rank DESC
                    LIMIT %s
                """,
                    (query, user_id, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT *, ts_rank(to_tsvector('english', content), query) as rank
                    FROM {self.schema_name}.memories,
                         plainto_tsquery('english', %s) query
                    WHERE to_tsvector('english', content) @@ query
                    ORDER BY rank DESC
                    LIMIT %s
                """,
                    (query, limit),
                )

            rows = cur.fetchall()
            return [self._row_to_memory(row, cur.description) for row in rows]

    def get_session_memories(
        self,
        session_id: str,
        order_by_turn: bool = True,
    ) -> list[EnrichedMemory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            order_by_turn: Order by turn index

        Returns:
            List of memories
        """
        self.connect()

        order = "turn_index" if order_by_turn else "created_at DESC"

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self.schema_name}.memories
                WHERE session_id = %s
                ORDER BY {order}
            """,
                (session_id,),
            )

            rows = cur.fetchall()
            return [self._row_to_memory(row, cur.description) for row in rows]

    def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[EnrichedMemory]:
        """Get all memories for a user.

        Args:
            user_id: User ID
            limit: Maximum results

        Returns:
            List of memories
        """
        return self.query_memories(user_id=user_id, limit=limit)

    def get_dataset_stats(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Get statistics for a dataset.

        Args:
            dataset_name: Dataset name (or all if None)

        Returns:
            Statistics dict
        """
        self.connect()

        with self._conn.cursor() as cur:
            if dataset_name:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(importance) as avg_importance,
                        array_agg(DISTINCT memory_type) as memory_types
                    FROM {self.schema_name}.memories
                    WHERE dataset_name = %s
                """,
                    (dataset_name,),
                )
            else:
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(importance) as avg_importance,
                        array_agg(DISTINCT memory_type) as memory_types,
                        array_agg(DISTINCT dataset_name) as datasets
                    FROM {self.schema_name}.memories
                """)

            row = cur.fetchone()

            return {
                "total_memories": row[0],
                "unique_users": row[1],
                "unique_sessions": row[2],
                "avg_importance": float(row[3]) if row[3] else 0,
                "memory_types": row[4] if row[4] else [],
                "datasets": row[5]
                if len(row) > 5 and row[5]
                else [dataset_name]
                if dataset_name
                else [],
            }

    def _row_to_memory(self, row: tuple, description: Any) -> EnrichedMemory:
        """Convert a database row to EnrichedMemory."""
        # Get column names from description
        columns = [col[0] for col in description]
        data = dict(zip(columns, row, strict=False))

        # Parse JSONB fields
        for field in ["topics", "categories", "entities"]:
            if field in data and isinstance(data[field], str):
                data[field] = json.loads(data[field])

        return EnrichedMemory.from_dict(data)

    def clear_dataset(self, dataset_name: str) -> int:
        """Clear all memories for a dataset.

        Args:
            dataset_name: Dataset name

        Returns:
            Number of deleted memories
        """
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {self.schema_name}.memories
                WHERE dataset_name = %s
            """,
                (dataset_name,),
            )
            count = cur.rowcount

            cur.execute(
                f"""
                DELETE FROM {self.schema_name}.sessions
                WHERE dataset_name = %s
            """,
                (dataset_name,),
            )

            self._conn.commit()

        logger.info(f"Cleared {count} memories for dataset: {dataset_name}")
        return count

    def export_to_json(self, path: str, dataset_name: str | None = None) -> int:
        """Export memories to JSON file.

        Args:
            path: Output file path
            dataset_name: Dataset to export (or all)

        Returns:
            Number of exported memories
        """
        memories = self.query_memories(dataset_name=dataset_name, limit=100000)

        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name,
            "total_memories": len(memories),
            "memories": [m.to_dict() for m in memories],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(memories)} memories to {path}")
        return len(memories)


def get_connection_string(
    host: str = "localhost",
    port: int = 5432,
    database: str = "mindcore_datasets",
    user: str = "postgres",
    password: str = "",
) -> str:
    """Build PostgreSQL connection string.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Username
        password: Password

    Returns:
        Connection string
    """
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"
