"""PostgreSQL storage backend for SAGE (Structured Augmented Generation Engine).

PostgreSQL-first architecture where core logic lives in SQL:
- Scoring via sage_score() function
- Session aggregates via triggers
- Full-text search via tsvector
- Fuzzy matching via pg_trgm

SVL acts as the KERNEL/COMPILER:
- Standard metadata: Enforced by system (message_type, intent, importance, etc.)
- User metadata: Assigned by user (topics, categories, custom tags)

Enhanced Search (optional):
    - pg_trgm: Fuzzy/trigram similarity matching
    - ParadeDB pg_search: BM25 full-text ranking
    - SQL ranking functions: Database-native scoring

See mindcore/storage/schema/ for setup instructions.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mindcore.clst.aggregates import SessionAggregate
from mindcore.exceptions import MemoryNotFoundError, StorageError
from mindcore.flr import Memory

from .base import BaseStorage

# Path to SQL files
_SQL_DIR = Path(__file__).parent


if TYPE_CHECKING:
    from .config import SearchConfig

logger = logging.getLogger(__name__)


class PostgresStorage(BaseStorage):
    """PostgreSQL storage backend.

    Production-ready storage with:
    - Full-text search via tsvector
    - JSONB for flexible metadata
    - Explicit connection pool management with configurable limits
    - Proper indexing

    Connection Pool Management:
        - Uses psycopg3's ConnectionPool for efficient connection reuse
        - Configurable min/max connections to match workload
        - Pool statistics available via get_stats()
        - Connection timeout prevents indefinite waits

    Example:
        storage = PostgresStorage(
            "postgresql://user:pass@localhost/mindcore",
            pool_size=10,
            max_overflow=20,
            connection_timeout=30.0,
        )
        memory_id = storage.store(memory)

        # Check pool health
        stats = storage.get_stats()
        print(f"Pool: {stats['connection_pool']}")
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        connection_timeout: float = 30.0,
        search_config: SearchConfig | None = None,
    ):
        """Initialize PostgreSQL storage with connection pool management.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Base connection pool size (min connections)
            max_overflow: Additional connections allowed above pool_size
            connection_timeout: Timeout for acquiring connections (seconds)
            search_config: Configuration for search features (pg_trgm, BM25).
                If None, uses default SearchConfig.

        Raises:
            ImportError: If psycopg is not installed
            StorageConnectionError: If database connection fails
        """
        try:
            import psycopg  # noqa: F401
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "psycopg v3 required for PostgreSQL. Install with:\n"
                "  pip install 'psycopg[binary,pool]'"
            )

        # Import SearchConfig here to avoid circular imports
        from .config import SearchConfig

        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._connection_timeout = connection_timeout
        self._search_config = search_config or SearchConfig()

        try:
            # Configure connection for better performance
            def configure_conn(conn):
                # Disable synchronous_commit for better write performance
                # Data is still safe (WAL is written), just not synced to disk immediately
                with conn.transaction():
                    conn.execute("SET synchronous_commit = off")

            self._pool = ConnectionPool(
                connection_string,
                min_size=pool_size,
                max_size=pool_size + max_overflow,
                timeout=connection_timeout,
                configure=configure_conn,
            )
            # Pre-warm the connection pool
            self._pool.wait()
        except Exception as e:
            raise StorageError(
                f"Failed to create connection pool: {e}",
                details={"connection_string": connection_string.split("@")[-1]},  # Hide credentials
            )

        self._initialize_schema()

        # Check for search extensions after schema init
        self._has_pg_trgm = self._check_extension("pg_trgm")
        self._has_pg_search = self._check_extension("pg_search")
        self._has_rank_memory = self._check_function("rank_memory")
        self._has_rank_session = self._check_function("rank_session")

        if self._search_config.use_trigram_search and not self._has_pg_trgm:
            logger.warning(
                "pg_trgm extension not available. Trigram search disabled. "
                "See mindcore/storage/schema/README.md for setup."
            )
        if self._search_config.use_bm25_search and not self._has_pg_search:
            logger.warning(
                "pg_search extension (ParadeDB) not available. BM25 search disabled. "
                "See mindcore/storage/schema/README.md for setup."
            )
        if self._search_config.use_sql_ranking and not self._has_rank_memory:
            logger.warning(
                "rank_memory() function not available. SQL ranking disabled. "
                "See mindcore/storage/schema/ranking_functions.sql for setup."
            )

    def _check_extension(self, extension_name: str) -> bool:
        """Check if a PostgreSQL extension is installed."""
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = %s",
                    (extension_name,),
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    def _check_function(self, function_name: str) -> bool:
        """Check if a PostgreSQL function exists."""
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM pg_proc p
                    JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE p.proname = %s AND n.nspname = 'public'
                    """,
                    (function_name,),
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        with self._pool.connection() as conn, conn.cursor() as cur:
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
                        session_id TEXT,
                        message_index INTEGER DEFAULT 0,
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

            # Session aggregates table for hierarchical retrieval
            cur.execute("""
                    CREATE TABLE IF NOT EXISTS session_aggregates (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        agent_id TEXT,
                        topic_weights JSONB DEFAULT '{}'::jsonb,
                        category_weights JSONB DEFAULT '{}'::jsonb,
                        entity_weights JSONB DEFAULT '{}'::jsonb,
                        intent_weights JSONB DEFAULT '{}'::jsonb,
                        sentiment_weights JSONB DEFAULT '{}'::jsonb,
                        importance_min REAL DEFAULT 1.0,
                        importance_max REAL DEFAULT 0.0,
                        importance_avg REAL DEFAULT 0.0,
                        importance_sum REAL DEFAULT 0.0,
                        confidence_min REAL DEFAULT 1.0,
                        confidence_max REAL DEFAULT 0.0,
                        confidence_avg REAL DEFAULT 0.0,
                        confidence_sum REAL DEFAULT 0.0,
                        memory_count INTEGER DEFAULT 0,
                        message_count INTEGER DEFAULT 0,
                        started_at TIMESTAMPTZ,
                        last_activity_at TIMESTAMPTZ,
                        dominant_topic TEXT,
                        dominant_category TEXT,
                        dominant_sentiment TEXT,
                        max_urgency TEXT,
                        access_level TEXT DEFAULT 'private',
                        summary_text TEXT,
                        summary_embedding JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
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

            # Session-related indexes on memories
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_session_id
                    ON memories(session_id)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_session_order
                    ON memories(session_id, message_index)
                """)

            # Session aggregates indexes for fast hierarchical queries
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_user_id
                    ON session_aggregates(user_id)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_agent_id
                    ON session_aggregates(agent_id)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_topics
                    ON session_aggregates USING GIN(topic_weights)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_categories
                    ON session_aggregates USING GIN(category_weights)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_importance
                    ON session_aggregates(importance_avg DESC)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_activity
                    ON session_aggregates(last_activity_at DESC)
                """)
            cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_user_activity
                    ON session_aggregates(user_id, last_activity_at DESC)
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

        with self._pool.connection() as conn, conn.cursor() as cur:
            # Get next message_index if session_id is set and message_index is 0
            if memory.session_id and memory.message_index == 0:
                memory.message_index = self._get_next_message_index_internal(cur, memory.session_id)

            cur.execute(
                """
                    INSERT INTO memories (
                        memory_id, content, memory_type, user_id, agent_id,
                        topics, categories, sentiment, importance, entities,
                        access_level, session_id, message_index,
                        created_at, last_accessed, expires_at,
                        reinforcement_score, access_count, vocabulary_version, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
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
                        session_id = EXCLUDED.session_id,
                        message_index = EXCLUDED.message_index,
                        last_accessed = EXCLUDED.last_accessed,
                        reinforcement_score = EXCLUDED.reinforcement_score,
                        access_count = EXCLUDED.access_count,
                        vocabulary_version = EXCLUDED.vocabulary_version,
                        embedding = EXCLUDED.embedding
                """,
                (
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
                    memory.session_id,
                    memory.message_index,
                    memory.created_at,
                    memory.last_accessed,
                    memory.expires_at,
                    memory.reinforcement_score,
                    memory.access_count,
                    memory.vocabulary_version,
                    json.dumps(memory.embedding) if memory.embedding else None,
                ),
            )

            # Update session aggregate if session_id is set
            # Skip if database trigger handles this automatically
            if memory.session_id and not self._has_session_trigger:
                self._update_session_aggregate_internal(cur, memory.session_id, memory)

            conn.commit()

        return memory.memory_id

    def _get_next_message_index_internal(self, cur, session_id: str) -> int:
        """Get next message index (internal, uses existing cursor)."""
        cur.execute(
            "SELECT COALESCE(MAX(message_index), -1) + 1 FROM memories WHERE session_id = %s",
            (session_id,),
        )
        result = cur.fetchone()
        # COALESCE guarantees a value, but add safety check
        return result[0] if result else 0

    def _update_session_aggregate_internal(self, cur, session_id: str, memory: Memory) -> None:
        """Update session aggregate incrementally (internal, uses existing cursor)."""
        now = datetime.now(timezone.utc)

        # Try to get existing aggregate
        cur.execute("SELECT * FROM session_aggregates WHERE session_id = %s", (session_id,))
        row = cur.fetchone()

        if row is None:
            # Create new aggregate
            aggregate = SessionAggregate(
                session_id=session_id,
                user_id=memory.user_id,
                agent_id=memory.agent_id,
            )
            aggregate.update_from_memory(memory)

            cur.execute(
                """
                INSERT INTO session_aggregates (
                    session_id, user_id, agent_id,
                    topic_weights, category_weights, entity_weights,
                    intent_weights, sentiment_weights,
                    importance_min, importance_max, importance_avg, importance_sum,
                    confidence_min, confidence_max, confidence_avg, confidence_sum,
                    memory_count, message_count,
                    started_at, last_activity_at,
                    dominant_topic, dominant_category, dominant_sentiment,
                    max_urgency, access_level,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s
                )
                """,
                (
                    aggregate.session_id,
                    aggregate.user_id,
                    aggregate.agent_id,
                    json.dumps(aggregate.topic_weights),
                    json.dumps(aggregate.category_weights),
                    json.dumps(aggregate.entity_weights),
                    json.dumps(aggregate.intent_weights),
                    json.dumps(aggregate.sentiment_weights),
                    aggregate.importance_min,
                    aggregate.importance_max,
                    aggregate.importance_avg,
                    aggregate.importance_sum,
                    aggregate.confidence_min,
                    aggregate.confidence_max,
                    aggregate.confidence_avg,
                    aggregate.confidence_sum,
                    aggregate.memory_count,
                    aggregate.message_count,
                    aggregate.started_at,
                    aggregate.last_activity_at,
                    aggregate.dominant_topic,
                    aggregate.dominant_category,
                    aggregate.dominant_sentiment,
                    aggregate.max_urgency,
                    aggregate.access_level,
                    now,
                    now,
                ),
            )
        else:
            # Load existing aggregate and update
            columns = [desc[0] for desc in cur.description]
            data = dict(zip(columns, row, strict=False))
            aggregate = self._row_to_session_aggregate(data)
            aggregate.update_from_memory(memory)

            cur.execute(
                """
                UPDATE session_aggregates SET
                    topic_weights = %s,
                    category_weights = %s,
                    entity_weights = %s,
                    sentiment_weights = %s,
                    importance_min = %s,
                    importance_max = %s,
                    importance_avg = %s,
                    importance_sum = %s,
                    memory_count = %s,
                    message_count = %s,
                    last_activity_at = %s,
                    dominant_topic = %s,
                    dominant_category = %s,
                    dominant_sentiment = %s,
                    access_level = %s,
                    updated_at = %s
                WHERE session_id = %s
                """,
                (
                    json.dumps(aggregate.topic_weights),
                    json.dumps(aggregate.category_weights),
                    json.dumps(aggregate.entity_weights),
                    json.dumps(aggregate.sentiment_weights),
                    aggregate.importance_min,
                    aggregate.importance_max,
                    aggregate.importance_avg,
                    aggregate.importance_sum,
                    aggregate.memory_count,
                    aggregate.message_count,
                    aggregate.last_activity_at,
                    aggregate.dominant_topic,
                    aggregate.dominant_category,
                    aggregate.dominant_sentiment,
                    aggregate.access_level,
                    now,
                    session_id,
                ),
            )

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM memories WHERE memory_id = %s", (memory_id,))
            row = cur.fetchone()

            if not row:
                return None

            return self._row_to_memory(row, cur.description)

    def update(self, memory: Memory) -> None:
        """Update an existing memory.

        Args:
            memory: Memory with updated fields

        Raises:
            MemoryNotFoundError: If memory doesn't exist
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
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
                """,
                (
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
                ),
            )
            conn.commit()

            if cur.rowcount == 0:
                raise MemoryNotFoundError(memory.memory_id)

    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Args:
            memory_id: Memory identifier

        Raises:
            MemoryNotFoundError: If memory doesn't exist
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE memory_id = %s", (memory_id,))
            conn.commit()

            if cur.rowcount == 0:
                raise MemoryNotFoundError(memory_id)

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
            order_by = (
                "ts_rank(search_vector, plainto_tsquery('english', %s)) DESC, created_at DESC"
            )
            params.append(query)

        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self._pool.connection() as conn, conn.cursor() as cur:
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

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_memory(row, cur.description) for row in rows]

    # =========================================================================
    # SAGE SCORING METHODS (PostgreSQL-first)
    # =========================================================================

    def search_scored(
        self,
        user_id: str,
        query: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[tuple[Memory, float]]:
        """Search memories using SAGE scoring function in PostgreSQL.

        This is the primary search method - scoring happens in SQL, not Python.

        Args:
            user_id: User identifier
            query: Full-text search query
            topics: Filter by topics
            categories: Filter by categories
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            session_id: Filter by session
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of (Memory, sage_score) tuples sorted by score
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM search_memories_scored(
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    user_id,
                    query,
                    topics,
                    categories,
                    memory_types,
                    min_importance,
                    session_id,
                    limit,
                    offset,
                ),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            results = []
            for row in rows:
                data = dict(zip(columns, row, strict=False))
                score = data.pop("sage_score", 0.0)
                memory = Memory(
                    memory_id=data["memory_id"],
                    content=data["content"],
                    memory_type=data["memory_type"],
                    user_id=user_id,
                    topics=data.get("topics", []),
                    categories=data.get("categories", []),
                    importance=data.get("importance", 0.5),
                    sentiment=data.get("sentiment", "neutral"),
                    reinforcement_score=data.get("reinforcement_score", 0.0),
                    session_id=data.get("session_id"),
                    created_at=data.get("created_at"),
                )
                results.append((memory, score))

            return results

    def search_fuzzy(
        self,
        user_id: str,
        query: str,
        similarity_threshold: float = 0.3,
        limit: int = 20,
    ) -> list[tuple[Memory, float, float]]:
        """Fuzzy search using pg_trgm similarity.

        Finds memories even with typos or partial matches.

        Args:
            user_id: User identifier
            query: Search query (can contain typos)
            similarity_threshold: Minimum similarity (0-1)
            limit: Maximum results

        Returns:
            List of (Memory, similarity, sage_score) tuples
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM search_memories_fuzzy(%s, %s, %s, %s)
                """,
                (user_id, query, similarity_threshold, limit),
            )
            rows = cur.fetchall()

            results = []
            for row in rows:
                memory = self.get(row[0])  # memory_id
                if memory:
                    results.append((memory, row[2], row[3]))  # similarity, sage_score

            return results

    def find_relevant_sessions(
        self,
        user_id: str,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        min_importance: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find relevant sessions by topic/category weights.

        Hierarchical retrieval: find sessions first, then drill down.

        Args:
            user_id: User identifier
            topics: Topic hints to match
            categories: Category hints to match
            min_importance: Minimum average importance
            limit: Maximum results

        Returns:
            List of session info dicts with relevance scores
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM find_relevant_sessions(%s, %s, %s, %s, %s)
                """,
                (user_id, topics, categories, min_importance, limit),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            return [dict(zip(columns, row, strict=False)) for row in rows]

    def update_reinforcement(self, memory_id: str, signal: float) -> float:
        """Update reinforcement score using PostgreSQL function.

        The reinforcement score is bounded to [-1.0, 1.0] and
        access_count/last_accessed are updated automatically.

        Args:
            memory_id: Memory identifier
            signal: Reinforcement signal to add

        Returns:
            New reinforcement score

        Raises:
            MemoryNotFoundError: If memory doesn't exist
            TypeError: If signal is not a valid number
        """
        if not isinstance(signal, int | float):
            raise TypeError(f"Signal must be a number, got {type(signal).__name__}")

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT update_reinforcement(%s, %s)", (memory_id, float(signal)))
            result = cur.fetchone()
            conn.commit()

            if result is None or result[0] is None:
                raise MemoryNotFoundError(memory_id)

            return result[0]

    def cleanup_expired(self) -> int:
        """Clean up expired memories using PostgreSQL function.

        Returns:
            Number of memories deleted
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT cleanup_expired_memories()")
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else 0

    def archive_inactive_sessions(self, days_inactive: int = 30) -> int:
        """Archive sessions that have been inactive.

        Args:
            days_inactive: Days of inactivity before archiving

        Returns:
            Number of sessions archived
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT archive_inactive_sessions(%s)", (days_inactive,))
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else 0

    def initialize_functions(self) -> None:
        """Initialize SQL functions from functions.sql.

        Call this once during setup to create all PostgreSQL functions.
        """
        functions_sql = _SQL_DIR / "functions.sql"
        if not functions_sql.exists():
            raise FileNotFoundError(f"SQL functions file not found: {functions_sql}")

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(functions_sql.read_text())
            conn.commit()

    def initialize_full_schema(self) -> None:
        """Initialize full schema from schema.sql and functions.sql.

        Call this for fresh database setup.
        """
        schema_sql = _SQL_DIR / "schema.sql"
        functions_sql = _SQL_DIR / "functions.sql"

        with self._pool.connection() as conn, conn.cursor() as cur:
            # Create extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

            # Execute schema
            if schema_sql.exists():
                cur.execute(schema_sql.read_text())

            # Execute functions
            if functions_sql.exists():
                cur.execute(functions_sql.read_text())

            conn.commit()

    def store_transfer(self, transfer_id: str, data: list[dict]) -> None:
        """Store transfer data."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO transfers (transfer_id, data)
                    VALUES (%s, %s)
                """,
                (transfer_id, json.dumps(data)),
            )
            conn.commit()

    def get_transfer(self, transfer_id: str) -> list[dict] | None:
        """Retrieve transfer data."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT data FROM transfers WHERE transfer_id = %s", (transfer_id,))
            row = cur.fetchone()
            if not row:
                return None
            return row[0]  # JSONB is automatically parsed

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics including connection pool info."""
        with self._pool.connection() as conn, conn.cursor() as cur:
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

            # Get pool statistics
            pool_stats = self._pool.get_stats()

            return {
                "total_memories": total,
                "by_memory_type": by_type,
                "unique_users": unique_users,
                "unique_agents": unique_agents,
                "database_size": db_size,
                "connection_pool": {
                    "pool_size": self._pool_size,
                    "max_size": self._pool_size + self._max_overflow,
                    "connection_timeout": self._connection_timeout,
                    "pool_stats": pool_stats,
                },
            }

    def close(self) -> None:
        """Close connection pool and release all resources."""
        self._pool.close()

    def _row_to_memory(self, row: tuple, description: Any) -> Memory:
        """Convert database row to Memory object."""
        # Build dict from row and description
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row, strict=False))

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
            session_id=data.get("session_id"),
            message_index=data.get("message_index", 0),
            created_at=created_at,
            last_accessed=last_accessed,
            expires_at=expires_at,
            reinforcement_score=data.get("reinforcement_score", 0.0),
            access_count=data.get("access_count", 0),
            vocabulary_version=data.get("vocabulary_version", "1.0.0"),
            embedding=embedding,
        )

    def _row_to_session_aggregate(self, data: dict[str, Any]) -> SessionAggregate:
        """Convert database row dict to SessionAggregate object."""
        return SessionAggregate(
            session_id=data["session_id"],
            user_id=data["user_id"],
            agent_id=data.get("agent_id"),
            topic_weights=data.get("topic_weights", {}),
            category_weights=data.get("category_weights", {}),
            entity_weights=data.get("entity_weights", {}),
            intent_weights=data.get("intent_weights", {}),
            sentiment_weights=data.get("sentiment_weights", {}),
            importance_min=data.get("importance_min", 1.0),
            importance_max=data.get("importance_max", 0.0),
            importance_avg=data.get("importance_avg", 0.0),
            importance_sum=data.get("importance_sum", 0.0),
            confidence_min=data.get("confidence_min", 1.0),
            confidence_max=data.get("confidence_max", 0.0),
            confidence_avg=data.get("confidence_avg", 0.0),
            confidence_sum=data.get("confidence_sum", 0.0),
            memory_count=data.get("memory_count", 0),
            message_count=data.get("message_count", 0),
            started_at=data.get("started_at"),
            last_activity_at=data.get("last_activity_at"),
            dominant_topic=data.get("dominant_topic"),
            dominant_category=data.get("dominant_category"),
            dominant_sentiment=data.get("dominant_sentiment"),
            max_urgency=data.get("max_urgency"),
            access_level=data.get("access_level", "private"),
            summary_text=data.get("summary_text"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # ==========================================================================
    # Session Aggregate Methods
    # ==========================================================================

    def store_session_aggregate(self, aggregate: SessionAggregate) -> str:
        """Store or update a session aggregate."""
        now = datetime.now(timezone.utc)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_aggregates (
                    session_id, user_id, agent_id,
                    topic_weights, category_weights, entity_weights,
                    intent_weights, sentiment_weights,
                    importance_min, importance_max, importance_avg, importance_sum,
                    confidence_min, confidence_max, confidence_avg, confidence_sum,
                    memory_count, message_count,
                    started_at, last_activity_at,
                    dominant_topic, dominant_category, dominant_sentiment,
                    max_urgency, access_level, summary_text,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    topic_weights = EXCLUDED.topic_weights,
                    category_weights = EXCLUDED.category_weights,
                    entity_weights = EXCLUDED.entity_weights,
                    intent_weights = EXCLUDED.intent_weights,
                    sentiment_weights = EXCLUDED.sentiment_weights,
                    importance_min = EXCLUDED.importance_min,
                    importance_max = EXCLUDED.importance_max,
                    importance_avg = EXCLUDED.importance_avg,
                    importance_sum = EXCLUDED.importance_sum,
                    memory_count = EXCLUDED.memory_count,
                    message_count = EXCLUDED.message_count,
                    last_activity_at = EXCLUDED.last_activity_at,
                    dominant_topic = EXCLUDED.dominant_topic,
                    dominant_category = EXCLUDED.dominant_category,
                    dominant_sentiment = EXCLUDED.dominant_sentiment,
                    max_urgency = EXCLUDED.max_urgency,
                    access_level = EXCLUDED.access_level,
                    summary_text = EXCLUDED.summary_text,
                    updated_at = %s
                """,
                (
                    aggregate.session_id,
                    aggregate.user_id,
                    aggregate.agent_id,
                    json.dumps(aggregate.topic_weights),
                    json.dumps(aggregate.category_weights),
                    json.dumps(aggregate.entity_weights),
                    json.dumps(aggregate.intent_weights),
                    json.dumps(aggregate.sentiment_weights),
                    aggregate.importance_min,
                    aggregate.importance_max,
                    aggregate.importance_avg,
                    aggregate.importance_sum,
                    aggregate.confidence_min,
                    aggregate.confidence_max,
                    aggregate.confidence_avg,
                    aggregate.confidence_sum,
                    aggregate.memory_count,
                    aggregate.message_count,
                    aggregate.started_at,
                    aggregate.last_activity_at,
                    aggregate.dominant_topic,
                    aggregate.dominant_category,
                    aggregate.dominant_sentiment,
                    aggregate.max_urgency,
                    aggregate.access_level,
                    aggregate.summary_text,
                    now,
                    now,
                    now,  # For the UPDATE clause
                ),
            )
            conn.commit()

        return aggregate.session_id

    def get_session_aggregate(self, session_id: str) -> SessionAggregate | None:
        """Retrieve a session aggregate by ID."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM session_aggregates WHERE session_id = %s", (session_id,))
            row = cur.fetchone()

            if not row:
                return None

            columns = [desc[0] for desc in cur.description]
            data = dict(zip(columns, row, strict=False))
            return self._row_to_session_aggregate(data)

    def query_sessions(
        self,
        user_id: str,
        topic_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        min_importance_avg: float | None = None,
        min_topic_weight: float = 0.0,
        agent_ids: list[str] | None = None,
        access_levels: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SessionAggregate]:
        """Query sessions by weighted metadata.

        This enables hierarchical retrieval - find relevant sessions first,
        then drill down to specific memories.
        """
        conditions = ["user_id = %s"]
        params: list[Any] = [user_id]

        if agent_ids:
            conditions.append("agent_id = ANY(%s)")
            params.append(agent_ids)

        if access_levels:
            conditions.append("access_level = ANY(%s)")
            params.append(access_levels)

        if min_importance_avg is not None:
            conditions.append("importance_avg >= %s")
            params.append(min_importance_avg)

        if start_date:
            conditions.append("started_at >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("last_activity_at <= %s")
            params.append(end_date)

        # Build topic matching condition with weight threshold
        if topic_hints:
            # Match sessions that have any of the topic hints with sufficient weight
            topic_conditions = []
            for topic in topic_hints:
                topic_conditions.append(
                    "(topic_weights ? %s AND (topic_weights->>%s)::float >= %s)"
                )
                params.extend([topic, topic, min_topic_weight])
            conditions.append(f"({' OR '.join(topic_conditions)})")

        if category_hints:
            category_conditions = []
            for category in category_hints:
                category_conditions.append("(category_weights ? %s)")
                params.append(category)
            conditions.append(f"({' OR '.join(category_conditions)})")

        where_clause = " AND ".join(conditions)

        # Build ORDER BY with weighted scoring
        order_parts = []
        if topic_hints:
            # Score by sum of topic weights for matching hints
            weight_calcs = []
            for topic in topic_hints:
                weight_calcs.append("COALESCE((topic_weights->>%s)::float, 0)")
                params.append(topic)
            order_parts.append(f"({' + '.join(weight_calcs)}) DESC")

        order_parts.append("importance_avg DESC")
        order_parts.append("last_activity_at DESC")
        order_by = ", ".join(order_parts)

        params.extend([limit, offset])

        sql = f"""
            SELECT * FROM session_aggregates
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s
        """

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return [
                self._row_to_session_aggregate(dict(zip(columns, row, strict=False)))
                for row in rows
            ]

    def query_memories_by_sessions(
        self,
        session_ids: list[str],
        min_importance: float | None = None,
        min_confidence: float | None = None,
        memory_types: list[str] | None = None,
        limit: int = 100,
        order_by_message_index: bool = True,
    ) -> list[Memory]:
        """Query memories from specific sessions, preserving event order."""
        if not session_ids:
            return []

        conditions = ["session_id = ANY(%s)"]
        params: list[Any] = [session_ids]

        if min_importance is not None:
            conditions.append("importance >= %s")
            params.append(min_importance)

        if memory_types:
            conditions.append("memory_type = ANY(%s)")
            params.append(memory_types)

        # Filter expired
        conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        where_clause = " AND ".join(conditions)

        order_by = "session_id, message_index" if order_by_message_index else "created_at DESC"

        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_memory(row, cur.description) for row in rows]

    def get_next_message_index(self, session_id: str) -> int:
        """Get the next message index for a session."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            return self._get_next_message_index_internal(cur, session_id)

    def update_session_aggregate_from_memory(
        self,
        session_id: str,
        memory: Memory,
    ) -> None:
        """Update session aggregate incrementally from a new memory."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            self._update_session_aggregate_internal(cur, session_id, memory)
            conn.commit()

    # ==========================================================================
    # Enhanced Search Methods (pg_trgm, ParadeDB BM25, SQL ranking)
    # ==========================================================================

    def search_ranked(
        self,
        query: str,
        user_id: str,
        attention_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        min_importance: float = 0.0,
        min_similarity: float = 0.1,
        memory_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[tuple[Memory, float]]:
        """Search memories with SQL-based ranking.

        Uses pg_trgm for fuzzy content matching and rank_memory()
        for multi-component scoring entirely in PostgreSQL.

        Requires pg_trgm extension and rank_memory() function.
        See mindcore/storage/schema/README.md for setup.

        Args:
            query: Search query text
            user_id: Filter by user
            attention_hints: Topics to prioritize in ranking
            category_hints: Categories to prioritize (used for filtering)
            min_importance: Minimum importance threshold
            min_similarity: Minimum trigram similarity (0-1)
            memory_types: Filter by memory types
            limit: Maximum results to return

        Returns:
            List of (Memory, score) tuples, sorted by score descending

        Raises:
            StorageError: If required extensions are not available
        """
        if not self._has_pg_trgm:
            raise StorageError(
                "pg_trgm extension required for search_ranked(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )
        if not self._has_rank_memory:
            raise StorageError(
                "rank_memory() function required for search_ranked(). "
                "Run mindcore/storage/schema/ranking_functions.sql to create it."
            )

        conditions = ["m.user_id = %s"]
        params: list[Any] = [user_id]

        # Expiration filter
        conditions.append("(m.expires_at IS NULL OR m.expires_at > NOW())")

        # Importance filter
        if min_importance > 0:
            conditions.append("m.importance >= %s")
            params.append(min_importance)

        # Memory type filter
        if memory_types:
            conditions.append("m.memory_type = ANY(%s)")
            params.append(memory_types)

        # Category filter
        if category_hints:
            conditions.append("m.categories ?| %s")
            params.append(category_hints)

        # Build search condition using trigram OR topic match OR full-text
        search_conditions = []

        # Trigram similarity on content
        if query:
            search_conditions.append("similarity(m.content, %s) >= %s")
            params.extend([query, min_similarity])

        # Topic match with attention hints
        if attention_hints:
            search_conditions.append("m.topics ?| %s")
            params.append(attention_hints)

        # Full-text search fallback
        if query:
            search_conditions.append("m.search_vector @@ plainto_tsquery('english', %s)")
            params.append(query)

        if search_conditions:
            conditions.append(f"({' OR '.join(search_conditions)})")

        where_clause = " AND ".join(conditions)

        # Build ranking call
        attention_array = f"ARRAY{attention_hints!r}" if attention_hints else "ARRAY[]::text[]"
        weights_json = self._search_config.to_sql_weights_json()

        sql = f"""
            SELECT m.*,
                   rank_memory(
                       m.content, m.topics, %s, {attention_array},
                       m.importance, m.reinforcement_score, m.created_at, m.access_count,
                       %s::jsonb
                   ) AS relevance_score
            FROM memories m
            WHERE {where_clause}
            ORDER BY relevance_score DESC
            LIMIT %s
        """
        params.extend([query or "", weights_json, limit])

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                # Last column is relevance_score
                memory = self._row_to_memory(row[:-1], cur.description[:-1])
                score = float(row[-1]) if row[-1] is not None else 0.0
                results.append((memory, score))

            return results

    def search_bm25(
        self,
        query: str,
        user_id: str,
        attention_hints: list[str] | None = None,
        min_importance: float = 0.0,
        memory_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[tuple[Memory, float]]:
        """Search using ParadeDB BM25 + custom ranking.

        Combines BM25 text relevance with custom scoring signals
        (topic match, recency, reinforcement, importance).

        Requires ParadeDB pg_search extension and rank_memory() function.
        See mindcore/storage/schema/README.md for setup.

        Args:
            query: Search query text
            user_id: Filter by user
            attention_hints: Topics to prioritize in ranking
            min_importance: Minimum importance threshold
            memory_types: Filter by memory types
            limit: Maximum results to return

        Returns:
            List of (Memory, score) tuples, sorted by combined score descending

        Raises:
            StorageError: If required extensions are not available
        """
        if not self._has_pg_search:
            raise StorageError(
                "ParadeDB pg_search extension required for search_bm25(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )
        if not self._has_rank_memory:
            raise StorageError(
                "rank_memory() function required for search_bm25(). "
                "Run mindcore/storage/schema/ranking_functions.sql to create it."
            )

        # Fetch more results for re-ranking
        fetch_limit = limit * self._search_config.bm25_fetch_multiplier

        conditions = ["m.user_id = %s"]
        params: list[Any] = [user_id]

        # Expiration filter
        conditions.append("(m.expires_at IS NULL OR m.expires_at > NOW())")

        # Importance filter
        if min_importance > 0:
            conditions.append("m.importance >= %s")
            params.append(min_importance)

        # Memory type filter
        if memory_types:
            conditions.append("m.memory_type = ANY(%s)")
            params.append(memory_types)

        where_clause = " AND ".join(conditions)

        # Build attention hints array for SQL
        attention_array = f"ARRAY{attention_hints!r}" if attention_hints else "ARRAY[]::text[]"

        # Custom weights for hybrid scoring (content weight reduced since BM25 handles text)
        hybrid_weights = {
            "content": 0.0,  # BM25 handles text relevance
            "topic": 0.25,
            "recency": 0.15,
            "reinforcement": 0.15,
            "importance": 0.1,
            "popularity": 0.0,
        }
        hybrid_weights_json = json.dumps(hybrid_weights)
        bm25_weight = self._search_config.bm25_weight

        sql = f"""
            WITH bm25_results AS (
                SELECT memory_id, score_bm25
                FROM memories_bm25.search(
                    query => paradedb.parse(%s),
                    limit_rows => %s
                )
            )
            SELECT m.*,
                   (
                       -- Normalized BM25 score
                       (b.score_bm25 / GREATEST(MAX(b.score_bm25) OVER (), 0.001)) * %s +
                       -- Custom ranking signals
                       rank_memory(
                           m.content, m.topics, %s, {attention_array},
                           m.importance, m.reinforcement_score, m.created_at, m.access_count,
                           %s::jsonb
                       ) * (1 - %s)
                   ) AS combined_score
            FROM memories m
            JOIN bm25_results b ON m.memory_id = b.memory_id
            WHERE {where_clause}
            ORDER BY combined_score DESC
            LIMIT %s
        """
        params_full = [
            query,
            fetch_limit,
            bm25_weight,
            query,
            hybrid_weights_json,
            bm25_weight,
            *params,
            limit,
        ]

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params_full)
            rows = cur.fetchall()

            results = []
            for row in rows:
                # Last column is combined_score
                memory = self._row_to_memory(row[:-1], cur.description[:-1])
                score = float(row[-1]) if row[-1] is not None else 0.0
                results.append((memory, score))

            return results

    def query_sessions_ranked(
        self,
        user_id: str,
        topic_hints: list[str] | None = None,
        category_hints: list[str] | None = None,
        min_importance_avg: float | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[SessionAggregate, float]]:
        """Query sessions with SQL-based ranking.

        Uses rank_session() SQL function for database-native scoring.
        Requires rank_session() function from ranking_functions.sql.

        Args:
            user_id: Filter by user
            topic_hints: Topics to match and rank by
            category_hints: Categories to match and rank by
            min_importance_avg: Minimum average importance
            agent_ids: Filter by agents
            limit: Maximum sessions to return

        Returns:
            List of (SessionAggregate, score) tuples, sorted by score descending

        Raises:
            StorageError: If rank_session() function is not available
        """
        if not self._has_rank_session:
            raise StorageError(
                "rank_session() function required for query_sessions_ranked(). "
                "Run mindcore/storage/schema/ranking_functions.sql to create it."
            )

        conditions = ["s.user_id = %s"]
        params: list[Any] = [user_id]

        if agent_ids:
            conditions.append("s.agent_id = ANY(%s)")
            params.append(agent_ids)

        if min_importance_avg is not None:
            conditions.append("s.importance_avg >= %s")
            params.append(min_importance_avg)

        where_clause = " AND ".join(conditions)

        # Build hint arrays for SQL
        topic_array = f"ARRAY{topic_hints!r}" if topic_hints else "ARRAY[]::text[]"
        category_array = f"ARRAY{category_hints!r}" if category_hints else "ARRAY[]::text[]"

        sql = f"""
            SELECT s.*,
                   rank_session(
                       s.topic_weights, s.category_weights, s.importance_avg,
                       s.last_activity_at, {topic_array}, {category_array}
                   ) AS relevance_score
            FROM session_aggregates s
            WHERE {where_clause}
            ORDER BY relevance_score DESC
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            columns = [desc[0] for desc in cur.description[:-1]]  # Exclude relevance_score
            for row in rows:
                data = dict(zip(columns, row[:-1], strict=False))
                session = self._row_to_session_aggregate(data)
                score = float(row[-1]) if row[-1] is not None else 0.0
                results.append((session, score))

            return results

    @property
    def search_capabilities(self) -> dict[str, bool]:
        """Get available search capabilities.

        Returns:
            Dict with capability names and availability status.
        """
        return {
            "trigram_search": self._has_pg_trgm,
            "bm25_search": self._has_pg_search,
            "sql_memory_ranking": self._has_rank_memory,
            "sql_session_ranking": self._has_rank_session,
            "vector_search": self._has_pgvector,
            "session_triggers": self._has_session_trigger,
        }

    @property
    def _has_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        if not hasattr(self, "_pgvector_available"):
            self._pgvector_available = self._check_extension("vector")
        return self._pgvector_available

    @property
    def _has_session_trigger(self) -> bool:
        """Check if session aggregate trigger is active."""
        if not hasattr(self, "_session_trigger_available"):
            try:
                with self._pool.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_trigger WHERE tgname = 'trg_memory_session_aggregate'"
                    )
                    self._session_trigger_available = cur.fetchone() is not None
            except Exception:
                self._session_trigger_available = False
        return self._session_trigger_available

    # ==========================================================================
    # Vector Search Methods (pgvector)
    # ==========================================================================

    def search_vector(
        self,
        embedding: list[float],
        user_id: str,
        topics: list[str] | None = None,
        min_importance: float = 0.0,
        min_similarity: float = 0.5,
        limit: int = 50,
    ) -> list[tuple[Memory, float]]:
        """Search memories by vector similarity.

        Uses pgvector for efficient approximate nearest neighbor search.
        Requires pgvector extension and embedding_vector column.

        Args:
            embedding: Query embedding vector (must match dimension, typically 1536)
            user_id: Filter by user
            topics: Optional topic filter
            min_importance: Minimum importance threshold
            min_similarity: Minimum cosine similarity (0-1)
            limit: Maximum results to return

        Returns:
            List of (Memory, similarity_score) tuples, sorted by similarity descending

        Raises:
            StorageError: If pgvector is not available
        """
        if not self._has_pgvector:
            raise StorageError(
                "pgvector extension required for search_vector(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )

        # Convert embedding to vector string format
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        conditions = ["m.user_id = %s", "m.embedding_vector IS NOT NULL"]
        params: list[Any] = [user_id]

        # Expiration filter
        conditions.append("(m.expires_at IS NULL OR m.expires_at > NOW())")

        # Importance filter
        if min_importance > 0:
            conditions.append("m.importance >= %s")
            params.append(min_importance)

        # Topic filter
        if topics:
            conditions.append("m.topics ?| %s")
            params.append(topics)

        # Similarity filter (cosine distance < 1 - min_similarity)
        conditions.append("1 - (m.embedding_vector <=> %s::vector) >= %s")
        params.extend([embedding_str, min_similarity])

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT m.*,
                   1 - (m.embedding_vector <=> %s::vector) AS similarity
            FROM memories m
            WHERE {where_clause}
            ORDER BY m.embedding_vector <=> %s::vector
            LIMIT %s
        """
        params.extend([embedding_str, embedding_str, limit])

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                # Last column is similarity
                memory = self._row_to_memory(row[:-1], cur.description[:-1])
                score = float(row[-1]) if row[-1] is not None else 0.0
                results.append((memory, score))

            return results

    def search_hybrid(
        self,
        query: str,
        embedding: list[float],
        user_id: str,
        attention_hints: list[str] | None = None,
        semantic_weight: float = 0.6,
        limit: int = 50,
    ) -> list[tuple[Memory, float]]:
        """Hybrid search combining vector similarity and keyword relevance.

        Uses Reciprocal Rank Fusion (RRF) to combine semantic and keyword rankings.
        Requires pgvector extension.

        Args:
            query: Text query for keyword matching
            embedding: Query embedding for semantic matching
            user_id: Filter by user
            attention_hints: Topics to prioritize
            semantic_weight: Weight for semantic vs keyword (0-1)
            limit: Maximum results to return

        Returns:
            List of (Memory, hybrid_score) tuples

        Raises:
            StorageError: If pgvector is not available
        """
        if not self._has_pgvector:
            raise StorageError(
                "pgvector extension required for search_hybrid(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )

        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        hints_array = f"ARRAY{attention_hints!r}" if attention_hints else "NULL"

        sql = f"""
            SELECT * FROM search_memories_hybrid(
                %s::vector,
                %s,
                %s,
                {hints_array}::text[],
                %s,
                %s
            )
        """
        params = [embedding_str, query, user_id, limit, semantic_weight]

        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    # Columns: memory_id, content, memory_type, topics, importance,
                    #          semantic_score, keyword_score, hybrid_score
                    memory = Memory(
                        memory_id=row[0],
                        content=row[1],
                        memory_type=row[2],
                        topics=row[3] if isinstance(row[3], list) else [],
                        importance=row[4] or 0.5,
                        user_id=user_id,
                    )
                    score = float(row[7]) if row[7] is not None else 0.0
                    results.append((memory, score))

                return results
        except Exception as e:
            # Fall back to vector-only search if hybrid function not available
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return self.search_vector(
                embedding=embedding,
                user_id=user_id,
                topics=attention_hints,
                limit=limit,
            )

    def find_similar_memories(
        self,
        memory_id: str,
        min_similarity: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Find memories similar to a given memory.

        Useful for deduplication, clustering, or "related memories" features.

        Args:
            memory_id: Source memory ID
            min_similarity: Minimum similarity threshold
            limit: Maximum results

        Returns:
            List of (Memory, similarity) tuples
        """
        if not self._has_pgvector:
            raise StorageError(
                "pgvector extension required for find_similar_memories(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )

        sql = """
            SELECT m.*, 1 - (m.embedding_vector <=> src.embedding_vector) AS similarity
            FROM memories m
            CROSS JOIN (SELECT embedding_vector, user_id FROM memories WHERE memory_id = %s) src
            WHERE m.memory_id != %s
              AND m.user_id = src.user_id
              AND m.embedding_vector IS NOT NULL
              AND 1 - (m.embedding_vector <=> src.embedding_vector) >= %s
            ORDER BY m.embedding_vector <=> src.embedding_vector
            LIMIT %s
        """

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (memory_id, memory_id, min_similarity, limit))
            rows = cur.fetchall()

            results = []
            for row in rows:
                memory = self._row_to_memory(row[:-1], cur.description[:-1])
                score = float(row[-1]) if row[-1] is not None else 0.0
                results.append((memory, score))

            return results

    def store_embedding(self, memory_id: str, embedding: list[float]) -> None:
        """Store or update embedding vector for a memory.

        Args:
            memory_id: Memory to update
            embedding: Embedding vector (must match configured dimension)

        Raises:
            MemoryNotFoundError: If memory doesn't exist
            StorageError: If pgvector is not available
        """
        if not self._has_pgvector:
            raise StorageError(
                "pgvector extension required for store_embedding(). "
                "See mindcore/storage/schema/README.md for setup instructions."
            )

        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET embedding_vector = %s::vector,
                    embedding = %s::jsonb
                WHERE memory_id = %s
                """,
                (embedding_str, json.dumps(embedding), memory_id),
            )
            conn.commit()

            if cur.rowcount == 0:
                raise MemoryNotFoundError(memory_id)

    # ==========================================================================
    # Materialized View Methods
    # ==========================================================================

    def refresh_materialized_views(self, critical_only: bool = False) -> dict[str, float]:
        """Refresh materialized views.

        Args:
            critical_only: If True, only refresh user_stats and session_stats

        Returns:
            Dict mapping view name to refresh time in seconds
        """
        results = {}

        if critical_only:
            views = ["mv_user_stats", "mv_session_stats"]
        else:
            views = [
                "mv_user_stats",
                "mv_session_stats",
                "mv_topic_analytics",
                "mv_memory_health",
                "mv_daily_stats",
            ]

        with self._pool.connection() as conn, conn.cursor() as cur:
            for view in views:
                try:
                    import time

                    start = time.perf_counter()
                    cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                    conn.commit()
                    results[view] = time.perf_counter() - start
                except Exception as e:
                    logger.warning(f"Failed to refresh {view}: {e}")
                    results[view] = -1.0

        return results

    def get_user_stats(self, user_id: str) -> dict[str, Any] | None:
        """Get pre-computed user statistics from materialized view.

        Args:
            user_id: User to get stats for

        Returns:
            Dict with user statistics or None if not found
        """
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM mv_user_stats WHERE user_id = %s", (user_id,))
                row = cur.fetchone()

                if not row:
                    return None

                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row, strict=False))
        except Exception as e:
            logger.warning(f"Failed to get user stats (view may not exist): {e}")
            return None

    def get_memory_health(self) -> dict[str, Any]:
        """Get system health metrics from materialized view.

        Returns:
            Dict with health metrics
        """
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM mv_memory_health LIMIT 1")
                row = cur.fetchone()

                if not row:
                    return {}

                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row, strict=False))
        except Exception as e:
            logger.warning(f"Failed to get memory health (view may not exist): {e}")
            return {}
