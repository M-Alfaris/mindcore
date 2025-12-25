"""Time-based partitioning for PostgreSQL storage.

Provides automatic table partitioning by time ranges for improved performance
at scale. Partitions can be managed automatically or manually.

Benefits:
- Faster queries when filtering by time
- Easier data archival and cleanup
- Parallel query execution across partitions
- Smaller indexes per partition

Example:
    from mindcore.v2.storage import PostgresStorage
    from mindcore.v2.storage.partitioning import PartitionManager

    storage = PostgresStorage("postgresql://...")
    partitions = PartitionManager(storage)

    # Setup partitioning (one-time)
    partitions.setup_partitioning(interval="monthly")

    # Create partitions for next 3 months
    partitions.create_future_partitions(months_ahead=3)

    # Maintenance: archive old partitions
    partitions.archive_partitions(older_than_months=12)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.v2.storage.postgres import PostgresStorage


class PartitionInterval(str, Enum):
    """Partition interval options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class PartitionInfo:
    """Information about a partition."""

    name: str
    start_date: datetime
    end_date: datetime
    row_count: int
    size_bytes: int
    size_pretty: str


@dataclass
class PartitioningStatus:
    """Status of partitioning setup."""

    is_partitioned: bool
    interval: PartitionInterval | None
    partitions: list[PartitionInfo]
    total_partitions: int
    total_rows: int
    total_size_bytes: int
    total_size_pretty: str
    oldest_partition: str | None
    newest_partition: str | None


class PartitionManager:
    """Manages time-based partitioning for PostgreSQL memories table.

    PostgreSQL native partitioning provides:
    - Partition pruning (only scan relevant partitions)
    - Parallel query execution
    - Easy maintenance (drop old partitions)
    - Better vacuum performance
    """

    # Partition naming pattern: memories_YYYY_MM or memories_YYYY_WW
    PARTITION_PREFIX = "memories_p"

    def __init__(self, storage: PostgresStorage):
        """Initialize partition manager.

        Args:
            storage: PostgresStorage instance with active connection pool
        """
        self.storage = storage
        self._pool = storage._pool

    def setup_partitioning(
        self,
        interval: PartitionInterval | str = PartitionInterval.MONTHLY,
        migrate_existing: bool = True,
    ) -> bool:
        """Setup partitioned table structure.

        This converts the existing memories table to a partitioned table.
        Should only be run once during initial setup.

        WARNING: This operation requires exclusive table access and may
        take time for large tables. Run during maintenance window.

        Args:
            interval: Partition interval (daily, weekly, monthly, quarterly, yearly)
            migrate_existing: If True, migrate existing data to partitions

        Returns:
            True if setup succeeded
        """
        if isinstance(interval, str):
            interval = PartitionInterval(interval)

        with self._pool.connection() as conn, conn.cursor() as cur:
            # Check if already partitioned
            cur.execute("""
                SELECT relkind FROM pg_class
                WHERE relname = 'memories'
            """)
            result = cur.fetchone()

            if result and result[0] == "p":
                # Already partitioned
                return True

            # Create partitioned table structure
            # First, rename existing table
            cur.execute("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'memories') THEN
                        ALTER TABLE memories RENAME TO memories_old;
                    END IF;
                END $$;
            """)

            # Create partitioned table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    -- Primary identifiers
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,

                    -- Conversation tracking
                    session_id TEXT,
                    thread_id TEXT,
                    parent_memory_id TEXT,
                    turn_index INTEGER,

                    -- Content and classification
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    message_role TEXT,

                    -- Metadata arrays (GIN indexed)
                    topics JSONB DEFAULT '[]'::jsonb,
                    categories JSONB DEFAULT '[]'::jsonb,
                    entities JSONB DEFAULT '[]'::jsonb,

                    -- Sentiment and scores
                    sentiment TEXT DEFAULT 'neutral',
                    importance REAL DEFAULT 0.5,
                    confidence_score REAL,
                    reinforcement_score REAL DEFAULT 0.0,

                    -- Access control
                    access_level TEXT DEFAULT 'private',
                    access_count INTEGER DEFAULT 0,

                    -- Timestamps (partition key)
                    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                    last_accessed TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ,

                    -- Extended semantic metadata
                    semantic_metadata JSONB DEFAULT '{}'::jsonb,

                    -- Versioning
                    vocabulary_version TEXT DEFAULT '1.0.0',

                    -- Embeddings
                    embedding JSONB,

                    -- Full-text search vector
                    search_vector tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(topics::text, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(entities::text, '')), 'C')
                    ) STORED,

                    -- Primary key includes partition key
                    PRIMARY KEY (memory_id, created_at)
                ) PARTITION BY RANGE (created_at);
            """)

            # Store partitioning metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _partition_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cur.execute(
                """
                INSERT INTO _partition_config (key, value)
                VALUES ('interval', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
                (interval.value,),
            )

            # Create default partition for any data that doesn't fit
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories_default
                PARTITION OF memories DEFAULT
            """)

            # Create initial partitions
            self._create_partition_range(cur, interval, months_back=1, months_ahead=3)

            # Migrate existing data if requested
            if migrate_existing:
                cur.execute("""
                    DO $$
                    BEGIN
                        IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'memories_old') THEN
                            INSERT INTO memories SELECT * FROM memories_old;
                            DROP TABLE memories_old;
                        END IF;
                    END $$;
                """)

            # Recreate indexes on partitioned table
            self._create_partition_indexes(cur)

            conn.commit()
            return True

    def create_future_partitions(
        self,
        months_ahead: int = 3,
    ) -> list[str]:
        """Create partitions for future months.

        Should be run periodically (e.g., monthly cron job) to ensure
        partitions exist before data arrives.

        Args:
            months_ahead: Number of months to create partitions for

        Returns:
            List of created partition names
        """
        interval = self._get_interval()
        if not interval:
            return []

        created = []
        with self._pool.connection() as conn, conn.cursor() as cur:
            now = datetime.now(timezone.utc)

            for i in range(months_ahead):
                if interval == PartitionInterval.MONTHLY:
                    target = now + timedelta(days=30 * (i + 1))
                    partition_name = self._create_monthly_partition(cur, target)
                elif interval == PartitionInterval.WEEKLY:
                    target = now + timedelta(weeks=i + 1)
                    partition_name = self._create_weekly_partition(cur, target)
                elif interval == PartitionInterval.DAILY:
                    target = now + timedelta(days=i + 1)
                    partition_name = self._create_daily_partition(cur, target)
                elif interval == PartitionInterval.QUARTERLY:
                    target = now + timedelta(days=90 * (i + 1))
                    partition_name = self._create_quarterly_partition(cur, target)
                elif interval == PartitionInterval.YEARLY:
                    target = now + timedelta(days=365 * (i + 1))
                    partition_name = self._create_yearly_partition(cur, target)
                else:
                    continue

                if partition_name:
                    created.append(partition_name)

            conn.commit()

        return created

    def archive_partitions(
        self,
        older_than_months: int = 12,
        detach_only: bool = True,
    ) -> list[str]:
        """Archive old partitions.

        Args:
            older_than_months: Archive partitions older than this
            detach_only: If True, detach but don't drop (safer)

        Returns:
            List of archived partition names
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * older_than_months)
        archived = []

        with self._pool.connection() as conn, conn.cursor() as cur:
            # Get all partitions
            cur.execute("""
                SELECT c.relname, pg_get_expr(c.relpartbound, c.oid)
                FROM pg_class c
                JOIN pg_inherits i ON c.oid = i.inhrelid
                JOIN pg_class p ON i.inhparent = p.oid
                WHERE p.relname = 'memories'
                  AND c.relname != 'memories_default'
            """)

            for row in cur.fetchall():
                partition_name = row[0]
                bound_expr = row[1]

                # Parse bound expression to get end date
                # Format: FOR VALUES FROM ('2024-01-01') TO ('2024-02-01')
                try:
                    if "TO ('" in bound_expr:
                        end_str = bound_expr.split("TO ('")[1].split("')")[0]
                        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

                        if end_date < cutoff:
                            if detach_only:
                                cur.execute(
                                    f"ALTER TABLE memories DETACH PARTITION {partition_name}"
                                )
                            else:
                                cur.execute(f"DROP TABLE {partition_name}")
                            archived.append(partition_name)
                except (IndexError, ValueError):
                    continue

            conn.commit()

        return archived

    def get_status(self) -> PartitioningStatus:
        """Get current partitioning status.

        Returns:
            PartitioningStatus with partition details
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            # Check if partitioned
            cur.execute("""
                SELECT relkind FROM pg_class WHERE relname = 'memories'
            """)
            result = cur.fetchone()
            is_partitioned = result and result[0] == "p"

            if not is_partitioned:
                return PartitioningStatus(
                    is_partitioned=False,
                    interval=None,
                    partitions=[],
                    total_partitions=0,
                    total_rows=0,
                    total_size_bytes=0,
                    total_size_pretty="0 bytes",
                    oldest_partition=None,
                    newest_partition=None,
                )

            # Get interval
            interval = self._get_interval()

            # Get partition info
            cur.execute("""
                SELECT
                    c.relname as partition_name,
                    pg_get_expr(c.relpartbound, c.oid) as bound,
                    COALESCE((SELECT reltuples::bigint FROM pg_class WHERE relname = c.relname), 0) as row_estimate,
                    pg_total_relation_size(c.oid) as size_bytes,
                    pg_size_pretty(pg_total_relation_size(c.oid)) as size_pretty
                FROM pg_class c
                JOIN pg_inherits i ON c.oid = i.inhrelid
                JOIN pg_class p ON i.inhparent = p.oid
                WHERE p.relname = 'memories'
                ORDER BY c.relname
            """)

            partitions = []
            total_rows = 0
            total_size = 0
            oldest = None
            newest = None

            for row in cur.fetchall():
                name = row[0]
                bound = row[1]
                rows = row[2]
                size = row[3]
                size_pretty = row[4]

                total_rows += rows
                total_size += size

                # Parse dates from bound
                start_date = None
                end_date = None
                try:
                    if "FROM ('" in bound:
                        start_str = bound.split("FROM ('")[1].split("')")[0]
                        start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    if "TO ('" in bound:
                        end_str = bound.split("TO ('")[1].split("')")[0]
                        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (IndexError, ValueError):
                    pass

                if start_date and end_date:
                    partitions.append(
                        PartitionInfo(
                            name=name,
                            start_date=start_date,
                            end_date=end_date,
                            row_count=rows,
                            size_bytes=size,
                            size_pretty=size_pretty,
                        )
                    )

                    if oldest is None or (start_date and start_date < partitions[0].start_date):
                        oldest = name
                    if newest is None or (end_date and end_date > partitions[-1].end_date):
                        newest = name

            # Get total size pretty
            cur.execute("SELECT pg_size_pretty(%s::bigint)", (total_size,))
            total_size_pretty = cur.fetchone()[0]

            return PartitioningStatus(
                is_partitioned=True,
                interval=interval,
                partitions=partitions,
                total_partitions=len(partitions),
                total_rows=total_rows,
                total_size_bytes=total_size,
                total_size_pretty=total_size_pretty,
                oldest_partition=oldest,
                newest_partition=newest,
            )

    def _get_interval(self) -> PartitionInterval | None:
        """Get configured partition interval."""
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT value FROM _partition_config WHERE key = 'interval'")
                result = cur.fetchone()
                if result:
                    return PartitionInterval(result[0])
        except Exception:
            pass
        return None

    def _create_partition_range(
        self,
        cur: Any,
        interval: PartitionInterval,
        months_back: int,
        months_ahead: int,
    ) -> None:
        """Create a range of partitions."""
        now = datetime.now(timezone.utc)

        for i in range(-months_back, months_ahead + 1):
            target = now + timedelta(days=30 * i)

            if interval == PartitionInterval.MONTHLY:
                self._create_monthly_partition(cur, target)
            elif interval == PartitionInterval.WEEKLY:
                # Create ~4 weeks per month
                for w in range(4):
                    week_target = target + timedelta(weeks=w)
                    self._create_weekly_partition(cur, week_target)
            elif interval == PartitionInterval.DAILY:
                # Create ~30 days per month
                for d in range(30):
                    day_target = target + timedelta(days=d)
                    self._create_daily_partition(cur, day_target)
            elif interval == PartitionInterval.QUARTERLY:
                self._create_quarterly_partition(cur, target)
            elif interval == PartitionInterval.YEARLY:
                self._create_yearly_partition(cur, target)

    def _create_monthly_partition(self, cur: Any, target: datetime) -> str | None:
        """Create a monthly partition."""
        year = target.year
        month = target.month
        partition_name = f"{self.PARTITION_PREFIX}{year}_{month:02d}"

        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        return self._create_partition(cur, partition_name, start, end)

    def _create_weekly_partition(self, cur: Any, target: datetime) -> str | None:
        """Create a weekly partition."""
        year, week, _ = target.isocalendar()
        partition_name = f"{self.PARTITION_PREFIX}{year}_w{week:02d}"

        # Get Monday of this week
        start = target - timedelta(days=target.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end = start + timedelta(weeks=1)

        return self._create_partition(cur, partition_name, start, end)

    def _create_daily_partition(self, cur: Any, target: datetime) -> str | None:
        """Create a daily partition."""
        partition_name = f"{self.PARTITION_PREFIX}{target.strftime('%Y_%m_%d')}"

        start = target.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        return self._create_partition(cur, partition_name, start, end)

    def _create_quarterly_partition(self, cur: Any, target: datetime) -> str | None:
        """Create a quarterly partition."""
        year = target.year
        quarter = (target.month - 1) // 3 + 1
        partition_name = f"{self.PARTITION_PREFIX}{year}_q{quarter}"

        start_month = (quarter - 1) * 3 + 1
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)

        if quarter == 4:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, start_month + 3, 1, tzinfo=timezone.utc)

        return self._create_partition(cur, partition_name, start, end)

    def _create_yearly_partition(self, cur: Any, target: datetime) -> str | None:
        """Create a yearly partition."""
        year = target.year
        partition_name = f"{self.PARTITION_PREFIX}{year}"

        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        return self._create_partition(cur, partition_name, start, end)

    def _create_partition(
        self,
        cur: Any,
        name: str,
        start: datetime,
        end: datetime,
    ) -> str | None:
        """Create a partition if it doesn't exist."""
        try:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {name}
                PARTITION OF memories
                FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')
            """)
            return name
        except Exception:
            # Partition might already exist or overlap
            return None

    def _create_partition_indexes(self, cur: Any) -> None:
        """Create indexes on the partitioned table."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_thread_id ON memories(thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_parent_id ON memories(parent_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_conversation ON memories(user_id, session_id, thread_id, turn_index)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_message_role ON memories(message_role)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_version ON memories(vocabulary_version)",
            "CREATE INDEX IF NOT EXISTS idx_memories_topics ON memories USING GIN(topics)",
            "CREATE INDEX IF NOT EXISTS idx_memories_categories ON memories USING GIN(categories)",
            "CREATE INDEX IF NOT EXISTS idx_memories_entities ON memories USING GIN(entities)",
            "CREATE INDEX IF NOT EXISTS idx_memories_semantic_metadata ON memories USING GIN(semantic_metadata)",
            "CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN(search_vector)",
            "CREATE INDEX IF NOT EXISTS idx_memories_access_level ON memories(access_level)",
            "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence_score) WHERE confidence_score IS NOT NULL",
        ]

        for idx in indexes:
            try:
                cur.execute(idx)
            except Exception:
                pass  # Index might already exist
