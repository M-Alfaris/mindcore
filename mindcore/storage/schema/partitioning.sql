-- Mindcore Table Partitioning
--
-- Implements table partitioning for the memories table to improve:
-- - Query performance for user-specific or time-range queries
-- - Maintenance operations (VACUUM, archive by partition)
-- - Parallel query execution
--
-- Partitioning Strategy:
--   - Primary: RANGE by created_at (monthly partitions)
--   - This allows efficient time-range queries and easy archival
--
-- Alternative strategies included:
--   - HASH by user_id (for user isolation)
--   - LIST by memory_type (for type-specific queries)
--
-- Requirements:
--   - PostgreSQL 12+ (for partitioning improvements)
--   - Downtime for migration (or use pg_partman for online)
--
-- IMPORTANT: This is a destructive migration. Back up data first!
-- Run with: psql $DATABASE_URL -f partitioning.sql

-- ==========================================================================
-- Strategy 1: RANGE Partitioning by created_at (Recommended)
-- ==========================================================================

-- Step 1: Create partitioned table structure
-- This creates a NEW table - does not modify existing table
CREATE TABLE IF NOT EXISTS memories_partitioned (
    memory_id TEXT NOT NULL,
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    reinforcement_score REAL DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    vocabulary_version TEXT DEFAULT '1.0.0',
    embedding JSONB,
    embedding_vector vector(1536),
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(topics::text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(entities::text, '')), 'C')
    ) STORED,
    PRIMARY KEY (memory_id, created_at)
) PARTITION BY RANGE (created_at);

-- Step 2: Create partitions for existing and future data
-- Monthly partitions for past year and next year

-- Past year partitions (adjust dates as needed)
CREATE TABLE IF NOT EXISTS memories_y2024m01 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE IF NOT EXISTS memories_y2024m02 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE IF NOT EXISTS memories_y2024m03 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE IF NOT EXISTS memories_y2024m04 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE IF NOT EXISTS memories_y2024m05 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE IF NOT EXISTS memories_y2024m06 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE IF NOT EXISTS memories_y2024m07 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE IF NOT EXISTS memories_y2024m08 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE IF NOT EXISTS memories_y2024m09 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE IF NOT EXISTS memories_y2024m10 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE IF NOT EXISTS memories_y2024m11 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE IF NOT EXISTS memories_y2024m12 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- 2025 partitions
CREATE TABLE IF NOT EXISTS memories_y2025m01 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE IF NOT EXISTS memories_y2025m02 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE IF NOT EXISTS memories_y2025m03 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE IF NOT EXISTS memories_y2025m04 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE IF NOT EXISTS memories_y2025m05 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE IF NOT EXISTS memories_y2025m06 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE IF NOT EXISTS memories_y2025m07 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE IF NOT EXISTS memories_y2025m08 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE IF NOT EXISTS memories_y2025m09 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE IF NOT EXISTS memories_y2025m10 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE IF NOT EXISTS memories_y2025m11 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE IF NOT EXISTS memories_y2025m12 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- 2026 partitions
CREATE TABLE IF NOT EXISTS memories_y2026m01 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS memories_y2026m02 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE IF NOT EXISTS memories_y2026m03 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS memories_y2026m04 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS memories_y2026m05 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE IF NOT EXISTS memories_y2026m06 PARTITION OF memories_partitioned
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

-- Default partition for any data outside defined ranges
CREATE TABLE IF NOT EXISTS memories_default PARTITION OF memories_partitioned
    DEFAULT;

-- Step 3: Create indexes on partitioned table
-- These will be automatically applied to all partitions

CREATE INDEX IF NOT EXISTS idx_memories_part_user_id
ON memories_partitioned (user_id);

CREATE INDEX IF NOT EXISTS idx_memories_part_session_id
ON memories_partitioned (session_id)
WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memories_part_type
ON memories_partitioned (memory_type);

CREATE INDEX IF NOT EXISTS idx_memories_part_created
ON memories_partitioned (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_part_topics
ON memories_partitioned USING GIN (topics);

CREATE INDEX IF NOT EXISTS idx_memories_part_search
ON memories_partitioned USING GIN (search_vector);

-- Vector index (if using pgvector)
-- CREATE INDEX IF NOT EXISTS idx_memories_part_embedding
-- ON memories_partitioned USING hnsw (embedding_vector vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- ==========================================================================
-- Migration Functions
-- ==========================================================================

-- Migrate data from old table to partitioned table
-- Run during maintenance window
CREATE OR REPLACE FUNCTION migrate_to_partitioned_table(
    p_batch_size INTEGER DEFAULT 10000,
    p_sleep_ms INTEGER DEFAULT 100
) RETURNS TABLE (
    batch_num INTEGER,
    rows_migrated INTEGER,
    last_id TEXT
) AS $$
DECLARE
    v_batch INTEGER := 0;
    v_count INTEGER;
    v_last_id TEXT := '';
    v_total INTEGER := 0;
BEGIN
    LOOP
        v_batch := v_batch + 1;

        -- Insert batch
        WITH batch AS (
            SELECT *
            FROM memories
            WHERE memory_id > v_last_id
            ORDER BY memory_id
            LIMIT p_batch_size
        )
        INSERT INTO memories_partitioned (
            memory_id, content, memory_type, user_id, agent_id,
            topics, categories, sentiment, importance, entities,
            access_level, session_id, message_index,
            created_at, last_accessed, expires_at,
            reinforcement_score, access_count, vocabulary_version,
            embedding, embedding_vector
        )
        SELECT
            memory_id, content, memory_type, user_id, agent_id,
            topics, categories, sentiment, importance, entities,
            access_level, session_id, message_index,
            COALESCE(created_at, NOW()), last_accessed, expires_at,
            reinforcement_score, access_count, vocabulary_version,
            embedding, embedding_vector
        FROM batch
        ON CONFLICT (memory_id, created_at) DO NOTHING;

        GET DIAGNOSTICS v_count = ROW_COUNT;

        IF v_count = 0 THEN
            EXIT;  -- No more rows
        END IF;

        -- Get last ID for next batch
        SELECT memory_id INTO v_last_id
        FROM memories
        WHERE memory_id > v_last_id
        ORDER BY memory_id
        OFFSET p_batch_size - 1
        LIMIT 1;

        IF v_last_id IS NULL THEN
            EXIT;
        END IF;

        v_total := v_total + v_count;

        batch_num := v_batch;
        rows_migrated := v_count;
        last_id := v_last_id;
        RETURN NEXT;

        -- Sleep to reduce load
        PERFORM pg_sleep(p_sleep_ms / 1000.0);
    END LOOP;

    RAISE NOTICE 'Migration complete. Total rows: %', v_total;
END;
$$ LANGUAGE plpgsql;

-- Swap tables after migration
-- WARNING: This renames tables - ensure migration is complete
CREATE OR REPLACE FUNCTION swap_to_partitioned_table()
RETURNS VOID AS $$
BEGIN
    -- Verify migration is complete
    IF (SELECT COUNT(*) FROM memories) > (SELECT COUNT(*) FROM memories_partitioned) THEN
        RAISE EXCEPTION 'Migration incomplete. Old table has more rows than new table.';
    END IF;

    -- Rename tables
    ALTER TABLE memories RENAME TO memories_old;
    ALTER TABLE memories_partitioned RENAME TO memories;

    -- Update sequences if any
    -- (memory_id is TEXT, so no sequence needed)

    RAISE NOTICE 'Tables swapped. Old table renamed to memories_old.';
    RAISE NOTICE 'Verify data, then run: DROP TABLE memories_old;';
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Partition Management Functions
-- ==========================================================================

-- Create partition for a new month
CREATE OR REPLACE FUNCTION create_monthly_partition(
    p_year INTEGER,
    p_month INTEGER
) RETURNS TEXT AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := format('memories_y%sm%s', p_year, LPAD(p_month::TEXT, 2, '0'));
    start_date := make_date(p_year, p_month, 1);
    end_date := start_date + INTERVAL '1 month';

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF memories
         FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );

    RETURN partition_name;
END;
$$ LANGUAGE plpgsql;

-- Create partitions for the next N months
CREATE OR REPLACE FUNCTION create_future_partitions(p_months INTEGER DEFAULT 12)
RETURNS TABLE (partition_name TEXT) AS $$
DECLARE
    v_date DATE := DATE_TRUNC('month', NOW());
    i INTEGER;
BEGIN
    FOR i IN 1..p_months LOOP
        v_date := v_date + INTERVAL '1 month';
        partition_name := create_monthly_partition(
            EXTRACT(YEAR FROM v_date)::INTEGER,
            EXTRACT(MONTH FROM v_date)::INTEGER
        );
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Detach old partition (for archival)
CREATE OR REPLACE FUNCTION detach_partition(p_partition_name TEXT)
RETURNS VOID AS $$
BEGIN
    EXECUTE format('ALTER TABLE memories DETACH PARTITION %I', p_partition_name);
    RAISE NOTICE 'Partition % detached. You can now archive or drop it.', p_partition_name;
END;
$$ LANGUAGE plpgsql;

-- Archive old partition to separate tablespace (if configured)
CREATE OR REPLACE FUNCTION archive_partition(
    p_partition_name TEXT,
    p_archive_tablespace TEXT DEFAULT 'pg_default'
) RETURNS VOID AS $$
BEGIN
    -- Move to archive tablespace
    EXECUTE format(
        'ALTER TABLE %I SET TABLESPACE %I',
        p_partition_name, p_archive_tablespace
    );

    RAISE NOTICE 'Partition % moved to tablespace %', p_partition_name, p_archive_tablespace;
END;
$$ LANGUAGE plpgsql;

-- List all partitions
CREATE OR REPLACE FUNCTION list_partitions()
RETURNS TABLE (
    partition_name TEXT,
    partition_expression TEXT,
    row_count BIGINT,
    size_bytes BIGINT,
    size_pretty TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.relname::TEXT,
        pg_get_expr(c.relpartbound, c.oid)::TEXT,
        (SELECT COUNT(*) FROM ONLY pg_class pc WHERE pc.oid = c.oid),
        pg_total_relation_size(c.oid),
        pg_size_pretty(pg_total_relation_size(c.oid))
    FROM pg_class c
    JOIN pg_inherits i ON c.oid = i.inhrelid
    JOIN pg_class p ON i.inhparent = p.oid
    WHERE p.relname = 'memories'
      AND p.relkind = 'p'
    ORDER BY c.relname;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Alternative: HASH Partitioning by user_id
-- ==========================================================================

-- Use this instead of RANGE if you have:
-- - Many concurrent users
-- - Most queries are user-specific
-- - Even distribution is more important than time-based queries

/*
CREATE TABLE IF NOT EXISTS memories_by_user (
    LIKE memories INCLUDING ALL
) PARTITION BY HASH (user_id);

-- Create 16 partitions (adjust based on expected user count)
CREATE TABLE memories_by_user_p0 PARTITION OF memories_by_user FOR VALUES WITH (MODULUS 16, REMAINDER 0);
CREATE TABLE memories_by_user_p1 PARTITION OF memories_by_user FOR VALUES WITH (MODULUS 16, REMAINDER 1);
-- ... repeat for p2 through p15
*/

-- ==========================================================================
-- Verification
-- ==========================================================================

-- Check partitions exist
-- SELECT * FROM list_partitions();

-- Check partition pruning works
-- EXPLAIN SELECT * FROM memories WHERE created_at = '2025-01-15';

-- Check data distribution
-- SELECT
--     tableoid::regclass AS partition,
--     COUNT(*) AS row_count
-- FROM memories
-- GROUP BY tableoid
-- ORDER BY tableoid;

-- ==========================================================================
-- Automatic Partition Maintenance (pg_cron job)
-- ==========================================================================

-- Schedule partition creation (run monthly)
-- SELECT safe_schedule('0 0 1 * *', 'SELECT create_future_partitions(3)', 'create_partitions');

-- Schedule old partition archival (run monthly)
-- Detach and archive partitions older than 2 years
-- SELECT safe_schedule('0 1 1 * *',
--     $$SELECT detach_partition(partition_name) FROM list_partitions()
--       WHERE partition_expression LIKE '%TO%' || (NOW() - INTERVAL '2 years')::DATE$$,
--     'archive_old_partitions');
