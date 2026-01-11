-- Mindcore pg_stat_statements Query Monitoring
--
-- Enables query performance monitoring and analysis.
-- Tracks execution statistics for all SQL statements.
--
-- Benefits:
-- - Identify slow queries
-- - Find most frequently executed queries
-- - Track query performance over time
-- - Detect inefficient query patterns
--
-- Requirements:
--   - PostgreSQL 12+ with pg_stat_statements extension
--   - pg_stat_statements must be configured in postgresql.conf:
--       shared_preload_libraries = 'pg_stat_statements'
--       pg_stat_statements.track = all
--       pg_stat_statements.max = 10000
--
-- Run with: psql $DATABASE_URL -f pg_stat_statements.sql

-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ==========================================================================
-- Query Analysis Views
-- ==========================================================================

-- Top slow queries (by total time)
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT
    queryid,
    LEFT(query, 100) AS query_preview,
    calls,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(min_exec_time::numeric, 2) AS min_time_ms,
    ROUND(max_exec_time::numeric, 2) AS max_time_ms,
    ROUND(stddev_exec_time::numeric, 2) AS stddev_ms,
    rows,
    ROUND((100.0 * total_exec_time / NULLIF(SUM(total_exec_time) OVER (), 0))::numeric, 2) AS pct_total_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
  AND query NOT LIKE 'COMMIT%'
  AND query NOT LIKE 'BEGIN%'
ORDER BY total_exec_time DESC
LIMIT 50;

-- Most frequently called queries
CREATE OR REPLACE VIEW v_frequent_queries AS
SELECT
    queryid,
    LEFT(query, 100) AS query_preview,
    calls,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    rows,
    ROUND(rows::numeric / NULLIF(calls, 0), 2) AS avg_rows_per_call,
    ROUND((100.0 * calls / NULLIF(SUM(calls) OVER (), 0))::numeric, 2) AS pct_total_calls
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
  AND query NOT LIKE 'COMMIT%'
  AND query NOT LIKE 'BEGIN%'
ORDER BY calls DESC
LIMIT 50;

-- Queries with high I/O
CREATE OR REPLACE VIEW v_io_intensive_queries AS
SELECT
    queryid,
    LEFT(query, 100) AS query_preview,
    calls,
    shared_blks_hit AS cache_hits,
    shared_blks_read AS disk_reads,
    ROUND((100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0))::numeric, 2) AS cache_hit_pct,
    shared_blks_written AS blocks_written,
    temp_blks_read + temp_blks_written AS temp_blocks
FROM pg_stat_statements
WHERE shared_blks_read > 100 OR temp_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 50;

-- Queries with high variance (inconsistent performance)
CREATE OR REPLACE VIEW v_variable_queries AS
SELECT
    queryid,
    LEFT(query, 100) AS query_preview,
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(min_exec_time::numeric, 2) AS min_time_ms,
    ROUND(max_exec_time::numeric, 2) AS max_time_ms,
    ROUND(stddev_exec_time::numeric, 2) AS stddev_ms,
    ROUND((stddev_exec_time / NULLIF(mean_exec_time, 0) * 100)::numeric, 2) AS cv_pct  -- Coefficient of variation
FROM pg_stat_statements
WHERE calls >= 10  -- Need enough samples
  AND stddev_exec_time > 0
  AND query NOT LIKE '%pg_stat_statements%'
ORDER BY (stddev_exec_time / NULLIF(mean_exec_time, 0)) DESC
LIMIT 50;

-- ==========================================================================
-- Mindcore-Specific Query Analysis
-- ==========================================================================

-- Memory search performance
CREATE OR REPLACE VIEW v_memory_search_stats AS
SELECT
    queryid,
    LEFT(query, 150) AS query_preview,
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    rows,
    ROUND(rows::numeric / NULLIF(calls, 0), 2) AS avg_rows
FROM pg_stat_statements
WHERE (
    query LIKE '%FROM memories%'
    OR query LIKE '%search_memories%'
    OR query LIKE '%rank_memory%'
    OR query LIKE '%similarity%'
)
AND query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 30;

-- Session aggregate performance
CREATE OR REPLACE VIEW v_session_aggregate_stats AS
SELECT
    queryid,
    LEFT(query, 150) AS query_preview,
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    rows
FROM pg_stat_statements
WHERE (
    query LIKE '%session_aggregates%'
    OR query LIKE '%rank_session%'
    OR query LIKE '%query_sessions%'
)
AND query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 30;

-- Vector search performance (if using pgvector)
CREATE OR REPLACE VIEW v_vector_search_stats AS
SELECT
    queryid,
    LEFT(query, 150) AS query_preview,
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    rows,
    ROUND(rows::numeric / NULLIF(calls, 0), 2) AS avg_rows
FROM pg_stat_statements
WHERE (
    query LIKE '%embedding_vector%'
    OR query LIKE '%<=>%'  -- Cosine distance operator
    OR query LIKE '%<->%'  -- L2 distance operator
    OR query LIKE '%search_memories_semantic%'
    OR query LIKE '%search_memories_hybrid%'
)
AND query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 30;

-- ==========================================================================
-- Analysis Functions
-- ==========================================================================

-- Get query performance summary
CREATE OR REPLACE FUNCTION get_query_summary()
RETURNS TABLE (
    metric TEXT,
    value TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'Total queries tracked'::TEXT, COUNT(*)::TEXT FROM pg_stat_statements;

    RETURN QUERY
    SELECT 'Total execution time (s)'::TEXT, ROUND(SUM(total_exec_time) / 1000)::TEXT
    FROM pg_stat_statements;

    RETURN QUERY
    SELECT 'Total calls'::TEXT, SUM(calls)::TEXT FROM pg_stat_statements;

    RETURN QUERY
    SELECT 'Avg query time (ms)'::TEXT, ROUND(AVG(mean_exec_time)::numeric, 2)::TEXT
    FROM pg_stat_statements WHERE calls > 0;

    RETURN QUERY
    SELECT 'Slowest query (ms)'::TEXT, ROUND(MAX(max_exec_time)::numeric, 2)::TEXT
    FROM pg_stat_statements;

    RETURN QUERY
    SELECT 'Cache hit ratio (%)'::TEXT,
        ROUND((100.0 * SUM(shared_blks_hit) / NULLIF(SUM(shared_blks_hit + shared_blks_read), 0))::numeric, 2)::TEXT
    FROM pg_stat_statements;
END;
$$ LANGUAGE plpgsql;

-- Find queries that might benefit from indexes
CREATE OR REPLACE FUNCTION suggest_indexes()
RETURNS TABLE (
    query_preview TEXT,
    calls BIGINT,
    avg_time_ms NUMERIC,
    suggestion TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        LEFT(s.query, 100),
        s.calls,
        ROUND(s.mean_exec_time::numeric, 2),
        CASE
            WHEN s.query LIKE '%WHERE%user_id%' AND s.mean_exec_time > 10
                THEN 'Consider index on user_id if not exists'
            WHEN s.query LIKE '%WHERE%session_id%' AND s.mean_exec_time > 10
                THEN 'Consider index on session_id if not exists'
            WHEN s.query LIKE '%WHERE%topics%' AND s.mean_exec_time > 20
                THEN 'Consider GIN index on topics JSONB'
            WHEN s.query LIKE '%ORDER BY created_at%' AND s.mean_exec_time > 10
                THEN 'Consider index on created_at'
            WHEN s.query LIKE '%embedding_vector%' AND s.mean_exec_time > 50
                THEN 'Consider HNSW index on embedding_vector'
            WHEN s.query LIKE '%similarity%' AND s.mean_exec_time > 30
                THEN 'Consider GIN trigram index'
            ELSE 'Review query plan with EXPLAIN ANALYZE'
        END
    FROM pg_stat_statements s
    WHERE s.mean_exec_time > 10
      AND s.calls >= 10
      AND s.query NOT LIKE '%pg_stat_statements%'
    ORDER BY s.mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Track query performance over time (call periodically to build history)
CREATE TABLE IF NOT EXISTS query_performance_history (
    snapshot_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    queryid BIGINT,
    query_preview TEXT,
    calls BIGINT,
    total_time_ms NUMERIC,
    avg_time_ms NUMERIC,
    rows_returned BIGINT
);

CREATE INDEX IF NOT EXISTS idx_query_history_time
ON query_performance_history (snapshot_time DESC);

CREATE OR REPLACE FUNCTION snapshot_query_performance()
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER;
BEGIN
    INSERT INTO query_performance_history (queryid, query_preview, calls, total_time_ms, avg_time_ms, rows_returned)
    SELECT
        queryid,
        LEFT(query, 200),
        calls,
        ROUND(total_exec_time::numeric, 2),
        ROUND(mean_exec_time::numeric, 2),
        rows
    FROM pg_stat_statements
    WHERE calls >= 10
      AND query NOT LIKE '%pg_stat_statements%'
      AND query NOT LIKE 'COMMIT%'
      AND query NOT LIKE 'BEGIN%';

    GET DIAGNOSTICS inserted_count = ROW_COUNT;

    -- Keep only last 7 days of history
    DELETE FROM query_performance_history
    WHERE snapshot_time < NOW() - INTERVAL '7 days';

    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Compare query performance between two time periods
CREATE OR REPLACE FUNCTION compare_query_performance(
    p_current_hours INTEGER DEFAULT 24,
    p_baseline_hours INTEGER DEFAULT 168  -- 7 days
) RETURNS TABLE (
    query_preview TEXT,
    current_avg_ms NUMERIC,
    baseline_avg_ms NUMERIC,
    change_pct NUMERIC,
    current_calls BIGINT,
    baseline_calls BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH current_period AS (
        SELECT queryid, query_preview, AVG(avg_time_ms) as avg_ms, SUM(calls) as total_calls
        FROM query_performance_history
        WHERE snapshot_time > NOW() - (p_current_hours || ' hours')::INTERVAL
        GROUP BY queryid, query_preview
    ),
    baseline_period AS (
        SELECT queryid, AVG(avg_time_ms) as avg_ms, SUM(calls) as total_calls
        FROM query_performance_history
        WHERE snapshot_time BETWEEN NOW() - (p_baseline_hours || ' hours')::INTERVAL
                                AND NOW() - (p_current_hours || ' hours')::INTERVAL
        GROUP BY queryid
    )
    SELECT
        c.query_preview,
        ROUND(c.avg_ms, 2),
        ROUND(b.avg_ms, 2),
        ROUND(((c.avg_ms - b.avg_ms) / NULLIF(b.avg_ms, 0) * 100)::numeric, 2),
        c.total_calls,
        b.total_calls
    FROM current_period c
    JOIN baseline_period b ON c.queryid = b.queryid
    WHERE ABS(c.avg_ms - b.avg_ms) > 5  -- Only significant changes
    ORDER BY ABS(c.avg_ms - b.avg_ms) DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Reset Functions
-- ==========================================================================

-- Reset statistics (use carefully - loses all history)
CREATE OR REPLACE FUNCTION reset_query_stats()
RETURNS VOID AS $$
BEGIN
    PERFORM pg_stat_statements_reset();
    RAISE NOTICE 'pg_stat_statements statistics have been reset';
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Verification
-- ==========================================================================

-- Check extension is installed
-- SELECT * FROM pg_extension WHERE extname = 'pg_stat_statements';

-- Get summary
-- SELECT * FROM get_query_summary();

-- View slow queries
-- SELECT * FROM v_slow_queries;

-- View memory search stats
-- SELECT * FROM v_memory_search_stats;

-- Get index suggestions
-- SELECT * FROM suggest_indexes();

-- Take a performance snapshot (run periodically or via pg_cron)
-- SELECT snapshot_query_performance();
