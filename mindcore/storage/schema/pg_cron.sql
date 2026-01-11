-- Mindcore pg_cron Scheduled Jobs
--
-- Automates database maintenance tasks:
-- - Expired memory cleanup
-- - Materialized view refresh
-- - Statistics updates
-- - Orphaned data cleanup
--
-- Requirements:
--   - PostgreSQL 12+ with pg_cron extension
--   - pg_cron must be configured in postgresql.conf:
--       shared_preload_libraries = 'pg_cron'
--       cron.database_name = 'your_database_name'
--
-- Run with: psql $DATABASE_URL -f pg_cron.sql
-- See README.md for installation guide

-- Enable pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Grant usage to the database user (run as superuser)
-- GRANT USAGE ON SCHEMA cron TO your_db_user;

-- ==========================================================================
-- Cleanup Functions
-- ==========================================================================

-- Delete expired memories
CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM memories
    WHERE expires_at IS NOT NULL
      AND expires_at <= NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    IF deleted_count > 0 THEN
        RAISE NOTICE 'Cleaned up % expired memories', deleted_count;
    END IF;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Delete orphaned session aggregates (no memories)
CREATE OR REPLACE FUNCTION cleanup_orphaned_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM session_aggregates sa
    WHERE NOT EXISTS (
        SELECT 1 FROM memories m WHERE m.session_id = sa.session_id
    );

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    IF deleted_count > 0 THEN
        RAISE NOTICE 'Cleaned up % orphaned session aggregates', deleted_count;
    END IF;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Clean up old transfer data (older than 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_transfers()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM transfers
    WHERE created_at < NOW() - INTERVAL '7 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    IF deleted_count > 0 THEN
        RAISE NOTICE 'Cleaned up % old transfers', deleted_count;
    END IF;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Archive old memories to a separate table (optional)
-- For compliance or historical analysis
CREATE OR REPLACE FUNCTION archive_old_memories(
    p_older_than INTERVAL DEFAULT '1 year',
    p_min_accesses INTEGER DEFAULT 0
) RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS memories_archive (LIKE memories INCLUDING ALL);

    -- Move old, rarely accessed memories to archive
    WITH moved AS (
        DELETE FROM memories
        WHERE created_at < NOW() - p_older_than
          AND access_count <= p_min_accesses
          AND (expires_at IS NULL OR expires_at > NOW())
        RETURNING *
    )
    INSERT INTO memories_archive SELECT * FROM moved;

    GET DIAGNOSTICS archived_count = ROW_COUNT;

    IF archived_count > 0 THEN
        RAISE NOTICE 'Archived % old memories', archived_count;
    END IF;

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Master cleanup function
CREATE OR REPLACE FUNCTION run_all_cleanup()
RETURNS TABLE (
    task TEXT,
    affected_rows INTEGER,
    duration INTERVAL
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    count INTEGER;
BEGIN
    -- Expired memories
    start_time := clock_timestamp();
    count := cleanup_expired_memories();
    end_time := clock_timestamp();
    task := 'cleanup_expired_memories';
    affected_rows := count;
    duration := end_time - start_time;
    RETURN NEXT;

    -- Orphaned sessions
    start_time := clock_timestamp();
    count := cleanup_orphaned_sessions();
    end_time := clock_timestamp();
    task := 'cleanup_orphaned_sessions';
    affected_rows := count;
    duration := end_time - start_time;
    RETURN NEXT;

    -- Old transfers
    start_time := clock_timestamp();
    count := cleanup_old_transfers();
    end_time := clock_timestamp();
    task := 'cleanup_old_transfers';
    affected_rows := count;
    duration := end_time - start_time;
    RETURN NEXT;

    -- Vacuum analyze (if needed)
    -- VACUUM ANALYZE memories;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Maintenance Functions
-- ==========================================================================

-- Update access statistics (for cold memories)
CREATE OR REPLACE FUNCTION decay_reinforcement_scores()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Apply time-based decay to reinforcement scores
    -- Memories not accessed in 30+ days decay toward 0
    UPDATE memories
    SET reinforcement_score = reinforcement_score * 0.95
    WHERE last_accessed < NOW() - INTERVAL '30 days'
      AND ABS(reinforcement_score) > 0.01;  -- Only update non-negligible scores

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Recalculate importance for stale memories
CREATE OR REPLACE FUNCTION recalculate_stale_importance()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Lower importance of memories not accessed in 90+ days
    UPDATE memories
    SET importance = GREATEST(0.1, importance * 0.9)
    WHERE last_accessed < NOW() - INTERVAL '90 days'
      AND importance > 0.1;

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Schedule Jobs (pg_cron)
-- ==========================================================================

-- Note: Job names must be unique. These commands will fail if jobs already exist.
-- Use cron.unschedule() to remove existing jobs first.

-- Helper to safely schedule (removes existing job with same name first)
CREATE OR REPLACE FUNCTION safe_schedule(
    p_schedule TEXT,
    p_command TEXT,
    p_job_name TEXT
) RETURNS BIGINT AS $$
DECLARE
    job_id BIGINT;
BEGIN
    -- Remove existing job if any
    DELETE FROM cron.job WHERE jobname = p_job_name;

    -- Schedule new job
    SELECT cron.schedule(p_job_name, p_schedule, p_command) INTO job_id;

    RETURN job_id;
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup jobs (run these commands to set up cron)
-- Uncomment and run after pg_cron is installed

-- Cleanup expired memories every hour
-- SELECT safe_schedule('0 * * * *', 'SELECT cleanup_expired_memories()', 'cleanup_expired_memories');

-- Cleanup orphaned sessions daily at 3 AM
-- SELECT safe_schedule('0 3 * * *', 'SELECT cleanup_orphaned_sessions()', 'cleanup_orphaned_sessions');

-- Cleanup old transfers daily at 3:30 AM
-- SELECT safe_schedule('30 3 * * *', 'SELECT cleanup_old_transfers()', 'cleanup_old_transfers');

-- Refresh materialized views every 15 minutes
-- SELECT safe_schedule('*/15 * * * *', 'SELECT refresh_critical_views()', 'refresh_critical_views');

-- Full materialized view refresh daily at 4 AM
-- SELECT safe_schedule('0 4 * * *', 'SELECT refresh_all_materialized_views()', 'refresh_all_views');

-- Decay reinforcement scores weekly on Sunday at 2 AM
-- SELECT safe_schedule('0 2 * * 0', 'SELECT decay_reinforcement_scores()', 'decay_reinforcement');

-- Archive old memories monthly on the 1st at 1 AM
-- SELECT safe_schedule('0 1 1 * *', 'SELECT archive_old_memories(''1 year''::interval, 0)', 'archive_old_memories');

-- VACUUM ANALYZE weekly on Saturday at 3 AM
-- SELECT safe_schedule('0 3 * * 6', 'VACUUM ANALYZE memories', 'vacuum_memories');

-- ==========================================================================
-- Job Management Functions
-- ==========================================================================

-- List all scheduled jobs
CREATE OR REPLACE FUNCTION list_scheduled_jobs()
RETURNS TABLE (
    jobid BIGINT,
    jobname TEXT,
    schedule TEXT,
    command TEXT,
    nodename TEXT,
    active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT j.jobid, j.jobname, j.schedule, j.command, j.nodename, j.active
    FROM cron.job j
    ORDER BY j.jobid;
END;
$$ LANGUAGE plpgsql;

-- Get job run history
CREATE OR REPLACE FUNCTION get_job_history(p_limit INTEGER DEFAULT 50)
RETURNS TABLE (
    runid BIGINT,
    jobid BIGINT,
    jobname TEXT,
    status TEXT,
    return_message TEXT,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.runid,
        r.jobid,
        j.jobname,
        r.status,
        r.return_message,
        r.start_time,
        r.end_time
    FROM cron.job_run_details r
    JOIN cron.job j ON r.jobid = j.jobid
    ORDER BY r.start_time DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Pause a job
CREATE OR REPLACE FUNCTION pause_job(p_jobname TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE cron.job SET active = false WHERE jobname = p_jobname;
END;
$$ LANGUAGE plpgsql;

-- Resume a job
CREATE OR REPLACE FUNCTION resume_job(p_jobname TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE cron.job SET active = true WHERE jobname = p_jobname;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Setup Commands (run these after installing pg_cron)
-- ==========================================================================

-- Uncomment and run to set up all scheduled jobs:
/*
SELECT safe_schedule('0 * * * *', 'SELECT cleanup_expired_memories()', 'cleanup_expired_memories');
SELECT safe_schedule('0 3 * * *', 'SELECT cleanup_orphaned_sessions()', 'cleanup_orphaned_sessions');
SELECT safe_schedule('30 3 * * *', 'SELECT cleanup_old_transfers()', 'cleanup_old_transfers');
SELECT safe_schedule('*/15 * * * *', 'SELECT refresh_critical_views()', 'refresh_critical_views');
SELECT safe_schedule('0 4 * * *', 'SELECT refresh_all_materialized_views()', 'refresh_all_views');
SELECT safe_schedule('0 2 * * 0', 'SELECT decay_reinforcement_scores()', 'decay_reinforcement');
SELECT safe_schedule('0 3 * * 6', 'VACUUM ANALYZE memories', 'vacuum_memories');
*/

-- ==========================================================================
-- Verification
-- ==========================================================================

-- Check pg_cron is installed
-- SELECT * FROM pg_extension WHERE extname = 'pg_cron';

-- List scheduled jobs
-- SELECT * FROM list_scheduled_jobs();

-- Check recent job runs
-- SELECT * FROM get_job_history(10);

-- Manually run cleanup
-- SELECT * FROM run_all_cleanup();
