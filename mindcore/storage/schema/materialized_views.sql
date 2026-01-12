-- Mindcore Materialized Views
--
-- Pre-computed views for common aggregations and statistics.
-- Refreshed periodically via pg_cron or application triggers.
--
-- Benefits:
-- - O(1) lookups for expensive aggregations
-- - Reduced query complexity for dashboards/analytics
-- - Consistent snapshot for reporting
--
-- Requirements:
--   - PostgreSQL 12+ (for REFRESH CONCURRENTLY)
--
-- Run with: psql $DATABASE_URL -f materialized_views.sql

-- ==========================================================================
-- User Statistics View
-- ==========================================================================

-- Pre-computed user-level statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_stats AS
SELECT
    user_id,
    COUNT(*) AS total_memories,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(*) FILTER (WHERE memory_type = 'episodic') AS episodic_count,
    COUNT(*) FILTER (WHERE memory_type = 'semantic') AS semantic_count,
    COUNT(*) FILTER (WHERE memory_type = 'procedural') AS procedural_count,
    COUNT(*) FILTER (WHERE memory_type = 'preference') AS preference_count,
    AVG(importance) AS avg_importance,
    AVG(reinforcement_score) AS avg_reinforcement,
    SUM(access_count) AS total_access_count,
    MIN(created_at) AS first_memory_at,
    MAX(created_at) AS last_memory_at,
    -- Active memories (accessed in last 30 days)
    COUNT(*) FILTER (WHERE last_accessed > NOW() - INTERVAL '30 days') AS active_memories_30d,
    -- Topic distribution (top 10)
    (SELECT jsonb_object_agg(topic, cnt)
     FROM (
         SELECT t.topic, COUNT(*) as cnt
         FROM memories m2, jsonb_array_elements_text(m2.topics) t(topic)
         WHERE m2.user_id = m.user_id
         GROUP BY t.topic
         ORDER BY cnt DESC
         LIMIT 10
     ) top_topics) AS top_topics,
    -- Recent session IDs
    ARRAY(
        SELECT DISTINCT session_id
        FROM memories m2
        WHERE m2.user_id = m.user_id AND m2.session_id IS NOT NULL
        ORDER BY session_id DESC
        LIMIT 5
    ) AS recent_sessions
FROM memories m
WHERE expires_at IS NULL OR expires_at > NOW()
GROUP BY user_id
WITH DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_user_stats_user
ON mv_user_stats (user_id);

-- ==========================================================================
-- Session Statistics View
-- ==========================================================================

-- Enhanced session statistics (supplements session_aggregates)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_session_stats AS
SELECT
    s.session_id,
    s.user_id,
    s.agent_id,
    s.memory_count,
    s.importance_avg,
    s.dominant_topic,
    s.dominant_category,
    s.started_at,
    s.last_activity_at,
    -- Computed fields
    EXTRACT(EPOCH FROM (s.last_activity_at - s.started_at)) / 60 AS duration_minutes,
    -- Memory type breakdown
    (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.session_id AND m.memory_type = 'episodic') AS episodic_count,
    (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.session_id AND m.memory_type = 'semantic') AS semantic_count,
    -- Sentiment analysis
    (SELECT AVG(CASE sentiment
        WHEN 'positive' THEN 1
        WHEN 'neutral' THEN 0
        WHEN 'negative' THEN -1
        ELSE 0
    END) FROM memories m WHERE m.session_id = s.session_id) AS sentiment_score,
    -- Engagement metrics
    (SELECT SUM(access_count) FROM memories m WHERE m.session_id = s.session_id) AS total_accesses,
    (SELECT AVG(reinforcement_score) FROM memories m WHERE m.session_id = s.session_id) AS avg_reinforcement
FROM session_aggregates s
WITH DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_session_stats_session
ON mv_session_stats (session_id);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_mv_session_stats_user
ON mv_session_stats (user_id, last_activity_at DESC);

-- ==========================================================================
-- Topic Analytics View
-- ==========================================================================

-- Global topic frequency and trends
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_topic_analytics AS
SELECT
    topic,
    COUNT(*) AS total_occurrences,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT session_id) AS unique_sessions,
    AVG(importance) AS avg_importance,
    AVG(reinforcement_score) AS avg_reinforcement,
    -- Time-based trends
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 day') AS occurrences_1d,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS occurrences_7d,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') AS occurrences_30d,
    MIN(created_at) AS first_seen,
    MAX(created_at) AS last_seen
FROM memories m, jsonb_array_elements_text(m.topics) t(topic)
WHERE m.expires_at IS NULL OR m.expires_at > NOW()
GROUP BY topic
HAVING COUNT(*) >= 5  -- Only topics with meaningful frequency
WITH DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_topic_analytics_topic
ON mv_topic_analytics (topic);

-- Index for popularity ranking
CREATE INDEX IF NOT EXISTS idx_mv_topic_analytics_popularity
ON mv_topic_analytics (total_occurrences DESC);

-- ==========================================================================
-- Memory Health View
-- ==========================================================================

-- System health and data quality metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_memory_health AS
SELECT
    -- Overall counts
    (SELECT COUNT(*) FROM memories) AS total_memories,
    (SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at <= NOW()) AS expired_memories,
    (SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL OR embedding_vector IS NOT NULL) AS embedded_memories,
    (SELECT COUNT(*) FROM session_aggregates) AS total_sessions,
    (SELECT COUNT(DISTINCT user_id) FROM memories) AS total_users,

    -- Data quality
    (SELECT COUNT(*) FROM memories WHERE jsonb_array_length(topics) = 0) AS memories_no_topics,
    (SELECT COUNT(*) FROM memories WHERE content IS NULL OR content = '') AS memories_no_content,
    (SELECT COUNT(*) FROM memories WHERE importance = 0.5) AS memories_default_importance,

    -- Age distribution
    (SELECT COUNT(*) FROM memories WHERE created_at > NOW() - INTERVAL '1 day') AS memories_1d,
    (SELECT COUNT(*) FROM memories WHERE created_at > NOW() - INTERVAL '7 days') AS memories_7d,
    (SELECT COUNT(*) FROM memories WHERE created_at > NOW() - INTERVAL '30 days') AS memories_30d,
    (SELECT COUNT(*) FROM memories WHERE created_at > NOW() - INTERVAL '90 days') AS memories_90d,

    -- Storage estimates (approximations)
    (SELECT pg_total_relation_size('memories')) AS memories_size_bytes,
    (SELECT pg_total_relation_size('session_aggregates')) AS sessions_size_bytes,

    -- Refresh timestamp
    NOW() AS refreshed_at
WITH DATA;

-- Single row, no unique index needed
-- But we add one for REFRESH CONCURRENTLY support
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_memory_health
ON mv_memory_health (refreshed_at);

-- ==========================================================================
-- Daily Aggregates View
-- ==========================================================================

-- Daily rollups for time-series analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_stats AS
SELECT
    DATE(created_at) AS date,
    COUNT(*) AS memories_created,
    COUNT(DISTINCT user_id) AS active_users,
    COUNT(DISTINCT session_id) AS sessions_started,
    AVG(importance) AS avg_importance,
    SUM(access_count) AS total_accesses,
    -- Memory types
    COUNT(*) FILTER (WHERE memory_type = 'episodic') AS episodic_count,
    COUNT(*) FILTER (WHERE memory_type = 'semantic') AS semantic_count,
    -- Sentiment
    COUNT(*) FILTER (WHERE sentiment = 'positive') AS positive_count,
    COUNT(*) FILTER (WHERE sentiment = 'negative') AS negative_count
FROM memories
WHERE created_at > NOW() - INTERVAL '90 days'  -- Rolling 90-day window
GROUP BY DATE(created_at)
WITH DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_stats_date
ON mv_daily_stats (date);

-- ==========================================================================
-- Refresh Functions
-- ==========================================================================

-- Refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS TABLE (view_name TEXT, refresh_time INTERVAL) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
BEGIN
    -- User stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_stats;
    end_time := clock_timestamp();
    view_name := 'mv_user_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- Session stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_session_stats;
    end_time := clock_timestamp();
    view_name := 'mv_session_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- Topic analytics
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_topic_analytics;
    end_time := clock_timestamp();
    view_name := 'mv_topic_analytics';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- Memory health
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_memory_health;
    end_time := clock_timestamp();
    view_name := 'mv_memory_health';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- Daily stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_stats;
    end_time := clock_timestamp();
    view_name := 'mv_daily_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Lightweight refresh for frequently needed views
CREATE OR REPLACE FUNCTION refresh_critical_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_session_stats;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Verification
-- ==========================================================================

-- Check materialized views exist
-- SELECT matviewname, ispopulated FROM pg_matviews WHERE schemaname = 'public';

-- Check refresh status
-- SELECT * FROM mv_memory_health;

-- Manual refresh
-- SELECT * FROM refresh_all_materialized_views();
