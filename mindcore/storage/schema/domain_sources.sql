-- =============================================================================
-- Domain Sources Schema for Mindcore
-- =============================================================================
--
-- This schema implements PostgreSQL-centric domain source management:
-- 1. Topic-to-table mapping configurations
-- 2. Automatic query triggering via functions
-- 3. Full audit trail for traceability
-- 4. Preference extraction metadata
--
-- Usage:
--   psql -d mindcore -f domain_sources.sql
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. DOMAIN SOURCE CONFIGURATIONS
-- -----------------------------------------------------------------------------

-- Table to store domain source configurations
CREATE TABLE IF NOT EXISTS domain_source_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('table', 'api', 'function', 'mcp')),
    description TEXT,

    -- Topics this source handles (array for many-to-many)
    topics TEXT[] NOT NULL DEFAULT '{}',

    -- Table source config
    table_name VARCHAR(255),
    query_template TEXT,
    param_mapping JSONB DEFAULT '{}',

    -- API source config
    api_url TEXT,
    api_method VARCHAR(10) DEFAULT 'GET',
    api_headers JSONB DEFAULT '{}',
    api_body_template JSONB,

    -- Function source config
    function_name VARCHAR(255),

    -- Common config
    cache_ttl_seconds INTEGER DEFAULT 60,
    timeout_seconds INTEGER DEFAULT 30,
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,  -- Higher = checked first

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for topic lookups
CREATE INDEX IF NOT EXISTS idx_domain_source_topics ON domain_source_configs USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_domain_source_enabled ON domain_source_configs (enabled) WHERE enabled = true;

-- -----------------------------------------------------------------------------
-- 2. DOMAIN SOURCE AUDIT LOG
-- -----------------------------------------------------------------------------

-- Full audit trail for all domain source operations
CREATE TABLE IF NOT EXISTS domain_source_audit_log (
    id BIGSERIAL PRIMARY KEY,

    -- Operation details
    operation VARCHAR(50) NOT NULL,  -- 'fetch', 'cache_hit', 'cache_miss', 'error'
    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,

    -- Context
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    topics TEXT[],

    -- Query details
    query_executed TEXT,
    params_used JSONB,

    -- Results
    success BOOLEAN NOT NULL,
    error_message TEXT,
    rows_returned INTEGER,
    latency_ms NUMERIC(10, 2),
    cached BOOLEAN DEFAULT false,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_log_user ON domain_source_audit_log (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_source ON domain_source_audit_log (source_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON domain_source_audit_log (operation, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON domain_source_audit_log (created_at DESC);

-- Auto-partition audit log by month for performance (optional)
-- CREATE TABLE domain_source_audit_log_y2025m01 PARTITION OF domain_source_audit_log
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- -----------------------------------------------------------------------------
-- 3. TOPIC-TO-SOURCE MAPPING CACHE
-- -----------------------------------------------------------------------------

-- Materialized view for fast topic lookups
CREATE MATERIALIZED VIEW IF NOT EXISTS topic_source_mapping AS
SELECT
    unnest(topics) AS topic,
    id AS source_id,
    name AS source_name,
    source_type,
    table_name,
    query_template,
    param_mapping,
    api_url,
    function_name,
    cache_ttl_seconds,
    priority
FROM domain_source_configs
WHERE enabled = true
ORDER BY priority DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_source_mapping ON topic_source_mapping (topic, source_id);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_topic_source_mapping()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY topic_source_mapping;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 4. CORE FUNCTIONS: FETCH DATA FOR TOPICS
-- -----------------------------------------------------------------------------

-- Main function: Get sources for a list of topics
CREATE OR REPLACE FUNCTION get_sources_for_topics(p_topics TEXT[])
RETURNS TABLE (
    source_id UUID,
    source_name VARCHAR(255),
    source_type VARCHAR(50),
    table_name VARCHAR(255),
    query_template TEXT,
    param_mapping JSONB,
    api_url TEXT,
    function_name VARCHAR(255),
    cache_ttl_seconds INTEGER,
    matched_topics TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (dsc.id)
        dsc.id,
        dsc.name,
        dsc.source_type,
        dsc.table_name,
        dsc.query_template,
        dsc.param_mapping,
        dsc.api_url,
        dsc.function_name,
        dsc.cache_ttl_seconds,
        ARRAY(
            SELECT unnest(dsc.topics)
            INTERSECT
            SELECT unnest(p_topics)
        ) AS matched_topics
    FROM domain_source_configs dsc
    WHERE dsc.enabled = true
      AND dsc.topics && p_topics  -- Array overlap operator
    ORDER BY dsc.id, dsc.priority DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Execute a table source query with parameter substitution
CREATE OR REPLACE FUNCTION execute_table_source(
    p_source_name VARCHAR(255),
    p_user_id VARCHAR(255),
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    v_source RECORD;
    v_query TEXT;
    v_result JSONB;
    v_start_time TIMESTAMPTZ;
    v_latency_ms NUMERIC;
    v_row_count INTEGER;
BEGIN
    v_start_time := clock_timestamp();

    -- Get source config
    SELECT * INTO v_source
    FROM domain_source_configs
    WHERE name = p_source_name AND enabled = true;

    IF v_source IS NULL THEN
        -- Log error
        INSERT INTO domain_source_audit_log (operation, source_name, source_type, user_id, success, error_message)
        VALUES ('fetch', p_source_name, 'unknown', p_user_id, false, 'Source not found or disabled');

        RETURN jsonb_build_object('success', false, 'error', 'Source not found');
    END IF;

    -- Build query with params
    v_query := v_source.query_template;

    -- Replace :user_id placeholder
    v_query := replace(v_query, ':user_id', quote_literal(p_user_id));

    -- Replace other params from JSONB
    FOR key, value IN SELECT * FROM jsonb_each_text(p_params) LOOP
        v_query := replace(v_query, ':' || key, quote_literal(value));
    END LOOP;

    -- Execute and get results as JSONB array
    EXECUTE format('SELECT jsonb_agg(row_to_json(t)) FROM (%s) t', v_query) INTO v_result;

    -- Get row count
    v_row_count := COALESCE(jsonb_array_length(v_result), 0);
    v_latency_ms := EXTRACT(EPOCH FROM (clock_timestamp() - v_start_time)) * 1000;

    -- Log success
    INSERT INTO domain_source_audit_log (
        operation, source_name, source_type, user_id,
        query_executed, params_used, success, rows_returned, latency_ms
    ) VALUES (
        'fetch', p_source_name, v_source.source_type, p_user_id,
        v_query, p_params, true, v_row_count, v_latency_ms
    );

    RETURN jsonb_build_object(
        'success', true,
        'data', COALESCE(v_result, '[]'::jsonb),
        'rows', v_row_count,
        'latency_ms', v_latency_ms,
        'source', p_source_name
    );

EXCEPTION WHEN OTHERS THEN
    v_latency_ms := EXTRACT(EPOCH FROM (clock_timestamp() - v_start_time)) * 1000;

    -- Log error
    INSERT INTO domain_source_audit_log (
        operation, source_name, source_type, user_id,
        query_executed, params_used, success, error_message, latency_ms
    ) VALUES (
        'fetch', p_source_name, COALESCE(v_source.source_type, 'unknown'), p_user_id,
        v_query, p_params, false, SQLERRM, v_latency_ms
    );

    RETURN jsonb_build_object(
        'success', false,
        'error', SQLERRM,
        'latency_ms', v_latency_ms
    );
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 5. PREFERENCE EXTRACTION SUPPORT
-- -----------------------------------------------------------------------------

-- Table to store extracted preferences per user
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,

    -- Preference categories
    preference_type VARCHAR(100) NOT NULL,  -- 'communication', 'product', 'ui', etc.
    preference_key VARCHAR(255) NOT NULL,
    preference_value JSONB NOT NULL,

    -- Confidence and source
    confidence NUMERIC(3, 2) DEFAULT 1.0,
    source_memory_id UUID,  -- Which memory this was extracted from

    -- Versioning (for preference changes over time)
    version INTEGER DEFAULT 1,
    superseded_by UUID,  -- Points to newer version if updated

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (user_id, preference_type, preference_key, version)
);

-- Index for fast preference lookups
CREATE INDEX IF NOT EXISTS idx_user_preferences_user ON user_preferences (user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_type ON user_preferences (user_id, preference_type);
CREATE INDEX IF NOT EXISTS idx_user_preferences_active ON user_preferences (user_id)
    WHERE superseded_by IS NULL;

-- Function to get active preferences for a user
CREATE OR REPLACE FUNCTION get_user_preferences(
    p_user_id VARCHAR(255),
    p_preference_type VARCHAR(100) DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_object_agg(
        preference_type || '.' || preference_key,
        jsonb_build_object(
            'value', preference_value,
            'confidence', confidence,
            'updated_at', updated_at
        )
    ) INTO v_result
    FROM user_preferences
    WHERE user_id = p_user_id
      AND superseded_by IS NULL
      AND (p_preference_type IS NULL OR preference_type = p_preference_type);

    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to upsert a preference
CREATE OR REPLACE FUNCTION upsert_preference(
    p_user_id VARCHAR(255),
    p_preference_type VARCHAR(100),
    p_preference_key VARCHAR(255),
    p_preference_value JSONB,
    p_confidence NUMERIC DEFAULT 1.0,
    p_source_memory_id UUID DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_existing_id UUID;
    v_new_id UUID;
    v_new_version INTEGER;
BEGIN
    -- Find existing active preference
    SELECT id, version INTO v_existing_id, v_new_version
    FROM user_preferences
    WHERE user_id = p_user_id
      AND preference_type = p_preference_type
      AND preference_key = p_preference_key
      AND superseded_by IS NULL;

    v_new_id := gen_random_uuid();
    v_new_version := COALESCE(v_new_version, 0) + 1;

    -- Mark existing as superseded
    IF v_existing_id IS NOT NULL THEN
        UPDATE user_preferences
        SET superseded_by = v_new_id, updated_at = NOW()
        WHERE id = v_existing_id;
    END IF;

    -- Insert new preference
    INSERT INTO user_preferences (
        id, user_id, preference_type, preference_key,
        preference_value, confidence, source_memory_id, version
    ) VALUES (
        v_new_id, p_user_id, p_preference_type, p_preference_key,
        p_preference_value, p_confidence, p_source_memory_id, v_new_version
    );

    RETURN v_new_id;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 6. AUTOMATIC TRIGGERS
-- -----------------------------------------------------------------------------

-- Trigger to refresh topic mapping when sources change
CREATE OR REPLACE FUNCTION trigger_refresh_topic_mapping()
RETURNS TRIGGER AS $$
BEGIN
    -- Async refresh (non-blocking)
    PERFORM refresh_topic_source_mapping();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_domain_source_refresh
AFTER INSERT OR UPDATE OR DELETE ON domain_source_configs
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_refresh_topic_mapping();

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION trigger_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_domain_source_timestamp
BEFORE UPDATE ON domain_source_configs
FOR EACH ROW
EXECUTE FUNCTION trigger_update_timestamp();

-- -----------------------------------------------------------------------------
-- 7. UTILITY FUNCTIONS
-- -----------------------------------------------------------------------------

-- Get audit log summary for a user
CREATE OR REPLACE FUNCTION get_audit_summary(
    p_user_id VARCHAR(255),
    p_since TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours'
)
RETURNS TABLE (
    source_name VARCHAR(255),
    total_calls BIGINT,
    successful_calls BIGINT,
    avg_latency_ms NUMERIC,
    cache_hit_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dal.source_name,
        COUNT(*) AS total_calls,
        COUNT(*) FILTER (WHERE success) AS successful_calls,
        AVG(latency_ms) AS avg_latency_ms,
        (COUNT(*) FILTER (WHERE cached))::NUMERIC / NULLIF(COUNT(*), 0) AS cache_hit_rate
    FROM domain_source_audit_log dal
    WHERE dal.user_id = p_user_id
      AND dal.created_at >= p_since
    GROUP BY dal.source_name;
END;
$$ LANGUAGE plpgsql STABLE;

-- Clean old audit logs (run via pg_cron)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(p_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM domain_source_audit_log
    WHERE created_at < NOW() - (p_days || ' days')::INTERVAL;

    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 8. INITIAL DATA / EXAMPLES
-- -----------------------------------------------------------------------------

-- Example: Register an orders table source
-- INSERT INTO domain_source_configs (name, source_type, topics, table_name, query_template, param_mapping)
-- VALUES (
--     'orders_source',
--     'table',
--     ARRAY['orders', 'purchases', 'transactions'],
--     'orders',
--     'SELECT order_id, status, total, created_at FROM orders WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 10',
--     '{"user_id": "user_id"}'::jsonb
-- );

-- Example: Register a preferences source
-- INSERT INTO domain_source_configs (name, source_type, topics, function_name)
-- VALUES (
--     'preferences_source',
--     'function',
--     ARRAY['preferences', 'settings'],
--     'get_user_preferences'
-- );

COMMENT ON TABLE domain_source_configs IS 'Configuration for domain data sources (tables, APIs, functions)';
COMMENT ON TABLE domain_source_audit_log IS 'Full audit trail of all domain source operations';
COMMENT ON TABLE user_preferences IS 'Extracted user preferences with versioning';
COMMENT ON FUNCTION get_sources_for_topics IS 'Get all sources that handle given topics';
COMMENT ON FUNCTION execute_table_source IS 'Execute a table source query with full logging';
COMMENT ON FUNCTION get_user_preferences IS 'Get active preferences for a user';
COMMENT ON FUNCTION upsert_preference IS 'Insert or update a preference with versioning';
