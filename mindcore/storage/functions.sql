-- =============================================================================
-- SAGE SQL Functions - Core Logic in PostgreSQL
-- =============================================================================
-- Moving scoring, ranking, and session management from Python to PostgreSQL
-- for deterministic, fast, and reliable operations.
-- =============================================================================

-- =============================================================================
-- SCORING FUNCTIONS
-- =============================================================================

-- Main SAGE scoring function (deterministic)
-- Combines: relevance, recency, reinforcement, importance, topic match
CREATE OR REPLACE FUNCTION sage_score(
    p_search_rank REAL,           -- ts_rank result (0-1)
    p_recency_hours REAL,         -- Hours since created
    p_reinforcement REAL,         -- Reinforcement score (-1 to 1)
    p_importance REAL,            -- Importance (0-1)
    p_confidence REAL,            -- Confidence (0-1)
    p_access_count INTEGER,       -- Access count for popularity
    p_topic_match_count INTEGER,  -- Number of matching topics
    p_total_topics INTEGER        -- Total topics requested
) RETURNS REAL AS $$
DECLARE
    v_relevance_weight CONSTANT REAL := 0.30;
    v_recency_weight CONSTANT REAL := 0.20;
    v_reinforcement_weight CONSTANT REAL := 0.15;
    v_importance_weight CONSTANT REAL := 0.15;
    v_confidence_weight CONSTANT REAL := 0.10;
    v_popularity_weight CONSTANT REAL := 0.05;
    v_topic_weight CONSTANT REAL := 0.05;

    v_recency_score REAL;
    v_popularity_score REAL;
    v_topic_score REAL;
    v_reinforcement_normalized REAL;
BEGIN
    -- Recency decay: score decreases over time (half-life of 24 hours)
    v_recency_score := 1.0 / (1.0 + p_recency_hours / 24.0);

    -- Popularity: logarithmic scaling
    v_popularity_score := LEAST(1.0, LN(1 + p_access_count) / LN(100));

    -- Topic match ratio
    v_topic_score := CASE
        WHEN p_total_topics > 0 THEN p_topic_match_count::REAL / p_total_topics::REAL
        ELSE 1.0
    END;

    -- Normalize reinforcement from [-1,1] to [0,1]
    v_reinforcement_normalized := (p_reinforcement + 1.0) / 2.0;

    -- Weighted combination
    RETURN (
        v_relevance_weight * COALESCE(p_search_rank, 0.5) +
        v_recency_weight * v_recency_score +
        v_reinforcement_weight * v_reinforcement_normalized +
        v_importance_weight * p_importance +
        v_confidence_weight * p_confidence +
        v_popularity_weight * v_popularity_score +
        v_topic_weight * v_topic_score
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION sage_score IS 'Deterministic scoring function for memory ranking';


-- Simplified scoring for hot path (no search, just metadata)
CREATE OR REPLACE FUNCTION sage_score_simple(
    p_recency_hours REAL,
    p_reinforcement REAL,
    p_importance REAL,
    p_access_count INTEGER
) RETURNS REAL AS $$
BEGIN
    RETURN sage_score(
        0.5,  -- Default relevance
        p_recency_hours,
        p_reinforcement,
        p_importance,
        0.8,  -- Default confidence
        p_access_count,
        1,    -- Assume topic match
        1
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- =============================================================================
-- TOPIC/CATEGORY WEIGHT FUNCTIONS
-- =============================================================================

-- Calculate topic match count between memory topics and query topics
CREATE OR REPLACE FUNCTION count_topic_matches(
    p_memory_topics JSONB,
    p_query_topics TEXT[]
) RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM jsonb_array_elements_text(p_memory_topics) AS t
        WHERE t = ANY(p_query_topics)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Update weight in JSONB object (increment or set)
CREATE OR REPLACE FUNCTION update_weight(
    p_weights JSONB,
    p_key TEXT,
    p_increment REAL DEFAULT 1.0
) RETURNS JSONB AS $$
DECLARE
    v_current REAL;
BEGIN
    v_current := COALESCE((p_weights ->> p_key)::REAL, 0.0);
    RETURN jsonb_set(
        COALESCE(p_weights, '{}'::jsonb),
        ARRAY[p_key],
        to_jsonb(v_current + p_increment)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Update multiple weights from an array
CREATE OR REPLACE FUNCTION update_weights_from_array(
    p_weights JSONB,
    p_items JSONB,
    p_increment REAL DEFAULT 1.0
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB := COALESCE(p_weights, '{}'::jsonb);
    v_item TEXT;
BEGIN
    FOR v_item IN SELECT value::text FROM jsonb_array_elements_text(p_items)
    LOOP
        v_result := update_weight(v_result, v_item, p_increment);
    END LOOP;
    RETURN v_result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Get dominant key from weights JSONB
CREATE OR REPLACE FUNCTION get_dominant_key(p_weights JSONB)
RETURNS TEXT AS $$
BEGIN
    RETURN (
        SELECT key
        FROM jsonb_each_text(p_weights)
        ORDER BY value::REAL DESC
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- =============================================================================
-- SESSION MANAGEMENT FUNCTIONS
-- =============================================================================

-- Update session aggregates when a memory is added
CREATE OR REPLACE FUNCTION update_session_on_memory_insert()
RETURNS TRIGGER AS $$
DECLARE
    v_session RECORD;
    v_new_count INTEGER;
    v_importance_sum REAL;
    v_confidence_sum REAL;
BEGIN
    -- Skip if no session_id
    IF NEW.session_id IS NULL THEN
        RETURN NEW;
    END IF;

    -- Get current session or create new one
    SELECT * INTO v_session FROM sessions WHERE session_id = NEW.session_id;

    IF v_session IS NULL THEN
        -- Create new session
        INSERT INTO sessions (
            session_id, user_id, agent_id, conversation_id,
            started_at, last_activity_at, memory_count
        ) VALUES (
            NEW.session_id, NEW.user_id, NEW.agent_id, NEW.conversation_id,
            NEW.created_at, NEW.created_at, 1
        );

        -- Set initial aggregates
        UPDATE sessions SET
            topic_weights = update_weights_from_array('{}'::jsonb, NEW.topics),
            category_weights = update_weights_from_array('{}'::jsonb, NEW.categories),
            entity_weights = update_weights_from_array('{}'::jsonb,
                (SELECT jsonb_agg(e->>'value') FROM jsonb_array_elements(NEW.entities) AS e)),
            importance_min = NEW.importance,
            importance_max = NEW.importance,
            importance_avg = NEW.importance,
            confidence_min = NEW.confidence,
            confidence_max = NEW.confidence,
            confidence_avg = NEW.confidence,
            dominant_topic = (SELECT value FROM jsonb_array_elements_text(NEW.topics) LIMIT 1),
            dominant_category = (SELECT value FROM jsonb_array_elements_text(NEW.categories) LIMIT 1),
            dominant_sentiment = NEW.sentiment,
            dominant_intent = NEW.message_intent
        WHERE session_id = NEW.session_id;
    ELSE
        -- Update existing session
        v_new_count := v_session.memory_count + 1;

        UPDATE sessions SET
            -- Update counts
            memory_count = v_new_count,
            message_count = CASE
                WHEN NEW.message_type IN ('question', 'answer', 'instruction')
                THEN message_count + 1
                ELSE message_count
            END,

            -- Update temporal
            last_activity_at = NEW.created_at,
            updated_at = NOW(),

            -- Update weights
            topic_weights = update_weights_from_array(topic_weights, NEW.topics),
            category_weights = update_weights_from_array(category_weights, NEW.categories),
            intent_weights = update_weight(intent_weights, NEW.message_intent),

            -- Update importance stats
            importance_min = LEAST(importance_min, NEW.importance),
            importance_max = GREATEST(importance_max, NEW.importance),
            importance_avg = (importance_avg * v_session.memory_count + NEW.importance) / v_new_count,

            -- Update confidence stats
            confidence_min = LEAST(confidence_min, NEW.confidence),
            confidence_max = GREATEST(confidence_max, NEW.confidence),
            confidence_avg = (confidence_avg * v_session.memory_count + NEW.confidence) / v_new_count,

            -- Update dominants
            dominant_topic = get_dominant_key(
                update_weights_from_array(topic_weights, NEW.topics)
            ),
            dominant_category = get_dominant_key(
                update_weights_from_array(category_weights, NEW.categories)
            ),
            dominant_sentiment = CASE
                WHEN NEW.importance > 0.7 THEN NEW.sentiment
                ELSE dominant_sentiment
            END,
            dominant_intent = get_dominant_key(
                update_weight(intent_weights, NEW.message_intent)
            )
        WHERE session_id = NEW.session_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Create trigger for automatic session updates
DROP TRIGGER IF EXISTS trg_memory_session_update ON memories;
CREATE TRIGGER trg_memory_session_update
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_session_on_memory_insert();


-- Auto-assign message_index within session
CREATE OR REPLACE FUNCTION assign_message_index()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.session_id IS NOT NULL AND NEW.message_index = 0 THEN
        SELECT COALESCE(MAX(message_index), 0) + 1
        INTO NEW.message_index
        FROM memories
        WHERE session_id = NEW.session_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_assign_message_index ON memories;
CREATE TRIGGER trg_assign_message_index
    BEFORE INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION assign_message_index();


-- =============================================================================
-- SEARCH FUNCTIONS
-- =============================================================================

-- Search memories with SAGE scoring
CREATE OR REPLACE FUNCTION search_memories_scored(
    p_user_id TEXT,
    p_query TEXT DEFAULT NULL,
    p_topics TEXT[] DEFAULT NULL,
    p_categories TEXT[] DEFAULT NULL,
    p_memory_types TEXT[] DEFAULT NULL,
    p_min_importance REAL DEFAULT NULL,
    p_session_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 20,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    memory_id TEXT,
    content TEXT,
    memory_type TEXT,
    message_type TEXT,
    message_intent TEXT,
    topics JSONB,
    categories JSONB,
    importance REAL,
    confidence REAL,
    sentiment TEXT,
    reinforcement_score REAL,
    session_id TEXT,
    created_at TIMESTAMPTZ,
    sage_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.memory_id,
        m.content,
        m.memory_type,
        m.message_type,
        m.message_intent,
        m.topics,
        m.categories,
        m.importance,
        m.confidence,
        m.sentiment,
        m.reinforcement_score,
        m.session_id,
        m.created_at,
        sage_score(
            CASE WHEN p_query IS NOT NULL
                THEN ts_rank(m.search_vector, plainto_tsquery('english', p_query))
                ELSE 0.5
            END,
            EXTRACT(EPOCH FROM NOW() - m.created_at) / 3600,
            m.reinforcement_score,
            m.importance,
            m.confidence,
            m.access_count,
            COALESCE(count_topic_matches(m.topics, p_topics), 0),
            COALESCE(array_length(p_topics, 1), 1)
        ) AS sage_score
    FROM memories m
    WHERE m.user_id = p_user_id
        AND (m.expires_at IS NULL OR m.expires_at > NOW())
        AND (p_query IS NULL OR m.search_vector @@ plainto_tsquery('english', p_query))
        AND (p_topics IS NULL OR m.topics ?| p_topics)
        AND (p_categories IS NULL OR m.categories ?| p_categories)
        AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
        AND (p_min_importance IS NULL OR m.importance >= p_min_importance)
        AND (p_session_id IS NULL OR m.session_id = p_session_id)
    ORDER BY sage_score DESC, m.created_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION search_memories_scored IS 'Search memories with SAGE scoring - main query function';


-- Fuzzy search using pg_trgm
CREATE OR REPLACE FUNCTION search_memories_fuzzy(
    p_user_id TEXT,
    p_query TEXT,
    p_similarity_threshold REAL DEFAULT 0.3,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE (
    memory_id TEXT,
    content TEXT,
    similarity REAL,
    sage_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.memory_id,
        m.content,
        similarity(m.content, p_query) AS similarity,
        sage_score(
            similarity(m.content, p_query),
            EXTRACT(EPOCH FROM NOW() - m.created_at) / 3600,
            m.reinforcement_score,
            m.importance,
            m.confidence,
            m.access_count,
            1, 1
        ) AS sage_score
    FROM memories m
    WHERE m.user_id = p_user_id
        AND m.content % p_query
        AND similarity(m.content, p_query) >= p_similarity_threshold
    ORDER BY similarity DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;


-- =============================================================================
-- SESSION QUERY FUNCTIONS
-- =============================================================================

-- Find relevant sessions by topic weights
CREATE OR REPLACE FUNCTION find_relevant_sessions(
    p_user_id TEXT,
    p_topics TEXT[] DEFAULT NULL,
    p_categories TEXT[] DEFAULT NULL,
    p_min_importance REAL DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE (
    session_id TEXT,
    dominant_topic TEXT,
    topic_weights JSONB,
    importance_avg REAL,
    memory_count INTEGER,
    last_activity_at TIMESTAMPTZ,
    relevance_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.session_id,
        s.dominant_topic,
        s.topic_weights,
        s.importance_avg,
        s.memory_count,
        s.last_activity_at,
        (
            -- Calculate session relevance
            COALESCE((
                SELECT SUM((s.topic_weights ->> t)::REAL)
                FROM unnest(p_topics) AS t
                WHERE s.topic_weights ? t
            ), 0) +
            s.importance_avg * 0.3 +
            (1.0 / (1.0 + EXTRACT(EPOCH FROM NOW() - s.last_activity_at) / 86400)) * 0.2
        )::REAL AS relevance_score
    FROM sessions s
    WHERE s.user_id = p_user_id
        AND s.status != 'archived'
        AND (p_topics IS NULL OR s.topic_weights ?| p_topics)
        AND (p_categories IS NULL OR s.category_weights ?| p_categories)
        AND (p_min_importance IS NULL OR s.importance_avg >= p_min_importance)
    ORDER BY relevance_score DESC, s.last_activity_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;


-- =============================================================================
-- REINFORCEMENT UPDATE FUNCTION
-- =============================================================================

-- Update reinforcement with bounds and update timestamp
CREATE OR REPLACE FUNCTION update_reinforcement(
    p_memory_id TEXT,
    p_signal REAL
) RETURNS REAL AS $$
DECLARE
    v_new_score REAL;
BEGIN
    UPDATE memories
    SET
        reinforcement_score = GREATEST(-1.0, LEAST(1.0, reinforcement_score + p_signal)),
        access_count = access_count + 1,
        last_accessed = NOW(),
        updated_at = NOW()
    WHERE memory_id = p_memory_id
    RETURNING reinforcement_score INTO v_new_score;

    RETURN v_new_score;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- CLEANUP FUNCTIONS
-- =============================================================================

-- Clean up expired memories
CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM memories
    WHERE expires_at IS NOT NULL AND expires_at < NOW();

    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$ LANGUAGE plpgsql;


-- Archive old sessions
CREATE OR REPLACE FUNCTION archive_inactive_sessions(
    p_days_inactive INTEGER DEFAULT 30
) RETURNS INTEGER AS $$
DECLARE
    v_archived INTEGER;
BEGIN
    UPDATE sessions
    SET status = 'archived', updated_at = NOW()
    WHERE status = 'active'
        AND last_activity_at < NOW() - (p_days_inactive || ' days')::INTERVAL;

    GET DIAGNOSTICS v_archived = ROW_COUNT;
    RETURN v_archived;
END;
$$ LANGUAGE plpgsql;
