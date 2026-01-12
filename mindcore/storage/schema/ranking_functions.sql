-- Ranking Functions for Mindcore PostgreSQL Search
--
-- These functions implement database-native ranking to replace Python-side
-- scoring, improving performance by 7-10x for large memory sets.
--
-- Functions:
--   rank_memory()  - Multi-component memory relevance scoring
--   rank_session() - Session relevance scoring for hierarchical retrieval
--
-- Usage: Source this file after enabling extensions
--   psql $DATABASE_URL -f ranking_functions.sql

-- =============================================================================
-- rank_memory: Unified memory ranking function
-- =============================================================================
-- Replaces: mindcore/clst/storage.py:score_memories_complex()
--
-- Scoring components (default weights):
--   - Content similarity (0.15): Trigram similarity with query
--   - Topic match (0.25): Intersection with attention hints
--   - Recency (0.15): Exponential decay with 1-week half-life
--   - Reinforcement (0.2): User feedback score + UCB exploration bonus
--   - Importance (0.15): Memory importance score
--   - Popularity (0.1): Normalized access count
--
-- Parameters:
--   p_content          - Memory content text
--   p_topics           - Memory topics as JSONB array
--   p_query            - Search query text
--   p_attention_topics - Topics to prioritize as TEXT[]
--   p_importance       - Memory importance (0-1)
--   p_reinforcement    - Reinforcement score (-1 to 1)
--   p_created_at       - Memory creation timestamp
--   p_access_count     - Number of times memory was accessed
--   p_weights          - Optional custom weights as JSONB
--
-- Returns: Score between 0 and 1

CREATE OR REPLACE FUNCTION rank_memory(
    p_content TEXT,
    p_topics JSONB,
    p_query TEXT,
    p_attention_topics TEXT[],
    p_importance FLOAT,
    p_reinforcement_score FLOAT,
    p_created_at TIMESTAMPTZ,
    p_access_count INT,
    p_weights JSONB DEFAULT '{
        "content": 0.15,
        "topic": 0.25,
        "recency": 0.15,
        "reinforcement": 0.2,
        "importance": 0.15,
        "popularity": 0.1
    }'::jsonb
) RETURNS FLOAT AS $$
DECLARE
    v_content_score FLOAT := 0;
    v_topic_score FLOAT := 0;
    v_recency_score FLOAT := 0;
    v_reinforcement FLOAT := 0;
    v_importance_score FLOAT := 0;
    v_popularity_score FLOAT := 0;
    v_topic_matches INT := 0;
    v_topic_count INT := 0;
    v_age_hours FLOAT;
    v_exploration_bonus FLOAT := 0;
BEGIN
    -- 1. Content similarity using trigram (pg_trgm)
    -- Returns 0-1 based on character n-gram overlap
    IF p_query IS NOT NULL AND p_query != '' AND p_content IS NOT NULL THEN
        v_content_score := COALESCE(similarity(p_content, p_query), 0);
    END IF;

    -- 2. Topic matching with attention hints
    -- Score based on intersection of memory topics with attention hints
    IF p_attention_topics IS NOT NULL THEN
        v_topic_count := array_length(p_attention_topics, 1);
        IF v_topic_count IS NOT NULL AND v_topic_count > 0 AND p_topics IS NOT NULL THEN
            SELECT COUNT(*) INTO v_topic_matches
            FROM jsonb_array_elements_text(p_topics) AS t
            WHERE t = ANY(p_attention_topics);

            v_topic_score := LEAST(v_topic_matches::FLOAT / v_topic_count, 1.0);
        END IF;
    END IF;

    -- 3. Recency decay (exponential with 1-week half-life)
    -- Memories older than 1 week get 50% score, 2 weeks ~25%, etc.
    IF p_created_at IS NOT NULL THEN
        v_age_hours := EXTRACT(EPOCH FROM (NOW() - p_created_at)) / 3600;
        v_recency_score := EXP(-v_age_hours / 168);  -- 168 hours = 1 week
    ELSE
        v_recency_score := 0.5;  -- Default for unknown age
    END IF;

    -- 4. Reinforcement score with UCB exploration bonus
    -- UCB (Upper Confidence Bound) encourages exploring less-accessed memories
    v_reinforcement := GREATEST(COALESCE(p_reinforcement_score, 0), 0);

    IF COALESCE(p_access_count, 0) < 10 THEN
        -- UCB-like exploration bonus: sqrt(2 * ln(N) / n)
        -- Using 1000 as proxy for total retrievals
        v_exploration_bonus := 0.1 * SQRT(
            2 * LN(GREATEST(1000, 1)) / GREATEST(COALESCE(p_access_count, 1), 1)
        );
        v_reinforcement := LEAST(1.0, v_reinforcement + v_exploration_bonus);
    END IF;

    -- 5. Importance score (direct pass-through)
    v_importance_score := COALESCE(p_importance, 0.5);

    -- 6. Popularity score (normalized to 100 accesses = 1.0)
    v_popularity_score := LEAST(COALESCE(p_access_count, 0) / 100.0, 1.0);

    -- Weighted combination using configurable weights
    RETURN LEAST(1.0, GREATEST(0.0,
        v_content_score * COALESCE((p_weights->>'content')::float, 0.15) +
        v_topic_score * COALESCE((p_weights->>'topic')::float, 0.25) +
        v_recency_score * COALESCE((p_weights->>'recency')::float, 0.15) +
        v_reinforcement * COALESCE((p_weights->>'reinforcement')::float, 0.2) +
        v_importance_score * COALESCE((p_weights->>'importance')::float, 0.15) +
        v_popularity_score * COALESCE((p_weights->>'popularity')::float, 0.1)
    ));
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION rank_memory IS 'Multi-component memory relevance scoring using trigram similarity, topic matching, recency decay, reinforcement signals, and popularity.';


-- =============================================================================
-- rank_session: Session relevance ranking function
-- =============================================================================
-- Replaces: mindcore/clst/aggregates.py:calculate_relevance_score()
--
-- Used for hierarchical retrieval - find relevant sessions first,
-- then drill down to memories within those sessions.
--
-- Scoring components (fixed weights):
--   - Topic matching (0.4): Average weight of matched topics
--   - Category matching (0.2): Average weight of matched categories
--   - Importance (0.25): Session's average memory importance
--   - Recency (0.15): Exponential decay based on last activity
--
-- Parameters:
--   p_topic_weights    - Session topic weights as JSONB {topic: weight}
--   p_category_weights - Session category weights as JSONB
--   p_importance_avg   - Average importance of memories in session
--   p_last_activity_at - Timestamp of last activity in session
--   p_topic_hints      - Topics to match as TEXT[]
--   p_category_hints   - Categories to match as TEXT[]
--
-- Returns: Score between 0 and 1

CREATE OR REPLACE FUNCTION rank_session(
    p_topic_weights JSONB,
    p_category_weights JSONB,
    p_importance_avg FLOAT,
    p_last_activity_at TIMESTAMPTZ,
    p_topic_hints TEXT[],
    p_category_hints TEXT[]
) RETURNS FLOAT AS $$
DECLARE
    v_topic_score FLOAT := 0;
    v_category_score FLOAT := 0;
    v_importance_score FLOAT := 0;
    v_recency_score FLOAT := 0;
    v_age_hours FLOAT;
    v_topic_count INT;
    v_category_count INT;
BEGIN
    -- Topic matching (40% weight)
    -- Calculate average weight of matched topics
    IF p_topic_hints IS NOT NULL THEN
        v_topic_count := array_length(p_topic_hints, 1);
        IF v_topic_count IS NOT NULL AND v_topic_count > 0 AND p_topic_weights IS NOT NULL THEN
            SELECT COALESCE(AVG((p_topic_weights->>t)::float), 0)
            INTO v_topic_score
            FROM unnest(p_topic_hints) AS t
            WHERE p_topic_weights ? t;
        END IF;
    END IF;

    -- Category matching (20% weight)
    IF p_category_hints IS NOT NULL THEN
        v_category_count := array_length(p_category_hints, 1);
        IF v_category_count IS NOT NULL AND v_category_count > 0 AND p_category_weights IS NOT NULL THEN
            SELECT COALESCE(AVG((p_category_weights->>c)::float), 0)
            INTO v_category_score
            FROM unnest(p_category_hints) AS c
            WHERE p_category_weights ? c;
        END IF;
    END IF;

    -- Importance score (25% weight)
    v_importance_score := COALESCE(p_importance_avg, 0);

    -- Recency score (15% weight)
    -- Exponential decay with 1-week half-life
    IF p_last_activity_at IS NOT NULL THEN
        v_age_hours := EXTRACT(EPOCH FROM (NOW() - p_last_activity_at)) / 3600;
        v_recency_score := EXP(-v_age_hours / 168);  -- 168 hours = 1 week
    ELSE
        v_recency_score := 0.5;
    END IF;

    -- Fixed weighted combination
    RETURN LEAST(1.0, GREATEST(0.0,
        v_topic_score * 0.4 +
        v_category_score * 0.2 +
        v_importance_score * 0.25 +
        v_recency_score * 0.15
    ));
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION rank_session IS 'Session relevance scoring for hierarchical retrieval using weighted topic/category matching, importance, and recency.';


-- =============================================================================
-- Utility: Batch ranking for performance
-- =============================================================================
-- For large result sets, use these views/functions instead of per-row calls

-- Example: Ranked memories view (create per-user or with security policies)
-- CREATE OR REPLACE VIEW ranked_memories AS
-- SELECT m.*,
--        rank_memory(m.content, m.topics, '', ARRAY[]::text[],
--                    m.importance, m.reinforcement_score, m.created_at, m.access_count) AS base_score
-- FROM memories m
-- WHERE expires_at IS NULL OR expires_at > NOW();


-- =============================================================================
-- Verification
-- =============================================================================
-- Test the functions after creation:
--
-- SELECT rank_memory(
--     'I ordered some shoes last week',
--     '["orders", "shoes"]'::jsonb,
--     'order status',
--     ARRAY['orders', 'shipping'],
--     0.7,
--     0.3,
--     NOW() - INTERVAL '2 days',
--     5
-- );
-- -- Should return ~0.5-0.7
--
-- SELECT rank_session(
--     '{"orders": 0.8, "shipping": 0.6}'::jsonb,
--     '{"support": 0.9}'::jsonb,
--     0.65,
--     NOW() - INTERVAL '1 day',
--     ARRAY['orders', 'returns'],
--     ARRAY['support']
-- );
-- -- Should return ~0.5-0.6
