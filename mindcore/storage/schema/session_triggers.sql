-- Mindcore Session Aggregate Triggers
--
-- Automatically updates session_aggregates when memories are inserted/updated.
-- Replaces Python-side update_session_aggregate_from_memory() for better performance.
--
-- Benefits:
-- - Atomic updates (no race conditions between insert and aggregate update)
-- - Reduced round trips (one transaction instead of two)
-- - Database-native JSON operations
--
-- Requirements:
--   - PostgreSQL 12+ (for generated columns and JSONB operations)
--
-- Run with: psql $DATABASE_URL -f session_triggers.sql

-- ==========================================================================
-- Helper Functions
-- ==========================================================================

-- Merge topic weights: adds new topic or updates existing weight
CREATE OR REPLACE FUNCTION merge_topic_weight(
    p_current_weights JSONB,
    p_topic TEXT,
    p_weight FLOAT DEFAULT 1.0
) RETURNS JSONB AS $$
DECLARE
    current_weight FLOAT;
    new_weight FLOAT;
BEGIN
    IF p_topic IS NULL OR p_topic = '' THEN
        RETURN p_current_weights;
    END IF;

    current_weight := COALESCE((p_current_weights->>p_topic)::float, 0);
    -- Weighted average with decay: new = old * 0.9 + new * 0.1
    new_weight := LEAST(1.0, current_weight * 0.9 + p_weight * 0.1);

    RETURN jsonb_set(
        COALESCE(p_current_weights, '{}'::jsonb),
        ARRAY[p_topic],
        to_jsonb(new_weight)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Merge multiple topics at once
CREATE OR REPLACE FUNCTION merge_topics_weights(
    p_current_weights JSONB,
    p_topics JSONB,
    p_base_weight FLOAT DEFAULT 1.0
) RETURNS JSONB AS $$
DECLARE
    result JSONB := COALESCE(p_current_weights, '{}'::jsonb);
    topic TEXT;
BEGIN
    IF p_topics IS NULL OR jsonb_array_length(p_topics) = 0 THEN
        RETURN result;
    END IF;

    FOR topic IN SELECT jsonb_array_elements_text(p_topics)
    LOOP
        result := merge_topic_weight(result, topic, p_base_weight);
    END LOOP;

    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Get dominant key from a JSONB weights object
CREATE OR REPLACE FUNCTION get_dominant_key(p_weights JSONB)
RETURNS TEXT AS $$
DECLARE
    max_key TEXT := NULL;
    max_val FLOAT := 0;
    key TEXT;
    val FLOAT;
BEGIN
    FOR key, val IN SELECT k, (v)::float FROM jsonb_each_text(p_weights) AS x(k, v)
    LOOP
        IF val > max_val THEN
            max_val := val;
            max_key := key;
        END IF;
    END LOOP;

    RETURN max_key;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ==========================================================================
-- Session Aggregate Update Function
-- ==========================================================================

CREATE OR REPLACE FUNCTION update_session_aggregate_from_memory()
RETURNS TRIGGER AS $$
DECLARE
    v_session_id TEXT;
    v_existing RECORD;
    v_new_count INT;
    v_new_importance_sum FLOAT;
    v_new_importance_avg FLOAT;
BEGIN
    -- Only process if session_id is set
    v_session_id := NEW.session_id;
    IF v_session_id IS NULL THEN
        RETURN NEW;
    END IF;

    -- Check if aggregate exists
    SELECT * INTO v_existing
    FROM session_aggregates
    WHERE session_id = v_session_id
    FOR UPDATE;

    IF v_existing IS NULL THEN
        -- Create new aggregate
        INSERT INTO session_aggregates (
            session_id,
            user_id,
            agent_id,
            topic_weights,
            category_weights,
            entity_weights,
            sentiment_weights,
            importance_min,
            importance_max,
            importance_avg,
            importance_sum,
            memory_count,
            message_count,
            started_at,
            last_activity_at,
            dominant_topic,
            dominant_category,
            dominant_sentiment,
            access_level,
            created_at,
            updated_at
        ) VALUES (
            v_session_id,
            NEW.user_id,
            NEW.agent_id,
            merge_topics_weights('{}'::jsonb, NEW.topics, NEW.importance),
            merge_topics_weights('{}'::jsonb, NEW.categories, NEW.importance),
            merge_topics_weights('{}'::jsonb, NEW.entities, NEW.importance),
            jsonb_build_object(COALESCE(NEW.sentiment, 'neutral'), NEW.importance),
            NEW.importance,
            NEW.importance,
            NEW.importance,
            NEW.importance,
            1,
            CASE WHEN NEW.message_index > 0 THEN NEW.message_index + 1 ELSE 1 END,
            NEW.created_at,
            NEW.created_at,
            (SELECT jsonb_array_elements_text(NEW.topics) LIMIT 1),
            (SELECT jsonb_array_elements_text(NEW.categories) LIMIT 1),
            NEW.sentiment,
            NEW.access_level,
            NOW(),
            NOW()
        );
    ELSE
        -- Update existing aggregate
        v_new_count := v_existing.memory_count + 1;
        v_new_importance_sum := v_existing.importance_sum + NEW.importance;
        v_new_importance_avg := v_new_importance_sum / v_new_count;

        UPDATE session_aggregates SET
            topic_weights = merge_topics_weights(topic_weights, NEW.topics, NEW.importance),
            category_weights = merge_topics_weights(category_weights, NEW.categories, NEW.importance),
            entity_weights = merge_topics_weights(entity_weights, NEW.entities, NEW.importance),
            sentiment_weights = merge_topic_weight(sentiment_weights, COALESCE(NEW.sentiment, 'neutral'), NEW.importance),
            importance_min = LEAST(importance_min, NEW.importance),
            importance_max = GREATEST(importance_max, NEW.importance),
            importance_avg = v_new_importance_avg,
            importance_sum = v_new_importance_sum,
            memory_count = v_new_count,
            message_count = GREATEST(message_count, NEW.message_index + 1),
            last_activity_at = GREATEST(last_activity_at, NEW.created_at),
            dominant_topic = get_dominant_key(merge_topics_weights(topic_weights, NEW.topics, NEW.importance)),
            dominant_category = get_dominant_key(merge_topics_weights(category_weights, NEW.categories, NEW.importance)),
            dominant_sentiment = get_dominant_key(merge_topic_weight(sentiment_weights, COALESCE(NEW.sentiment, 'neutral'), NEW.importance)),
            -- Promote access level: if new memory is more permissive, update
            access_level = CASE
                WHEN NEW.access_level = 'public' THEN 'public'
                WHEN NEW.access_level = 'shared' AND access_level != 'public' THEN 'shared'
                ELSE access_level
            END,
            updated_at = NOW()
        WHERE session_id = v_session_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Create Triggers
-- ==========================================================================

-- Drop existing trigger if exists (for idempotent migrations)
DROP TRIGGER IF EXISTS trg_memory_session_aggregate ON memories;

-- Create trigger for INSERT
CREATE TRIGGER trg_memory_session_aggregate
    AFTER INSERT ON memories
    FOR EACH ROW
    WHEN (NEW.session_id IS NOT NULL)
    EXECUTE FUNCTION update_session_aggregate_from_memory();

-- Optional: Create trigger for UPDATE (if you want aggregates to reflect edits)
-- This can be expensive if you frequently update memories
-- DROP TRIGGER IF EXISTS trg_memory_session_aggregate_update ON memories;
-- CREATE TRIGGER trg_memory_session_aggregate_update
--     AFTER UPDATE OF topics, categories, entities, sentiment, importance ON memories
--     FOR EACH ROW
--     WHEN (NEW.session_id IS NOT NULL)
--     EXECUTE FUNCTION update_session_aggregate_from_memory();

-- ==========================================================================
-- Recalculate Function (for existing data or repairs)
-- ==========================================================================

-- Recalculate all session aggregates from scratch
-- Use this after migration or if aggregates become inconsistent
CREATE OR REPLACE FUNCTION recalculate_session_aggregates()
RETURNS INTEGER AS $$
DECLARE
    affected_count INTEGER := 0;
BEGIN
    -- Clear existing aggregates
    DELETE FROM session_aggregates;

    -- Recalculate from memories
    INSERT INTO session_aggregates (
        session_id,
        user_id,
        agent_id,
        topic_weights,
        category_weights,
        entity_weights,
        sentiment_weights,
        importance_min,
        importance_max,
        importance_avg,
        importance_sum,
        memory_count,
        message_count,
        started_at,
        last_activity_at,
        dominant_topic,
        dominant_category,
        access_level,
        created_at,
        updated_at
    )
    SELECT
        session_id,
        user_id,
        agent_id,
        -- Aggregate topic weights (simplified: just count occurrences)
        (SELECT jsonb_object_agg(topic, cnt::float / total::float)
         FROM (
             SELECT t.topic, COUNT(*) as cnt, SUM(COUNT(*)) OVER () as total
             FROM memories m2, jsonb_array_elements_text(m2.topics) t(topic)
             WHERE m2.session_id = m.session_id
             GROUP BY t.topic
         ) topic_counts),
        -- Aggregate category weights
        (SELECT jsonb_object_agg(cat, cnt::float / total::float)
         FROM (
             SELECT c.cat, COUNT(*) as cnt, SUM(COUNT(*)) OVER () as total
             FROM memories m2, jsonb_array_elements_text(m2.categories) c(cat)
             WHERE m2.session_id = m.session_id
             GROUP BY c.cat
         ) cat_counts),
        -- Entity weights
        (SELECT jsonb_object_agg(ent, cnt::float / total::float)
         FROM (
             SELECT e.ent, COUNT(*) as cnt, SUM(COUNT(*)) OVER () as total
             FROM memories m2, jsonb_array_elements_text(m2.entities) e(ent)
             WHERE m2.session_id = m.session_id
             GROUP BY e.ent
         ) ent_counts),
        -- Sentiment weights
        jsonb_build_object(
            'positive', (COUNT(*) FILTER (WHERE sentiment = 'positive'))::float / COUNT(*)::float,
            'neutral', (COUNT(*) FILTER (WHERE sentiment = 'neutral'))::float / COUNT(*)::float,
            'negative', (COUNT(*) FILTER (WHERE sentiment = 'negative'))::float / COUNT(*)::float
        ),
        MIN(importance),
        MAX(importance),
        AVG(importance),
        SUM(importance),
        COUNT(*),
        MAX(message_index) + 1,
        MIN(created_at),
        MAX(created_at),
        -- Dominant topic
        (SELECT t.topic FROM memories m2, jsonb_array_elements_text(m2.topics) t(topic)
         WHERE m2.session_id = m.session_id
         GROUP BY t.topic ORDER BY COUNT(*) DESC LIMIT 1),
        -- Dominant category
        (SELECT c.cat FROM memories m2, jsonb_array_elements_text(m2.categories) c(cat)
         WHERE m2.session_id = m.session_id
         GROUP BY c.cat ORDER BY COUNT(*) DESC LIMIT 1),
        -- Most permissive access level
        CASE
            WHEN bool_or(access_level = 'public') THEN 'public'
            WHEN bool_or(access_level = 'shared') THEN 'shared'
            ELSE 'private'
        END,
        NOW(),
        NOW()
    FROM memories m
    WHERE session_id IS NOT NULL
    GROUP BY session_id, user_id, agent_id;

    GET DIAGNOSTICS affected_count = ROW_COUNT;
    RETURN affected_count;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- Verification
-- ==========================================================================

-- Check trigger exists
-- SELECT tgname, tgtype, tgenabled FROM pg_trigger WHERE tgname = 'trg_memory_session_aggregate';

-- Check trigger function
-- SELECT proname, prosrc FROM pg_proc WHERE proname = 'update_session_aggregate_from_memory';

-- Test trigger (insert a memory with session_id and check aggregate)
-- INSERT INTO memories (memory_id, content, memory_type, user_id, session_id, topics)
-- VALUES ('test_mem', 'Test content', 'episodic', 'user1', 'session1', '["orders"]');
-- SELECT * FROM session_aggregates WHERE session_id = 'session1';
