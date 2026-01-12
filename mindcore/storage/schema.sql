-- =============================================================================
-- SAGE (Structured Augmented Generation Engine) - PostgreSQL Schema
-- =============================================================================
-- SVL acts as the KERNEL/COMPILER:
-- - Standard metadata: Enforced by system (message_type, intent, importance, etc.)
-- - User metadata: Assigned by user (topics, categories, custom tags)
-- - All data flows through SVL Gate for validation/canonicalization
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Fuzzy text matching

-- =============================================================================
-- SESSIONS TABLE (Parent - represents conversation threads)
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    -- Primary identifiers
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    conversation_id TEXT,  -- Thread ID for grouping related sessions

    -- Temporal
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),

    -- Session state
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'paused', 'ended', 'archived')),

    -- Aggregated metadata (computed from memories)
    memory_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,

    -- Weighted topic/category aggregates (JSONB for flexibility)
    topic_weights JSONB DEFAULT '{}'::jsonb,
    category_weights JSONB DEFAULT '{}'::jsonb,
    entity_weights JSONB DEFAULT '{}'::jsonb,
    intent_weights JSONB DEFAULT '{}'::jsonb,

    -- Dominant values (most frequent/weighted)
    dominant_topic TEXT,
    dominant_category TEXT,
    dominant_sentiment TEXT,
    dominant_intent TEXT,

    -- Importance/confidence statistics
    importance_min REAL DEFAULT 1.0,
    importance_max REAL DEFAULT 0.0,
    importance_avg REAL DEFAULT 0.0,
    confidence_min REAL DEFAULT 1.0,
    confidence_max REAL DEFAULT 0.0,
    confidence_avg REAL DEFAULT 0.0,

    -- Session-level summary (for hierarchical retrieval)
    summary_text TEXT,

    -- Access control
    access_level TEXT DEFAULT 'private' CHECK (access_level IN ('private', 'team', 'shared', 'global')),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- MEMORIES TABLE (Child - individual memory units)
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    -- ===================
    -- PRIMARY IDENTIFIERS (Standard - System Enforced)
    -- ===================
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    conversation_id TEXT,  -- Thread ID

    -- ===================
    -- CONTENT
    -- ===================
    content TEXT NOT NULL,
    content_hash TEXT,  -- For deduplication

    -- ===================
    -- STANDARD METADATA (System Enforced by SVL Kernel)
    -- ===================
    -- Message classification
    message_type TEXT NOT NULL DEFAULT 'general'
        CHECK (message_type IN (
            'general', 'question', 'answer', 'instruction', 'feedback',
            'clarification', 'confirmation', 'error', 'system', 'tool_call', 'tool_result'
        )),
    message_intent TEXT DEFAULT 'inform'
        CHECK (message_intent IN (
            'inform', 'request', 'confirm', 'deny', 'clarify', 'greet',
            'farewell', 'thank', 'apologize', 'complain', 'suggest', 'command'
        )),
    message_role TEXT DEFAULT 'user'
        CHECK (message_role IN ('user', 'assistant', 'system', 'tool')),

    -- Memory classification
    memory_type TEXT NOT NULL DEFAULT 'episodic'
        CHECK (memory_type IN (
            'episodic', 'semantic', 'procedural', 'preference',
            'entity', 'relationship', 'temporal', 'working'
        )),

    -- Scoring (System computed)
    importance REAL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    confidence REAL DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),

    -- Sentiment analysis
    sentiment TEXT DEFAULT 'neutral'
        CHECK (sentiment IN ('positive', 'negative', 'neutral', 'mixed')),
    sentiment_score REAL DEFAULT 0.0 CHECK (sentiment_score >= -1 AND sentiment_score <= 1),

    -- Ordering within session
    message_index INTEGER DEFAULT 0,

    -- ===================
    -- USER-ASSIGNABLE METADATA (Flexible via SVL)
    -- ===================
    topics JSONB DEFAULT '[]'::jsonb,        -- ["orders", "shipping", "returns"]
    categories JSONB DEFAULT '[]'::jsonb,    -- ["support", "billing"]
    tags JSONB DEFAULT '[]'::jsonb,          -- User-defined tags
    entities JSONB DEFAULT '[]'::jsonb,      -- Extracted entities [{"type": "person", "value": "John"}]

    -- ===================
    -- REINFORCEMENT & ACCESS (System Managed)
    -- ===================
    reinforcement_score REAL DEFAULT 0.0 CHECK (reinforcement_score >= -1 AND reinforcement_score <= 1),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,

    -- Access control
    access_level TEXT DEFAULT 'private'
        CHECK (access_level IN ('private', 'team', 'shared', 'global')),

    -- ===================
    -- TEMPORAL (System Managed)
    -- ===================
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,  -- For temporal memories

    -- ===================
    -- VERSIONING (SVL Schema Version)
    -- ===================
    vocabulary_version TEXT DEFAULT '1.0.0',

    -- ===================
    -- CUSTOM METADATA (User extensible)
    -- ===================
    metadata JSONB DEFAULT '{}'::jsonb,

    -- ===================
    -- FULL-TEXT SEARCH (Auto-generated)
    -- ===================
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(
            (SELECT string_agg(value::text, ' ') FROM jsonb_array_elements_text(topics)), ''
        )), 'B') ||
        setweight(to_tsvector('simple', coalesce(
            (SELECT string_agg(value::text, ' ') FROM jsonb_array_elements_text(categories)), ''
        )), 'C')
    ) STORED
);

-- =============================================================================
-- INDEXES - Optimized for SAGE query patterns
-- =============================================================================

-- Primary lookup indexes
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_conversation_id ON memories(conversation_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id) WHERE agent_id IS NOT NULL;

-- Temporal indexes
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_user_created ON memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session_order ON memories(session_id, message_index);

-- Type indexes
CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_message_type ON memories(message_type);
CREATE INDEX IF NOT EXISTS idx_memories_message_intent ON memories(message_intent);

-- Score indexes (for filtering by importance/confidence)
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC) WHERE importance >= 0.5;
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_memories_reinforcement ON memories(reinforcement_score DESC);

-- JSONB indexes (GIN for array containment queries)
CREATE INDEX IF NOT EXISTS idx_memories_topics ON memories USING GIN(topics jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_categories ON memories USING GIN(categories jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_entities ON memories USING GIN(entities jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN(metadata jsonb_path_ops);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN(search_vector);

-- Fuzzy search index (pg_trgm)
CREATE INDEX IF NOT EXISTS idx_memories_content_trgm ON memories USING GIN(content gin_trgm_ops);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_user_type_created
    ON memories(user_id, memory_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_user_topics
    ON memories USING GIN(user_id, topics jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_session_type
    ON memories(session_id, memory_type) WHERE session_id IS NOT NULL;

-- Access level index
CREATE INDEX IF NOT EXISTS idx_memories_access_level ON memories(access_level);

-- Expiration index (for cleanup)
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at)
    WHERE expires_at IS NOT NULL;

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_conversation_id ON sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user_activity ON sessions(user_id, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_topics ON sessions USING GIN(topic_weights jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_sessions_categories ON sessions USING GIN(category_weights jsonb_path_ops);

-- =============================================================================
-- COMMENTS (Documentation)
-- =============================================================================
COMMENT ON TABLE sessions IS 'Conversation sessions/threads - parent of memories';
COMMENT ON TABLE memories IS 'Individual memory units with standard + user metadata';

COMMENT ON COLUMN memories.message_type IS 'Standard: Type of message (question, answer, etc.) - SVL enforced';
COMMENT ON COLUMN memories.message_intent IS 'Standard: Intent behind message - SVL enforced';
COMMENT ON COLUMN memories.memory_type IS 'Standard: Memory classification - SVL enforced';
COMMENT ON COLUMN memories.importance IS 'Standard: Importance score 0-1 - SVL computed';
COMMENT ON COLUMN memories.confidence IS 'Standard: Confidence score 0-1 - SVL computed';
COMMENT ON COLUMN memories.topics IS 'User-assignable: Array of topic strings';
COMMENT ON COLUMN memories.categories IS 'User-assignable: Array of category strings';
COMMENT ON COLUMN memories.tags IS 'User-assignable: Custom tags array';

COMMENT ON COLUMN sessions.topic_weights IS 'Aggregated topic weights from session memories';
COMMENT ON COLUMN sessions.dominant_topic IS 'Most weighted topic in session';
