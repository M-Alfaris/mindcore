-- Mindcore Search Extensions Schema
--
-- Enables enhanced search capabilities for PostgreSQL storage backend:
-- 1. pg_trgm extension for trigram similarity matching (fuzzy search)
-- 2. pg_search extension (ParadeDB) for BM25 full-text search
-- 3. GIN indexes for efficient trigram and BM25 lookups
--
-- Requirements:
--   - PostgreSQL 12+ (for pg_trgm)
--   - ParadeDB (optional, for BM25) - comment out pg_search lines if unavailable
--
-- Run with: psql $DATABASE_URL -f extensions.sql
-- See README.md for full installation guide

-- Enable pg_trgm extension for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable ParadeDB pg_search extension for BM25
-- Note: Requires ParadeDB PostgreSQL or the pg_search extension installed
-- Comment out if ParadeDB is not available
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Add trigram index on content for fuzzy matching
-- Uses GIN index for efficient similarity lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_content_trgm
ON memories USING GIN (content gin_trgm_ops);

-- Add trigram index on topics (cast to text for trigram matching)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_topics_text_trgm
ON memories USING GIN ((topics::text) gin_trgm_ops);

-- Add trigram index on entities for fuzzy entity matching
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_entities_text_trgm
ON memories USING GIN ((entities::text) gin_trgm_ops);

-- Configure similarity threshold (can be tuned per deployment)
-- Default is 0.3, lower values = more matches (less strict)
-- This is a session-level setting, consider adding to connection config
-- SET pg_trgm.similarity_threshold = 0.2;

-- Create ParadeDB BM25 index on memories table
-- This enables efficient full-text search with BM25 ranking
-- Note: Comment out if ParadeDB is not available
CALL paradedb.create_bm25(
    index_name => 'memories_bm25',
    table_name => 'memories',
    key_field => 'memory_id',
    text_fields => paradedb.field(
        'content',
        tokenizer => paradedb.tokenizer('en_stem')
    )
);

-- Create BM25 index on session summaries for session-level search
-- Only useful if summary_text is populated
CALL paradedb.create_bm25(
    index_name => 'sessions_bm25',
    table_name => 'session_aggregates',
    key_field => 'session_id',
    text_fields => paradedb.field(
        'summary_text',
        tokenizer => paradedb.tokenizer('en_stem')
    )
);

-- Verification queries (run after migration)
-- SELECT * FROM pg_extension WHERE extname IN ('pg_trgm', 'pg_search');
-- SELECT similarity('orders', 'ordrs');  -- Should return ~0.5
-- SELECT * FROM paradedb.indexes;
