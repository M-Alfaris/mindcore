-- Mindcore pgvector Schema
--
-- Enables vector similarity search using pgvector 0.8.0+
-- Replaces JSONB embedding storage with native vector type for:
-- - Efficient similarity search (cosine, L2, inner product)
-- - HNSW and IVFFlat indexes for fast approximate nearest neighbor
-- - Native vector operations in PostgreSQL
--
-- Requirements:
--   - PostgreSQL 14+ (for optimal performance)
--   - pgvector 0.8.0+ (for halfvec and improved HNSW)
--
-- Run with: psql $DATABASE_URL -f pgvector.sql
-- See README.md for full installation guide

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Default embedding dimension (OpenAI ada-002 = 1536, text-embedding-3-small = 1536)
-- Adjust based on your embedding model
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'memories' AND column_name = 'embedding_vector') THEN
        -- Add vector column for embeddings
        -- Using 1536 dimensions (common for OpenAI embeddings)
        -- Change to 768 for sentence-transformers, 384 for MiniLM, etc.
        ALTER TABLE memories ADD COLUMN embedding_vector vector(1536);

        RAISE NOTICE 'Added embedding_vector column (1536 dimensions)';
    END IF;
END $$;

-- Migrate existing JSONB embeddings to vector format
-- Only run once, then can be removed
DO $$
DECLARE
    migrated_count INTEGER := 0;
BEGIN
    UPDATE memories
    SET embedding_vector = embedding::text::vector
    WHERE embedding IS NOT NULL
      AND embedding_vector IS NULL
      AND jsonb_array_length(embedding) = 1536;

    GET DIAGNOSTICS migrated_count = ROW_COUNT;

    IF migrated_count > 0 THEN
        RAISE NOTICE 'Migrated % embeddings from JSONB to vector', migrated_count;
    END IF;
END $$;

-- Create HNSW index for fast approximate nearest neighbor search
-- HNSW is faster for queries, IVFFlat is faster to build
-- Using cosine distance (most common for text embeddings)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_embedding_hnsw
ON memories USING hnsw (embedding_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (faster to build, slower queries)
-- Uncomment if you have >1M vectors and need faster index builds
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_embedding_ivfflat
-- ON memories USING ivfflat (embedding_vector vector_cosine_ops)
-- WITH (lists = 100);

-- Create partial index for user-scoped vector search
-- Useful if you frequently search within a single user's memories
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_embedding
ON memories (user_id, embedding_vector)
WHERE embedding_vector IS NOT NULL;

-- ==========================================================================
-- Vector Search Functions
-- ==========================================================================

-- Semantic search function with filtering
-- Returns memories similar to input vector, filtered by user/topics
CREATE OR REPLACE FUNCTION search_memories_semantic(
    p_embedding vector(1536),
    p_user_id TEXT,
    p_topics TEXT[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0,
    p_limit INT DEFAULT 50,
    p_min_similarity FLOAT DEFAULT 0.5
) RETURNS TABLE (
    memory_id TEXT,
    content TEXT,
    memory_type TEXT,
    topics JSONB,
    importance FLOAT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.memory_id,
        m.content,
        m.memory_type,
        m.topics,
        m.importance,
        1 - (m.embedding_vector <=> p_embedding) AS similarity
    FROM memories m
    WHERE m.user_id = p_user_id
      AND m.embedding_vector IS NOT NULL
      AND (m.expires_at IS NULL OR m.expires_at > NOW())
      AND m.importance >= p_min_importance
      AND (p_topics IS NULL OR m.topics ?| p_topics)
      AND 1 - (m.embedding_vector <=> p_embedding) >= p_min_similarity
    ORDER BY m.embedding_vector <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

-- Hybrid search: combines semantic similarity with keyword relevance
-- Uses RRF (Reciprocal Rank Fusion) to merge rankings
CREATE OR REPLACE FUNCTION search_memories_hybrid(
    p_embedding vector(1536),
    p_query TEXT,
    p_user_id TEXT,
    p_attention_hints TEXT[] DEFAULT NULL,
    p_limit INT DEFAULT 50,
    p_semantic_weight FLOAT DEFAULT 0.6  -- Weight for semantic vs keyword
) RETURNS TABLE (
    memory_id TEXT,
    content TEXT,
    memory_type TEXT,
    topics JSONB,
    importance FLOAT,
    semantic_score FLOAT,
    keyword_score FLOAT,
    hybrid_score FLOAT
) AS $$
WITH semantic_results AS (
    SELECT
        m.memory_id,
        m.content,
        m.memory_type,
        m.topics,
        m.importance,
        1 - (m.embedding_vector <=> p_embedding) AS sim_score,
        ROW_NUMBER() OVER (ORDER BY m.embedding_vector <=> p_embedding) AS sem_rank
    FROM memories m
    WHERE m.user_id = p_user_id
      AND m.embedding_vector IS NOT NULL
      AND (m.expires_at IS NULL OR m.expires_at > NOW())
    ORDER BY m.embedding_vector <=> p_embedding
    LIMIT p_limit * 2
),
keyword_results AS (
    SELECT
        m.memory_id,
        ts_rank(m.search_vector, plainto_tsquery('english', p_query)) AS kw_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(m.search_vector, plainto_tsquery('english', p_query)) DESC) AS kw_rank
    FROM memories m
    WHERE m.user_id = p_user_id
      AND m.search_vector @@ plainto_tsquery('english', p_query)
      AND (m.expires_at IS NULL OR m.expires_at > NOW())
    ORDER BY kw_score DESC
    LIMIT p_limit * 2
),
combined AS (
    SELECT
        COALESCE(s.memory_id, k.memory_id) AS memory_id,
        s.content,
        s.memory_type,
        s.topics,
        s.importance,
        COALESCE(s.sim_score, 0) AS semantic_score,
        COALESCE(k.kw_score, 0) AS keyword_score,
        -- RRF formula: 1/(k + rank) where k = 60 is standard
        (p_semantic_weight * (1.0 / (60 + COALESCE(s.sem_rank, 1000)))) +
        ((1 - p_semantic_weight) * (1.0 / (60 + COALESCE(k.kw_rank, 1000)))) AS hybrid_score
    FROM semantic_results s
    FULL OUTER JOIN keyword_results k ON s.memory_id = k.memory_id
)
SELECT
    c.memory_id,
    c.content,
    c.memory_type,
    c.topics,
    c.importance,
    c.semantic_score,
    c.keyword_score,
    c.hybrid_score
FROM combined c
WHERE c.memory_id IS NOT NULL
ORDER BY c.hybrid_score DESC
LIMIT p_limit;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Find similar memories to a given memory
-- Useful for deduplication, clustering, or "related memories" features
CREATE OR REPLACE FUNCTION find_similar_memories(
    p_memory_id TEXT,
    p_limit INT DEFAULT 10,
    p_min_similarity FLOAT DEFAULT 0.8
) RETURNS TABLE (
    memory_id TEXT,
    content TEXT,
    similarity FLOAT
) AS $$
DECLARE
    source_embedding vector(1536);
    source_user_id TEXT;
BEGIN
    -- Get the source memory's embedding and user_id
    SELECT m.embedding_vector, m.user_id
    INTO source_embedding, source_user_id
    FROM memories m
    WHERE m.memory_id = p_memory_id;

    IF source_embedding IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        m.memory_id,
        m.content,
        1 - (m.embedding_vector <=> source_embedding) AS similarity
    FROM memories m
    WHERE m.memory_id != p_memory_id
      AND m.user_id = source_user_id
      AND m.embedding_vector IS NOT NULL
      AND 1 - (m.embedding_vector <=> source_embedding) >= p_min_similarity
    ORDER BY m.embedding_vector <=> source_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

-- ==========================================================================
-- Utility Functions
-- ==========================================================================

-- Convert JSONB array to vector (for backward compatibility)
CREATE OR REPLACE FUNCTION jsonb_to_vector(p_jsonb JSONB, p_dim INT DEFAULT 1536)
RETURNS vector AS $$
BEGIN
    IF p_jsonb IS NULL OR jsonb_array_length(p_jsonb) != p_dim THEN
        RETURN NULL;
    END IF;
    RETURN p_jsonb::text::vector;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Average multiple vectors (for cluster centroids)
CREATE OR REPLACE FUNCTION avg_vectors(p_vectors vector[])
RETURNS vector AS $$
DECLARE
    result vector;
    dim INT;
    i INT;
    sum_val FLOAT;
BEGIN
    IF array_length(p_vectors, 1) IS NULL OR array_length(p_vectors, 1) = 0 THEN
        RETURN NULL;
    END IF;

    dim := vector_dims(p_vectors[1]);

    -- Use pgvector's built-in average
    SELECT AVG(v) INTO result FROM unnest(p_vectors) v;

    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- ==========================================================================
-- Verification Queries
-- ==========================================================================

-- Check pgvector is installed
-- SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check vector column exists
-- SELECT column_name, data_type, udt_name
-- FROM information_schema.columns
-- WHERE table_name = 'memories' AND column_name = 'embedding_vector';

-- Test similarity search (after inserting some vectors)
-- SELECT * FROM search_memories_semantic(
--     '[0.1, 0.2, ...]'::vector(1536),  -- Your query embedding
--     'user_123',
--     ARRAY['orders', 'shipping'],
--     0.0,
--     10,
--     0.5
-- );

-- Check index usage
-- EXPLAIN ANALYZE SELECT * FROM memories
-- ORDER BY embedding_vector <=> '[0.1, 0.2, ...]'::vector(1536)
-- LIMIT 10;
