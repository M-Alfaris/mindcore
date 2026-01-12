-- Mindcore Search Columns Schema
--
-- Adds combined search_text column for comprehensive text search:
-- 1. Generated column combining content + topics + entities
-- 2. Trigram index for fuzzy search across all fields
--
-- This enables single-query search across all text fields, improving recall
-- for queries that span content and metadata.
--
-- Requirements:
--   - PostgreSQL 12+ (for generated columns)
--   - pg_trgm extension (from extensions.sql)
--
-- Run with: psql $DATABASE_URL -f search_columns.sql
-- See README.md for trigger-based alternative for older PostgreSQL

-- Add combined search text column
-- This is a STORED generated column that concatenates searchable fields
ALTER TABLE memories ADD COLUMN IF NOT EXISTS search_text TEXT
GENERATED ALWAYS AS (
    COALESCE(content, '') || ' ' ||
    COALESCE(
        (SELECT string_agg(elem, ' ')
         FROM jsonb_array_elements_text(COALESCE(topics, '[]'::jsonb)) AS elem),
        ''
    ) || ' ' ||
    COALESCE(
        (SELECT string_agg(elem, ' ')
         FROM jsonb_array_elements_text(COALESCE(entities, '[]'::jsonb)) AS elem),
        ''
    )
) STORED;

-- Create trigram index on combined search text
-- This enables fuzzy search across all text fields simultaneously
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_search_text_trgm
ON memories USING GIN (search_text gin_trgm_ops);

-- Alternative: If the above generated column syntax doesn't work on your
-- PostgreSQL version (requires PG 12+), use a trigger-based approach:
--
-- ALTER TABLE memories ADD COLUMN IF NOT EXISTS search_text TEXT;
--
-- CREATE OR REPLACE FUNCTION update_search_text()
-- RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.search_text := COALESCE(NEW.content, '') || ' ' ||
--         COALESCE(array_to_string(
--             ARRAY(SELECT jsonb_array_elements_text(COALESCE(NEW.topics, '[]'::jsonb))),
--             ' '
--         ), '') || ' ' ||
--         COALESCE(array_to_string(
--             ARRAY(SELECT jsonb_array_elements_text(COALESCE(NEW.entities, '[]'::jsonb))),
--             ' '
--         ), '');
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;
--
-- CREATE TRIGGER trg_update_search_text
-- BEFORE INSERT OR UPDATE ON memories
-- FOR EACH ROW EXECUTE FUNCTION update_search_text();
--
-- -- Backfill existing rows
-- UPDATE memories SET search_text = search_text;

-- Verification
-- SELECT memory_id, LEFT(search_text, 100) FROM memories LIMIT 5;
