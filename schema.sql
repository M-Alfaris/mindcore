-- Mindcore Database Schema
-- Version: 0.1.0
--
-- This schema is automatically created by Mindcore when you initialize
-- the DatabaseManager. You can also run this file manually to set up
-- the database.
--
-- Usage:
--   psql -U postgres -d mindcore -f schema.sql

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    raw_text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_user_thread
    ON messages(user_id, thread_id);

CREATE INDEX IF NOT EXISTS idx_thread_created
    ON messages(thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_metadata_topics
    ON messages USING GIN ((metadata->'topics'));

CREATE INDEX IF NOT EXISTS idx_created_at
    ON messages(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_created
    ON messages(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_session
    ON messages(session_id);

-- Add comments for documentation
COMMENT ON TABLE messages IS 'Stores all messages with enriched metadata';
COMMENT ON COLUMN messages.message_id IS 'Unique identifier for the message';
COMMENT ON COLUMN messages.user_id IS 'Identifier for the user who sent the message';
COMMENT ON COLUMN messages.thread_id IS 'Identifier for the conversation thread';
COMMENT ON COLUMN messages.session_id IS 'Identifier for the session';
COMMENT ON COLUMN messages.role IS 'Role of the message sender (user, assistant, system, tool)';
COMMENT ON COLUMN messages.raw_text IS 'Original message text content';
COMMENT ON COLUMN messages.metadata IS 'JSON metadata including topics, sentiment, intent, etc.';
COMMENT ON COLUMN messages.created_at IS 'Timestamp when the message was created';

-- Example queries for reference:

-- Get recent messages for a thread:
-- SELECT * FROM messages
-- WHERE thread_id = 'thread_456'
-- ORDER BY created_at DESC
-- LIMIT 50;

-- Search messages by topic:
-- SELECT * FROM messages
-- WHERE metadata->'topics' ? 'AI'
-- ORDER BY created_at DESC;

-- Get messages with high importance:
-- SELECT * FROM messages
-- WHERE (metadata->>'importance')::float > 0.7
-- ORDER BY created_at DESC;

-- Get messages by intent:
-- SELECT * FROM messages
-- WHERE metadata->>'intent' = 'ask_question'
-- ORDER BY created_at DESC;
