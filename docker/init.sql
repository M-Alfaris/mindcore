-- Mindcore PostgreSQL Initialization Script
-- This script runs automatically when the PostgreSQL container starts for the first time.

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Messages table (core storage)
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    raw_text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    -- Multi-agent support
    agent_id VARCHAR(255),
    visibility VARCHAR(50) DEFAULT 'private',
    sharing_groups TEXT[] DEFAULT '{}'
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_messages_user_thread ON messages(user_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON messages(thread_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_id) WHERE agent_id IS NOT NULL;

-- JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_messages_topics ON messages USING GIN ((metadata->'topics'));
CREATE INDEX IF NOT EXISTS idx_messages_categories ON messages USING GIN ((metadata->'categories'));
CREATE INDEX IF NOT EXISTS idx_messages_tags ON messages USING GIN ((metadata->'tags'));
CREATE INDEX IF NOT EXISTS idx_messages_intent ON messages ((metadata->>'intent'));
CREATE INDEX IF NOT EXISTS idx_messages_importance ON messages (((metadata->>'importance')::float));

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_messages_text_search ON messages USING GIN (to_tsvector('english', raw_text));

-- Agents table (for multi-agent mode)
CREATE TABLE IF NOT EXISTS agents (
    agent_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    owner_id VARCHAR(255) NOT NULL,
    groups TEXT[] DEFAULT '{}',
    roles TEXT[] DEFAULT '{}',
    api_key_hash VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agents_owner ON agents(owner_id);
CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(is_active) WHERE is_active = true;

-- Access policies table
CREATE TABLE IF NOT EXISTS access_policies (
    resource_id VARCHAR(255) PRIMARY KEY,
    resource_type VARCHAR(100) NOT NULL DEFAULT 'message',
    owner_id VARCHAR(255) NOT NULL,
    owner_org VARCHAR(255),
    visibility VARCHAR(50) DEFAULT 'private',
    sharing_groups TEXT[] DEFAULT '{}',
    acl JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_policies_owner ON access_policies(owner_id);
CREATE INDEX IF NOT EXISTS idx_policies_visibility ON access_policies(visibility);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR(255),
    resource_id VARCHAR(255),
    resource_type VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    success BOOLEAN DEFAULT true,
    details JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp DESC);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Thread summaries table (for context compression)
CREATE TABLE IF NOT EXISTS thread_summaries (
    thread_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    summary TEXT,
    key_points JSONB DEFAULT '[]',
    message_count INTEGER DEFAULT 0,
    last_message_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_summaries_user ON thread_summaries(user_id);

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mindcore_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mindcore_user;

-- Helpful comment
COMMENT ON DATABASE mindcore IS 'Mindcore: Intelligent memory and context management for AI agents';
