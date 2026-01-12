-- Mindcore SAGE - PostgreSQL Initialization
-- This script runs automatically when the container first starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mindcore TO mindcore;

-- Log success
DO $$
BEGIN
    RAISE NOTICE 'Mindcore database initialized with pg_trgm extension';
END $$;
