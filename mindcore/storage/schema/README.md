# Mindcore PostgreSQL Schema Extensions

This folder contains PostgreSQL schema extensions for enhanced search, performance, and operational capabilities. These extensions transform Mindcore from a basic storage system into a production-ready, high-performance memory platform.

## Quick Start

```bash
# Core extensions (required for advanced search)
psql $DATABASE_URL -f extensions.sql
psql $DATABASE_URL -f search_columns.sql
psql $DATABASE_URL -f ranking_functions.sql

# Vector search (recommended for semantic search)
psql $DATABASE_URL -f pgvector.sql

# Automatic session updates (recommended)
psql $DATABASE_URL -f session_triggers.sql

# Analytics and monitoring (recommended for production)
psql $DATABASE_URL -f materialized_views.sql
psql $DATABASE_URL -f pg_stat_statements.sql

# Scheduled maintenance (requires pg_cron)
psql $DATABASE_URL -f pg_cron.sql

# Table partitioning (for large deployments, requires migration)
# See partitioning section below
```

## Requirements Matrix

| File | PostgreSQL | Extensions | Notes |
|------|------------|------------|-------|
| extensions.sql | 12+ | pg_trgm, pg_search (optional) | ParadeDB for BM25 |
| search_columns.sql | 12+ | - | Generated columns |
| ranking_functions.sql | 12+ | pg_trgm | SQL ranking functions |
| pgvector.sql | 14+ | pgvector 0.8.0+ | Vector similarity search |
| session_triggers.sql | 12+ | - | Automatic aggregates |
| materialized_views.sql | 12+ | - | Pre-computed stats |
| pg_cron.sql | 12+ | pg_cron | Scheduled jobs |
| pg_stat_statements.sql | 12+ | pg_stat_statements | Query monitoring |
| partitioning.sql | 12+ | - | Time-based partitioning |

## Schema Files

### Core Search (`extensions.sql`, `search_columns.sql`, `ranking_functions.sql`)

Enables enhanced text search with fuzzy matching and custom ranking.

**What it does:**

- `pg_trgm`: Trigram similarity for fuzzy/typo-tolerant search
- `pg_search` (ParadeDB): BM25 full-text ranking
- `rank_memory()`: Multi-component scoring (content + topics + recency + importance)
- `rank_session()`: Session-level relevance ranking

**Performance impact:** 7-10x faster than Python-side scoring.

### Vector Search (`pgvector.sql`)

Enables semantic similarity search using embedding vectors.

**What it does:**

- Adds `embedding_vector` column (vector type, 1536 dimensions)
- HNSW index for fast approximate nearest neighbor search
- `search_memories_semantic()`: Pure vector similarity search
- `search_memories_hybrid()`: Combines vector + keyword search with RRF
- `find_similar_memories()`: Find duplicates or related memories

**Use cases:**

- Semantic search ("find memories about customer frustration" vs exact keyword match)
- Memory deduplication
- Clustering and topic modeling
- RAG (Retrieval Augmented Generation) pipelines

**Setup:**

```bash
# Install pgvector
# Docker: Use pgvector/pgvector:pg16 image
# Ubuntu: sudo apt install postgresql-16-pgvector
# macOS: brew install pgvector

psql $DATABASE_URL -f pgvector.sql

# Migrate existing JSONB embeddings (automatic in script)
# Test similarity search
SELECT * FROM search_memories_semantic(
    '[0.1, 0.2, ...]'::vector(1536),
    'user_123',
    ARRAY['orders'],
    0.0,
    10,
    0.5
);
```

### Session Triggers (`session_triggers.sql`)

Automatically updates session aggregates when memories are inserted.

**What it does:**

- Trigger on `memories` INSERT → updates `session_aggregates`
- Atomic updates (no race conditions)
- Weighted topic/category aggregation
- Replaces Python-side `update_session_aggregate_from_memory()`

**Benefits:**

- Single transaction (insert + aggregate update)
- No Python round-trips
- Consistent aggregates even with concurrent inserts

**Functions:**

- `merge_topic_weight()`: Weighted average for topic weights
- `get_dominant_key()`: Find highest-weighted key
- `recalculate_session_aggregates()`: Full rebuild for repairs

### Materialized Views (`materialized_views.sql`)

Pre-computed statistics for dashboards and analytics.

**Views:**

- `mv_user_stats`: Per-user memory counts, topic distribution, activity
- `mv_session_stats`: Enhanced session metrics (duration, sentiment score)
- `mv_topic_analytics`: Global topic frequency and trends
- `mv_memory_health`: System health and data quality metrics
- `mv_daily_stats`: Time-series rollups for analytics

**Refresh:**

```sql
-- Manual refresh
SELECT * FROM refresh_all_materialized_views();

-- Critical views only (faster)
SELECT refresh_critical_views();
```

**Scheduling:** Use pg_cron (see below) for automatic refresh.

### Scheduled Jobs (`pg_cron.sql`)

Automates maintenance tasks via pg_cron.

**Jobs:**

| Job | Schedule | Function |
|-----|----------|----------|
| Cleanup expired memories | Hourly | `cleanup_expired_memories()` |
| Cleanup orphaned sessions | Daily 3 AM | `cleanup_orphaned_sessions()` |
| Cleanup old transfers | Daily 3:30 AM | `cleanup_old_transfers()` |
| Refresh critical views | Every 15 min | `refresh_critical_views()` |
| Full view refresh | Daily 4 AM | `refresh_all_materialized_views()` |
| Decay reinforcement | Weekly Sun 2 AM | `decay_reinforcement_scores()` |
| VACUUM ANALYZE | Weekly Sat 3 AM | `VACUUM ANALYZE memories` |

**Setup:**

```bash
# 1. Enable pg_cron in postgresql.conf
shared_preload_libraries = 'pg_cron'
cron.database_name = 'mindcore'

# 2. Restart PostgreSQL
sudo systemctl restart postgresql

# 3. Run schema
psql $DATABASE_URL -f pg_cron.sql

# 4. Schedule jobs (uncomment in file or run manually)
SELECT safe_schedule('0 * * * *', 'SELECT cleanup_expired_memories()', 'cleanup_expired_memories');
```

**Management:**

```sql
-- List jobs
SELECT * FROM list_scheduled_jobs();

-- Check history
SELECT * FROM get_job_history(10);

-- Pause/resume
SELECT pause_job('cleanup_expired_memories');
SELECT resume_job('cleanup_expired_memories');
```

### Query Monitoring (`pg_stat_statements.sql`)

Tracks query performance for optimization.

**Views:**

- `v_slow_queries`: Top queries by total execution time
- `v_frequent_queries`: Most called queries
- `v_io_intensive_queries`: Queries with high disk I/O
- `v_variable_queries`: Queries with inconsistent performance
- `v_memory_search_stats`: Mindcore search query performance
- `v_vector_search_stats`: Vector search performance

**Functions:**

- `get_query_summary()`: Overall statistics
- `suggest_indexes()`: Index recommendations
- `snapshot_query_performance()`: Save stats for trending
- `compare_query_performance()`: Compare periods

**Setup:**

```bash
# Enable in postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.max = 10000

# Restart and run schema
psql $DATABASE_URL -f pg_stat_statements.sql
```

**Usage:**

```sql
-- Get summary
SELECT * FROM get_query_summary();

-- Find slow queries
SELECT * FROM v_slow_queries;

-- Get index suggestions
SELECT * FROM suggest_indexes();
```

### Table Partitioning (`partitioning.sql`)

Splits the memories table into monthly partitions for large deployments.

**Benefits:**

- Faster queries with partition pruning
- Parallel query execution
- Easy archival (detach old partitions)
- Reduced VACUUM time

**Strategy:** RANGE partitioning by `created_at` (monthly)

**When to use:**

- 10M+ memories
- Frequent time-range queries
- Need to archive old data
- Performance issues with large table scans

**Migration Process:**

```sql
-- 1. Create partitioned structure (non-destructive)
\i partitioning.sql

-- 2. Migrate data (during low-traffic period)
SELECT * FROM migrate_to_partitioned_table(10000, 100);

-- 3. Verify counts match
SELECT COUNT(*) FROM memories;
SELECT COUNT(*) FROM memories_partitioned;

-- 4. Swap tables (brief lock)
SELECT swap_to_partitioned_table();

-- 5. Verify and cleanup
DROP TABLE memories_old;  -- After verification
```

**Maintenance:**

```sql
-- Create future partitions
SELECT * FROM create_future_partitions(12);

-- List partitions
SELECT * FROM list_partitions();

-- Archive old partition
SELECT detach_partition('memories_y2024m01');
```

## Installation Order

For a fresh deployment:

```bash
# 1. Core schema (via PostgresStorage auto-init)
# Tables created automatically on first connection

# 2. Search extensions
psql $DATABASE_URL -f extensions.sql
psql $DATABASE_URL -f search_columns.sql
psql $DATABASE_URL -f ranking_functions.sql

# 3. Vector search
psql $DATABASE_URL -f pgvector.sql

# 4. Triggers
psql $DATABASE_URL -f session_triggers.sql

# 5. Analytics
psql $DATABASE_URL -f materialized_views.sql
psql $DATABASE_URL -f pg_stat_statements.sql

# 6. Scheduled jobs (requires pg_cron configured)
psql $DATABASE_URL -f pg_cron.sql

# 7. Partitioning (optional, for large deployments)
# See migration process above
```

## Configuration

### SearchConfig (Python)

```python
from mindcore.storage.config import SearchConfig

config = SearchConfig(
    use_trigram_search=True,
    use_bm25_search=True,
    trigram_similarity_threshold=0.2,
    ranking_weights={
        "content": 0.15,
        "topic": 0.25,
        "recency": 0.15,
        "reinforcement": 0.2,
        "importance": 0.15,
        "popularity": 0.1,
    }
)

storage = PostgresStorage(connection_string, search_config=config)
```

### PostgreSQL Settings

Recommended `postgresql.conf` for production:

```ini
# Extensions
shared_preload_libraries = 'pg_cron,pg_stat_statements'
cron.database_name = 'mindcore'
pg_stat_statements.track = all
pg_stat_statements.max = 10000

# Memory (adjust based on available RAM)
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Parallelism
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# Vector search (pgvector)
# Increase for large datasets
# hnsw.ef_search = 100  # Higher = more accurate, slower
```

## Troubleshooting

### Extension not found

```sql
-- Check available extensions
SELECT * FROM pg_available_extensions WHERE name IN ('pg_trgm', 'vector', 'pg_cron');

-- Install missing (requires superuser)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Trigger not firing

```sql
-- Check trigger exists
SELECT * FROM pg_trigger WHERE tgname = 'trg_memory_session_aggregate';

-- Check trigger is enabled
SELECT tgenabled FROM pg_trigger WHERE tgname = 'trg_memory_session_aggregate';
-- 'O' = enabled, 'D' = disabled
```

### Materialized view stale

```sql
-- Check last refresh
SELECT * FROM mv_memory_health;

-- Force refresh
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_stats;
```

### Slow vector search

```sql
-- Check index exists
SELECT * FROM pg_indexes WHERE indexname LIKE '%embedding%';

-- Check index is being used
EXPLAIN ANALYZE SELECT * FROM memories
ORDER BY embedding_vector <=> '[...]'::vector(1536)
LIMIT 10;

-- Increase ef_search for better accuracy (but slower)
SET hnsw.ef_search = 200;
```

### Partition not found for date

```sql
-- Create missing partition
SELECT create_monthly_partition(2026, 7);

-- Or create next 12 months
SELECT * FROM create_future_partitions(12);
```

## Performance Benchmarks

Tested with 1M memories on PostgreSQL 16 / 32GB RAM:

| Operation | Without Extensions | With Extensions |
|-----------|-------------------|-----------------|
| Fuzzy search | ~200ms | ~15ms |
| Vector search (top 50) | N/A | ~25ms |
| Hybrid search | N/A | ~40ms |
| Memory scoring | ~150ms (Python) | ~20ms (SQL) |
| Session ranking | ~50ms (Python) | ~8ms (SQL) |
| User stats lookup | ~500ms | ~2ms (materialized) |

Run `benchmarks/search_comparison.py` for your deployment metrics.
