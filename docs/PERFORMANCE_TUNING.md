# Mindcore Performance Tuning Guide

This guide covers performance optimization strategies for Mindcore deployments handling varying loads from hobbyist (100s of messages/day) to production (10k+ conversations/day).

## Table of Contents

1. [Quick Start Performance Settings](#quick-start-performance-settings)
2. [Worker Pool Configuration](#worker-pool-configuration)
3. [Database Optimization](#database-optimization)
4. [Cache Configuration](#cache-configuration)
5. [LLM Provider Tuning](#llm-provider-tuning)
6. [Monitoring with Prometheus](#monitoring-with-prometheus)
7. [Load-Specific Configurations](#load-specific-configurations)
8. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

---

## Quick Start Performance Settings

### Environment Variables

```bash
# Worker pool size (1-16, default: 1)
export MINDCORE_ENRICHMENT_WORKERS=4

# LLM provider configuration
export MINDCORE_LLAMA_MODEL_PATH=/path/to/model.gguf  # Local inference
export OPENAI_API_KEY=sk-...                          # Cloud fallback

# Database connection pool
export MINDCORE_DB_POOL_SIZE=10
export MINDCORE_DB_MAX_OVERFLOW=20

# Cache settings
export MINDCORE_CACHE_MAX_SIZE=100
export MINDCORE_CACHE_TTL=3600
```

### Python Configuration

```python
from mindcore import MindcoreClient, MultiAgentConfig

# High-throughput configuration
client = MindcoreClient(
    use_sqlite=False,           # Use PostgreSQL for production
    persistent_cache=True,       # Disk-backed cache
    enrichment_workers=4,        # Parallel enrichment
)

# Start Prometheus metrics server
client.start_metrics_server(port=9090)
```

---

## Worker Pool Configuration

### Understanding Worker Pools

Mindcore uses a background worker pool for message enrichment. Each worker:
- Pulls messages from a persistent queue (SQLite-backed, crash-resilient)
- Detects trivial messages (skips LLM call for greetings, confirmations)
- Calls LLM for full enrichment when needed
- Updates database and cache

### Recommended Worker Counts

| Load Level | Messages/Day | Recommended Workers | Notes |
|------------|-------------|---------------------|-------|
| Hobbyist   | < 1,000     | 1                   | Default, minimal resources |
| Small      | 1k - 5k     | 2                   | Good balance |
| Medium     | 5k - 20k    | 4                   | Recommended for most production |
| High       | 20k - 50k   | 8                   | Requires good LLM capacity |
| Enterprise | 50k+        | 8-16                | Consider distributed setup |

### Configuration Methods

```python
# Method 1: Constructor parameter (highest priority)
client = MindcoreClient(enrichment_workers=4)

# Method 2: Environment variable
# export MINDCORE_ENRICHMENT_WORKERS=4
client = MindcoreClient()  # Will use env var

# Check current configuration
print(f"Workers: {client.enrichment_worker_count}")
```

### Monitoring Worker Health

```python
# Get worker health status
health = client.get_worker_health()

print(f"Pool size: {health['worker_pool_size']}")
print(f"Healthy: {health['healthy']}")
print(f"Issues: {health.get('issues', [])}")

# Get detailed enrichment metrics
metrics = client.get_enrichment_metrics()
print(f"Processed: {metrics['processed_count']}")
print(f"Trivial skipped: {metrics['trivial_skip_count']}")
print(f"Error rate: {metrics['error_rate']:.2%}")
print(f"Avg processing time: {metrics['avg_processing_time_ms']:.1f}ms")
```

---

## Database Optimization

### SQLite (Development/Small Scale)

Good for:
- Development and testing
- < 5k messages/day
- Single-server deployments

```python
client = MindcoreClient(
    use_sqlite=True,
    sqlite_path="mindcore.db"
)
```

**SQLite Tuning:**
```sql
-- Applied automatically, but useful to know:
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache
PRAGMA temp_store = MEMORY;
```

### PostgreSQL (Production)

Recommended for:
- > 5k messages/day
- Multi-server deployments
- High availability requirements

```python
# Configure via environment or config.yaml
client = MindcoreClient(
    use_sqlite=False,
    config_path="config.yaml"
)
```

**PostgreSQL Connection Pool:**
```yaml
# config.yaml
database:
  host: localhost
  port: 5432
  name: mindcore
  user: mindcore_user
  password: ${MINDCORE_DB_PASSWORD}
  pool_size: 10
  max_overflow: 20
```

**Recommended Indexes:**
```sql
-- These are created automatically, but verify they exist:
CREATE INDEX IF NOT EXISTS idx_messages_user_thread
    ON messages(user_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_metadata_topics
    ON messages USING GIN ((metadata->'topics'));
CREATE INDEX IF NOT EXISTS idx_messages_created_at
    ON messages(created_at DESC);
```

---

## Cache Configuration

### In-Memory Cache (cachetools)

Fast but volatile. Good for development.

```python
client = MindcoreClient(
    persistent_cache=False,  # Use in-memory
)
```

### Disk-Backed Cache (diskcache)

Recommended for production. Survives restarts.

```python
client = MindcoreClient(
    persistent_cache=True,   # Default
    cache_dir="/var/cache/mindcore"  # Optional custom path
)
```

### Cache Size Tuning

```yaml
# config.yaml
cache:
  max_size: 100      # Messages per thread (default: 50)
  ttl: 3600          # Seconds (default: None = no TTL)
```

**Sizing Guidelines:**

| Use Case | max_size | ttl |
|----------|----------|-----|
| Short sessions (< 1hr) | 30 | 1800 |
| Normal sessions | 50 | 3600 |
| Long sessions (support) | 100 | 7200 |
| Context-heavy (agents) | 150 | None |

### Cache Invalidation

```python
# Invalidate by topic (when underlying data changes)
count = client.invalidate_cache_by_topic("billing")

# Invalidate stale entries
count = client.invalidate_stale_cache(max_age_seconds=3600)

# Get cache statistics
stats = client.get_cache_stats()
print(f"Hit rate: {stats['invalidation']['hit_rate']:.1%}")
```

---

## LLM Provider Tuning

### Local Inference (llama.cpp)

Best for:
- Cost-sensitive deployments
- Privacy requirements
- Consistent latency

```yaml
# config.yaml
llm:
  provider: llama_cpp  # or "auto" for fallback support
  llama_cpp:
    model_path: /models/llama-3-8b.gguf
    n_ctx: 4096        # Context window
    n_threads: 8       # CPU threads (leave ~2 for system)
    n_gpu_layers: 0    # GPU layers (if CUDA available)
    verbose: false
```

**Model Recommendations:**

| Model Size | RAM Required | Speed | Quality |
|------------|-------------|-------|---------|
| 7B-8B Q4   | 6-8 GB      | Fast  | Good    |
| 13B Q4     | 10-12 GB    | Medium| Better  |
| 7B-8B Q8   | 10-12 GB    | Medium| Better  |

### Cloud Inference (OpenAI)

Best for:
- Scaling without infrastructure
- Highest quality enrichment
- Burst capacity

```yaml
# config.yaml
llm:
  provider: openai  # or "auto"
  openai:
    model: gpt-4o-mini  # Cost-effective
    timeout: 60
    max_retries: 3
```

### Fallback Strategy (Recommended)

```yaml
# config.yaml
llm:
  provider: auto  # Try llama.cpp first, fallback to OpenAI
  llama_cpp:
    model_path: /models/llama-3-8b.gguf
  openai:
    model: gpt-4o-mini
```

### Trivial Message Detection

Mindcore automatically skips LLM calls for trivial messages (greetings, confirmations, etc.), saving 20-30% of LLM costs.

```python
# Check savings
metrics = client.get_enrichment_metrics()
savings = metrics.get('savings_from_trivial', {})
print(f"LLM calls saved: {savings.get('llm_calls_saved', 0)}")
print(f"Estimated cost saved: ${savings.get('estimated_cost_saved_usd', 0):.2f}")
```

---

## Monitoring with Prometheus

### Quick Setup

```python
from mindcore import MindcoreClient

client = MindcoreClient(enrichment_workers=4)

# Start Prometheus metrics server
if client.start_metrics_server(port=9090):
    print("Metrics available at http://localhost:9090/metrics")
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `mindcore_messages_ingested_total` | Counter | Total messages ingested |
| `mindcore_messages_enriched_total` | Counter | Messages enriched (by method) |
| `mindcore_context_requests_total` | Counter | Context retrieval requests |
| `mindcore_cache_operations_total` | Counter | Cache hits/misses |
| `mindcore_errors_total` | Counter | Errors by type |
| `mindcore_enrichment_queue_depth` | Gauge | Queue depth |
| `mindcore_active_workers` | Gauge | Active workers |
| `mindcore_enrichment_duration_seconds` | Histogram | Enrichment latency |
| `mindcore_context_retrieval_duration_seconds` | Histogram | Context latency |

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'mindcore'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### Grafana Dashboard Queries

```promql
# Ingestion rate (messages/minute)
rate(mindcore_messages_ingested_total[5m]) * 60

# Enrichment throughput
rate(mindcore_messages_enriched_total[5m]) * 60

# Cache hit rate
sum(rate(mindcore_cache_operations_total{result="hit"}[5m])) /
sum(rate(mindcore_cache_operations_total[5m]))

# Queue depth (should stay low)
mindcore_enrichment_queue_depth

# P95 enrichment latency
histogram_quantile(0.95, rate(mindcore_enrichment_duration_seconds_bucket[5m]))

# Error rate
rate(mindcore_errors_total[5m])
```

### Programmatic Metrics Access

```python
# Without Prometheus server
metrics = client.get_prometheus_metrics()
print(f"Messages ingested: {metrics.get('messages_ingested', 0)}")
print(f"Queue depth: {metrics.get('queue_depth', 0)}")
```

---

## Load-Specific Configurations

### Hobbyist (< 1k messages/day)

```python
from mindcore import MindcoreClient

client = MindcoreClient(
    use_sqlite=True,
    sqlite_path="mindcore.db",
    persistent_cache=False,  # In-memory OK
    enrichment_workers=1,    # Single worker sufficient
)
```

### Small Production (1k-10k messages/day)

```python
client = MindcoreClient(
    use_sqlite=True,          # SQLite OK up to ~10k/day
    persistent_cache=True,    # Persist cache
    enrichment_workers=2,
)
```

### Medium Production (10k-50k messages/day)

```python
client = MindcoreClient(
    use_sqlite=False,         # PostgreSQL recommended
    persistent_cache=True,
    enrichment_workers=4,
)

# Enable monitoring
client.start_metrics_server(port=9090)
```

### High Production (50k+ messages/day)

```python
client = MindcoreClient(
    use_sqlite=False,         # PostgreSQL required
    persistent_cache=True,
    enrichment_workers=8,
)

# Must have monitoring
client.start_metrics_server(port=9090)
```

**Additional recommendations for high load:**
- Use connection pooling (PgBouncer)
- Consider read replicas for context queries
- Implement rate limiting at API layer
- Run tier migration daily: `client.run_tier_migration()`

---

## Troubleshooting Performance Issues

### High Queue Depth

**Symptom:** `mindcore_enrichment_queue_depth` growing continuously

**Causes & Solutions:**
1. **Insufficient workers**: Increase `enrichment_workers`
2. **Slow LLM**: Switch to faster model or cloud provider
3. **Database bottleneck**: Check DB connection pool, add indexes

```python
# Check queue depth
health = client.get_worker_health()
print(f"Queue depth: {health['workers']['enrichment']['metrics']['queue_size']}")

# Increase workers
client = MindcoreClient(enrichment_workers=8)
```

### High Enrichment Latency

**Symptom:** P95 enrichment > 2s

**Solutions:**
1. Use faster LLM model (7B instead of 13B)
2. Increase LLM threads: `n_threads: 12`
3. Enable GPU acceleration: `n_gpu_layers: 35`
4. Check trivial detection is working

```python
metrics = client.get_enrichment_metrics()
trivial_rate = metrics['trivial_skip_count'] / max(metrics['processed_count'], 1)
print(f"Trivial skip rate: {trivial_rate:.1%}")  # Should be 20-30%
```

### Low Cache Hit Rate

**Symptom:** Cache hit rate < 50%

**Solutions:**
1. Increase cache size
2. Extend TTL
3. Check cache invalidation patterns

```python
stats = client.get_cache_stats()
print(f"Hit rate: {stats['invalidation']['hit_rate']:.1%}")
```

### Memory Issues

**Symptom:** OOM errors or high memory usage

**Solutions:**
1. Reduce cache size
2. Use smaller LLM model
3. Run tier migration more frequently

```python
# Run tier migration to archive old messages
result = client.run_tier_migration()
print(f"Migrated: {result['messages_migrated']} messages")
```

### Database Connection Exhaustion

**Symptom:** "too many connections" errors

**Solutions:**
1. Increase pool size in config
2. Use connection pooler (PgBouncer)
3. Reduce worker count if DB is bottleneck

```yaml
database:
  pool_size: 20
  max_overflow: 40
```

---

## Performance Checklist

Before going to production:

- [ ] Set appropriate `enrichment_workers` for expected load
- [ ] Configure persistent cache with appropriate size
- [ ] Enable Prometheus metrics monitoring
- [ ] Set up Grafana dashboards for key metrics
- [ ] Configure database indexes
- [ ] Test with expected load using load testing tool
- [ ] Set up alerts for queue depth and error rate
- [ ] Schedule regular tier migration

```python
# Quick health check script
def check_mindcore_health(client):
    health = client.get_worker_health()
    metrics = client.get_enrichment_metrics()
    cache = client.get_cache_stats()

    issues = []

    if not health['healthy']:
        issues.append(f"Worker issues: {health.get('issues', [])}")

    if metrics.get('error_rate', 0) > 0.05:
        issues.append(f"High error rate: {metrics['error_rate']:.1%}")

    if cache['invalidation']['hit_rate'] < 0.5:
        issues.append(f"Low cache hit rate: {cache['invalidation']['hit_rate']:.1%}")

    if health['workers']['enrichment']['metrics'].get('queue_size', 0) > 100:
        issues.append(f"High queue depth: {health['workers']['enrichment']['metrics']['queue_size']}")

    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'metrics': {
            'workers': health['worker_pool_size'],
            'processed': metrics['processed_count'],
            'error_rate': metrics.get('error_rate', 0),
            'cache_hit_rate': cache['invalidation']['hit_rate'],
        }
    }

# Usage
result = check_mindcore_health(client)
if not result['healthy']:
    print("Issues found:", result['issues'])
```
