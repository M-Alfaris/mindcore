# Mindcore Dashboard

Modern web dashboard for monitoring, observability, and management of Mindcore AI agents.

## Features

- **Overview** - High-level metrics, charts, system health
- **Performance** - Response times, latency distribution, P95/P99 metrics
- **Messages** - Browse, search, filter all messages with full metadata
- **Threads** - Conversation thread management with details
- **Sessions** - Session analytics and tracking
- **Logs** - Real-time system logs with level filtering
- **Models** - Visual model selector for cloud and local LLMs
- **Configuration** - LLM settings, memory, cache configuration

## Quick Start

### Start the Backend

```bash
# From project root
python -m mindcore.api.server

# Or with custom port
python -m mindcore.api.server --port 8000
```

### Start the Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard runs on `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

## Database Schema

The dashboard uses SQLite (or PostgreSQL) for persistent storage.

### Core Tables

#### `messages`
Stores all conversation messages.

| Column | Type | Description |
|--------|------|-------------|
| message_id | TEXT | Primary key, unique message identifier |
| user_id | TEXT | User identifier |
| thread_id | TEXT | Conversation thread identifier |
| session_id | TEXT | Session identifier |
| role | TEXT | Message role (user, assistant, system, tool) |
| raw_text | TEXT | Message content |
| metadata | TEXT | JSON metadata (topics, intent, sentiment) |
| created_at | TIMESTAMP | Creation timestamp |

### Metrics Tables

#### `performance_metrics`
Stores timing data for LLM calls, enrichment, and retrieval operations.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| metric_type | TEXT | Type: 'llm_call', 'enrichment', 'retrieval', 'storage' |
| operation | TEXT | Specific operation name |
| model | TEXT | LLM model name (for llm_call type) |
| prompt_tokens | INTEGER | Input token count |
| completion_tokens | INTEGER | Output token count |
| total_time_ms | INTEGER | Execution time in milliseconds |
| success | INTEGER | 1=success, 0=failure |
| error_message | TEXT | Error message if failed |
| user_id | TEXT | Associated user |
| thread_id | TEXT | Associated thread |
| message_id | TEXT | Associated message |
| metadata | TEXT | Additional JSON metadata |
| created_at | TIMESTAMP | Timestamp |

#### `tool_calls`
Tracks every tool invocation with timing and status.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| tool_call_id | TEXT | Unique tool call identifier |
| tool_name | TEXT | Name of the tool |
| message_id | TEXT | Associated message ID |
| thread_id | TEXT | Associated thread ID |
| user_id | TEXT | Associated user ID |
| execution_time_ms | INTEGER | Execution time in milliseconds |
| success | INTEGER | 1=success, 0=failure |
| error_message | TEXT | Error message if failed |
| input_data | TEXT | Tool input (JSON) |
| output_data | TEXT | Tool output (truncated) |
| created_at | TIMESTAMP | Timestamp |

#### `sessions`
Aggregated session analytics.

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | Primary key |
| user_id | TEXT | User identifier |
| started_at | TIMESTAMP | Session start time |
| last_activity_at | TIMESTAMP | Last activity timestamp |
| thread_count | INTEGER | Number of threads in session |
| message_count | INTEGER | Total messages in session |
| total_llm_calls | INTEGER | Total LLM API calls |
| total_tool_calls | INTEGER | Total tool invocations |
| total_latency_ms | INTEGER | Cumulative latency |
| avg_latency_ms | INTEGER | Average latency |
| metadata | TEXT | Additional metadata |

#### `dashboard_settings`
User preferences and configuration.

| Column | Type | Description |
|--------|------|-------------|
| setting_key | TEXT | Primary key, setting name |
| setting_value | TEXT | Setting value (stored as string) |
| setting_type | TEXT | Type: 'string', 'number', 'boolean', 'json' |
| description | TEXT | Human-readable description |
| updated_at | TIMESTAMP | Last update time |

## Dashboard Settings

Default settings stored in `dashboard_settings` table:

| Setting | Default | Description |
|---------|---------|-------------|
| `metrics_retention_days` | 30 | Days to retain performance metrics |
| `auto_refresh_interval` | 30 | Dashboard auto-refresh interval (seconds) |
| `max_log_entries` | 1000 | Maximum log entries in memory |
| `enable_performance_tracking` | true | Enable performance metric collection |
| `enable_tool_tracking` | true | Enable tool call tracking |
| `latency_warning_threshold_ms` | 1000 | Latency threshold for warnings |
| `latency_critical_threshold_ms` | 2000 | Latency threshold for critical alerts |

## API Endpoints

### Dashboard Stats

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/stats` | GET | Dashboard overview statistics |
| `/api/dashboard/messages-by-time` | GET | Message count by day |
| `/api/dashboard/messages-by-role` | GET | Message count by role |

### Messages

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/messages` | GET | Paginated messages with filters |
| `/api/dashboard/messages/{id}` | GET | Single message by ID |
| `/api/dashboard/messages/{id}` | DELETE | Delete a message |

### Threads

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/threads` | GET | List conversation threads |
| `/api/dashboard/threads/{id}/messages` | GET | Messages for a thread |

### Performance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/performance` | GET | Performance metrics (range: 1h, 24h, 7d) |
| `/api/dashboard/tools` | GET | Tool usage statistics |
| `/api/dashboard/tool-calls` | GET | Tool call history |
| `/api/dashboard/users/performance` | GET | Per-user performance |

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/sessions/stats` | GET | Session statistics (total, active, avg duration) |
| `/api/dashboard/sessions` | GET | Paginated list of sessions |
| `/api/dashboard/sessions/{id}` | GET | Session detail with threads |
| `/api/dashboard/sessions/{id}/messages` | GET | Messages for a session |
| `/api/dashboard/sessions/{id}` | DELETE | Delete a session and all its messages |

### Logs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/logs` | GET | System logs (level filter, limit) |
| `/api/dashboard/logs` | DELETE | Clear all logs |

### Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/config` | GET | Current configuration (all settings) |
| `/api/dashboard/config` | PUT | Update configuration |
| `/api/dashboard/config/status` | GET | System status (enabled, server, model) |
| `/api/dashboard/config/restart` | POST | Signal server restart |
| `/api/dashboard/config/env` | GET | Environment variables (masked) |
| `/api/dashboard/config/env` | PUT | Update environment variables |
| `/api/dashboard/config/database/test` | POST | Test database connection |
| `/api/dashboard/config/database/vacuum` | POST | Vacuum SQLite database |
| `/api/dashboard/config/database/clear-metrics` | POST | Clear old metrics (by days) |
| `/api/dashboard/config/database/reset` | POST | Reset database (destructive) |

### Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/models` | GET | Available models |
| `/api/dashboard/models/active` | GET | Currently active model |
| `/api/dashboard/models/active` | POST | Set active model |

## Recording Metrics

### Using the Metrics Utility

```python
from mindcore.utils import (
    timed,
    measure_time,
    record_llm_call,
    record_tool_call,
    record_enrichment,
    record_retrieval
)

# Decorator for timing functions
@timed("my_operation")
def my_function():
    # Function body
    pass

# Context manager for timing
with measure_time("llm_call") as timer:
    result = llm.generate(prompt)
print(f"Took {timer.elapsed_ms}ms")

# Manual recording
record_llm_call(
    model="gpt-4o-mini",
    latency_ms=450,
    prompt_tokens=150,
    completion_tokens=200,
    success=True
)

record_tool_call(
    tool_name="search",
    execution_time_ms=120,
    success=True,
    message_id="msg_123"
)
```

### Using the MetricsManager Directly

```python
from mindcore.core import MetricsManager

metrics = MetricsManager(db_path="mindcore.db")

# Record LLM call
metrics.record_llm_call(
    model="gpt-4o-mini",
    total_time_ms=450,
    prompt_tokens=150,
    completion_tokens=200,
    success=True,
    user_id="user_123",
    thread_id="thread_456"
)

# Record tool call
metrics.record_tool_call(
    tool_name="web_search",
    execution_time_ms=320,
    success=True,
    message_id="msg_789"
)

# Get performance stats
stats = metrics.get_performance_stats(time_range="24h")
print(f"Avg response time: {stats['avg_response_time_ms']}ms")
print(f"P95 latency: {stats['p95_response_time_ms']}ms")

# Get tool stats
tool_stats = metrics.get_tool_stats(time_range="24h")
print(f"Tool success rate: {tool_stats['success_rate']}%")

# Manage settings
metrics.set_setting("auto_refresh_interval", 60)
interval = metrics.get_setting("auto_refresh_interval")

# Cleanup old data
deleted = metrics.cleanup_old_metrics(days=30)
```

## Theme

The dashboard uses a light blue-purple color scheme:

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#7C6EE4` | Buttons, links, accents |
| Primary Light | `#9D91ED` | Hover states |
| Primary Dark | `#5B4FCF` | Active states |
| Background | `#F0EEFF` | Page background |

## Tech Stack

- **Frontend**: Vue 3, Vite, Ant Design Vue 4.1.2
- **Charts**: Chart.js + vue-chartjs
- **State**: Pinia
- **HTTP**: Axios
- **Backend**: FastAPI, SQLite/PostgreSQL

## Development

### Project Structure

```
dashboard/
├── src/
│   ├── api/           # API client
│   ├── assets/        # Styles
│   ├── stores/        # Pinia stores
│   ├── views/         # Page components
│   │   ├── Overview.vue
│   │   ├── Performance.vue
│   │   ├── Messages.vue
│   │   ├── Threads.vue
│   │   ├── ThreadDetail.vue
│   │   ├── Sessions.vue
│   │   ├── Logs.vue
│   │   ├── Configuration.vue
│   │   └── Models.vue
│   ├── App.vue
│   ├── main.js
│   └── router.js
├── package.json
└── vite.config.js
```

### Build for Production

```bash
cd dashboard
npm run build
```

Output will be in `dashboard/dist/`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDCORE_DB_PATH` | `mindcore.db` | SQLite database path |
| `MINDCORE_API_PORT` | `8000` | API server port |
| `MINDCORE_LOG_LEVEL` | `INFO` | Logging level |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT
