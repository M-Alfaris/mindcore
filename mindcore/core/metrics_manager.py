"""Metrics and observability data manager for Mindcore dashboard.

This module provides persistent storage for:
- Performance metrics (LLM response times, enrichment times)
- Tool call tracking (success/failure, execution times)
- Session analytics
- Dashboard settings

Database Schema:
    - performance_metrics: Stores timing data for LLM calls, enrichment, retrieval
    - tool_calls: Tracks tool execution with success/failure status
    - sessions: Aggregated session data
    - dashboard_settings: User dashboard preferences and configuration

Usage:
    from mindcore.core import MetricsManager

    metrics = MetricsManager(db_path="mindcore.db")

    # Record an LLM call
    metrics.record_llm_call(
        model="gpt-4o-mini",
        prompt_tokens=150,
        completion_tokens=200,
        latency_ms=450,
        success=True
    )

    # Record a tool call
    metrics.record_tool_call(
        tool_name="search",
        message_id="msg_123",
        execution_time_ms=120,
        success=True
    )

    # Get performance stats
    stats = metrics.get_performance_stats(time_range="24h")
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


class MetricsManager:
    """Manages metrics and observability data storage.

    Provides persistent storage for dashboard analytics including:
    - LLM performance metrics
    - Tool call tracking
    - Session analytics
    - Dashboard user settings

    Thread-safe implementation using connection-per-thread pattern.
    """

    def __init__(self, db_path: str = "mindcore.db"):
        """Initialize metrics manager.

        Args:
            db_path: Path to SQLite database file.
                    Uses same database as SQLiteManager by default.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize schema
        self.initialize_schema()
        logger.info(f"MetricsManager initialized with database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
        except sqlite3.Error as e:
            logger.exception(f"SQLite error: {e}")
            conn.rollback()
            raise
        finally:
            pass

    def initialize_schema(self) -> None:
        """Create metrics database schema.

        Tables:
            - performance_metrics: LLM calls, enrichment, retrieval timing
            - tool_calls: Tool execution tracking
            - sessions: Session analytics
            - dashboard_settings: User preferences
        """
        schema_sql = """
        -- Performance metrics table
        -- Stores timing data for LLM calls, enrichment, retrieval operations
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,          -- 'llm_call', 'enrichment', 'retrieval', 'storage'
            operation TEXT NOT NULL,            -- Specific operation name
            model TEXT,                         -- LLM model name (for llm_call type)
            prompt_tokens INTEGER DEFAULT 0,    -- Input tokens
            completion_tokens INTEGER DEFAULT 0,-- Output tokens
            total_time_ms INTEGER NOT NULL,     -- Total execution time in milliseconds
            success INTEGER DEFAULT 1,          -- 1=success, 0=failure
            error_message TEXT,                 -- Error message if failed
            user_id TEXT,                       -- Associated user
            thread_id TEXT,                     -- Associated thread
            message_id TEXT,                    -- Associated message
            metadata TEXT DEFAULT '{}',         -- Additional JSON metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_perf_type_created
            ON performance_metrics(metric_type, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_perf_created
            ON performance_metrics(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_perf_model
            ON performance_metrics(model);

        -- Tool calls tracking table
        -- Records every tool invocation with timing and status
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_call_id TEXT UNIQUE,           -- Unique tool call identifier
            tool_name TEXT NOT NULL,            -- Name of the tool
            message_id TEXT,                    -- Associated message ID
            thread_id TEXT,                     -- Associated thread ID
            user_id TEXT,                       -- Associated user ID
            execution_time_ms INTEGER NOT NULL, -- Execution time in milliseconds
            success INTEGER DEFAULT 1,          -- 1=success, 0=failure
            error_message TEXT,                 -- Error message if failed
            input_data TEXT,                    -- Tool input (JSON)
            output_data TEXT,                   -- Tool output (JSON, truncated)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_tool_name_created
            ON tool_calls(tool_name, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_tool_success
            ON tool_calls(success);

        CREATE INDEX IF NOT EXISTS idx_tool_message
            ON tool_calls(message_id);

        -- Sessions table
        -- Aggregated session analytics
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL,
            last_activity_at TIMESTAMP NOT NULL,
            thread_count INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            total_llm_calls INTEGER DEFAULT 0,
            total_tool_calls INTEGER DEFAULT 0,
            total_latency_ms INTEGER DEFAULT 0,
            avg_latency_ms INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_session_user
            ON sessions(user_id);

        CREATE INDEX IF NOT EXISTS idx_session_activity
            ON sessions(last_activity_at DESC);

        -- Dashboard settings table
        -- User preferences and configuration
        CREATE TABLE IF NOT EXISTS dashboard_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            setting_type TEXT DEFAULT 'string', -- 'string', 'number', 'boolean', 'json'
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Insert default settings
        INSERT OR IGNORE INTO dashboard_settings (setting_key, setting_value, setting_type, description)
        VALUES
            ('metrics_retention_days', '30', 'number', 'Days to retain performance metrics'),
            ('auto_refresh_interval', '30', 'number', 'Dashboard auto-refresh interval in seconds'),
            ('max_log_entries', '1000', 'number', 'Maximum log entries to keep in memory'),
            ('enable_performance_tracking', 'true', 'boolean', 'Enable performance metric collection'),
            ('enable_tool_tracking', 'true', 'boolean', 'Enable tool call tracking'),
            ('latency_warning_threshold_ms', '1000', 'number', 'Latency threshold for warnings'),
            ('latency_critical_threshold_ms', '2000', 'number', 'Latency threshold for critical alerts');
        """

        with self._lock:
            conn = self._get_connection()
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Metrics schema initialized")

    # =========================================================================
    # Performance Metrics
    # =========================================================================

    def record_llm_call(
        self,
        model: str,
        total_time_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error_message: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        message_id: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Record an LLM API call metric.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            total_time_ms: Total response time in milliseconds
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            success: Whether the call succeeded
            error_message: Error message if failed
            user_id: Associated user ID
            thread_id: Associated thread ID
            message_id: Associated message ID
            metadata: Additional metadata

        Returns:
            True if recorded successfully
        """
        return self._record_metric(
            metric_type="llm_call",
            operation="generate",
            model=model,
            total_time_ms=total_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error_message=error_message,
            user_id=user_id,
            thread_id=thread_id,
            message_id=message_id,
            metadata=metadata,
        )

    def record_enrichment(
        self,
        message_id: str,
        total_time_ms: int,
        success: bool = True,
        metadata: dict | None = None,
    ) -> bool:
        """Record message enrichment timing."""
        return self._record_metric(
            metric_type="enrichment",
            operation="enrich_message",
            total_time_ms=total_time_ms,
            success=success,
            message_id=message_id,
            metadata=metadata,
        )

    def record_retrieval(
        self,
        thread_id: str,
        total_time_ms: int,
        messages_retrieved: int = 0,
        cache_hit: bool = False,
        success: bool = True,
    ) -> bool:
        """Record context retrieval timing."""
        return self._record_metric(
            metric_type="retrieval",
            operation="get_context",
            total_time_ms=total_time_ms,
            success=success,
            thread_id=thread_id,
            metadata={"messages_retrieved": messages_retrieved, "cache_hit": cache_hit},
        )

    def _record_metric(
        self,
        metric_type: str,
        operation: str,
        total_time_ms: int,
        model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error_message: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        message_id: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Internal method to record a performance metric."""
        insert_sql = """
        INSERT INTO performance_metrics (
            metric_type, operation, model, prompt_tokens, completion_tokens,
            total_time_ms, success, error_message, user_id, thread_id,
            message_id, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                conn.execute(
                    insert_sql,
                    (
                        metric_type,
                        operation,
                        model,
                        prompt_tokens,
                        completion_tokens,
                        total_time_ms,
                        1 if success else 0,
                        error_message,
                        user_id,
                        thread_id,
                        message_id,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.exception(f"Failed to record metric: {e}")
            return False

    # =========================================================================
    # Tool Calls
    # =========================================================================

    def record_tool_call(
        self,
        tool_name: str,
        execution_time_ms: int,
        tool_call_id: str | None = None,
        message_id: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        success: bool = True,
        error_message: str | None = None,
        input_data: dict | None = None,
        output_data: str | None = None,
    ) -> bool:
        """Record a tool call execution.

        Args:
            tool_name: Name of the tool
            execution_time_ms: Execution time in milliseconds
            tool_call_id: Unique identifier for this tool call
            message_id: Associated message ID
            thread_id: Associated thread ID
            user_id: Associated user ID
            success: Whether the call succeeded
            error_message: Error message if failed
            input_data: Tool input parameters
            output_data: Tool output (truncated if too long)

        Returns:
            True if recorded successfully
        """
        import uuid

        tool_call_id = tool_call_id or str(uuid.uuid4())

        insert_sql = """
        INSERT INTO tool_calls (
            tool_call_id, tool_name, message_id, thread_id, user_id,
            execution_time_ms, success, error_message, input_data, output_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                conn.execute(
                    insert_sql,
                    (
                        tool_call_id,
                        tool_name,
                        message_id,
                        thread_id,
                        user_id,
                        execution_time_ms,
                        1 if success else 0,
                        error_message,
                        json.dumps(input_data) if input_data else None,
                        output_data[:1000] if output_data else None,  # Truncate output
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.exception(f"Failed to record tool call: {e}")
            return False

    # =========================================================================
    # Analytics Queries
    # =========================================================================

    def get_performance_stats(
        self, time_range: str = "24h", metric_type: str | None = None
    ) -> dict[str, Any]:
        """Get performance statistics for the dashboard.

        Args:
            time_range: Time range - '1h', '24h', '7d', '30d'
            metric_type: Optional filter by metric type

        Returns:
            Dictionary with performance statistics
        """
        # Calculate time boundary
        now = datetime.now(timezone.utc)
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "24h":
            start_time = now - timedelta(hours=24)
        elif time_range == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(days=30)

        try:
            with self.get_connection() as conn:
                # Base query conditions
                conditions = ["created_at >= ?"]
                params = [start_time.isoformat()]

                if metric_type:
                    conditions.append("metric_type = ?")
                    params.append(metric_type)

                where_clause = " AND ".join(conditions)

                # Get aggregate stats
                stats_sql = f"""
                SELECT
                    COUNT(*) as total_calls,
                    AVG(total_time_ms) as avg_time_ms,
                    MAX(total_time_ms) as max_time_ms,
                    MIN(total_time_ms) as min_time_ms,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
                FROM performance_metrics
                WHERE {where_clause}
                """

                cursor = conn.execute(stats_sql, params)
                row = cursor.fetchone()

                # Calculate P95
                p95_sql = f"""
                SELECT total_time_ms
                FROM performance_metrics
                WHERE {where_clause}
                ORDER BY total_time_ms
                LIMIT 1 OFFSET (
                    SELECT CAST(COUNT(*) * 0.95 AS INTEGER)
                    FROM performance_metrics
                    WHERE {where_clause}
                )
                """
                p95_cursor = conn.execute(p95_sql, params + params)
                p95_row = p95_cursor.fetchone()

                # Get latency distribution
                dist_sql = f"""
                SELECT
                    SUM(CASE WHEN total_time_ms < 100 THEN 1 ELSE 0 END) as bucket_0,
                    SUM(CASE WHEN total_time_ms >= 100 AND total_time_ms < 500 THEN 1 ELSE 0 END) as bucket_1,
                    SUM(CASE WHEN total_time_ms >= 500 AND total_time_ms < 1000 THEN 1 ELSE 0 END) as bucket_2,
                    SUM(CASE WHEN total_time_ms >= 1000 AND total_time_ms < 2000 THEN 1 ELSE 0 END) as bucket_3,
                    SUM(CASE WHEN total_time_ms >= 2000 THEN 1 ELSE 0 END) as bucket_4
                FROM performance_metrics
                WHERE {where_clause}
                """
                dist_cursor = conn.execute(dist_sql, params)
                dist_row = dist_cursor.fetchone()

                return {
                    "total_llm_calls": row["total_calls"] or 0,
                    "avg_response_time_ms": int(row["avg_time_ms"] or 0),
                    "max_response_time_ms": row["max_time_ms"] or 0,
                    "min_response_time_ms": row["min_time_ms"] or 0,
                    "p95_response_time_ms": p95_row["total_time_ms"] if p95_row else 0,
                    "total_prompt_tokens": row["total_prompt_tokens"] or 0,
                    "total_completion_tokens": row["total_completion_tokens"] or 0,
                    "error_count": row["error_count"] or 0,
                    "success_rate": round(
                        (1 - (row["error_count"] or 0) / max(row["total_calls"] or 1, 1)) * 100, 1
                    ),
                    "latency_distribution": [
                        dist_row["bucket_0"] or 0,
                        dist_row["bucket_1"] or 0,
                        dist_row["bucket_2"] or 0,
                        dist_row["bucket_3"] or 0,
                        dist_row["bucket_4"] or 0,
                    ],
                }
        except Exception as e:
            logger.exception(f"Failed to get performance stats: {e}")
            return {
                "total_llm_calls": 0,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "latency_distribution": [0, 0, 0, 0, 0],
            }

    def get_tool_stats(self, time_range: str = "24h") -> dict[str, Any]:
        """Get tool usage statistics.

        Args:
            time_range: Time range - '1h', '24h', '7d'

        Returns:
            Dictionary with tool statistics
        """
        now = datetime.now(timezone.utc)
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "24h":
            start_time = now - timedelta(hours=24)
        else:
            start_time = now - timedelta(days=7)

        try:
            with self.get_connection() as conn:
                # Get per-tool stats
                sql = """
                SELECT
                    tool_name,
                    COUNT(*) as call_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    AVG(execution_time_ms) as avg_time_ms
                FROM tool_calls
                WHERE created_at >= ?
                GROUP BY tool_name
                ORDER BY call_count DESC
                """

                cursor = conn.execute(sql, (start_time.isoformat(),))
                rows = cursor.fetchall()

                tools = []
                total_calls = 0
                total_success = 0

                for row in rows:
                    total_calls += row["call_count"]
                    total_success += row["success_count"]
                    tools.append(
                        {
                            "name": row["tool_name"],
                            "call_count": row["call_count"],
                            "success_rate": round(
                                row["success_count"] / row["call_count"] * 100, 1
                            ),
                            "avg_time_ms": int(row["avg_time_ms"] or 0),
                        }
                    )

                return {
                    "total_calls": total_calls,
                    "success_rate": round(total_success / max(total_calls, 1) * 100, 1),
                    "tools": tools,
                }
        except Exception as e:
            logger.exception(f"Failed to get tool stats: {e}")
            return {"total_calls": 0, "success_rate": 100, "tools": []}

    # =========================================================================
    # Settings
    # =========================================================================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a dashboard setting value."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT setting_value, setting_type FROM dashboard_settings WHERE setting_key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    value = row["setting_value"]
                    setting_type = row["setting_type"]
                    if setting_type == "number":
                        return int(value)
                    if setting_type == "boolean":
                        return value.lower() == "true"
                    if setting_type == "json":
                        return json.loads(value)
                    return value
                return default
        except Exception as e:
            logger.exception(f"Failed to get setting {key}: {e}")
            return default

    def set_setting(self, key: str, value: Any, description: str | None = None) -> bool:
        """Set a dashboard setting value."""
        try:
            with self.get_connection() as conn:
                # Determine type
                if isinstance(value, bool):
                    setting_type = "boolean"
                    str_value = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    setting_type = "number"
                    str_value = str(value)
                elif isinstance(value, dict):
                    setting_type = "json"
                    str_value = json.dumps(value)
                else:
                    setting_type = "string"
                    str_value = str(value)

                sql = """
                INSERT OR REPLACE INTO dashboard_settings
                    (setting_key, setting_value, setting_type, description, updated_at)
                VALUES (?, ?, ?, COALESCE(?, (SELECT description FROM dashboard_settings WHERE setting_key = ?)), CURRENT_TIMESTAMP)
                """
                conn.execute(sql, (key, str_value, setting_type, description, key))
                conn.commit()
                return True
        except Exception as e:
            logger.exception(f"Failed to set setting {key}: {e}")
            return False

    def get_all_settings(self) -> dict[str, Any]:
        """Get all dashboard settings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT setting_key, setting_value, setting_type, description FROM dashboard_settings"
                )
                settings = {}
                for row in cursor.fetchall():
                    value = row["setting_value"]
                    if row["setting_type"] == "number":
                        value = int(value)
                    elif row["setting_type"] == "boolean":
                        value = value.lower() == "true"
                    elif row["setting_type"] == "json":
                        value = json.loads(value)
                    settings[row["setting_key"]] = {
                        "value": value,
                        "type": row["setting_type"],
                        "description": row["description"],
                    }
                return settings
        except Exception as e:
            logger.exception(f"Failed to get all settings: {e}")
            return {}

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_old_metrics(self, days: int | None = None) -> int:
        """Delete metrics older than specified days.

        Args:
            days: Days to retain. Uses setting if not specified.

        Returns:
            Number of records deleted
        """
        if days is None:
            days = self.get_setting("metrics_retention_days", 30)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            with self.get_connection() as conn:
                # Delete old performance metrics
                cursor = conn.execute(
                    "DELETE FROM performance_metrics WHERE created_at < ?", (cutoff.isoformat(),)
                )
                perf_deleted = cursor.rowcount

                # Delete old tool calls
                cursor = conn.execute(
                    "DELETE FROM tool_calls WHERE created_at < ?", (cutoff.isoformat(),)
                )
                tool_deleted = cursor.rowcount

                conn.commit()

                total = perf_deleted + tool_deleted
                logger.info(f"Cleaned up {total} old metric records")
                return total
        except Exception as e:
            logger.exception(f"Failed to cleanup old metrics: {e}")
            return 0

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.info("MetricsManager connection closed")
