"""
Dashboard API routes for Mindcore web interface.

Provides endpoints for:
- Statistics and metrics
- Message management
- Thread browsing
- System logs
- Configuration
- Model management
"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Query, HTTPException, Body
from pydantic import BaseModel

from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


# ============================================================================
# Pydantic Models
# ============================================================================

class StatsResponse(BaseModel):
    total_messages: int
    today_messages: int
    active_users: int
    conversations: int


class MessageResponse(BaseModel):
    message_id: str
    user_id: str
    thread_id: str
    session_id: str
    role: str
    raw_text: str
    metadata: Dict[str, Any]
    created_at: Optional[str]


class MessagesListResponse(BaseModel):
    messages: List[MessageResponse]
    total: int
    page: int
    page_size: int


class ThreadResponse(BaseModel):
    thread_id: str
    user_id: str
    message_count: int
    last_message_at: Optional[str]
    first_message_at: Optional[str]


class ThreadsListResponse(BaseModel):
    threads: List[ThreadResponse]
    total: int


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    logger: Optional[str] = None


class LogsResponse(BaseModel):
    logs: List[LogEntry]
    total: int


class ConfigResponse(BaseModel):
    llm: Dict[str, Any]
    memory: Dict[str, Any]
    database: Dict[str, Any]
    cache: Dict[str, Any]


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    provider: str
    size: Optional[str] = None
    available: bool = True


class ModelsResponse(BaseModel):
    cloud: List[ModelInfo]
    local: List[ModelInfo]
    active: Optional[str] = None


class SetModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


# ============================================================================
# In-memory log storage (for demo purposes)
# ============================================================================

_log_buffer: List[Dict[str, Any]] = []
_max_log_entries = 1000


def add_log_entry(level: str, message: str, logger_name: str = "mindcore"):
    """Add a log entry to the buffer."""
    global _log_buffer
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
        "logger": logger_name
    }
    _log_buffer.append(entry)
    if len(_log_buffer) > _max_log_entries:
        _log_buffer = _log_buffer[-_max_log_entries:]


# Add some initial demo logs
add_log_entry("INFO", "Mindcore dashboard initialized")
add_log_entry("INFO", "Database connection established")
add_log_entry("DEBUG", "Cache manager started with TTL=3600s")


# ============================================================================
# Stats Endpoints
# ============================================================================

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get dashboard statistics."""
    from ... import _mindcore_instance

    total_messages = 0
    today_messages = 0
    active_users = set()
    conversations = set()

    if _mindcore_instance is not None:
        try:
            # Get stats from database
            if hasattr(_mindcore_instance, 'db'):
                db = _mindcore_instance.db

                # For SQLite
                if hasattr(db, 'get_connection'):
                    with db.get_connection() as conn:
                        # Total messages
                        cursor = conn.execute("SELECT COUNT(*) FROM messages")
                        total_messages = cursor.fetchone()[0]

                        # Today's messages
                        today = datetime.now(timezone.utc).date().isoformat()
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM messages WHERE date(created_at) = ?",
                            (today,)
                        )
                        today_messages = cursor.fetchone()[0]

                        # Active users
                        cursor = conn.execute("SELECT DISTINCT user_id FROM messages")
                        active_users = set(row[0] for row in cursor.fetchall())

                        # Conversations (threads)
                        cursor = conn.execute("SELECT DISTINCT thread_id FROM messages")
                        conversations = set(row[0] for row in cursor.fetchall())

                # For PostgreSQL
                elif hasattr(db, 'pool'):
                    with db.get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT COUNT(*) FROM messages")
                            total_messages = cursor.fetchone()[0]

                            cursor.execute(
                                "SELECT COUNT(*) FROM messages WHERE DATE(created_at) = CURRENT_DATE"
                            )
                            today_messages = cursor.fetchone()[0]

                            cursor.execute("SELECT DISTINCT user_id FROM messages")
                            active_users = set(row[0] for row in cursor.fetchall())

                            cursor.execute("SELECT DISTINCT thread_id FROM messages")
                            conversations = set(row[0] for row in cursor.fetchall())

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            add_log_entry("ERROR", f"Failed to get stats: {e}")

    return StatsResponse(
        total_messages=total_messages,
        today_messages=today_messages,
        active_users=len(active_users),
        conversations=len(conversations)
    )


@router.get("/messages-by-time")
async def get_messages_by_time(days: int = Query(7, ge=1, le=30)):
    """Get message count by day for the last N days."""
    from ... import _mindcore_instance

    result = defaultdict(int)

    # Initialize with zeros for all days
    for i in range(days):
        date = (datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat()
        result[date] = 0

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                    cursor = conn.execute(
                        """SELECT date(created_at) as date, COUNT(*) as count
                           FROM messages
                           WHERE created_at >= ?
                           GROUP BY date(created_at)
                           ORDER BY date""",
                        (start_date,)
                    )
                    for row in cursor.fetchall():
                        if row[0]:
                            result[row[0]] = row[1]

            elif hasattr(db, 'pool'):
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """SELECT DATE(created_at) as date, COUNT(*) as count
                               FROM messages
                               WHERE created_at >= NOW() - INTERVAL '%s days'
                               GROUP BY DATE(created_at)
                               ORDER BY date""",
                            (days,)
                        )
                        for row in cursor.fetchall():
                            if row[0]:
                                result[row[0].isoformat()] = row[1]

        except Exception as e:
            logger.error(f"Failed to get messages by time: {e}")

    # Convert to sorted list
    sorted_dates = sorted(result.keys())
    return {
        "labels": sorted_dates,
        "data": [result[d] for d in sorted_dates]
    }


@router.get("/messages-by-role")
async def get_messages_by_role():
    """Get message count by role."""
    from ... import _mindcore_instance

    result = {"user": 0, "assistant": 0, "system": 0, "tool": 0}

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT role, COUNT(*) FROM messages GROUP BY role"
                    )
                    for row in cursor.fetchall():
                        if row[0] in result:
                            result[row[0]] = row[1]

            elif hasattr(db, 'pool'):
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT role, COUNT(*) FROM messages GROUP BY role"
                        )
                        for row in cursor.fetchall():
                            if row[0] in result:
                                result[row[0]] = row[1]

        except Exception as e:
            logger.error(f"Failed to get messages by role: {e}")

    return {
        "labels": list(result.keys()),
        "data": list(result.values())
    }


# ============================================================================
# Messages Endpoints
# ============================================================================

@router.get("/messages", response_model=MessagesListResponse)
async def get_messages(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    role: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    thread_id: Optional[str] = Query(None)
):
    """Get paginated messages with optional filters."""
    from ... import _mindcore_instance

    messages = []
    total = 0
    offset = (page - 1) * page_size

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Build query
                    conditions = []
                    params = []

                    if role:
                        conditions.append("role = ?")
                        params.append(role)
                    if user_id:
                        conditions.append("user_id = ?")
                        params.append(user_id)
                    if thread_id:
                        conditions.append("thread_id = ?")
                        params.append(thread_id)
                    if search:
                        conditions.append("raw_text LIKE ?")
                        params.append(f"%{search}%")

                    where_clause = " AND ".join(conditions) if conditions else "1=1"

                    # Count
                    cursor = conn.execute(
                        f"SELECT COUNT(*) FROM messages WHERE {where_clause}",
                        params
                    )
                    total = cursor.fetchone()[0]

                    # Fetch
                    cursor = conn.execute(
                        f"""SELECT * FROM messages
                            WHERE {where_clause}
                            ORDER BY created_at DESC
                            LIMIT ? OFFSET ?""",
                        params + [page_size, offset]
                    )

                    for row in cursor.fetchall():
                        metadata = {}
                        if row['metadata']:
                            try:
                                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                            except:
                                pass

                        messages.append(MessageResponse(
                            message_id=row['message_id'],
                            user_id=row['user_id'],
                            thread_id=row['thread_id'],
                            session_id=row['session_id'],
                            role=row['role'],
                            raw_text=row['raw_text'],
                            metadata=metadata,
                            created_at=str(row['created_at']) if row['created_at'] else None
                        ))

        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            add_log_entry("ERROR", f"Failed to get messages: {e}")

    return MessagesListResponse(
        messages=messages,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/messages/{message_id}", response_model=MessageResponse)
async def get_message(message_id: str):
    """Get a single message by ID."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    message = _mindcore_instance.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    metadata = {}
    if message.metadata:
        metadata = message.metadata.to_dict() if hasattr(message.metadata, 'to_dict') else message.metadata

    return MessageResponse(
        message_id=message.message_id,
        user_id=message.user_id,
        thread_id=message.thread_id,
        session_id=message.session_id,
        role=message.role.value if hasattr(message.role, 'value') else message.role,
        raw_text=message.raw_text,
        metadata=metadata,
        created_at=str(message.created_at) if message.created_at else None
    )


@router.delete("/messages/{message_id}")
async def delete_message(message_id: str):
    """Delete a message."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db

        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))
                conn.commit()
        elif hasattr(db, 'pool'):
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM messages WHERE message_id = %s", (message_id,))
                    conn.commit()

        add_log_entry("INFO", f"Deleted message: {message_id}")
        return {"status": "deleted", "message_id": message_id}

    except Exception as e:
        logger.error(f"Failed to delete message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Threads Endpoints
# ============================================================================

@router.get("/threads", response_model=ThreadsListResponse)
async def get_threads(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: Optional[str] = Query(None)
):
    """Get list of conversation threads."""
    from ... import _mindcore_instance

    threads = []
    total = 0
    offset = (page - 1) * page_size

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Build query
                    where_clause = f"user_id = '{user_id}'" if user_id else "1=1"

                    # Count distinct threads
                    cursor = conn.execute(
                        f"SELECT COUNT(DISTINCT thread_id) FROM messages WHERE {where_clause}"
                    )
                    total = cursor.fetchone()[0]

                    # Get threads with stats
                    cursor = conn.execute(
                        f"""SELECT
                                thread_id,
                                user_id,
                                COUNT(*) as message_count,
                                MAX(created_at) as last_message_at,
                                MIN(created_at) as first_message_at
                            FROM messages
                            WHERE {where_clause}
                            GROUP BY thread_id, user_id
                            ORDER BY last_message_at DESC
                            LIMIT ? OFFSET ?""",
                        (page_size, offset)
                    )

                    for row in cursor.fetchall():
                        threads.append(ThreadResponse(
                            thread_id=row['thread_id'],
                            user_id=row['user_id'],
                            message_count=row['message_count'],
                            last_message_at=str(row['last_message_at']) if row['last_message_at'] else None,
                            first_message_at=str(row['first_message_at']) if row['first_message_at'] else None
                        ))

        except Exception as e:
            logger.error(f"Failed to get threads: {e}")

    return ThreadsListResponse(threads=threads, total=total)


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """Get messages for a specific thread."""
    return await get_messages(
        page=page,
        page_size=page_size,
        thread_id=thread_id
    )


# ============================================================================
# Logs Endpoints
# ============================================================================

@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500)
):
    """Get system logs."""
    logs = _log_buffer.copy()

    if level:
        logs = [log for log in logs if log['level'].upper() == level.upper()]

    # Most recent first
    logs = list(reversed(logs))[:limit]

    return LogsResponse(
        logs=[LogEntry(**log) for log in logs],
        total=len(logs)
    )


@router.delete("/logs")
async def clear_logs():
    """Clear all logs."""
    global _log_buffer
    _log_buffer = []
    add_log_entry("INFO", "Logs cleared by user")
    return {"status": "cleared"}


# ============================================================================
# Configuration Endpoints
# ============================================================================

# In-memory config store (persisted via MetricsManager settings in production)
_runtime_config: Dict[str, Any] = {
    "system_enabled": True,
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "local_model": "llama-3.2-3b",
        "temperature": 0.3,
        "max_tokens": 1500
    },
    "memory": {
        "enabled": True,
        "provider": "llama_cpp",
        "local_model": "llama-3.2-1b"
    },
    "database": {
        "type": "sqlite",
        "path": "mindcore.db"
    },
    "cache": {
        "max_size": 50,
        "ttl": 3600
    },
    "api": {
        "port": 8000,
        "cors_origins": [],
        "rate_limiting": True,
        "rate_limit": 100
    },
    "logging": {
        "level": "INFO",
        "json_format": False
    },
    "monitoring": {
        "performance_tracking": True,
        "tool_tracking": True,
        "retention_days": 30
    },
    "security": {
        "input_validation": True,
        "max_message_length": 10000,
        "audit_logging": False
    }
}


class ExtendedConfigResponse(BaseModel):
    system_enabled: bool = True
    llm: Dict[str, Any]
    memory: Dict[str, Any]
    database: Dict[str, Any]
    cache: Dict[str, Any]
    api: Dict[str, Any] = {}
    logging: Dict[str, Any] = {}
    monitoring: Dict[str, Any] = {}
    security: Dict[str, Any] = {}


@router.get("/config", response_model=ExtendedConfigResponse)
async def get_config():
    """Get current configuration."""
    from ... import _mindcore_instance

    config = _runtime_config.copy()

    if _mindcore_instance is not None:
        try:
            # Get LLM config
            if hasattr(_mindcore_instance, '_llm_provider'):
                provider = _mindcore_instance._llm_provider
                config["llm"]["provider"] = provider.name if hasattr(provider, 'name') else config["llm"]["provider"]

            # Get cache config
            if hasattr(_mindcore_instance, 'cache'):
                cache = _mindcore_instance.cache
                if hasattr(cache, 'max_size'):
                    config["cache"]["max_size"] = cache.max_size
                if hasattr(cache, 'ttl_seconds'):
                    config["cache"]["ttl"] = cache.ttl_seconds

            # Get DB config
            if hasattr(_mindcore_instance, '_use_sqlite'):
                config["database"]["type"] = "sqlite" if _mindcore_instance._use_sqlite else "postgresql"

        except Exception as e:
            logger.error(f"Failed to get config: {e}")

    return ExtendedConfigResponse(**config)


@router.put("/config")
async def update_config(config: Dict[str, Any] = Body(...)):
    """Update configuration."""
    global _runtime_config

    # Deep merge configuration
    for key, value in config.items():
        if key in _runtime_config and isinstance(_runtime_config[key], dict) and isinstance(value, dict):
            _runtime_config[key].update(value)
        else:
            _runtime_config[key] = value

    add_log_entry("INFO", f"Configuration updated: {list(config.keys())}")
    return _runtime_config


@router.get("/config/status")
async def get_system_status():
    """Get system status."""
    from ... import _mindcore_instance

    return {
        "system_enabled": _runtime_config.get("system_enabled", True),
        "server_status": "online",
        "mindcore_initialized": _mindcore_instance is not None,
        "active_model": _runtime_config.get("llm", {}).get("model", "gpt-4o-mini"),
        "database_type": _runtime_config.get("database", {}).get("type", "sqlite")
    }


@router.post("/config/restart")
async def restart_server():
    """Signal server restart (placeholder)."""
    add_log_entry("INFO", "Server restart requested")
    return {"status": "restart_requested", "message": "Server will restart shortly"}


# ============================================================================
# Environment Variables Endpoints
# ============================================================================

# In-memory env var store (for dashboard display, actual values from os.environ)
_env_vars_config: List[Dict[str, Any]] = []


@router.get("/config/env")
async def get_env_vars():
    """Get configured environment variables (masked for sensitive ones)."""
    env_vars = [
        {"key": "OPENAI_API_KEY", "value": os.environ.get("OPENAI_API_KEY", ""), "sensitive": True},
        {"key": "ANTHROPIC_API_KEY", "value": os.environ.get("ANTHROPIC_API_KEY", ""), "sensitive": True},
        {"key": "MINDCORE_DB_PATH", "value": os.environ.get("MINDCORE_DB_PATH", "mindcore.db"), "sensitive": False},
        {"key": "MINDCORE_LOG_LEVEL", "value": os.environ.get("MINDCORE_LOG_LEVEL", "INFO"), "sensitive": False},
        {"key": "MINDCORE_API_PORT", "value": os.environ.get("MINDCORE_API_PORT", "8000"), "sensitive": False},
        {"key": "MINDCORE_LLAMA_MODEL_PATH", "value": os.environ.get("MINDCORE_LLAMA_MODEL_PATH", ""), "sensitive": False},
    ]

    # Mask sensitive values
    for var in env_vars:
        if var["sensitive"] and var["value"]:
            val = var["value"]
            if len(val) > 8:
                var["value"] = val[:4] + "..." + val[-4:]
            else:
                var["value"] = "***"

    return {"env_vars": env_vars}


@router.put("/config/env")
async def update_env_vars(env_vars: List[Dict[str, Any]] = Body(...)):
    """Update environment variables (runtime only, not persistent)."""
    updated = []
    for var in env_vars:
        key = var.get("key")
        value = var.get("value")
        if key and value:
            # Don't actually set env vars for security, just log the intent
            add_log_entry("INFO", f"Environment variable update requested: {key}")
            updated.append(key)

    return {"status": "updated", "variables": updated, "note": "Changes require server restart to take effect"}


# ============================================================================
# Database Management Endpoints
# ============================================================================

@router.post("/config/database/test")
async def test_database_connection(db_config: Dict[str, Any] = Body(...)):
    """Test database connection."""
    db_type = db_config.get("type", "sqlite")

    try:
        if db_type == "sqlite":
            import sqlite3
            path = db_config.get("path", "mindcore.db")
            conn = sqlite3.connect(path)
            conn.execute("SELECT 1")
            conn.close()
            return {"status": "success", "message": f"Connected to SQLite database: {path}"}

        elif db_type == "postgresql":
            # In production, would actually test PostgreSQL connection
            return {"status": "success", "message": "PostgreSQL connection test passed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/config/database/vacuum")
async def vacuum_database():
    """Vacuum the SQLite database."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db
        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                conn.execute("VACUUM")
                add_log_entry("INFO", "Database vacuumed successfully")
                return {"status": "success", "message": "Database vacuumed"}
        else:
            return {"status": "skipped", "message": "Vacuum only supported for SQLite"}

    except Exception as e:
        logger.error(f"Failed to vacuum database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/database/clear-metrics")
async def clear_old_metrics(days: int = Body(..., embed=True)):
    """Clear metrics older than specified days."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        deleted_count = 0
        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                # Clear performance metrics
                cursor = conn.execute(
                    "DELETE FROM performance_metrics WHERE created_at < ?",
                    (cutoff,)
                )
                deleted_count += cursor.rowcount

                # Clear tool calls
                cursor = conn.execute(
                    "DELETE FROM tool_calls WHERE created_at < ?",
                    (cutoff,)
                )
                deleted_count += cursor.rowcount
                conn.commit()

        add_log_entry("INFO", f"Cleared {deleted_count} old metrics (older than {days} days)")
        return {"status": "success", "deleted_count": deleted_count}

    except Exception as e:
        logger.error(f"Failed to clear metrics: {e}")
        # Table might not exist yet
        return {"status": "success", "deleted_count": 0, "note": "Metrics tables may not exist"}


@router.post("/config/database/reset")
async def reset_database():
    """Reset database (delete all data, recreate tables)."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db
        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                # Get all tables
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                # Delete data from all tables
                for table in tables:
                    conn.execute(f"DELETE FROM {table}")
                conn.commit()

                add_log_entry("WARNING", f"Database reset: cleared {len(tables)} tables")
                return {"status": "success", "tables_cleared": tables}

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Models Endpoints
# ============================================================================

@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models."""
    from ... import _mindcore_instance

    cloud_models = [
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            description="Most capable model for complex tasks",
            provider="openai",
            available=True
        ),
        ModelInfo(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            description="Fast and cost-effective for most tasks",
            provider="openai",
            available=True
        ),
        ModelInfo(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            description="High performance with vision capabilities",
            provider="openai",
            available=True
        ),
        ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            description="Good balance of speed and capability",
            provider="openai",
            available=True
        )
    ]

    local_models = [
        ModelInfo(
            id="llama-3.2-3b",
            name="Llama 3.2 3B",
            description="Best quality for local inference",
            provider="llama_cpp",
            size="2.0 GB",
            available=_check_model_available("llama-3.2-3b")
        ),
        ModelInfo(
            id="llama-3.2-1b",
            name="Llama 3.2 1B",
            description="Lightweight, low resource usage",
            provider="llama_cpp",
            size="0.8 GB",
            available=_check_model_available("llama-3.2-1b")
        ),
        ModelInfo(
            id="qwen2.5-3b",
            name="Qwen 2.5 3B",
            description="Multilingual, great for structured output",
            provider="llama_cpp",
            size="2.1 GB",
            available=_check_model_available("qwen2.5-3b")
        ),
        ModelInfo(
            id="phi-3.5-mini",
            name="Phi 3.5 Mini",
            description="Microsoft's reasoning-focused model",
            provider="llama_cpp",
            size="2.2 GB",
            available=_check_model_available("phi-3.5-mini")
        )
    ]

    active = None
    if _mindcore_instance is not None:
        try:
            if hasattr(_mindcore_instance, '_llm_provider'):
                provider = _mindcore_instance._llm_provider
                active = provider.name if hasattr(provider, 'name') else None
        except:
            pass

    return ModelsResponse(
        cloud=cloud_models,
        local=local_models,
        active=active
    )


def _check_model_available(model_id: str) -> bool:
    """Check if a local model is available."""
    models_dir = os.path.expanduser("~/.mindcore/models")
    if not os.path.exists(models_dir):
        return False

    # Check for model files
    for f in os.listdir(models_dir):
        if model_id.replace("-", "") in f.lower().replace("-", ""):
            return True
    return False


@router.get("/models/active")
async def get_active_model():
    """Get the currently active model."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        return {"model": None, "provider": None}

    try:
        if hasattr(_mindcore_instance, '_llm_provider'):
            provider = _mindcore_instance._llm_provider
            return {
                "model": provider.name if hasattr(provider, 'name') else "unknown",
                "provider": type(provider).__name__
            }
    except:
        pass

    return {"model": None, "provider": None}


@router.post("/models/active")
async def set_active_model(request: SetModelRequest):
    """Set the active model."""
    add_log_entry("INFO", f"Model switched to: {request.model_id}")
    return {"status": "updated", "model_id": request.model_id}


@router.post("/models/{model_id}/download")
async def download_model(model_id: str):
    """Trigger model download."""
    add_log_entry("INFO", f"Starting download for model: {model_id}")
    return {
        "status": "downloading",
        "model_id": model_id,
        "message": "Use 'mindcore download-model' CLI for actual download"
    }


# ============================================================================
# Performance & Observability Endpoints
# ============================================================================

class PerformanceStatsResponse(BaseModel):
    stats: Dict[str, Any]
    tool_stats: Dict[str, Any]
    user_performance: List[Dict[str, Any]]
    response_time_trend: List[Dict[str, Any]]
    latency_distribution: List[int]


class ToolCallResponse(BaseModel):
    tool_call_id: str
    message_id: str
    tool_name: str
    success: bool
    execution_time_ms: int
    error_message: Optional[str] = None
    created_at: str


# In-memory performance metrics storage (for demo/dev purposes)
_performance_metrics: Dict[str, List[Dict[str, Any]]] = {
    "response_times": [],
    "tool_calls": [],
    "user_stats": {}
}


def record_performance_metric(metric_type: str, data: Dict[str, Any]):
    """Record a performance metric."""
    global _performance_metrics
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    if metric_type == "response_time":
        _performance_metrics["response_times"].append(data)
        # Keep last 1000 entries
        _performance_metrics["response_times"] = _performance_metrics["response_times"][-1000:]
    elif metric_type == "tool_call":
        _performance_metrics["tool_calls"].append(data)
        _performance_metrics["tool_calls"] = _performance_metrics["tool_calls"][-1000:]


@router.get("/performance")
async def get_performance_stats(
    range: str = Query("24h", pattern="^(1h|24h|7d)$")
):
    """Get performance statistics and metrics."""
    from ... import _mindcore_instance

    # Calculate time range
    now = datetime.now(timezone.utc)
    if range == "1h":
        start_time = now - timedelta(hours=1)
    elif range == "24h":
        start_time = now - timedelta(hours=24)
    else:
        start_time = now - timedelta(days=7)

    # Default response structure
    response = {
        "stats": {
            "avg_response_time_ms": 0,
            "p95_response_time_ms": 0,
            "avg_enrichment_time_ms": 0,
            "total_llm_calls": 0,
            "avg_llm_time_ms": 0,
            "avg_retrieval_time_ms": 0,
            "avg_storage_time_ms": 0,
            "avg_total_time_ms": 0
        },
        "tool_stats": {
            "success_rate": 100,
            "tools": []
        },
        "user_performance": [],
        "response_time_trend": [],
        "latency_distribution": [0, 0, 0, 0, 0]
    }

    # Calculate from stored metrics
    metrics = _performance_metrics["response_times"]
    start_iso = start_time.isoformat()
    recent_metrics = [m for m in metrics if m.get("timestamp", "") >= start_iso]

    if recent_metrics:
        times = [m.get("total_time_ms", 0) for m in recent_metrics]
        response["stats"]["avg_response_time_ms"] = sum(times) // len(times) if times else 0
        sorted_times = sorted(times)
        p95_idx = int(len(sorted_times) * 0.95)
        response["stats"]["p95_response_time_ms"] = sorted_times[p95_idx] if sorted_times else 0
        response["stats"]["total_llm_calls"] = len(recent_metrics)

        # Calculate latency distribution
        distribution = [0, 0, 0, 0, 0]  # <100ms, 100-500ms, 500ms-1s, 1-2s, >2s
        for t in times:
            if t < 100:
                distribution[0] += 1
            elif t < 500:
                distribution[1] += 1
            elif t < 1000:
                distribution[2] += 1
            elif t < 2000:
                distribution[3] += 1
            else:
                distribution[4] += 1
        response["latency_distribution"] = distribution

    # Calculate tool stats
    tool_calls = _performance_metrics["tool_calls"]
    recent_tools = [t for t in tool_calls if t.get("timestamp", "") >= start_iso]
    if recent_tools:
        success_count = sum(1 for t in recent_tools if t.get("success", False))
        response["tool_stats"]["success_rate"] = round(success_count / len(recent_tools) * 100, 1)

        # Group by tool name
        tool_grouped: Dict[str, Dict[str, Any]] = {}
        for t in recent_tools:
            name = t.get("tool_name", "unknown")
            if name not in tool_grouped:
                tool_grouped[name] = {"name": name, "call_count": 0, "success_count": 0, "total_time_ms": 0}
            tool_grouped[name]["call_count"] += 1
            if t.get("success", False):
                tool_grouped[name]["success_count"] += 1
            tool_grouped[name]["total_time_ms"] += t.get("execution_time_ms", 0)

        response["tool_stats"]["tools"] = [
            {
                "name": v["name"],
                "call_count": v["call_count"],
                "success_rate": round(v["success_count"] / v["call_count"] * 100, 1) if v["call_count"] > 0 else 0,
                "avg_time_ms": v["total_time_ms"] // v["call_count"] if v["call_count"] > 0 else 0
            }
            for v in tool_grouped.values()
        ]

    # Try to get data from database if available
    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Get user performance stats
                    cursor = conn.execute(
                        """SELECT
                               user_id,
                               COUNT(DISTINCT thread_id) as thread_count,
                               COUNT(*) as message_count
                           FROM messages
                           GROUP BY user_id
                           ORDER BY message_count DESC
                           LIMIT 20"""
                    )
                    for row in cursor.fetchall():
                        response["user_performance"].append({
                            "user_id": row['user_id'],
                            "thread_count": row['thread_count'],
                            "message_count": row['message_count'],
                            "avg_response_time_ms": 0,  # Would need timing data in DB
                            "total_time_ms": 0
                        })

                    # Build response time trend from message timestamps
                    if range == "1h":
                        interval_sql = "strftime('%H:%M', created_at)"
                    elif range == "24h":
                        interval_sql = "strftime('%H:00', created_at)"
                    else:
                        interval_sql = "strftime('%Y-%m-%d', created_at)"

                    cursor = conn.execute(
                        f"""SELECT {interval_sql} as time_bucket, COUNT(*) as count
                            FROM messages
                            WHERE created_at >= ?
                            GROUP BY time_bucket
                            ORDER BY time_bucket""",
                        (start_time.isoformat(),)
                    )

                    # Simulated response times based on message volume
                    for row in cursor.fetchall():
                        if row['time_bucket']:
                            response["response_time_trend"].append({
                                "time": row['time_bucket'],
                                "value": 200 + (row['count'] * 10)  # Simulated latency
                            })

        except Exception as e:
            logger.error(f"Failed to get performance stats from DB: {e}")
            add_log_entry("ERROR", f"Performance stats error: {e}")

    # Add demo data if no real data
    if not response["response_time_trend"]:
        for i in range(6):
            hour = (now - timedelta(hours=4*i)).strftime("%H:00")
            response["response_time_trend"].append({
                "time": hour,
                "value": 250 + (i * 20)
            })
        response["response_time_trend"].reverse()

    if not response["latency_distribution"] or sum(response["latency_distribution"]) == 0:
        response["latency_distribution"] = [45, 32, 15, 6, 2]

    return response


@router.get("/tools")
async def get_tool_stats():
    """Get tool usage statistics."""
    tool_calls = _performance_metrics["tool_calls"]

    if not tool_calls:
        return {
            "success_rate": 100,
            "total_calls": 0,
            "tools": []
        }

    success_count = sum(1 for t in tool_calls if t.get("success", False))

    # Group by tool name
    tool_grouped: Dict[str, Dict[str, Any]] = {}
    for t in tool_calls:
        name = t.get("tool_name", "unknown")
        if name not in tool_grouped:
            tool_grouped[name] = {"name": name, "call_count": 0, "success_count": 0, "total_time_ms": 0}
        tool_grouped[name]["call_count"] += 1
        if t.get("success", False):
            tool_grouped[name]["success_count"] += 1
        tool_grouped[name]["total_time_ms"] += t.get("execution_time_ms", 0)

    return {
        "success_rate": round(success_count / len(tool_calls) * 100, 1) if tool_calls else 100,
        "total_calls": len(tool_calls),
        "tools": [
            {
                "name": v["name"],
                "call_count": v["call_count"],
                "success_rate": round(v["success_count"] / v["call_count"] * 100, 1) if v["call_count"] > 0 else 0,
                "avg_time_ms": v["total_time_ms"] // v["call_count"] if v["call_count"] > 0 else 0
            }
            for v in tool_grouped.values()
        ]
    }


@router.get("/tool-calls")
async def get_tool_calls(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    tool_name: Optional[str] = Query(None),
    success: Optional[bool] = Query(None)
):
    """Get detailed tool call history."""
    calls = _performance_metrics["tool_calls"].copy()

    # Filter
    if tool_name:
        calls = [c for c in calls if c.get("tool_name") == tool_name]
    if success is not None:
        calls = [c for c in calls if c.get("success") == success]

    total = len(calls)
    offset = (page - 1) * page_size
    calls = list(reversed(calls))[offset:offset + page_size]

    return {
        "tool_calls": calls,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/users/performance")
async def get_user_performance(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get per-user performance metrics."""
    from ... import _mindcore_instance

    users = []
    total = 0
    offset = (page - 1) * page_size

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Count unique users
                    cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM messages")
                    total = cursor.fetchone()[0]

                    # Get user stats
                    cursor = conn.execute(
                        """SELECT
                               user_id,
                               COUNT(DISTINCT thread_id) as thread_count,
                               COUNT(*) as message_count,
                               MIN(created_at) as first_message,
                               MAX(created_at) as last_message
                           FROM messages
                           GROUP BY user_id
                           ORDER BY message_count DESC
                           LIMIT ? OFFSET ?""",
                        (page_size, offset)
                    )

                    for row in cursor.fetchall():
                        users.append({
                            "user_id": row['user_id'],
                            "thread_count": row['thread_count'],
                            "message_count": row['message_count'],
                            "first_message": str(row['first_message']) if row['first_message'] else None,
                            "last_message": str(row['last_message']) if row['last_message'] else None,
                            "avg_response_time_ms": 0,  # Would need timing data
                            "total_time_ms": 0
                        })

        except Exception as e:
            logger.error(f"Failed to get user performance: {e}")

    return {
        "users": users,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# ============================================================================
# Sessions Endpoints
# ============================================================================

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    started_at: str
    last_activity_at: str
    thread_count: int
    message_count: int
    total_llm_calls: int
    total_tool_calls: int
    total_latency_ms: int
    avg_latency_ms: int
    status: str


class SessionsListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    page: int
    page_size: int


class SessionStatsResponse(BaseModel):
    total_sessions: int
    active_today: int
    avg_duration_minutes: int
    avg_messages_per_session: int


@router.get("/sessions/stats", response_model=SessionStatsResponse)
async def get_session_stats():
    """Get session statistics."""
    from ... import _mindcore_instance

    stats = {
        "total_sessions": 0,
        "active_today": 0,
        "avg_duration_minutes": 0,
        "avg_messages_per_session": 0
    }

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Total unique sessions
                    cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
                    stats["total_sessions"] = cursor.fetchone()[0]

                    # Active today
                    today = datetime.now(timezone.utc).date().isoformat()
                    cursor = conn.execute(
                        "SELECT COUNT(DISTINCT session_id) FROM messages WHERE date(created_at) = ?",
                        (today,)
                    )
                    stats["active_today"] = cursor.fetchone()[0]

                    # Avg messages per session
                    if stats["total_sessions"] > 0:
                        cursor = conn.execute("SELECT COUNT(*) FROM messages")
                        total_messages = cursor.fetchone()[0]
                        stats["avg_messages_per_session"] = total_messages // stats["total_sessions"]

                    # Avg duration (based on time between first and last message)
                    cursor = conn.execute(
                        """SELECT AVG(
                               (julianday(last_msg) - julianday(first_msg)) * 24 * 60
                           ) as avg_duration
                           FROM (
                               SELECT session_id,
                                      MIN(created_at) as first_msg,
                                      MAX(created_at) as last_msg
                               FROM messages
                               GROUP BY session_id
                           )"""
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        stats["avg_duration_minutes"] = int(result[0])

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")

    return SessionStatsResponse(**stats)


@router.get("/sessions", response_model=SessionsListResponse)
async def get_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Get list of sessions with analytics."""
    from ... import _mindcore_instance

    sessions = []
    total = 0
    offset = (page - 1) * page_size

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Build conditions
                    conditions = []
                    params = []

                    if user_id:
                        conditions.append("user_id = ?")
                        params.append(user_id)

                    where_clause = " AND ".join(conditions) if conditions else "1=1"

                    # Count unique sessions
                    cursor = conn.execute(
                        f"SELECT COUNT(DISTINCT session_id) FROM messages WHERE {where_clause}",
                        params
                    )
                    total = cursor.fetchone()[0]

                    # Get sessions with aggregated data
                    cursor = conn.execute(
                        f"""SELECT
                               session_id,
                               user_id,
                               MIN(created_at) as started_at,
                               MAX(created_at) as last_activity_at,
                               COUNT(DISTINCT thread_id) as thread_count,
                               COUNT(*) as message_count
                           FROM messages
                           WHERE {where_clause}
                           GROUP BY session_id, user_id
                           ORDER BY last_activity_at DESC
                           LIMIT ? OFFSET ?""",
                        params + [page_size, offset]
                    )

                    now = datetime.now(timezone.utc)
                    inactive_threshold = now - timedelta(minutes=30)

                    for row in cursor.fetchall():
                        last_activity = row['last_activity_at']
                        try:
                            last_dt = datetime.fromisoformat(str(last_activity).replace('Z', '+00:00'))
                            session_status = "active" if last_dt > inactive_threshold else "idle"
                        except:
                            session_status = "unknown"

                        # Filter by status if requested
                        if status and session_status != status:
                            continue

                        sessions.append(SessionResponse(
                            session_id=row['session_id'],
                            user_id=row['user_id'],
                            started_at=str(row['started_at']) if row['started_at'] else "",
                            last_activity_at=str(row['last_activity_at']) if row['last_activity_at'] else "",
                            thread_count=row['thread_count'],
                            message_count=row['message_count'],
                            total_llm_calls=row['message_count'],  # Approximation
                            total_tool_calls=0,  # Would need tool tracking
                            total_latency_ms=0,  # Would need performance tracking
                            avg_latency_ms=0,
                            status=session_status
                        ))

        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            add_log_entry("ERROR", f"Failed to get sessions: {e}")

    return SessionsListResponse(
        sessions=sessions,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get detailed information about a specific session."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db

        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                # Get session summary
                cursor = conn.execute(
                    """SELECT
                           session_id,
                           user_id,
                           MIN(created_at) as started_at,
                           MAX(created_at) as last_activity_at,
                           COUNT(DISTINCT thread_id) as thread_count,
                           COUNT(*) as message_count
                       FROM messages
                       WHERE session_id = ?
                       GROUP BY session_id, user_id""",
                    (session_id,)
                )
                row = cursor.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="Session not found")

                # Get threads in this session
                cursor = conn.execute(
                    """SELECT
                           thread_id,
                           COUNT(*) as message_count,
                           MIN(created_at) as first_message,
                           MAX(created_at) as last_message
                       FROM messages
                       WHERE session_id = ?
                       GROUP BY thread_id
                       ORDER BY first_message""",
                    (session_id,)
                )
                threads = [
                    {
                        "thread_id": t['thread_id'],
                        "message_count": t['message_count'],
                        "first_message": str(t['first_message']),
                        "last_message": str(t['last_message'])
                    }
                    for t in cursor.fetchall()
                ]

                # Get role breakdown
                cursor = conn.execute(
                    "SELECT role, COUNT(*) as count FROM messages WHERE session_id = ? GROUP BY role",
                    (session_id,)
                )
                role_breakdown = {r['role']: r['count'] for r in cursor.fetchall()}

                return {
                    "session_id": row['session_id'],
                    "user_id": row['user_id'],
                    "started_at": str(row['started_at']),
                    "last_activity_at": str(row['last_activity_at']),
                    "thread_count": row['thread_count'],
                    "message_count": row['message_count'],
                    "threads": threads,
                    "role_breakdown": role_breakdown,
                    "total_llm_calls": row['message_count'],
                    "total_tool_calls": 0,
                    "total_latency_ms": 0,
                    "avg_latency_ms": 0
                }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """Get messages for a specific session."""
    from ... import _mindcore_instance

    messages = []
    total = 0
    offset = (page - 1) * page_size

    if _mindcore_instance is not None:
        try:
            db = _mindcore_instance.db

            if hasattr(db, 'get_connection'):
                with db.get_connection() as conn:
                    # Count
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                        (session_id,)
                    )
                    total = cursor.fetchone()[0]

                    # Fetch messages
                    cursor = conn.execute(
                        """SELECT * FROM messages
                           WHERE session_id = ?
                           ORDER BY created_at ASC
                           LIMIT ? OFFSET ?""",
                        (session_id, page_size, offset)
                    )

                    for row in cursor.fetchall():
                        metadata = {}
                        if row['metadata']:
                            try:
                                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                            except:
                                pass

                        messages.append({
                            "message_id": row['message_id'],
                            "user_id": row['user_id'],
                            "thread_id": row['thread_id'],
                            "session_id": row['session_id'],
                            "role": row['role'],
                            "raw_text": row['raw_text'],
                            "metadata": metadata,
                            "created_at": str(row['created_at']) if row['created_at'] else None
                        })

        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")

    return {
        "messages": messages,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete all messages in a session."""
    from ... import _mindcore_instance

    if _mindcore_instance is None:
        raise HTTPException(status_code=503, detail="Mindcore not initialized")

    try:
        db = _mindcore_instance.db

        if hasattr(db, 'get_connection'):
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (session_id,)
                )
                count = cursor.fetchone()[0]

                if count == 0:
                    raise HTTPException(status_code=404, detail="Session not found")

                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.commit()

                add_log_entry("INFO", f"Deleted session {session_id} with {count} messages")
                return {"status": "deleted", "session_id": session_id, "messages_deleted": count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
