"""
Logging utility for Mindcore framework.

Uses structlog for structured, cloud-ready logging with JSON output support.
"""
import logging
import sys
import os
from typing import Optional, Any

import structlog


def _configure_structlog(json_logs: bool = False, log_level: str = "INFO") -> None:
    """
    Configure structlog with appropriate processors.

    Args:
        json_logs: If True, output JSON logs (for cloud/production).
        log_level: Logging level.
    """
    # Shared processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_logs:
        # Production: JSON output for log aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: colored console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Auto-configure on module load
_json_logs = os.getenv("MINDCORE_JSON_LOGS", "").lower() in ("1", "true", "yes")
_log_level = os.getenv("MINDCORE_LOG_LEVEL", "INFO")
_configure_structlog(json_logs=_json_logs, log_level=_log_level)


def get_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None  # Ignored, kept for backward compatibility
) -> Any:
    """
    Get a configured structlog logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
              Note: Level is set globally via MINDCORE_LOG_LEVEL env var.
        format_string: Ignored (for backward compatibility).

    Returns:
        Configured structlog logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing message", user_id="123", message_id="msg_456")
        >>> logger.error("Failed to process", error="connection timeout", retries=3)
    """
    return structlog.get_logger(name)


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,  # Ignored
    json_logs: Optional[bool] = None
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Logging level.
        format_string: Ignored (for backward compatibility).
        json_logs: If True, output JSON logs. If None, uses MINDCORE_JSON_LOGS env var.

    Example:
        >>> # Development (colored console)
        >>> configure_logging(level="DEBUG")
        >>>
        >>> # Production (JSON for log aggregation)
        >>> configure_logging(level="INFO", json_logs=True)
    """
    use_json = json_logs if json_logs is not None else _json_logs
    _configure_structlog(json_logs=use_json, log_level=level)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent log messages.

    Useful for adding request IDs, user IDs, etc. to all logs in a request.

    Args:
        **kwargs: Key-value pairs to add to log context.

    Example:
        >>> bind_context(request_id="req_123", user_id="user_456")
        >>> logger.info("Processing")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
