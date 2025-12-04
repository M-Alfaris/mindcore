"""Logging utility for Mindcore framework.

Uses structlog for structured, cloud-ready logging with JSON output support.
Provides configurable logging with fine-grained control over log levels
for different components.

Configuration:
    Environment variables:
    - MINDCORE_LOG_LEVEL: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - MINDCORE_JSON_LOGS: Enable JSON output ("1", "true", "yes")
    - MINDCORE_LOG_CONTEXT: Enable context agent logs ("1", "true", "yes")
    - MINDCORE_LOG_ENRICHMENT: Enable enrichment logs ("1", "true", "yes")
    - MINDCORE_LOG_TOOLS: Enable tool calling logs ("1", "true", "yes")

    Or use LogConfig programmatically:
    >>> from mindcore.utils.logger import configure_logging, LogConfig
    >>> config = LogConfig(
    ...     level="DEBUG",
    ...     json_logs=False,
    ...     enable_context_logs=True,
    ...     enable_enrichment_logs=True,
    ...     enable_tool_logs=True
    ... )
    >>> configure_logging(config=config)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog


class LogCategory(Enum):
    """Categories of logs that can be enabled/disabled."""

    CONTEXT = "context"  # SmartContextAgent, context assembly
    ENRICHMENT = "enrichment"  # Message enrichment, metadata extraction
    TOOLS = "tools"  # Tool calling, tool execution
    DATABASE = "database"  # Database operations
    CACHE = "cache"  # Cache operations
    LLM = "llm"  # LLM provider calls
    API = "api"  # API requests/responses
    GENERAL = "general"  # General framework logs


@dataclass
class LogConfig:
    """Configuration for Mindcore logging.

    Allows fine-grained control over which log categories are enabled
    and at what level.

    Example:
        >>> config = LogConfig(
        ...     level="INFO",
        ...     json_logs=False,
        ...     enable_context_logs=True,  # Enable SmartContextAgent logs
        ...     enable_enrichment_logs=False,  # Disable enrichment spam
        ...     enable_tool_logs=True,
        ...     errors_always_logged=True  # Always log errors regardless of category
        ... )
    """

    # Global log level
    level: str = "INFO"

    # Output format
    json_logs: bool = False

    # Category-specific enables (default: errors only for verbose categories)
    enable_context_logs: bool = False  # SmartContextAgent detailed logs
    enable_enrichment_logs: bool = False  # Enrichment agent logs
    enable_tool_logs: bool = False  # Tool calling logs
    enable_database_logs: bool = False  # Database operation logs
    enable_cache_logs: bool = False  # Cache operation logs
    enable_llm_logs: bool = True  # LLM call logs (useful for debugging)
    enable_api_logs: bool = True  # API request logs

    # Always log errors regardless of category settings
    errors_always_logged: bool = True

    # Always log warnings regardless of category settings
    warnings_always_logged: bool = True

    # Minimum message count before context warnings (for early conversation handling)
    min_messages_for_context_warning: int = 3

    # Custom log levels per category (overrides global level)
    category_levels: dict = field(default_factory=dict)

    def is_category_enabled(self, category: LogCategory) -> bool:
        """Check if a log category is enabled."""
        category_map = {
            LogCategory.CONTEXT: self.enable_context_logs,
            LogCategory.ENRICHMENT: self.enable_enrichment_logs,
            LogCategory.TOOLS: self.enable_tool_logs,
            LogCategory.DATABASE: self.enable_database_logs,
            LogCategory.CACHE: self.enable_cache_logs,
            LogCategory.LLM: self.enable_llm_logs,
            LogCategory.API: self.enable_api_logs,
            LogCategory.GENERAL: True,  # Always enabled
        }
        return category_map.get(category, True)

    def get_category_level(self, category: LogCategory) -> str:
        """Get the log level for a specific category."""
        return self.category_levels.get(category.value, self.level)

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create LogConfig from environment variables."""

        def parse_bool(val: str | None) -> bool:
            return val is not None and val.lower() in ("1", "true", "yes")

        return cls(
            level=os.getenv("MINDCORE_LOG_LEVEL", "INFO"),
            json_logs=parse_bool(os.getenv("MINDCORE_JSON_LOGS")),
            enable_context_logs=parse_bool(os.getenv("MINDCORE_LOG_CONTEXT")),
            enable_enrichment_logs=parse_bool(os.getenv("MINDCORE_LOG_ENRICHMENT")),
            enable_tool_logs=parse_bool(os.getenv("MINDCORE_LOG_TOOLS")),
            enable_database_logs=parse_bool(os.getenv("MINDCORE_LOG_DATABASE")),
            enable_cache_logs=parse_bool(os.getenv("MINDCORE_LOG_CACHE")),
            enable_llm_logs=parse_bool(os.getenv("MINDCORE_LOG_LLM", "1")),  # Default enabled
            enable_api_logs=parse_bool(os.getenv("MINDCORE_LOG_API", "1")),  # Default enabled
            errors_always_logged=True,
            warnings_always_logged=True,
        )


# Global config instance
_log_config: LogConfig = LogConfig.from_env()


def _make_category_filter(config: LogConfig):
    """Create a structlog processor that filters logs based on category config.

    This processor checks if a log should be emitted based on:
    1. The log level (errors/warnings always pass if configured)
    2. The log category (if category is disabled, log is dropped)
    """

    def category_filter(logger, method_name, event_dict):
        # Get log level
        log_level = event_dict.get("level", "info").lower()

        # Always allow errors if configured
        if config.errors_always_logged and log_level in ("error", "critical", "exception"):
            return event_dict

        # Always allow warnings if configured
        if config.warnings_always_logged and log_level == "warning":
            return event_dict

        # Check category
        category_str = event_dict.get("category", "general")
        try:
            category = LogCategory(category_str)
        except ValueError:
            category = LogCategory.GENERAL

        if not config.is_category_enabled(category):
            raise structlog.DropEvent

        return event_dict

    return category_filter


def _configure_structlog(config: LogConfig) -> None:
    """Configure structlog with appropriate processors.

    Args:
        config: LogConfig instance with logging settings.
    """
    # Shared processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _make_category_filter(config),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if config.json_logs:
        # Production: JSON output for log aggregation
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        processors = [*shared_processors, structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Auto-configure on module load
_configure_structlog(_log_config)


def get_logger(
    name: str,
    level: str | None = None,
    format_string: str | None = None,  # Ignored, kept for backward compatibility
    category: LogCategory | None = None,
) -> Any:
    """Get a configured structlog logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
              Note: Level is set globally via MINDCORE_LOG_LEVEL env var.
        format_string: Ignored (for backward compatibility).
        category: Optional log category for filtering.

    Returns:
        Configured structlog logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing message", user_id="123", message_id="msg_456")
        >>> logger.error("Failed to process", error="connection timeout", retries=3)

        >>> # With category for filtering
        >>> logger = get_logger(__name__, category=LogCategory.CONTEXT)
        >>> logger.debug("Context assembly started", query="test")  # Only if context logs enabled
    """
    base_logger = structlog.get_logger(name)

    # Bind category if specified
    if category is not None:
        return base_logger.bind(category=category.value)

    return base_logger


def get_config() -> LogConfig:
    """Get the current logging configuration."""
    return _log_config


def configure_logging(
    level: str = "INFO",
    format_string: str | None = None,  # Ignored
    json_logs: bool | None = None,
    config: LogConfig | None = None,
) -> None:
    """Configure global logging settings.

    Args:
        level: Logging level.
        format_string: Ignored (for backward compatibility).
        json_logs: If True, output JSON logs. If None, uses MINDCORE_JSON_LOGS env var.
        config: Optional LogConfig instance for full control.

    Example:
        >>> # Simple configuration
        >>> configure_logging(level="DEBUG")
        >>>
        >>> # Production (JSON for log aggregation)
        >>> configure_logging(level="INFO", json_logs=True)
        >>>
        >>> # Full control with LogConfig
        >>> config = LogConfig(
        ...     level="DEBUG",
        ...     enable_context_logs=True,
        ...     enable_tool_logs=True
        ... )
        >>> configure_logging(config=config)
    """
    global _log_config

    if config is not None:
        _log_config = config
    else:
        _log_config = LogConfig(
            level=level,
            json_logs=json_logs if json_logs is not None else _log_config.json_logs,
            enable_context_logs=_log_config.enable_context_logs,
            enable_enrichment_logs=_log_config.enable_enrichment_logs,
            enable_tool_logs=_log_config.enable_tool_logs,
        )

    _configure_structlog(_log_config)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log messages.

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


# Convenience function for creating category-specific loggers
def get_context_logger(name: str) -> Any:
    """Get a logger for context-related operations."""
    return get_logger(name, category=LogCategory.CONTEXT)


def get_enrichment_logger(name: str) -> Any:
    """Get a logger for enrichment-related operations."""
    return get_logger(name, category=LogCategory.ENRICHMENT)


def get_tool_logger(name: str) -> Any:
    """Get a logger for tool-related operations."""
    return get_logger(name, category=LogCategory.TOOLS)
