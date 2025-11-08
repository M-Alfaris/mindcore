"""
Logging utility for Mindcore framework.
"""
import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger

    # Set level
    log_level = getattr(logging, level.upper()) if level else logging.INFO
    logger.setLevel(log_level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Create formatter
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string or default_format)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def configure_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Configure global logging settings.

    Args:
        level: Logging level.
        format_string: Custom format string.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=log_level,
        format=format_string or default_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
