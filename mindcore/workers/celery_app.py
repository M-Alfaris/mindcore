"""Celery application configuration for distributed Mindcore workers.

Provides optional Celery integration for scaling enrichment
across multiple workers/machines.

Requires: pip install celery[redis] or celery[rabbitmq]
"""

from dataclasses import dataclass, field
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


def is_celery_available() -> bool:
    """Check if Celery is installed."""
    try:
        import celery  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class CeleryConfig:
    """Configuration for Celery application.

    Example:
        config = CeleryConfig(
            broker_url="redis://localhost:6379/0",
            result_backend="redis://localhost:6379/1",
            task_default_queue="mindcore",
        )
    """

    # Broker configuration
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str | None = "redis://localhost:6379/1"

    # Queue configuration
    task_default_queue: str = "mindcore"
    task_queues: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Worker configuration
    worker_prefetch_multiplier: int = 4
    worker_concurrency: int | None = None  # None = CPU count
    worker_max_tasks_per_child: int = 1000  # Restart workers periodically

    # Task configuration
    task_acks_late: bool = True  # Ack after task completes (safer)
    task_reject_on_worker_lost: bool = True
    task_time_limit: int = 300  # 5 minutes
    task_soft_time_limit: int = 240  # 4 minutes (warning)

    # Retry configuration
    task_default_retry_delay: int = 60  # 1 minute
    task_max_retries: int = 3

    # Serialization
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = field(default_factory=lambda: ["json"])

    # Result expiration
    result_expires: int = 3600  # 1 hour

    # Monitoring
    worker_send_task_events: bool = True
    task_send_sent_event: bool = True

    def to_celery_config(self) -> dict[str, Any]:
        """Convert to Celery configuration dictionary."""
        config = {
            "broker_url": self.broker_url,
            "result_backend": self.result_backend,
            "task_default_queue": self.task_default_queue,
            "worker_prefetch_multiplier": self.worker_prefetch_multiplier,
            "worker_max_tasks_per_child": self.worker_max_tasks_per_child,
            "task_acks_late": self.task_acks_late,
            "task_reject_on_worker_lost": self.task_reject_on_worker_lost,
            "task_time_limit": self.task_time_limit,
            "task_soft_time_limit": self.task_soft_time_limit,
            "task_default_retry_delay": self.task_default_retry_delay,
            "task_serializer": self.task_serializer,
            "result_serializer": self.result_serializer,
            "accept_content": self.accept_content,
            "result_expires": self.result_expires,
            "worker_send_task_events": self.worker_send_task_events,
            "task_send_sent_event": self.task_send_sent_event,
        }

        if self.worker_concurrency:
            config["worker_concurrency"] = self.worker_concurrency

        if self.task_queues:
            config["task_queues"] = self.task_queues

        return config


def create_celery_app(
    config: CeleryConfig | None = None,
    app_name: str = "mindcore",
) -> Any:
    """Create and configure a Celery application.

    Args:
        config: Celery configuration (uses defaults if not provided)
        app_name: Name for the Celery application

    Returns:
        Configured Celery application

    Raises:
        ImportError: If Celery is not installed

    Example:
        from mindcore.workers import CeleryConfig, create_celery_app

        config = CeleryConfig(
            broker_url="redis://localhost:6379/0",
        )

        app = create_celery_app(config)

        # Run worker: celery -A mindcore.workers.celery_app worker -l info
    """
    if not is_celery_available():
        raise ImportError(
            "Celery is not installed. Install with: pip install celery[redis]"
        )

    from celery import Celery

    config = config or CeleryConfig()

    # Create Celery app
    app = Celery(app_name)

    # Apply configuration
    app.config_from_object(config.to_celery_config())

    logger.info(
        f"Celery app created: name={app_name}, "
        f"broker={config.broker_url}, "
        f"queue={config.task_default_queue}"
    )

    return app


# Default app instance (can be imported by Celery worker)
_default_app: Any = None


def get_celery_app(config: CeleryConfig | None = None) -> Any:
    """Get or create the default Celery application.

    Args:
        config: Optional configuration (only used if app doesn't exist)

    Returns:
        Celery application instance
    """
    global _default_app

    if _default_app is None:
        _default_app = create_celery_app(config)

    return _default_app


def reset_celery_app() -> None:
    """Reset the default Celery application (for testing)."""
    global _default_app
    _default_app = None
