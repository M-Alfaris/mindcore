"""Worker manager for Mindcore.

Provides a unified interface for managing workers regardless
of backend (thread pool, Celery, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from mindcore.utils.logger import get_logger


if TYPE_CHECKING:
    from mindcore import MindcoreClient

logger = get_logger(__name__)


class WorkerBackend(str, Enum):
    """Available worker backends."""

    THREAD = "thread"  # Built-in ThreadPoolExecutor (default)
    CELERY = "celery"  # Distributed Celery workers


@dataclass
class WorkerConfig:
    """Configuration for workers.

    Example:
        # Thread pool (default)
        config = WorkerConfig(
            backend=WorkerBackend.THREAD,
            workers=4,
        )

        # Celery
        config = WorkerConfig(
            backend=WorkerBackend.CELERY,
            celery_broker="redis://localhost:6379/0",
            celery_result_backend="redis://localhost:6379/1",
        )
    """

    backend: WorkerBackend = WorkerBackend.THREAD

    # Thread pool settings
    workers: int = 4
    max_workers: int = 16

    # Celery settings
    celery_broker: str | None = None
    celery_result_backend: str | None = None
    celery_queue: str = "mindcore.enrichment"

    # Common settings
    task_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 60

    # Queue settings
    queue_path: str | None = None  # For persistent queue
    max_queue_size: int = 10000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "workers": self.workers,
            "max_workers": self.max_workers,
            "celery_broker": self.celery_broker,
            "celery_queue": self.celery_queue,
            "task_timeout": self.task_timeout,
            "max_retries": self.max_retries,
        }


class WorkerManager:
    """Unified worker manager.

    Manages enrichment workers regardless of backend.
    Provides a consistent interface for:
    - Submitting tasks
    - Monitoring queue depth
    - Getting worker health
    - Scaling workers

    Example:
        from mindcore.workers import WorkerManager, WorkerConfig, WorkerBackend

        # Use thread pool (default, good for development)
        manager = WorkerManager(
            client=mindcore_client,
            config=WorkerConfig(backend=WorkerBackend.THREAD, workers=4),
        )

        # Or use Celery (production, distributed)
        manager = WorkerManager(
            client=mindcore_client,
            config=WorkerConfig(
                backend=WorkerBackend.CELERY,
                celery_broker="redis://localhost:6379/0",
            ),
        )

        # Submit task
        manager.submit_enrichment(message_dict)

        # Get status
        status = manager.get_status()
    """

    def __init__(
        self,
        client: "MindcoreClient",
        config: WorkerConfig | None = None,
    ):
        """Initialize worker manager.

        Args:
            client: MindcoreClient instance
            config: Worker configuration
        """
        self.client = client
        self.config = config or WorkerConfig()

        self._celery_worker = None
        self._initialized = False

        self._setup_backend()

    def _setup_backend(self) -> None:
        """Set up the worker backend."""
        if self.config.backend == WorkerBackend.THREAD:
            # Thread pool is already set up in MindcoreClient
            logger.info(
                f"Using thread pool backend: workers={self.client.enrichment_worker_count}"
            )
            self._initialized = True

        elif self.config.backend == WorkerBackend.CELERY:
            self._setup_celery()

    def _setup_celery(self) -> None:
        """Set up Celery backend."""
        from .celery_app import CeleryConfig, create_celery_app, is_celery_available
        from .enrichment_worker import DistributedEnrichmentWorker

        if not is_celery_available():
            raise ImportError(
                "Celery is not installed. Install with: pip install celery[redis]"
            )

        if not self.config.celery_broker:
            raise ValueError("celery_broker must be configured for Celery backend")

        # Create Celery app
        celery_config = CeleryConfig(
            broker_url=self.config.celery_broker,
            result_backend=self.config.celery_result_backend,
            task_default_queue=self.config.celery_queue,
            task_time_limit=self.config.task_timeout,
            task_max_retries=self.config.max_retries,
            task_default_retry_delay=self.config.retry_delay,
        )

        celery_app = create_celery_app(celery_config)

        # Create distributed worker
        self._celery_worker = DistributedEnrichmentWorker(
            celery_app=celery_app,
            client=self.client,
            queue_name=self.config.celery_queue,
        )

        self._celery_worker.register_tasks()
        self._initialized = True

        logger.info(
            f"Using Celery backend: broker={self.config.celery_broker}, "
            f"queue={self.config.celery_queue}"
        )

    def submit_enrichment(
        self,
        task_data: dict[str, Any],
        priority: int = 5,
    ) -> Any:
        """Submit an enrichment task.

        Args:
            task_data: Message data to enrich
            priority: Task priority (1-10, lower is higher)

        Returns:
            Task result or identifier
        """
        if not self._initialized:
            raise RuntimeError("Worker manager not initialized")

        if self.config.backend == WorkerBackend.THREAD:
            # Use built-in queue
            self.client._enrichment_queue.put(task_data)
            return task_data.get("message_id")

        elif self.config.backend == WorkerBackend.CELERY:
            return self._celery_worker.submit_enrichment(task_data, priority=priority)

    def submit_batch(
        self,
        tasks: list[dict[str, Any]],
        priority: int = 5,
    ) -> list[Any]:
        """Submit multiple enrichment tasks.

        Args:
            tasks: List of message data dictionaries
            priority: Task priority

        Returns:
            List of task results/identifiers
        """
        results = []
        for task_data in tasks:
            result = self.submit_enrichment(task_data, priority=priority)
            results.append(result)

        logger.info(f"Submitted batch of {len(tasks)} tasks")
        return results

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        if self.config.backend == WorkerBackend.THREAD:
            try:
                return self.client._enrichment_queue.qsize()
            except Exception:
                return -1

        elif self.config.backend == WorkerBackend.CELERY:
            stats = self._celery_worker.get_queue_stats()
            return stats.get("total_pending", -1)

        return -1

    def get_status(self) -> dict[str, Any]:
        """Get worker status.

        Returns:
            Status including backend, queue depth, worker count, health
        """
        status = {
            "backend": self.config.backend.value,
            "initialized": self._initialized,
            "queue_depth": self.get_queue_depth(),
        }

        if self.config.backend == WorkerBackend.THREAD:
            status["workers"] = self.client.enrichment_worker_count
            health = self.client.get_worker_health()
            status["health"] = health

        elif self.config.backend == WorkerBackend.CELERY:
            stats = self._celery_worker.get_queue_stats()
            status["celery"] = stats
            status["workers"] = len(stats.get("workers", []))

        return status

    def get_health(self) -> dict[str, Any]:
        """Get worker health check.

        Returns:
            Health status with any issues detected
        """
        queue_depth = self.get_queue_depth()
        issues = []

        if queue_depth > self.config.max_queue_size * 0.8:
            issues.append({
                "type": "high_queue_depth",
                "value": queue_depth,
                "threshold": self.config.max_queue_size * 0.8,
                "message": f"Queue depth is {queue_depth}, approaching limit",
            })

        if self.config.backend == WorkerBackend.THREAD:
            worker_health = self.client.get_worker_health()
            if not worker_health.get("healthy", True):
                issues.extend(worker_health.get("issues", []))

        return {
            "healthy": len(issues) == 0,
            "backend": self.config.backend.value,
            "queue_depth": queue_depth,
            "issues": issues,
        }

    def scale_workers(self, count: int) -> bool:
        """Scale worker count (where supported).

        Args:
            count: New worker count

        Returns:
            True if scaling succeeded
        """
        if self.config.backend == WorkerBackend.THREAD:
            # Can't dynamically scale thread pool easily
            logger.warning(
                "Thread pool cannot be dynamically scaled. "
                "Restart with enrichment_workers parameter."
            )
            return False

        elif self.config.backend == WorkerBackend.CELERY:
            # Celery workers are scaled externally (e.g., via k8s, supervisor)
            logger.info(
                "Celery workers are scaled externally. "
                "Adjust your orchestration configuration."
            )
            return False

        return False

    def close(self) -> None:
        """Clean up worker resources."""
        logger.info(f"WorkerManager closed (backend={self.config.backend.value})")
