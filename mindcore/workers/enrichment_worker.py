"""Distributed enrichment worker using Celery.

Provides scalable message enrichment across multiple workers.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from mindcore.core.schemas import EnrichmentSource, MessageMetadata
from mindcore.utils.logger import get_logger


if TYPE_CHECKING:
    from celery import Celery

    from mindcore import MindcoreClient

logger = get_logger(__name__)


@dataclass
class EnrichmentTask:
    """Represents an enrichment task."""

    message_id: str
    user_id: str
    thread_id: str
    session_id: str
    role: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: int = 5  # 1-10, lower is higher priority
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "role": self.role,
            "text": self.text,
            "created_at": self.created_at,
            "priority": self.priority,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnrichmentTask":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            thread_id=data["thread_id"],
            session_id=data["session_id"],
            role=data["role"],
            text=data["text"],
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            priority=data.get("priority", 5),
            retry_count=data.get("retry_count", 0),
        )


class DistributedEnrichmentWorker:
    """Distributed enrichment worker using Celery.

    Replaces the built-in ThreadPoolExecutor with a Celery-based
    distributed task queue for horizontal scaling.

    Example:
        from mindcore.workers import (
            CeleryConfig,
            create_celery_app,
            DistributedEnrichmentWorker,
        )

        config = CeleryConfig(broker_url="redis://localhost:6379/0")
        celery_app = create_celery_app(config)

        worker = DistributedEnrichmentWorker(
            celery_app=celery_app,
            client=mindcore_client,
        )

        # Register tasks with Celery
        worker.register_tasks()

        # Submit enrichment task
        worker.submit_enrichment({
            "message_id": "msg123",
            "user_id": "user456",
            "thread_id": "thread789",
            "session_id": "session123",
            "role": "user",
            "text": "Hello, how are you?",
        })
    """

    def __init__(
        self,
        celery_app: "Celery",
        client: "MindcoreClient",
        queue_name: str = "mindcore.enrichment",
        priority_queue_name: str = "mindcore.enrichment.priority",
    ):
        """Initialize distributed worker.

        Args:
            celery_app: Celery application instance
            client: MindcoreClient for enrichment
            queue_name: Name of the enrichment queue
            priority_queue_name: Name of the priority queue
        """
        self.celery_app = celery_app
        self.client = client
        self.queue_name = queue_name
        self.priority_queue_name = priority_queue_name

        self._tasks_registered = False
        self._enrich_task = None

    def register_tasks(self) -> None:
        """Register Celery tasks.

        Call this once before submitting tasks.
        """
        if self._tasks_registered:
            return

        @self.celery_app.task(
            name="mindcore.enrich_message",
            bind=True,
            max_retries=3,
            default_retry_delay=60,
            acks_late=True,
        )
        def enrich_message_task(self_task, task_data: dict[str, Any]) -> dict[str, Any]:
            """Celery task for message enrichment."""
            task = EnrichmentTask.from_dict(task_data)
            start_time = time.time()

            try:
                # Get trivial detector
                trivial_detector = self.client._trivial_detector
                trivial_result = trivial_detector.detect(task.text)

                if trivial_result.is_trivial:
                    # Auto-enrich without LLM
                    enriched_message = trivial_detector.auto_enrich(
                        text=task.text,
                        user_id=task.user_id,
                        thread_id=task.thread_id,
                        session_id=task.session_id,
                        role=task.role,
                        message_id=task.message_id,
                    )
                    source = EnrichmentSource.TRIVIAL.value
                else:
                    # Full LLM enrichment
                    enriched_message = self.client._metadata_agent.process({
                        "message_id": task.message_id,
                        "user_id": task.user_id,
                        "thread_id": task.thread_id,
                        "session_id": task.session_id,
                        "role": task.role,
                        "text": task.text,
                    })
                    source = EnrichmentSource.LLM.value

                # Update database
                self.client.db.update_message_metadata(
                    message_id=task.message_id,
                    metadata=enriched_message.metadata,
                )

                # Notify cache invalidation
                self.client._cache_invalidation.notify_enrichment_complete(enriched_message)

                # Learn from metadata
                if not trivial_result.is_trivial:
                    self.client._adaptive_learner.process_message_metadata(
                        user_id=enriched_message.user_id,
                        metadata=enriched_message.metadata,
                    )

                latency_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Celery enrichment complete: message_id={task.message_id}, "
                    f"source={source}, latency={latency_ms:.0f}ms"
                )

                return {
                    "success": True,
                    "message_id": task.message_id,
                    "source": source,
                    "confidence": enriched_message.metadata.confidence_score,
                    "latency_ms": latency_ms,
                }

            except Exception as e:
                logger.exception(f"Celery enrichment failed: {e}")

                # Update with failure metadata
                failed_metadata = MessageMetadata(
                    enrichment_failed=True,
                    enrichment_error=str(e),
                    confidence_score=0.0,
                    enrichment_source=EnrichmentSource.FALLBACK.value,
                )
                self.client.db.update_message_metadata(task.message_id, failed_metadata)

                # Retry if appropriate
                if task.retry_count < 3:
                    raise self_task.retry(exc=e)

                return {
                    "success": False,
                    "message_id": task.message_id,
                    "error": str(e),
                }

        self._enrich_task = enrich_message_task
        self._tasks_registered = True

        logger.info("Celery enrichment tasks registered")

    def submit_enrichment(
        self,
        task_data: dict[str, Any],
        priority: int = 5,
        countdown: int | None = None,
    ) -> Any:
        """Submit an enrichment task.

        Args:
            task_data: Message data to enrich
            priority: Task priority (1-10, lower is higher)
            countdown: Delay in seconds before execution

        Returns:
            Celery AsyncResult
        """
        if not self._tasks_registered:
            self.register_tasks()

        # Create task
        task = EnrichmentTask(
            message_id=task_data.get("message_id", ""),
            user_id=task_data["user_id"],
            thread_id=task_data["thread_id"],
            session_id=task_data["session_id"],
            role=task_data["role"],
            text=task_data["text"],
            priority=priority,
        )

        # Choose queue based on priority
        queue = self.priority_queue_name if priority <= 3 else self.queue_name

        # Submit task
        result = self._enrich_task.apply_async(
            args=[task.to_dict()],
            queue=queue,
            priority=priority,
            countdown=countdown,
        )

        logger.debug(f"Enrichment task submitted: {task.message_id}, queue={queue}")
        return result

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
            List of Celery AsyncResults
        """
        results = []
        for task_data in tasks:
            result = self.submit_enrichment(task_data, priority=priority)
            results.append(result)

        logger.info(f"Submitted batch of {len(tasks)} enrichment tasks")
        return results

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Queue depth and worker information
        """
        try:
            inspect = self.celery_app.control.inspect()

            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
            scheduled = inspect.scheduled() or {}

            total_active = sum(len(tasks) for tasks in active.values())
            total_reserved = sum(len(tasks) for tasks in reserved.values())
            total_scheduled = sum(len(tasks) for tasks in scheduled.values())

            return {
                "workers": list(active.keys()),
                "active_tasks": total_active,
                "reserved_tasks": total_reserved,
                "scheduled_tasks": total_scheduled,
                "total_pending": total_reserved + total_scheduled,
            }
        except Exception as e:
            logger.warning(f"Failed to get queue stats: {e}")
            return {"error": str(e)}

    def purge_queue(self, queue_name: str | None = None) -> int:
        """Purge all tasks from a queue.

        Args:
            queue_name: Queue to purge (defaults to enrichment queue)

        Returns:
            Number of tasks purged
        """
        queue = queue_name or self.queue_name

        try:
            purged = self.celery_app.control.purge()
            logger.warning(f"Purged {purged} tasks from queue")
            return purged
        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return 0
