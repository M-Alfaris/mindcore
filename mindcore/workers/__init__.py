"""Distributed workers for Mindcore.

Optional package for scaling enrichment and context processing
using Celery for distributed task execution.

Installation:
    pip install mindcore[celery]

Usage:
    # Configure Celery
    from mindcore.workers import CeleryConfig, create_celery_app

    config = CeleryConfig(
        broker_url="redis://localhost:6379/0",
        result_backend="redis://localhost:6379/1",
    )

    celery_app = create_celery_app(config)

    # Use distributed enrichment
    from mindcore.workers import DistributedEnrichmentWorker

    worker = DistributedEnrichmentWorker(celery_app, mindcore_client)
    worker.start()

    # Submit tasks
    worker.submit_enrichment(message_dict)
"""

from .celery_app import CeleryConfig, create_celery_app, is_celery_available
from .enrichment_worker import DistributedEnrichmentWorker, EnrichmentTask
from .worker_manager import WorkerBackend, WorkerConfig, WorkerManager


__all__ = [
    "CeleryConfig",
    "DistributedEnrichmentWorker",
    "EnrichmentTask",
    "WorkerBackend",
    "WorkerConfig",
    "WorkerManager",
    "create_celery_app",
    "is_celery_available",
]
