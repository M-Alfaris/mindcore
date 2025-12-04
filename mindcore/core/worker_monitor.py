"""Worker monitoring for background tasks.

Provides monitoring, metrics, and health checks for background workers
like the enrichment worker.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class WorkerStatus(str, Enum):
    """Worker status states."""

    IDLE = "idle"
    PROCESSING = "processing"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"


@dataclass
class WorkerMetrics:
    """Metrics for a worker."""

    worker_name: str
    status: WorkerStatus = WorkerStatus.IDLE
    started_at: datetime | None = None
    last_activity: datetime | None = None
    processed_count: int = 0
    error_count: int = 0
    trivial_skip_count: int = 0
    avg_processing_time_ms: float = 0.0
    queue_size: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None

    # Rolling window for processing times (last 100)
    _processing_times: list[float] = field(default_factory=list)

    def record_processing(self, duration_ms: float) -> None:
        """Record a successful processing."""
        self.processed_count += 1
        self.last_activity = datetime.now(timezone.utc)

        # Update rolling average
        self._processing_times.append(duration_ms)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)

        self.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)

    def record_trivial_skip(self) -> None:
        """Record a trivial message skip (no LLM call)."""
        self.trivial_skip_count += 1
        self.last_activity = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worker_name": self.worker_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "trivial_skip_count": self.trivial_skip_count,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "queue_size": self.queue_size,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "uptime_seconds": self._get_uptime_seconds(),
            "error_rate": self._get_error_rate(),
            "savings_from_trivial": self._get_trivial_savings(),
        }

    def _get_uptime_seconds(self) -> float | None:
        """Get worker uptime in seconds."""
        if not self.started_at:
            return None
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def _get_error_rate(self) -> float:
        """Get error rate as percentage."""
        total = self.processed_count + self.error_count
        if total == 0:
            return 0.0
        return round(self.error_count / total * 100, 2)

    def _get_trivial_savings(self) -> dict[str, Any]:
        """Get estimated savings from trivial message detection."""
        total = self.processed_count + self.trivial_skip_count
        if total == 0:
            return {"percentage": 0.0, "llm_calls_saved": 0}

        return {
            "percentage": round(self.trivial_skip_count / total * 100, 2),
            "llm_calls_saved": self.trivial_skip_count,
        }


class WorkerMonitor:
    """Monitor for background workers.

    Tracks metrics, health status, and provides callbacks for alerts.

    Example:
        >>> monitor = WorkerMonitor()
        >>> metrics = monitor.register_worker("enrichment")
        >>>
        >>> # In worker loop
        >>> metrics.status = WorkerStatus.PROCESSING
        >>> start = time.time()
        >>> # ... do work ...
        >>> metrics.record_processing((time.time() - start) * 1000)
        >>>
        >>> # Get health check
        >>> health = monitor.get_health()
    """

    def __init__(self):
        """Initialize the worker monitor."""
        self._workers: dict[str, WorkerMetrics] = {}
        self._lock = threading.Lock()
        self._alert_callbacks: list[Callable[[str, WorkerMetrics], None]] = []
        self._health_check_interval = 30  # seconds
        self._last_health_check: datetime | None = None

    def register_worker(self, name: str) -> WorkerMetrics:
        """Register a new worker for monitoring.

        Args:
            name: Unique worker name

        Returns:
            WorkerMetrics object to update
        """
        with self._lock:
            if name in self._workers:
                return self._workers[name]

            metrics = WorkerMetrics(
                worker_name=name,
                status=WorkerStatus.STARTING,
                started_at=datetime.now(timezone.utc),
            )
            self._workers[name] = metrics
            return metrics

    def unregister_worker(self, name: str) -> None:
        """Unregister a worker."""
        with self._lock:
            if name in self._workers:
                self._workers[name].status = WorkerStatus.STOPPED

    def get_metrics(self, name: str) -> WorkerMetrics | None:
        """Get metrics for a specific worker."""
        return self._workers.get(name)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all workers."""
        with self._lock:
            return {name: metrics.to_dict() for name, metrics in self._workers.items()}

    def get_health(self) -> dict[str, Any]:
        """Get overall health status.

        Returns:
            Health check result with status and details
        """
        self._last_health_check = datetime.now(timezone.utc)

        with self._lock:
            workers_health = {}
            overall_healthy = True

            for name, metrics in self._workers.items():
                worker_healthy = True
                issues = []

                # Check if worker is responsive
                if metrics.status == WorkerStatus.ERROR:
                    worker_healthy = False
                    issues.append("Worker in error state")

                # Check for stale worker (no activity in 5 minutes when processing)
                if metrics.status == WorkerStatus.PROCESSING:
                    if metrics.last_activity:
                        stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
                        if metrics.last_activity < stale_threshold:
                            worker_healthy = False
                            issues.append("Worker appears stale (no activity)")

                # Check error rate (> 10% is concerning)
                error_rate = metrics._get_error_rate()
                if error_rate > 10:
                    worker_healthy = False
                    issues.append(f"High error rate: {error_rate}%")

                workers_health[name] = {
                    "healthy": worker_healthy,
                    "status": metrics.status.value,
                    "issues": issues,
                    "metrics": metrics.to_dict(),
                }

                if not worker_healthy:
                    overall_healthy = False
                    self._trigger_alerts(name, metrics)

            return {
                "healthy": overall_healthy,
                "timestamp": self._last_health_check.isoformat(),
                "workers": workers_health,
            }

    def add_alert_callback(self, callback: Callable[[str, WorkerMetrics], None]) -> None:
        """Add a callback for worker alerts.

        Args:
            callback: Function called with (worker_name, metrics) on issues
        """
        self._alert_callbacks.append(callback)

    def _trigger_alerts(self, name: str, metrics: WorkerMetrics) -> None:
        """Trigger alert callbacks for a worker."""
        for callback in self._alert_callbacks:
            try:
                callback(name, metrics)
            except Exception:
                pass  # Don't let callback errors affect monitoring

    def update_queue_size(self, name: str, size: int) -> None:
        """Update the queue size for a worker."""
        with self._lock:
            if name in self._workers:
                self._workers[name].queue_size = size


# Singleton instance
_monitor: WorkerMonitor | None = None
_monitor_lock = threading.Lock()


def get_worker_monitor() -> WorkerMonitor:
    """Get the singleton worker monitor."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = WorkerMonitor()
    return _monitor


def reset_worker_monitor() -> None:
    """Reset the singleton monitor (for testing)."""
    global _monitor
    with _monitor_lock:
        _monitor = None
