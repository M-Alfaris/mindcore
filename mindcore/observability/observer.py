"""Main observer class for Mindcore observability.

Provides a unified interface for metrics, alerts, and quality monitoring.
"""

import threading
import time
from typing import TYPE_CHECKING, Any

from mindcore.utils.logger import get_logger

from .alerts import AlertConfig, AlertHandler, AlertManager, LoggingAlertHandler
from .metrics import MetricsCollector, ObservabilityBackend
from .quality import EnrichmentQualityMonitor


if TYPE_CHECKING:
    from mindcore import MindcoreClient

logger = get_logger(__name__)


class MindcoreObserver:
    """Unified observability for Mindcore.

    Combines:
    - Metrics collection (enrichment, context, workers)
    - Alerting (configurable thresholds and handlers)
    - Quality monitoring (confidence tracking, trend detection)

    Example:
        >>> from mindcore.observability import MindcoreObserver, AlertConfig
        >>>
        >>> # Create observer with alerts
        >>> observer = MindcoreObserver(
        ...     backend="prometheus",
        ...     alerts=AlertConfig(
        ...         enrichment_queue_depth={"threshold": 1000, "severity": "warning"},
        ...         enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
        ...         low_confidence_rate={"threshold": 0.2, "severity": "warning"},
        ...     )
        ... )
        >>>
        >>> # Attach to client for automatic instrumentation
        >>> observer.attach(client)
        >>>
        >>> # Or use standalone
        >>> observer.record_enrichment(
        ...     message_id="msg123",
        ...     confidence_score=0.85,
        ...     latency_ms=150,
        ...     enrichment_source="llm",
        ...     vocabulary_match_rate=0.9,
        ...     success=True,
        ... )
        >>>
        >>> # Check alerts
        >>> alerts = observer.check_alerts()
        >>>
        >>> # Get quality report
        >>> report = observer.get_quality_report()
    """

    def __init__(
        self,
        backend: str | ObservabilityBackend = "memory",
        alerts: AlertConfig | None = None,
        alert_handlers: list[AlertHandler] | None = None,
        quality_window_size: int = 1000,
        low_confidence_threshold: float = 0.3,
        alert_check_interval: float = 60.0,
        auto_check_alerts: bool = False,
    ):
        """Initialize Mindcore observer.

        Args:
            backend: Metrics backend ("memory", "prometheus", or ObservabilityBackend)
            alerts: Alert configuration (optional)
            alert_handlers: Custom alert handlers (defaults to logging)
            quality_window_size: Number of samples for quality tracking
            low_confidence_threshold: Threshold for low confidence alerts
            alert_check_interval: Seconds between automatic alert checks
            auto_check_alerts: Whether to start background alert checking
        """
        # Parse backend
        if isinstance(backend, str):
            backend = ObservabilityBackend(backend)

        # Initialize components
        self._metrics = MetricsCollector(backend=backend)

        self._quality_monitor = EnrichmentQualityMonitor(
            window_size=quality_window_size,
            low_confidence_threshold=low_confidence_threshold,
        )

        # Initialize alerting
        self._alert_config = alerts or AlertConfig()
        handlers = alert_handlers or [LoggingAlertHandler()]
        self._alert_manager = AlertManager(self._alert_config, handlers)

        # Background alert checking
        self._alert_check_interval = alert_check_interval
        self._alert_thread: threading.Thread | None = None
        self._running = False

        if auto_check_alerts and alerts:
            self.start_alert_checking()

        logger.info(
            f"MindcoreObserver initialized: backend={backend.value}, "
            f"alerts_configured={len(self._alert_manager.rules)}"
        )

    def attach(self, client: "MindcoreClient") -> None:
        """Attach observer to a MindcoreClient for automatic instrumentation.

        This wraps key methods to automatically record metrics.

        Args:
            client: MindcoreClient to instrument
        """
        # Store reference for queue monitoring
        self._client = client

        # Wrap the enrichment worker notification
        original_notify = getattr(
            client._cache_invalidation, "notify_enrichment_complete", None
        )

        if original_notify:
            def instrumented_notify(message):
                # Record metrics
                if hasattr(message, "metadata"):
                    self.record_enrichment(
                        message_id=message.message_id,
                        confidence_score=message.metadata.confidence_score,
                        latency_ms=message.metadata.enrichment_latency_ms or 0,
                        enrichment_source=message.metadata.enrichment_source,
                        vocabulary_match_rate=message.metadata.vocabulary_match_rate,
                        success=not message.metadata.enrichment_failed,
                    )
                return original_notify(message)

            client._cache_invalidation.notify_enrichment_complete = instrumented_notify

        logger.info("Observer attached to MindcoreClient")

    def record_enrichment(
        self,
        message_id: str,
        confidence_score: float,
        latency_ms: float,
        enrichment_source: str,
        vocabulary_match_rate: float,
        success: bool,
    ) -> None:
        """Record enrichment metrics.

        Args:
            message_id: ID of enriched message
            confidence_score: Confidence score (0.0-1.0)
            latency_ms: Processing time in milliseconds
            enrichment_source: Source (llm, trivial, fallback)
            vocabulary_match_rate: Rate of vocabulary matches
            success: Whether enrichment succeeded
        """
        self._metrics.record_enrichment(
            message_id=message_id,
            confidence_score=confidence_score,
            latency_ms=latency_ms,
            enrichment_source=enrichment_source,
            vocabulary_match_rate=vocabulary_match_rate,
            success=success,
        )

        self._quality_monitor.record(
            message_id=message_id,
            confidence_score=confidence_score,
            enrichment_source=enrichment_source,
            vocabulary_match_rate=vocabulary_match_rate,
            latency_ms=latency_ms,
        )

    def record_context_retrieval(
        self,
        latency_ms: float,
        llm_calls: int,
        fast_path_used: bool,
        confidence: str,
        context_source: str,
    ) -> None:
        """Record context retrieval metrics.

        Args:
            latency_ms: Total retrieval time
            llm_calls: Number of LLM calls made
            fast_path_used: Whether fast path was used
            confidence: Confidence level (high/medium/low)
            context_source: Source of context
        """
        self._metrics.record_context_retrieval(
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            fast_path_used=fast_path_used,
            confidence=confidence,
            context_source=context_source,
        )

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record queue depth for a named queue."""
        self._metrics.record_queue_depth(queue_name, depth)

    def record_worker_status(
        self,
        worker_name: str,
        processed: int,
        errors: int,
        idle_time_ms: float,
    ) -> None:
        """Record worker status metrics."""
        self._metrics.record_worker_status(worker_name, processed, errors, idle_time_ms)

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check all alert rules against current metrics.

        Returns:
            List of triggered alerts
        """
        metrics = self._metrics.get_alert_metrics()

        # Add queue depth if client is attached
        if hasattr(self, "_client") and self._client:
            try:
                queue_size = self._client._enrichment_queue.qsize()
                metrics["enrichment_queue_depth"] = queue_size
            except Exception:
                pass

        triggered = self._alert_manager.check(metrics)
        return [alert.to_dict() for alert in triggered]

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get currently active (unresolved) alerts."""
        return [alert.to_dict() for alert in self._alert_manager.active_alerts]

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return self._metrics.get_metrics()

    def get_quality_report(self) -> dict[str, Any]:
        """Get comprehensive quality report.

        Includes:
        - Confidence statistics
        - Trend analysis
        - Latency stats
        - Vocabulary match rates
        - Source distribution
        - Low confidence samples for review
        """
        return self._quality_monitor.get_report()

    def get_health_check(self) -> dict[str, Any]:
        """Get quick health check status.

        Returns:
            Health status with any detected issues
        """
        quality_health = self._quality_monitor.get_health_check()
        active_alerts = self.get_active_alerts()

        return {
            "healthy": quality_health["healthy"] and len(active_alerts) == 0,
            "quality": quality_health,
            "active_alerts": active_alerts,
            "alert_count": len(active_alerts),
        }

    def start_alert_checking(self) -> None:
        """Start background thread for periodic alert checking."""
        if self._running:
            return

        self._running = True
        self._alert_thread = threading.Thread(
            target=self._alert_check_loop,
            name="mindcore_alert_checker",
            daemon=True,
        )
        self._alert_thread.start()
        logger.info(f"Started alert checking (interval: {self._alert_check_interval}s)")

    def stop_alert_checking(self) -> None:
        """Stop background alert checking."""
        self._running = False
        if self._alert_thread:
            self._alert_thread.join(timeout=5.0)
            self._alert_thread = None
        logger.info("Stopped alert checking")

    def _alert_check_loop(self) -> None:
        """Background loop for checking alerts."""
        while self._running:
            try:
                self.check_alerts()
            except Exception as e:
                logger.error(f"Alert check error: {e}")

            time.sleep(self._alert_check_interval)

    def get_status(self) -> dict[str, Any]:
        """Get observer status summary."""
        return {
            "backend": self._metrics.backend_type.value,
            "alert_rules": len(self._alert_manager.rules),
            "active_alerts": len(self._alert_manager.active_alerts),
            "alert_checking_active": self._running,
            "quality_stats": self._quality_monitor.get_report().get("confidence", {}),
        }

    def close(self) -> None:
        """Clean up resources."""
        self.stop_alert_checking()
        logger.info("MindcoreObserver closed")
