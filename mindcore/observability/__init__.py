"""Observability package for Mindcore.

Optional package for monitoring, alerting, and quality tracking.
Supports multiple backends: Prometheus, logging, custom handlers.

Installation:
    pip install mindcore[observability]

Usage:
    from mindcore.observability import MindcoreObserver, AlertConfig

    observer = MindcoreObserver(
        backend="prometheus",
        alerts=AlertConfig(
            enrichment_queue_depth={"threshold": 1000, "severity": "warning"},
            enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
            low_confidence_rate={"threshold": 0.2, "severity": "warning"},
        )
    )

    # Record metrics
    observer.record_enrichment(
        message_id="msg123",
        confidence_score=0.85,
        latency_ms=150,
        enrichment_source="llm",
        vocabulary_match_rate=0.9,
        success=True,
    )
    observer.check_alerts()
"""

from .alerts import AlertConfig, AlertHandler, AlertSeverity, LoggingAlertHandler
from .metrics import EnrichmentMetrics, MetricsCollector, ObservabilityBackend
from .observer import MindcoreObserver
from .quality import ConfidenceTracker, EnrichmentQualityMonitor


__all__ = [
    "AlertConfig",
    "AlertHandler",
    "AlertSeverity",
    "ConfidenceTracker",
    "EnrichmentMetrics",
    "EnrichmentQualityMonitor",
    "LoggingAlertHandler",
    "MetricsCollector",
    "MindcoreObserver",
    "ObservabilityBackend",
]
