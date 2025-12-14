"""Metrics collection for Mindcore observability.

Supports multiple backends: in-memory, Prometheus, custom.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


class ObservabilityBackend(str, Enum):
    """Supported observability backends."""

    MEMORY = "memory"  # In-memory metrics (default)
    PROMETHEUS = "prometheus"  # Prometheus client
    LOGGING = "logging"  # Log-based metrics
    CUSTOM = "custom"  # Custom backend


@dataclass
class EnrichmentMetrics:
    """Metrics for a single enrichment operation."""

    message_id: str
    confidence_score: float
    latency_ms: float
    enrichment_source: str
    vocabulary_match_rate: float
    success: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "enrichment_source": self.enrichment_source,
            "vocabulary_match_rate": self.vocabulary_match_rate,
            "success": self.success,
            "timestamp": self.timestamp,
        }


@dataclass
class ContextMetrics:
    """Metrics for a context retrieval operation."""

    query_preview: str
    latency_ms: float
    llm_calls: int
    fast_path_used: bool
    confidence: str
    context_source: str
    timestamp: float = field(default_factory=time.time)


class MetricsBackend(ABC):
    """Abstract base for metrics backends."""

    @abstractmethod
    def record_counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Record a counter metric."""

    @abstractmethod
    def record_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""

    @abstractmethod
    def record_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get all current metrics."""


class MemoryMetricsBackend(MetricsBackend):
    """In-memory metrics backend for development/testing."""

    def __init__(self, window_size: int = 1000):
        """Initialize memory backend.

        Args:
            window_size: Number of samples to keep for histograms
        """
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, deque] = {}
        self._window_size = window_size
        self._lock = threading.Lock()

    def record_counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Record a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def record_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def record_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = deque(maxlen=self._window_size)
            self._histograms[key].append(value)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metrics(self) -> dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    result["histograms"][key] = {
                        "count": n,
                        "sum": sum(sorted_values),
                        "avg": sum(sorted_values) / n,
                        "min": sorted_values[0],
                        "max": sorted_values[-1],
                        "p50": sorted_values[n // 2],
                        "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                        "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
                    }

            return result

    def get_summary_metrics(self) -> dict[str, float]:
        """Get flat summary metrics for alerting."""
        metrics = self.get_metrics()
        result = {}

        # Flatten counters and gauges
        result.update(metrics["counters"])
        result.update(metrics["gauges"])

        # Add histogram percentiles
        for key, hist in metrics["histograms"].items():
            result[f"{key}_p99"] = hist.get("p99", 0)
            result[f"{key}_avg"] = hist.get("avg", 0)

        return result


class PrometheusMetricsBackend(MetricsBackend):
    """Prometheus metrics backend."""

    def __init__(self, prefix: str = "mindcore"):
        """Initialize Prometheus backend.

        Args:
            prefix: Metric name prefix
        """
        self._prefix = prefix
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

        try:
            from prometheus_client import Counter, Gauge, Histogram
            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("prometheus_client not installed, falling back to memory backend")

    def _get_counter(self, name: str, labels: dict[str, str] | None) -> Any:
        """Get or create a counter."""
        if not self._available:
            return None

        label_names = tuple(sorted(labels.keys())) if labels else ()
        key = (name, label_names)

        if key not in self._counters:
            self._counters[key] = self._Counter(
                f"{self._prefix}_{name}",
                f"Counter for {name}",
                list(label_names),
            )

        return self._counters[key]

    def _get_gauge(self, name: str, labels: dict[str, str] | None) -> Any:
        """Get or create a gauge."""
        if not self._available:
            return None

        label_names = tuple(sorted(labels.keys())) if labels else ()
        key = (name, label_names)

        if key not in self._gauges:
            self._gauges[key] = self._Gauge(
                f"{self._prefix}_{name}",
                f"Gauge for {name}",
                list(label_names),
            )

        return self._gauges[key]

    def _get_histogram(self, name: str, labels: dict[str, str] | None) -> Any:
        """Get or create a histogram."""
        if not self._available:
            return None

        label_names = tuple(sorted(labels.keys())) if labels else ()
        key = (name, label_names)

        if key not in self._histograms:
            self._histograms[key] = self._Histogram(
                f"{self._prefix}_{name}",
                f"Histogram for {name}",
                list(label_names),
            )

        return self._histograms[key]

    def record_counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Record a counter metric."""
        counter = self._get_counter(name, labels)
        if counter:
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)

    def record_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        gauge = self._get_gauge(name, labels)
        if gauge:
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)

    def record_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        histogram = self._get_histogram(name, labels)
        if histogram:
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)

    def get_metrics(self) -> dict[str, Any]:
        """Get all current metrics (returns Prometheus format info)."""
        return {
            "backend": "prometheus",
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "available": self._available,
        }


class MetricsCollector:
    """Central metrics collector for Mindcore."""

    def __init__(
        self,
        backend: ObservabilityBackend = ObservabilityBackend.MEMORY,
        **backend_kwargs,
    ):
        """Initialize metrics collector.

        Args:
            backend: Which backend to use
            **backend_kwargs: Backend-specific configuration
        """
        self.backend_type = backend

        if backend == ObservabilityBackend.PROMETHEUS:
            self._backend = PrometheusMetricsBackend(**backend_kwargs)
        else:
            self._backend = MemoryMetricsBackend(**backend_kwargs)

        # Track enrichment stats for alerting
        self._enrichment_count = 0
        self._enrichment_errors = 0
        self._low_confidence_count = 0
        self._context_fallback_count = 0
        self._context_count = 0

    def record_enrichment(
        self,
        message_id: str,
        confidence_score: float,
        latency_ms: float,
        enrichment_source: str,
        vocabulary_match_rate: float,
        success: bool,
    ) -> None:
        """Record enrichment metrics."""
        self._enrichment_count += 1

        labels = {"source": enrichment_source}

        self._backend.record_counter("enrichment_total", 1, labels)
        self._backend.record_histogram("enrichment_latency_ms", latency_ms, labels)
        self._backend.record_histogram("enrichment_confidence", confidence_score, labels)
        self._backend.record_histogram("vocabulary_match_rate", vocabulary_match_rate, labels)

        if not success:
            self._enrichment_errors += 1
            self._backend.record_counter("enrichment_errors_total", 1, labels)

        if confidence_score < 0.3:
            self._low_confidence_count += 1
            self._backend.record_counter("enrichment_low_confidence_total", 1)

        # Update gauges
        if self._enrichment_count > 0:
            self._backend.record_gauge(
                "enrichment_error_rate",
                self._enrichment_errors / self._enrichment_count,
            )
            self._backend.record_gauge(
                "low_confidence_rate",
                self._low_confidence_count / self._enrichment_count,
            )

    def record_context_retrieval(
        self,
        latency_ms: float,
        llm_calls: int,
        fast_path_used: bool,
        confidence: str,
        context_source: str,
    ) -> None:
        """Record context retrieval metrics."""
        self._context_count += 1

        labels = {"source": context_source}

        self._backend.record_counter("context_retrieval_total", 1, labels)
        self._backend.record_histogram("context_latency_ms", latency_ms, labels)
        self._backend.record_counter("llm_calls_total", llm_calls)

        if fast_path_used:
            self._backend.record_counter("context_fast_path_total", 1)

        if context_source == "fallback":
            self._context_fallback_count += 1
            self._backend.record_counter("context_fallback_total", 1)

        # Update fallback rate
        if self._context_count > 0:
            self._backend.record_gauge(
                "context_fallback_rate",
                self._context_fallback_count / self._context_count,
            )

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record queue depth metric."""
        self._backend.record_gauge(
            f"{queue_name}_queue_depth",
            depth,
        )

    def record_worker_status(
        self,
        worker_name: str,
        processed: int,
        errors: int,
        idle_time_ms: float,
    ) -> None:
        """Record worker status metrics."""
        labels = {"worker": worker_name}

        self._backend.record_gauge("worker_processed_total", processed, labels)
        self._backend.record_gauge("worker_errors_total", errors, labels)
        self._backend.record_gauge("worker_idle_time_ms", idle_time_ms, labels)

        if processed > 0:
            self._backend.record_gauge(
                "worker_error_rate",
                errors / processed,
                labels,
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return self._backend.get_metrics()

    def get_alert_metrics(self) -> dict[str, float]:
        """Get metrics formatted for alerting."""
        if isinstance(self._backend, MemoryMetricsBackend):
            return self._backend.get_summary_metrics()

        # For Prometheus, return what we track internally
        return {
            "enrichment_error_rate": (
                self._enrichment_errors / self._enrichment_count
                if self._enrichment_count > 0 else 0
            ),
            "low_confidence_rate": (
                self._low_confidence_count / self._enrichment_count
                if self._enrichment_count > 0 else 0
            ),
            "context_fallback_rate": (
                self._context_fallback_count / self._context_count
                if self._context_count > 0 else 0
            ),
        }
