"""Prometheus Metrics Export for Mindcore.

Provides Prometheus-compatible metrics for monitoring Mindcore in production.
Metrics are exposed via a simple HTTP endpoint or can be collected programmatically.

Dependencies:
    pip install prometheus-client

Usage:
    >>> from mindcore.core.prometheus_metrics import (
    ...     MindcoreMetrics, get_metrics, start_metrics_server
    ... )
    >>>
    >>> # Get metrics instance
    >>> metrics = get_metrics()
    >>>
    >>> # Record events
    >>> metrics.record_ingestion()
    >>> metrics.record_enrichment(duration_ms=150.0, trivial=False)
    >>> metrics.record_context_retrieval(duration_ms=200.0)
    >>>
    >>> # Start HTTP server for Prometheus scraping
    >>> start_metrics_server(port=9090)
"""

import threading
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)

# Optional prometheus_client import
try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed. Install with: pip install prometheus-client")


class MindcoreMetrics:
    """Prometheus metrics collector for Mindcore.

    Tracks:
    - Message ingestion rate and latency
    - Enrichment processing time and throughput
    - Context retrieval performance
    - Cache hit/miss rates
    - Worker pool status
    - Queue depth
    - Error rates

    Example:
        >>> metrics = MindcoreMetrics()
        >>>
        >>> # Record metrics
        >>> metrics.record_ingestion()
        >>> metrics.record_enrichment(duration_ms=150.0)
        >>>
        >>> # Get current values
        >>> print(metrics.get_metrics_dict())
    """

    def __init__(self, registry: Any | None = None):
        """Initialize metrics collector.

        Args:
            registry: Optional Prometheus CollectorRegistry.
                     Uses default registry if None.
        """
        self._lock = threading.Lock()
        self._enabled = PROMETHEUS_AVAILABLE

        if not self._enabled:
            # Create dummy counters for when prometheus is not available
            self._counters: dict[str, int] = {
                "messages_ingested": 0,
                "messages_enriched": 0,
                "trivial_skipped": 0,
                "context_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "errors": 0,
            }
            self._gauges: dict[str, float] = {
                "queue_depth": 0,
                "active_workers": 0,
                "worker_pool_size": 1,
            }
            self._histograms: dict[str, list] = {
                "enrichment_duration_ms": [],
                "context_duration_ms": [],
                "ingestion_duration_ms": [],
            }
            return

        # Use provided registry or default
        self._registry = registry or REGISTRY

        # Info metric (static labels)
        self.info = Info("mindcore", "Mindcore framework information", registry=self._registry)
        self.info.info({"version": "0.3.0", "component": "memory_management"})

        # Counters (monotonically increasing)
        self.messages_ingested_total = Counter(
            "mindcore_messages_ingested_total",
            "Total number of messages ingested",
            registry=self._registry,
        )

        self.messages_enriched_total = Counter(
            "mindcore_messages_enriched_total",
            "Total number of messages enriched by LLM",
            ["method"],  # 'llm' or 'trivial'
            registry=self._registry,
        )

        self.context_requests_total = Counter(
            "mindcore_context_requests_total",
            "Total number of context retrieval requests",
            registry=self._registry,
        )

        self.cache_operations_total = Counter(
            "mindcore_cache_operations_total",
            "Total cache operations",
            ["result"],  # 'hit' or 'miss'
            registry=self._registry,
        )

        self.errors_total = Counter(
            "mindcore_errors_total",
            "Total errors by type",
            ["error_type"],  # 'enrichment', 'context', 'database', etc.
            registry=self._registry,
        )

        # Gauges (can go up or down)
        self.queue_depth = Gauge(
            "mindcore_enrichment_queue_depth",
            "Current number of messages in enrichment queue",
            registry=self._registry,
        )

        self.active_workers = Gauge(
            "mindcore_active_workers",
            "Number of currently active enrichment workers",
            registry=self._registry,
        )

        self.worker_pool_size = Gauge(
            "mindcore_worker_pool_size", "Configured worker pool size", registry=self._registry
        )

        self.cache_size = Gauge(
            "mindcore_cache_size_bytes", "Approximate cache size in bytes", registry=self._registry
        )

        # Histograms (for latency distributions)
        self.enrichment_duration = Histogram(
            "mindcore_enrichment_duration_seconds",
            "Time spent enriching messages",
            ["method"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        self.context_retrieval_duration = Histogram(
            "mindcore_context_retrieval_duration_seconds",
            "Time spent retrieving context",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self._registry,
        )

        self.ingestion_duration = Histogram(
            "mindcore_ingestion_duration_seconds",
            "Time spent ingesting messages (without enrichment)",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
            registry=self._registry,
        )

        # Summaries (for percentiles)
        self.llm_tokens_used = Summary(
            "mindcore_llm_tokens_used",
            "LLM tokens used per operation",
            ["operation"],  # 'enrichment', 'context'
            registry=self._registry,
        )

        logger.info("Prometheus metrics initialized")

    @property
    def enabled(self) -> bool:
        """Check if Prometheus metrics are available."""
        return self._enabled

    def record_ingestion(self, duration_ms: float | None = None) -> None:
        """Record a message ingestion event.

        Args:
            duration_ms: Optional ingestion duration in milliseconds
        """
        if self._enabled:
            self.messages_ingested_total.inc()
            if duration_ms is not None:
                self.ingestion_duration.observe(duration_ms / 1000.0)
        else:
            with self._lock:
                self._counters["messages_ingested"] += 1
                if duration_ms is not None:
                    self._histograms["ingestion_duration_ms"].append(duration_ms)

    def record_enrichment(
        self, duration_ms: float, trivial: bool = False, tokens_used: int | None = None
    ) -> None:
        """Record an enrichment event.

        Args:
            duration_ms: Enrichment duration in milliseconds
            trivial: True if message was trivially enriched (no LLM)
            tokens_used: Optional number of LLM tokens used
        """
        method = "trivial" if trivial else "llm"

        if self._enabled:
            self.messages_enriched_total.labels(method=method).inc()
            self.enrichment_duration.labels(method=method).observe(duration_ms / 1000.0)
            if tokens_used is not None:
                self.llm_tokens_used.labels(operation="enrichment").observe(tokens_used)
        else:
            with self._lock:
                self._counters["messages_enriched"] += 1
                if trivial:
                    self._counters["trivial_skipped"] += 1
                self._histograms["enrichment_duration_ms"].append(duration_ms)

    def record_context_retrieval(self, duration_ms: float, tokens_used: int | None = None) -> None:
        """Record a context retrieval event.

        Args:
            duration_ms: Retrieval duration in milliseconds
            tokens_used: Optional number of LLM tokens used
        """
        if self._enabled:
            self.context_requests_total.inc()
            self.context_retrieval_duration.observe(duration_ms / 1000.0)
            if tokens_used is not None:
                self.llm_tokens_used.labels(operation="context").observe(tokens_used)
        else:
            with self._lock:
                self._counters["context_requests"] += 1
                self._histograms["context_duration_ms"].append(duration_ms)

    def record_cache_access(self, hit: bool) -> None:
        """Record a cache access.

        Args:
            hit: True if cache hit, False if miss
        """
        if self._enabled:
            result = "hit" if hit else "miss"
            self.cache_operations_total.labels(result=result).inc()
        else:
            with self._lock:
                if hit:
                    self._counters["cache_hits"] += 1
                else:
                    self._counters["cache_misses"] += 1

    def record_error(self, error_type: str) -> None:
        """Record an error.

        Args:
            error_type: Type of error (enrichment, context, database, etc.)
        """
        if self._enabled:
            self.errors_total.labels(error_type=error_type).inc()
        else:
            with self._lock:
                self._counters["errors"] += 1

    def set_queue_depth(self, depth: int) -> None:
        """Set current enrichment queue depth."""
        if self._enabled:
            self.queue_depth.set(depth)
        else:
            with self._lock:
                self._gauges["queue_depth"] = depth

    def set_active_workers(self, count: int) -> None:
        """Set number of currently active workers."""
        if self._enabled:
            self.active_workers.set(count)
        else:
            with self._lock:
                self._gauges["active_workers"] = count

    def set_worker_pool_size(self, size: int) -> None:
        """Set configured worker pool size."""
        if self._enabled:
            self.worker_pool_size.set(size)
        else:
            with self._lock:
                self._gauges["worker_pool_size"] = size

    def set_cache_size(self, size_bytes: int) -> None:
        """Set approximate cache size in bytes."""
        if self._enabled:
            self.cache_size.set(size_bytes)

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get current metrics as a dictionary.

        Useful for non-Prometheus monitoring or debugging.

        Returns:
            Dictionary with all current metric values
        """
        if not self._enabled:
            with self._lock:
                # Calculate histogram stats
                def calc_stats(values):
                    if not values:
                        return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
                    sorted_vals = sorted(values)
                    n = len(sorted_vals)
                    return {
                        "count": n,
                        "avg": sum(sorted_vals) / n,
                        "p50": sorted_vals[int(n * 0.5)] if n > 0 else 0,
                        "p95": sorted_vals[int(n * 0.95)] if n > 0 else 0,
                        "p99": sorted_vals[int(n * 0.99)] if n > 0 else 0,
                    }

                cache_total = self._counters["cache_hits"] + self._counters["cache_misses"]
                cache_hit_rate = (
                    self._counters["cache_hits"] / cache_total if cache_total > 0 else 0
                )

                return {
                    "counters": dict(self._counters),
                    "gauges": dict(self._gauges),
                    "histograms": {k: calc_stats(v) for k, v in self._histograms.items()},
                    "cache_hit_rate": round(cache_hit_rate, 4),
                    "prometheus_enabled": False,
                }

        # When Prometheus is available, collect from registry
        return {
            "messages_ingested": self.messages_ingested_total._value.get(),
            "context_requests": self.context_requests_total._value.get(),
            "queue_depth": self.queue_depth._value.get(),
            "active_workers": self.active_workers._value.get(),
            "worker_pool_size": self.worker_pool_size._value.get(),
            "prometheus_enabled": True,
        }

    def generate_prometheus_output(self) -> str:
        """Generate Prometheus exposition format output.

        Returns:
            Prometheus-formatted metrics string

        Raises:
            RuntimeError: If prometheus_client is not installed
        """
        if not self._enabled:
            raise RuntimeError(
                "Prometheus metrics not available. Install with: pip install prometheus-client"
            )

        return generate_latest(self._registry).decode("utf-8")


# Singleton instance
_metrics: MindcoreMetrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MindcoreMetrics:
    """Get or create the singleton metrics instance."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = MindcoreMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset the singleton metrics instance (for testing)."""
    global _metrics
    with _metrics_lock:
        _metrics = None


def start_metrics_server(port: int = 9090, addr: str = "") -> bool:
    """Start HTTP server for Prometheus metrics scraping.

    Args:
        port: Port to listen on (default: 9090)
        addr: Address to bind to (default: all interfaces)

    Returns:
        True if server started, False if prometheus_client not available

    Example:
        >>> # Start metrics server
        >>> if start_metrics_server(port=9090):
        ...     print("Metrics available at http://localhost:9090/metrics")
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning(
            "Cannot start metrics server: prometheus_client not installed. "
            "Install with: pip install prometheus-client"
        )
        return False

    try:
        start_http_server(port, addr)
        logger.info(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        logger.exception(f"Failed to start metrics server: {e}")
        return False


def is_prometheus_available() -> bool:
    """Check if prometheus_client is installed."""
    return PROMETHEUS_AVAILABLE
