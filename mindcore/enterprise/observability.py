"""Observability module using OpenTelemetry.

Provides metrics and tracing for Mindcore operations using OpenTelemetry,
the industry standard for observability.

Requirements:
    pip install opentelemetry-api opentelemetry-sdk

Optional exporters:
    pip install opentelemetry-exporter-otlp  # For OTLP/gRPC export
    pip install opentelemetry-exporter-prometheus  # For Prometheus

Example:
    from mindcore.enterprise import ObservabilityConfig, MindcoreMetrics

    # Configure with OTLP exporter
    config = ObservabilityConfig(
        service_name="my-ai-service",
        otlp_endpoint="http://localhost:4317",
        enable_tracing=True,
        enable_metrics=True,
    )

    metrics = MindcoreMetrics(config)

    # Record operations
    metrics.record_store(user_id="user123", latency_ms=45.2)
    metrics.record_recall(user_id="user123", memories_returned=5, latency_ms=120.5)

    # Or use context manager for automatic tracing
    with metrics.trace_operation("store_memory", user_id="user123"):
        # ... operation code
        pass

References:
    - https://opentelemetry.io/docs/languages/python/
    - https://opentelemetry-python.readthedocs.io/en/latest/sdk/metrics.html
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator


# Type hints for optional dependencies
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    metrics = None
    trace = None


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class SpanKind(str, Enum):
    """Span types for tracing."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"


@dataclass
class ObservabilityConfig:
    """Configuration for observability features.

    Attributes:
        service_name: Name of the service for telemetry identification
        service_version: Version of the service
        environment: Deployment environment (production, staging, development)
        enable_tracing: Enable distributed tracing
        enable_metrics: Enable metrics collection
        otlp_endpoint: OTLP exporter endpoint (e.g., "http://localhost:4317")
        console_export: Export to console (useful for development)
        metrics_export_interval_ms: How often to export metrics (default: 60000ms)
        sample_rate: Trace sampling rate (0.0 to 1.0, default: 1.0)
        custom_attributes: Additional attributes to include in all telemetry

    Example:
        config = ObservabilityConfig(
            service_name="mindcore-service",
            environment="production",
            otlp_endpoint="http://otel-collector:4317",
            sample_rate=0.1,  # Sample 10% of traces
        )
    """

    service_name: str = "mindcore"
    service_version: str = "2.0.0"
    environment: str = "development"
    enable_tracing: bool = True
    enable_metrics: bool = True
    otlp_endpoint: str | None = None
    console_export: bool = False
    metrics_export_interval_ms: int = 60000
    sample_rate: float = 1.0
    custom_attributes: dict[str, str] = field(default_factory=dict)

    def get_resource_attributes(self) -> dict[str, str]:
        """Get OpenTelemetry resource attributes."""
        attrs = {
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
        }
        attrs.update(self.custom_attributes)
        return attrs


class MindcoreMetrics:
    """Metrics collector for Mindcore operations.

    Uses OpenTelemetry for metrics collection, supporting multiple
    export backends (OTLP, Prometheus, Console).

    Example:
        metrics = MindcoreMetrics(ObservabilityConfig(service_name="my-service"))

        # Record a store operation
        metrics.record_store(user_id="user123", latency_ms=45.2)

        # Record a recall operation
        metrics.record_recall(
            user_id="user123",
            memories_returned=5,
            latency_ms=120.5,
            cache_hit=True,
        )

        # Record custom metric
        metrics.record_custom("custom_operation_count", 1, {"type": "special"})
    """

    def __init__(self, config: ObservabilityConfig | None = None):
        """Initialize metrics collector.

        Args:
            config: Observability configuration. If None, uses defaults.

        Raises:
            ImportError: If opentelemetry packages are not installed
        """
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError(
                "OpenTelemetry packages required for observability. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        self.config = config or ObservabilityConfig()
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize OpenTelemetry metrics."""
        resource = Resource.create(self.config.get_resource_attributes())

        # Configure exporter
        readers = []
        if self.config.console_export:
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=self.config.metrics_export_interval_ms,
                )
            )

        if self.config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )

                readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(endpoint=self.config.otlp_endpoint),
                        export_interval_millis=self.config.metrics_export_interval_ms,
                    )
                )
            except ImportError:
                pass  # OTLP exporter not installed

        # Create provider and set globally
        if readers:
            provider = MeterProvider(resource=resource, metric_readers=readers)
            metrics.set_meter_provider(provider)

        # Get meter
        self._meter = metrics.get_meter(
            self.config.service_name,
            self.config.service_version,
        )

        # Create instruments
        self._store_counter = self._meter.create_counter(
            name="mindcore.store.count",
            description="Number of store operations",
            unit="1",
        )
        self._store_latency = self._meter.create_histogram(
            name="mindcore.store.latency",
            description="Store operation latency",
            unit="ms",
        )
        self._recall_counter = self._meter.create_counter(
            name="mindcore.recall.count",
            description="Number of recall operations",
            unit="1",
        )
        self._recall_latency = self._meter.create_histogram(
            name="mindcore.recall.latency",
            description="Recall operation latency",
            unit="ms",
        )
        self._recall_memories = self._meter.create_histogram(
            name="mindcore.recall.memories_returned",
            description="Number of memories returned per recall",
            unit="1",
        )
        self._cache_hits = self._meter.create_counter(
            name="mindcore.cache.hits",
            description="Cache hit count",
            unit="1",
        )
        self._cache_misses = self._meter.create_counter(
            name="mindcore.cache.misses",
            description="Cache miss count",
            unit="1",
        )
        self._errors = self._meter.create_counter(
            name="mindcore.errors",
            description="Error count by type",
            unit="1",
        )
        self._active_memories = self._meter.create_up_down_counter(
            name="mindcore.memories.active",
            description="Number of active memories",
            unit="1",
        )

    def record_store(
        self,
        user_id: str,
        latency_ms: float,
        memory_type: str = "unknown",
        success: bool = True,
        **extra_attributes: Any,
    ) -> None:
        """Record a store operation.

        Args:
            user_id: User identifier
            latency_ms: Operation latency in milliseconds
            memory_type: Type of memory stored
            success: Whether operation succeeded
            **extra_attributes: Additional metric attributes
        """
        attrs = {
            "user_id": user_id,
            "memory_type": memory_type,
            "success": str(success),
            **{k: str(v) for k, v in extra_attributes.items()},
        }
        self._store_counter.add(1, attrs)
        self._store_latency.record(latency_ms, attrs)
        if success:
            self._active_memories.add(1, {"user_id": user_id})

    def record_recall(
        self,
        user_id: str,
        memories_returned: int,
        latency_ms: float,
        cache_hit: bool = False,
        success: bool = True,
        **extra_attributes: Any,
    ) -> None:
        """Record a recall operation.

        Args:
            user_id: User identifier
            memories_returned: Number of memories returned
            latency_ms: Operation latency in milliseconds
            cache_hit: Whether result was from cache
            success: Whether operation succeeded
            **extra_attributes: Additional metric attributes
        """
        attrs = {
            "user_id": user_id,
            "cache_hit": str(cache_hit),
            "success": str(success),
            **{k: str(v) for k, v in extra_attributes.items()},
        }
        self._recall_counter.add(1, attrs)
        self._recall_latency.record(latency_ms, attrs)
        self._recall_memories.record(memories_returned, attrs)

        if cache_hit:
            self._cache_hits.add(1, {"user_id": user_id})
        else:
            self._cache_misses.add(1, {"user_id": user_id})

    def record_delete(self, user_id: str, count: int = 1) -> None:
        """Record memory deletion.

        Args:
            user_id: User identifier
            count: Number of memories deleted
        """
        self._active_memories.add(-count, {"user_id": user_id})

    def record_error(
        self,
        error_type: str,
        operation: str,
        user_id: str | None = None,
    ) -> None:
        """Record an error.

        Args:
            error_type: Type/class of error
            operation: Operation that failed
            user_id: Optional user identifier
        """
        attrs = {"error_type": error_type, "operation": operation}
        if user_id:
            attrs["user_id"] = user_id
        self._errors.add(1, attrs)

    def record_custom(
        self,
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
        metric_type: MetricType = MetricType.COUNTER,
    ) -> None:
        """Record a custom metric.

        Args:
            name: Metric name (will be prefixed with "mindcore.custom.")
            value: Metric value
            attributes: Optional attributes
            metric_type: Type of metric (counter, histogram, gauge)
        """
        full_name = f"mindcore.custom.{name}"
        attrs = attributes or {}

        # Create instrument on demand
        if metric_type == MetricType.COUNTER:
            counter = self._meter.create_counter(full_name, unit="1")
            counter.add(int(value), attrs)
        elif metric_type == MetricType.HISTOGRAM:
            histogram = self._meter.create_histogram(full_name, unit="1")
            histogram.record(value, attrs)
        elif metric_type == MetricType.GAUGE:
            gauge = self._meter.create_up_down_counter(full_name, unit="1")
            gauge.add(value, attrs)


class MindcoreTracer:
    """Distributed tracing for Mindcore operations.

    Uses OpenTelemetry for distributed tracing, enabling request
    correlation across services.

    Example:
        tracer = MindcoreTracer(ObservabilityConfig(service_name="my-service"))

        # Trace an operation
        with tracer.trace_operation("store_memory", user_id="user123") as span:
            span.set_attribute("memory_type", "semantic")
            # ... operation code
            span.add_event("memory_stored", {"memory_id": "mem_123"})

        # Or use decorator
        @tracer.traced("recall_memories")
        def recall(user_id: str, query: str):
            # ... implementation
            pass
    """

    def __init__(self, config: ObservabilityConfig | None = None):
        """Initialize tracer.

        Args:
            config: Observability configuration. If None, uses defaults.

        Raises:
            ImportError: If opentelemetry packages are not installed
        """
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError(
                "OpenTelemetry packages required for tracing. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        self.config = config or ObservabilityConfig()
        self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        resource = Resource.create(self.config.get_resource_attributes())

        # Create provider
        provider = TracerProvider(resource=resource)

        # Add exporters
        if self.config.console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        if self.config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=self.config.otlp_endpoint))
                )
            except ImportError:
                pass  # OTLP exporter not installed

        # Set globally
        trace.set_tracer_provider(provider)

        # Get tracer
        self._tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version,
        )

    @contextmanager
    def trace_operation(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        **attributes: Any,
    ) -> Generator[Any, None, None]:
        """Context manager for tracing an operation.

        Args:
            name: Operation name
            kind: Span kind (internal, client, server)
            **attributes: Span attributes

        Yields:
            OpenTelemetry span object

        Example:
            with tracer.trace_operation("store_memory", user_id="user123") as span:
                span.set_attribute("memory_type", "semantic")
                result = do_store()
                span.add_event("stored", {"memory_id": result.id})
        """
        span_kind_map = {
            SpanKind.INTERNAL: trace.SpanKind.INTERNAL,
            SpanKind.CLIENT: trace.SpanKind.CLIENT,
            SpanKind.SERVER: trace.SpanKind.SERVER,
        }

        with self._tracer.start_as_current_span(
            f"mindcore.{name}",
            kind=span_kind_map.get(kind, trace.SpanKind.INTERNAL),
        ) as span:
            # Set initial attributes
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration_ms)

    def traced(self, name: str, **default_attributes: Any):
        """Decorator for tracing a function.

        Args:
            name: Operation name
            **default_attributes: Default span attributes

        Example:
            @tracer.traced("recall_memories")
            def recall(user_id: str, query: str):
                # ... implementation
                pass
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Merge default attributes with runtime ones
                attrs = {**default_attributes}
                if "user_id" in kwargs:
                    attrs["user_id"] = kwargs["user_id"]

                with self.trace_operation(name, **attrs):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_current_trace_id(self) -> str | None:
        """Get the current trace ID for correlation.

        Returns:
            Trace ID as hex string, or None if no active trace
        """
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, "032x")
        return None

    def get_current_span_id(self) -> str | None:
        """Get the current span ID for correlation.

        Returns:
            Span ID as hex string, or None if no active span
        """
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, "016x")
        return None
