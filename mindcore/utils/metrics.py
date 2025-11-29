"""
Performance metrics and instrumentation for Mindcore.

Provides utilities for:
- Timing function execution
- Recording performance metrics
- Tool call tracking
- Latency monitoring
"""
import time
import functools
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from datetime import datetime, timezone


class PerformanceTimer:
    """Context manager and decorator for timing operations."""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: Optional[int] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = int((self.end_time - self.start_time) * 1000)
        return False

    @property
    def elapsed(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.elapsed_ms is not None:
            return self.elapsed_ms
        if self.start_time is None:
            return 0
        current = time.perf_counter()
        return int((current - self.start_time) * 1000)


def timed(operation_name: Optional[str] = None):
    """Decorator for timing function execution.

    Usage:
        @timed("my_operation")
        def my_function():
            ...

        @timed()
        async def my_async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with PerformanceTimer(name) as timer:
                result = func(*args, **kwargs)
            _record_timing(name, timer.elapsed_ms)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(name) as timer:
                result = await func(*args, **kwargs)
            _record_timing(name, timer.elapsed_ms)
            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def measure_time(operation_name: str = "operation"):
    """Context manager for measuring execution time.

    Usage:
        with measure_time("llm_call") as timer:
            result = llm.generate(prompt)
        print(f"LLM call took {timer.elapsed_ms}ms")
    """
    timer = PerformanceTimer(operation_name)
    timer.__enter__()
    try:
        yield timer
    finally:
        timer.__exit__(None, None, None)


# Global metrics storage (shared with dashboard API)
_metrics_callback: Optional[Callable] = None


def set_metrics_callback(callback: Callable[[str, Dict[str, Any]], None]):
    """Set a callback function to receive metrics.

    The callback receives (metric_type, data) where:
    - metric_type: 'response_time', 'tool_call', etc.
    - data: Dict with metric details
    """
    global _metrics_callback
    _metrics_callback = callback


def _record_timing(operation_name: str, elapsed_ms: int):
    """Internal function to record timing metrics."""
    if _metrics_callback:
        _metrics_callback("response_time", {
            "operation": operation_name,
            "total_time_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


def record_llm_call(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error: Optional[str] = None
):
    """Record an LLM API call metric.

    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        latency_ms: Response time in milliseconds
        success: Whether the call succeeded
        error: Error message if failed
    """
    if _metrics_callback:
        _metrics_callback("response_time", {
            "operation": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time_ms": latency_ms,
            "success": success,
            "error": error
        })


def record_tool_call(
    tool_name: str,
    execution_time_ms: int,
    success: bool = True,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None
):
    """Record a tool execution metric.

    Args:
        tool_name: Name of the tool
        execution_time_ms: Execution time in milliseconds
        success: Whether the tool call succeeded
        message_id: Associated message ID
        error_message: Error message if failed
    """
    if _metrics_callback:
        _metrics_callback("tool_call", {
            "tool_name": tool_name,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "message_id": message_id,
            "error_message": error_message
        })


def record_enrichment(
    message_id: str,
    enrichment_time_ms: int,
    topics_extracted: int = 0,
    intent_detected: Optional[str] = None
):
    """Record message enrichment metrics.

    Args:
        message_id: The message ID
        enrichment_time_ms: Time to enrich the message
        topics_extracted: Number of topics extracted
        intent_detected: Detected intent
    """
    if _metrics_callback:
        _metrics_callback("response_time", {
            "operation": "enrichment",
            "message_id": message_id,
            "total_time_ms": enrichment_time_ms,
            "topics_extracted": topics_extracted,
            "intent_detected": intent_detected
        })


def record_retrieval(
    thread_id: str,
    retrieval_time_ms: int,
    messages_retrieved: int = 0,
    cache_hit: bool = False
):
    """Record context retrieval metrics.

    Args:
        thread_id: The thread ID
        retrieval_time_ms: Time to retrieve context
        messages_retrieved: Number of messages retrieved
        cache_hit: Whether the result was cached
    """
    if _metrics_callback:
        _metrics_callback("response_time", {
            "operation": "retrieval",
            "thread_id": thread_id,
            "total_time_ms": retrieval_time_ms,
            "messages_retrieved": messages_retrieved,
            "cache_hit": cache_hit
        })


# Initialize metrics connection with dashboard API
def _init_dashboard_metrics():
    """Initialize metrics callback to dashboard API."""
    try:
        from ..api.routes.dashboard import record_performance_metric
        set_metrics_callback(record_performance_metric)
    except ImportError:
        pass  # Dashboard not available


# Auto-initialize on import if dashboard is available
try:
    _init_dashboard_metrics()
except:
    pass
