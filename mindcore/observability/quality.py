"""Quality monitoring for enrichment and context retrieval.

Tracks confidence scores, vocabulary match rates, and quality trends.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ConfidenceSample:
    """A single confidence score sample."""

    message_id: str
    confidence_score: float
    enrichment_source: str
    vocabulary_match_rate: float
    timestamp: float = field(default_factory=time.time)


class ConfidenceTracker:
    """Tracks confidence scores over time for quality monitoring.

    Provides:
    - Rolling statistics (avg, p50, p95, etc.)
    - Trend detection (improving/degrading)
    - Low confidence alerts
    """

    def __init__(
        self,
        window_size: int = 1000,
        low_confidence_threshold: float = 0.3,
        high_confidence_threshold: float = 0.7,
    ):
        """Initialize confidence tracker.

        Args:
            window_size: Number of samples to track
            low_confidence_threshold: Below this is "low confidence"
            high_confidence_threshold: Above this is "high confidence"
        """
        self._samples: deque[ConfidenceSample] = deque(maxlen=window_size)
        self._low_threshold = low_confidence_threshold
        self._high_threshold = high_confidence_threshold
        self._lock = threading.Lock()

        # Counters
        self._total_count = 0
        self._low_confidence_count = 0
        self._high_confidence_count = 0

    def record(
        self,
        message_id: str,
        confidence_score: float,
        enrichment_source: str,
        vocabulary_match_rate: float,
    ) -> None:
        """Record a confidence sample."""
        sample = ConfidenceSample(
            message_id=message_id,
            confidence_score=confidence_score,
            enrichment_source=enrichment_source,
            vocabulary_match_rate=vocabulary_match_rate,
        )

        with self._lock:
            self._samples.append(sample)
            self._total_count += 1

            if confidence_score < self._low_threshold:
                self._low_confidence_count += 1
            elif confidence_score >= self._high_threshold:
                self._high_confidence_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get confidence statistics."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "avg": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p95": 0,
                    "low_confidence_rate": 0,
                    "high_confidence_rate": 0,
                }

            scores = [s.confidence_score for s in self._samples]
            sorted_scores = sorted(scores)
            n = len(sorted_scores)

            return {
                "count": n,
                "total_processed": self._total_count,
                "avg": sum(scores) / n,
                "min": sorted_scores[0],
                "max": sorted_scores[-1],
                "p50": sorted_scores[n // 2],
                "p95": sorted_scores[int(n * 0.95)] if n >= 20 else sorted_scores[-1],
                "p99": sorted_scores[int(n * 0.99)] if n >= 100 else sorted_scores[-1],
                "low_confidence_rate": (
                    self._low_confidence_count / self._total_count
                    if self._total_count > 0 else 0
                ),
                "high_confidence_rate": (
                    self._high_confidence_count / self._total_count
                    if self._total_count > 0 else 0
                ),
            }

    def get_trend(self, window: int = 100) -> dict[str, Any]:
        """Analyze confidence trend.

        Compares recent samples to older samples to detect
        improving or degrading quality.

        Args:
            window: Number of samples for each window

        Returns:
            Trend analysis with direction and magnitude
        """
        with self._lock:
            samples = list(self._samples)

        if len(samples) < window * 2:
            return {
                "trend": "insufficient_data",
                "recent_avg": 0,
                "older_avg": 0,
                "change": 0,
            }

        recent = samples[-window:]
        older = samples[-(window * 2) : -window]

        recent_avg = sum(s.confidence_score for s in recent) / len(recent)
        older_avg = sum(s.confidence_score for s in older) / len(older)

        change = recent_avg - older_avg
        change_pct = (change / older_avg * 100) if older_avg > 0 else 0

        if change_pct > 5:
            trend = "improving"
        elif change_pct < -5:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "change": change,
            "change_pct": change_pct,
        }

    def get_low_confidence_samples(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent low confidence samples for review.

        Args:
            limit: Maximum samples to return

        Returns:
            List of low confidence samples
        """
        with self._lock:
            low_samples = [
                s for s in self._samples
                if s.confidence_score < self._low_threshold
            ]

        # Return most recent
        result = []
        for sample in reversed(low_samples[-limit:]):
            result.append({
                "message_id": sample.message_id,
                "confidence_score": sample.confidence_score,
                "enrichment_source": sample.enrichment_source,
                "vocabulary_match_rate": sample.vocabulary_match_rate,
                "timestamp": sample.timestamp,
            })

        return result


class EnrichmentQualityMonitor:
    """Monitors enrichment quality across multiple dimensions.

    Tracks:
    - Confidence scores
    - Vocabulary match rates
    - Enrichment sources
    - Processing latencies
    """

    def __init__(
        self,
        window_size: int = 1000,
        low_confidence_threshold: float = 0.3,
    ):
        """Initialize quality monitor.

        Args:
            window_size: Number of samples to track
            low_confidence_threshold: Threshold for low confidence
        """
        self._confidence_tracker = ConfidenceTracker(
            window_size=window_size,
            low_confidence_threshold=low_confidence_threshold,
        )
        self._source_counts: dict[str, int] = {}
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._vocab_match_rates: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(
        self,
        message_id: str,
        confidence_score: float,
        enrichment_source: str,
        vocabulary_match_rate: float,
        latency_ms: float,
    ) -> None:
        """Record an enrichment result."""
        self._confidence_tracker.record(
            message_id=message_id,
            confidence_score=confidence_score,
            enrichment_source=enrichment_source,
            vocabulary_match_rate=vocabulary_match_rate,
        )

        with self._lock:
            self._source_counts[enrichment_source] = (
                self._source_counts.get(enrichment_source, 0) + 1
            )
            self._latencies.append(latency_ms)
            self._vocab_match_rates.append(vocabulary_match_rate)

    def get_report(self) -> dict[str, Any]:
        """Get comprehensive quality report."""
        confidence_stats = self._confidence_tracker.get_stats()
        trend = self._confidence_tracker.get_trend()

        with self._lock:
            latencies = list(self._latencies)
            vocab_rates = list(self._vocab_match_rates)
            sources = dict(self._source_counts)

        # Calculate latency stats
        latency_stats = {}
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            latency_stats = {
                "avg_ms": sum(latencies) / n,
                "p50_ms": sorted_latencies[n // 2],
                "p95_ms": sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
                "p99_ms": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            }

        # Calculate vocab match rate stats
        vocab_stats = {}
        if vocab_rates:
            vocab_stats = {
                "avg": sum(vocab_rates) / len(vocab_rates),
                "min": min(vocab_rates),
            }

        # Calculate source distribution
        total = sum(sources.values()) if sources else 0
        source_distribution = {
            k: {"count": v, "pct": v / total * 100 if total > 0 else 0}
            for k, v in sources.items()
        }

        return {
            "confidence": confidence_stats,
            "confidence_trend": trend,
            "latency": latency_stats,
            "vocabulary_match_rate": vocab_stats,
            "enrichment_sources": source_distribution,
            "low_confidence_samples": self._confidence_tracker.get_low_confidence_samples(5),
        }

    def get_health_check(self) -> dict[str, Any]:
        """Get a quick health check result.

        Returns:
            Health status with any issues detected
        """
        stats = self._confidence_tracker.get_stats()
        trend = self._confidence_tracker.get_trend()

        issues = []

        if stats["low_confidence_rate"] > 0.2:
            issues.append({
                "type": "high_low_confidence_rate",
                "value": stats["low_confidence_rate"],
                "threshold": 0.2,
                "message": f"Low confidence rate is {stats['low_confidence_rate']:.1%}",
            })

        if trend["trend"] == "degrading" and abs(trend["change_pct"]) > 10:
            issues.append({
                "type": "degrading_quality",
                "value": trend["change_pct"],
                "message": f"Confidence trending down by {abs(trend['change_pct']):.1f}%",
            })

        with self._lock:
            vocab_rates = list(self._vocab_match_rates)
            if vocab_rates and sum(vocab_rates) / len(vocab_rates) < 0.7:
                avg_rate = sum(vocab_rates) / len(vocab_rates)
                issues.append({
                    "type": "low_vocabulary_match",
                    "value": avg_rate,
                    "threshold": 0.7,
                    "message": f"Vocabulary match rate is {avg_rate:.1%}",
                })

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "stats": {
                "avg_confidence": stats["avg"],
                "low_confidence_rate": stats["low_confidence_rate"],
                "trend": trend["trend"],
            },
        }
