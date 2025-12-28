"""Benchmark metrics definitions and calculations.

Metrics are organized into categories that matter for production AI memory:

1. Performance Metrics - Latency, QPS, throughput
2. Quality Metrics - Recall, precision, accuracy
3. Determinism Metrics - Replay variance, hash stability
4. Operational Metrics - Audit time, cost, drift rate
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricCategory(str, Enum):
    """Categories of benchmark metrics."""

    PERFORMANCE = "performance"
    QUALITY = "quality"
    DETERMINISM = "determinism"
    OPERATIONAL = "operational"
    COST = "cost"
    ROBUSTNESS = "robustness"


@dataclass
class LatencyMetrics:
    """Latency measurements with percentiles."""

    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p90(self) -> float:
        return self._percentile(90)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    def _percentile(self, p: int) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        k = (len(sorted_samples) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_samples) else f
        return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

    def add(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "median_ms": round(self.median, 3),
            "p50_ms": round(self.p50, 3),
            "p90_ms": round(self.p90, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "std_ms": round(self.std, 3),
        }


@dataclass
class QualityMetrics:
    """Quality metrics for retrieval."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_queries: int = 0

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class DeterminismMetrics:
    """Metrics for measuring determinism and reproducibility."""

    replay_runs: int = 0
    identical_results: int = 0
    result_hashes: list[str] = field(default_factory=list)
    memory_injection_variances: list[float] = field(default_factory=list)

    @property
    def replay_consistency(self) -> float:
        """Percentage of replays that produced identical results."""
        if self.replay_runs == 0:
            return 0.0
        return self.identical_results / self.replay_runs

    @property
    def hash_stability(self) -> float:
        """Measure of hash consistency across replays."""
        if len(self.result_hashes) < 2:
            return 1.0
        unique_hashes = len(set(self.result_hashes))
        return 1.0 / unique_hashes if unique_hashes > 0 else 0.0

    @property
    def avg_injection_variance(self) -> float:
        """Average variance in memory injection across replays."""
        if not self.memory_injection_variances:
            return 0.0
        return statistics.mean(self.memory_injection_variances)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_runs": self.replay_runs,
            "identical_results": self.identical_results,
            "replay_consistency": round(self.replay_consistency, 4),
            "hash_stability": round(self.hash_stability, 4),
            "unique_hashes": len(set(self.result_hashes)),
            "avg_injection_variance": round(self.avg_injection_variance, 6),
        }


@dataclass
class CostMetrics:
    """Metrics for operational cost measurement."""

    flr_queries: int = 0
    clst_queries: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    flr_latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    clst_latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    @property
    def flr_ratio(self) -> float:
        """Ratio of queries served by FLR (hot path)."""
        total = self.flr_queries + self.clst_queries
        return self.flr_queries / total if total > 0 else 0.0

    @property
    def avg_cost_per_query(self) -> float:
        """Estimated cost per query (relative units)."""
        # FLR is ~10x cheaper than CLST
        flr_cost = self.flr_queries * 1
        clst_cost = self.clst_queries * 10
        total = self.flr_queries + self.clst_queries
        return (flr_cost + clst_cost) / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flr_queries": self.flr_queries,
            "clst_queries": self.clst_queries,
            "flr_ratio": round(self.flr_ratio, 4),
            "avg_cost_per_query": round(self.avg_cost_per_query, 4),
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "flr_latency": self.flr_latency.to_dict(),
            "clst_latency": self.clst_latency.to_dict(),
        }


@dataclass
class DriftMetrics:
    """Metrics for measuring memory drift over time."""

    time_points: list[datetime] = field(default_factory=list)
    preference_stability_scores: list[float] = field(default_factory=list)
    correction_latencies: list[float] = field(default_factory=list)
    false_preference_promotions: int = 0
    total_preferences_tracked: int = 0

    @property
    def drift_rate(self) -> float:
        """Rate of preference drift over time."""
        if len(self.preference_stability_scores) < 2:
            return 0.0
        # Calculate rate of change
        changes = [
            abs(self.preference_stability_scores[i] - self.preference_stability_scores[i - 1])
            for i in range(1, len(self.preference_stability_scores))
        ]
        return statistics.mean(changes) if changes else 0.0

    @property
    def avg_correction_latency(self) -> float:
        """Average time to correct a preference after contradiction."""
        return statistics.mean(self.correction_latencies) if self.correction_latencies else 0.0

    @property
    def false_promotion_rate(self) -> float:
        """Rate of incorrectly promoted preferences."""
        if self.total_preferences_tracked == 0:
            return 0.0
        return self.false_preference_promotions / self.total_preferences_tracked

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_rate": round(self.drift_rate, 6),
            "avg_correction_latency_ms": round(self.avg_correction_latency, 3),
            "false_promotion_rate": round(self.false_promotion_rate, 6),
            "false_preference_promotions": self.false_preference_promotions,
            "total_preferences_tracked": self.total_preferences_tracked,
            "stability_scores": [round(s, 4) for s in self.preference_stability_scores[-10:]],
        }


@dataclass
class RobustnessMetrics:
    """Metrics for adversarial and failure robustness."""

    noisy_inputs: int = 0
    correctly_handled: int = 0
    memory_pollution_events: int = 0
    recovery_successes: int = 0
    recovery_failures: int = 0

    @property
    def noise_resistance(self) -> float:
        """Percentage of noisy inputs handled correctly."""
        return self.correctly_handled / self.noisy_inputs if self.noisy_inputs > 0 else 1.0

    @property
    def recovery_rate(self) -> float:
        """Rate of successful recovery from failures."""
        total = self.recovery_successes + self.recovery_failures
        return self.recovery_successes / total if total > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_resistance": round(self.noise_resistance, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "memory_pollution_events": self.memory_pollution_events,
            "noisy_inputs": self.noisy_inputs,
            "correctly_handled": self.correctly_handled,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""

    name: str
    category: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    # Sub-metrics
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    determinism: DeterminismMetrics = field(default_factory=DeterminismMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    drift: DriftMetrics = field(default_factory=DriftMetrics)
    robustness: RobustnessMetrics = field(default_factory=RobustnessMetrics)

    # Throughput
    total_operations: int = 0
    duration_seconds: float = 0.0

    @property
    def qps(self) -> float:
        """Queries per second."""
        return self.total_operations / self.duration_seconds if self.duration_seconds > 0 else 0.0

    def complete(self) -> None:
        """Mark benchmark as complete."""
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_operations": self.total_operations,
            "qps": round(self.qps, 2),
            "latency": self.latency.to_dict(),
            "quality": self.quality.to_dict(),
            "determinism": self.determinism.to_dict(),
            "cost": self.cost.to_dict(),
            "drift": self.drift.to_dict(),
            "robustness": self.robustness.to_dict(),
        }


def compute_result_hash(result: Any) -> str:
    """Compute a stable hash of a result for determinism checking."""
    if isinstance(result, dict):
        # Sort keys for consistent hashing
        serialized = str(sorted(result.items()))
    elif isinstance(result, list):
        serialized = str(sorted(str(item) for item in result))
    else:
        serialized = str(result)

    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class LatencyTimer:
    """Context manager for measuring latency."""

    def __init__(self, metrics: LatencyMetrics):
        self.metrics = metrics
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.metrics.add(elapsed_ms)
        return False
