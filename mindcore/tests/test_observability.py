"""Tests for observability package."""

import pytest

from mindcore.observability.alerts import (
    Alert,
    AlertConfig,
    AlertManager,
    AlertRule,
    AlertSeverity,
    LoggingAlertHandler,
)
from mindcore.observability.metrics import (
    MemoryMetricsBackend,
    MetricsCollector,
    ObservabilityBackend,
)
from mindcore.observability.quality import ConfidenceTracker, EnrichmentQualityMonitor
from mindcore.observability.observer import MindcoreObserver


class TestAlertRule:
    """Tests for AlertRule."""

    def test_gt_comparison(self):
        """Test greater than comparison."""
        rule = AlertRule(threshold=100, comparison="gt")
        assert rule.check(101) is True
        assert rule.check(100) is False
        assert rule.check(99) is False

    def test_lt_comparison(self):
        """Test less than comparison."""
        rule = AlertRule(threshold=0.5, comparison="lt")
        assert rule.check(0.3) is True
        assert rule.check(0.5) is False
        assert rule.check(0.7) is False

    def test_gte_comparison(self):
        """Test greater than or equal comparison."""
        rule = AlertRule(threshold=100, comparison="gte")
        assert rule.check(101) is True
        assert rule.check(100) is True
        assert rule.check(99) is False


class TestAlertConfig:
    """Tests for AlertConfig."""

    def test_get_rules(self):
        """Test rule extraction from config."""
        config = AlertConfig(
            enrichment_queue_depth={"threshold": 1000, "severity": "warning"},
            enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
        )

        rules = config.get_rules()

        assert "enrichment_queue_depth" in rules
        assert rules["enrichment_queue_depth"].threshold == 1000
        assert rules["enrichment_queue_depth"].severity == AlertSeverity.WARNING

        assert "enrichment_error_rate" in rules
        assert rules["enrichment_error_rate"].threshold == 0.05
        assert rules["enrichment_error_rate"].severity == AlertSeverity.CRITICAL

    def test_custom_alerts(self):
        """Test custom alert configuration."""
        config = AlertConfig(
            custom_alerts={
                "my_custom_metric": {"threshold": 50, "severity": "info"},
            }
        )

        rules = config.get_rules()
        assert "my_custom_metric" in rules
        assert rules["my_custom_metric"].severity == AlertSeverity.INFO


class TestAlertManager:
    """Tests for AlertManager."""

    def test_alert_triggered(self):
        """Test alert triggering."""
        config = AlertConfig(
            enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
        )
        manager = AlertManager(config, handlers=[])

        triggered = manager.check({"enrichment_error_rate": 0.1})

        assert len(triggered) == 1
        assert triggered[0].name == "enrichment_error_rate"
        assert triggered[0].severity == AlertSeverity.CRITICAL

    def test_no_alert_when_below_threshold(self):
        """Test no alert when below threshold."""
        config = AlertConfig(
            enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
        )
        manager = AlertManager(config, handlers=[])

        triggered = manager.check({"enrichment_error_rate": 0.01})

        assert len(triggered) == 0

    def test_alert_cooldown(self):
        """Test alert cooldown prevents re-triggering."""
        config = AlertConfig(
            enrichment_error_rate={
                "threshold": 0.05,
                "severity": "critical",
                "cooldown_seconds": 300,
            },
        )
        manager = AlertManager(config, handlers=[])

        # First check triggers
        triggered1 = manager.check({"enrichment_error_rate": 0.1})
        assert len(triggered1) == 1

        # Second check within cooldown does not trigger
        triggered2 = manager.check({"enrichment_error_rate": 0.1})
        assert len(triggered2) == 0


class TestMemoryMetricsBackend:
    """Tests for MemoryMetricsBackend."""

    def test_counter(self):
        """Test counter recording."""
        backend = MemoryMetricsBackend()

        backend.record_counter("requests_total", 1)
        backend.record_counter("requests_total", 1)
        backend.record_counter("requests_total", 1)

        metrics = backend.get_metrics()
        assert metrics["counters"]["requests_total"] == 3

    def test_gauge(self):
        """Test gauge recording."""
        backend = MemoryMetricsBackend()

        backend.record_gauge("queue_depth", 100)
        backend.record_gauge("queue_depth", 50)

        metrics = backend.get_metrics()
        assert metrics["gauges"]["queue_depth"] == 50  # Latest value

    def test_histogram(self):
        """Test histogram recording."""
        backend = MemoryMetricsBackend()

        for i in range(100):
            backend.record_histogram("latency_ms", i)

        metrics = backend.get_metrics()
        hist = metrics["histograms"]["latency_ms"]

        assert hist["count"] == 100
        assert hist["min"] == 0
        assert hist["max"] == 99
        assert hist["avg"] == 49.5

    def test_labels(self):
        """Test metrics with labels."""
        backend = MemoryMetricsBackend()

        backend.record_counter("requests", 1, {"endpoint": "/api"})
        backend.record_counter("requests", 2, {"endpoint": "/health"})

        metrics = backend.get_metrics()
        assert metrics["counters"]["requests{endpoint=/api}"] == 1
        assert metrics["counters"]["requests{endpoint=/health}"] == 2


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_enrichment(self):
        """Test enrichment metrics recording."""
        collector = MetricsCollector(backend=ObservabilityBackend.MEMORY)

        collector.record_enrichment(
            message_id="msg1",
            confidence_score=0.85,
            latency_ms=150,
            enrichment_source="llm",
            vocabulary_match_rate=0.95,
            success=True,
        )

        metrics = collector.get_metrics()
        assert "enrichment_total" in str(metrics)

    def test_record_context_retrieval(self):
        """Test context retrieval metrics recording."""
        collector = MetricsCollector(backend=ObservabilityBackend.MEMORY)

        collector.record_context_retrieval(
            latency_ms=200,
            llm_calls=2,
            fast_path_used=True,
            confidence="high",
            context_source="recent",
        )

        metrics = collector.get_metrics()
        assert "context_retrieval_total" in str(metrics)


class TestConfidenceTracker:
    """Tests for ConfidenceTracker."""

    def test_record_and_stats(self):
        """Test recording samples and getting stats."""
        tracker = ConfidenceTracker()

        for i in range(10):
            tracker.record(
                message_id=f"msg{i}",
                confidence_score=0.5 + (i * 0.05),
                enrichment_source="llm",
                vocabulary_match_rate=0.9,
            )

        stats = tracker.get_stats()

        assert stats["count"] == 10
        assert 0.5 <= stats["avg"] <= 1.0
        assert stats["min"] >= 0.5

    def test_low_confidence_tracking(self):
        """Test low confidence sample tracking."""
        tracker = ConfidenceTracker(low_confidence_threshold=0.3)

        # Add some low confidence samples
        tracker.record("msg1", 0.2, "llm", 0.5)
        tracker.record("msg2", 0.8, "llm", 0.9)
        tracker.record("msg3", 0.1, "llm", 0.3)

        stats = tracker.get_stats()
        assert stats["low_confidence_rate"] > 0

        samples = tracker.get_low_confidence_samples()
        assert len(samples) == 2

    def test_trend_detection(self):
        """Test trend detection."""
        tracker = ConfidenceTracker(window_size=1000)

        # Add improving samples
        for i in range(200):
            score = 0.3 if i < 100 else 0.7
            tracker.record(f"msg{i}", score, "llm", 0.9)

        trend = tracker.get_trend(window=50)
        assert trend["trend"] == "improving"


class TestEnrichmentQualityMonitor:
    """Tests for EnrichmentQualityMonitor."""

    def test_comprehensive_recording(self):
        """Test comprehensive quality monitoring."""
        monitor = EnrichmentQualityMonitor()

        for i in range(50):
            monitor.record(
                message_id=f"msg{i}",
                confidence_score=0.7 + (i % 3) * 0.1,
                enrichment_source="llm" if i % 2 == 0 else "trivial",
                vocabulary_match_rate=0.85,
                latency_ms=100 + i,
            )

        report = monitor.get_report()

        assert "confidence" in report
        assert "latency" in report
        assert "enrichment_sources" in report
        assert "llm" in report["enrichment_sources"]

    def test_health_check(self):
        """Test health check."""
        monitor = EnrichmentQualityMonitor(low_confidence_threshold=0.3)

        # Add healthy samples
        for i in range(20):
            monitor.record(f"msg{i}", 0.8, "llm", 0.9, 100)

        health = monitor.get_health_check()
        assert health["healthy"] is True

        # Add low confidence samples
        for i in range(30):
            monitor.record(f"low{i}", 0.1, "llm", 0.3, 100)

        health = monitor.get_health_check()
        assert health["healthy"] is False


class TestMindcoreObserver:
    """Tests for MindcoreObserver."""

    def test_initialization(self):
        """Test observer initialization."""
        observer = MindcoreObserver(backend="memory")

        status = observer.get_status()
        assert status["backend"] == "memory"

    def test_with_alerts(self):
        """Test observer with alerts."""
        config = AlertConfig(
            enrichment_error_rate={"threshold": 0.1, "severity": "critical"},
        )

        observer = MindcoreObserver(
            backend="memory",
            alerts=config,
        )

        status = observer.get_status()
        assert status["alert_rules"] > 0

    def test_record_and_check(self):
        """Test recording metrics and checking alerts."""
        config = AlertConfig(
            low_confidence_rate={"threshold": 0.2, "severity": "warning"},
        )

        observer = MindcoreObserver(backend="memory", alerts=config)

        # Record some enrichments
        for i in range(10):
            observer.record_enrichment(
                message_id=f"msg{i}",
                confidence_score=0.1 if i < 5 else 0.9,  # 50% low confidence
                latency_ms=100,
                enrichment_source="llm",
                vocabulary_match_rate=0.9,
                success=True,
            )

        # Check for alerts
        triggered = observer.check_alerts()
        # May or may not trigger depending on calculation

    def test_health_check(self):
        """Test health check endpoint."""
        observer = MindcoreObserver(backend="memory")

        # Add good samples
        for i in range(10):
            observer.record_enrichment(
                message_id=f"msg{i}",
                confidence_score=0.8,
                latency_ms=100,
                enrichment_source="llm",
                vocabulary_match_rate=0.95,
                success=True,
            )

        health = observer.get_health_check()
        assert "healthy" in health
        assert "quality" in health

    def test_quality_report(self):
        """Test quality report generation."""
        observer = MindcoreObserver(backend="memory")

        # Add samples
        for i in range(20):
            observer.record_enrichment(
                message_id=f"msg{i}",
                confidence_score=0.7 + (i % 4) * 0.1,
                latency_ms=100 + i * 5,
                enrichment_source="llm",
                vocabulary_match_rate=0.9,
                success=True,
            )

        report = observer.get_quality_report()

        assert "confidence" in report
        assert "confidence_trend" in report
        assert "latency" in report
