"""Alert configuration and handlers for Mindcore observability.

Provides flexible alerting with multiple severity levels and handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a triggered alert."""

    name: str
    message: str
    severity: AlertSeverity
    value: float
    threshold: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "value": self.value,
            "threshold": self.threshold,
            "triggered_at": self.triggered_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """Configuration for a single alert rule."""

    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    comparison: str = "gt"  # gt, lt, gte, lte, eq
    cooldown_seconds: int = 300  # Don't re-alert within this window
    description: str = ""

    def check(self, value: float) -> bool:
        """Check if value triggers this alert."""
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        return False


@dataclass
class AlertConfig:
    """Configuration for all alert rules.

    Example:
        config = AlertConfig(
            enrichment_queue_depth={"threshold": 1000, "severity": "warning"},
            enrichment_error_rate={"threshold": 0.05, "severity": "critical"},
            low_confidence_rate={"threshold": 0.2, "severity": "warning"},
            context_latency_p99={"threshold": 5000, "severity": "warning"},
        )
    """

    # Enrichment alerts
    enrichment_queue_depth: dict[str, Any] | None = None
    enrichment_error_rate: dict[str, Any] | None = None
    enrichment_latency_p99: dict[str, Any] | None = None

    # Confidence alerts
    low_confidence_rate: dict[str, Any] | None = None
    average_confidence: dict[str, Any] | None = None

    # Context retrieval alerts
    context_latency_p99: dict[str, Any] | None = None
    context_fallback_rate: dict[str, Any] | None = None

    # Worker alerts
    worker_error_rate: dict[str, Any] | None = None
    worker_idle_time: dict[str, Any] | None = None

    # Custom alerts
    custom_alerts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_rules(self) -> dict[str, AlertRule]:
        """Convert config to AlertRule objects."""
        rules = {}

        # Process built-in alerts
        alert_defs = [
            ("enrichment_queue_depth", self.enrichment_queue_depth, "gt"),
            ("enrichment_error_rate", self.enrichment_error_rate, "gt"),
            ("enrichment_latency_p99", self.enrichment_latency_p99, "gt"),
            ("low_confidence_rate", self.low_confidence_rate, "gt"),
            ("average_confidence", self.average_confidence, "lt"),
            ("context_latency_p99", self.context_latency_p99, "gt"),
            ("context_fallback_rate", self.context_fallback_rate, "gt"),
            ("worker_error_rate", self.worker_error_rate, "gt"),
            ("worker_idle_time", self.worker_idle_time, "gt"),
        ]

        for name, config, default_comparison in alert_defs:
            if config:
                rules[name] = AlertRule(
                    threshold=config.get("threshold", 0),
                    severity=AlertSeverity(config.get("severity", "warning")),
                    comparison=config.get("comparison", default_comparison),
                    cooldown_seconds=config.get("cooldown_seconds", 300),
                    description=config.get("description", ""),
                )

        # Process custom alerts
        for name, config in self.custom_alerts.items():
            rules[name] = AlertRule(
                threshold=config.get("threshold", 0),
                severity=AlertSeverity(config.get("severity", "warning")),
                comparison=config.get("comparison", "gt"),
                cooldown_seconds=config.get("cooldown_seconds", 300),
                description=config.get("description", ""),
            )

        return rules


class AlertHandler(ABC):
    """Base class for alert handlers."""

    @abstractmethod
    def handle(self, alert: Alert) -> None:
        """Handle a triggered alert."""

    @abstractmethod
    def handle_resolved(self, alert_name: str) -> None:
        """Handle an alert being resolved."""


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts."""

    def handle(self, alert: Alert) -> None:
        """Log the alert."""
        log_method = logger.warning if alert.severity == AlertSeverity.WARNING else logger.error
        if alert.severity == AlertSeverity.INFO:
            log_method = logger.info

        log_method(
            f"[ALERT:{alert.severity.value.upper()}] {alert.name}: {alert.message}",
            value=alert.value,
            threshold=alert.threshold,
            metadata=alert.metadata,
        )

    def handle_resolved(self, alert_name: str) -> None:
        """Log alert resolution."""
        logger.info(f"[ALERT:RESOLVED] {alert_name}")


class WebhookAlertHandler(AlertHandler):
    """Alert handler that sends webhooks."""

    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        """Initialize webhook handler.

        Args:
            webhook_url: URL to POST alerts to
            headers: Optional headers to include
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}

    def handle(self, alert: Alert) -> None:
        """Send alert to webhook."""
        import json
        import urllib.request

        try:
            data = json.dumps(alert.to_dict()).encode("utf-8")
            headers = {"Content-Type": "application/json", **self.headers}
            req = urllib.request.Request(self.webhook_url, data=data, headers=headers)
            urllib.request.urlopen(req, timeout=10)
            logger.debug(f"Alert sent to webhook: {alert.name}")
        except Exception as e:
            logger.error(f"Failed to send alert webhook: {e}")

    def handle_resolved(self, alert_name: str) -> None:
        """Send resolution to webhook."""
        import json
        import urllib.request

        try:
            data = json.dumps({"name": alert_name, "status": "resolved"}).encode("utf-8")
            headers = {"Content-Type": "application/json", **self.headers}
            req = urllib.request.Request(self.webhook_url, data=data, headers=headers)
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send resolution webhook: {e}")


class AlertManager:
    """Manages alert rules, state, and handlers."""

    def __init__(
        self,
        config: AlertConfig,
        handlers: list[AlertHandler] | None = None,
    ):
        """Initialize alert manager.

        Args:
            config: Alert configuration
            handlers: List of alert handlers (defaults to LoggingAlertHandler)
        """
        self.rules = config.get_rules()
        self.handlers = handlers or [LoggingAlertHandler()]
        self._active_alerts: dict[str, Alert] = {}
        self._last_triggered: dict[str, datetime] = {}

    def check(self, metrics: dict[str, float]) -> list[Alert]:
        """Check metrics against alert rules.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            List of triggered alerts
        """
        triggered = []
        now = datetime.now(timezone.utc)

        for name, rule in self.rules.items():
            if name not in metrics:
                continue

            value = metrics[name]

            if rule.check(value):
                # Check cooldown
                last = self._last_triggered.get(name)
                if last and (now - last).total_seconds() < rule.cooldown_seconds:
                    continue

                alert = Alert(
                    name=name,
                    message=f"{name} is {value:.2f} (threshold: {rule.threshold})",
                    severity=rule.severity,
                    value=value,
                    threshold=rule.threshold,
                )

                self._active_alerts[name] = alert
                self._last_triggered[name] = now
                triggered.append(alert)

                # Notify handlers
                for handler in self.handlers:
                    try:
                        handler.handle(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")

            elif name in self._active_alerts:
                # Alert resolved
                del self._active_alerts[name]
                for handler in self.handlers:
                    try:
                        handler.handle_resolved(name)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")

        return triggered

    @property
    def active_alerts(self) -> list[Alert]:
        """Get list of currently active alerts."""
        return list(self._active_alerts.values())

    def get_status(self) -> dict[str, Any]:
        """Get alert manager status."""
        return {
            "rules_count": len(self.rules),
            "active_alerts": len(self._active_alerts),
            "alerts": [a.to_dict() for a in self._active_alerts.values()],
        }
