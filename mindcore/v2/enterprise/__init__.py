"""Enterprise features for Mindcore.

This module provides production-ready features for enterprise deployments:
- Observability: OpenTelemetry-based metrics and tracing
- Rate Limiting: Configurable rate limits with multiple backends
- Audit Trail: Structured audit logging for compliance
- Encryption: At-rest encryption for sensitive memory content

Requirements (install with `pip install mindcore[enterprise]`):
    - opentelemetry-api>=1.20.0
    - opentelemetry-sdk>=1.20.0
    - limits>=3.0.0
    - cryptography>=41.0.0
    - structlog>=23.0.0

Example:
    from mindcore import Mindcore
    from mindcore.v2.enterprise import (
        ObservabilityConfig,
        RateLimiter,
        AuditLogger,
        EncryptionConfig,
        create_enterprise_mindcore,
    )

    # Quick setup with all enterprise features
    mc = create_enterprise_mindcore(
        storage="postgresql://localhost/mindcore",
        encryption_key="your-secret-key",
        rate_limit="100/minute",
        enable_tracing=True,
    )

    # Or configure individually
    mc = Mindcore(storage="postgresql://...")
    mc.enable_observability(ObservabilityConfig(service_name="my-service"))
    mc.enable_rate_limiting(RateLimiter(limit="1000/hour"))
    mc.enable_audit_logging(AuditLogger(output="file", path="/var/log/mindcore"))
    mc.enable_encryption(EncryptionConfig(key_from_env="MINDCORE_ENCRYPTION_KEY"))
"""

from .audit import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLogger,
)
from .encryption import (
    EncryptionConfig,
    EncryptionError,
    FieldEncryptor,
    KeyRotator,
)
from .observability import (
    MetricType,
    MindcoreMetrics,
    MindcoreTracer,
    ObservabilityConfig,
    SpanKind,
)
from .rate_limiting import (
    RateLimitBackend,
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError,
)


# Backwards compatibility alias
RateLimitExceeded = RateLimitExceededError


__all__ = [
    "AuditConfig",
    "AuditEvent",
    "AuditEventType",
    # Audit
    "AuditLogger",
    # Encryption
    "EncryptionConfig",
    "EncryptionError",
    "FieldEncryptor",
    "KeyRotator",
    "MetricType",
    "MindcoreMetrics",
    "MindcoreTracer",
    # Observability
    "ObservabilityConfig",
    "RateLimitBackend",
    "RateLimitConfig",
    "RateLimitExceeded",  # Backwards compatibility alias
    "RateLimitExceededError",
    # Rate Limiting
    "RateLimiter",
    "SpanKind",
]
