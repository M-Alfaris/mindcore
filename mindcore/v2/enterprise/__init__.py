"""Enterprise features for Mindcore.

This module provides production-ready features for enterprise deployments:
- Observability: OpenTelemetry-based metrics and tracing
- Rate Limiting: Configurable rate limits with multiple backends
- Audit Trail: Structured audit logging for compliance
- Encryption: At-rest encryption for sensitive memory content
- Compliance: GDPR/CCPA compliance tools with data retention

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
        ComplianceManager,
        RetentionPolicy,
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

    # GDPR/CCPA compliance
    compliance = mc.get_compliance_manager()
    export = compliance.export_user_data("user_123")
    compliance.delete_user_data("user_123")
"""

from .audit import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLogger,
)
from .compliance import (
    AnonymizationResult,
    AnonymizationStrategy,
    ComplianceEventType,
    ComplianceManager,
    GDPRDeleteResult,
    GDPRExportResult,
    RetentionEnforcementResult,
    RetentionPolicy,
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
    # Compliance
    "AnonymizationResult",
    "AnonymizationStrategy",
    "AuditConfig",
    "AuditEvent",
    "AuditEventType",
    # Audit
    "AuditLogger",
    "ComplianceEventType",
    "ComplianceManager",
    # Encryption
    "EncryptionConfig",
    "EncryptionError",
    "FieldEncryptor",
    "GDPRDeleteResult",
    "GDPRExportResult",
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
    "RetentionEnforcementResult",
    "RetentionPolicy",
    "SpanKind",
]
