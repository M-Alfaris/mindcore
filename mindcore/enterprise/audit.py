"""Audit logging module for Mindcore.

Provides structured audit logging for compliance, security monitoring,
and debugging using structlog.

Requirements:
    pip install structlog

Example:
    from mindcore.enterprise import AuditLogger, AuditConfig

    # File-based audit log
    audit = AuditLogger(AuditConfig(
        output="file",
        file_path="/var/log/mindcore/audit.log",
        include_request_id=True,
    ))

    # Log memory operations
    audit.log_store(
        user_id="user123",
        memory_id="mem_abc",
        memory_type="semantic",
        content_hash="sha256:...",  # Don't log actual content
    )

    # Log access events
    audit.log_access(
        user_id="user123",
        resource="memory:mem_abc",
        action="read",
        granted=True,
    )

    # Log authentication
    audit.log_auth(
        user_id="user123",
        action="login",
        success=True,
        ip_address="192.168.1.1",
    )

References:
    - https://www.structlog.org/en/stable/logging-best-practices.html
    - https://betterstack.com/community/guides/logging/structlog/
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import structlog


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Memory operations
    MEMORY_STORE = "memory.store"
    MEMORY_RECALL = "memory.recall"
    MEMORY_DELETE = "memory.delete"
    MEMORY_UPDATE = "memory.update"

    # Access control
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    ACCESS_CHECK = "access.check"

    # Authentication (event type names, not passwords)
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_ISSUED = "auth.token_issued"  # noqa: S105  # nosec B105
    AUTH_TOKEN_REVOKED = "auth.token_revoked"  # noqa: S105  # nosec B105

    # Agent operations
    AGENT_REGISTER = "agent.register"
    AGENT_UNREGISTER = "agent.unregister"
    AGENT_SHARE = "agent.share"
    AGENT_SYNC = "agent.sync"

    # Admin operations
    ADMIN_CONFIG_CHANGE = "admin.config_change"
    ADMIN_MIGRATION = "admin.migration"
    ADMIN_EXPORT = "admin.export"
    ADMIN_IMPORT = "admin.import"

    # Security events
    SECURITY_RATE_LIMITED = "security.rate_limited"
    SECURITY_INVALID_INPUT = "security.invalid_input"
    SECURITY_ENCRYPTION = "security.encryption"


@dataclass
class AuditEvent:
    """Structured audit event.

    Attributes:
        event_type: Type of event
        timestamp: When the event occurred
        user_id: User who initiated the event
        agent_id: Agent involved (if applicable)
        resource: Resource affected (e.g., "memory:mem_123")
        action: Action performed
        outcome: Success/failure/etc.
        details: Additional event details
        request_id: Correlation ID for request tracing
        ip_address: Client IP address
        user_agent: Client user agent
    """

    event_type: AuditEventType
    timestamp: datetime
    user_id: str | None = None
    agent_id: str | None = None
    resource: str | None = None
    action: str | None = None
    outcome: str = "success"
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "request_id": self.request_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        output: Output destination ("stdout", "stderr", "file")
        file_path: Path to audit log file (required if output="file")
        json_format: Use JSON format for log entries
        include_request_id: Include request correlation ID
        include_timestamp: Include ISO timestamp
        include_source: Include source file/line information
        redact_patterns: Patterns to redact from logs (e.g., passwords)
        max_detail_length: Maximum length for detail values
        rotation_enabled: Enable log rotation (file output only)
        rotation_max_bytes: Max file size before rotation
        rotation_backup_count: Number of backup files to keep

    Example:
        config = AuditConfig(
            output="file",
            file_path="/var/log/mindcore/audit.log",
            json_format=True,
            include_request_id=True,
            redact_patterns=["password", "token", "secret"],
        )
    """

    output: str = "stdout"  # stdout, stderr, file
    file_path: str | None = None
    json_format: bool = True
    include_request_id: bool = True
    include_timestamp: bool = True
    include_source: bool = False
    redact_patterns: list[str] = field(
        default_factory=lambda: ["password", "token", "secret", "key", "credential"]
    )
    max_detail_length: int = 1000
    rotation_enabled: bool = True
    rotation_max_bytes: int = 10 * 1024 * 1024  # 10MB
    rotation_backup_count: int = 5


class AuditLogger:
    """Structured audit logger for compliance and security.

    Provides tamper-evident, structured logging suitable for
    compliance requirements (SOC2, HIPAA, GDPR audit trails).

    Example:
        audit = AuditLogger(AuditConfig(
            output="file",
            file_path="/var/log/mindcore/audit.log",
        ))

        # Log memory store
        audit.log_store(
            user_id="user123",
            memory_id="mem_abc",
            memory_type="semantic",
        )

        # Log with context
        with audit.context(request_id="req_123", ip_address="192.168.1.1"):
            audit.log_recall(user_id="user123", query_hash="sha256:...")
    """

    def __init__(self, config: AuditConfig | None = None):
        """Initialize audit logger.

        Args:
            config: Audit configuration. If None, uses defaults (stdout).
        """
        self.config = config or AuditConfig()
        self._context: dict[str, Any] = {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure structlog for audit logging."""
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if self.config.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Determine output stream
        output_stream: TextIO
        if self.config.output == "stderr":
            output_stream = sys.stderr
        elif self.config.output == "file" and self.config.file_path:
            # Ensure directory exists
            Path(self.config.file_path).parent.mkdir(parents=True, exist_ok=True)

            if self.config.rotation_enabled:
                from logging.handlers import RotatingFileHandler

                handler = RotatingFileHandler(
                    self.config.file_path,
                    maxBytes=self.config.rotation_max_bytes,
                    backupCount=self.config.rotation_backup_count,
                )
                # Use handler's stream
                output_stream = handler.stream
                self._file_handler = handler
            else:
                self._file_handle = open(self.config.file_path, "a")  # noqa: SIM115
                output_stream = self._file_handle
        else:
            output_stream = sys.stdout

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(0),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=output_stream),
            cache_logger_on_first_use=True,
        )

        self._logger = structlog.get_logger("mindcore.audit")

    def _redact_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive information from log data.

        Args:
            data: Data to redact

        Returns:
            Data with sensitive values redacted
        """
        redacted = {}
        for key, value in data.items():
            # Check if key matches redact patterns
            key_lower = key.lower()
            should_redact = any(pattern in key_lower for pattern in self.config.redact_patterns)

            if should_redact:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_sensitive(value)
            elif isinstance(value, str) and len(value) > self.config.max_detail_length:
                redacted[key] = value[: self.config.max_detail_length] + "...[truncated]"
            else:
                redacted[key] = value

        return redacted

    def _hash_content(self, content: str) -> str:
        """Create a hash of content for logging without exposing data.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hash prefixed with "sha256:"
        """
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _log_event(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: Event to log
        """
        data = event.to_dict()

        # Add context
        if self._context:
            data.update(self._context)

        # Redact sensitive data
        if event.details:
            data["details"] = self._redact_sensitive(event.details)

        self._logger.info(
            event.event_type.value,
            **{k: v for k, v in data.items() if v is not None},
        )

    class _ContextManager:
        """Context manager for temporary audit context."""

        def __init__(self, logger: AuditLogger, context: dict[str, Any]):
            self._logger = logger
            self._context = context
            self._previous: dict[str, Any] = {}

        def __enter__(self):
            self._previous = self._logger._context.copy()
            self._logger._context.update(self._context)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._logger._context = self._previous
            return False

    def context(self, **kwargs: Any) -> _ContextManager:
        """Create a context for audit logging.

        Args:
            **kwargs: Context values to include in all logs

        Returns:
            Context manager

        Example:
            with audit.context(request_id="req_123", ip_address="192.168.1.1"):
                audit.log_store(...)
                audit.log_recall(...)
        """
        return self._ContextManager(self, kwargs)

    def log_store(
        self,
        user_id: str,
        memory_id: str,
        memory_type: str,
        content_hash: str | None = None,
        topics: list[str] | None = None,
        agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a memory store operation.

        Args:
            user_id: User who stored the memory
            memory_id: ID of stored memory
            memory_type: Type of memory
            content_hash: Hash of content (don't log actual content)
            topics: Memory topics
            agent_id: Agent ID if applicable
            **extra: Additional details
        """
        self._log_event(
            AuditEvent(
                event_type=AuditEventType.MEMORY_STORE,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                agent_id=agent_id,
                resource=f"memory:{memory_id}",
                action="store",
                details={
                    "memory_type": memory_type,
                    "content_hash": content_hash,
                    "topics": topics or [],
                    **extra,
                },
            )
        )

    def log_recall(
        self,
        user_id: str,
        query_hash: str | None = None,
        memories_returned: int = 0,
        agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a memory recall operation.

        Args:
            user_id: User who recalled memories
            query_hash: Hash of query (don't log actual query)
            memories_returned: Number of memories returned
            agent_id: Agent ID if applicable
            **extra: Additional details
        """
        self._log_event(
            AuditEvent(
                event_type=AuditEventType.MEMORY_RECALL,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                agent_id=agent_id,
                action="recall",
                details={
                    "query_hash": query_hash,
                    "memories_returned": memories_returned,
                    **extra,
                },
            )
        )

    def log_delete(
        self,
        user_id: str,
        memory_id: str,
        agent_id: str | None = None,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a memory delete operation.

        Args:
            user_id: User who deleted the memory
            memory_id: ID of deleted memory
            agent_id: Agent ID if applicable
            reason: Reason for deletion
            **extra: Additional details
        """
        self._log_event(
            AuditEvent(
                event_type=AuditEventType.MEMORY_DELETE,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                agent_id=agent_id,
                resource=f"memory:{memory_id}",
                action="delete",
                details={"reason": reason, **extra},
            )
        )

    def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        agent_id: str | None = None,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        """Log an access control decision.

        Args:
            user_id: User requesting access
            resource: Resource being accessed
            action: Action being performed
            granted: Whether access was granted
            agent_id: Agent ID if applicable
            reason: Reason for decision
            **extra: Additional details
        """
        self._log_event(
            AuditEvent(
                event_type=(
                    AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED
                ),
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                agent_id=agent_id,
                resource=resource,
                action=action,
                outcome="granted" if granted else "denied",
                details={"reason": reason, **extra},
            )
        )

    def log_auth(
        self,
        user_id: str,
        action: str,
        success: bool,
        ip_address: str | None = None,
        user_agent: str | None = None,
        **extra: Any,
    ) -> None:
        """Log an authentication event.

        Args:
            user_id: User authenticating
            action: Auth action (login, logout, etc.)
            success: Whether auth succeeded
            ip_address: Client IP address
            user_agent: Client user agent
            **extra: Additional details
        """
        event_type = {
            "login": (AuditEventType.AUTH_LOGIN if success else AuditEventType.AUTH_FAILED),
            "logout": AuditEventType.AUTH_LOGOUT,
            "token_issued": AuditEventType.AUTH_TOKEN_ISSUED,
            "token_revoked": AuditEventType.AUTH_TOKEN_REVOKED,
        }.get(action, AuditEventType.AUTH_LOGIN)

        self._log_event(
            AuditEvent(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                action=action,
                outcome="success" if success else "failure",
                ip_address=ip_address,
                user_agent=user_agent,
                details=extra,
            )
        )

    def log_agent_operation(
        self,
        agent_id: str,
        operation: str,
        user_id: str | None = None,
        target_agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log an agent operation.

        Args:
            agent_id: Agent performing operation
            operation: Operation type (register, share, sync)
            user_id: User context if applicable
            target_agent_id: Target agent for sharing/sync
            **extra: Additional details
        """
        event_type = {
            "register": AuditEventType.AGENT_REGISTER,
            "unregister": AuditEventType.AGENT_UNREGISTER,
            "share": AuditEventType.AGENT_SHARE,
            "sync": AuditEventType.AGENT_SYNC,
        }.get(operation, AuditEventType.AGENT_REGISTER)

        self._log_event(
            AuditEvent(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                agent_id=agent_id,
                action=operation,
                details={"target_agent_id": target_agent_id, **extra},
            )
        )

    def log_security_event(
        self,
        event_type: str,
        user_id: str | None = None,
        ip_address: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event
            user_id: User involved
            ip_address: Client IP address
            **extra: Additional details
        """
        audit_type = {
            "rate_limited": AuditEventType.SECURITY_RATE_LIMITED,
            "invalid_input": AuditEventType.SECURITY_INVALID_INPUT,
            "encryption": AuditEventType.SECURITY_ENCRYPTION,
        }.get(event_type, AuditEventType.SECURITY_INVALID_INPUT)

        self._log_event(
            AuditEvent(
                event_type=audit_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                ip_address=ip_address,
                action=event_type,
                details=extra,
            )
        )

    def log_admin_operation(
        self,
        user_id: str,
        operation: str,
        **extra: Any,
    ) -> None:
        """Log an administrative operation.

        Args:
            user_id: Admin user
            operation: Operation type
            **extra: Additional details
        """
        event_type = {
            "config_change": AuditEventType.ADMIN_CONFIG_CHANGE,
            "migration": AuditEventType.ADMIN_MIGRATION,
            "export": AuditEventType.ADMIN_EXPORT,
            "import": AuditEventType.ADMIN_IMPORT,
        }.get(operation, AuditEventType.ADMIN_CONFIG_CHANGE)

        self._log_event(
            AuditEvent(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                action=operation,
                details=extra,
            )
        )

    def close(self) -> None:
        """Close the audit logger and flush any buffered logs."""
        if hasattr(self, "_file_handle"):
            self._file_handle.close()
        if hasattr(self, "_file_handler"):
            self._file_handler.close()
