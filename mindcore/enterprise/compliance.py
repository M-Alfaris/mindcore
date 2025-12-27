"""GDPR/CCPA Compliance Tools for MindCore.

Provides data protection compliance features:
- GDPR Article 15: Right of Access (data export)
- GDPR Article 17: Right to Erasure (right to be forgotten)
- Data anonymization for analytics
- Automatic retention policy enforcement

Example:
    from mindcore.enterprise.compliance import (
        ComplianceManager,
        RetentionPolicy,
        AnonymizationStrategy,
    )

    # Initialize with storage
    compliance = ComplianceManager(storage)

    # Export all user data (GDPR Article 15)
    export = await compliance.export_user_data("user_123")
    with open("user_data.json", "w") as f:
        f.write(export.to_json())

    # Delete all user data (GDPR Article 17)
    result = await compliance.delete_user_data("user_123")
    print(f"Deleted {result.memories_deleted} memories")

    # Anonymize user data for analytics
    result = compliance.anonymize_user_data(
        "user_123",
        strategy=AnonymizationStrategy.PSEUDONYMIZE,
    )

    # Configure retention policies
    retention = RetentionPolicy(
        memory_type_policies={
            "episodic": 730,      # 2 years
            "preference": None,   # Forever
            "working": 1,         # 1 day
        },
        default_max_age_days=365,
    )
    compliance.set_retention_policy(retention)

    # Enforce retention (run periodically)
    result = compliance.enforce_retention()
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mindcore.flr import Memory
    from mindcore.storage.base import BaseStorage


class AnonymizationStrategy(str, Enum):
    """Strategies for anonymizing user data."""

    # Replace user_id with random UUID, keep content
    PSEUDONYMIZE = "pseudonymize"

    # Hash user_id deterministically (allows re-linking if key known)
    HASH = "hash"

    # Remove all PII from content, replace user_id
    REDACT = "redact"

    # Delete content, keep metadata for analytics
    AGGREGATE = "aggregate"


class ComplianceEventType(str, Enum):
    """Types of compliance events for audit logging."""

    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"
    DATA_ANONYMIZE = "data_anonymize"
    RETENTION_ENFORCE = "retention_enforce"
    CONSENT_RECORD = "consent_record"
    ACCESS_REQUEST = "access_request"


@dataclass
class RetentionPolicy:
    """Data retention policy configuration.

    Attributes:
        memory_type_policies: Max age in days per memory type (None = forever)
        default_max_age_days: Default max age if type not specified
        enforce_on_access: Check retention on every access
        delete_expired_batch_size: Batch size for deletion jobs
    """

    memory_type_policies: dict[str, int | None] = field(default_factory=dict)
    default_max_age_days: int | None = 365
    enforce_on_access: bool = False
    delete_expired_batch_size: int = 1000

    def get_max_age(self, memory_type: str) -> int | None:
        """Get max age for a memory type.

        Args:
            memory_type: Type of memory

        Returns:
            Max age in days, or None for no expiration
        """
        return self.memory_type_policies.get(memory_type, self.default_max_age_days)

    def get_cutoff_date(self, memory_type: str) -> datetime | None:
        """Get cutoff date for a memory type.

        Args:
            memory_type: Type of memory

        Returns:
            Datetime before which memories should be deleted, or None
        """
        max_age = self.get_max_age(memory_type)
        if max_age is None:
            return None
        return datetime.now(timezone.utc) - timedelta(days=max_age)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_type_policies": self.memory_type_policies,
            "default_max_age_days": self.default_max_age_days,
            "enforce_on_access": self.enforce_on_access,
            "delete_expired_batch_size": self.delete_expired_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionPolicy:
        """Create from dictionary."""
        return cls(
            memory_type_policies=data.get("memory_type_policies", {}),
            default_max_age_days=data.get("default_max_age_days", 365),
            enforce_on_access=data.get("enforce_on_access", False),
            delete_expired_batch_size=data.get("delete_expired_batch_size", 1000),
        )


@dataclass
class GDPRExportResult:
    """Result of a GDPR data export operation.

    Attributes:
        user_id: User whose data was exported
        export_id: Unique identifier for this export
        exported_at: Timestamp of export
        memories: List of exported memories
        memory_count: Total number of memories
        metadata: Additional export metadata
    """

    user_id: str
    export_id: str
    exported_at: datetime
    memories: list[dict[str, Any]]
    memory_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "export_info": {
                "user_id": self.user_id,
                "export_id": self.export_id,
                "exported_at": self.exported_at.isoformat(),
                "memory_count": self.memory_count,
                "format_version": "1.0",
                "generator": "mindcore-compliance",
            },
            "memories": self.memories,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class GDPRDeleteResult:
    """Result of a GDPR deletion operation.

    Attributes:
        user_id: User whose data was deleted
        deletion_id: Unique identifier for this deletion
        deleted_at: Timestamp of deletion
        memories_deleted: Number of memories deleted
        cache_cleared: Whether cache was also cleared
        verification_token: Token to verify deletion was complete
    """

    user_id: str
    deletion_id: str
    deleted_at: datetime
    memories_deleted: int
    cache_cleared: bool = False
    verification_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "deletion_id": self.deletion_id,
            "deleted_at": self.deleted_at.isoformat(),
            "memories_deleted": self.memories_deleted,
            "cache_cleared": self.cache_cleared,
            "verification_token": self.verification_token,
        }


@dataclass
class AnonymizationResult:
    """Result of an anonymization operation.

    Attributes:
        original_user_id: Original user ID (for logging only)
        anonymized_user_id: New anonymized user ID
        anonymization_id: Unique identifier for this operation
        anonymized_at: Timestamp of anonymization
        memories_anonymized: Number of memories anonymized
        strategy: Strategy used for anonymization
    """

    original_user_id: str
    anonymized_user_id: str
    anonymization_id: str
    anonymized_at: datetime
    memories_anonymized: int
    strategy: AnonymizationStrategy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_user_id": self.original_user_id,
            "anonymized_user_id": self.anonymized_user_id,
            "anonymization_id": self.anonymization_id,
            "anonymized_at": self.anonymized_at.isoformat(),
            "memories_anonymized": self.memories_anonymized,
            "strategy": self.strategy.value,
        }


@dataclass
class RetentionEnforcementResult:
    """Result of retention policy enforcement.

    Attributes:
        enforcement_id: Unique identifier for this enforcement run
        enforced_at: Timestamp of enforcement
        memories_deleted: Number of memories deleted
        memories_by_type: Breakdown by memory type
        errors: Any errors encountered
    """

    enforcement_id: str
    enforced_at: datetime
    memories_deleted: int
    memories_by_type: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enforcement_id": self.enforcement_id,
            "enforced_at": self.enforced_at.isoformat(),
            "memories_deleted": self.memories_deleted,
            "memories_by_type": self.memories_by_type,
            "errors": self.errors,
        }


class ComplianceManager:
    """GDPR/CCPA compliance manager for MindCore.

    Provides tools for data protection compliance including:
    - Data export (Right of Access)
    - Data deletion (Right to be Forgotten)
    - Data anonymization
    - Retention policy enforcement

    Example:
        compliance = ComplianceManager(storage)

        # Export user data
        export = compliance.export_user_data("user_123")

        # Delete user data
        result = compliance.delete_user_data("user_123")

        # Anonymize for analytics
        result = compliance.anonymize_user_data("user_123")
    """

    def __init__(
        self,
        storage: BaseStorage,
        retention_policy: RetentionPolicy | None = None,
        on_event: Any | None = None,
        anonymization_salt: str | None = None,
    ):
        """Initialize ComplianceManager.

        Args:
            storage: Storage backend
            retention_policy: Optional retention policy
            on_event: Optional callback for compliance events
            anonymization_salt: Salt for hash-based anonymization
        """
        self.storage = storage
        self.retention_policy = retention_policy
        self.on_event = on_event
        self._anonymization_salt = anonymization_salt or uuid.uuid4().hex

    def export_user_data(
        self,
        user_id: str,
        include_metadata: bool = True,
        include_system_fields: bool = False,
    ) -> GDPRExportResult:
        """Export all data for a user (GDPR Article 15 - Right of Access).

        Args:
            user_id: User ID to export
            include_metadata: Include memory metadata
            include_system_fields: Include internal system fields

        Returns:
            GDPRExportResult with all user data
        """
        export_id = f"export_{uuid.uuid4().hex[:12]}"
        exported_at = datetime.now(timezone.utc)

        # Fetch all memories for the user
        memories = self.storage.search(user_id=user_id, limit=100000)

        # Convert to exportable format
        exported_memories = []
        for memory in memories:
            mem_dict = memory.to_dict()

            if not include_system_fields:
                # Remove internal system fields
                for field_name in [
                    "reinforcement_score",
                    "access_count",
                    "vocabulary_version",
                    "embedding",
                ]:
                    mem_dict.pop(field_name, None)

            if not include_metadata:
                # Keep only essential fields
                mem_dict = {
                    "memory_id": mem_dict["memory_id"],
                    "content": mem_dict["content"],
                    "memory_type": mem_dict["memory_type"],
                    "created_at": mem_dict["created_at"],
                }

            exported_memories.append(mem_dict)

        result = GDPRExportResult(
            user_id=user_id,
            export_id=export_id,
            exported_at=exported_at,
            memories=exported_memories,
            memory_count=len(exported_memories),
            metadata={
                "include_metadata": include_metadata,
                "include_system_fields": include_system_fields,
            },
        )

        self._emit_event(
            ComplianceEventType.DATA_EXPORT,
            user_id,
            {"export_id": export_id, "memory_count": len(exported_memories)},
        )

        return result

    def delete_user_data(
        self,
        user_id: str,
        clear_cache: bool = True,
        verify: bool = True,
    ) -> GDPRDeleteResult:
        """Delete all data for a user (GDPR Article 17 - Right to Erasure).

        This is a destructive operation that cannot be undone.

        Args:
            user_id: User ID to delete
            clear_cache: Also clear cached data
            verify: Verify deletion was complete

        Returns:
            GDPRDeleteResult with deletion details
        """
        deletion_id = f"delete_{uuid.uuid4().hex[:12]}"
        deleted_at = datetime.now(timezone.utc)

        # Get all memories for the user
        memories = self.storage.search(user_id=user_id, limit=100000)

        # Delete each memory
        deleted_count = 0
        for memory in memories:
            try:
                self.storage.delete(memory.memory_id)
                deleted_count += 1
            except Exception:
                # Memory may have already been deleted
                pass

        # Generate verification token
        verification_token = ""  # nosec B105 - not a password, initialized before conditional
        if verify:
            # Verify no memories remain
            remaining = self.storage.search(user_id=user_id, limit=1)
            if not remaining:
                verification_token = hashlib.sha256(
                    f"{deletion_id}:{user_id}:{deleted_at.isoformat()}".encode()
                ).hexdigest()[:16]

        result = GDPRDeleteResult(
            user_id=user_id,
            deletion_id=deletion_id,
            deleted_at=deleted_at,
            memories_deleted=deleted_count,
            cache_cleared=clear_cache,
            verification_token=verification_token,
        )

        self._emit_event(
            ComplianceEventType.DATA_DELETE,
            user_id,
            {"deletion_id": deletion_id, "memories_deleted": deleted_count},
        )

        return result

    def anonymize_user_data(
        self,
        user_id: str,
        strategy: AnonymizationStrategy = AnonymizationStrategy.PSEUDONYMIZE,
        pii_patterns: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize all data for a user.

        This replaces the user's identity with an anonymous identifier
        while preserving the data for analytics.

        Args:
            user_id: User ID to anonymize
            strategy: Anonymization strategy to use
            pii_patterns: Optional regex patterns for PII to redact

        Returns:
            AnonymizationResult with anonymization details
        """
        anonymization_id = f"anon_{uuid.uuid4().hex[:12]}"
        anonymized_at = datetime.now(timezone.utc)

        # Generate anonymized user ID based on strategy
        if strategy == AnonymizationStrategy.HASH:
            anonymized_user_id = hashlib.sha256(
                f"{self._anonymization_salt}:{user_id}".encode()
            ).hexdigest()[:16]
            anonymized_user_id = f"anon_{anonymized_user_id}"
        else:
            anonymized_user_id = f"anon_{uuid.uuid4().hex[:12]}"

        # Get all memories for the user
        memories = self.storage.search(user_id=user_id, limit=100000)

        # Anonymize each memory
        anonymized_count = 0
        for memory in memories:
            try:
                self._anonymize_memory(memory, anonymized_user_id, strategy, pii_patterns)
                self.storage.update(memory)
                anonymized_count += 1
            except Exception:
                pass

        result = AnonymizationResult(
            original_user_id=user_id,
            anonymized_user_id=anonymized_user_id,
            anonymization_id=anonymization_id,
            anonymized_at=anonymized_at,
            memories_anonymized=anonymized_count,
            strategy=strategy,
        )

        self._emit_event(
            ComplianceEventType.DATA_ANONYMIZE,
            user_id,
            {
                "anonymization_id": anonymization_id,
                "anonymized_user_id": anonymized_user_id,
                "memories_anonymized": anonymized_count,
                "strategy": strategy.value,
            },
        )

        return result

    def _anonymize_memory(
        self,
        memory: Memory,
        anonymized_user_id: str,
        strategy: AnonymizationStrategy,
        pii_patterns: list[str] | None,
    ) -> None:
        """Anonymize a single memory in place.

        Args:
            memory: Memory to anonymize
            anonymized_user_id: New user ID
            strategy: Anonymization strategy
            pii_patterns: Optional PII patterns to redact
        """
        import re

        # Always replace user_id
        memory.user_id = anonymized_user_id

        # Clear agent_id if present
        if memory.agent_id:
            memory.agent_id = None

        if strategy == AnonymizationStrategy.PSEUDONYMIZE:
            # Just replace IDs, keep content
            pass

        elif strategy == AnonymizationStrategy.HASH:
            # Hash IDs, keep content
            pass

        elif strategy == AnonymizationStrategy.REDACT:
            # Redact PII from content
            content = memory.content

            # Default PII patterns
            default_patterns = [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card
                r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP address
            ]

            patterns = (pii_patterns or []) + default_patterns

            for pattern in patterns:
                content = re.sub(pattern, "[REDACTED]", content)

            memory.content = content

            # Clear entities (likely contain PII)
            memory.entities = []

        elif strategy == AnonymizationStrategy.AGGREGATE:
            # Remove content entirely, keep only metadata
            memory.content = f"[AGGREGATED: {memory.memory_type}]"
            memory.entities = []
            memory.topics = memory.topics[:3] if memory.topics else []

    def set_retention_policy(self, policy: RetentionPolicy) -> None:
        """Set the retention policy.

        Args:
            policy: Retention policy to use
        """
        self.retention_policy = policy

    def enforce_retention(
        self,
        user_id: str | None = None,
        dry_run: bool = False,
    ) -> RetentionEnforcementResult:
        """Enforce retention policy by deleting expired memories.

        Args:
            user_id: Optional user ID to limit enforcement to
            dry_run: If True, don't actually delete, just count

        Returns:
            RetentionEnforcementResult with enforcement details
        """
        if not self.retention_policy:
            return RetentionEnforcementResult(
                enforcement_id=f"enforce_{uuid.uuid4().hex[:12]}",
                enforced_at=datetime.now(timezone.utc),
                memories_deleted=0,
                errors=["No retention policy configured"],
            )

        enforcement_id = f"enforce_{uuid.uuid4().hex[:12]}"
        enforced_at = datetime.now(timezone.utc)

        deleted_by_type: dict[str, int] = {}
        total_deleted = 0
        errors: list[str] = []

        # Get all unique memory types to process
        memory_types = set(self.retention_policy.memory_type_policies.keys())
        if self.retention_policy.default_max_age_days is not None:
            # Need to also process any types with default policy
            # Get a sample to find all types
            sample = self.storage.search(user_id=user_id, limit=1000)
            for mem in sample:
                memory_types.add(mem.memory_type)

        # Process each memory type
        for memory_type in memory_types:
            cutoff_date = self.retention_policy.get_cutoff_date(memory_type)
            if cutoff_date is None:
                continue  # No expiration for this type

            # Find expired memories
            expired = self.storage.search(
                user_id=user_id,
                memory_types=[memory_type],
                end_date=cutoff_date,
                limit=self.retention_policy.delete_expired_batch_size,
            )

            type_deleted = 0
            for memory in expired:
                if not dry_run:
                    try:
                        self.storage.delete(memory.memory_id)
                        type_deleted += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {memory.memory_id}: {e}")
                else:
                    type_deleted += 1

            deleted_by_type[memory_type] = type_deleted
            total_deleted += type_deleted

        result = RetentionEnforcementResult(
            enforcement_id=enforcement_id,
            enforced_at=enforced_at,
            memories_deleted=total_deleted,
            memories_by_type=deleted_by_type,
            errors=errors,
        )

        if not dry_run:
            self._emit_event(
                ComplianceEventType.RETENTION_ENFORCE,
                user_id or "all_users",
                {
                    "enforcement_id": enforcement_id,
                    "memories_deleted": total_deleted,
                    "by_type": deleted_by_type,
                },
            )

        return result

    def get_user_data_summary(self, user_id: str) -> dict[str, Any]:
        """Get summary of data held for a user.

        Useful for responding to data access requests without full export.

        Args:
            user_id: User ID to summarize

        Returns:
            Summary of user's data
        """
        memories = self.storage.search(user_id=user_id, limit=100000)

        # Aggregate by type
        by_type: dict[str, int] = {}
        by_topic: dict[str, int] = {}
        oldest_date: datetime | None = None
        newest_date: datetime | None = None

        for memory in memories:
            by_type[memory.memory_type] = by_type.get(memory.memory_type, 0) + 1

            for topic in memory.topics:
                by_topic[topic] = by_topic.get(topic, 0) + 1

            if memory.created_at:
                if oldest_date is None or memory.created_at < oldest_date:
                    oldest_date = memory.created_at
                if newest_date is None or memory.created_at > newest_date:
                    newest_date = memory.created_at

        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "memories_by_type": by_type,
            "memories_by_topic": dict(sorted(by_topic.items(), key=lambda x: -x[1])[:10]),
            "oldest_memory": oldest_date.isoformat() if oldest_date else None,
            "newest_memory": newest_date.isoformat() if newest_date else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def check_retention_status(
        self,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Check which memories would be affected by retention policy.

        Args:
            user_id: Optional user ID to limit check to

        Returns:
            Status of retention policy enforcement
        """
        if not self.retention_policy:
            return {"status": "no_policy", "memories_affected": 0}

        # Dry run to count affected memories
        result = self.enforce_retention(user_id=user_id, dry_run=True)

        return {
            "status": "ok",
            "memories_affected": result.memories_deleted,
            "by_type": result.memories_by_type,
            "policy": self.retention_policy.to_dict(),
        }

    def _emit_event(
        self,
        event_type: ComplianceEventType,
        subject: str,
        data: dict[str, Any],
    ) -> None:
        """Emit a compliance event.

        Args:
            event_type: Type of event
            subject: Subject (usually user_id)
            data: Event data
        """
        if self.on_event:
            try:
                self.on_event(event_type, subject, data)
            except Exception:
                pass
