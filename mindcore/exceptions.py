"""Mindcore Exceptions - Standardized error handling.

All mindcore operations that fail raise exceptions rather than returning
False/None. This follows the "fail hard" philosophy for predictable behavior.

Exception Hierarchy:
    MindcoreError (base)
    ├── StorageError
    │   ├── MemoryNotFoundError
    │   └── StorageConnectionError
    ├── ValidationError
    │   ├── VocabularyValidationError
    │   └── MemoryValidationError
    ├── MigrationError
    │   ├── MigrationPathError
    │   └── RollbackError
    ├── AccessError
    │   ├── PermissionDeniedError
    │   └── AgentNotFoundError
    └── ConfigurationError
"""

from __future__ import annotations


class MindcoreError(Exception):
    """Base exception for all Mindcore errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# === Storage Errors ===


class StorageError(MindcoreError):
    """Base exception for storage-related errors."""


class MemoryNotFoundError(StorageError):
    """Raised when a memory is not found in storage."""

    def __init__(self, memory_id: str):
        super().__init__(f"Memory '{memory_id}' not found", details={"memory_id": memory_id})
        self.memory_id = memory_id


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""

    def __init__(self, message: str, backend: str | None = None):
        super().__init__(message, details={"backend": backend} if backend else {})
        self.backend = backend


# === Validation Errors ===


class ValidationError(MindcoreError):
    """Base exception for validation errors."""


class VocabularyValidationError(ValidationError):
    """Raised when vocabulary validation fails."""

    def __init__(self, errors: list[str], memory_data: dict | None = None):
        message = f"Vocabulary validation failed: {', '.join(errors)}"
        super().__init__(message, details={"validation_errors": errors})
        self.errors = errors
        self.memory_data = memory_data


class MemoryValidationError(ValidationError):
    """Raised when memory data validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, details={"field": field} if field else {})
        self.field = field


# === Migration Errors ===


class MigrationError(MindcoreError):
    """Base exception for migration errors."""


class MigrationPathError(MigrationError):
    """Raised when no migration path exists between versions."""

    def __init__(self, from_version: str, to_version: str):
        super().__init__(
            f"No migration path from version '{from_version}' to '{to_version}'",
            details={"from_version": from_version, "to_version": to_version},
        )
        self.from_version = from_version
        self.to_version = to_version


class RollbackError(MigrationError):
    """Raised when migration rollback fails."""

    def __init__(self, message: str, checkpoint_id: str | None = None):
        super().__init__(message, details={"checkpoint_id": checkpoint_id} if checkpoint_id else {})
        self.checkpoint_id = checkpoint_id


# === Access Control Errors ===


class AccessError(MindcoreError):
    """Base exception for access control errors."""


class PermissionDeniedError(AccessError):
    """Raised when an operation is not permitted."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        required_permission: str | None = None,
        permission: str | None = None,  # Alias for required_permission
    ):
        # Support both 'permission' and 'required_permission' for backwards compatibility
        effective_permission = required_permission or permission
        details = {}
        if agent_id:
            details["agent_id"] = agent_id
        if effective_permission:
            details["required_permission"] = effective_permission
        super().__init__(message, details=details)
        self.agent_id = agent_id
        self.required_permission = effective_permission
        self.permission = effective_permission  # Alias


class AgentNotFoundError(AccessError):
    """Raised when an agent is not found in the registry."""

    def __init__(self, agent_id: str):
        super().__init__(f"Agent '{agent_id}' not found", details={"agent_id": agent_id})
        self.agent_id = agent_id


# === Configuration Errors ===


class ConfigurationError(MindcoreError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message, details={"config_key": config_key} if config_key else {})
        self.config_key = config_key


class MultiAgentNotEnabledError(ConfigurationError):
    """Raised when multi-agent operations are attempted without enabling."""

    def __init__(self):
        super().__init__(
            "Multi-agent not enabled. Initialize Mindcore with enable_multi_agent=True",
            config_key="enable_multi_agent",
        )
