"""Rate limiting module for Mindcore.

Provides configurable rate limiting with multiple backend support
using the `limits` library.

Requirements:
    pip install limits

Optional backends:
    pip install redis  # For Redis backend
    pip install pymemcache  # For Memcached backend

Example:
    from mindcore.enterprise import RateLimiter, RateLimitConfig

    # Simple in-memory rate limiter
    limiter = RateLimiter(limit="100/minute")

    # Check if operation is allowed
    if limiter.is_allowed("user123", "store"):
        # Perform operation
        pass
    else:
        raise RateLimitExceededError("Rate limit exceeded")

    # Or use context manager
    with limiter.limit("user123", "recall"):
        # Perform operation
        pass

    # With Redis backend for distributed systems
    limiter = RateLimiter(
        limit="1000/hour",
        backend="redis",
        backend_uri="redis://localhost:6379",
    )

References:
    - https://limits.readthedocs.io/en/stable/
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator


# Type hints for optional dependency
try:
    from limits import parse
    from limits.storage import MemoryStorage, storage_from_string
    from limits.strategies import (
        FixedWindowRateLimiter,
        MovingWindowRateLimiter,
    )

    LIMITS_AVAILABLE = True
except ImportError:
    LIMITS_AVAILABLE = False


class RateLimitBackend(str, Enum):
    """Supported rate limit storage backends."""

    MEMORY = "memory"  # In-process memory (not distributed)
    REDIS = "redis"  # Redis for distributed rate limiting
    MEMCACHED = "memcached"  # Memcached backend
    MONGODB = "mongodb"  # MongoDB backend


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded.

    Attributes:
        identifier: The identifier that exceeded the limit
        operation: The operation that was limited
        limit: The limit that was exceeded
        retry_after: Seconds until the limit resets (if available)
    """

    def __init__(
        self,
        message: str,
        identifier: str = "",
        operation: str = "",
        limit: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.identifier = identifier
        self.operation = operation
        self.limit = limit
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        default_limit: Default rate limit string (e.g., "100/minute")
        backend: Storage backend type
        backend_uri: Connection URI for backend (required for redis/memcached)
        strategy: Rate limiting strategy ("fixed" or "moving")
        operation_limits: Per-operation rate limits
        user_tier_limits: Per-user-tier rate limits
        burst_enabled: Allow burst requests above limit temporarily
        burst_multiplier: Multiplier for burst limit (e.g., 1.5x normal limit)

    Rate Limit String Format:
        - "100/minute" - 100 requests per minute
        - "1000/hour" - 1000 requests per hour
        - "10000/day" - 10000 requests per day
        - "5/second" - 5 requests per second

    Example:
        config = RateLimitConfig(
            default_limit="100/minute",
            backend=RateLimitBackend.REDIS,
            backend_uri="redis://localhost:6379",
            operation_limits={
                "store": "50/minute",  # More restrictive for writes
                "recall": "200/minute",  # Less restrictive for reads
            },
            user_tier_limits={
                "free": "100/hour",
                "pro": "1000/hour",
                "enterprise": "10000/hour",
            },
        )
    """

    default_limit: str = "100/minute"
    backend: RateLimitBackend = RateLimitBackend.MEMORY
    backend_uri: str | None = None
    strategy: str = "moving"  # "fixed" or "moving"
    operation_limits: dict[str, str] = field(default_factory=dict)
    user_tier_limits: dict[str, str] = field(default_factory=dict)
    burst_enabled: bool = False
    burst_multiplier: float = 1.5


class RateLimiter:
    """Rate limiter for Mindcore operations.

    Supports multiple backends and strategies for flexible rate limiting
    in both single-node and distributed deployments.

    Example:
        # In-memory rate limiter (single process)
        limiter = RateLimiter(limit="100/minute")

        # Redis-backed rate limiter (distributed)
        limiter = RateLimiter(
            limit="1000/hour",
            backend="redis",
            backend_uri="redis://localhost:6379",
        )

        # Check and consume
        if limiter.is_allowed("user123", "store"):
            perform_store_operation()

        # With automatic enforcement
        with limiter.limit("user123", "recall"):
            perform_recall_operation()

        # Get remaining quota
        remaining = limiter.get_remaining("user123", "store")
        print(f"Remaining: {remaining} requests")
    """

    def __init__(
        self,
        limit: str = "100/minute",
        backend: str | RateLimitBackend = RateLimitBackend.MEMORY,
        backend_uri: str | None = None,
        config: RateLimitConfig | None = None,
    ):
        """Initialize rate limiter.

        Args:
            limit: Rate limit string (e.g., "100/minute")
            backend: Storage backend
            backend_uri: Connection URI for backend
            config: Full configuration (overrides other args)

        Raises:
            ImportError: If limits library is not installed
        """
        if not LIMITS_AVAILABLE:
            raise ImportError(
                "limits library required for rate limiting. Install with: pip install limits"
            )

        if config:
            self.config = config
        else:
            backend_enum = (
                backend if isinstance(backend, RateLimitBackend) else RateLimitBackend(backend)
            )
            self.config = RateLimitConfig(
                default_limit=limit,
                backend=backend_enum,
                backend_uri=backend_uri,
            )

        self._setup_limiter()

    def _setup_limiter(self) -> None:
        """Initialize the rate limiter storage and strategy."""
        # Setup storage backend
        if self.config.backend == RateLimitBackend.MEMORY:
            self._storage = MemoryStorage()
        elif self.config.backend_uri:
            self._storage = storage_from_string(self.config.backend_uri)
        else:
            raise ValueError(f"backend_uri required for {self.config.backend.value} backend")

        # Setup strategy
        if self.config.strategy == "fixed":
            self._limiter = FixedWindowRateLimiter(self._storage)
        else:
            self._limiter = MovingWindowRateLimiter(self._storage)

        # Parse default limit
        self._default_limit = parse(self.config.default_limit)

        # Parse operation-specific limits
        self._operation_limits = {
            op: parse(limit_str) for op, limit_str in self.config.operation_limits.items()
        }

        # Parse tier-specific limits
        self._tier_limits = {
            tier: parse(limit_str) for tier, limit_str in self.config.user_tier_limits.items()
        }

    def _get_limit(
        self,
        operation: str | None = None,
        user_tier: str | None = None,
    ):
        """Get the applicable rate limit.

        Args:
            operation: Operation name
            user_tier: User tier for tiered limits

        Returns:
            Parsed rate limit
        """
        # Check tier-specific first (most specific)
        if user_tier and user_tier in self._tier_limits:
            return self._tier_limits[user_tier]

        # Check operation-specific
        if operation and operation in self._operation_limits:
            return self._operation_limits[operation]

        # Fall back to default
        return self._default_limit

    def _make_key(self, identifier: str, operation: str | None = None) -> str:
        """Create a storage key for the identifier and operation.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name

        Returns:
            Storage key string
        """
        if operation:
            return f"mindcore:{identifier}:{operation}"
        return f"mindcore:{identifier}"

    def is_allowed(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
        cost: int = 1,
    ) -> bool:
        """Check if an operation is allowed and consume quota.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name for op-specific limits
            user_tier: Optional user tier for tiered limits
            cost: Cost of this operation (default: 1)

        Returns:
            True if allowed (quota consumed), False if rate limited
        """
        limit = self._get_limit(operation, user_tier)
        key = self._make_key(identifier, operation)

        return self._limiter.hit(limit, key, cost=cost)

    def check(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
    ) -> bool:
        """Check if an operation would be allowed without consuming quota.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
            user_tier: Optional user tier

        Returns:
            True if would be allowed, False otherwise
        """
        limit = self._get_limit(operation, user_tier)
        key = self._make_key(identifier, operation)

        return self._limiter.test(limit, key)

    def get_remaining(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
    ) -> int:
        """Get remaining quota for an identifier.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
            user_tier: Optional user tier

        Returns:
            Number of remaining requests in current window
        """
        limit = self._get_limit(operation, user_tier)
        key = self._make_key(identifier, operation)

        window_stats = self._limiter.get_window_stats(limit, key)
        # window_stats.remaining is the number of remaining requests available
        return window_stats.remaining

    def get_reset_time(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
    ) -> float:
        """Get time until rate limit resets.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
            user_tier: Optional user tier

        Returns:
            Seconds until limit resets
        """
        limit = self._get_limit(operation, user_tier)
        key = self._make_key(identifier, operation)

        window_stats = self._limiter.get_window_stats(limit, key)
        return max(0, window_stats.reset_time - time.time())

    @contextmanager
    def limit(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
        cost: int = 1,
    ) -> Generator[None, None, None]:
        """Context manager that enforces rate limiting.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
            user_tier: Optional user tier
            cost: Cost of this operation

        Raises:
            RateLimitExceededError: If rate limit is exceeded

        Example:
            with limiter.limit("user123", "store"):
                perform_store_operation()
        """
        if not self.is_allowed(identifier, operation, user_tier, cost):
            retry_after = self.get_reset_time(identifier, operation, user_tier)
            limit = self._get_limit(operation, user_tier)

            raise RateLimitExceededError(
                f"Rate limit exceeded for {identifier}",
                identifier=identifier,
                operation=operation or "default",
                limit=str(limit),
                retry_after=retry_after,
            )

        yield

    def clear(self, identifier: str, operation: str | None = None) -> None:
        """Clear rate limit state for an identifier.

        Useful for testing or administrative purposes.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
        """
        key = self._make_key(identifier, operation)
        self._storage.clear(key)

    def get_stats(
        self,
        identifier: str,
        operation: str | None = None,
        user_tier: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed rate limit statistics.

        Args:
            identifier: User/agent identifier
            operation: Optional operation name
            user_tier: Optional user tier

        Returns:
            Dict with limit, remaining, reset_time, and used
        """
        limit = self._get_limit(operation, user_tier)
        key = self._make_key(identifier, operation)

        window_stats = self._limiter.get_window_stats(limit, key)

        return {
            "limit": limit.amount,
            "window_seconds": limit.multiples,
            "remaining": window_stats.remaining,
            "used": limit.amount - window_stats.remaining,
            "reset_time": window_stats.reset_time,
            "reset_in_seconds": max(0, window_stats.reset_time - time.time()),
        }
