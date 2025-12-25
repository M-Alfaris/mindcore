"""Common Patterns for MindCore.

This module provides usage patterns and thin helpers showing how to use
the core FLR, CLST, SVL, and Federation layers for common scenarios.

Available Patterns:
- customer_facing: For AI agents interacting with customers/users
- (future) autonomous: For fully autonomous agent systems
- (future) multi_model: For using multiple LLM providers

These are NOT new abstractions - just convenience wrappers and documentation
showing how to use the existing layers effectively.
"""

from .customer_facing import (
    UserMemoryHelper,
    consent_to_access_level,
    mask_pii,
    contains_pii,
)


__all__ = [
    "UserMemoryHelper",
    "consent_to_access_level",
    "mask_pii",
    "contains_pii",
]
