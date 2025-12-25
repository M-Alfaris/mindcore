"""Patterns for Customer-Facing AI Agents.

This module shows how to use the existing FLR, CLST, SVL, and Federation
layers for user-facing AI agents. No new abstractions needed - just
patterns for common use cases.

The core insight: Users are just namespaces with access control.

Key Patterns:
1. User as Namespace: Each user gets their own namespace
2. Consent as AccessLevel: Map user consent to federation access levels
3. Multi-session via CLST: Store persistent preferences in user namespace
4. Session via FLR: Hot memory for current conversation
5. Privacy via Filtering: Apply PII filters before storage (app-level)

Example:
    from mindcore.v2.federation import quick_setup, AccessLevel
    from mindcore.v2.patterns.customer_facing import UserMemoryHelper

    # Setup federation (once for your org)
    federation = quick_setup(
        org_id="my-company",
        departments={"support": ["tier-1"]},
    )

    # Create customer-facing agent
    agent = federation.create_agent(
        agent_id="support-bot",
        agent_type="support-bot",
        department="support",
        team="tier-1",
    )

    # Helper for user-specific operations
    user_helper = UserMemoryHelper(agent)

    # Remember something for a user
    user_helper.remember_for_user(
        user_id="user-123",
        content="User prefers dark mode",
        memory_type="preference",
    )

    # Recall user's context across sessions
    context = user_helper.get_user_context("user-123")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mindcore.v2.federation import AgentMemoryBridge, AccessLevel


@dataclass
class UserMemoryHelper:
    """Thin helper for user-specific memory operations.

    Wraps an AgentMemoryBridge to provide user-centric convenience methods.
    All storage uses the existing FLR/CLST infrastructure.
    """

    agent: AgentMemoryBridge
    user_namespace_prefix: str = "user"

    # Simple in-memory session tracking (production: use Redis/DB)
    _active_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def remember_for_user(
        self,
        user_id: str,
        content: str,
        memory_type: str = "general",
        persist: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory for a specific user.

        Uses the existing FLR.remember() with user_id as metadata.
        """
        full_metadata = {
            "user_id": user_id,
            "memory_type": memory_type,
            "stored_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        return self.agent.remember(
            content=content,
            user_id=user_id,
            metadata=full_metadata,
            persist=persist,
        )

    def recall_for_user(
        self,
        user_id: str,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Recall memories for a specific user.

        Filters FLR/CLST results by user_id.
        """
        results = self.agent.recall(
            query=query,
            user_id=user_id,
            include_federated=True,
            limit=limit * 2,  # Over-fetch to filter
        )

        # Filter by memory type if specified
        if memory_type:
            results = [
                m for m in results
                if getattr(m, 'metadata', {}).get('memory_type') == memory_type
            ]

        return results[:limit]

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Get stored preferences for a user."""
        prefs = self.recall_for_user(
            user_id=user_id,
            memory_type="preference",
        )

        # Build preferences dict from memories
        result = {}
        for mem in prefs:
            content = getattr(mem, 'content', '')
            metadata = getattr(mem, 'metadata', {})
            if 'preference_key' in metadata:
                result[metadata['preference_key']] = metadata.get('preference_value', content)

        return result

    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
    ) -> str:
        """Set a preference for a user."""
        return self.remember_for_user(
            user_id=user_id,
            content=f"Preference: {key} = {value}",
            memory_type="preference",
            metadata={
                "preference_key": key,
                "preference_value": value,
            },
        )

    def get_user_context(
        self,
        user_id: str,
        include_history: bool = True,
        include_preferences: bool = True,
        history_limit: int = 5,
    ) -> dict[str, Any]:
        """Get full context for a user.

        Combines preferences, recent history, and session info.
        """
        context: dict[str, Any] = {
            "user_id": user_id,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

        if include_preferences:
            context["preferences"] = self.get_user_preferences(user_id)

        if include_history:
            history = self.recall_for_user(
                user_id=user_id,
                memory_type="conversation",
                limit=history_limit,
            )
            context["recent_history"] = [
                {
                    "content": getattr(m, 'content', ''),
                    "metadata": getattr(m, 'metadata', {}),
                }
                for m in history
            ]

        # Check for active session
        if user_id in self._active_sessions:
            context["active_session"] = self._active_sessions[user_id]

        return context

    def start_session(
        self,
        user_id: str,
        channel: str = "web",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a session for a user."""
        import uuid
        session_id = str(uuid.uuid4())

        self._active_sessions[user_id] = {
            "session_id": session_id,
            "started_at": datetime.utcnow().isoformat(),
            "channel": channel,
            "message_count": 0,
            **(metadata or {}),
        }

        return session_id

    def end_session(self, user_id: str) -> None:
        """End a user's session."""
        if user_id in self._active_sessions:
            session = self._active_sessions.pop(user_id)

            # Optionally store session summary
            self.remember_for_user(
                user_id=user_id,
                content=f"Session ended after {session.get('message_count', 0)} messages",
                memory_type="session_summary",
                metadata=session,
                persist=True,
            )

    def reinforce_for_user(
        self,
        user_id: str,
        memory_id: str,
        signal: float,
    ) -> float:
        """Reinforce a memory with user attribution."""
        return self.agent.reinforce(
            memory_id=memory_id,
            signal=signal,
            context={"user_id": user_id},
        )

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (right to erasure).

        Note: This only clears local FLR. For CLST deletion,
        you need to call the storage backend directly.

        Returns number of memories removed from local cache.
        """
        count = 0
        to_remove = []

        for mem_id, mem in self.agent.local_memories.items():
            if mem.user_id == user_id:
                to_remove.append(mem_id)

        for mem_id in to_remove:
            del self.agent.local_memories[mem_id]
            count += 1

        # Clear session
        self._active_sessions.pop(user_id, None)

        return count


# =============================================================================
# Consent to AccessLevel Mapping
# =============================================================================

def consent_to_access_level(consent: str) -> Any:
    """Map user consent string to AccessLevel.

    Usage:
        from mindcore.v2.federation import AccessLevel

        level = consent_to_access_level("full")  # -> AccessLevel.ORGANIZATION
        level = consent_to_access_level("minimal")  # -> AccessLevel.PRIVATE
    """
    from mindcore.v2.federation import AccessLevel

    mapping = {
        # Minimal consent - only agent can access
        "none": AccessLevel.PRIVATE,
        "minimal": AccessLevel.PRIVATE,

        # Functional - same agent type can share
        "functional": AccessLevel.AGENT_TYPE,

        # Team sharing
        "team": AccessLevel.TEAM,

        # Department sharing
        "department": AccessLevel.DEPARTMENT,

        # Full org access
        "full": AccessLevel.ORGANIZATION,
        "organization": AccessLevel.ORGANIZATION,
    }

    return mapping.get(consent.lower(), AccessLevel.PRIVATE)


# =============================================================================
# Simple PII Filter (Optional - App Level)
# =============================================================================

import re

# Common PII patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
}


def mask_pii(text: str, types: list[str] | None = None) -> str:
    """Simple PII masking for content before storage.

    This is a basic helper. For production, use a dedicated
    PII detection service.

    Args:
        text: Text to mask
        types: PII types to mask (default: all)

    Returns:
        Text with PII masked

    Example:
        safe = mask_pii("Email me at john@example.com")
        # -> "Email me at [EMAIL]"
    """
    types = types or list(PII_PATTERNS.keys())
    result = text

    for pii_type in types:
        if pii_type in PII_PATTERNS:
            result = PII_PATTERNS[pii_type].sub(f"[{pii_type.upper()}]", result)

    return result


def contains_pii(text: str) -> bool:
    """Check if text contains PII."""
    for pattern in PII_PATTERNS.values():
        if pattern.search(text):
            return True
    return False


# =============================================================================
# Usage Patterns
# =============================================================================

"""
PATTERN 1: Basic Customer Support Bot
=====================================

from mindcore.v2.federation import quick_setup
from mindcore.v2.patterns.customer_facing import UserMemoryHelper

# Setup
federation = quick_setup(org_id="company", departments=["support"])
agent = federation.create_agent("bot-1", "support", "support")
helper = UserMemoryHelper(agent)

# Handle user message
def handle_message(user_id: str, message: str):
    # Get context
    context = helper.get_user_context(user_id)

    # Remember the interaction
    helper.remember_for_user(user_id, message, "conversation")

    # Generate response using context
    response = llm.generate(context=context, message=message)

    # Remember response
    helper.remember_for_user(user_id, response, "conversation")

    return response


PATTERN 2: Multi-Session Personalization
========================================

# First session
helper.set_user_preference("user-123", "theme", "dark")
helper.set_user_preference("user-123", "language", "en")

# Later session (even different agent instance)
prefs = helper.get_user_preferences("user-123")
# -> {"theme": "dark", "language": "en"}


PATTERN 3: Privacy-Conscious Storage
====================================

from mindcore.v2.patterns.customer_facing import mask_pii, contains_pii

def safe_remember(helper, user_id, content):
    # Check and mask PII before storage
    if contains_pii(content):
        content = mask_pii(content)

    helper.remember_for_user(user_id, content)


PATTERN 4: Right to Erasure (GDPR)
==================================

# User requests deletion
def handle_deletion_request(user_id: str):
    # Clear local FLR
    count = helper.delete_user_memories(user_id)

    # For CLST, call storage directly
    # storage.delete_by_filter({"user_id": user_id})

    return f"Deleted {count} memories"


PATTERN 5: Consent-Based Access
===============================

from mindcore.v2.patterns.customer_facing import consent_to_access_level

def remember_with_consent(helper, user_id, content, consent):
    access_level = consent_to_access_level(consent)

    helper.remember_for_user(
        user_id=user_id,
        content=content,
        metadata={"access_level": access_level.value},
    )
"""
