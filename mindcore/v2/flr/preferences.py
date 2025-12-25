"""Preference Manager - Temporal preference handling for MindCore.

Handles mutable user preferences that change over time, as opposed to
immutable facts (like orders, events). Provides:

1. Preference versioning with supersession chains
2. Temporal validity (valid_from, valid_until)
3. Automatic expiration of old preferences
4. Reinforcement signal adjustment for outdated preferences
5. Preference history tracking

Example:
    from mindcore.v2.flr import PreferenceManager, FLR
    from mindcore.v2.storage import SQLiteStorage

    storage = SQLiteStorage("mindcore.db")
    flr = FLR(storage)
    prefs = PreferenceManager(storage, flr)

    # Set initial preference
    pref = prefs.set_preference(
        user_id="user_123",
        key="theme",
        value="User prefers light mode",
        categories=["ui", "display"],
    )

    # Later, user changes preference
    new_pref = prefs.update_preference(
        user_id="user_123",
        key="theme",
        value="User now prefers dark mode with purple accents",
    )

    # Get current preference
    current = prefs.get_preference("user_123", "theme")
    print(current.content)  # "User now prefers dark mode..."

    # View preference history
    history = prefs.get_preference_history("user_123", "theme")
    for version in history:
        print(f"v{version.turn_index}: {version.content}")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from mindcore.v2.flr.recall import Memory


if TYPE_CHECKING:
    from mindcore.v2.flr.recall import FLR
    from mindcore.v2.storage.base import BaseStorage


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicting preferences from multiple agents."""

    NEWER_WINS = "newer_wins"  # Most recent timestamp wins (default)
    HIGHER_CONFIDENCE = "higher_confidence"  # Higher confidence score wins
    LLM_MERGE = "llm_merge"  # Use LLM to intelligently merge
    HUMAN_REVIEW = "human_review"  # Flag for human review
    AGENT_PRIORITY = "agent_priority"  # Use agent priority ranking
    KEEP_BOTH = "keep_both"  # Keep both as separate preferences


class ConflictStatus(str, Enum):
    """Status of a preference conflict."""

    DETECTED = "detected"  # Conflict detected, not yet resolved
    RESOLVED = "resolved"  # Conflict resolved automatically
    PENDING_REVIEW = "pending_review"  # Waiting for human review
    MERGED = "merged"  # Preferences merged via LLM
    DISMISSED = "dismissed"  # Conflict dismissed (not a real conflict)


@dataclass
class PreferenceConflict:
    """Represents a conflict between preferences from different agents."""

    conflict_id: str
    user_id: str
    preference_key: str
    existing_preference: Memory
    conflicting_preference: Memory
    existing_agent: str | None
    conflicting_agent: str | None
    status: ConflictStatus
    resolution_strategy: ConflictResolutionStrategy | None
    resolved_preference: Memory | None
    detected_at: datetime
    resolved_at: datetime | None
    resolution_reason: str | None
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "user_id": self.user_id,
            "preference_key": self.preference_key,
            "existing_preference": self.existing_preference.to_dict(),
            "conflicting_preference": self.conflicting_preference.to_dict(),
            "existing_agent": self.existing_agent,
            "conflicting_agent": self.conflicting_agent,
            "status": self.status.value,
            "resolution_strategy": self.resolution_strategy.value
            if self.resolution_strategy
            else None,
            "resolved_preference": self.resolved_preference.to_dict()
            if self.resolved_preference
            else None,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_reason": self.resolution_reason,
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class ConflictResolutionResult:
    """Result of conflict resolution."""

    conflict: PreferenceConflict
    resolved: bool
    final_preference: Memory | None
    strategy_used: ConflictResolutionStrategy
    message: str


@dataclass
class PreferenceUpdate:
    """Result of a preference update operation."""

    new_preference: Memory
    old_preference: Memory | None
    preference_key: str
    version: int
    superseded: bool
    user_id: str


@dataclass
class PreferenceHistory:
    """Complete history of a preference key."""

    preference_key: str
    user_id: str
    versions: list[Memory]
    current: Memory | None
    total_versions: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "preference_key": self.preference_key,
            "user_id": self.user_id,
            "versions": [v.to_dict() for v in self.versions],
            "current": self.current.to_dict() if self.current else None,
            "total_versions": self.total_versions,
        }


@dataclass
class PreferenceSummary:
    """Summary of all preferences for a user."""

    user_id: str
    preferences: dict[str, Memory]  # key -> current preference
    total_keys: int
    categories: list[str]  # All unique categories across preferences

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "total_keys": self.total_keys,
            "categories": self.categories,
        }


class PreferenceManager:
    """Manages temporal user preferences with versioning and history.

    Preferences are different from facts:
    - Facts (orders, events) are immutable - they happened
    - Preferences are mutable - they change over time

    This manager handles:
    - Creating new preferences
    - Updating preferences (creates new version, expires old)
    - Getting current valid preference
    - Viewing preference history
    - Soft-deleting preferences

    All preferences use memory_type="preference" and store metadata:
    - preference_key: Unique key for this preference type
    - valid_from: When this version became active
    - valid_until: When this version expired (if superseded)
    - supersedes: memory_id of the previous version
    - version: Version number (1, 2, 3, ...)
    """

    # Reinforcement signal to apply to expired preferences
    EXPIRED_PREFERENCE_SIGNAL = -0.3

    # Reinforcement signal for newly set preferences
    NEW_PREFERENCE_SIGNAL = 0.2

    def __init__(
        self,
        storage: BaseStorage,
        flr: FLR | None = None,
        default_importance: float = 0.6,
        default_access_level: str = "private",
    ):
        """Initialize PreferenceManager.

        Args:
            storage: Storage backend for persistence
            flr: Optional FLR instance for reinforcement signals
            default_importance: Default importance for new preferences
            default_access_level: Default access level for preferences
        """
        self.storage = storage
        self.flr = flr
        self.default_importance = default_importance
        self.default_access_level = default_access_level

        # Track pending conflicts awaiting resolution
        self._pending_conflicts: dict[str, PreferenceConflict] = {}

    def set_preference(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        importance: float | None = None,
        access_level: str | None = None,
        confidence_score: float | None = None,
        session_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Set a new preference (or update existing one).

        This is a convenience method that checks if a preference exists
        and either creates it or updates it.

        Args:
            user_id: User identifier
            key: Preference key (e.g., "theme", "language", "notifications")
            value: Preference value/description
            agent_id: Optional agent that set this preference
            topics: Optional topics for categorization
            categories: Optional categories
            importance: Optional importance override
            access_level: Optional access level override
            confidence_score: Confidence in this preference (0.0-1.0)
            session_id: Session where preference was set
            extra_metadata: Additional metadata to store

        Returns:
            The created or updated preference Memory
        """
        existing = self.get_preference(user_id, key)

        if existing:
            result = self.update_preference(
                user_id=user_id,
                key=key,
                value=value,
                agent_id=agent_id,
                topics=topics,
                categories=categories,
                importance=importance,
                access_level=access_level,
                confidence_score=confidence_score,
                session_id=session_id,
                extra_metadata=extra_metadata,
            )
            return result.new_preference

        return self._create_preference(
            user_id=user_id,
            key=key,
            value=value,
            version=1,
            supersedes=None,
            agent_id=agent_id,
            topics=topics,
            categories=categories,
            importance=importance,
            access_level=access_level,
            confidence_score=confidence_score,
            session_id=session_id,
            extra_metadata=extra_metadata,
        )

    def update_preference(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        importance: float | None = None,
        access_level: str | None = None,
        confidence_score: float | None = None,
        session_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> PreferenceUpdate:
        """Update an existing preference, creating a new version.

        The old preference is:
        1. Marked as expired (valid_until set to now)
        2. Negatively reinforced (ranks lower in future queries)

        The new preference:
        1. Links to old via 'supersedes' field
        2. Has incremented version number
        3. Is positively reinforced

        Args:
            user_id: User identifier
            key: Preference key to update
            value: New preference value
            agent_id: Agent making the update
            topics: Optional new topics (inherits from old if not provided)
            categories: Optional new categories (inherits from old if not provided)
            importance: Optional importance override
            access_level: Optional access level override
            confidence_score: Confidence in this preference
            session_id: Session where update occurred
            extra_metadata: Additional metadata

        Returns:
            PreferenceUpdate with new and old preference details
        """
        now = datetime.now(timezone.utc)
        old_preference = self.get_preference(user_id, key)

        # Determine version and supersedes
        version = 1
        supersedes = None
        inherited_topics = topics
        inherited_categories = categories

        if old_preference:
            # Get old version number
            old_version = old_preference.semantic_metadata.get("version", 1)
            version = old_version + 1
            supersedes = old_preference.memory_id

            # Inherit topics/categories if not provided
            if inherited_topics is None:
                inherited_topics = old_preference.topics
            if inherited_categories is None:
                inherited_categories = old_preference.categories

            # Expire the old preference
            old_preference.semantic_metadata["valid_until"] = now.isoformat()
            old_preference.semantic_metadata["superseded_by"] = None  # Will be set after
            self.storage.update(old_preference)

            # Negatively reinforce old preference
            if self.flr:
                self.flr.reinforce(old_preference.memory_id, self.EXPIRED_PREFERENCE_SIGNAL)

        # Create new preference
        new_preference = self._create_preference(
            user_id=user_id,
            key=key,
            value=value,
            version=version,
            supersedes=supersedes,
            agent_id=agent_id,
            topics=inherited_topics,
            categories=inherited_categories,
            importance=importance,
            access_level=access_level,
            confidence_score=confidence_score,
            session_id=session_id,
            extra_metadata=extra_metadata,
        )

        # Update old preference with superseded_by reference
        if old_preference:
            old_preference.semantic_metadata["superseded_by"] = new_preference.memory_id
            self.storage.update(old_preference)

        return PreferenceUpdate(
            new_preference=new_preference,
            old_preference=old_preference,
            preference_key=key,
            version=version,
            superseded=old_preference is not None,
            user_id=user_id,
        )

    def get_preference(
        self,
        user_id: str,
        key: str,
        *,
        include_expired: bool = False,
    ) -> Memory | None:
        """Get the current (non-expired) preference for a key.

        Args:
            user_id: User identifier
            key: Preference key
            include_expired: If True, return most recent even if expired

        Returns:
            Current preference Memory, or None if not found
        """
        # Search for preferences with this key
        results = self.storage.search(
            user_id=user_id,
            memory_types=["preference"],
            limit=100,  # Get enough to find all versions
        )

        now = datetime.now(timezone.utc)
        candidates = []

        for mem in results:
            meta = mem.semantic_metadata
            if meta.get("preference_key") != key:
                continue

            # Check if expired
            valid_until = meta.get("valid_until")
            is_expired = False
            if valid_until:
                try:
                    expiry = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
                    is_expired = expiry < now
                except (ValueError, TypeError):
                    pass

            if is_expired and not include_expired:
                continue

            candidates.append((mem, meta.get("version", 1), is_expired))

        if not candidates:
            return None

        # Sort by version descending, prefer non-expired
        candidates.sort(key=lambda x: (not x[2], x[1]), reverse=True)
        return candidates[0][0]

    def get_preference_history(
        self,
        user_id: str,
        key: str,
        *,
        limit: int = 50,
    ) -> PreferenceHistory:
        """Get the complete history of a preference.

        Returns all versions in chronological order (oldest first).

        Args:
            user_id: User identifier
            key: Preference key
            limit: Maximum versions to return

        Returns:
            PreferenceHistory with all versions
        """
        results = self.storage.search(
            user_id=user_id,
            memory_types=["preference"],
            limit=limit * 2,  # Buffer for filtering
        )

        versions = []
        for mem in results:
            meta = mem.semantic_metadata
            if meta.get("preference_key") == key:
                versions.append(mem)

        # Sort by version number
        versions.sort(key=lambda m: m.semantic_metadata.get("version", 1))

        # Limit results
        versions = versions[:limit]

        # Get current (non-expired) preference
        current = self.get_preference(user_id, key)

        return PreferenceHistory(
            preference_key=key,
            user_id=user_id,
            versions=versions,
            current=current,
            total_versions=len(versions),
        )

    def delete_preference(
        self,
        user_id: str,
        key: str,
        *,
        hard_delete: bool = False,
    ) -> bool:
        """Delete a preference.

        By default, performs a soft delete (expires the preference).
        Use hard_delete=True to permanently remove.

        Args:
            user_id: User identifier
            key: Preference key
            hard_delete: If True, permanently remove from storage

        Returns:
            True if preference was found and deleted
        """
        preference = self.get_preference(user_id, key)
        if not preference:
            return False

        if hard_delete:
            # Get all versions and delete them
            history = self.get_preference_history(user_id, key)
            for version in history.versions:
                self.storage.delete(version.memory_id)
        else:
            # Soft delete - just expire it
            now = datetime.now(timezone.utc)
            preference.semantic_metadata["valid_until"] = now.isoformat()
            preference.semantic_metadata["deleted"] = True
            self.storage.update(preference)

            # Strongly negative reinforce
            if self.flr:
                self.flr.reinforce(preference.memory_id, -0.5)

        return True

    def list_preferences(
        self,
        user_id: str,
        *,
        category: str | None = None,
        include_expired: bool = False,
    ) -> PreferenceSummary:
        """List all current preferences for a user.

        Args:
            user_id: User identifier
            category: Optional filter by category
            include_expired: Include expired preferences

        Returns:
            PreferenceSummary with all current preferences
        """
        # Search for all preferences
        results = self.storage.search(
            user_id=user_id,
            memory_types=["preference"],
            categories=[category] if category else None,
            limit=500,
        )

        now = datetime.now(timezone.utc)
        preferences: dict[str, Memory] = {}
        all_categories: set[str] = set()

        for mem in results:
            meta = mem.semantic_metadata
            key = meta.get("preference_key")
            if not key:
                continue

            # Check expiration
            valid_until = meta.get("valid_until")
            is_expired = False
            if valid_until:
                try:
                    expiry = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
                    is_expired = expiry < now
                except (ValueError, TypeError):
                    pass

            # Check if deleted
            if meta.get("deleted") and not include_expired:
                continue

            if is_expired and not include_expired:
                continue

            # Track categories
            all_categories.update(mem.categories)

            # Keep highest version for each key
            if key not in preferences:
                preferences[key] = mem
            else:
                existing_version = preferences[key].semantic_metadata.get("version", 1)
                new_version = meta.get("version", 1)
                if new_version > existing_version:
                    preferences[key] = mem

        return PreferenceSummary(
            user_id=user_id,
            preferences=preferences,
            total_keys=len(preferences),
            categories=sorted(all_categories),
        )

    def merge_preferences(
        self,
        user_id: str,
        keys: list[str],
        new_key: str,
        new_value: str,
        *,
        delete_old: bool = True,
    ) -> Memory:
        """Merge multiple preferences into a single new preference.

        Useful for consolidating related preferences.

        Args:
            user_id: User identifier
            keys: List of preference keys to merge
            new_key: Key for the merged preference
            new_value: Value for the merged preference
            delete_old: If True, soft-delete the merged preferences

        Returns:
            The new merged preference
        """
        all_topics: set[str] = set()
        all_categories: set[str] = set()
        merged_from: list[str] = []

        for key in keys:
            pref = self.get_preference(user_id, key)
            if pref:
                all_topics.update(pref.topics)
                all_categories.update(pref.categories)
                merged_from.append(pref.memory_id)

                if delete_old:
                    self.delete_preference(user_id, key, hard_delete=False)

        return self._create_preference(
            user_id=user_id,
            key=new_key,
            value=new_value,
            version=1,
            supersedes=None,
            topics=list(all_topics),
            categories=list(all_categories),
            extra_metadata={
                "merged_from": merged_from,
                "merge_keys": keys,
            },
        )

    def _create_preference(
        self,
        user_id: str,
        key: str,
        value: str,
        version: int,
        supersedes: str | None,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        importance: float | None = None,
        access_level: str | None = None,
        confidence_score: float | None = None,
        session_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Create a new preference memory.

        Internal method - use set_preference or update_preference instead.
        """
        now = datetime.now(timezone.utc)

        # Build semantic metadata
        semantic_metadata: dict[str, Any] = {
            "preference_key": key,
            "version": version,
            "valid_from": now.isoformat(),
        }

        if supersedes:
            semantic_metadata["supersedes"] = supersedes

        if extra_metadata:
            semantic_metadata.update(extra_metadata)

        # Default topics include the key
        default_topics = ["preferences", key]
        if topics:
            default_topics = list(set(default_topics + topics))

        # Default categories
        default_categories = ["user_preferences"]
        if categories:
            default_categories = list(set(default_categories + categories))

        memory = Memory(
            memory_id=f"pref_{key}_{uuid.uuid4().hex[:8]}",
            content=value,
            memory_type="preference",
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            turn_index=version,  # Use turn_index for version tracking
            message_role="preference",
            topics=default_topics,
            categories=default_categories,
            importance=importance or self.default_importance,
            access_level=access_level or self.default_access_level,
            confidence_score=confidence_score,
            semantic_metadata=semantic_metadata,
        )

        self.storage.store(memory)

        # Positively reinforce new preference
        if self.flr:
            self.flr.reinforce(memory.memory_id, self.NEW_PREFERENCE_SIGNAL)

        return memory

    def get_stats(self) -> dict[str, Any]:
        """Get preference manager statistics."""
        return {
            "default_importance": self.default_importance,
            "default_access_level": self.default_access_level,
            "has_flr": self.flr is not None,
            "expired_signal": self.EXPIRED_PREFERENCE_SIGNAL,
            "new_signal": self.NEW_PREFERENCE_SIGNAL,
            "pending_conflicts": len(self._pending_conflicts),
        }

    # -------------------------------------------------------------------------
    # Multi-Agent Conflict Resolution
    # -------------------------------------------------------------------------

    def set_preference_with_conflict_check(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        agent_id: str | None = None,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.NEWER_WINS,
        llm_merge_fn: Callable[[Memory, Memory], str] | None = None,
        agent_priorities: dict[str, int] | None = None,
        conflict_threshold: float = 0.3,
        **kwargs: Any,
    ) -> PreferenceUpdate | ConflictResolutionResult:
        """Set preference with multi-agent conflict detection and resolution.

        When a different agent sets a preference that may conflict with an existing
        one, this method detects the conflict and resolves it according to the
        specified strategy.

        Example conflict scenarios:
        - Agent A: "User prefers premium products"
        - Agent B: "User is price-sensitive"

        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
            agent_id: Agent setting this preference
            conflict_strategy: How to resolve conflicts
            llm_merge_fn: Function to merge preferences using LLM (required for LLM_MERGE)
            agent_priorities: Dict mapping agent_id -> priority (higher = more trusted)
            conflict_threshold: Semantic similarity threshold for conflict detection
            **kwargs: Additional arguments passed to set_preference

        Returns:
            PreferenceUpdate if no conflict, ConflictResolutionResult if conflict resolved
        """
        existing = self.get_preference(user_id, key)

        # No existing preference - just set it
        if not existing:
            pref = self.set_preference(
                user_id=user_id,
                key=key,
                value=value,
                agent_id=agent_id,
                **kwargs,
            )
            return PreferenceUpdate(
                new_preference=pref,
                old_preference=None,
                preference_key=key,
                version=1,
                superseded=False,
                user_id=user_id,
            )

        # Check if this is from the same agent (not a conflict, just an update)
        if existing.agent_id == agent_id:
            return self.update_preference(
                user_id=user_id,
                key=key,
                value=value,
                agent_id=agent_id,
                **kwargs,
            )

        # Different agent - check for conflict
        is_conflict = self._detect_conflict(existing, value, conflict_threshold)

        if not is_conflict:
            # Not a conflict, treat as update
            return self.update_preference(
                user_id=user_id,
                key=key,
                value=value,
                agent_id=agent_id,
                **kwargs,
            )

        # Create conflict record
        conflict = self._create_conflict(
            user_id=user_id,
            key=key,
            existing=existing,
            new_value=value,
            new_agent=agent_id,
            strategy=conflict_strategy,
        )

        # Resolve based on strategy
        return self._resolve_conflict(
            conflict=conflict,
            strategy=conflict_strategy,
            llm_merge_fn=llm_merge_fn,
            agent_priorities=agent_priorities,
            **kwargs,
        )

    def _detect_conflict(
        self,
        existing: Memory,
        new_value: str,
        threshold: float,
    ) -> bool:
        """Detect if new preference conflicts with existing one.

        Uses simple heuristics to detect potential conflicts:
        - Contradictory keywords (premium vs budget, likes vs dislikes)
        - Semantic opposition indicators

        For more accurate detection, use embeddings or LLM.
        """
        existing_lower = existing.content.lower()
        new_lower = new_value.lower()

        # Contradiction pairs to check
        contradictions = [
            ("premium", "budget"),
            ("premium", "cheap"),
            ("premium", "price-sensitive"),
            ("expensive", "affordable"),
            ("expensive", "cheap"),
            ("likes", "dislikes"),
            ("prefers", "avoids"),
            ("wants", "doesn't want"),
            ("loves", "hates"),
            ("always", "never"),
            ("morning", "evening"),
            ("early", "late"),
            ("yes", "no"),
            ("true", "false"),
            ("dark", "light"),
            ("hot", "cold"),
            ("fast", "slow"),
            ("more", "less"),
            ("high", "low"),
            ("large", "small"),
            ("frequent", "rare"),
        ]

        for word1, word2 in contradictions:
            # Check if existing has word1 and new has word2
            if word1 in existing_lower and word2 in new_lower:
                return True
            # Check reverse
            if word2 in existing_lower and word1 in new_lower:
                return True

        # Check for explicit negation patterns
        negation_patterns = [
            ("not ", ""),
            ("doesn't ", "does "),
            ("don't ", "do "),
            ("isn't ", "is "),
            ("aren't ", "are "),
            ("won't ", "will "),
            ("can't ", "can "),
            ("no longer", "still"),
        ]

        for neg, pos in negation_patterns:
            # If one has negation and other doesn't for same base phrase
            if neg in existing_lower and pos in new_lower:
                return True
            if neg in new_lower and pos in existing_lower:
                return True

        return False

    def _create_conflict(
        self,
        user_id: str,
        key: str,
        existing: Memory,
        new_value: str,
        new_agent: str | None,
        strategy: ConflictResolutionStrategy,
    ) -> PreferenceConflict:
        """Create a conflict record."""
        now = datetime.now(timezone.utc)

        # Create temporary memory for the new preference
        new_preference = Memory(
            memory_id=f"conflict_{uuid.uuid4().hex[:8]}",
            content=new_value,
            memory_type="preference",
            user_id=user_id,
            agent_id=new_agent,
            created_at=now,
            semantic_metadata={
                "preference_key": key,
                "is_conflict_candidate": True,
            },
        )

        conflict = PreferenceConflict(
            conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            preference_key=key,
            existing_preference=existing,
            conflicting_preference=new_preference,
            existing_agent=existing.agent_id,
            conflicting_agent=new_agent,
            status=ConflictStatus.DETECTED,
            resolution_strategy=strategy,
            resolved_preference=None,
            detected_at=now,
            resolved_at=None,
            resolution_reason=None,
            requires_human_review=(strategy == ConflictResolutionStrategy.HUMAN_REVIEW),
        )

        # Track pending conflicts
        self._pending_conflicts[conflict.conflict_id] = conflict

        return conflict

    def _resolve_conflict(
        self,
        conflict: PreferenceConflict,
        strategy: ConflictResolutionStrategy,
        llm_merge_fn: Callable[[Memory, Memory], str] | None = None,
        agent_priorities: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> ConflictResolutionResult:
        """Resolve a preference conflict according to strategy."""
        now = datetime.now(timezone.utc)
        resolved_pref: Memory | None = None
        message = ""

        if strategy == ConflictResolutionStrategy.NEWER_WINS:
            # New preference wins - update to new value
            result = self.update_preference(
                user_id=conflict.user_id,
                key=conflict.preference_key,
                value=conflict.conflicting_preference.content,
                agent_id=conflict.conflicting_agent,
                extra_metadata={
                    "conflict_resolved": True,
                    "resolution_strategy": strategy.value,
                    "superseded_agent": conflict.existing_agent,
                },
                **kwargs,
            )
            resolved_pref = result.new_preference
            message = "Newer preference from agent wins"
            conflict.status = ConflictStatus.RESOLVED

        elif strategy == ConflictResolutionStrategy.HIGHER_CONFIDENCE:
            # Compare confidence scores
            existing_conf = conflict.existing_preference.confidence_score or 0.5
            new_conf = conflict.conflicting_preference.confidence_score or 0.5

            if new_conf > existing_conf:
                result = self.update_preference(
                    user_id=conflict.user_id,
                    key=conflict.preference_key,
                    value=conflict.conflicting_preference.content,
                    agent_id=conflict.conflicting_agent,
                    extra_metadata={
                        "conflict_resolved": True,
                        "resolution_strategy": strategy.value,
                        "winning_confidence": new_conf,
                    },
                    **kwargs,
                )
                resolved_pref = result.new_preference
                message = f"Higher confidence ({new_conf:.2f} > {existing_conf:.2f}) wins"
            else:
                resolved_pref = conflict.existing_preference
                message = (
                    f"Existing preference kept (confidence {existing_conf:.2f} >= {new_conf:.2f})"
                )

            conflict.status = ConflictStatus.RESOLVED

        elif strategy == ConflictResolutionStrategy.AGENT_PRIORITY:
            # Use agent priority ranking
            priorities = agent_priorities or {}
            existing_priority = priorities.get(conflict.existing_agent or "", 0)
            new_priority = priorities.get(conflict.conflicting_agent or "", 0)

            if new_priority > existing_priority:
                result = self.update_preference(
                    user_id=conflict.user_id,
                    key=conflict.preference_key,
                    value=conflict.conflicting_preference.content,
                    agent_id=conflict.conflicting_agent,
                    extra_metadata={
                        "conflict_resolved": True,
                        "resolution_strategy": strategy.value,
                        "agent_priority": new_priority,
                    },
                    **kwargs,
                )
                resolved_pref = result.new_preference
                message = f"Higher priority agent ({conflict.conflicting_agent}) wins"
            else:
                resolved_pref = conflict.existing_preference
                message = f"Existing preference kept (agent priority {existing_priority} >= {new_priority})"

            conflict.status = ConflictStatus.RESOLVED

        elif strategy == ConflictResolutionStrategy.LLM_MERGE:
            if llm_merge_fn is None:
                # Fall back to NEWER_WINS if no LLM function provided
                return self._resolve_conflict(
                    conflict=conflict,
                    strategy=ConflictResolutionStrategy.NEWER_WINS,
                    **kwargs,
                )

            # Use LLM to merge preferences
            merged_value = llm_merge_fn(
                conflict.existing_preference,
                conflict.conflicting_preference,
            )

            result = self.update_preference(
                user_id=conflict.user_id,
                key=conflict.preference_key,
                value=merged_value,
                agent_id=conflict.conflicting_agent,  # New agent takes ownership
                extra_metadata={
                    "conflict_resolved": True,
                    "resolution_strategy": strategy.value,
                    "merged_from_agents": [
                        conflict.existing_agent,
                        conflict.conflicting_agent,
                    ],
                    "llm_merged": True,
                },
                **kwargs,
            )
            resolved_pref = result.new_preference
            message = "Preferences merged using LLM"
            conflict.status = ConflictStatus.MERGED

        elif strategy == ConflictResolutionStrategy.HUMAN_REVIEW:
            # Flag for human review - don't resolve yet
            conflict.status = ConflictStatus.PENDING_REVIEW
            conflict.requires_human_review = True
            message = "Conflict flagged for human review"
            # Store the conflicting preference temporarily
            self.storage.store(conflict.conflicting_preference)

        elif strategy == ConflictResolutionStrategy.KEEP_BOTH:
            # Create new preference with different key
            new_key = f"{conflict.preference_key}_{conflict.conflicting_agent}"
            new_pref = self.set_preference(
                user_id=conflict.user_id,
                key=new_key,
                value=conflict.conflicting_preference.content,
                agent_id=conflict.conflicting_agent,
                extra_metadata={
                    "original_key": conflict.preference_key,
                    "conflict_preserved": True,
                    "conflict_with": conflict.existing_preference.memory_id,
                },
                **kwargs,
            )
            resolved_pref = new_pref
            message = f"Both preferences kept (new key: {new_key})"
            conflict.status = ConflictStatus.RESOLVED

        # Update conflict record
        conflict.resolved_at = now
        conflict.resolved_preference = resolved_pref
        conflict.resolution_reason = message

        # Remove from pending if resolved
        if conflict.status in (ConflictStatus.RESOLVED, ConflictStatus.MERGED):
            self._pending_conflicts.pop(conflict.conflict_id, None)

        return ConflictResolutionResult(
            conflict=conflict,
            resolved=(conflict.status != ConflictStatus.PENDING_REVIEW),
            final_preference=resolved_pref,
            strategy_used=strategy,
            message=message,
        )

    def resolve_pending_conflict(
        self,
        conflict_id: str,
        chosen_preference: str,  # "existing" or "new" or custom value
        resolution_note: str | None = None,
    ) -> ConflictResolutionResult:
        """Resolve a conflict that was flagged for human review.

        Args:
            conflict_id: ID of the pending conflict
            chosen_preference: Which preference to keep:
                - "existing": Keep the original preference
                - "new": Use the conflicting preference
                - Any other string: Use as custom merged value
            resolution_note: Optional note about the resolution

        Returns:
            ConflictResolutionResult
        """
        conflict = self._pending_conflicts.get(conflict_id)
        if not conflict:
            raise ValueError(f"No pending conflict found with ID: {conflict_id}")

        now = datetime.now(timezone.utc)

        if chosen_preference == "existing":
            resolved_pref = conflict.existing_preference
            message = "Human chose existing preference"
        elif chosen_preference == "new":
            result = self.update_preference(
                user_id=conflict.user_id,
                key=conflict.preference_key,
                value=conflict.conflicting_preference.content,
                agent_id=conflict.conflicting_agent,
                extra_metadata={
                    "human_resolved": True,
                    "resolution_note": resolution_note,
                },
            )
            resolved_pref = result.new_preference
            message = "Human chose new preference"
            # Clean up temporary conflicting preference
            try:
                self.storage.delete(conflict.conflicting_preference.memory_id)
            except Exception:
                pass
        else:
            # Custom merged value
            result = self.update_preference(
                user_id=conflict.user_id,
                key=conflict.preference_key,
                value=chosen_preference,
                extra_metadata={
                    "human_resolved": True,
                    "human_merged": True,
                    "resolution_note": resolution_note,
                },
            )
            resolved_pref = result.new_preference
            message = "Human provided custom resolution"
            # Clean up temporary conflicting preference
            try:
                self.storage.delete(conflict.conflicting_preference.memory_id)
            except Exception:
                pass

        # Update conflict record
        conflict.status = ConflictStatus.RESOLVED
        conflict.resolved_at = now
        conflict.resolved_preference = resolved_pref
        conflict.resolution_reason = f"{message}. {resolution_note or ''}"

        # Remove from pending
        self._pending_conflicts.pop(conflict_id, None)

        return ConflictResolutionResult(
            conflict=conflict,
            resolved=True,
            final_preference=resolved_pref,
            strategy_used=ConflictResolutionStrategy.HUMAN_REVIEW,
            message=message,
        )

    def get_pending_conflicts(
        self,
        user_id: str | None = None,
    ) -> list[PreferenceConflict]:
        """Get all pending conflicts awaiting human review.

        Args:
            user_id: Optional filter by user

        Returns:
            List of pending conflicts
        """
        conflicts = list(self._pending_conflicts.values())

        if user_id:
            conflicts = [c for c in conflicts if c.user_id == user_id]

        return conflicts

    def dismiss_conflict(self, conflict_id: str, reason: str = "Not a real conflict") -> bool:
        """Dismiss a conflict as not being a real conflict.

        Args:
            conflict_id: ID of the conflict to dismiss
            reason: Reason for dismissal

        Returns:
            True if dismissed, False if not found
        """
        conflict = self._pending_conflicts.get(conflict_id)
        if not conflict:
            return False

        conflict.status = ConflictStatus.DISMISSED
        conflict.resolved_at = datetime.now(timezone.utc)
        conflict.resolution_reason = reason

        # Clean up temporary preference if stored
        try:
            self.storage.delete(conflict.conflicting_preference.memory_id)
        except Exception:
            pass

        self._pending_conflicts.pop(conflict_id, None)
        return True
