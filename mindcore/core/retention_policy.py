"""
Data Retention Policy Manager.

Manages data lifecycle including:
- Tier migration (mid-term to long-term memory)
- Data archival and deletion
- Importance decay with recency
- Context window management
"""
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import math

from .schemas import Message, MessageMetadata, ThreadSummary
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryTier(str, Enum):
    """Memory tier levels."""
    SHORT_TERM = "short_term"  # Recent messages, full detail
    MID_TERM = "mid_term"      # Summarized threads, enriched messages
    LONG_TERM = "long_term"    # Compressed summaries, key facts only
    ARCHIVE = "archive"        # Cold storage, minimal retrieval


@dataclass
class RetentionConfig:
    """Configuration for data retention policies."""
    # Tier thresholds (in days)
    short_term_days: int = 1       # Messages < 1 day old
    mid_term_days: int = 30        # Messages 1-30 days old
    long_term_days: int = 365      # Messages 30-365 days old
    archive_after_days: int = 365  # Archive after 1 year

    # Deletion policies
    delete_trivial_after_days: int = 7    # Delete low-importance messages after 7 days
    delete_archived_after_days: int = 730  # Delete archived data after 2 years

    # Summarization triggers
    summarize_thread_after_messages: int = 50  # Summarize thread after 50 messages
    summarize_thread_after_days: int = 7       # Summarize inactive thread after 7 days

    # Importance decay
    importance_half_life_days: float = 14.0  # Importance halves every 14 days
    min_importance: float = 0.05             # Minimum importance floor

    # Context window limits
    max_context_messages: int = 50   # Max messages in context window
    max_context_tokens: int = 8000   # Max tokens in context window (estimated)


@dataclass
class TierMigrationResult:
    """Result of a tier migration operation."""
    source_tier: MemoryTier
    target_tier: MemoryTier
    messages_migrated: int
    threads_summarized: int
    messages_deleted: int
    errors: List[str] = field(default_factory=list)


class RetentionPolicyManager:
    """
    Manages data retention and tier migration.

    Handles:
    - Automatic tier migration based on age and importance
    - Thread summarization for older conversations
    - Importance decay over time
    - Context window management

    Example:
        >>> policy = RetentionPolicyManager(db, config=RetentionConfig())
        >>>
        >>> # Run migration (typically scheduled)
        >>> result = policy.run_migration()
        >>> print(f"Migrated {result.messages_migrated} messages")
        >>>
        >>> # Get importance with decay
        >>> importance = policy.get_decayed_importance(message)
    """

    def __init__(
        self,
        db_manager,
        summarization_agent=None,
        config: Optional[RetentionConfig] = None
    ):
        """
        Initialize retention policy manager.

        Args:
            db_manager: Database manager instance
            summarization_agent: Optional SummarizationAgent for thread summaries
            config: Retention configuration
        """
        self.db = db_manager
        self.summarization_agent = summarization_agent
        self.config = config or RetentionConfig()
        self._lock = threading.Lock()

    def get_decayed_importance(
        self,
        message: Message,
        base_importance: Optional[float] = None,
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate importance with time-based decay.

        Uses exponential decay where importance halves every half_life_days.

        Args:
            message: Message to calculate importance for
            base_importance: Override base importance (uses message.metadata.importance if not provided)
            reference_time: Time to calculate decay from (defaults to now)

        Returns:
            Decayed importance score (0.0 to 1.0)
        """
        # Get base importance
        if base_importance is not None:
            importance = base_importance
        elif hasattr(message.metadata, 'importance'):
            importance = message.metadata.importance or 0.5
        else:
            importance = 0.5

        # Get message timestamp
        msg_time = message.created_at or message.timestamp
        if msg_time is None:
            return importance

        # Ensure timezone awareness
        if isinstance(msg_time, str):
            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)

        # Calculate reference time
        ref_time = reference_time or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_days = (ref_time - msg_time).total_seconds() / 86400

        if age_days <= 0:
            return importance

        # Exponential decay: importance * 2^(-age/half_life)
        decay_factor = math.pow(2, -age_days / self.config.importance_half_life_days)
        decayed = importance * decay_factor

        # Apply minimum floor
        return max(decayed, self.config.min_importance)

    def get_message_tier(self, message: Message) -> MemoryTier:
        """
        Determine the appropriate tier for a message.

        Args:
            message: Message to categorize

        Returns:
            Appropriate MemoryTier
        """
        msg_time = message.created_at or message.timestamp
        if msg_time is None:
            return MemoryTier.SHORT_TERM

        if isinstance(msg_time, str):
            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - msg_time).total_seconds() / 86400

        if age_days < self.config.short_term_days:
            return MemoryTier.SHORT_TERM
        elif age_days < self.config.mid_term_days:
            return MemoryTier.MID_TERM
        elif age_days < self.config.long_term_days:
            return MemoryTier.LONG_TERM
        else:
            return MemoryTier.ARCHIVE

    def should_delete_message(self, message: Message) -> bool:
        """
        Check if a message should be deleted based on retention policy.

        Args:
            message: Message to check

        Returns:
            True if message should be deleted
        """
        msg_time = message.created_at or message.timestamp
        if msg_time is None:
            return False

        if isinstance(msg_time, str):
            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - msg_time).total_seconds() / 86400

        # Check trivial message deletion
        importance = getattr(message.metadata, 'importance', 0.5) or 0.5
        if importance < 0.2 and age_days > self.config.delete_trivial_after_days:
            return True

        # Check archived data deletion
        if age_days > self.config.delete_archived_after_days:
            return True

        return False

    def should_summarize_thread(
        self,
        thread_id: str,
        message_count: int,
        last_message_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if a thread should be summarized.

        Args:
            thread_id: Thread identifier
            message_count: Number of messages in thread
            last_message_time: Time of last message

        Returns:
            True if thread should be summarized
        """
        # Check message count threshold
        if message_count >= self.config.summarize_thread_after_messages:
            return True

        # Check inactivity threshold
        if last_message_time:
            if isinstance(last_message_time, str):
                last_message_time = datetime.fromisoformat(
                    last_message_time.replace('Z', '+00:00')
                )
            if last_message_time.tzinfo is None:
                last_message_time = last_message_time.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            inactive_days = (now - last_message_time).total_seconds() / 86400

            if inactive_days >= self.config.summarize_thread_after_days:
                return True

        return False

    def run_migration(
        self,
        user_id: Optional[str] = None,
        dry_run: bool = False
    ) -> TierMigrationResult:
        """
        Run tier migration for all or specific user's data.

        Args:
            user_id: Optional user to migrate (all users if None)
            dry_run: If True, don't actually migrate, just report

        Returns:
            TierMigrationResult with migration statistics
        """
        result = TierMigrationResult(
            source_tier=MemoryTier.SHORT_TERM,
            target_tier=MemoryTier.MID_TERM,
            messages_migrated=0,
            threads_summarized=0,
            messages_deleted=0
        )

        with self._lock:
            try:
                # Get messages that need tier migration
                messages = self._get_messages_for_migration(user_id)

                for message in messages:
                    current_tier = self.get_message_tier(message)

                    # Check for deletion
                    if self.should_delete_message(message):
                        if not dry_run:
                            self._delete_message(message)
                        result.messages_deleted += 1
                        continue

                    # Update tier if needed
                    stored_tier = getattr(message.metadata, 'memory_tier', None)
                    if stored_tier != current_tier.value:
                        if not dry_run:
                            self._update_message_tier(message, current_tier)
                        result.messages_migrated += 1

                # Check for thread summarization
                threads = self._get_threads_for_summarization(user_id)
                for thread_info in threads:
                    thread_id = thread_info['thread_id']
                    msg_count = thread_info['message_count']
                    last_msg = thread_info.get('last_message_time')

                    if self.should_summarize_thread(thread_id, msg_count, last_msg):
                        if not dry_run:
                            self._summarize_thread(thread_info)
                        result.threads_summarized += 1

                logger.info(
                    f"Migration complete: {result.messages_migrated} migrated, "
                    f"{result.threads_summarized} threads summarized, "
                    f"{result.messages_deleted} deleted"
                )

            except Exception as e:
                result.errors.append(str(e))
                logger.error(f"Migration error: {e}")

        return result

    def _get_messages_for_migration(
        self,
        user_id: Optional[str] = None
    ) -> List[Message]:
        """Get messages that may need tier migration."""
        # Query messages older than short-term threshold
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.short_term_days
        )

        # This should be implemented in the database manager
        if hasattr(self.db, 'get_messages_before'):
            return self.db.get_messages_before(cutoff, user_id=user_id)

        # Fallback: fetch recent and filter
        if user_id:
            messages = self.db.fetch_messages_by_user(user_id, limit=1000)
        else:
            messages = []  # Would need all-users query

        return [m for m in messages if self.get_message_tier(m) != MemoryTier.SHORT_TERM]

    def _get_threads_for_summarization(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get threads that may need summarization."""
        if hasattr(self.db, 'get_thread_stats'):
            return self.db.get_thread_stats(user_id=user_id)
        return []

    def _update_message_tier(self, message: Message, tier: MemoryTier) -> bool:
        """Update message tier in database."""
        if not hasattr(message.metadata, 'memory_tier'):
            return False

        message.metadata.memory_tier = tier.value
        return self.db.update_message_metadata(
            message.message_id,
            message.metadata
        )

    def _delete_message(self, message: Message) -> bool:
        """Delete a message from database."""
        if hasattr(self.db, 'delete_message'):
            return self.db.delete_message(message.message_id)
        return False

    def _summarize_thread(self, thread_info: Dict[str, Any]) -> Optional[ThreadSummary]:
        """Summarize a thread using the summarization agent."""
        if not self.summarization_agent:
            return None

        thread_id = thread_info['thread_id']
        user_id = thread_info.get('user_id')

        # Get thread messages
        messages = self.db.fetch_thread_messages(thread_id, limit=100)
        if not messages:
            return None

        # Generate summary
        try:
            summary = self.summarization_agent.summarize_thread(messages)

            # Store summary
            if hasattr(self.db, 'save_thread_summary'):
                self.db.save_thread_summary(thread_id, summary)

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize thread {thread_id}: {e}")
            return None

    def get_context_window(
        self,
        user_id: str,
        thread_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_importance: float = 0.1
    ) -> List[Message]:
        """
        Get optimized context window for a conversation.

        Selects messages based on recency, importance, and token budget.

        Args:
            user_id: User identifier
            thread_id: Thread identifier
            max_messages: Maximum messages to include
            max_tokens: Maximum estimated tokens
            min_importance: Minimum decayed importance to include

        Returns:
            List of messages optimized for context window
        """
        max_msgs = max_messages or self.config.max_context_messages
        max_toks = max_tokens or self.config.max_context_tokens

        # Fetch recent messages
        messages = self.db.fetch_recent_messages(
            user_id, thread_id,
            limit=max_msgs * 2  # Fetch extra for filtering
        )

        # Score and filter messages
        scored_messages = []
        for msg in messages:
            decayed_importance = self.get_decayed_importance(msg)
            if decayed_importance >= min_importance:
                scored_messages.append((msg, decayed_importance))

        # Sort by importance (descending) but keep chronological order for ties
        scored_messages.sort(key=lambda x: (-x[1], x[0].created_at or x[0].timestamp))

        # Select messages within token budget
        selected = []
        estimated_tokens = 0

        for msg, importance in scored_messages:
            # Rough token estimation (4 chars per token)
            msg_tokens = len(msg.raw_text or msg.content or '') // 4

            if estimated_tokens + msg_tokens > max_toks:
                break
            if len(selected) >= max_msgs:
                break

            selected.append(msg)
            estimated_tokens += msg_tokens

        # Return in chronological order
        selected.sort(key=lambda m: m.created_at or m.timestamp or datetime.min)

        return selected


# Singleton instance
_policy_manager: Optional[RetentionPolicyManager] = None
_policy_lock = threading.Lock()


def get_retention_policy(
    db_manager=None,
    summarization_agent=None,
    config: Optional[RetentionConfig] = None
) -> RetentionPolicyManager:
    """Get or create the singleton retention policy manager."""
    global _policy_manager
    if _policy_manager is None:
        with _policy_lock:
            if _policy_manager is None:
                if db_manager is None:
                    raise ValueError(
                        "First call to get_retention_policy requires db_manager"
                    )
                _policy_manager = RetentionPolicyManager(
                    db_manager, summarization_agent, config
                )
    return _policy_manager


def reset_retention_policy() -> None:
    """Reset the singleton policy manager (for testing)."""
    global _policy_manager
    with _policy_lock:
        _policy_manager = None
