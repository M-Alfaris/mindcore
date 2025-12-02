"""
Adaptive User Preferences Learner.

Automatically learns and updates user preferences based on enriched
message metadata, without explicit user instruction.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import Counter
import threading

from .schemas import UserPreferences, MessageMetadata
from .preferences_manager import PreferencesManager
from .vocabulary import Intent, Sentiment, CommunicationStyle, get_vocabulary
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreferenceSignal:
    """A signal that may indicate a user preference."""

    signal_type: str  # e.g., "topic_interest", "communication_style"
    value: str
    weight: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive preferences learning."""

    # Minimum signals before updating a preference
    min_signals_for_topic: int = 3
    min_signals_for_style: int = 5
    min_signals_for_interest: int = 3

    # Signal decay (older signals matter less)
    signal_decay_days: int = 30  # Signals older than this are discounted

    # Thresholds for preference updates
    topic_interest_threshold: float = 0.6  # Topic appears in >60% of messages
    style_consistency_threshold: float = 0.7  # Style consistent in >70% of messages

    # Maximum items in preference lists
    max_interests: int = 20
    max_notification_topics: int = 10


class AdaptivePreferencesLearner:
    """
    Learns user preferences from message metadata.

    Analyzes patterns in enriched message metadata to infer and update
    user preferences automatically.

    Example:
        >>> learner = AdaptivePreferencesLearner(preferences_manager, db)
        >>>
        >>> # Called after each message is enriched
        >>> learner.process_message_metadata(
        ...     user_id="user123",
        ...     metadata=enriched_message.metadata
        ... )
        >>>
        >>> # Periodically update preferences based on signals
        >>> updates = learner.apply_updates("user123")
    """

    def __init__(
        self,
        preferences_manager: PreferencesManager,
        db_manager,
        config: Optional[AdaptiveConfig] = None,
    ):
        """
        Initialize the adaptive preferences learner.

        Args:
            preferences_manager: PreferencesManager instance
            db_manager: Database manager for querying message history
            config: Optional configuration overrides
        """
        self.preferences = preferences_manager
        self.db = db_manager
        self.config = config or AdaptiveConfig()
        self.vocabulary = get_vocabulary()
        self._lock = threading.Lock()

        # In-memory signal buffers per user (for real-time learning)
        self._signal_buffers: Dict[str, List[PreferenceSignal]] = {}

    def process_message_metadata(self, user_id: str, metadata: MessageMetadata) -> None:
        """
        Process enriched message metadata and extract preference signals.

        Args:
            user_id: User identifier
            metadata: Enriched message metadata
        """
        signals = self._extract_signals(metadata)

        with self._lock:
            if user_id not in self._signal_buffers:
                self._signal_buffers[user_id] = []

            self._signal_buffers[user_id].extend(signals)

            # Keep buffer size manageable (last 500 signals)
            if len(self._signal_buffers[user_id]) > 500:
                self._signal_buffers[user_id] = self._signal_buffers[user_id][-500:]

        logger.debug(f"Extracted {len(signals)} signals for user {user_id}")

    def _extract_signals(self, metadata: MessageMetadata) -> List[PreferenceSignal]:
        """Extract preference signals from message metadata."""
        signals = []
        base_weight = metadata.importance if hasattr(metadata, "importance") else 0.5

        # Skip low importance messages for preference learning
        if base_weight < 0.3:
            return signals

        # Topic interest signals
        for topic in getattr(metadata, "topics", []):
            if topic and topic != "general":
                signals.append(
                    PreferenceSignal(signal_type="topic_interest", value=topic, weight=base_weight)
                )

        # Communication style signals from sentiment and intent
        sentiment = getattr(metadata, "sentiment", None)
        intent = getattr(metadata, "intent", None)

        if sentiment:
            # Handle both string and dict formats for sentiment
            sentiment_value = sentiment
            if isinstance(sentiment, dict):
                sentiment_value = sentiment.get("overall", "")
            style = self._infer_style_from_sentiment(sentiment_value)
            if style:
                signals.append(
                    PreferenceSignal(
                        signal_type="communication_style",
                        value=style,
                        weight=base_weight * 0.5,  # Lower weight for inferred style
                    )
                )

        # Entity-based interest signals
        # Entities can be either dicts with type/value keys or plain strings
        for entity in getattr(metadata, "entities", []):
            if isinstance(entity, dict):
                entity_type = entity.get("type", "")
                entity_value = entity.get("value", "")
            elif isinstance(entity, str):
                # Parse string format like "product: iPhone" or just "iPhone"
                if ":" in entity:
                    parts = entity.split(":", 1)
                    entity_type = parts[0].strip().lower()
                    entity_value = parts[1].strip()
                else:
                    # Plain string entity - treat as generic interest
                    entity_type = "entity"
                    entity_value = entity.strip()
            else:
                continue

            if entity_type in ["product", "service", "feature", "entity"]:
                signals.append(
                    PreferenceSignal(signal_type="interest", value=entity_value, weight=base_weight)
                )

        # Keyword-based interest signals (high importance keywords only)
        for keyword in getattr(metadata, "keywords", []):
            if keyword and len(keyword) > 3:  # Skip short keywords
                signals.append(
                    PreferenceSignal(
                        signal_type="keyword_interest",
                        value=keyword.lower(),
                        weight=base_weight * 0.3,
                    )
                )

        return signals

    def _infer_style_from_sentiment(self, sentiment: str) -> Optional[str]:
        """Infer communication style from sentiment patterns."""
        # This is a simplified inference - could be more sophisticated
        style_map = {
            "positive": "casual",
            "negative": "formal",  # Complaints tend to be more formal
            "neutral": "balanced",
        }
        return style_map.get(sentiment)

    def apply_updates(self, user_id: str, force: bool = False) -> List[Tuple[str, str, Any]]:
        """
        Apply learned preferences based on accumulated signals.

        Args:
            user_id: User identifier
            force: If True, apply updates even with fewer signals

        Returns:
            List of (field, action, value) tuples for updates applied
        """
        updates = []

        with self._lock:
            signals = self._signal_buffers.get(user_id, [])
            if not signals:
                return updates

            # Apply decay to signals
            now = datetime.now(timezone.utc)
            decay_cutoff = now - timedelta(days=self.config.signal_decay_days)

            valid_signals = []
            for signal in signals:
                if signal.timestamp >= decay_cutoff:
                    # Apply time-based decay
                    age_days = (now - signal.timestamp).days
                    decay_factor = 1.0 - (age_days / self.config.signal_decay_days * 0.5)
                    signal.weight *= decay_factor
                    valid_signals.append(signal)

            if not valid_signals:
                return updates

        # Group signals by type
        topic_signals = [s for s in valid_signals if s.signal_type == "topic_interest"]
        style_signals = [s for s in valid_signals if s.signal_type == "communication_style"]
        interest_signals = [s for s in valid_signals if s.signal_type == "interest"]
        keyword_signals = [s for s in valid_signals if s.signal_type == "keyword_interest"]

        # Update topic-based interests
        if len(topic_signals) >= self.config.min_signals_for_topic or force:
            topic_updates = self._compute_topic_updates(user_id, topic_signals)
            updates.extend(topic_updates)

        # Update communication style
        if len(style_signals) >= self.config.min_signals_for_style or force:
            style_update = self._compute_style_update(user_id, style_signals)
            if style_update:
                updates.append(style_update)

        # Update interests from entities and keywords
        all_interest_signals = interest_signals + keyword_signals
        if len(all_interest_signals) >= self.config.min_signals_for_interest or force:
            interest_updates = self._compute_interest_updates(user_id, all_interest_signals)
            updates.extend(interest_updates)

        return updates

    def _compute_topic_updates(
        self, user_id: str, signals: List[PreferenceSignal]
    ) -> List[Tuple[str, str, Any]]:
        """Compute topic-based preference updates."""
        updates = []

        # Weight-adjusted topic counts
        topic_weights = Counter()
        for signal in signals:
            topic_weights[signal.value] += signal.weight

        total_weight = sum(topic_weights.values())
        if total_weight == 0:
            return updates

        # Get current preferences
        prefs = self.preferences.get_preferences(user_id)
        current_topics = set(prefs.notification_topics)

        # Find topics that exceed threshold
        for topic, weight in topic_weights.most_common(self.config.max_notification_topics):
            ratio = weight / total_weight
            if ratio >= self.config.topic_interest_threshold:
                if topic not in current_topics:
                    # Add to notification topics
                    prefs.notification_topics.append(topic)
                    updates.append(("notification_topics", "add", topic))
                    logger.info(f"Learned notification topic '{topic}' for user {user_id}")

        # Save if updated
        if updates:
            self.preferences.db.save_preferences(prefs)

        return updates

    def _compute_style_update(
        self, user_id: str, signals: List[PreferenceSignal]
    ) -> Optional[Tuple[str, str, str]]:
        """Compute communication style update."""
        # Weight-adjusted style counts
        style_weights = Counter()
        for signal in signals:
            style_weights[signal.value] += signal.weight

        if not style_weights:
            return None

        total_weight = sum(style_weights.values())
        dominant_style, dominant_weight = style_weights.most_common(1)[0]

        # Check if dominant style is consistent enough
        if dominant_weight / total_weight >= self.config.style_consistency_threshold:
            prefs = self.preferences.get_preferences(user_id)
            if prefs.communication_style != dominant_style:
                success, _ = self.preferences.update_preference(
                    user_id, "communication_style", dominant_style
                )
                if success:
                    logger.info(
                        f"Learned communication style '{dominant_style}' for user {user_id}"
                    )
                    return ("communication_style", "set", dominant_style)

        return None

    def _compute_interest_updates(
        self, user_id: str, signals: List[PreferenceSignal]
    ) -> List[Tuple[str, str, Any]]:
        """Compute interest updates from entity and keyword signals."""
        updates = []

        # Weight-adjusted interest counts
        interest_weights = Counter()
        for signal in signals:
            interest_weights[signal.value] += signal.weight

        if not interest_weights:
            return updates

        # Get current preferences
        prefs = self.preferences.get_preferences(user_id)
        current_interests = set(prefs.interests)

        # Add top interests that aren't already tracked
        for interest, weight in interest_weights.most_common(self.config.max_interests):
            if weight >= 1.0 and interest not in current_interests:  # At least 1.0 total weight
                success, _ = self.preferences.add_interest(user_id, interest)
                if success:
                    updates.append(("interests", "add", interest))
                    logger.info(f"Learned interest '{interest}' for user {user_id}")

                    # Respect max interests limit
                    if len(prefs.interests) >= self.config.max_interests:
                        break

        return updates

    def get_signal_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of accumulated signals for a user.

        Returns:
            Dictionary with signal counts and top signals by type
        """
        with self._lock:
            signals = self._signal_buffers.get(user_id, [])

        if not signals:
            return {"total_signals": 0, "by_type": {}}

        # Group by type
        by_type = {}
        for signal in signals:
            if signal.signal_type not in by_type:
                by_type[signal.signal_type] = []
            by_type[signal.signal_type].append(signal)

        # Summarize each type
        summary = {"total_signals": len(signals), "by_type": {}}

        for signal_type, type_signals in by_type.items():
            weights = Counter()
            for s in type_signals:
                weights[s.value] += s.weight

            summary["by_type"][signal_type] = {
                "count": len(type_signals),
                "top_values": dict(weights.most_common(5)),
            }

        return summary

    def clear_signals(self, user_id: str) -> None:
        """Clear accumulated signals for a user."""
        with self._lock:
            if user_id in self._signal_buffers:
                del self._signal_buffers[user_id]


# Singleton instance
_learner: Optional[AdaptivePreferencesLearner] = None
_learner_lock = threading.Lock()


def get_adaptive_learner(
    preferences_manager: Optional[PreferencesManager] = None, db_manager=None
) -> AdaptivePreferencesLearner:
    """Get or create the singleton adaptive learner."""
    global _learner
    if _learner is None:
        with _learner_lock:
            if _learner is None:
                if preferences_manager is None or db_manager is None:
                    raise ValueError(
                        "First call to get_adaptive_learner requires "
                        "preferences_manager and db_manager"
                    )
                _learner = AdaptivePreferencesLearner(preferences_manager, db_manager)
    return _learner


def reset_adaptive_learner() -> None:
    """Reset the singleton learner (for testing)."""
    global _learner
    with _learner_lock:
        _learner = None
