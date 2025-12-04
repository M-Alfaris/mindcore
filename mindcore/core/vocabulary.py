"""VocabularyManager - Central vocabulary control for Mindcore.

Provides a single source of truth for topics, categories, intents, sentiments,
and entity types used throughout the system. Supports extensibility for
custom domains and integration with external systems.

Usage:
    from mindcore.core.vocabulary import VocabularyManager, get_vocabulary

    # Get the global vocabulary instance
    vocab = get_vocabulary()

    # Validate a topic
    if vocab.is_valid_topic("billing"):
        ...

    # Extend with custom vocabulary
    vocab.register_topics(["custom_topic1", "custom_topic2"], source="my_system")

    # Map external system values to Mindcore vocabulary
    vocab.add_topic_mapping("customer_service", "support")  # external -> internal

    # Get all topics (including custom)
    all_topics = vocab.get_topics()
"""

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


class VocabularySource(str, Enum):
    """Source of vocabulary entries."""

    CORE = "core"  # Built-in Mindcore vocabulary
    CONNECTOR = "connector"  # Registered by connectors
    USER = "user"  # User-defined extensions
    EXTERNAL = "external"  # Loaded from external system


class Intent(str, Enum):
    """Predefined message intents."""

    ASK_QUESTION = "ask_question"
    REQUEST_ACTION = "request_action"
    PROVIDE_INFO = "provide_info"
    EXPRESS_OPINION = "express_opinion"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class Sentiment(str, Enum):
    """Predefined sentiment values."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class CommunicationStyle(str, Enum):
    """User communication style preferences."""

    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    BALANCED = "balanced"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class EntityType(str, Enum):
    """Predefined entity types for extraction."""

    ORDER_ID = "order_id"
    INVOICE_ID = "invoice_id"
    TRACKING_NUMBER = "tracking_number"
    DATE = "date"
    AMOUNT = "amount"
    PRODUCT = "product"
    PERSON = "person"
    ORGANIZATION = "organization"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CUSTOM = "custom"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


@dataclass
class VocabularyEntry:
    """A vocabulary entry with metadata."""

    value: str
    source: VocabularySource
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    parent: str | None = None  # For hierarchical categories
    metadata: dict[str, Any] = field(default_factory=dict)


class VocabularyManager:
    """Central vocabulary manager for Mindcore.

    Manages topics, categories, intents, sentiments, and entity types
    with support for:
    - Core vocabulary (built-in)
    - Connector vocabulary (registered by connectors)
    - User extensions (custom domain-specific terms)
    - External mappings (integration with external systems)
    - Synonyms and aliases

    Thread-safe singleton pattern ensures consistency across the application.

    Example:
        >>> vocab = VocabularyManager()
        >>>
        >>> # Check if topic is valid
        >>> vocab.is_valid_topic("billing")
        True
        >>>
        >>> # Register custom topics
        >>> vocab.register_topics(["my_custom_topic"], source="my_app")
        >>>
        >>> # Map external values to internal vocabulary
        >>> vocab.add_topic_mapping("customer_inquiry", "support")
        >>> vocab.resolve_topic("customer_inquiry")
        "support"
        >>>
        >>> # Get vocabulary for LLM prompts
        >>> vocab.to_prompt_list()
        "Available Topics: billing, support, ..."
    """

    # Core topics - general purpose, always available
    CORE_TOPICS = [
        # General conversation
        "greeting",
        "farewell",
        "thanks",
        "help",
        "feedback",
        # Support
        "issue",
        "bug",
        "error",
        "problem",
        "complaint",
        # Billing & Orders (aligned with connectors)
        "billing",
        "payment",
        "refund",
        "cancellation",
        "orders",
        "order",
        "purchase",
        "delivery",
        "shipping",
        "tracking",
        "invoice",
        "subscription",
        "charge",
        "receipt",
        # Product
        "feature",
        "product",
        "service",
        "pricing",
        "demo",
        # Technical
        "api",
        "integration",
        "setup",
        "configuration",
        "documentation",
        # Account
        "account",
        "login",
        "password",
        "settings",
        "profile",
        "security",
        # General
        "general",
        "other",
        "unknown",
    ]

    # Core categories - high-level classification
    CORE_CATEGORIES = [
        "support",  # Customer support inquiries
        "billing",  # Payment, refunds, subscriptions
        "orders",  # Order management, delivery, tracking
        "technical",  # API, integrations, bugs
        "account",  # User account management
        "product",  # Product features, demos
        "feedback",  # User feedback, suggestions
        "general",  # General conversation
        "urgent",  # High priority issues
    ]

    def __init__(self):
        """Initialize vocabulary manager with core vocabulary."""
        self._lock = threading.RLock()

        # Topic management
        self._topics: dict[str, VocabularyEntry] = {}
        self._topic_mappings: dict[str, str] = {}  # external -> internal

        # Category management
        self._categories: dict[str, VocabularyEntry] = {}
        self._category_mappings: dict[str, str] = {}

        # Connector topic registry (for routing)
        self._connector_topics: dict[str, set[str]] = {}  # connector_name -> topics

        # Custom vocabulary loaders
        self._external_loaders: list[Callable[[], dict[str, Any]]] = []

        # Initialize core vocabulary
        self._initialize_core_vocabulary()

        logger.debug("VocabularyManager initialized with core vocabulary")

    def _initialize_core_vocabulary(self) -> None:
        """Initialize built-in vocabulary."""
        # Register core topics
        for topic in self.CORE_TOPICS:
            self._topics[topic] = VocabularyEntry(
                value=topic, source=VocabularySource.CORE, description=f"Core topic: {topic}"
            )

        # Register core categories
        for category in self.CORE_CATEGORIES:
            self._categories[category] = VocabularyEntry(
                value=category,
                source=VocabularySource.CORE,
                description=f"Core category: {category}",
            )

        # Set up common aliases/mappings
        self._topic_mappings.update(
            {
                "order": "orders",
                "ship": "shipping",
                "deliver": "delivery",
                "pay": "payment",
                "bill": "billing",
                "subscribe": "subscription",
                "cancel": "cancellation",
                "return": "refund",
                "pwd": "password",
                "auth": "login",
                "config": "configuration",
                "docs": "documentation",
                "doc": "documentation",
            }
        )

    # -------------------------------------------------------------------------
    # Topic Management
    # -------------------------------------------------------------------------

    def get_topics(self, include_sources: list[VocabularySource] | None = None) -> list[str]:
        """Get all registered topics.

        Args:
            include_sources: Optional filter by source types.
                            If None, returns all topics.

        Returns:
            List of topic strings.
        """
        with self._lock:
            if include_sources is None:
                return list(self._topics.keys())
            return [
                entry.value for entry in self._topics.values() if entry.source in include_sources
            ]

    def is_valid_topic(self, topic: str) -> bool:
        """Check if a topic is in the vocabulary."""
        with self._lock:
            normalized = topic.lower().strip()
            # Check direct match
            if normalized in self._topics:
                return True
            # Check mappings
            return normalized in self._topic_mappings

    def resolve_topic(self, topic: str) -> str | None:
        """Resolve a topic to its canonical form.

        Handles aliases and external mappings.

        Args:
            topic: Topic string (may be alias or external value)

        Returns:
            Canonical topic string, or None if not found.
        """
        with self._lock:
            normalized = topic.lower().strip()
            # Direct match
            if normalized in self._topics:
                return normalized
            # Check mappings
            if normalized in self._topic_mappings:
                return self._topic_mappings[normalized]
            return None

    def validate_topics(self, topics: list[str]) -> list[str]:
        """Validate and resolve a list of topics.

        Args:
            topics: List of topic strings.

        Returns:
            List of valid, resolved topics (invalid ones filtered out).
        """
        valid = []
        for topic in topics:
            resolved = self.resolve_topic(topic)
            if resolved and resolved not in valid:
                valid.append(resolved)
        return valid

    def register_topics(
        self, topics: list[str], source: str = "user", descriptions: dict[str, str] | None = None
    ) -> None:
        """Register new topics.

        Args:
            topics: List of topic strings to register.
            source: Source identifier (e.g., "user", "my_connector").
            descriptions: Optional descriptions for each topic.
        """
        with self._lock:
            vocab_source = VocabularySource.USER
            if source == "connector":
                vocab_source = VocabularySource.CONNECTOR
            elif source == "external":
                vocab_source = VocabularySource.EXTERNAL

            for topic in topics:
                normalized = topic.lower().strip()
                if normalized not in self._topics:
                    self._topics[normalized] = VocabularyEntry(
                        value=normalized,
                        source=vocab_source,
                        description=descriptions.get(topic) if descriptions else None,
                    )
                    logger.debug(f"Registered topic '{normalized}' from {source}")

    def add_topic_mapping(self, external_value: str, internal_topic: str) -> bool:
        """Add a mapping from external system value to internal topic.

        Args:
            external_value: Value from external system.
            internal_topic: Mindcore topic to map to.

        Returns:
            True if mapping added, False if internal_topic is invalid.
        """
        with self._lock:
            external = external_value.lower().strip()
            internal = internal_topic.lower().strip()

            if internal not in self._topics:
                logger.warning(f"Cannot map to unknown topic: {internal}")
                return False

            self._topic_mappings[external] = internal
            logger.debug(f"Added topic mapping: {external} -> {internal}")
            return True

    # -------------------------------------------------------------------------
    # Category Management
    # -------------------------------------------------------------------------

    def get_categories(self, include_sources: list[VocabularySource] | None = None) -> list[str]:
        """Get all registered categories."""
        with self._lock:
            if include_sources is None:
                return list(self._categories.keys())
            return [
                entry.value
                for entry in self._categories.values()
                if entry.source in include_sources
            ]

    def is_valid_category(self, category: str) -> bool:
        """Check if a category is valid."""
        with self._lock:
            normalized = category.lower().strip()
            if normalized in self._categories:
                return True
            return normalized in self._category_mappings

    def resolve_category(self, category: str) -> str | None:
        """Resolve a category to its canonical form."""
        with self._lock:
            normalized = category.lower().strip()
            if normalized in self._categories:
                return normalized
            if normalized in self._category_mappings:
                return self._category_mappings[normalized]
            return None

    def validate_categories(self, categories: list[str]) -> list[str]:
        """Validate and resolve a list of categories."""
        valid = []
        for category in categories:
            resolved = self.resolve_category(category)
            if resolved and resolved not in valid:
                valid.append(resolved)
        return valid

    def register_categories(
        self,
        categories: list[str],
        source: str = "user",
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Register new categories."""
        with self._lock:
            vocab_source = VocabularySource.USER
            if source == "connector":
                vocab_source = VocabularySource.CONNECTOR
            elif source == "external":
                vocab_source = VocabularySource.EXTERNAL

            for category in categories:
                normalized = category.lower().strip()
                if normalized not in self._categories:
                    self._categories[normalized] = VocabularyEntry(
                        value=normalized,
                        source=vocab_source,
                        description=descriptions.get(category) if descriptions else None,
                    )
                    logger.debug(f"Registered category '{normalized}' from {source}")

    # -------------------------------------------------------------------------
    # Connector Integration
    # -------------------------------------------------------------------------

    def register_connector_topics(self, connector_name: str, topics: list[str]) -> None:
        """Register topics handled by a connector.

        This allows the VocabularyManager to know which connector
        handles which topics for routing purposes.

        Args:
            connector_name: Name of the connector (e.g., "orders", "billing")
            topics: Topics this connector handles.
        """
        with self._lock:
            # Ensure topics are registered
            for topic in topics:
                normalized = topic.lower().strip()
                if normalized not in self._topics:
                    self._topics[normalized] = VocabularyEntry(
                        value=normalized,
                        source=VocabularySource.CONNECTOR,
                        description=f"Topic from {connector_name} connector",
                    )

            # Register connector -> topics mapping
            self._connector_topics[connector_name] = {t.lower().strip() for t in topics}
            logger.info(f"Registered connector '{connector_name}' with topics: {topics}")

    def get_connector_for_topics(self, topics: list[str]) -> list[str]:
        """Get connectors that handle the given topics.

        Args:
            topics: List of topics to check.

        Returns:
            List of connector names that handle any of the topics.
        """
        with self._lock:
            matching_connectors = []
            topic_set = {t.lower().strip() for t in topics}

            for connector_name, connector_topics in self._connector_topics.items():
                if topic_set & connector_topics:  # Intersection
                    matching_connectors.append(connector_name)

            return matching_connectors

    # -------------------------------------------------------------------------
    # Intent and Sentiment (Enum-based, strict)
    # -------------------------------------------------------------------------

    def get_intents(self) -> list[str]:
        """Get all valid intent values."""
        return Intent.values()

    def is_valid_intent(self, intent: str) -> bool:
        """Check if intent is valid."""
        return intent.lower().strip() in Intent.values()

    def resolve_intent(self, intent: str) -> str | None:
        """Resolve intent to canonical form."""
        normalized = intent.lower().strip()
        if normalized in Intent.values():
            return normalized
        # Common mappings
        intent_mappings = {
            "question": "ask_question",
            "ask": "ask_question",
            "request": "request_action",
            "action": "request_action",
            "info": "provide_info",
            "information": "provide_info",
            "opinion": "express_opinion",
            "confirm": "confirmation",
            "clarify": "clarification",
            "hello": "greeting",
            "hi": "greeting",
            "bye": "farewell",
            "goodbye": "farewell",
        }
        return intent_mappings.get(normalized)

    def get_sentiments(self) -> list[str]:
        """Get all valid sentiment values."""
        return Sentiment.values()

    def is_valid_sentiment(self, sentiment: str) -> bool:
        """Check if sentiment is valid."""
        return sentiment.lower().strip() in Sentiment.values()

    def get_entity_types(self) -> list[str]:
        """Get all valid entity types."""
        return EntityType.values()

    def get_communication_styles(self) -> list[str]:
        """Get all valid communication styles."""
        return CommunicationStyle.values()

    # -------------------------------------------------------------------------
    # LLM Prompt Generation
    # -------------------------------------------------------------------------

    def to_prompt_list(self, include_descriptions: bool = False) -> str:
        """Format vocabulary as a prompt-friendly list for the LLM.

        Args:
            include_descriptions: If True, include descriptions for each item.

        Returns:
            Formatted string for LLM prompts.
        """
        topics = self.get_topics()
        categories = self.get_categories()
        intents = self.get_intents()
        sentiments = self.get_sentiments()

        parts = [
            f"Available Topics: {', '.join(sorted(topics))}",
            "",
            f"Available Categories: {', '.join(sorted(categories))}",
            "",
            f"Available Intents: {', '.join(intents)}",
            "",
            f"Available Sentiments: {', '.join(sentiments)}",
        ]

        return "\n".join(parts)

    def to_json_schema(self) -> dict[str, Any]:
        """Generate JSON schema for vocabulary (useful for tool definitions).

        Returns:
            Dictionary with enum definitions for topics, categories, etc.
        """
        return {
            "topics": {
                "type": "array",
                "items": {"type": "string", "enum": self.get_topics()},
                "description": "Topics from the predefined vocabulary",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": self.get_categories()},
                "description": "Categories from the predefined vocabulary",
            },
            "intent": {"type": "string", "enum": self.get_intents(), "description": "User intent"},
            "sentiment": {
                "type": "string",
                "enum": self.get_sentiments(),
                "description": "Message sentiment",
            },
        }

    # -------------------------------------------------------------------------
    # External Integration
    # -------------------------------------------------------------------------

    def load_from_json(self, file_path: str) -> None:
        """Load vocabulary extensions from a JSON file.

        Expected format:
        {
            "topics": ["custom_topic1", "custom_topic2"],
            "categories": ["custom_category"],
            "topic_mappings": {"external_term": "internal_topic"},
            "category_mappings": {"external_cat": "internal_cat"}
        }

        Args:
            file_path: Path to JSON file.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Vocabulary file not found: {file_path}")
            return

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Load topics
            if "topics" in data:
                self.register_topics(data["topics"], source="external")

            # Load categories
            if "categories" in data:
                self.register_categories(data["categories"], source="external")

            # Load mappings
            if "topic_mappings" in data:
                for external, internal in data["topic_mappings"].items():
                    self.add_topic_mapping(external, internal)

            if "category_mappings" in data:
                for external, internal in data["category_mappings"].items():
                    with self._lock:
                        self._category_mappings[external.lower()] = internal.lower()

            logger.info(f"Loaded vocabulary extensions from {file_path}")

        except Exception as e:
            logger.exception(f"Failed to load vocabulary from {file_path}: {e}")

    def register_external_loader(self, loader: Callable[[], dict[str, Any]]) -> None:
        """Register a custom vocabulary loader function.

        The loader should return a dict with keys:
        - topics: List[str]
        - categories: List[str]
        - topic_mappings: Dict[str, str]

        This allows integration with external systems (databases, APIs, etc.)

        Args:
            loader: Callable that returns vocabulary data.
        """
        self._external_loaders.append(loader)

    def refresh_from_external(self) -> None:
        """Refresh vocabulary from all registered external loaders."""
        for loader in self._external_loaders:
            try:
                data = loader()
                if "topics" in data:
                    self.register_topics(data["topics"], source="external")
                if "categories" in data:
                    self.register_categories(data["categories"], source="external")
                if "topic_mappings" in data:
                    for ext, internal in data["topic_mappings"].items():
                        self.add_topic_mapping(ext, internal)
            except Exception as e:
                logger.exception(f"External vocabulary loader failed: {e}")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get vocabulary statistics."""
        with self._lock:
            return {
                "total_topics": len(self._topics),
                "total_categories": len(self._categories),
                "topic_mappings": len(self._topic_mappings),
                "category_mappings": len(self._category_mappings),
                "registered_connectors": list(self._connector_topics.keys()),
                "topics_by_source": {
                    source.value: len([e for e in self._topics.values() if e.source == source])
                    for source in VocabularySource
                },
            }


# Global singleton instance
_vocabulary_instance: VocabularyManager | None = None
_vocabulary_lock = threading.Lock()


def get_vocabulary() -> VocabularyManager:
    """Get the global VocabularyManager instance (thread-safe singleton).

    Returns:
        VocabularyManager instance.

    Example:
        >>> vocab = get_vocabulary()
        >>> vocab.is_valid_topic("billing")
        True
    """
    global _vocabulary_instance
    if _vocabulary_instance is None:
        with _vocabulary_lock:
            if _vocabulary_instance is None:
                _vocabulary_instance = VocabularyManager()
    return _vocabulary_instance


def reset_vocabulary() -> None:
    """Reset the global vocabulary instance (mainly for testing)."""
    global _vocabulary_instance
    with _vocabulary_lock:
        _vocabulary_instance = None


# Convenience exports
__all__ = [
    "CommunicationStyle",
    "EntityType",
    "Intent",
    "Sentiment",
    "VocabularyEntry",
    "VocabularyManager",
    "VocabularySource",
    "get_vocabulary",
    "reset_vocabulary",
]
