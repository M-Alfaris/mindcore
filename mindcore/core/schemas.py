"""
Data schemas and models for Mindcore framework.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, ClassVar
from datetime import datetime, timezone
from enum import Enum


@dataclass
class MetadataSchema:
    """
    Predefined metadata schema for consistent enrichment.

    The EnrichmentAgent uses these lists to assign standardized metadata,
    ensuring consistency across messages and enabling efficient retrieval.

    Developers can customize these lists based on their domain.
    """

    # Predefined topic categories (domain-specific)
    topics: List[str] = field(
        default_factory=lambda: [
            # General
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
            "refund",
            "cancellation",
            "billing",
            "payment",
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
        ]
    )

    # Predefined categories (high-level classification)
    categories: List[str] = field(
        default_factory=lambda: [
            "support",  # Customer support inquiries
            "billing",  # Payment, refunds, subscriptions
            "technical",  # API, integrations, bugs
            "account",  # User account management
            "product",  # Product features, demos
            "feedback",  # User feedback, suggestions
            "general",  # General conversation
            "urgent",  # High priority issues
        ]
    )

    # Predefined intents
    intents: List[str] = field(
        default_factory=lambda: [
            "ask_question",  # User asking a question
            "request_action",  # User requesting something to be done
            "provide_info",  # User providing information
            "express_opinion",  # User expressing opinion/feedback
            "complaint",  # User complaining about something
            "greeting",  # Greeting/farewell
            "confirmation",  # User confirming something
            "clarification",  # User asking for clarification
        ]
    )

    # Predefined sentiment values
    sentiments: List[str] = field(
        default_factory=lambda: [
            "positive",
            "negative",
            "neutral",
            "mixed",
        ]
    )

    def to_prompt_list(self) -> str:
        """Format schema as a prompt-friendly list for the LLM."""
        return f"""Available Topics: {', '.join(self.topics)}

Available Categories: {', '.join(self.categories)}

Available Intents: {', '.join(self.intents)}

Available Sentiments: {', '.join(self.sentiments)}"""

    def validate_topics(self, topics: List[str]) -> List[str]:
        """Filter topics to only include predefined ones."""
        valid = set(self.topics)
        return [t for t in topics if t.lower() in valid or t in valid]

    def validate_categories(self, categories: List[str]) -> List[str]:
        """Filter categories to only include predefined ones."""
        valid = set(self.categories)
        return [c for c in categories if c.lower() in valid or c in valid]

    def validate_intent(self, intent: Optional[str]) -> Optional[str]:
        """Validate intent is in predefined list."""
        if intent and intent.lower() in [i.lower() for i in self.intents]:
            return intent.lower()
        return None


# Default schema instance
DEFAULT_METADATA_SCHEMA = MetadataSchema()


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class MessageMetadata:
    """Metadata enrichment for messages."""

    topics: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    importance: float = 0.5
    sentiment: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    # Enrichment status tracking
    enrichment_failed: bool = False
    enrichment_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def is_enriched(self) -> bool:
        """Check if metadata was successfully enriched."""
        return not self.enrichment_failed and (
            bool(self.topics)
            or bool(self.categories)
            or bool(self.tags)
            or bool(self.entities)
            or self.intent is not None
        )


class KnowledgeVisibility(str, Enum):
    """Visibility levels for knowledge items in multi-agent mode."""

    PRIVATE = "private"  # Only the owning agent can access
    SHARED = "shared"  # Shared with specific agents/groups
    PUBLIC = "public"  # Accessible by all agents in organization


@dataclass
class Message:
    """
    Core message structure.

    Supports both single-agent and multi-agent deployments:
    - Single-agent: agent_id and visibility are optional/ignored
    - Multi-agent: agent_id identifies owning agent, visibility controls access
    """

    message_id: str
    user_id: str
    thread_id: str
    session_id: str
    role: MessageRole
    raw_text: str
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    created_at: Optional[datetime] = None

    # Multi-agent support (optional - ignored in single-agent mode)
    agent_id: Optional[str] = None  # Agent that created/owns this message
    visibility: str = "private"  # "private", "shared", "public"
    sharing_groups: List[str] = field(default_factory=list)  # Groups with access

    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
        if isinstance(self.metadata, dict):
            self.metadata = MessageMetadata(**self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "role": self.role.value,
            "raw_text": self.raw_text,
            "metadata": (
                self.metadata.to_dict()
                if isinstance(self.metadata, MessageMetadata)
                else self.metadata
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            # Multi-agent fields
            "agent_id": self.agent_id,
            "visibility": self.visibility,
            "sharing_groups": self.sharing_groups,
        }


@dataclass
class AssembledContext:
    """Assembled context from Context Assembler Agent."""

    assembled_context: str
    key_points: List[str]
    relevant_message_ids: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ContextRequest:
    """Request for context assembly."""

    user_id: str
    thread_id: str
    query: str
    max_messages: int = 50
    include_metadata: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class IngestRequest:
    """Request for message ingestion."""

    user_id: str
    thread_id: str
    session_id: str
    role: str
    text: str
    message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ThreadSummary:
    """
    Compressed summary of a thread/session.

    Used to reduce storage and speed up retrieval for older conversations.
    Instead of fetching all messages, the system can retrieve the summary.
    """

    summary_id: str
    user_id: str
    thread_id: str
    session_id: Optional[str] = None

    # Summary content
    summary: str = ""  # LLM-generated summary
    key_facts: List[str] = field(default_factory=list)  # Extractable facts
    topics: List[str] = field(default_factory=list)  # Aggregated topics
    categories: List[str] = field(default_factory=list)  # Aggregated categories
    overall_sentiment: str = "neutral"  # Overall sentiment

    # Metadata
    message_count: int = 0  # Original message count
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    summarized_at: Optional[datetime] = None

    # Entities extracted (order IDs, dates, etc.)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    # Example: {"order_ids": ["#12345"], "dates": ["2024-03-15"]}

    # Status
    messages_deleted: bool = False  # Whether raw messages were purged

    def __post_init__(self):
        """Post-initialization processing."""
        if self.summarized_at is None:
            self.summarized_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "topics": self.topics,
            "categories": self.categories,
            "overall_sentiment": self.overall_sentiment,
            "message_count": self.message_count,
            "first_message_at": (
                self.first_message_at.isoformat() if self.first_message_at else None
            ),
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "summarized_at": self.summarized_at.isoformat() if self.summarized_at else None,
            "entities": self.entities,
            "messages_deleted": self.messages_deleted,
        }

    def to_context_string(self) -> str:
        """Format summary for inclusion in AI context."""
        parts = [f"Summary: {self.summary}"]
        if self.key_facts:
            parts.append(f"Key facts: {'; '.join(self.key_facts)}")
        if self.topics:
            parts.append(f"Topics discussed: {', '.join(self.topics)}")
        return "\n".join(parts)


@dataclass
class UserPreferences:
    """
    Amendable user preferences.

    These can be updated by user request through the AI agent.
    Separate from read-only system data (orders, billing, etc.).
    """

    user_id: str

    # Communication preferences
    language: str = "en"
    timezone: str = "UTC"
    communication_style: str = "balanced"  # formal, casual, technical, balanced

    # Personalization
    interests: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    preferred_name: Optional[str] = None

    # Context hints (user-provided context that should always be included)
    custom_context: Dict[str, Any] = field(default_factory=dict)
    # Example: {"role": "developer", "company": "Acme Inc"}

    # Notification preferences
    notification_topics: List[str] = field(default_factory=list)

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Define which fields are amendable by AI agent
    AMENDABLE_FIELDS: ClassVar[Set[str]] = {
        "language",
        "timezone",
        "communication_style",
        "interests",
        "goals",
        "preferred_name",
        "custom_context",
        "notification_topics",
    }

    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at

    def update(self, field_name: str, value: Any) -> bool:
        """
        Update a preference field if amendable.

        Args:
            field_name: Name of the field to update
            value: New value for the field

        Returns:
            True if update succeeded, False if field not amendable
        """
        if field_name not in self.AMENDABLE_FIELDS:
            return False
        setattr(self, field_name, value)
        self.updated_at = datetime.now(timezone.utc)
        return True

    def add_to_list(self, field_name: str, value: str) -> bool:
        """Add a value to a list field (interests, goals, notification_topics)."""
        if field_name not in {"interests", "goals", "notification_topics"}:
            return False
        current_list = getattr(self, field_name, [])
        if value not in current_list:
            current_list.append(value)
            setattr(self, field_name, current_list)
            self.updated_at = datetime.now(timezone.utc)
        return True

    def remove_from_list(self, field_name: str, value: str) -> bool:
        """Remove a value from a list field."""
        if field_name not in {"interests", "goals", "notification_topics"}:
            return False
        current_list = getattr(self, field_name, [])
        if value in current_list:
            current_list.remove(value)
            setattr(self, field_name, current_list)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "language": self.language,
            "timezone": self.timezone,
            "communication_style": self.communication_style,
            "interests": self.interests,
            "goals": self.goals,
            "preferred_name": self.preferred_name,
            "custom_context": self.custom_context,
            "notification_topics": self.notification_topics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_context_string(self) -> str:
        """Format preferences for inclusion in AI context."""
        parts = []
        if self.preferred_name:
            parts.append(f"User prefers to be called: {self.preferred_name}")
        if self.language != "en":
            parts.append(f"Preferred language: {self.language}")
        if self.communication_style != "balanced":
            parts.append(f"Communication style: {self.communication_style}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests)}")
        if self.goals:
            parts.append(f"Goals: {', '.join(self.goals)}")
        if self.custom_context:
            for k, v in self.custom_context.items():
                parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else ""
