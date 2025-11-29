"""
Data schemas and models for Mindcore framework.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
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
    topics: List[str] = field(default_factory=lambda: [
        # General
        "greeting", "farewell", "thanks", "help", "feedback",
        # Support
        "issue", "bug", "error", "problem", "complaint",
        "refund", "cancellation", "billing", "payment",
        # Product
        "feature", "product", "service", "pricing", "demo",
        # Technical
        "api", "integration", "setup", "configuration", "documentation",
        # Account
        "account", "login", "password", "settings", "profile",
    ])

    # Predefined categories (high-level classification)
    categories: List[str] = field(default_factory=lambda: [
        "support",      # Customer support inquiries
        "billing",      # Payment, refunds, subscriptions
        "technical",    # API, integrations, bugs
        "account",      # User account management
        "product",      # Product features, demos
        "feedback",     # User feedback, suggestions
        "general",      # General conversation
        "urgent",       # High priority issues
    ])

    # Predefined intents
    intents: List[str] = field(default_factory=lambda: [
        "ask_question",     # User asking a question
        "request_action",   # User requesting something to be done
        "provide_info",     # User providing information
        "express_opinion",  # User expressing opinion/feedback
        "complaint",        # User complaining about something
        "greeting",         # Greeting/farewell
        "confirmation",     # User confirming something
        "clarification",    # User asking for clarification
    ])

    # Predefined sentiment values
    sentiments: List[str] = field(default_factory=lambda: [
        "positive", "negative", "neutral", "mixed",
    ])

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
            bool(self.topics) or bool(self.categories) or
            bool(self.tags) or bool(self.entities) or
            self.intent is not None
        )


@dataclass
class Message:
    """Core message structure."""
    message_id: str
    user_id: str
    thread_id: str
    session_id: str
    role: MessageRole
    raw_text: str
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    created_at: Optional[datetime] = None

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
            "metadata": self.metadata.to_dict() if isinstance(self.metadata, MessageMetadata) else self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
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
