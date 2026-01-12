"""SVL Ontology - Core semantic definitions for the Structured Validation Layer.

The ontology defines all standardized values for memory metadata including:
- Message types and intents
- Temporal qualifiers
- Emotional classifications
- User roles and preference types
- Domain labels

All memories pass through this vocabulary layer for consistent semantic tagging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Role of the message sender in a conversation.

    These follow the standard LLM API message roles used by OpenAI, Anthropic,
    and other providers, plus additional roles for agentic workflows.
    """

    # Core LLM roles
    SYSTEM = "system"  # System instructions/prompts
    USER = "user"  # Human user input
    ASSISTANT = "assistant"  # AI model response

    # Tool/Function calling roles
    TOOL = "tool"  # Tool/function call request from assistant
    TOOL_RESULT = "tool_result"  # Result returned from tool execution
    FUNCTION = "function"  # Legacy function call (OpenAI style)
    FUNCTION_RESULT = "function_result"  # Legacy function result

    # Extended roles for agentic systems
    DEVELOPER = "developer"  # Developer-level instructions (Anthropic)
    CONTEXT = "context"  # Injected context (RAG results, retrieved memories)
    AGENT = "agent"  # Inter-agent communication
    ORCHESTRATOR = "orchestrator"  # Multi-agent orchestrator messages
    PLANNER = "planner"  # Planning/reasoning step
    EXECUTOR = "executor"  # Execution step
    CRITIC = "critic"  # Self-critique or evaluation
    SUMMARIZER = "summarizer"  # Summarization step

    # Memory-specific roles
    MEMORY = "memory"  # Retrieved memory injection
    PREFERENCE = "preference"  # User preference injection


class MessageType(str, Enum):
    """Types of messages in conversations."""

    # User-initiated
    QUERY = "query"  # User asking a question
    COMMAND = "command"  # User giving an instruction
    STATEMENT = "statement"  # User providing information
    FEEDBACK = "feedback"  # User giving feedback/opinion

    # Agent-initiated
    RESPONSE = "response"  # Agent answering a query
    CLARIFICATION = "clarification"  # Agent asking for more info
    SUGGESTION = "suggestion"  # Agent suggesting an action
    CONFIRMATION = "confirmation"  # Agent confirming understanding
    NOTIFICATION = "notification"  # Agent proactive notification

    # Tool/Function calling
    TOOL_CALL = "tool_call"  # Request to execute a tool
    TOOL_RESPONSE = "tool_response"  # Response from tool execution
    FUNCTION_CALL = "function_call"  # Legacy function call
    FUNCTION_RESPONSE = "function_response"  # Legacy function response

    # System
    SYSTEM = "system"  # System messages
    ERROR = "error"  # Error messages
    STATUS = "status"  # Status updates

    # Reasoning/Planning
    THOUGHT = "thought"  # Internal reasoning step (chain-of-thought)
    PLAN = "plan"  # Planning step
    REFLECTION = "reflection"  # Self-reflection
    EVALUATION = "evaluation"  # Quality evaluation


class MessageIntent(str, Enum):
    """Intent behind a message."""

    # Information seeking
    ASK_QUESTION = "ask_question"
    SEEK_CLARIFICATION = "seek_clarification"
    REQUEST_INFORMATION = "request_information"

    # Action oriented
    REQUEST_ACTION = "request_action"
    GIVE_COMMAND = "give_command"
    CANCEL_REQUEST = "cancel_request"

    # Information sharing
    PROVIDE_INFO = "provide_info"
    CORRECT_INFO = "correct_info"
    UPDATE_INFO = "update_info"

    # Opinion/Feedback
    EXPRESS_OPINION = "express_opinion"
    GIVE_FEEDBACK = "give_feedback"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"

    # Social
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    APOLOGY = "apology"

    # Confirmation
    CONFIRM = "confirm"
    DENY = "deny"
    ACCEPT = "accept"
    REJECT = "reject"


class TemporalQualifier(str, Enum):
    """Time-based qualifiers for memories."""

    # Frequency
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    RECURRING = "recurring"
    ONE_TIME = "one_time"

    # Reference
    PAST_EVENT = "past_event"
    CURRENT = "current"
    FUTURE_PLAN = "future_plan"
    DEADLINE = "deadline"

    # Duration
    SHORT_TERM = "short_term"  # Hours to days
    MEDIUM_TERM = "medium_term"  # Days to weeks
    LONG_TERM = "long_term"  # Weeks to months
    PERMANENT = "permanent"  # Indefinite

    # Session context
    THIS_SESSION = "this_session"
    CROSS_SESSION = "cross_session"


class EmotionalClassification(str, Enum):
    """Emotional content classification."""

    # Primary emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

    # Secondary emotions
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    CONFUSION = "confusion"
    EXCITEMENT = "excitement"
    DISAPPOINTMENT = "disappointment"
    RELIEF = "relief"
    ANXIETY = "anxiety"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

    # Neutral
    NEUTRAL = "neutral"


class UserRole(str, Enum):
    """User role classifications."""

    # Standard roles
    END_USER = "end_user"
    ADMIN = "admin"
    DEVELOPER = "developer"
    SUPPORT_AGENT = "support_agent"
    MANAGER = "manager"

    # Domain-specific
    CUSTOMER = "customer"
    PROSPECT = "prospect"
    VIP = "vip"
    INTERNAL = "internal"
    EXTERNAL = "external"

    # Technical
    API_USER = "api_user"
    SYSTEM = "system"
    SERVICE = "service"


class PreferenceType(str, Enum):
    """Types of user preferences."""

    # Communication
    COMMUNICATION_STYLE = "communication_style"
    LANGUAGE = "language"
    TONE = "tone"
    VERBOSITY = "verbosity"
    FORMAT = "format"

    # Behavior
    NOTIFICATION = "notification"
    REMINDER = "reminder"
    SCHEDULE = "schedule"
    FREQUENCY = "frequency"

    # Interface
    THEME = "theme"
    LAYOUT = "layout"
    ACCESSIBILITY = "accessibility"

    # Privacy
    DATA_SHARING = "data_sharing"
    VISIBILITY = "visibility"
    RETENTION = "retention"

    # Domain-specific
    PRODUCT = "product"
    SERVICE = "service"
    PAYMENT = "payment"
    SHIPPING = "shipping"


class DomainLabel(str, Enum):
    """High-level domain classifications."""

    # Business domains
    CUSTOMER_SERVICE = "customer_service"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    OPERATIONS = "operations"
    HR = "hr"

    # Technical domains
    ENGINEERING = "engineering"
    PRODUCT = "product"
    SECURITY = "security"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"

    # Industry domains
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    RETAIL = "retail"
    LEGAL = "legal"
    MEDIA = "media"

    # General
    GENERAL = "general"
    PERSONAL = "personal"


class Urgency(str, Enum):
    """Urgency levels for memories."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action required soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # No rush
    INFORMATIONAL = "informational"  # No action needed


class Confidence(str, Enum):
    """Confidence levels for extracted information."""

    HIGH = "high"  # Very confident
    MEDIUM = "medium"  # Reasonably confident
    LOW = "low"  # Uncertain
    INFERRED = "inferred"  # Derived, not explicit


@dataclass
class SemanticMetadata:
    """Complete semantic metadata for a memory.

    This represents the full SVL metadata that can be attached to any memory.
    All fields are designed to be stored as PostgreSQL columns with appropriate indexes.
    """

    # =========================================================================
    # Conversation tracking (required for proper message threading)
    # =========================================================================
    session_id: str | None = None  # Current session identifier
    thread_id: str | None = None  # Conversation thread (groups related messages)
    parent_memory_id: str | None = None  # For reply chains / threading
    turn_index: int | None = None  # Position in conversation (0-based)

    # =========================================================================
    # Message context
    # =========================================================================
    message_role: MessageRole | str | None = None  # Who sent this message
    message_type: MessageType | str | None = None  # What kind of message
    message_intent: MessageIntent | str | None = None  # Purpose of message

    # =========================================================================
    # Confidence and quality metrics
    # =========================================================================
    confidence_score: float | None = None  # 0.0-1.0 numeric confidence
    confidence: Confidence | str | None = None  # Categorical confidence level
    quality_score: float | None = None  # 0.0-1.0 quality assessment
    relevance_score: float | None = None  # 0.0-1.0 relevance to context

    # =========================================================================
    # Temporal
    # =========================================================================
    temporal_qualifier: TemporalQualifier | str | None = None
    expires_at: str | None = None  # ISO datetime string
    valid_from: str | None = None  # When this info becomes valid
    valid_until: str | None = None  # When this info expires

    # =========================================================================
    # Emotional
    # =========================================================================
    emotional_classification: EmotionalClassification | str | None = None
    emotional_intensity: float = 0.5  # 0-1 scale

    # =========================================================================
    # User context
    # =========================================================================
    user_role: UserRole | str | None = None
    preference_type: PreferenceType | str | None = None

    # =========================================================================
    # Domain
    # =========================================================================
    domain_label: DomainLabel | str | None = None
    subdomain: str | None = None

    # =========================================================================
    # Priority and urgency
    # =========================================================================
    urgency: Urgency | str | None = None
    priority: int | None = None  # Numeric priority (higher = more important)

    # =========================================================================
    # Tool/Function calling metadata
    # =========================================================================
    tool_name: str | None = None  # Name of tool being called
    tool_call_id: str | None = None  # Unique ID for tool call correlation
    function_name: str | None = None  # Legacy function name
    function_call_id: str | None = None  # Legacy function call ID

    # =========================================================================
    # Source and provenance
    # =========================================================================
    source: str | None = None  # Where this memory came from
    source_type: str | None = None  # Type of source (api, user, system, etc.)
    model_id: str | None = None  # Model that generated this (if AI-generated)
    model_version: str | None = None  # Version of the model

    # =========================================================================
    # Custom extensions
    # =========================================================================
    custom_tags: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def _enum_to_value(self, value: Enum | str | None) -> str | None:
        """Convert enum to string value."""
        if value is None:
            return None
        return value.value if isinstance(value, Enum) else value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        result: dict[str, Any] = {}

        # Conversation tracking
        if self.session_id:
            result["session_id"] = self.session_id
        if self.thread_id:
            result["thread_id"] = self.thread_id
        if self.parent_memory_id:
            result["parent_memory_id"] = self.parent_memory_id
        if self.turn_index is not None:
            result["turn_index"] = self.turn_index

        # Message context
        if self.message_role:
            result["message_role"] = self._enum_to_value(self.message_role)
        if self.message_type:
            result["message_type"] = self._enum_to_value(self.message_type)
        if self.message_intent:
            result["message_intent"] = self._enum_to_value(self.message_intent)

        # Confidence and quality
        if self.confidence_score is not None:
            result["confidence_score"] = self.confidence_score
        if self.confidence:
            result["confidence"] = self._enum_to_value(self.confidence)
        if self.quality_score is not None:
            result["quality_score"] = self.quality_score
        if self.relevance_score is not None:
            result["relevance_score"] = self.relevance_score

        # Temporal
        if self.temporal_qualifier:
            result["temporal_qualifier"] = self._enum_to_value(self.temporal_qualifier)
        if self.expires_at:
            result["expires_at"] = self.expires_at
        if self.valid_from:
            result["valid_from"] = self.valid_from
        if self.valid_until:
            result["valid_until"] = self.valid_until

        # Emotional
        if self.emotional_classification:
            result["emotional_classification"] = self._enum_to_value(self.emotional_classification)
        if self.emotional_intensity != 0.5:
            result["emotional_intensity"] = self.emotional_intensity

        # User context
        if self.user_role:
            result["user_role"] = self._enum_to_value(self.user_role)
        if self.preference_type:
            result["preference_type"] = self._enum_to_value(self.preference_type)

        # Domain
        if self.domain_label:
            result["domain_label"] = self._enum_to_value(self.domain_label)
        if self.subdomain:
            result["subdomain"] = self.subdomain

        # Priority and urgency
        if self.urgency:
            result["urgency"] = self._enum_to_value(self.urgency)
        if self.priority is not None:
            result["priority"] = self.priority

        # Tool/Function calling
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.function_name:
            result["function_name"] = self.function_name
        if self.function_call_id:
            result["function_call_id"] = self.function_call_id

        # Source and provenance
        if self.source:
            result["source"] = self.source
        if self.source_type:
            result["source_type"] = self.source_type
        if self.model_id:
            result["model_id"] = self.model_id
        if self.model_version:
            result["model_version"] = self.model_version

        # Custom extensions
        if self.custom_tags:
            result["custom_tags"] = self.custom_tags
        if self.custom_metadata:
            result["custom_metadata"] = self.custom_metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticMetadata:
        """Create from dictionary."""
        return cls(
            # Conversation tracking
            session_id=data.get("session_id"),
            thread_id=data.get("thread_id"),
            parent_memory_id=data.get("parent_memory_id"),
            turn_index=data.get("turn_index"),
            # Message context
            message_role=data.get("message_role"),
            message_type=data.get("message_type"),
            message_intent=data.get("message_intent"),
            # Confidence and quality
            confidence_score=data.get("confidence_score"),
            confidence=data.get("confidence"),
            quality_score=data.get("quality_score"),
            relevance_score=data.get("relevance_score"),
            # Temporal
            temporal_qualifier=data.get("temporal_qualifier"),
            expires_at=data.get("expires_at"),
            valid_from=data.get("valid_from"),
            valid_until=data.get("valid_until"),
            # Emotional
            emotional_classification=data.get("emotional_classification"),
            emotional_intensity=data.get("emotional_intensity", 0.5),
            # User context
            user_role=data.get("user_role"),
            preference_type=data.get("preference_type"),
            # Domain
            domain_label=data.get("domain_label"),
            subdomain=data.get("subdomain"),
            # Priority and urgency
            urgency=data.get("urgency"),
            priority=data.get("priority"),
            # Tool/Function calling
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
            function_name=data.get("function_name"),
            function_call_id=data.get("function_call_id"),
            # Source and provenance
            source=data.get("source"),
            source_type=data.get("source_type"),
            model_id=data.get("model_id"),
            model_version=data.get("model_version"),
            # Custom extensions
            custom_tags=data.get("custom_tags", []),
            custom_metadata=data.get("custom_metadata", {}),
        )


# Convenience functions to get all values
def get_message_roles() -> list[str]:
    """Get all message role values."""
    return [r.value for r in MessageRole]


def get_message_types() -> list[str]:
    """Get all message type values."""
    return [t.value for t in MessageType]


def get_message_intents() -> list[str]:
    """Get all message intent values."""
    return [i.value for i in MessageIntent]


def get_temporal_qualifiers() -> list[str]:
    """Get all temporal qualifier values."""
    return [t.value for t in TemporalQualifier]


def get_emotional_classifications() -> list[str]:
    """Get all emotional classification values."""
    return [e.value for e in EmotionalClassification]


def get_user_roles() -> list[str]:
    """Get all user role values."""
    return [r.value for r in UserRole]


def get_preference_types() -> list[str]:
    """Get all preference type values."""
    return [p.value for p in PreferenceType]


def get_domain_labels() -> list[str]:
    """Get all domain label values."""
    return [d.value for d in DomainLabel]


def get_urgency_levels() -> list[str]:
    """Get all urgency level values."""
    return [u.value for u in Urgency]


def get_confidence_levels() -> list[str]:
    """Get all confidence level values."""
    return [c.value for c in Confidence]
