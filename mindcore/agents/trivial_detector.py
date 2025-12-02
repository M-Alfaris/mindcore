"""
Trivial message detector for auto-enrichment without LLM.

Detects greetings, confirmations, fillers, and other trivial messages
that don't require LLM enrichment. Auto-enriches them with low importance
metadata to reduce costs and improve performance.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.schemas import Message, MessageMetadata, MessageRole
from ..core.vocabulary import Intent, Sentiment, get_vocabulary


class TrivialCategory(str, Enum):
    """Categories of trivial messages."""

    GREETING = "greeting"
    FAREWELL = "farewell"
    CONFIRMATION = "confirmation"
    THANKS = "thanks"
    FILLER = "filler"
    QUESTION_WORD = "question_word"
    ACKNOWLEDGMENT = "acknowledgment"
    APOLOGY = "apology"


@dataclass
class TrivialMatch:
    """Result of trivial message detection."""

    is_trivial: bool
    category: Optional[TrivialCategory] = None
    confidence: float = 0.0
    matched_pattern: Optional[str] = None


class TrivialMessageDetector:
    """
    Detects trivial messages using regex patterns.

    Trivial messages are auto-enriched with low importance metadata,
    skipping the LLM call entirely. This reduces costs and latency
    while still properly categorizing these messages.

    Example:
        >>> detector = TrivialMessageDetector()
        >>> result = detector.detect("Hello!")
        >>> result.is_trivial
        True
        >>> result.category
        TrivialCategory.GREETING
    """

    # Pattern definitions: (pattern, category, base_importance)
    # Patterns are case-insensitive and match whole message or start/end
    PATTERNS: List[Tuple[str, TrivialCategory, float]] = [
        # Greetings (very low importance for retrieval)
        (r"^(hi|hey|hello|hiya|yo|howdy|greetings?)[\s!.,?]*$", TrivialCategory.GREETING, 0.1),
        (r"^good\s+(morning|afternoon|evening|day)[\s!.,?]*$", TrivialCategory.GREETING, 0.1),
        (r"^(what'?s\s+up|sup|wassup)[\s!.,?]*$", TrivialCategory.GREETING, 0.1),
        # Farewells
        (
            r"^(bye|goodbye|cya|see\s+you|later|take\s+care|cheers)[\s!.,?]*$",
            TrivialCategory.FAREWELL,
            0.1,
        ),
        (r"^(good\s*night|gn|ttyl|talk\s+later)[\s!.,?]*$", TrivialCategory.FAREWELL, 0.1),
        # Confirmations (low importance)
        (
            r"^(yes|yeah|yep|yup|yea|aye|affirmative|correct|right|exactly)[\s!.,?]*$",
            TrivialCategory.CONFIRMATION,
            0.15,
        ),
        (r"^(no|nope|nah|negative|wrong|incorrect)[\s!.,?]*$", TrivialCategory.CONFIRMATION, 0.15),
        (
            r"^(ok|okay|k|kk|alright|all\s*right|sure|fine|got\s+it)[\s!.,?]*$",
            TrivialCategory.CONFIRMATION,
            0.15,
        ),
        (
            r"^(go\s+ahead|proceed|continue|confirmed?|agreed?)[\s!.,?]*$",
            TrivialCategory.CONFIRMATION,
            0.15,
        ),
        (
            r"^(sounds?\s+good|works?\s+for\s+me|that'?s?\s+fine)[\s!.,?]*$",
            TrivialCategory.CONFIRMATION,
            0.15,
        ),
        # Thanks (low importance)
        (
            r"^(thanks?|thank\s+you|thx|ty|cheers|appreciated?)[\s!.,?]*$",
            TrivialCategory.THANKS,
            0.1,
        ),
        (r"^(thanks?\s+(a\s+lot|so\s+much|very\s+much))[\s!.,?]*$", TrivialCategory.THANKS, 0.1),
        (r"^(much\s+appreciated|many\s+thanks)[\s!.,?]*$", TrivialCategory.THANKS, 0.1),
        # Fillers (very low importance)
        (r"^(um+|uh+|hmm+|hm+|er+|ah+|oh+)[\s!.,?]*$", TrivialCategory.FILLER, 0.05),
        (r"^(well|so|anyway|anyways|like|basically)[\s!.,?]*$", TrivialCategory.FILLER, 0.05),
        (r"^(i\s+see|i\s+understand|understood|noted)[\s!.,?]*$", TrivialCategory.FILLER, 0.1),
        # Single question words (need context, low importance alone)
        (r"^(what|where|when|why|how|who|which)[\s?!.,]*$", TrivialCategory.QUESTION_WORD, 0.2),
        (r"^(pardon|sorry|excuse\s+me)[\s?!.,]*$", TrivialCategory.QUESTION_WORD, 0.15),
        # Acknowledgments
        (
            r"^(nice|great|awesome|cool|perfect|excellent|wonderful)[\s!.,?]*$",
            TrivialCategory.ACKNOWLEDGMENT,
            0.15,
        ),
        (r"^(interesting|fascinating|intriguing)[\s!.,?]*$", TrivialCategory.ACKNOWLEDGMENT, 0.15),
        (r"^(good|bad|amazing|terrible)[\s!.,?]*$", TrivialCategory.ACKNOWLEDGMENT, 0.15),
        # Apologies
        (r"^(sorry|apologies|my\s+bad|oops|whoops)[\s!.,?]*$", TrivialCategory.APOLOGY, 0.1),
        (r"^(i'?m\s+sorry|excuse\s+me|pardon\s+me)[\s!.,?]*$", TrivialCategory.APOLOGY, 0.1),
    ]

    # Compiled patterns for performance
    _compiled_patterns: List[Tuple[re.Pattern, TrivialCategory, float]] = []

    def __init__(
        self,
        custom_patterns: Optional[List[Tuple[str, TrivialCategory, float]]] = None,
        min_confidence: float = 0.8,
    ):
        """
        Initialize the trivial message detector.

        Args:
            custom_patterns: Additional patterns to detect
            min_confidence: Minimum confidence to consider a match
        """
        self.min_confidence = min_confidence

        # Compile all patterns
        all_patterns = self.PATTERNS.copy()
        if custom_patterns:
            all_patterns.extend(custom_patterns)

        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), category, importance)
            for pattern, category, importance in all_patterns
        ]

    def detect(self, text: str) -> TrivialMatch:
        """
        Detect if a message is trivial.

        Args:
            text: The message text to analyze

        Returns:
            TrivialMatch with detection results
        """
        if not text:
            return TrivialMatch(is_trivial=False)

        # Normalize text
        normalized = text.strip()

        # Check for very short messages (likely trivial)
        word_count = len(normalized.split())

        # Try each pattern
        for pattern, category, base_importance in self._compiled_patterns:
            match = pattern.match(normalized)
            if match:
                # Confidence based on how much of the text matched
                match_ratio = len(match.group()) / len(normalized) if normalized else 0
                confidence = min(1.0, match_ratio * 1.2)  # Boost slightly

                if confidence >= self.min_confidence:
                    return TrivialMatch(
                        is_trivial=True,
                        category=category,
                        confidence=confidence,
                        matched_pattern=pattern.pattern,
                    )

        # Check for very short messages that might be trivial
        if word_count <= 2 and len(normalized) <= 15:
            # Could be trivial but not matched - return low confidence
            return TrivialMatch(is_trivial=False, confidence=0.3)

        return TrivialMatch(is_trivial=False)

    def auto_enrich(
        self,
        text: str,
        user_id: str,
        thread_id: str,
        session_id: str,
        role: str = "user",
        message_id: Optional[str] = None,
        **extra_metadata,
    ) -> Optional[Message]:
        """
        Auto-enrich a trivial message without LLM.

        Args:
            text: Message text
            user_id: User identifier
            thread_id: Thread identifier
            session_id: Session identifier
            role: Message role (user/assistant)
            message_id: Optional message ID
            **extra_metadata: Additional metadata to include

        Returns:
            Enriched Message if trivial, None if needs LLM enrichment
        """
        result = self.detect(text)

        if not result.is_trivial:
            return None

        # Map trivial category to intent and topics
        intent, sentiment, topics = self._map_category(result.category)

        # Calculate importance (very low for trivial messages)
        importance = self._calculate_importance(result)

        # Create metadata
        metadata = MessageMetadata(
            topics=topics,
            categories=["general"],
            entities=[],
            intent=intent.value if intent else "acknowledge",
            sentiment=sentiment.value if sentiment else "neutral",
            importance=importance,
            keywords=[],
            **extra_metadata,
        )

        # Create message
        import uuid
        from datetime import datetime

        message = Message(
            id=message_id or str(uuid.uuid4()),
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id,
            role=MessageRole(role) if isinstance(role, str) else role,
            content=text,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )

        return message

    def _map_category(
        self, category: TrivialCategory
    ) -> Tuple[Optional[Intent], Optional[Sentiment], List[str]]:
        """Map trivial category to intent, sentiment, and topics."""
        mappings = {
            TrivialCategory.GREETING: (Intent.GREET, Sentiment.POSITIVE, ["greeting"]),
            TrivialCategory.FAREWELL: (Intent.GREET, Sentiment.POSITIVE, ["farewell"]),
            TrivialCategory.CONFIRMATION: (Intent.CONFIRM, Sentiment.NEUTRAL, ["general"]),
            TrivialCategory.THANKS: (Intent.THANK, Sentiment.POSITIVE, ["thanks"]),
            TrivialCategory.FILLER: (None, Sentiment.NEUTRAL, ["general"]),
            TrivialCategory.QUESTION_WORD: (Intent.ASK_QUESTION, Sentiment.NEUTRAL, ["general"]),
            TrivialCategory.ACKNOWLEDGMENT: (Intent.CONFIRM, Sentiment.POSITIVE, ["general"]),
            TrivialCategory.APOLOGY: (Intent.APOLOGIZE, Sentiment.NEUTRAL, ["general"]),
        }
        return mappings.get(category, (None, Sentiment.NEUTRAL, ["general"]))

    def _calculate_importance(self, result: TrivialMatch) -> float:
        """
        Calculate importance score for trivial message.

        Trivial messages get very low importance (0.05-0.2)
        to effectively exclude them from context retrieval
        while still including them in thread summaries.
        """
        # Base importance from pattern
        base = 0.1
        for pattern, category, importance in self._compiled_patterns:
            if category == result.category:
                base = importance
                break

        # Adjust by confidence
        return base * result.confidence


# Singleton instance for convenience
_detector: Optional[TrivialMessageDetector] = None


def get_trivial_detector() -> TrivialMessageDetector:
    """Get the singleton trivial message detector."""
    global _detector
    if _detector is None:
        _detector = TrivialMessageDetector()
    return _detector


def reset_trivial_detector() -> None:
    """Reset the singleton detector (useful for testing)."""
    global _detector
    _detector = None
