"""Reasoning extraction for Mindcore Proxy.

Extracts reasoning, decisions, and learnings from Claude Code sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ReasoningType(str, Enum):
    """Types of reasoning that can be extracted."""

    DECISION = "decision"
    LEARNING = "learning"
    PREFERENCE = "preference"
    INSIGHT = "insight"
    ERROR_RECOVERY = "error_recovery"
    PATTERN = "pattern"


@dataclass
class ExtractedReasoning:
    """Represents reasoning extracted from a session.

    Attributes:
        reasoning_type: Type of reasoning
        content: The reasoning content
        confidence: Confidence score (0-1)
        context: Surrounding context that led to this reasoning
        topics: Related topics
        metadata: Additional metadata
        extracted_at: When this was extracted
    """

    reasoning_type: ReasoningType
    content: str
    confidence: float = 0.8
    context: str = ""
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reasoning_type": self.reasoning_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "context": self.context,
            "topics": self.topics,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at.isoformat(),
        }

    def to_memory_dict(self, user_id: str) -> dict[str, Any]:
        """Convert to a memory-compatible dictionary.

        Args:
            user_id: User ID for the memory

        Returns:
            Dictionary suitable for storing as a memory
        """
        # Map reasoning type to memory type
        type_mapping = {
            ReasoningType.DECISION: "episodic",
            ReasoningType.LEARNING: "semantic",
            ReasoningType.PREFERENCE: "preference",
            ReasoningType.INSIGHT: "semantic",
            ReasoningType.ERROR_RECOVERY: "procedural",
            ReasoningType.PATTERN: "semantic",
        }

        return {
            "content": self.content,
            "memory_type": type_mapping.get(self.reasoning_type, "semantic"),
            "user_id": user_id,
            "topics": self.topics,
            "importance": self.confidence,
            "metadata": {
                "reasoning_type": self.reasoning_type.value,
                "context": self.context,
                **self.metadata,
            },
        }


class ReasoningExtractor:
    """Extracts reasoning from Claude Code session messages.

    Uses pattern matching and heuristics to identify reasoning,
    decisions, and learnings from conversation history.
    """

    # Patterns that indicate different types of reasoning
    DECISION_PATTERNS = [
        "I'll use",
        "I decided to",
        "I'm going to",
        "The best approach is",
        "I chose",
        "Let me",
    ]

    LEARNING_PATTERNS = [
        "I learned",
        "I discovered",
        "I found out",
        "It turns out",
        "I realized",
        "Now I understand",
    ]

    PREFERENCE_PATTERNS = [
        "User prefers",
        "They prefer",
        "They like",
        "They want",
        "Their preference",
    ]

    INSIGHT_PATTERNS = [
        "I noticed",
        "Interestingly",
        "It seems",
        "This suggests",
        "Based on this",
    ]

    ERROR_PATTERNS = [
        "The error",
        "Fixed by",
        "The issue was",
        "The problem was",
        "To fix this",
    ]

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize the extractor.

        Args:
            min_confidence: Minimum confidence threshold for extraction
        """
        self.min_confidence = min_confidence

    def extract_from_message(
        self,
        content: str,
        role: str = "assistant",
        context: str = "",
    ) -> list[ExtractedReasoning]:
        """Extract reasoning from a single message.

        Args:
            content: Message content
            role: Message role (usually assistant)
            context: Surrounding context

        Returns:
            List of extracted reasoning items
        """
        if role != "assistant":
            return []

        results = []

        # Check for decisions
        for pattern in self.DECISION_PATTERNS:
            if pattern.lower() in content.lower():
                reasoning = self._extract_with_pattern(
                    content=content,
                    pattern=pattern,
                    reasoning_type=ReasoningType.DECISION,
                    context=context,
                )
                if reasoning and reasoning.confidence >= self.min_confidence:
                    results.append(reasoning)
                break

        # Check for learnings
        for pattern in self.LEARNING_PATTERNS:
            if pattern.lower() in content.lower():
                reasoning = self._extract_with_pattern(
                    content=content,
                    pattern=pattern,
                    reasoning_type=ReasoningType.LEARNING,
                    context=context,
                )
                if reasoning and reasoning.confidence >= self.min_confidence:
                    results.append(reasoning)
                break

        # Check for preferences
        for pattern in self.PREFERENCE_PATTERNS:
            if pattern.lower() in content.lower():
                reasoning = self._extract_with_pattern(
                    content=content,
                    pattern=pattern,
                    reasoning_type=ReasoningType.PREFERENCE,
                    context=context,
                )
                if reasoning and reasoning.confidence >= self.min_confidence:
                    results.append(reasoning)
                break

        # Check for insights
        for pattern in self.INSIGHT_PATTERNS:
            if pattern.lower() in content.lower():
                reasoning = self._extract_with_pattern(
                    content=content,
                    pattern=pattern,
                    reasoning_type=ReasoningType.INSIGHT,
                    context=context,
                )
                if reasoning and reasoning.confidence >= self.min_confidence:
                    results.append(reasoning)
                break

        # Check for error recovery
        for pattern in self.ERROR_PATTERNS:
            if pattern.lower() in content.lower():
                reasoning = self._extract_with_pattern(
                    content=content,
                    pattern=pattern,
                    reasoning_type=ReasoningType.ERROR_RECOVERY,
                    context=context,
                )
                if reasoning and reasoning.confidence >= self.min_confidence:
                    results.append(reasoning)
                break

        return results

    def extract_from_session(
        self,
        messages: list[dict[str, Any]],
    ) -> list[ExtractedReasoning]:
        """Extract all reasoning from a session's messages.

        Args:
            messages: List of session messages

        Returns:
            List of all extracted reasoning items
        """
        results = []

        for i, message in enumerate(messages):
            # Build context from previous messages
            context_messages = messages[max(0, i - 2):i]
            context = " ".join(
                m.get("content", "")[:200] for m in context_messages
            )

            extracted = self.extract_from_message(
                content=message.get("content", ""),
                role=message.get("role", "user"),
                context=context,
            )
            results.extend(extracted)

        return results

    def _extract_with_pattern(
        self,
        content: str,
        pattern: str,
        reasoning_type: ReasoningType,
        context: str,
    ) -> ExtractedReasoning | None:
        """Extract reasoning using a specific pattern.

        Args:
            content: Full message content
            pattern: Pattern that triggered extraction
            reasoning_type: Type of reasoning
            context: Surrounding context

        Returns:
            Extracted reasoning or None
        """
        # Find the sentence containing the pattern
        sentences = content.replace("\n", " ").split(".")
        matching_sentence = ""

        for sentence in sentences:
            if pattern.lower() in sentence.lower():
                matching_sentence = sentence.strip()
                break

        if not matching_sentence:
            return None

        # Calculate confidence based on sentence length and specificity
        confidence = 0.7
        if len(matching_sentence) > 50:
            confidence += 0.1
        if len(matching_sentence) > 100:
            confidence += 0.1

        # Extract potential topics from the content
        topics = self._extract_topics(matching_sentence)

        return ExtractedReasoning(
            reasoning_type=reasoning_type,
            content=matching_sentence,
            confidence=min(confidence, 1.0),
            context=context[:500] if context else "",
            topics=topics,
        )

    def _extract_topics(self, content: str) -> list[str]:
        """Extract potential topics from content.

        Args:
            content: Content to extract topics from

        Returns:
            List of topic strings
        """
        # Simple keyword extraction
        tech_keywords = [
            "python", "javascript", "typescript", "react", "api",
            "database", "sql", "file", "function", "class", "test",
            "error", "bug", "feature", "config", "settings",
        ]

        topics = []
        content_lower = content.lower()

        for keyword in tech_keywords:
            if keyword in content_lower:
                topics.append(keyword)

        return topics[:5]  # Limit to 5 topics
