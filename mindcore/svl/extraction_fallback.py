"""SVL Metadata Extraction Fallback Strategies.

When LLM fails to assign valid SVL-compliant metadata, these strategies
provide progressively degraded but functional fallback mechanisms.

Strategy Order (most to least accurate):
1. Retry with error feedback - Tell LLM what failed, ask to fix
2. Retry with simplified schema - Reduce required fields
3. Batch context extraction - Collect failed, process with neighbor context
4. Rule-based extraction - Keyword/pattern matching to SVL vocabulary
5. Default assignment - Safe defaults with needs_review flag

Design Philosophy:
- Never lose a memory due to metadata extraction failure
- Degrade gracefully with clear quality indicators
- Enable later reprocessing of low-quality extractions
- Track failure patterns to improve SVL vocabulary

Example:
    from mindcore.svl.extraction_fallback import (
        ResilientMetadataExtractor,
        ExtractionStrategy,
    )

    extractor = ResilientMetadataExtractor(svl=svl)

    # This will try all strategies until one succeeds
    result = extractor.extract_with_fallback(
        user_message="I want to cancel my subscription",
        llm_call=my_llm_function,
        max_retries=3,
    )

    if result.strategy_used != ExtractionStrategy.PRIMARY:
        print(f"Used fallback: {result.strategy_used}")
        print(f"Quality score: {result.quality_score}")
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from mindcore.svl import SharedVocabularyLayer


class ExtractionStrategy(str, Enum):
    """Strategy used for metadata extraction."""

    PRIMARY = "primary"  # First attempt with full schema
    RETRY_WITH_ERRORS = "retry_with_errors"  # Retry with error feedback
    RETRY_SIMPLIFIED = "retry_simplified"  # Retry with fewer required fields
    BATCH_CONTEXT = "batch_context"  # Batch processing with neighbor context
    RULE_BASED = "rule_based"  # Keyword/pattern matching
    DEFAULT_ASSIGNMENT = "default_assignment"  # Safe defaults


class ExtractionFailureType(str, Enum):
    """Types of extraction failures."""

    INVALID_JSON = "invalid_json"
    MISSING_REQUIRED = "missing_required"
    INVALID_VOCABULARY = "invalid_vocabulary"
    OUT_OF_BOUNDS = "out_of_bounds"  # e.g., importance > 1.0
    LLM_REFUSAL = "llm_refusal"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ExtractionAttempt:
    """Record of a single extraction attempt."""

    strategy: ExtractionStrategy
    success: bool
    failure_type: ExtractionFailureType | None = None
    failure_details: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExtractionResult:
    """Result of metadata extraction with fallback."""

    success: bool
    metadata: dict[str, Any] | None = None
    strategy_used: ExtractionStrategy = ExtractionStrategy.PRIMARY
    quality_score: float = 1.0  # 1.0 = primary success, degrades with fallbacks
    needs_review: bool = False  # True if used low-quality fallback
    attempts: list[ExtractionAttempt] = field(default_factory=list)

    # For batch reprocessing
    queued_for_batch: bool = False
    batch_context_ids: list[str] = field(default_factory=list)

    # Validation warnings (non-fatal issues)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "metadata": self.metadata,
            "strategy_used": self.strategy_used.value,
            "quality_score": self.quality_score,
            "needs_review": self.needs_review,
            "attempt_count": len(self.attempts),
            "warnings": self.warnings,
        }


@dataclass
class BatchItem:
    """Item in the batch reprocessing queue."""

    message_id: str
    user_message: str
    agent_response: str | None
    session_id: str
    user_id: str
    failure_count: int = 0
    last_failure: ExtractionFailureType | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    neighbor_ids: list[str] = field(default_factory=list)


class RuleBasedExtractor:
    """Fallback extractor using keyword/pattern matching.

    When LLM extraction fails, this provides a simple but reliable
    extraction based on pattern matching against SVL vocabulary.
    """

    def __init__(self, svl: SharedVocabularyLayer | None = None):
        self._svl = svl
        self._topic_patterns: dict[str, list[str]] = {}
        self._category_patterns: dict[str, list[str]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile keyword patterns from SVL vocabulary."""
        if not self._svl:
            return

        # Build keyword associations for topics
        for topic in self._svl.schema.get_all_topics():
            keywords = self._generate_keywords(topic)
            self._topic_patterns[topic] = keywords

        # Build keyword associations for categories
        for category in self._svl.schema.get_all_categories():
            keywords = self._generate_keywords(category)
            self._category_patterns[category] = keywords

    def _generate_keywords(self, term: str) -> list[str]:
        """Generate keywords for a term."""
        # Split compound terms
        keywords = [term.lower()]
        keywords.extend(term.lower().replace("_", " ").replace("-", " ").split())

        # Add common variations
        if "_" in term:
            keywords.append(term.replace("_", " ").lower())
        if "-" in term:
            keywords.append(term.replace("-", " ").lower())

        return list(set(keywords))

    def extract(
        self,
        user_message: str,
        agent_response: str | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Extract metadata using rule-based matching.

        Args:
            user_message: User's message
            agent_response: Optional agent response
            user_id: User ID
            session_id: Session ID

        Returns:
            Metadata dict (may have empty lists if no matches)
        """
        combined_text = user_message.lower()
        if agent_response:
            combined_text += " " + agent_response.lower()

        # Match topics
        topics = []
        for topic, keywords in self._topic_patterns.items():
            for kw in keywords:
                if kw in combined_text:
                    topics.append(topic)
                    break
        topics = topics[:5] if topics else ["general"]

        # Match categories
        categories = []
        for category, keywords in self._category_patterns.items():
            for kw in keywords:
                if kw in combined_text:
                    categories.append(category)
                    break
        categories = categories[:3] if categories else ["uncategorized"]

        # Detect message type
        message_type = self._detect_message_type(user_message)

        # Detect intent
        intent = self._detect_intent(user_message)

        # Estimate importance based on signals
        importance = self._estimate_importance(user_message)

        # Detect sentiment
        sentiment = self._detect_sentiment(user_message)

        # Detect urgency
        urgency = self._detect_urgency(user_message)

        return {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": session_id,
            "topics": topics,
            "categories": categories,
            "entities": self._extract_entities(user_message),
            "message_type": message_type,
            "message_intent": intent,
            "importance": importance,
            "confidence": 0.4,  # Lower confidence for rule-based
            "urgency": urgency,
            "sentiment": sentiment,
            "emotional_classification": "neutral",
            "memory_type": "episodic",
            "access_level": "private",
            "_extraction_method": "rule_based",
        }

    def _detect_message_type(self, text: str) -> str:
        """Detect message type from text patterns."""
        text_lower = text.lower()

        if text_lower.endswith("?") or any(
            w in text_lower for w in ["what", "how", "why", "when", "where", "who", "which"]
        ):
            return "query"
        if any(
            w in text_lower
            for w in ["please", "can you", "could you", "would you", "i need", "i want"]
        ):
            return "command"
        if any(
            w in text_lower for w in ["thanks", "thank you", "great", "good", "bad", "terrible"]
        ):
            return "feedback"

        return "statement"

    def _detect_intent(self, text: str) -> str:
        """Detect message intent from text patterns."""
        text_lower = text.lower()

        if "?" in text or any(w in text_lower for w in ["what", "how", "explain"]):
            return "ask_question"
        if any(w in text_lower for w in ["please", "can you", "do this", "help me"]):
            return "request_action"
        if any(w in text_lower for w in ["i think", "in my opinion", "i feel"]):
            return "give_feedback"
        if any(w in text_lower for w in ["hi", "hello", "hey"]):
            return "greeting"
        if any(w in text_lower for w in ["bye", "goodbye", "see you"]):
            return "farewell"

        return "provide_info"

    def _estimate_importance(self, text: str) -> float:
        """Estimate importance from text signals."""
        text_lower = text.lower()
        importance = 0.5  # Default

        # High importance signals
        if any(w in text_lower for w in ["urgent", "asap", "immediately", "critical", "emergency"]):
            importance = 0.9
        elif any(w in text_lower for w in ["important", "priority", "deadline"]):
            importance = 0.8
        # Low importance signals
        elif any(w in text_lower for w in ["just curious", "wondering", "random", "btw"]):
            importance = 0.3

        return importance

    def _detect_sentiment(self, text: str) -> str:
        """Detect sentiment from text."""
        text_lower = text.lower()

        positive_words = [
            "great",
            "good",
            "excellent",
            "amazing",
            "love",
            "thanks",
            "helpful",
            "wonderful",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "frustrated",
            "angry",
            "disappointed",
            "worst",
        ]

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        if neg_count > pos_count:
            return "negative"
        return "neutral"

    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["emergency", "asap", "immediately", "now"]):
            return "critical"
        if any(w in text_lower for w in ["urgent", "soon", "quickly"]):
            return "high"
        if any(w in text_lower for w in ["when you can", "no rush", "whenever"]):
            return "low"

        return "medium"

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities from text."""
        entities = []

        # Numbers and IDs
        entities.extend(re.findall(r"#\w+", text))
        entities.extend(re.findall(r"\b\d{4,}\b", text))  # Long numbers (IDs, order numbers)

        # Emails
        entities.extend(re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text))

        # Capitalized phrases (potential names)
        entities.extend(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))

        return list(set(entities))[:10]


class ResilientMetadataExtractor:
    """Metadata extractor with comprehensive fallback strategies.

    This class wraps the standard MetadataExtractor and adds fallback
    mechanisms to ensure extraction never completely fails.

    Strategy Order:
    1. Primary extraction - Full LLM call with complete schema
    2. Retry with errors - Same call but include validation errors in prompt
    3. Retry simplified - Reduced required fields, more lenient validation
    4. Batch context - Queue for batch processing with neighbor context
    5. Rule-based - Simple keyword/pattern matching
    6. Default assignment - Safe defaults with review flag

    Example:
        extractor = ResilientMetadataExtractor(svl=svl)

        # Define your LLM call function
        def my_llm_call(prompt: str) -> str:
            return openai.chat(messages=[{"role": "user", "content": prompt}])

        result = extractor.extract_with_fallback(
            user_message="Cancel my subscription",
            llm_call=my_llm_call,
        )
    """

    def __init__(
        self,
        svl: SharedVocabularyLayer | None = None,
        max_retries: int = 2,
        enable_batch_queue: bool = True,
        batch_queue_max_size: int = 100,
        batch_process_threshold: int = 5,
    ):
        """Initialize resilient extractor.

        Args:
            svl: SharedVocabularyLayer for vocabulary access
            max_retries: Maximum retries before falling back
            enable_batch_queue: Enable batch queue for failed extractions
            batch_queue_max_size: Maximum items in batch queue
            batch_process_threshold: Process batch when this many items queued
        """
        self._svl = svl
        self.max_retries = max_retries
        self.enable_batch_queue = enable_batch_queue
        self.batch_queue_max_size = batch_queue_max_size
        self.batch_process_threshold = batch_process_threshold

        # Fallback extractors
        self._rule_based = RuleBasedExtractor(svl)

        # Batch queue
        self._batch_queue: list[BatchItem] = []

        # Statistics
        self._stats = {
            "total_extractions": 0,
            "primary_success": 0,
            "retry_success": 0,
            "batch_success": 0,
            "rule_based_fallback": 0,
            "default_fallback": 0,
            "failure_types": {},
        }

    def extract_with_fallback(
        self,
        user_message: str,
        llm_call: Callable[[str], str],
        agent_response: str | None = None,
        user_id: str = "",
        session_id: str = "",
        neighbor_messages: list[str] | None = None,
        skip_strategies: list[ExtractionStrategy] | None = None,
    ) -> ExtractionResult:
        """Extract metadata with automatic fallback.

        Args:
            user_message: The user's message to extract metadata from
            llm_call: Function that calls LLM with prompt and returns response
            agent_response: Optional agent response
            user_id: User ID
            session_id: Session ID
            neighbor_messages: Previous/next messages for context
            skip_strategies: Strategies to skip (for testing)

        Returns:
            ExtractionResult with metadata or failure info
        """
        self._stats["total_extractions"] += 1
        skip = skip_strategies or []
        result = ExtractionResult(success=False)

        # Strategy 1: Primary extraction
        if ExtractionStrategy.PRIMARY not in skip:
            attempt = self._try_primary_extraction(
                user_message, llm_call, agent_response, user_id, session_id
            )
            result.attempts.append(attempt)

            if attempt.success:
                result.success = True
                result.strategy_used = ExtractionStrategy.PRIMARY
                result.quality_score = 1.0
                self._stats["primary_success"] += 1
                return result

        # Strategy 2: Retry with error feedback
        if ExtractionStrategy.RETRY_WITH_ERRORS not in skip and result.attempts:
            last_error = result.attempts[-1].failure_details
            for retry in range(self.max_retries):
                attempt = self._try_retry_with_errors(
                    user_message,
                    llm_call,
                    agent_response,
                    user_id,
                    session_id,
                    last_error,
                    retry + 1,
                )
                result.attempts.append(attempt)

                if attempt.success:
                    result.success = True
                    result.strategy_used = ExtractionStrategy.RETRY_WITH_ERRORS
                    result.quality_score = 0.9 - (retry * 0.05)
                    self._stats["retry_success"] += 1
                    return result

                last_error = attempt.failure_details

        # Strategy 3: Retry with simplified schema
        if ExtractionStrategy.RETRY_SIMPLIFIED not in skip:
            attempt = self._try_simplified_extraction(
                user_message, llm_call, agent_response, user_id, session_id
            )
            result.attempts.append(attempt)

            if attempt.success:
                result.success = True
                result.strategy_used = ExtractionStrategy.RETRY_SIMPLIFIED
                result.quality_score = 0.75
                result.warnings.append("Used simplified schema - some fields may be missing")
                self._stats["retry_success"] += 1
                return result

        # Strategy 4: Batch context extraction (queue for later)
        if (
            ExtractionStrategy.BATCH_CONTEXT not in skip
            and self.enable_batch_queue
            and neighbor_messages
        ):
            attempt = self._try_batch_context_extraction(
                user_message, llm_call, agent_response, user_id, session_id, neighbor_messages
            )
            result.attempts.append(attempt)

            if attempt.success:
                result.success = True
                result.strategy_used = ExtractionStrategy.BATCH_CONTEXT
                result.quality_score = 0.7
                self._stats["batch_success"] += 1
                return result

        # Strategy 5: Rule-based extraction
        if ExtractionStrategy.RULE_BASED not in skip:
            attempt, metadata = self._try_rule_based_extraction(
                user_message, agent_response, user_id, session_id
            )
            result.attempts.append(attempt)

            if attempt.success:
                result.success = True
                result.metadata = metadata
                result.strategy_used = ExtractionStrategy.RULE_BASED
                result.quality_score = 0.5
                result.needs_review = True
                result.warnings.append("Used rule-based fallback - consider manual review")
                self._stats["rule_based_fallback"] += 1
                return result

        # Strategy 6: Default assignment (last resort)
        attempt, metadata = self._apply_defaults(user_message, user_id, session_id)
        result.attempts.append(attempt)
        result.success = True
        result.metadata = metadata
        result.strategy_used = ExtractionStrategy.DEFAULT_ASSIGNMENT
        result.quality_score = 0.2
        result.needs_review = True
        result.warnings.append("Used default assignment - manual review required")
        self._stats["default_fallback"] += 1

        # Queue for batch reprocessing
        if self.enable_batch_queue:
            self._queue_for_batch(user_message, agent_response, user_id, session_id)
            result.queued_for_batch = True

        return result

    def _try_primary_extraction(
        self,
        user_message: str,
        llm_call: Callable[[str], str],
        agent_response: str | None,
        user_id: str,
        session_id: str,
    ) -> ExtractionAttempt:
        """Try primary LLM extraction with full schema."""
        from .enforced_metadata import MetadataExtractor

        extractor = MetadataExtractor(svl=self._svl, strict_validation=True)
        prompt = extractor.get_extraction_prompt(
            user_message=user_message,
            agent_response=agent_response,
            session_id=session_id,
            user_id=user_id,
        )

        start = datetime.now(timezone.utc)
        try:
            response = llm_call(prompt)
            metadata, _ = extractor.parse_metadata(response, user_id, session_id)

            return ExtractionAttempt(
                strategy=ExtractionStrategy.PRIMARY,
                success=True,
                duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
        except json.JSONDecodeError as e:
            return self._create_failure(
                ExtractionStrategy.PRIMARY,
                ExtractionFailureType.INVALID_JSON,
                f"JSON parse error: {e}",
                start,
            )
        except ValueError as e:
            error_str = str(e)
            if "validation failed" in error_str.lower():
                return self._create_failure(
                    ExtractionStrategy.PRIMARY,
                    ExtractionFailureType.INVALID_VOCABULARY,
                    error_str,
                    start,
                )
            return self._create_failure(
                ExtractionStrategy.PRIMARY, ExtractionFailureType.UNKNOWN, error_str, start
            )
        except Exception as e:
            return self._create_failure(
                ExtractionStrategy.PRIMARY, ExtractionFailureType.UNKNOWN, str(e), start
            )

    def _try_retry_with_errors(
        self,
        user_message: str,
        llm_call: Callable[[str], str],
        agent_response: str | None,
        user_id: str,
        session_id: str,
        last_error: str,
        retry_num: int,
    ) -> ExtractionAttempt:
        """Retry with error feedback in prompt."""
        from .enforced_metadata import MetadataExtractor

        extractor = MetadataExtractor(svl=self._svl, strict_validation=True)
        base_prompt = extractor.get_extraction_prompt(
            user_message=user_message,
            agent_response=agent_response,
            session_id=session_id,
            user_id=user_id,
        )

        # Inject error feedback
        error_guidance = f"""
## PREVIOUS ATTEMPT FAILED - Please Fix

Your previous response had the following error:
{last_error}

Please ensure:
1. Topics and categories MUST be from the provided vocabulary lists
2. All required fields must be present
3. Response must be valid JSON only

Retry attempt {retry_num}/{self.max_retries}. Be more careful this time.

---

{base_prompt}
"""

        start = datetime.now(timezone.utc)
        try:
            response = llm_call(error_guidance)
            metadata, _ = extractor.parse_metadata(response, user_id, session_id)

            return ExtractionAttempt(
                strategy=ExtractionStrategy.RETRY_WITH_ERRORS,
                success=True,
                duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
        except Exception as e:
            return self._create_failure(
                ExtractionStrategy.RETRY_WITH_ERRORS, ExtractionFailureType.UNKNOWN, str(e), start
            )

    def _try_simplified_extraction(
        self,
        user_message: str,
        llm_call: Callable[[str], str],
        agent_response: str | None,
        user_id: str,
        session_id: str,
    ) -> ExtractionAttempt:
        """Try with simplified schema (fewer required fields)."""
        simplified_prompt = f"""Extract basic metadata from this message. Be simple and direct.

Message: {user_message}
{f"Response: {agent_response}" if agent_response else ""}

Return ONLY valid JSON with these fields:
{{
    "topics": ["list", "of", "topics"],
    "categories": ["list"],
    "message_type": "query|statement|command|feedback",
    "importance": 0.5,
    "sentiment": "positive|negative|neutral"
}}

JSON only, no explanation:"""

        start = datetime.now(timezone.utc)
        try:
            response = llm_call(simplified_prompt)
            data = self._extract_json(response)

            # Apply defaults for missing fields
            data.setdefault("message_id", f"msg_{uuid.uuid4().hex[:12]}")
            data.setdefault("user_id", user_id)
            data.setdefault("session_id", session_id)
            data.setdefault("topics", ["general"])
            data.setdefault("categories", ["uncategorized"])
            data.setdefault("message_type", "statement")
            data.setdefault("message_intent", "provide_info")
            data.setdefault("importance", 0.5)
            data.setdefault("confidence", 0.6)
            data.setdefault("sentiment", "neutral")
            data.setdefault("urgency", "medium")
            data.setdefault("entities", [])
            data.setdefault("memory_type", "episodic")
            data.setdefault("access_level", "private")

            # Light validation (non-strict)
            if not isinstance(data.get("topics"), list):
                data["topics"] = ["general"]
            if not isinstance(data.get("categories"), list):
                data["categories"] = ["uncategorized"]

            return ExtractionAttempt(
                strategy=ExtractionStrategy.RETRY_SIMPLIFIED,
                success=True,
                duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
        except Exception as e:
            return self._create_failure(
                ExtractionStrategy.RETRY_SIMPLIFIED, ExtractionFailureType.UNKNOWN, str(e), start
            )

    def _try_batch_context_extraction(
        self,
        user_message: str,
        llm_call: Callable[[str], str],
        agent_response: str | None,
        user_id: str,
        session_id: str,
        neighbor_messages: list[str],
    ) -> ExtractionAttempt:
        """Try extraction with neighbor message context.

        This implements the user's idea of "circulating untagged memories
        with next memory" but in a cleaner way - we provide explicit
        context from neighbors to help the LLM understand the message.
        """
        context_str = "\n".join([f"- {msg}" for msg in neighbor_messages[:3]])

        context_prompt = f"""Extract metadata from the TARGET message below. Use the context messages to understand the topic and intent better.

## Context (previous/related messages):
{context_str}

## TARGET MESSAGE (extract metadata for this):
{user_message}
{f"Agent response: {agent_response}" if agent_response else ""}

Return JSON with: topics, categories, message_type, importance, sentiment

JSON only:"""

        start = datetime.now(timezone.utc)
        try:
            response = llm_call(context_prompt)
            data = self._extract_json(response)

            # Apply defaults
            data.setdefault("message_id", f"msg_{uuid.uuid4().hex[:12]}")
            data.setdefault("user_id", user_id)
            data.setdefault("session_id", session_id)
            data.setdefault("message_intent", "provide_info")
            data.setdefault("confidence", 0.6)
            data.setdefault("urgency", "medium")
            data.setdefault("entities", [])
            data.setdefault("memory_type", "episodic")
            data.setdefault("access_level", "private")

            return ExtractionAttempt(
                strategy=ExtractionStrategy.BATCH_CONTEXT,
                success=True,
                duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
        except Exception as e:
            return self._create_failure(
                ExtractionStrategy.BATCH_CONTEXT, ExtractionFailureType.UNKNOWN, str(e), start
            )

    def _try_rule_based_extraction(
        self,
        user_message: str,
        agent_response: str | None,
        user_id: str,
        session_id: str,
    ) -> tuple[ExtractionAttempt, dict[str, Any]]:
        """Try rule-based extraction (no LLM)."""
        start = datetime.now(timezone.utc)

        try:
            metadata = self._rule_based.extract(
                user_message=user_message,
                agent_response=agent_response,
                user_id=user_id,
                session_id=session_id,
            )

            return (
                ExtractionAttempt(
                    strategy=ExtractionStrategy.RULE_BASED,
                    success=True,
                    duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                ),
                metadata,
            )
        except Exception as e:
            return (
                self._create_failure(
                    ExtractionStrategy.RULE_BASED, ExtractionFailureType.UNKNOWN, str(e), start
                ),
                {},
            )

    def _apply_defaults(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
    ) -> tuple[ExtractionAttempt, dict[str, Any]]:
        """Apply safe default values (last resort)."""
        start = datetime.now(timezone.utc)

        # Extract any entities we can find
        entities = []
        entities.extend(re.findall(r"#\w+", user_message))
        entities.extend(re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", user_message))

        metadata = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": session_id,
            "topics": ["untagged"],
            "categories": ["needs_review"],
            "entities": entities[:5],
            "message_type": "statement",
            "message_intent": "provide_info",
            "importance": 0.5,
            "confidence": 0.1,  # Very low - needs review
            "urgency": "medium",
            "sentiment": "neutral",
            "emotional_classification": "neutral",
            "memory_type": "episodic",
            "access_level": "private",
            "_needs_reprocessing": True,
            "_extraction_method": "default",
        }

        return (
            ExtractionAttempt(
                strategy=ExtractionStrategy.DEFAULT_ASSIGNMENT,
                success=True,
                duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            ),
            metadata,
        )

    def _queue_for_batch(
        self,
        user_message: str,
        agent_response: str | None,
        user_id: str,
        session_id: str,
    ) -> None:
        """Queue failed extraction for batch reprocessing."""
        if len(self._batch_queue) >= self.batch_queue_max_size:
            # Remove oldest item
            self._batch_queue.pop(0)

        item = BatchItem(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            user_message=user_message,
            agent_response=agent_response,
            session_id=session_id,
            user_id=user_id,
        )
        self._batch_queue.append(item)

    def process_batch_queue(
        self,
        llm_call: Callable[[str], str],
    ) -> list[ExtractionResult]:
        """Process the batch queue with context from neighbors.

        This processes multiple failed extractions together, using
        context from adjacent messages to help the LLM.

        Args:
            llm_call: Function to call LLM

        Returns:
            List of extraction results
        """
        if not self._batch_queue:
            return []

        results = []
        queue_copy = self._batch_queue.copy()
        self._batch_queue.clear()

        for i, item in enumerate(queue_copy):
            # Get neighbor context
            neighbors = []
            if i > 0:
                neighbors.append(queue_copy[i - 1].user_message)
            if i < len(queue_copy) - 1:
                neighbors.append(queue_copy[i + 1].user_message)

            # Try extraction with neighbor context
            result = self.extract_with_fallback(
                user_message=item.user_message,
                llm_call=llm_call,
                agent_response=item.agent_response,
                user_id=item.user_id,
                session_id=item.session_id,
                neighbor_messages=neighbors if neighbors else None,
                skip_strategies=[ExtractionStrategy.PRIMARY],  # Already failed primary
            )
            results.append(result)

        return results

    def _create_failure(
        self,
        strategy: ExtractionStrategy,
        failure_type: ExtractionFailureType,
        details: str,
        start: datetime,
    ) -> ExtractionAttempt:
        """Create a failure attempt record."""
        # Track failure type
        self._stats["failure_types"][failure_type.value] = (
            self._stats["failure_types"].get(failure_type.value, 0) + 1
        )

        return ExtractionAttempt(
            strategy=strategy,
            success=False,
            failure_type=failure_type,
            failure_details=details,
            duration_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from: {text[:100]}")

    def get_stats(self) -> dict[str, Any]:
        """Get extraction statistics."""
        total = self._stats["total_extractions"]
        if total == 0:
            return {"total_extractions": 0}

        return {
            "total_extractions": total,
            "primary_success_rate": self._stats["primary_success"] / total,
            "retry_success_rate": self._stats["retry_success"] / total,
            "rule_based_rate": self._stats["rule_based_fallback"] / total,
            "default_rate": self._stats["default_fallback"] / total,
            "failure_types": self._stats["failure_types"],
            "batch_queue_size": len(self._batch_queue),
        }

    def get_items_needing_review(self) -> list[BatchItem]:
        """Get items in batch queue that need manual review."""
        return [item for item in self._batch_queue if item.failure_count >= 2]
