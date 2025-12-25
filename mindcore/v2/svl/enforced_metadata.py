"""SVL Enforced Metadata - Forces LLM to assign SVL-compliant metadata.

This module provides:
1. EnforcedMetadata schema with all required fields
2. LLM prompt templates that enforce SVL vocabulary usage
3. HistoricalContextNeeded decision model
4. Validation and extraction utilities

The LLM is forced to assign metadata from the SVL vocabulary, ensuring
consistent, queryable, and traceable memory metadata.

Example:
    from mindcore.v2.svl.enforced_metadata import (
        MetadataExtractor,
        EnforcedMetadata,
        ContextDecision,
    )

    extractor = MetadataExtractor(svl=shared_vocabulary_layer)

    # Get prompt for LLM to extract metadata
    prompt = extractor.get_extraction_prompt(
        user_message="What's my order status?",
        session_id="session_123",
    )

    # Validate LLM response
    metadata = extractor.parse_response(llm_response)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mindcore.v2.svl import SharedVocabularyLayer


class HistoricalContextNeeded(str, Enum):
    """LLM decision on whether historical context is needed."""

    TRUE = "True"
    FALSE = "False"


@dataclass
class ContextDecision:
    """LLM's decision on context requirements.

    The LLM analyzes the user query and decides:
    - Whether historical context from CLST is needed
    - Suggested topics and categories to query
    - Urgency level for the response
    """

    historical_context_needed: HistoricalContextNeeded
    suggested_topics: list[str] = field(default_factory=list)
    suggested_categories: list[str] = field(default_factory=list)
    reasoning: str = ""
    urgency: str = "medium"
    confidence: str = "high"

    def needs_clst(self) -> bool:
        """Check if CLST query is needed."""
        return self.historical_context_needed == HistoricalContextNeeded.TRUE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "historical_context_needed": self.historical_context_needed.value,
            "suggested_topics": self.suggested_topics,
            "suggested_categories": self.suggested_categories,
            "reasoning": self.reasoning,
            "urgency": self.urgency,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextDecision:
        """Create from dictionary (LLM response)."""
        hcn = data.get("historical_context_needed", "False")
        if isinstance(hcn, str):
            hcn = HistoricalContextNeeded.TRUE if hcn.lower() == "true" else HistoricalContextNeeded.FALSE
        return cls(
            historical_context_needed=hcn,
            suggested_topics=data.get("suggested_topics", []),
            suggested_categories=data.get("suggested_categories", []),
            reasoning=data.get("reasoning", ""),
            urgency=data.get("urgency", "medium"),
            confidence=data.get("confidence", "high"),
        )


@dataclass
class EnforcedMetadata:
    """SVL-enforced metadata that LLM must assign.

    All fields are derived from SVL vocabulary to ensure consistency
    and queryability across the system.
    """

    # Required identifiers (system-assigned or LLM-provided)
    message_id: str
    user_id: str
    session_id: str
    agent_id: str | None = None
    thread_id: str | None = None  # For multi-thread conversations

    # SVL-enforced classifications (LLM must choose from vocabulary)
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    message_type: str = "statement"
    message_intent: str = "provide_info"

    # Scores (LLM-assigned within bounds)
    importance: float = 0.5
    confidence: float = 0.8
    urgency: str = "medium"

    # Additional SVL fields
    sentiment: str = "neutral"
    emotional_classification: str = "neutral"
    temporal_qualifier: str | None = None
    domain_label: str | None = None

    # Memory type (for storage classification)
    memory_type: str = "episodic"
    access_level: str = "private"

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "topics": self.topics,
            "categories": self.categories,
            "entities": self.entities,
            "message_type": self.message_type,
            "message_intent": self.message_intent,
            "importance": self.importance,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "emotional_classification": self.emotional_classification,
            "temporal_qualifier": self.temporal_qualifier,
            "domain_label": self.domain_label,
            "memory_type": self.memory_type,
            "access_level": self.access_level,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnforcedMetadata:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            message_id=data.get("message_id", f"msg_{uuid.uuid4().hex[:12]}"),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id"),
            thread_id=data.get("thread_id"),
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            entities=data.get("entities", []),
            message_type=data.get("message_type", "statement"),
            message_intent=data.get("message_intent", "provide_info"),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.8),
            urgency=data.get("urgency", "medium"),
            sentiment=data.get("sentiment", "neutral"),
            emotional_classification=data.get("emotional_classification", "neutral"),
            temporal_qualifier=data.get("temporal_qualifier"),
            domain_label=data.get("domain_label"),
            memory_type=data.get("memory_type", "episodic"),
            access_level=data.get("access_level", "private"),
            created_at=created_at,
        )


class MetadataExtractor:
    """Extracts and validates SVL-enforced metadata from LLM responses.

    Forces the LLM to assign metadata from the SVL vocabulary through
    structured prompts and validation.
    """

    def __init__(
        self,
        svl: SharedVocabularyLayer | None = None,
        strict_validation: bool = True,
    ):
        """Initialize extractor.

        Args:
            svl: SharedVocabularyLayer for vocabulary access
            strict_validation: Raise errors on invalid vocabulary
        """
        self._svl = svl
        self._strict = strict_validation

    def get_context_decision_prompt(
        self,
        user_message: str,
        session_context: str | None = None,
    ) -> str:
        """Get prompt for LLM to decide if historical context is needed.

        Args:
            user_message: The user's query
            session_context: Optional current session summary

        Returns:
            Prompt string for LLM
        """
        topics = self._get_topics_list()
        categories = self._get_categories_list()

        return f"""Analyze this user message and decide if historical context is needed.

## User Message
{user_message}

{f"## Current Session Context{chr(10)}{session_context}" if session_context else ""}

## Your Task
Decide whether this query requires historical context from previous sessions (CLST)
or if it can be answered using only the current session context (FLR).

Use historical context (HistoricalContextNeeded = "True") when:
- User references past interactions ("last time", "before", "you told me")
- User asks about preferences, history, or patterns
- Query requires understanding of long-term user behavior
- User mentions specific past events or decisions

Do NOT use historical context (HistoricalContextNeeded = "False") when:
- User asks a new, standalone question
- Query is about current session only
- Simple greetings or acknowledgments
- Real-time information requests

## Available Topics (choose from these)
{topics}

## Available Categories (choose from these)
{categories}

## Response Format (JSON)
{{
    "historical_context_needed": "True" or "False",
    "suggested_topics": ["topic1", "topic2"],
    "suggested_categories": ["category1"],
    "reasoning": "Brief explanation of decision",
    "urgency": "critical|high|medium|low|informational",
    "confidence": "high|medium|low"
}}

Respond with valid JSON only."""

    def get_extraction_prompt(
        self,
        user_message: str,
        agent_response: str | None = None,
        session_id: str = "",
        user_id: str = "",
        include_memory_extraction: bool = True,
    ) -> str:
        """Get prompt for LLM to extract enforced metadata.

        Args:
            user_message: The user's message
            agent_response: Optional agent response to classify
            session_id: Current session ID
            user_id: User ID
            include_memory_extraction: Include memory content extraction

        Returns:
            Prompt string for LLM
        """
        topics = self._get_topics_list()
        categories = self._get_categories_list()
        message_types = self._get_message_types_list()
        message_intents = self._get_message_intents_list()
        memory_types = self._get_memory_types_list()
        sentiments = self._get_sentiments_list()
        urgency_levels = self._get_urgency_list()
        emotional = self._get_emotional_list()
        temporal = self._get_temporal_list()

        memory_section = ""
        if include_memory_extraction:
            memory_section = f"""
## Memory Extraction (if applicable)
If this message contains information worth remembering, extract it as a memory:
{{
    "content": "The actual information to remember",
    "memory_type": {memory_types},
    "importance": 0.0-1.0,
    "access_level": "private|team|shared|global"
}}
"""

        return f"""Extract SVL-compliant metadata from this message.

## User Message
{user_message}

{f"## Agent Response{chr(10)}{agent_response}" if agent_response else ""}

## REQUIRED: Assign Metadata from SVL Vocabulary

You MUST assign values from the following vocabularies. Do NOT invent new values.

### Topics (choose 1-5 from this list)
{topics}

### Categories (choose 1-3 from this list)
{categories}

### Message Type (choose exactly 1)
{message_types}

### Message Intent (choose exactly 1)
{message_intents}

### Sentiment (choose exactly 1)
{sentiments}

### Urgency (choose exactly 1)
{urgency_levels}

### Emotional Classification (choose exactly 1)
{emotional}

### Temporal Qualifier (choose 0-1, optional)
{temporal}
{memory_section}
## Response Format (JSON)
{{
    "message_id": "auto-generated or provided",
    "user_id": "{user_id}",
    "session_id": "{session_id}",
    "thread_id": null,

    "topics": ["topic1", "topic2"],
    "categories": ["category1"],
    "entities": ["extracted entity names"],

    "message_type": "query|command|statement|...",
    "message_intent": "ask_question|request_action|...",

    "importance": 0.5,
    "confidence": 0.8,
    "urgency": "medium",

    "sentiment": "neutral",
    "emotional_classification": "neutral",
    "temporal_qualifier": null,

    "memory_type": "episodic",
    "access_level": "private",

    "memories_to_store": [
        {{
            "content": "...",
            "memory_type": "...",
            "importance": 0.5
        }}
    ]
}}

IMPORTANT:
- You MUST choose values from the provided vocabularies
- Topics and categories MUST match exactly (case-sensitive)
- Extract all relevant entities mentioned
- Assign appropriate importance (0.1=trivial, 0.5=normal, 0.9=critical)

Respond with valid JSON only."""

    def get_json_schema(self) -> dict[str, Any]:
        """Get JSON Schema for enforced metadata.

        Use this for structured output with LLMs that support it.

        Returns:
            JSON Schema dict
        """
        topics = self._get_topics_raw()
        categories = self._get_categories_raw()
        message_types = self._get_message_types_raw()
        message_intents = self._get_message_intents_raw()
        sentiments = self._get_sentiments_raw()
        urgency_levels = ["critical", "high", "medium", "low", "informational"]
        emotional = self._get_emotional_raw()
        temporal = self._get_temporal_raw()
        memory_types = self._get_memory_types_raw()
        access_levels = ["private", "team", "shared", "global"]

        return {
            "type": "object",
            "properties": {
                "historical_context_needed": {
                    "type": "string",
                    "enum": ["True", "False"],
                    "description": "Whether historical context from CLST is needed",
                },
                "message_id": {
                    "type": "string",
                    "description": "Unique message identifier",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier",
                },
                "thread_id": {
                    "type": ["string", "null"],
                    "description": "Thread identifier for multi-thread conversations",
                },
                "agent_id": {
                    "type": ["string", "null"],
                    "description": "Agent identifier",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "enum": topics} if topics else {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "Topics from SVL vocabulary",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": categories} if categories else {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "Categories from SVL vocabulary",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extracted entity names",
                },
                "message_type": {
                    "type": "string",
                    "enum": message_types if message_types else [
                        "query", "command", "statement", "feedback",
                        "response", "clarification", "suggestion", "confirmation",
                    ],
                    "description": "Message type from SVL",
                },
                "message_intent": {
                    "type": "string",
                    "enum": message_intents if message_intents else [
                        "ask_question", "request_action", "provide_info",
                        "give_feedback", "greeting", "farewell",
                    ],
                    "description": "Message intent from SVL",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score (0.1=trivial, 0.5=normal, 0.9=critical)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in classification",
                },
                "urgency": {
                    "type": "string",
                    "enum": urgency_levels,
                    "description": "Urgency level",
                },
                "sentiment": {
                    "type": "string",
                    "enum": sentiments if sentiments else ["positive", "negative", "neutral", "mixed"],
                    "description": "Sentiment classification",
                },
                "emotional_classification": {
                    "type": "string",
                    "enum": emotional if emotional else ["neutral", "joy", "sadness", "anger", "fear"],
                    "description": "Emotional classification from SVL",
                },
                "temporal_qualifier": {
                    "type": ["string", "null"],
                    "enum": temporal + [None] if temporal else None,
                    "description": "Temporal qualifier if applicable",
                },
                "domain_label": {
                    "type": ["string", "null"],
                    "description": "Domain label if applicable",
                },
                "memory_type": {
                    "type": "string",
                    "enum": memory_types if memory_types else [
                        "episodic", "semantic", "procedural", "preference", "entity", "working",
                    ],
                    "description": "Memory type for storage",
                },
                "access_level": {
                    "type": "string",
                    "enum": access_levels,
                    "description": "Access level for multi-agent",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for context decision",
                },
                "memories_to_store": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "memory_type": {"type": "string"},
                            "importance": {"type": "number"},
                        },
                        "required": ["content"],
                    },
                    "description": "Memories to extract and store",
                },
            },
            "required": [
                "topics",
                "categories",
                "message_type",
                "message_intent",
                "importance",
                "sentiment",
            ],
        }

    def parse_context_decision(self, llm_response: str | dict) -> ContextDecision:
        """Parse LLM response into ContextDecision.

        Args:
            llm_response: JSON string or dict from LLM

        Returns:
            ContextDecision object
        """
        if isinstance(llm_response, str):
            # Extract JSON from response
            data = self._extract_json(llm_response)
        else:
            data = llm_response

        return ContextDecision.from_dict(data)

    def parse_metadata(
        self,
        llm_response: str | dict,
        user_id: str = "",
        session_id: str = "",
    ) -> tuple[EnforcedMetadata, list[dict[str, Any]]]:
        """Parse LLM response into EnforcedMetadata.

        Args:
            llm_response: JSON string or dict from LLM
            user_id: User ID to inject
            session_id: Session ID to inject

        Returns:
            Tuple of (EnforcedMetadata, list of memories to store)
        """
        if isinstance(llm_response, str):
            data = self._extract_json(llm_response)
        else:
            data = llm_response

        # Inject system values
        if user_id:
            data["user_id"] = user_id
        if session_id:
            data["session_id"] = session_id
        if "message_id" not in data or not data["message_id"]:
            data["message_id"] = f"msg_{uuid.uuid4().hex[:12]}"

        # Validate against SVL vocabulary
        if self._strict:
            self._validate_vocabulary(data)

        metadata = EnforcedMetadata.from_dict(data)
        memories = data.get("memories_to_store", [])

        return metadata, memories

    def validate(self, metadata: EnforcedMetadata | dict) -> tuple[bool, list[str]]:
        """Validate metadata against SVL vocabulary.

        Args:
            metadata: Metadata to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        if isinstance(metadata, EnforcedMetadata):
            data = metadata.to_dict()
        else:
            data = metadata

        errors = []
        errors.extend(self._validate_vocabulary(data, raise_errors=False))
        return len(errors) == 0, errors

    def _validate_vocabulary(
        self,
        data: dict[str, Any],
        raise_errors: bool = True,
    ) -> list[str]:
        """Validate data against SVL vocabulary."""
        errors = []

        if self._svl:
            # Validate topics
            valid_topics = set(self._svl.schema.get_all_topics())
            if valid_topics and "topics" in data:
                invalid = [t for t in data["topics"] if t not in valid_topics]
                if invalid:
                    errors.append(f"Invalid topics: {invalid}")

            # Validate categories
            valid_cats = set(self._svl.schema.get_all_categories())
            if valid_cats and "categories" in data:
                invalid = [c for c in data["categories"] if c not in valid_cats]
                if invalid:
                    errors.append(f"Invalid categories: {invalid}")

            # Validate message_type
            valid_types = set(self._svl.schema.get_message_types())
            if valid_types and data.get("message_type") not in valid_types:
                errors.append(f"Invalid message_type: {data.get('message_type')}")

            # Validate message_intent
            valid_intents = set(self._svl.schema.get_message_intents())
            if valid_intents and data.get("message_intent") not in valid_intents:
                errors.append(f"Invalid message_intent: {data.get('message_intent')}")

        if errors and raise_errors:
            raise ValueError(f"SVL validation failed: {errors}")

        return errors

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.debug("Direct JSON parse failed: %s", e)

        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.debug("JSON block parse failed: %s", e)

        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    # Helper methods to get vocabulary lists
    def _get_topics_list(self) -> str:
        topics = self._get_topics_raw()
        return ", ".join(topics[:30]) + ("..." if len(topics) > 30 else "")

    def _get_topics_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_all_topics()
        return []

    def _get_categories_list(self) -> str:
        cats = self._get_categories_raw()
        return ", ".join(cats[:20]) + ("..." if len(cats) > 20 else "")

    def _get_categories_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_all_categories()
        return []

    def _get_message_types_list(self) -> str:
        types = self._get_message_types_raw()
        return ", ".join(types)

    def _get_message_types_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_message_types()
        return ["query", "command", "statement", "feedback", "response", "clarification"]

    def _get_message_intents_list(self) -> str:
        intents = self._get_message_intents_raw()
        return ", ".join(intents[:15]) + ("..." if len(intents) > 15 else "")

    def _get_message_intents_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_message_intents()
        return ["ask_question", "request_action", "provide_info", "give_feedback"]

    def _get_memory_types_list(self) -> str:
        return ", ".join(self._get_memory_types_raw())

    def _get_memory_types_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.memory_types
        return ["episodic", "semantic", "procedural", "preference", "entity", "working"]

    def _get_sentiments_list(self) -> str:
        return ", ".join(self._get_sentiments_raw())

    def _get_sentiments_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.sentiments
        return ["positive", "negative", "neutral", "mixed"]

    def _get_urgency_list(self) -> str:
        return "critical, high, medium, low, informational"

    def _get_emotional_list(self) -> str:
        emotional = self._get_emotional_raw()
        return ", ".join(emotional[:10]) + ("..." if len(emotional) > 10 else "")

    def _get_emotional_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_emotional_classifications()
        return ["neutral", "joy", "sadness", "anger", "fear", "surprise"]

    def _get_temporal_list(self) -> str:
        temporal = self._get_temporal_raw()
        return ", ".join(temporal[:8]) + ("..." if len(temporal) > 8 else "")

    def _get_temporal_raw(self) -> list[str]:
        if self._svl:
            return self._svl.schema.get_temporal_qualifiers()
        return ["past_event", "current", "future_plan", "recurring"]

    # =========================================================================
    # Provider Integration Methods
    # =========================================================================

    def get_provider_request(
        self,
        provider: str,
        user_message: str,
        request_type: str = "metadata",
        session_id: str = "",
        user_id: str = "",
        **provider_kwargs: Any,
    ) -> dict[str, Any]:
        """Get a complete request ready for an LLM provider.

        This method combines the prompt, schema, and provider-specific
        configuration into a ready-to-send request.

        Args:
            provider: Provider name (openai, anthropic, google, etc.)
            user_message: The user's message to process
            request_type: Type of extraction ("metadata" or "context_decision")
            session_id: Session ID for metadata
            user_id: User ID for metadata
            **provider_kwargs: Provider-specific configuration options

        Returns:
            Dict with all request parameters for the provider

        Example:
            # OpenAI GPT-5 with reasoning
            request = extractor.get_provider_request(
                provider="openai",
                user_message="What's my order status?",
                request_type="metadata",
                session_id="session_123",
                user_id="user_456",
                model="gpt-5",
                reasoning_effort="high",
            )

            # Claude with extended thinking
            request = extractor.get_provider_request(
                provider="anthropic",
                user_message="What's my order status?",
                request_type="context_decision",
                thinking_budget=16000,
            )
        """
        from .llm_providers import get_provider_config

        # Get provider configuration
        config = get_provider_config(provider, **provider_kwargs)

        # Get appropriate schema
        schema = self.get_json_schema()

        # Get provider-specific parameters
        params = config.get_request_params(schema, include_reasoning=True)
        headers = config.get_headers()

        # Build the prompt
        if request_type == "context_decision":
            prompt = self.get_context_decision_prompt(user_message)
        else:
            prompt = self.get_extraction_prompt(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
            )

        # Build messages based on provider
        if config.get_provider_name() == "openai":
            params["input"] = prompt  # Responses API uses 'input'
        elif config.get_provider_name() == "anthropic":
            params["messages"] = [{"role": "user", "content": prompt}]
        elif config.get_provider_name() == "google":
            params["contents"] = [{"parts": [{"text": prompt}]}]
        else:
            params["messages"] = [{"role": "user", "content": prompt}]

        return {
            "params": params,
            "headers": headers,
            "provider": config.get_provider_name(),
        }

    def get_openai_request(
        self,
        user_message: str,
        request_type: str = "metadata",
        session_id: str = "",
        user_id: str = "",
        model: str = "gpt-5",
        reasoning_effort: str = "high",
        use_responses_api: bool = True,
    ) -> dict[str, Any]:
        """Get OpenAI GPT-5 request with Responses API.

        Uses the new Responses API for:
        - Preserved reasoning across turns
        - Better intelligence (3-5% improvement)
        - Lower costs (40-80% better cache utilization)

        Args:
            user_message: The message to process
            request_type: "metadata" or "context_decision"
            session_id: Session ID
            user_id: User ID
            model: Model name (gpt-5, gpt-5-pro, gpt-5-mini)
            reasoning_effort: low, medium, high, or xhigh
            use_responses_api: Use Responses API (recommended)

        Returns:
            Request parameters for OpenAI API

        Reference: https://platform.openai.com/docs/guides/responses-vs-chat-completions
        """
        from .llm_providers import OpenAIConfig, ReasoningEffort

        effort = ReasoningEffort(reasoning_effort)
        config = OpenAIConfig(
            model=model,
            reasoning_effort=effort,
            use_responses_api=use_responses_api,
            temperature=0.0,
        )

        schema = self.get_json_schema()
        params = config.get_request_params(schema, include_reasoning=True)

        if request_type == "context_decision":
            prompt = self.get_context_decision_prompt(user_message)
        else:
            prompt = self.get_extraction_prompt(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
            )

        if use_responses_api:
            params["input"] = prompt
        else:
            params["messages"] = [{"role": "user", "content": prompt}]

        return params

    def get_claude_request(
        self,
        user_message: str,
        request_type: str = "metadata",
        session_id: str = "",
        user_id: str = "",
        model: str = "claude-sonnet-4-5-20250514",
        thinking_budget: int = 16000,
        use_extended_thinking: bool = True,
        use_interleaved_thinking: bool = True,
    ) -> dict[str, Any]:
        """Get Claude request with Extended Thinking.

        Uses Extended Thinking for:
        - Deep reasoning before responding
        - Better metadata classification
        - Interleaved thinking for complex queries

        Note: Temperature not supported with extended thinking.

        Args:
            user_message: The message to process
            request_type: "metadata" or "context_decision"
            session_id: Session ID
            user_id: User ID
            model: Model name
            thinking_budget: Max tokens for internal reasoning
            use_extended_thinking: Enable extended thinking
            use_interleaved_thinking: Enable thinking between tool calls

        Returns:
            Dict with params and headers for Anthropic API

        Reference: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
        """
        from .llm_providers import ClaudeConfig

        config = ClaudeConfig(
            model=model,
            thinking_budget=thinking_budget,
            use_extended_thinking=use_extended_thinking,
            use_interleaved_thinking=use_interleaved_thinking,
        )

        schema = self.get_json_schema()
        params = config.get_request_params(schema, include_reasoning=True)
        headers = config.get_headers()

        if request_type == "context_decision":
            prompt = self.get_context_decision_prompt(user_message)
        else:
            prompt = self.get_extraction_prompt(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
            )

        params["messages"] = [{"role": "user", "content": prompt}]

        return {
            "params": params,
            "headers": headers,
        }

    def get_gemini_request(
        self,
        user_message: str,
        request_type: str = "metadata",
        session_id: str = "",
        user_id: str = "",
        model: str = "gemini-2.5-flash",
        thinking_mode: str = "dynamic",
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """Get Gemini request with Thinking Mode.

        Uses Thinking Mode for:
        - Automatic complexity-based thinking budget
        - Better reasoning for metadata extraction
        - Works with all Gemini tools

        Args:
            user_message: The message to process
            request_type: "metadata" or "context_decision"
            session_id: Session ID
            user_id: User ID
            model: Model name
            thinking_mode: "disabled", "dynamic", or "fixed"
            thinking_budget: Budget for "fixed" mode

        Returns:
            Request parameters for Gemini API

        Reference: https://ai.google.dev/gemini-api/docs/thinking
        """
        from .llm_providers import GeminiConfig, ThinkingMode

        mode = ThinkingMode(thinking_mode) if thinking_mode in ["disabled", "dynamic", "fixed"] else ThinkingMode.DYNAMIC

        config = GeminiConfig(
            model=model,
            thinking_mode=mode,
            thinking_budget=thinking_budget,
            temperature=0.0,
        )

        schema = self.get_json_schema()
        params = config.get_request_params(schema, include_reasoning=True)

        if request_type == "context_decision":
            prompt = self.get_context_decision_prompt(user_message)
        else:
            prompt = self.get_extraction_prompt(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
            )

        params["contents"] = [{"parts": [{"text": prompt}]}]

        return params

    # =========================================================================
    # Feedback-Enhanced Metadata Extraction
    # =========================================================================

    def get_extraction_prompt_with_feedback(
        self,
        user_message: str,
        feedback: dict[str, Any],
        agent_response: str | None = None,
        session_id: str = "",
        user_id: str = "",
    ) -> str:
        """Get extraction prompt enhanced with effectiveness feedback.

        This method injects feedback from previous retrievals to improve
        metadata assignment quality. High-effectiveness topics/categories
        are recommended, while low-effectiveness ones are flagged.

        Args:
            user_message: The user's message
            feedback: Feedback from FLR.get_metadata_feedback_for_extractor()
            agent_response: Optional agent response
            session_id: Session ID
            user_id: User ID

        Returns:
            Enhanced prompt with feedback guidance

        Example:
            feedback = flr.get_metadata_feedback_for_extractor()
            prompt = extractor.get_extraction_prompt_with_feedback(
                user_message="How do I get a refund?",
                feedback=feedback,
            )
            # Prompt now includes:
            # "Prefer high-quality topics: 'refund', 'billing'"
            # "Avoid low-quality topics: 'general', 'misc'"
        """
        # Get base prompt
        base_prompt = self.get_extraction_prompt(
            user_message=user_message,
            agent_response=agent_response,
            session_id=session_id,
            user_id=user_id,
        )

        # Build feedback guidance section
        guidance_lines = []

        # High-quality topics
        high_topics = feedback.get("high_quality_topics", [])
        if high_topics:
            topics_str = ", ".join([f"'{t[0]}'" for t in high_topics[:10]])
            guidance_lines.append(f"✓ PREFER these topics (proven effective): {topics_str}")

        # Low-quality topics
        low_topics = feedback.get("low_quality_topics", [])
        if low_topics:
            topics_str = ", ".join([f"'{t[0]}'" for t in low_topics[:5]])
            guidance_lines.append(f"✗ AVOID these topics (low effectiveness): {topics_str}")

        # High-quality categories
        high_cats = feedback.get("high_quality_categories", [])
        if high_cats:
            cats_str = ", ".join([f"'{c[0]}'" for c in high_cats[:10]])
            guidance_lines.append(f"✓ PREFER these categories (proven effective): {cats_str}")

        # Low-quality categories
        low_cats = feedback.get("low_quality_categories", [])
        if low_cats:
            cats_str = ", ".join([f"'{c[0]}'" for c in low_cats[:5]])
            guidance_lines.append(f"✗ AVOID these categories (low effectiveness): {cats_str}")

        # Natural language guidance
        if feedback.get("guidance"):
            guidance_lines.append(f"\nAdditional guidance:\n{feedback['guidance']}")

        if not guidance_lines:
            return base_prompt

        feedback_section = """
## Quality Feedback (from retrieval analytics)
The following guidance is based on which metadata assignments led to
successful retrievals. Use this to improve assignment quality.

""" + "\n".join(guidance_lines)

        # Insert before the response format section
        insert_marker = "## Response Format"
        if insert_marker in base_prompt:
            parts = base_prompt.split(insert_marker)
            return parts[0] + feedback_section + "\n\n" + insert_marker + parts[1]

        # Fallback: append at end
        return base_prompt + "\n" + feedback_section

    def set_feedback_source(self, feedback_getter: callable) -> None:
        """Set a callback to automatically get feedback for prompts.

        This allows the MetadataExtractor to automatically enhance
        prompts with effectiveness feedback without explicit calls.

        Args:
            feedback_getter: Callable that returns feedback dict

        Example:
            extractor.set_feedback_source(flr.get_metadata_feedback_for_extractor)

            # Now all prompts automatically include feedback
            prompt = extractor.get_extraction_prompt(...)
        """
        self._feedback_getter = feedback_getter

    def get_effectiveness_enhanced_schema(
        self,
        feedback: dict[str, Any],
    ) -> dict[str, Any]:
        """Get JSON schema with effectiveness annotations.

        Modifies the schema to include effectiveness scores as descriptions,
        helping LLMs make better choices.

        Args:
            feedback: Feedback from FLR.get_metadata_feedback_for_extractor()

        Returns:
            Enhanced JSON schema
        """
        schema = self.get_json_schema()

        # Enhance topics with effectiveness info
        high_topics = {t[0]: t[1] for t in feedback.get("high_quality_topics", [])}
        low_topics = {t[0]: t[1] for t in feedback.get("low_quality_topics", [])}

        if "properties" in schema and "topics" in schema["properties"]:
            topic_desc = schema["properties"]["topics"].get("description", "")
            if high_topics:
                effective_list = ", ".join([f"{t} ({s:.0%})" for t, s in list(high_topics.items())[:5]])
                topic_desc += f" Effective topics: {effective_list}."
            if low_topics:
                avoid_list = ", ".join([f"{t}" for t in list(low_topics.keys())[:3]])
                topic_desc += f" Avoid: {avoid_list}."
            schema["properties"]["topics"]["description"] = topic_desc

        return schema

