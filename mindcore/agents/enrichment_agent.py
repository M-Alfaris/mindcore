"""Metadata Enrichment AI Agent.

Enriches messages with intelligent metadata using LLM providers.
Uses VocabularyManager for controlled vocabulary.
Includes confidence scoring for quality monitoring.
"""

import time
from typing import TYPE_CHECKING, Any

from mindcore.core.schemas import EnrichmentSource, Message, MessageMetadata
from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.helper import generate_message_id
from mindcore.utils.logger import get_logger

from .base_agent import BaseAgent


if TYPE_CHECKING:
    from mindcore.llm import BaseLLMProvider

logger = get_logger(__name__)


class EnrichmentAgent(BaseAgent):
    """AI agent that enriches messages with metadata.

    Analyzes message content and generates:
    - Topics (from controlled vocabulary)
    - Categories (from controlled vocabulary)
    - Importance score
    - Sentiment (enum: positive, negative, neutral, mixed)
    - Intent (enum: ask_question, request_action, etc.)
    - Tags (free-form keywords)
    - Entities (extracted named entities)
    - Key phrases

    Uses VocabularyManager to ensure consistent, controlled vocabulary
    that integrates with connectors and external systems.

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.AUTO, ...)
        >>> agent = EnrichmentAgent(provider)
        >>> message = agent.process({
        ...     "user_id": "user1",
        ...     "thread_id": "thread1",
        ...     "session_id": "session1",
        ...     "role": "user",
        ...     "text": "How do I check my order status?"
        ... })
        >>> print(message.metadata.topics)
        ['orders', 'tracking']
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.3,
        max_tokens: int = 800,
        vocabulary: VocabularyManager | None = None,
    ):
        """Initialize enrichment agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            vocabulary: VocabularyManager instance. If None, uses global instance.
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.vocabulary = vocabulary or get_vocabulary()
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for enrichment with vocabulary constraints."""
        vocab_prompt = self.vocabulary.to_prompt_list()

        return f"""You are a metadata enrichment AI agent. Your task is to analyze messages and extract structured metadata.

CRITICAL: You MUST ONLY use values from the predefined lists below. Do NOT invent new topics, categories, intents, or sentiments.

{vocab_prompt}

For each message, return a JSON object with these fields:

{{
  "topics": ["select 1-3 topics from Available Topics"],
  "categories": ["select 1-2 categories from Available Categories"],
  "importance": 0.0-1.0 (float, where 1.0 is most important),
  "sentiment": {{
    "overall": "select from Available Sentiments",
    "score": 0.0-1.0 (float, 0=very negative, 1=very positive)
  }},
  "intent": "select ONE intent from Available Intents",
  "tags": ["relevant keywords extracted from the message"],
  "entities": ["named entities: people, places, products, order IDs, etc."],
  "key_phrases": ["important phrases from the message"]
}}

Rules:
- ONLY use values from the predefined lists for topics, categories, intent, and sentiment
- If no topic matches well, use "general" or the closest match
- importance: 0.0-0.3 for greetings/casual, 0.4-0.6 for normal, 0.7-1.0 for urgent/important
- For order-related queries, use topics like "orders", "tracking", "delivery"
- For billing queries, use topics like "billing", "payment", "invoice", "refund"
- Be concise and accurate. Return ONLY valid JSON."""

    def process(self, message_dict: dict[str, Any]) -> Message:
        """Enrich a message with metadata.

        Args:
            message_dict: Dictionary containing message fields:
                - user_id (str): User identifier
                - thread_id (str): Thread identifier
                - session_id (str): Session identifier
                - role (str): Message role (user, assistant, system, tool)
                - text (str): Message content
                - message_id (str, optional): Message ID (auto-generated if not provided)

        Returns:
            Enriched Message object. Check message.metadata.enrichment_failed
            to determine if enrichment was successful.
            Check message.metadata.confidence_score for quality assessment.
        """
        text = message_dict.get("text", "")
        role = message_dict.get("role", "user")
        start_time = time.time()

        logger.debug(f"Enriching message ({self.provider_name}): {text[:100]}...")

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Message to analyze:\nRole: {role}\nText: {text}"},
        ]

        metadata = None
        try:
            # Call LLM
            response = self._call_llm(messages, json_mode=True)

            # Parse response
            metadata_dict = self._parse_json_response(response)

            # Validate against vocabulary (strict - no fallback to invalid values)
            raw_topics = metadata_dict.get("topics", [])
            raw_categories = metadata_dict.get("categories", [])
            raw_intent = metadata_dict.get("intent")
            raw_sentiment = metadata_dict.get("sentiment", {})

            # Validate using VocabularyManager
            validated_topics = self.vocabulary.validate_topics(raw_topics)
            validated_categories = self.vocabulary.validate_categories(raw_categories)
            validated_intent = self.vocabulary.resolve_intent(raw_intent) if raw_intent else None

            # Calculate vocabulary match rate for confidence scoring
            vocab_match_stats = self._calculate_vocabulary_match_rate(
                raw_topics=raw_topics,
                validated_topics=validated_topics,
                raw_categories=raw_categories,
                validated_categories=validated_categories,
                raw_intent=raw_intent,
                validated_intent=validated_intent,
            )

            # Validate sentiment
            sentiment_overall = raw_sentiment.get("overall", "neutral")
            sentiment_valid = self.vocabulary.is_valid_sentiment(sentiment_overall)
            if not sentiment_valid:
                sentiment_overall = "neutral"

            # If no valid topics found, default to "general"
            used_default_topics = False
            if not validated_topics:
                validated_topics = ["general"]
                used_default_topics = True

            # If no valid categories found, default to "general"
            used_default_categories = False
            if not validated_categories:
                validated_categories = ["general"]
                used_default_categories = True

            # Calculate confidence score
            latency_ms = (time.time() - start_time) * 1000
            confidence_score = self._calculate_confidence_score(
                vocab_match_rate=vocab_match_stats["match_rate"],
                used_default_topics=used_default_topics,
                used_default_categories=used_default_categories,
                sentiment_valid=sentiment_valid,
                has_entities=bool(metadata_dict.get("entities")),
                has_key_phrases=bool(metadata_dict.get("key_phrases")),
                text_length=len(text),
            )

            # Create MessageMetadata object with validated values
            metadata = MessageMetadata(
                topics=validated_topics,
                categories=validated_categories,
                importance=max(0.0, min(1.0, metadata_dict.get("importance", 0.5))),
                sentiment={
                    "overall": sentiment_overall,
                    "score": max(0.0, min(1.0, raw_sentiment.get("score", 0.5))),
                },
                intent=validated_intent,
                tags=metadata_dict.get("tags", []),
                entities=metadata_dict.get("entities", []),
                key_phrases=metadata_dict.get("key_phrases", []),
                enrichment_failed=False,
                enrichment_error=None,
                # New confidence fields
                confidence_score=confidence_score,
                enrichment_source=EnrichmentSource.LLM.value,
                enrichment_latency_ms=latency_ms,
                vocabulary_match_rate=vocab_match_stats["match_rate"],
            )

            logger.info(
                f"Enriched message: topics={metadata.topics}, "
                f"confidence={confidence_score:.2f}, latency={latency_ms:.0f}ms"
            )

        except Exception as e:
            error_msg = str(e)
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Enrichment failed: {error_msg}")

            # Create metadata that indicates failure with zero confidence
            metadata = MessageMetadata(
                enrichment_failed=True,
                enrichment_error=error_msg,
                confidence_score=0.0,
                enrichment_source=EnrichmentSource.FALLBACK.value,
                enrichment_latency_ms=latency_ms,
                vocabulary_match_rate=0.0,
            )

        # Create Message object
        return Message(
            message_id=message_dict.get("message_id") or generate_message_id(),
            user_id=message_dict["user_id"],
            thread_id=message_dict["thread_id"],
            session_id=message_dict["session_id"],
            role=role,
            raw_text=text,
            metadata=metadata,
        )

    def _calculate_vocabulary_match_rate(
        self,
        raw_topics: list[str],
        validated_topics: list[str],
        raw_categories: list[str],
        validated_categories: list[str],
        raw_intent: str | None,
        validated_intent: str | None,
    ) -> dict[str, Any]:
        """Calculate how well LLM outputs matched the controlled vocabulary.

        Returns:
            Dictionary with match statistics.
        """
        total_items = 0
        matched_items = 0

        # Topics matching
        if raw_topics:
            total_items += len(raw_topics)
            matched_items += len(validated_topics)

        # Categories matching
        if raw_categories:
            total_items += len(raw_categories)
            matched_items += len(validated_categories)

        # Intent matching
        if raw_intent:
            total_items += 1
            if validated_intent:
                matched_items += 1

        match_rate = matched_items / total_items if total_items > 0 else 1.0

        return {
            "match_rate": match_rate,
            "total_items": total_items,
            "matched_items": matched_items,
            "topics_matched": len(validated_topics),
            "topics_total": len(raw_topics) if raw_topics else 0,
            "categories_matched": len(validated_categories),
            "categories_total": len(raw_categories) if raw_categories else 0,
            "intent_matched": validated_intent is not None if raw_intent else True,
        }

    def _calculate_confidence_score(
        self,
        vocab_match_rate: float,
        used_default_topics: bool,
        used_default_categories: bool,
        sentiment_valid: bool,
        has_entities: bool,
        has_key_phrases: bool,
        text_length: int,
    ) -> float:
        """Calculate overall confidence score for the enrichment.

        Factors considered:
        - Vocabulary match rate (40% weight)
        - Whether defaults were used (20% weight)
        - Sentiment validity (10% weight)
        - Richness of extraction (20% weight)
        - Text length appropriateness (10% weight)

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.0

        # Vocabulary match rate (40% weight) - most important
        score += vocab_match_rate * 0.40

        # Default usage penalty (20% weight)
        default_score = 1.0
        if used_default_topics:
            default_score -= 0.5
        if used_default_categories:
            default_score -= 0.5
        score += max(0.0, default_score) * 0.20

        # Sentiment validity (10% weight)
        score += (1.0 if sentiment_valid else 0.5) * 0.10

        # Richness of extraction (20% weight)
        richness_score = 0.0
        if has_entities:
            richness_score += 0.5
        if has_key_phrases:
            richness_score += 0.5
        score += richness_score * 0.20

        # Text length appropriateness (10% weight)
        # Very short texts (<10 chars) or very long texts (>5000 chars) get lower confidence
        if text_length < 10:
            length_score = 0.3
        elif text_length < 50:
            length_score = 0.7
        elif text_length <= 5000:
            length_score = 1.0
        else:
            length_score = 0.8
        score += length_score * 0.10

        return min(1.0, max(0.0, score))

    def enrich_batch(self, messages: list) -> list:
        """Enrich multiple messages.

        Args:
            messages: List of message dictionaries.

        Returns:
            List of enriched Message objects.
        """
        enriched = []
        for msg_dict in messages:
            try:
                enriched_msg = self.process(msg_dict)
                enriched.append(enriched_msg)
            except Exception as e:
                logger.exception(f"Failed to enrich message: {e}")
                continue

        logger.info(f"Enriched {len(enriched)}/{len(messages)} messages")
        return enriched

    def refresh_vocabulary(self) -> None:
        """Refresh the vocabulary from external sources."""
        self.vocabulary.refresh_from_external()
        self.system_prompt = self._create_system_prompt()
        logger.info("Enrichment agent vocabulary refreshed")
