"""Retrieval Query Agent.

Uses LLM with tool calling to analyze queries and extract search parameters.
NO keyword matching or regex - pure LLM understanding of semantic intent.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mindcore.core.schemas import DEFAULT_METADATA_SCHEMA, MetadataSchema
from mindcore.utils.logger import get_logger

from .base_agent import BaseAgent


if TYPE_CHECKING:
    from mindcore.llm import BaseLLMProvider

logger = get_logger(__name__)


@dataclass
class QueryIntent:
    """Structured query intent from LLM analysis."""

    topics: list[str]
    categories: list[str]
    intent: str | None
    entities: list[dict[str, str]]
    time_scope: str
    search_current_thread_only: bool
    confidence: str


class RetrievalQueryAgent(BaseAgent):
    """AI agent that analyzes queries to extract search parameters.

    Uses LLM with tool calling to understand semantic INTENT of queries,
    not just keywords. This is critical for handling cases like:
    - "I don't need a refund" -> should NOT search for refund-related messages
    - "What did we discuss about my order?" -> should search cross-thread

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.AUTO, ...)
        >>> agent = RetrievalQueryAgent(provider)
        >>> intent = agent.analyze_query(
        ...     "I don't need help with refunds, I need understanding",
        ...     recent_context="User asked about policy..."
        ... )
        >>> print(intent.topics)
        ['general_inquiry']  # NOT 'refund'!
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.2,
        max_tokens: int = 500,
        metadata_schema: MetadataSchema | None = None,
    ):
        """Initialize retrieval query agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation (lower for consistency)
            max_tokens: Maximum tokens in response
            metadata_schema: Predefined metadata schema for vocabulary constraints
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.metadata_schema = metadata_schema or DEFAULT_METADATA_SCHEMA
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for query analysis."""
        schema_list = self.metadata_schema.to_prompt_list()

        return f"""You are a query analysis agent. Your task is to understand the semantic INTENT of a user's query and extract search parameters.

CRITICAL: Focus on what the user WANTS, not keywords they mention.
- "I don't want a refund" → topic should be 'general' or 'feedback', NOT 'refund'
- "I'm not here for billing help" → category should NOT be 'billing'
- Consider negations and context carefully

{schema_list}

Analyze the query and return a JSON object with:
{{
  "topics": ["select 1-3 topics based on semantic INTENT"],
  "categories": ["select 1-2 categories based on PURPOSE"],
  "intent": "the user's intent from Available Intents",
  "entities": [{{"type": "entity_type", "value": "extracted value"}}],
  "time_scope": "today|this_week|this_month|all_time",
  "search_current_thread_only": true/false,
  "confidence": "high|medium|low"
}}

Guidelines for search_current_thread_only:
- true: Query is about current conversation ("what did we just discuss", "earlier you said")
- false: Query references past/other conversations ("last time", "previously", "in our other chat")
- false: Query is about a new topic or general history

Return ONLY valid JSON."""

    def process(self, query: str, recent_context: str | None = None) -> QueryIntent:
        """Analyze a query to extract search parameters.

        Args:
            query: User's query string.
            recent_context: Optional recent conversation context.

        Returns:
            QueryIntent with extracted search parameters.
        """
        return self.analyze_query(query, recent_context)

    def analyze_query(self, query: str, recent_context: str | None = None) -> QueryIntent:
        """Analyze query using LLM to extract search parameters.

        This method uses LLM understanding instead of keyword matching,
        which correctly handles negations and semantic nuance.

        Args:
            query: User's query string.
            recent_context: Optional recent messages for context.

        Returns:
            QueryIntent with topics, categories, intent, etc.
        """
        logger.debug(f"Analyzing query: {query[:100]}...")

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]

        if recent_context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Recent conversation:\n{recent_context}\n\nQuery to analyze: {query}",
                }
            )
        else:
            messages.append({"role": "user", "content": f"Query to analyze: {query}"})

        try:
            # Call LLM
            response = self._call_llm(messages, json_mode=True)

            # Parse response
            data = self._parse_json_response(response)

            # Validate against schema
            raw_topics = data.get("topics", [])
            raw_categories = data.get("categories", [])
            raw_intent = data.get("intent")

            validated_topics = self.metadata_schema.validate_topics(raw_topics)
            validated_categories = self.metadata_schema.validate_categories(raw_categories)
            validated_intent = self.metadata_schema.validate_intent(raw_intent)

            # Use validated values, fall back to raw if validation returns empty
            topics = validated_topics if validated_topics else raw_topics[:3]
            categories = validated_categories if validated_categories else raw_categories[:2]
            intent = validated_intent or raw_intent

            result = QueryIntent(
                topics=topics,
                categories=categories,
                intent=intent,
                entities=data.get("entities", []),
                time_scope=data.get("time_scope", "all_time"),
                search_current_thread_only=data.get("search_current_thread_only", False),
                confidence=data.get("confidence", "medium"),
            )

            logger.info(
                f"Query analyzed: topics={result.topics}, "
                f"categories={result.categories}, intent={result.intent}"
            )
            return result

        except Exception as e:
            logger.warning(f"Query analysis failed: {e}, using defaults")
            # Return safe defaults on failure
            return QueryIntent(
                topics=[],
                categories=[],
                intent=None,
                entities=[],
                time_scope="all_time",
                search_current_thread_only=False,
                confidence="low",
            )
