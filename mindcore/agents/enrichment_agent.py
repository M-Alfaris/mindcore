"""
Metadata Enrichment AI Agent.

Enriches messages with intelligent metadata using LLM providers.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..core.schemas import Message, MessageMetadata, MetadataSchema, DEFAULT_METADATA_SCHEMA
from ..utils.logger import get_logger
from ..utils.helper import generate_message_id

if TYPE_CHECKING:
    from ..llm import BaseLLMProvider

logger = get_logger(__name__)


class EnrichmentAgent(BaseAgent):
    """
    AI agent that enriches messages with metadata.

    Analyzes message content and generates:
    - Topics
    - Categories
    - Importance score
    - Sentiment
    - Intent
    - Tags
    - Entities
    - Key phrases

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.AUTO, ...)
        >>> agent = EnrichmentAgent(provider)
        >>> message = agent.process({
        ...     "user_id": "user1",
        ...     "thread_id": "thread1",
        ...     "session_id": "session1",
        ...     "role": "user",
        ...     "text": "How do I implement caching?"
        ... })
        >>> print(message.metadata.topics)
        ['caching', 'implementation']
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.3,
        max_tokens: int = 800,
        metadata_schema: Optional[MetadataSchema] = None
    ):
        """
        Initialize enrichment agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            metadata_schema: Predefined metadata schema for consistent enrichment.
                           If None, uses DEFAULT_METADATA_SCHEMA.
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.metadata_schema = metadata_schema or DEFAULT_METADATA_SCHEMA
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for enrichment with predefined schema."""
        schema_list = self.metadata_schema.to_prompt_list()

        return f"""You are a metadata enrichment AI agent. Your task is to analyze messages and extract structured metadata.

IMPORTANT: You must ONLY use values from the predefined lists below. Do NOT invent new topics, categories, or intents.

{schema_list}

For each message, return a JSON object with these fields:

{{
  "topics": ["select 1-3 topics from the Available Topics list above"],
  "categories": ["select 1-2 categories from the Available Categories list above"],
  "importance": 0.0-1.0 (float, where 1.0 is most important),
  "sentiment": {{
    "overall": "select from Available Sentiments above",
    "score": 0.0-1.0 (float)
  }},
  "intent": "select ONE intent from Available Intents above",
  "tags": ["relevant keywords from the message"],
  "entities": ["named entities like people, places, technologies, products"],
  "key_phrases": ["important phrases from the message"]
}}

Rules:
- ONLY use values from the predefined lists for topics, categories, intent, and sentiment
- If no topic matches, use the closest one or "general"
- importance: 0.0-0.3 for greetings/casual, 0.4-0.6 for normal, 0.7-1.0 for urgent/important
- Be concise and accurate. Return ONLY valid JSON."""

    def process(self, message_dict: Dict[str, Any]) -> Message:
        """
        Enrich a message with metadata.

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
        """
        text = message_dict.get('text', '')
        role = message_dict.get('role', 'user')

        logger.debug(f"Enriching message ({self.provider_name}): {text[:100]}...")

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Message to analyze:\nRole: {role}\nText: {text}"}
        ]

        metadata = None
        try:
            # Call LLM
            response = self._call_llm(messages, json_mode=True)

            # Parse response
            metadata_dict = self._parse_json_response(response)

            # Validate and normalize against predefined schema
            raw_topics = metadata_dict.get('topics', [])
            raw_categories = metadata_dict.get('categories', [])
            raw_intent = metadata_dict.get('intent')

            # Use schema validation (allows LLM flexibility but ensures consistency)
            validated_topics = self.metadata_schema.validate_topics(raw_topics) or raw_topics[:3]
            validated_categories = self.metadata_schema.validate_categories(raw_categories) or raw_categories[:2]
            validated_intent = self.metadata_schema.validate_intent(raw_intent) or raw_intent

            # Create MessageMetadata object with validated values
            metadata = MessageMetadata(
                topics=validated_topics,
                categories=validated_categories,
                importance=max(0.0, min(1.0, metadata_dict.get('importance', 0.5))),
                sentiment=metadata_dict.get('sentiment', {}),
                intent=validated_intent,
                tags=metadata_dict.get('tags', []),
                entities=metadata_dict.get('entities', []),
                key_phrases=metadata_dict.get('key_phrases', []),
                enrichment_failed=False,
                enrichment_error=None
            )

            logger.info(f"Successfully enriched message with topics: {metadata.topics}")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Enrichment failed: {error_msg}")

            # Create metadata that indicates failure
            metadata = MessageMetadata(
                enrichment_failed=True,
                enrichment_error=error_msg
            )

        # Create Message object
        message = Message(
            message_id=message_dict.get('message_id') or generate_message_id(),
            user_id=message_dict['user_id'],
            thread_id=message_dict['thread_id'],
            session_id=message_dict['session_id'],
            role=role,
            raw_text=text,
            metadata=metadata
        )

        return message

    def enrich_batch(self, messages: list) -> list:
        """
        Enrich multiple messages.

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
                logger.error(f"Failed to enrich message: {e}")
                continue

        logger.info(f"Enriched {len(enriched)}/{len(messages)} messages")
        return enriched
