"""
Metadata Enrichment AI Agent.

Enriches messages with intelligent metadata using LLM providers.
"""
from typing import Dict, Any, TYPE_CHECKING

from .base_agent import BaseAgent
from ..core.schemas import Message, MessageMetadata
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
        max_tokens: int = 800
    ):
        """
        Initialize enrichment agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for enrichment."""
        return """You are a metadata enrichment AI agent. Your task is to analyze messages and extract structured metadata.

For each message, you must return a JSON object with the following fields:

{
  "topics": ["list of main topics discussed"],
  "categories": ["list of categories like 'question', 'statement', 'command', 'code', 'technical', 'casual', etc."],
  "importance": 0.0-1.0 (float, where 1.0 is most important),
  "sentiment": {
    "overall": "positive/negative/neutral",
    "score": 0.0-1.0 (float)
  },
  "intent": "primary intent of the message (e.g., 'ask_question', 'provide_info', 'request_action', 'express_opinion', 'greeting', etc.)",
  "tags": ["relevant tags or keywords"],
  "entities": ["named entities like people, places, technologies, products"],
  "key_phrases": ["important phrases from the message"]
}

Be concise and accurate. Focus on extracting the most relevant information. Return ONLY valid JSON."""

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

            # Create MessageMetadata object
            metadata = MessageMetadata(
                topics=metadata_dict.get('topics', []),
                categories=metadata_dict.get('categories', []),
                importance=metadata_dict.get('importance', 0.5),
                sentiment=metadata_dict.get('sentiment', {}),
                intent=metadata_dict.get('intent'),
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
