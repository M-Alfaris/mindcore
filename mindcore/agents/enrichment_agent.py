"""
Metadata Enrichment AI Agent.

Enriches messages with intelligent metadata.
"""
import json
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..core.schemas import Message, MessageMetadata
from ..utils.logger import get_logger
from ..utils.helper import generate_message_id

if TYPE_CHECKING:
    from ..llm_providers import LLMProvider
    from ..importance import ImportanceAlgorithm

logger = get_logger(__name__)


class EnrichmentAgent(BaseAgent):
    """
    AI agent that enriches messages with metadata.

    Analyzes message content and generates:
    - Topics
    - Categories
    - Importance score (using configurable algorithm)
    - Sentiment
    - Intent
    - Tags
    - Entities
    - Key phrases
    """

    def __init__(
        self,
        llm_provider: Optional['LLMProvider'] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        importance_algorithm: Optional['ImportanceAlgorithm'] = None
    ):
        """
        Initialize enrichment agent.

        Args:
            llm_provider: Optional LLM provider instance.
            api_key: API key (used if llm_provider not provided).
            model: Model name.
            temperature: Temperature for generation.
            system_prompt: Optional custom system prompt. If not provided, uses default.
            importance_algorithm: Optional importance algorithm. If not provided, uses LLM-based.
        """
        super().__init__(llm_provider, api_key, model, temperature, max_tokens=800)

        # Use custom prompt or default
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            # Import here to avoid circular dependency
            from ..prompts import ENRICHMENT_SYSTEM_PROMPT
            self.system_prompt = ENRICHMENT_SYSTEM_PROMPT

        # Set importance algorithm
        if importance_algorithm:
            self.importance_algorithm = importance_algorithm
        else:
            # Default to LLM-based importance
            from ..importance import LLMBasedImportance
            self.importance_algorithm = LLMBasedImportance()

        logger.info(f"EnrichmentAgent using importance algorithm: {self.importance_algorithm.__class__.__name__}")

    def _create_system_prompt(self) -> str:
        """
        Create system prompt for enrichment.

        Deprecated: Use system_prompt parameter in __init__ instead.
        Kept for backward compatibility.

        Returns:
            System prompt string.
        """
        from ..prompts import ENRICHMENT_SYSTEM_PROMPT
        return ENRICHMENT_SYSTEM_PROMPT

    def process(self, message_dict: Dict[str, Any]) -> Message:
        """
        Enrich a message with metadata.

        Args:
            message_dict: Dictionary containing message fields.

        Returns:
            Enriched Message object.
        """
        # Extract message text
        text = message_dict.get('text', '')
        role = message_dict.get('role', 'user')

        logger.debug(f"Enriching message: {text[:100]}...")

        # Prepare messages for LLM using prompts module
        from ..prompts import get_enrichment_prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": get_enrichment_prompt(role=role, text=text)}
        ]

        try:
            # Call LLM
            response = self._call_openai(messages)

            # Parse response
            metadata_dict = self._parse_json_response(response)

            # Override importance with custom algorithm if not using LLM-based
            from ..importance import LLMBasedImportance
            if not isinstance(self.importance_algorithm, LLMBasedImportance):
                # Use custom importance algorithm
                importance = self.importance_algorithm.calculate(text, metadata=metadata_dict)
                metadata_dict['importance'] = importance
                logger.debug(f"Calculated importance: {importance:.2f} using {self.importance_algorithm.__class__.__name__}")

            # Create MessageMetadata object
            metadata = MessageMetadata(
                topics=metadata_dict.get('topics', []),
                categories=metadata_dict.get('categories', []),
                importance=metadata_dict.get('importance', 0.5),
                sentiment=metadata_dict.get('sentiment', {}),
                intent=metadata_dict.get('intent'),
                tags=metadata_dict.get('tags', []),
                entities=metadata_dict.get('entities', []),
                key_phrases=metadata_dict.get('key_phrases', [])
            )

            logger.info(f"Successfully enriched message with topics: {metadata.topics}, importance: {metadata.importance:.2f}")

        except Exception as e:
            logger.warning(f"Enrichment failed, using default metadata: {e}")
            # Fallback to default metadata
            metadata = MessageMetadata()

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
                # Skip failed messages or add with default metadata
                continue

        logger.info(f"Enriched {len(enriched)}/{len(messages)} messages")
        return enriched
