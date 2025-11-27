"""
Context Assembly AI Agent.

Retrieves and assembles relevant historical context using LLM providers.
"""
from typing import List, Dict, Any, TYPE_CHECKING

from .base_agent import BaseAgent
from ..core.schemas import Message, AssembledContext
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm import BaseLLMProvider

logger = get_logger(__name__)


class ContextAssemblerAgent(BaseAgent):
    """
    AI agent that assembles relevant context from historical messages.

    Uses LLM reasoning to:
    - Analyze message history
    - Identify relevant messages
    - Summarize key information
    - Extract key points
    - Return structured context

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.AUTO, ...)
        >>> agent = ContextAssemblerAgent(provider)
        >>> context = agent.process(messages, "What did we discuss about caching?")
        >>> print(context.key_points)
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.3,
        max_tokens: int = 1500
    ):
        """
        Initialize context assembler agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for context assembly."""
        return """You are a context assembly AI agent. Your task is to analyze a conversation history and current query to extract and summarize the most relevant context.

You will receive:
1. A list of historical messages with metadata
2. A current query or topic

Your task is to return a JSON object with:

{
  "assembled_context": "A clear, concise summary of relevant historical context that would help understand the current query. Focus on key facts, decisions, and relevant background.",
  "key_points": ["List of 3-5 most important points from the history"],
  "relevant_message_ids": ["List of message IDs that were most relevant"],
  "metadata": {
    "topics": ["main topics covered"],
    "sentiment": {
      "overall": "positive/negative/neutral",
      "trend": "improving/declining/stable"
    },
    "importance": 0.0-1.0 (overall importance of this context)
  }
}

Be selective and concise. Only include truly relevant information. Return ONLY valid JSON."""

    def process(self, messages: List[Message], query: str) -> AssembledContext:
        """
        Assemble context from messages based on query.

        Args:
            messages: List of Message objects with metadata.
            query: Current query or topic to find relevant context for.

        Returns:
            AssembledContext object containing:
            - assembled_context: Summarized relevant context
            - key_points: List of key points
            - relevant_message_ids: IDs of relevant messages
            - metadata: Topics, sentiment, importance
        """
        logger.debug(
            f"Assembling context from {len(messages)} messages "
            f"({self.provider_name}) for query: {query[:100]}..."
        )

        # Format messages for the prompt
        formatted_messages = self._format_messages(messages)

        # Prepare messages for LLM
        prompt = f"""Historical Messages:
{formatted_messages}

Current Query/Topic:
{query}

Analyze the messages and provide relevant context for the current query."""

        api_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            # Call LLM
            response = self._call_llm(api_messages, json_mode=True)

            # Parse response
            context_dict = self._parse_json_response(response)

            # Create AssembledContext object
            assembled_context = AssembledContext(
                assembled_context=context_dict.get('assembled_context', ''),
                key_points=context_dict.get('key_points', []),
                relevant_message_ids=context_dict.get('relevant_message_ids', []),
                metadata=context_dict.get('metadata', {})
            )

            logger.info(f"Successfully assembled context with {len(assembled_context.key_points)} key points")
            return assembled_context

        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            # Return empty context on failure
            return AssembledContext(
                assembled_context="",
                key_points=[],
                relevant_message_ids=[],
                metadata={
                    "assembly_failed": True,
                    "error_type": type(e).__name__
                }
            )

    def _format_messages(self, messages: List[Message]) -> str:
        """
        Format messages for inclusion in prompt.

        Args:
            messages: List of Message objects.

        Returns:
            Formatted string with message content and metadata.
        """
        formatted = []

        for msg in messages:
            metadata_str = ""
            if msg.metadata:
                topics = msg.metadata.topics if hasattr(msg.metadata, 'topics') else []
                intent = msg.metadata.intent if hasattr(msg.metadata, 'intent') else None
                importance = msg.metadata.importance if hasattr(msg.metadata, 'importance') else 0.5

                metadata_parts = []
                if topics:
                    metadata_parts.append(f"topics: {', '.join(topics)}")
                if intent:
                    metadata_parts.append(f"intent: {intent}")
                metadata_parts.append(f"importance: {importance:.2f}")

                metadata_str = f" [{' | '.join(metadata_parts)}]"

            formatted.append(
                f"[{msg.message_id}] {msg.role.value}: {msg.raw_text}{metadata_str}"
            )

        return "\n".join(formatted)

    def assemble_for_prompt(self, messages: List[Message], query: str) -> str:
        """
        Assemble context and format it for direct inclusion in a prompt.

        Args:
            messages: List of Message objects.
            query: Current query or topic.

        Returns:
            Formatted context string ready for prompt injection.
        """
        context = self.process(messages, query)

        parts = []

        if context.assembled_context:
            parts.append("## Historical Context")
            parts.append(context.assembled_context)
            parts.append("")

        if context.key_points:
            parts.append("## Key Points from History")
            for i, point in enumerate(context.key_points, 1):
                parts.append(f"{i}. {point}")
            parts.append("")

        return "\n".join(parts)
