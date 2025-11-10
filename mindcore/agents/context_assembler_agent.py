"""
Context Assembly AI Agent.

Retrieves and assembles relevant historical context.
"""
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..core.schemas import Message, AssembledContext
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm_providers import LLMProvider

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
    """

    def __init__(
        self,
        llm_provider: Optional['LLMProvider'] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize context assembler agent.

        Args:
            llm_provider: Optional LLM provider instance.
            api_key: API key (used if llm_provider not provided).
            model: Model name.
            temperature: Temperature for generation.
            system_prompt: Optional custom system prompt. If not provided, uses default.
        """
        super().__init__(llm_provider, api_key, model, temperature, max_tokens=1500)

        # Use custom prompt or default
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            # Import here to avoid circular dependency
            from ..prompts import CONTEXT_ASSEMBLY_SYSTEM_PROMPT
            self.system_prompt = CONTEXT_ASSEMBLY_SYSTEM_PROMPT

    def _create_system_prompt(self) -> str:
        """
        Create system prompt for context assembly.

        Deprecated: Use system_prompt parameter in __init__ instead.
        Kept for backward compatibility.

        Returns:
            System prompt string.
        """
        from ..prompts import CONTEXT_ASSEMBLY_SYSTEM_PROMPT
        return CONTEXT_ASSEMBLY_SYSTEM_PROMPT

    def process(self, messages: List[Message], query: str) -> AssembledContext:
        """
        Assemble context from messages based on query.

        Args:
            messages: List of Message objects.
            query: Current query or topic.

        Returns:
            AssembledContext object.
        """
        logger.debug(f"Assembling context from {len(messages)} messages for query: {query[:100]}...")

        # Format messages for the prompt
        formatted_messages = self._format_messages(messages)

        # Use prompts module for user prompt
        from ..prompts import get_context_assembly_prompt

        user_prompt = get_context_assembly_prompt(
            formatted_messages=formatted_messages,
            query=query
        )

        api_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Call LLM
            response = self._call_openai(api_messages)

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
                assembled_context="Context assembly failed due to an error.",
                key_points=[],
                relevant_message_ids=[],
                metadata={"error": str(e)}
            )

    def _format_messages(self, messages: List[Message]) -> str:
        """
        Format messages for inclusion in prompt.

        Args:
            messages: List of Message objects.

        Returns:
            Formatted string.
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
