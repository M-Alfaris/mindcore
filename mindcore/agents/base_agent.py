"""
Base agent class for Mindcore AI agents.
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm_providers import LLMProvider

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Provides common functionality for LLM provider interactions.
    Supports multiple LLM providers (OpenAI, Ollama, Anthropic, etc.)
    """

    def __init__(
        self,
        llm_provider: Optional['LLMProvider'] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize base agent.

        Args:
            llm_provider: Optional LLMProvider instance. If provided, uses this provider.
                         Otherwise, creates OpenAI provider with api_key and model.
            api_key: API key (used if llm_provider not provided).
            model: Model name (default: gpt-4o-mini).
            temperature: Temperature for generation.
            max_tokens: Maximum tokens in response.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use provided LLM provider or create default OpenAI provider
        if llm_provider:
            self.llm_provider = llm_provider
            self.model = llm_provider.model
        else:
            # Lazy import to avoid circular dependency
            from ..llm_providers import OpenAIProvider
            self.llm_provider = OpenAIProvider(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.model = model

        # Legacy attributes for backward compatibility
        self.api_key = api_key

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model}")

    def _call_openai(
        self,
        messages: list,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call LLM provider (supports OpenAI, Ollama, Anthropic, etc.)

        Note: Method name kept as _call_openai for backward compatibility,
        but now works with any LLM provider.

        Args:
            messages: List of message dictionaries.
            response_format: Optional response format specification (ignored for some providers).
            temperature: Override temperature.
            max_tokens: Override max_tokens.

        Returns:
            Response content as string.
        """
        try:
            content = self.llm_provider.chat_completion(
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            logger.debug(f"LLM API call successful: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from OpenAI.

        Args:
            response: Response string.

        Returns:
            Parsed JSON dictionary.
        """
        try:
            # Try to extract JSON from code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process method to be implemented by subclasses.

        Returns:
            Processing result.
        """
        pass
