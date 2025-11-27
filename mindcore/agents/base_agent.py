"""
Base agent class for Mindcore AI agents.

Uses the LLM provider abstraction layer for all inference.
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm import BaseLLMProvider

logger = get_logger(__name__)


class AgentInitializationError(Exception):
    """Raised when agent initialization fails."""
    pass


class APICallError(Exception):
    """Raised when LLM call fails."""
    pass


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Uses the LLM provider abstraction layer for all inference operations.
    Supports llama.cpp, OpenAI, and automatic fallback between them.

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(
        ...     ProviderType.AUTO,
        ...     llama_config={"model_path": "~/.mindcore/models/model.gguf"},
        ...     openai_config={"api_key": "sk-..."}
        ... )
        >>> agent = MyAgent(llm_provider=provider)
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize base agent.

        Args:
            llm_provider: LLM provider instance from mindcore.llm
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens in response

        Raises:
            AgentInitializationError: If provider is None or invalid
        """
        if llm_provider is None:
            raise AgentInitializationError(
                "LLM provider is required. Use create_provider() from mindcore.llm:\n"
                "  from mindcore.llm import create_provider, ProviderType\n"
                "  provider = create_provider(ProviderType.AUTO, ...)"
            )

        self._llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"{llm_provider.name} provider"
        )

    @property
    def provider_name(self) -> str:
        """Get the name of the active LLM provider."""
        return self._llm_provider.name

    @property
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        return self._llm_provider.is_available()

    def _call_llm(
        self,
        messages: list,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            json_mode: If True, request JSON output format.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Response content as string.

        Raises:
            APICallError: If the LLM call fails.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self._llm_provider.generate(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                json_mode=json_mode
            )
            latency_str = f"{response.latency_ms:.0f}ms" if response.latency_ms else "N/A"
            logger.debug(
                f"LLM call successful ({self._llm_provider.name}): "
                f"{len(response.content)} chars, {latency_str}"
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise APICallError(f"LLM call failed: {e}") from e

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM.

        Handles various formats including:
        - Plain JSON
        - JSON in markdown code blocks
        - JSON embedded in text

        Args:
            response: Response string from LLM.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If response cannot be parsed as JSON.
        """
        if not response or not isinstance(response, str):
            raise ValueError("Empty or invalid response from LLM")

        original_response = response

        try:
            # Try to extract JSON from code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()

            # Try parsing the extracted/original response
            return json.loads(response)

        except json.JSONDecodeError as e:
            # Try to find JSON object in the response
            try:
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end > start:
                    json_str = response[start:end + 1]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {original_response[:500]}...")
            raise ValueError(
                f"Failed to parse LLM response as JSON. Response started with: "
                f"{original_response[:100]}..."
            ) from e

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process method to be implemented by subclasses.

        Returns:
            Processing result.
        """
        pass
