"""
OpenAI LLM Provider for Mindcore.

Provides OpenAI API integration as a fallback provider.
Also supports any OpenAI-compatible API (vLLM, Ollama, LocalAI, etc.) via base_url.
"""
import time
from typing import List, Dict, Optional, Any

from .base_provider import (
    BaseLLMProvider,
    LLMResponse,
    LLMProviderError,
    GenerationError,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider for Mindcore.

    Serves as the fallback provider when llama.cpp is unavailable or fails.
    Uses gpt-4o-mini by default for cost-effective inference.

    Also supports any OpenAI-compatible API server via the base_url parameter:
    - vLLM: base_url="http://localhost:8000/v1"
    - Ollama: base_url="http://localhost:11434/v1"
    - LocalAI: base_url="http://localhost:8080/v1"
    - text-generation-webui: base_url="http://localhost:5000/v1"
    - Any other OpenAI-compatible server

    Features:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - JSON mode support
    - Connection timeout handling
    - Custom base_url for self-hosted LLMs

    Example:
        >>> # Using OpenAI API
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="gpt-4o-mini"
        ... )
        >>>
        >>> # Using self-hosted vLLM server
        >>> provider = OpenAIProvider(
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="not-needed",  # Some servers require any non-empty string
        ...     model="meta-llama/Llama-3.2-3B-Instruct"
        ... )
        >>>
        >>> response = provider.generate([
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.content)
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # Base delay in seconds
    RETRY_DELAY_MAX = 30.0  # Maximum delay in seconds
    REQUEST_TIMEOUT = 60  # Timeout in seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
                For self-hosted servers, use any non-empty string if required.
            model: Model to use (default: gpt-4o-mini)
            base_url: Custom API base URL for OpenAI-compatible servers.
                Examples:
                - vLLM: "http://localhost:8000/v1"
                - Ollama: "http://localhost:11434/v1"
                - LocalAI: "http://localhost:8080/v1"
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
            **kwargs: Additional arguments for OpenAI client

        Note:
            Provider will be unavailable (is_available() = False) if no API key
            is provided. This allows graceful fallback behavior.
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._api_key = api_key

        if api_key:
            self._init_client(api_key, base_url, **kwargs)
        else:
            logger.warning(
                "OpenAI API key not provided. OpenAI provider will be unavailable. "
                "Set OPENAI_API_KEY environment variable or provide api_key parameter."
            )

    def _init_client(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            client_kwargs = {
                "api_key": api_key,
                "timeout": self.timeout,
                **kwargs
            }

            if base_url:
                client_kwargs["base_url"] = base_url

            self._client = OpenAI(**client_kwargs)

            if base_url:
                logger.info(
                    f"OpenAI-compatible provider initialized: "
                    f"model={self.model}, base_url={base_url}"
                )
            else:
                logger.info(f"OpenAI provider initialized with model: {self.model}")
        except ImportError as e:
            raise LLMProviderError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise LLMProviderError(f"Failed to initialize OpenAI: {e}") from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate a response using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            json_mode: If True, constrain output to valid JSON

        Returns:
            LLMResponse with generated content

        Raises:
            GenerationError: If generation fails after all retries
        """
        if not self.is_available():
            raise GenerationError(
                "OpenAI provider not available. "
                "Provide API key or set OPENAI_API_KEY environment variable."
            )

        # Import error types
        from openai import (
            APIError,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        )

        start_time = time.time()

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else None
                latency_ms = (time.time() - start_time) * 1000

                logger.debug(
                    f"OpenAI generation complete: {len(content)} chars, "
                    f"{tokens_used or 'N/A'} tokens, {latency_ms:.0f}ms"
                )

                return LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used=tokens_used,
                    provider="openai",
                    latency_ms=latency_ms,
                    metadata={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                        "completion_tokens": response.usage.completion_tokens if response.usage else None,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                )

            except RateLimitError as e:
                last_exception = e
                delay = min(
                    self.RETRY_DELAY_BASE * (2 ** attempt),
                    self.RETRY_DELAY_MAX
                )
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                delay = min(
                    self.RETRY_DELAY_BASE * (2 ** attempt),
                    self.RETRY_DELAY_MAX
                )
                logger.warning(
                    f"Connection error, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)

            except APIError as e:
                # Retry on 5xx server errors
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    last_exception = e
                    delay = min(
                        self.RETRY_DELAY_BASE * (2 ** attempt),
                        self.RETRY_DELAY_MAX
                    )
                    logger.warning(
                        f"Server error {e.status_code}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    # Don't retry client errors (4xx)
                    logger.error(f"OpenAI API client error: {e}")
                    raise GenerationError(f"OpenAI API error: {e}") from e

            except Exception as e:
                # Unknown error, don't retry
                logger.error(f"OpenAI generation failed with unexpected error: {e}")
                raise GenerationError(f"OpenAI generation failed: {e}") from e

        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"OpenAI generation failed after {self.max_retries} retries "
            f"({latency_ms:.0f}ms): {last_exception}"
        )
        raise GenerationError(
            f"OpenAI API call failed after {self.max_retries} retries"
        ) from last_exception

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate a response with tool calling support.

        Args:
            messages: List of message dicts (can include tool calls and results)
            tools: List of tool definitions in OpenAI format
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}

        Returns:
            Dict with 'content' and optionally 'tool_calls'

        Raises:
            GenerationError: If generation fails after all retries
        """
        if not self.is_available():
            raise GenerationError(
                "OpenAI provider not available. "
                "Provide API key or set OPENAI_API_KEY environment variable."
            )

        from openai import (
            APIError,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        )

        start_time = time.time()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)

                message = response.choices[0].message
                latency_ms = (time.time() - start_time) * 1000

                result = {
                    "content": message.content,
                    "tool_calls": None
                }

                # Extract tool calls if present
                if message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]

                logger.debug(
                    f"OpenAI tool call complete: {latency_ms:.0f}ms, "
                    f"tool_calls={len(result['tool_calls'] or [])}"
                )

                return result

            except RateLimitError as e:
                last_exception = e
                delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                logger.warning(f"Rate limit hit, retrying in {delay:.1f}s")
                time.sleep(delay)

            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                logger.warning(f"Connection error, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)

            except APIError as e:
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    last_exception = e
                    delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                    logger.warning(f"Server error, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise GenerationError(f"OpenAI API error: {e}") from e

            except Exception as e:
                raise GenerationError(f"OpenAI tool call failed: {e}") from e

        raise GenerationError(
            f"OpenAI tool call failed after {self.max_retries} retries"
        ) from last_exception

    def is_available(self) -> bool:
        """Check if the OpenAI client is initialized."""
        return self._client is not None

    @property
    def name(self) -> str:
        """Provider name."""
        if self.base_url:
            return "openai-compatible"
        return "openai"

    def close(self) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            # OpenAI client doesn't require explicit cleanup
            self._client = None
            logger.debug("OpenAI provider closed")

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        if self.base_url:
            return f"OpenAIProvider(model={self.model}, base_url={self.base_url}, {status})"
        return f"OpenAIProvider(model={self.model}, {status})"
