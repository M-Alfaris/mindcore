"""Base LLM Provider interface for Mindcore.

Provides a unified abstraction layer for different LLM backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Unified response from any LLM provider.

    Attributes:
        content: The generated text response
        model: Model identifier used for generation
        tokens_used: Total tokens consumed (if available)
        provider: Name of the provider (e.g., "llama.cpp", "openai")
        latency_ms: Response time in milliseconds
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    tokens_used: int | None = None
    provider: str = "unknown"
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class ModelNotFoundError(LLMProviderError):
    """Raised when the specified model cannot be found."""


class GenerationError(LLMProviderError):
    """Raised when text generation fails."""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    This interface allows Mindcore to work with different LLM backends
    (llama.cpp, OpenAI, etc.) through a unified API.

    Implementations should handle:
    - Model initialization and lifecycle
    - Chat completion generation
    - Error handling and retries
    - Resource cleanup
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles: 'system', 'user', 'assistant'
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in the response
            json_mode: If True, constrain output to valid JSON

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            GenerationError: If generation fails
            LLMProviderError: For other provider-specific errors
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to generate.

        Returns:
            True if the provider can accept generation requests
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification.

        Returns:
            Human-readable provider name (e.g., "llama.cpp", "openai")
        """

    def close(self) -> None:
        """Clean up provider resources.

        Override this method if the provider needs cleanup
        (e.g., unloading models, closing connections).
        """

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False
