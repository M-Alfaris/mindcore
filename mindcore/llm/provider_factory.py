"""
LLM Provider Factory for Mindcore.

Provides factory functions to create and configure LLM providers
with automatic fallback support.
"""
from enum import Enum
from typing import Optional, Dict, Any, List

from .base_provider import (
    BaseLLMProvider,
    LLMResponse,
    LLMProviderError,
    GenerationError,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProviderType(Enum):
    """Available LLM provider types."""
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    AUTO = "auto"  # Try llama.cpp first, fallback to OpenAI


class FallbackProvider(BaseLLMProvider):
    """
    Provider wrapper that implements automatic fallback.

    Tries the primary provider first, and if it fails, automatically
    falls back to the secondary provider. This enables reliable inference
    with llama.cpp as primary (fast, free) and OpenAI as backup (reliable).

    Example:
        >>> primary = LlamaCppProvider(model_path="...")
        >>> fallback = OpenAIProvider(api_key="...")
        >>> provider = FallbackProvider(primary, fallback)
        >>> # Uses llama.cpp, falls back to OpenAI on failure
        >>> response = provider.generate(messages)
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
        fallback_on_error: bool = True
    ):
        """
        Initialize fallback provider.

        Args:
            primary: Primary provider (tried first)
            fallback: Fallback provider (used if primary fails)
            fallback_on_error: If True, fall back on any error.
                              If False, only fall back if primary is unavailable.
        """
        self.primary = primary
        self.fallback = fallback
        self.fallback_on_error = fallback_on_error
        self._last_used_provider: Optional[str] = None

        logger.info(
            f"FallbackProvider initialized: {primary.name} -> {fallback.name}"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate response with automatic fallback.

        Tries primary provider first. If it fails or is unavailable,
        automatically falls back to the secondary provider.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Enable JSON output mode

        Returns:
            LLMResponse from whichever provider succeeded

        Raises:
            GenerationError: If both providers fail
        """
        primary_error = None

        # Try primary provider
        if self.primary.is_available():
            try:
                response = self.primary.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode
                )
                self._last_used_provider = self.primary.name
                return response

            except Exception as e:
                primary_error = e
                if self.fallback_on_error:
                    logger.warning(
                        f"Primary provider ({self.primary.name}) failed: {e}. "
                        f"Falling back to {self.fallback.name}"
                    )
                else:
                    raise
        else:
            logger.info(
                f"Primary provider ({self.primary.name}) unavailable, "
                f"using fallback ({self.fallback.name})"
            )

        # Try fallback provider
        if self.fallback.is_available():
            try:
                response = self.fallback.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode
                )
                self._last_used_provider = self.fallback.name

                # Add fallback metadata
                response.metadata["fallback_used"] = True
                response.metadata["primary_error"] = str(primary_error) if primary_error else None

                return response

            except Exception as e:
                logger.error(f"Fallback provider ({self.fallback.name}) also failed: {e}")
                if primary_error:
                    raise GenerationError(
                        f"Both providers failed. "
                        f"Primary ({self.primary.name}): {primary_error}. "
                        f"Fallback ({self.fallback.name}): {e}"
                    ) from e
                raise

        # Neither provider available
        raise GenerationError(
            f"No providers available. "
            f"Primary ({self.primary.name}): unavailable. "
            f"Fallback ({self.fallback.name}): unavailable."
        )

    def is_available(self) -> bool:
        """Check if either provider is available."""
        return self.primary.is_available() or self.fallback.is_available()

    @property
    def name(self) -> str:
        """Combined provider name."""
        return f"{self.primary.name}+{self.fallback.name}"

    @property
    def last_used_provider(self) -> Optional[str]:
        """Get the name of the last provider that was used."""
        return self._last_used_provider

    def close(self) -> None:
        """Close both providers."""
        self.primary.close()
        self.fallback.close()

    def get_status(self) -> Dict[str, Any]:
        """Get status of both providers."""
        return {
            "primary": {
                "name": self.primary.name,
                "available": self.primary.is_available(),
            },
            "fallback": {
                "name": self.fallback.name,
                "available": self.fallback.is_available(),
            },
            "last_used": self._last_used_provider,
        }

    def __repr__(self) -> str:
        primary_status = "ok" if self.primary.is_available() else "unavailable"
        fallback_status = "ok" if self.fallback.is_available() else "unavailable"
        return (
            f"FallbackProvider("
            f"{self.primary.name}={primary_status}, "
            f"{self.fallback.name}={fallback_status})"
        )


def create_provider(
    provider_type: ProviderType = ProviderType.AUTO,
    llama_config: Optional[Dict[str, Any]] = None,
    openai_config: Optional[Dict[str, Any]] = None
) -> BaseLLMProvider:
    """
    Factory function to create LLM providers.

    Creates the appropriate provider based on the specified type and
    configuration. Supports automatic fallback from llama.cpp to OpenAI.

    Args:
        provider_type: Type of provider to create:
            - LLAMA_CPP: Local llama.cpp inference only
            - OPENAI: OpenAI API only
            - AUTO: llama.cpp primary with OpenAI fallback (recommended)
        llama_config: Configuration for llama.cpp provider:
            - model_path (str, required): Path to GGUF model
            - n_ctx (int): Context window size (default: 4096)
            - n_threads (int): CPU threads (default: auto)
            - n_gpu_layers (int): GPU layers (default: 0 = CPU only)
        openai_config: Configuration for OpenAI provider:
            - api_key (str): OpenAI API key
            - model (str): Model name (default: gpt-4o-mini)

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If no provider could be initialized
        LLMProviderError: If provider initialization fails

    Example:
        >>> # Auto mode with fallback (recommended)
        >>> provider = create_provider(
        ...     provider_type=ProviderType.AUTO,
        ...     llama_config={"model_path": "~/.mindcore/models/llama.gguf"},
        ...     openai_config={"api_key": "sk-..."}
        ... )

        >>> # llama.cpp only (offline)
        >>> provider = create_provider(
        ...     provider_type=ProviderType.LLAMA_CPP,
        ...     llama_config={"model_path": "~/.mindcore/models/llama.gguf"}
        ... )

        >>> # OpenAI only
        >>> provider = create_provider(
        ...     provider_type=ProviderType.OPENAI,
        ...     openai_config={"api_key": "sk-..."}
        ... )
    """
    llama_config = llama_config or {}
    openai_config = openai_config or {}

    # Import providers here to avoid circular imports
    from .llama_cpp_provider import LlamaCppProvider
    from .openai_provider import OpenAIProvider

    if provider_type == ProviderType.LLAMA_CPP:
        if not llama_config.get("model_path"):
            raise ValueError(
                "model_path is required for llama.cpp provider. "
                "Set MINDCORE_LLAMA_MODEL_PATH or provide in config."
            )
        return LlamaCppProvider(**llama_config)

    elif provider_type == ProviderType.OPENAI:
        if not openai_config.get("api_key"):
            raise ValueError(
                "api_key is required for OpenAI provider. "
                "Set OPENAI_API_KEY or provide in config."
            )
        return OpenAIProvider(**openai_config)

    elif provider_type == ProviderType.AUTO:
        # Create providers, allowing failures
        primary = None
        fallback = None
        errors = []

        # Try to create llama.cpp provider
        if llama_config.get("model_path"):
            try:
                primary = LlamaCppProvider(**llama_config)
                logger.info(f"Primary provider (llama.cpp) initialized")
            except Exception as e:
                errors.append(f"llama.cpp: {e}")
                logger.warning(f"Could not initialize llama.cpp provider: {e}")

        # Try to create OpenAI provider
        if openai_config.get("api_key"):
            try:
                fallback = OpenAIProvider(**openai_config)
                logger.info(f"Fallback provider (OpenAI) initialized")
            except Exception as e:
                errors.append(f"OpenAI: {e}")
                logger.warning(f"Could not initialize OpenAI provider: {e}")

        # Return appropriate provider
        if primary and fallback:
            return FallbackProvider(primary, fallback)
        elif primary:
            logger.info("Using llama.cpp only (no OpenAI fallback configured)")
            return primary
        elif fallback:
            logger.info("Using OpenAI only (llama.cpp not configured or failed)")
            return fallback
        else:
            raise ValueError(
                f"No LLM provider could be initialized. Errors: {'; '.join(errors)}\n"
                "Configure either:\n"
                "  - MINDCORE_LLAMA_MODEL_PATH for local inference, or\n"
                "  - OPENAI_API_KEY for cloud inference"
            )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def get_provider_type(name: str) -> ProviderType:
    """
    Convert string provider name to ProviderType enum.

    Args:
        name: Provider name string ("llama_cpp", "openai", "auto")

    Returns:
        Corresponding ProviderType enum value
    """
    name_lower = name.lower().strip()
    mapping = {
        "llama_cpp": ProviderType.LLAMA_CPP,
        "llama-cpp": ProviderType.LLAMA_CPP,
        "llamacpp": ProviderType.LLAMA_CPP,
        "llama": ProviderType.LLAMA_CPP,
        "local": ProviderType.LLAMA_CPP,
        "openai": ProviderType.OPENAI,
        "gpt": ProviderType.OPENAI,
        "auto": ProviderType.AUTO,
        "default": ProviderType.AUTO,
    }
    if name_lower in mapping:
        return mapping[name_lower]
    raise ValueError(
        f"Unknown provider type: {name}. "
        f"Valid options: {list(mapping.keys())}"
    )
