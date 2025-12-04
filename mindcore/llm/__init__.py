"""Mindcore LLM Provider Layer.

Provides a unified abstraction for different LLM backends:
- llama.cpp: CPU-optimized local inference (primary)
- OpenAI: Cloud API fallback

Quick Start:
-----------
    from mindcore.llm import create_provider, ProviderType

    # Auto mode: llama.cpp primary, OpenAI fallback
    provider = create_provider(
        provider_type=ProviderType.AUTO,
        llama_config={"model_path": "~/.mindcore/models/model.gguf"},
        openai_config={"api_key": "sk-..."}
    )

    # Generate response
    response = provider.generate([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ])
    print(response.content)

Provider Types:
--------------
- ProviderType.AUTO: llama.cpp with OpenAI fallback (recommended)
- ProviderType.LLAMA_CPP: Local llama.cpp only
- ProviderType.OPENAI: OpenAI API only
"""

from .base_provider import (
    BaseLLMProvider,
    GenerationError,
    LLMProviderError,
    LLMResponse,
    ModelNotFoundError,
)
from .llama_cpp_provider import LlamaCppProvider
from .openai_provider import OpenAIProvider
from .provider_factory import (
    FallbackProvider,
    ProviderType,
    create_provider,
    get_provider_type,
)


__all__ = [
    # Base classes
    "BaseLLMProvider",
    "FallbackProvider",
    "GenerationError",
    # Exceptions
    "LLMProviderError",
    "LLMResponse",
    # Providers
    "LlamaCppProvider",
    "ModelNotFoundError",
    "OpenAIProvider",
    # Factory
    "ProviderType",
    "create_provider",
    "get_provider_type",
]
