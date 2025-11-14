"""
LLM Provider abstraction for Mindcore.

Supports multiple LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Ollama (local models: llama2, mistral, etc.)
- LM Studio (local models)
- Anthropic (Claude)
- Any OpenAI-compatible API

Usage:
    from mindcore.llm_providers import OpenAIProvider, OllamaProvider

    # OpenAI
    provider = OpenAIProvider(api_key="key", model="gpt-4o-mini")

    # Ollama (local)
    provider = OllamaProvider(model="llama2", base_url="http://localhost:11434")

    # Use provider
    response = provider.chat_completion(messages=[...])
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Custom providers should inherit from this class.
    """

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1000):
        """
        Initialize LLM provider.

        Args:
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Get chat completion from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Provider-specific kwargs

        Returns:
            Response text

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = provider.chat_completion(messages)
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider (GPT-4o, GPT-4o-mini, etc.).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o, gpt-4o-mini, etc.)
            temperature: Temperature
            max_tokens: Max tokens
            base_url: Optional custom base URL
        """
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Get completion from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM models.

    Supports: llama2, mistral, codellama, etc.
    Requires Ollama running locally: https://ollama.ai
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (llama2, mistral, etc.)
            base_url: Ollama API base URL
            temperature: Temperature
            max_tokens: Max tokens

        Example:
            >>> provider = OllamaProvider(model="llama2")
            >>> response = provider.chat_completion([...])
        """
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url.rstrip('/')

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Get completion from Ollama."""
        import requests

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")


class LMStudioProvider(LLMProvider):
    """
    LM Studio provider for local models.

    LM Studio provides OpenAI-compatible API.
    Default URL: http://localhost:1234/v1
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize LM Studio provider.

        Args:
            model: Model name (as shown in LM Studio)
            base_url: LM Studio API base URL
            temperature: Temperature
            max_tokens: Max tokens
        """
        super().__init__(model, temperature, max_tokens)

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="lm-studio",  # LM Studio doesn't require real API key
                base_url=base_url
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Get completion from LM Studio."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider (Claude models).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Claude model name
            temperature: Temperature
            max_tokens: Max tokens
        """
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Get completion from Anthropic."""
        # Anthropic API format is slightly different
        # Extract system message if present
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            system=system_msg,
            messages=chat_messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )

        return response.content[0].text


class OpenAICompatibleProvider(LLMProvider):
    """
    Generic OpenAI-compatible API provider.

    Works with any API that follows OpenAI's chat completion format.
    Examples: Together AI, Anyscale, etc.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI-compatible provider.

        Args:
            api_key: API key for the service
            model: Model name
            base_url: Base URL for the API
            temperature: Temperature
            max_tokens: Max tokens
        """
        super().__init__(model, temperature, max_tokens)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Get completion from OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_llm_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Get LLM provider by name.

    Args:
        provider_name: Provider name
            - "openai": OpenAI (GPT-4o, GPT-4o-mini)
            - "ollama": Ollama (local models)
            - "lmstudio": LM Studio (local models)
            - "anthropic": Anthropic (Claude)
        api_key: API key (not required for local providers)
        model: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance

    Example:
        >>> # OpenAI
        >>> provider = get_llm_provider("openai", api_key="key", model="gpt-4o-mini")
        >>>
        >>> # Ollama (local)
        >>> provider = get_llm_provider("ollama", model="llama2")
        >>>
        >>> # LM Studio (local)
        >>> provider = get_llm_provider("lmstudio", model="local-model")
    """
    providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
        "anthropic": AnthropicProvider,
        "openai_compatible": OpenAICompatibleProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Choose from: {list(providers.keys())}"
        )

    provider_class = providers[provider_name]

    # Build kwargs based on provider
    if provider_name == "openai":
        if not api_key:
            raise ValueError("api_key required for OpenAI")
        return provider_class(api_key=api_key, model=model or "gpt-4o-mini", **kwargs)

    elif provider_name == "ollama":
        return provider_class(model=model or "llama2", **kwargs)

    elif provider_name == "lmstudio":
        return provider_class(model=model or "local-model", **kwargs)

    elif provider_name == "anthropic":
        if not api_key:
            raise ValueError("api_key required for Anthropic")
        return provider_class(api_key=api_key, model=model or "claude-3-haiku-20240307", **kwargs)

    elif provider_name == "openai_compatible":
        if not api_key or not kwargs.get("base_url"):
            raise ValueError("api_key and base_url required for OpenAI-compatible provider")
        return provider_class(api_key=api_key, model=model, **kwargs)


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "get_llm_provider",
]
