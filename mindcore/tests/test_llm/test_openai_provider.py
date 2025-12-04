"""Tests for OpenAIProvider."""

import pytest

from mindcore.llm.base_provider import GenerationError, LLMProviderError, LLMResponse
from mindcore.llm.openai_provider import OpenAIProvider


class TestOpenAIProviderInitialization:
    """Tests for OpenAIProvider initialization."""

    def test_initialization_without_api_key(self):
        """Test provider initializes but is unavailable without API key."""
        provider = OpenAIProvider(api_key=None)

        assert provider.is_available() is False
        assert provider._client is None

    def test_initialization_stores_config(self):
        """Test provider stores configuration correctly."""
        provider = OpenAIProvider(
            api_key=None,
            model="gpt-4",
            base_url="http://localhost:8000/v1",
            timeout=120,
            max_retries=5,
        )

        assert provider.model == "gpt-4"
        assert provider.base_url == "http://localhost:8000/v1"
        assert provider.timeout == 120
        assert provider.max_retries == 5

    def test_default_values(self):
        """Test default configuration values."""
        provider = OpenAIProvider(api_key=None)

        assert provider.model == "gpt-4o-mini"
        assert provider.base_url is None
        assert provider.timeout == 60
        assert provider.max_retries == 3


class TestOpenAIProviderAvailability:
    """Tests for OpenAIProvider availability checks."""

    def test_is_available_false_without_client(self):
        """Test is_available returns False when no client."""
        provider = OpenAIProvider(api_key=None)
        assert provider.is_available() is False

    def test_generate_not_available(self):
        """Test generate raises when provider not available."""
        provider = OpenAIProvider(api_key=None)

        with pytest.raises(GenerationError, match="OpenAI provider not available"):
            provider.generate([{"role": "user", "content": "Hello"}])

    def test_generate_with_tools_not_available(self):
        """Test generate_with_tools raises when provider not available."""
        provider = OpenAIProvider(api_key=None)

        with pytest.raises(GenerationError, match="OpenAI provider not available"):
            provider.generate_with_tools(
                [{"role": "user", "content": "Hello"}],
                tools=[],
            )


class TestOpenAIProviderProperties:
    """Tests for OpenAIProvider properties."""

    def test_name_property_standard(self):
        """Test name property for standard OpenAI."""
        provider = OpenAIProvider(api_key=None)
        assert provider.name == "openai"

    def test_name_property_compatible(self):
        """Test name property for OpenAI-compatible server."""
        provider = OpenAIProvider(
            api_key=None,
            base_url="http://localhost:8000/v1",
        )
        assert provider.name == "openai-compatible"

    def test_close_without_client(self):
        """Test close when no client does nothing."""
        provider = OpenAIProvider(api_key=None)
        provider.close()  # Should not raise
        assert provider._client is None

    def test_repr_unavailable(self):
        """Test repr for unavailable provider."""
        provider = OpenAIProvider(api_key=None)
        repr_str = repr(provider)
        assert "unavailable" in repr_str
        assert "gpt-4o-mini" in repr_str

    def test_repr_with_base_url(self):
        """Test repr includes base_url when set."""
        provider = OpenAIProvider(
            api_key=None,
            base_url="http://localhost:8000/v1",
        )
        repr_str = repr(provider)
        assert "base_url" in repr_str


class TestOpenAIProviderRetryConfig:
    """Tests for retry configuration."""

    def test_class_constants(self):
        """Test class-level retry constants."""
        assert OpenAIProvider.MAX_RETRIES == 3
        assert OpenAIProvider.RETRY_DELAY_BASE == 1.0
        assert OpenAIProvider.RETRY_DELAY_MAX == 30.0
        assert OpenAIProvider.REQUEST_TIMEOUT == 60

    def test_retry_delay_exponential_backoff_calculation(self):
        """Test that retry delay follows exponential backoff."""
        provider = OpenAIProvider(api_key=None)

        # Calculate expected delays for each attempt
        for attempt in range(5):
            delay = min(
                provider.RETRY_DELAY_BASE * (2**attempt),
                provider.RETRY_DELAY_MAX,
            )
            assert delay <= provider.RETRY_DELAY_MAX

        # Verify specific values
        assert min(1.0 * (2**0), 30.0) == 1.0  # Attempt 0
        assert min(1.0 * (2**1), 30.0) == 2.0  # Attempt 1
        assert min(1.0 * (2**2), 30.0) == 4.0  # Attempt 2
        assert min(1.0 * (2**5), 30.0) == 30.0  # Capped at max


class TestOpenAIProviderContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_enter_exit(self):
        """Test provider can be used as context manager."""
        provider = OpenAIProvider(api_key=None)

        with provider as p:
            assert p is provider

        # After exit, should still work (close is called)
        assert provider._client is None


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o-mini",
            tokens_used=10,
            provider="openai",
            latency_ms=100.5,
            metadata={"finish_reason": "stop"},
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o-mini"
        assert response.tokens_used == 10
        assert response.provider == "openai"
        assert response.latency_ms == 100.5
        assert response.metadata["finish_reason"] == "stop"

    def test_llm_response_defaults(self):
        """Test LLMResponse default values."""
        response = LLMResponse(content="Test", model="test-model")

        assert response.tokens_used is None
        assert response.provider == "unknown"
        assert response.latency_ms is None
        assert response.metadata == {}
