"""
Tests for LLM providers module.

Tests all provider implementations with mocking to avoid actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mindcore.llm_providers import (
    LLMProvider,
    OpenAIProvider,
    OllamaProvider,
    LMStudioProvider,
    AnthropicProvider,
    OpenAICompatibleProvider,
    get_llm_provider
)


class TestLLMProviderBase:
    """Test base LLMProvider class."""

    def test_abstract_class_cannot_instantiate(self):
        """LLMProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LLMProvider(model="test")

    def test_custom_provider_implementation(self):
        """Custom provider can be created by subclassing."""

        class CustomProvider(LLMProvider):
            def chat_completion(self, messages, temperature=None, max_tokens=None, **kwargs):
                return "Custom response"

        provider = CustomProvider(model="custom", temperature=0.5, max_tokens=500)
        assert provider.model == "custom"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 500

        response = provider.chat_completion([{"role": "user", "content": "test"}])
        assert response == "Custom response"


class TestOpenAIProvider:
    """Test OpenAI provider."""

    @patch('mindcore.llm_providers.OpenAI')
    def test_initialization(self, mock_openai_class):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000
        )

        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o-mini"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 1000
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)

    @patch('mindcore.llm_providers.OpenAI')
    def test_initialization_with_base_url(self, mock_openai_class):
        """Test OpenAI provider with custom base URL."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
            base_url="https://custom-endpoint.com"
        )

        assert provider.base_url == "https://custom-endpoint.com"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom-endpoint.com"
        )

    @patch('mindcore.llm_providers.OpenAI')
    def test_chat_completion(self, mock_openai_class):
        """Test OpenAI chat completion."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]

        response = provider.chat_completion(messages, temperature=0.5, max_tokens=500)

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )

    @patch('mindcore.llm_providers.OpenAI')
    def test_chat_completion_default_params(self, mock_openai_class):
        """Test chat completion uses default parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="key", model="gpt-4o-mini", temperature=0.7, max_tokens=800)

        response = provider.chat_completion([{"role": "user", "content": "Hi"}])

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=800
        )


class TestOllamaProvider:
    """Test Ollama provider."""

    def test_initialization(self):
        """Test Ollama provider initialization."""
        provider = OllamaProvider(
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=1000
        )

        assert provider.model == "llama2"
        assert provider.base_url == "http://localhost:11434"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 1000

    def test_base_url_strip_trailing_slash(self):
        """Test base URL strips trailing slash."""
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    @patch('mindcore.llm_providers.requests.post')
    def test_chat_completion(self, mock_post):
        """Test Ollama chat completion."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Ollama response"}
        }
        mock_post.return_value = mock_response

        provider = OllamaProvider(model="llama2")

        messages = [{"role": "user", "content": "Hello!"}]
        response = provider.chat_completion(messages, temperature=0.5, max_tokens=500)

        assert response == "Ollama response"

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"

        payload = call_args[1]['json']
        assert payload['model'] == "llama2"
        assert payload['messages'] == messages
        assert payload['stream'] is False
        assert payload['options']['temperature'] == 0.5
        assert payload['options']['num_predict'] == 500

    @patch('mindcore.llm_providers.requests.post')
    def test_chat_completion_error_handling(self, mock_post):
        """Test Ollama error handling."""
        mock_post.side_effect = Exception("Connection error")

        provider = OllamaProvider(model="llama2")

        with pytest.raises(RuntimeError, match="Ollama API call failed"):
            provider.chat_completion([{"role": "user", "content": "Hi"}])


class TestLMStudioProvider:
    """Test LM Studio provider."""

    @patch('mindcore.llm_providers.OpenAI')
    def test_initialization(self, mock_openai_class):
        """Test LM Studio provider initialization."""
        provider = LMStudioProvider(
            model="local-model",
            base_url="http://localhost:1234/v1"
        )

        assert provider.model == "local-model"
        mock_openai_class.assert_called_once_with(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )

    @patch('mindcore.llm_providers.OpenAI')
    def test_chat_completion(self, mock_openai_class):
        """Test LM Studio chat completion."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "LM Studio response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = LMStudioProvider(model="local-model")

        messages = [{"role": "user", "content": "Test"}]
        response = provider.chat_completion(messages)

        assert response == "LM Studio response"


class TestAnthropicProvider:
    """Test Anthropic provider."""

    @patch('mindcore.llm_providers.Anthropic')
    def test_initialization(self, mock_anthropic_class):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-haiku-20240307",
            temperature=0.3,
            max_tokens=1000
        )

        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-haiku-20240307"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch('mindcore.llm_providers.Anthropic')
    def test_chat_completion(self, mock_anthropic_class):
        """Test Anthropic chat completion."""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key", model="claude-3-haiku-20240307")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]

        response = provider.chat_completion(messages)

        assert response == "Claude response"

        # Verify system message extracted
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args['system'] == "You are helpful."
        assert call_args['messages'] == [{"role": "user", "content": "Hello!"}]

    @patch('mindcore.llm_providers.Anthropic')
    def test_chat_completion_no_system_message(self, mock_anthropic_class):
        """Test Anthropic with no system message."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = AnthropicProvider(api_key="key", model="claude-3-haiku-20240307")

        messages = [{"role": "user", "content": "Hi"}]
        provider.chat_completion(messages)

        call_args = mock_client.messages.create.call_args[1]
        assert call_args['system'] is None


class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider."""

    @patch('mindcore.llm_providers.OpenAI')
    def test_initialization(self, mock_openai_class):
        """Test OpenAI-compatible provider initialization."""
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="custom-model",
            base_url="https://custom-api.com/v1"
        )

        assert provider.model == "custom-model"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom-api.com/v1"
        )


class TestGetLLMProvider:
    """Test get_llm_provider factory function."""

    @patch('mindcore.llm_providers.OpenAI')
    def test_get_openai_provider(self, mock_openai_class):
        """Test getting OpenAI provider."""
        provider = get_llm_provider("openai", api_key="key", model="gpt-4o-mini")

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_get_ollama_provider(self):
        """Test getting Ollama provider."""
        provider = get_llm_provider("ollama", model="llama2")

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama2"

    @patch('mindcore.llm_providers.OpenAI')
    def test_get_lmstudio_provider(self, mock_openai_class):
        """Test getting LM Studio provider."""
        provider = get_llm_provider("lmstudio", model="local-model")

        assert isinstance(provider, LMStudioProvider)
        assert provider.model == "local-model"

    @patch('mindcore.llm_providers.Anthropic')
    def test_get_anthropic_provider(self, mock_anthropic_class):
        """Test getting Anthropic provider."""
        provider = get_llm_provider("anthropic", api_key="key", model="claude-3-haiku-20240307")

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-haiku-20240307"

    def test_get_provider_invalid_name(self):
        """Test invalid provider name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("invalid_provider")

    def test_get_openai_provider_missing_api_key(self):
        """Test OpenAI without API key raises error."""
        with pytest.raises(ValueError, match="api_key required for OpenAI"):
            get_llm_provider("openai", model="gpt-4o-mini")

    def test_get_anthropic_provider_missing_api_key(self):
        """Test Anthropic without API key raises error."""
        with pytest.raises(ValueError, match="api_key required for Anthropic"):
            get_llm_provider("anthropic", model="claude-3-haiku-20240307")

    @patch('mindcore.llm_providers.OpenAI')
    def test_get_openai_compatible_provider(self, mock_openai_class):
        """Test getting OpenAI-compatible provider."""
        provider = get_llm_provider(
            "openai_compatible",
            api_key="key",
            model="custom",
            base_url="https://api.custom.com"
        )

        assert isinstance(provider, OpenAICompatibleProvider)

    def test_get_openai_compatible_missing_base_url(self):
        """Test OpenAI-compatible without base_url raises error."""
        with pytest.raises(ValueError, match="api_key and base_url required"):
            get_llm_provider("openai_compatible", api_key="key", model="custom")

    def test_default_models(self):
        """Test default models for each provider."""
        # OpenAI default
        with patch('mindcore.llm_providers.OpenAI'):
            provider = get_llm_provider("openai", api_key="key")
            assert provider.model == "gpt-4o-mini"

        # Ollama default
        provider = get_llm_provider("ollama")
        assert provider.model == "llama2"

        # LM Studio default
        with patch('mindcore.llm_providers.OpenAI'):
            provider = get_llm_provider("lmstudio")
            assert provider.model == "local-model"

        # Anthropic default
        with patch('mindcore.llm_providers.Anthropic'):
            provider = get_llm_provider("anthropic", api_key="key")
            assert provider.model == "claude-3-haiku-20240307"


class TestImportErrors:
    """Test handling of missing dependencies."""

    @patch('mindcore.llm_providers.OpenAI', side_effect=ImportError)
    def test_openai_import_error(self, mock_import):
        """Test error when OpenAI package not installed."""
        with pytest.raises(ImportError, match="OpenAI package not installed"):
            OpenAIProvider(api_key="key", model="gpt-4o-mini")

    @patch('mindcore.llm_providers.Anthropic', side_effect=ImportError)
    def test_anthropic_import_error(self, mock_import):
        """Test error when Anthropic package not installed."""
        with pytest.raises(ImportError, match="Anthropic package not installed"):
            AnthropicProvider(api_key="key", model="claude-3-haiku-20240307")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
