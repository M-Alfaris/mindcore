"""Tests for LlamaCppProvider."""

import pytest

from mindcore.llm.base_provider import (
    GenerationError,
    LLMProviderError,
    LLMResponse,
    ModelNotFoundError,
)
from mindcore.llm.llama_cpp_provider import LlamaCppProvider


class TestLlamaCppProviderInitialization:
    """Tests for LlamaCppProvider initialization."""

    def test_model_not_found_error(self):
        """Test provider raises error when model file doesn't exist."""
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            LlamaCppProvider(model_path="/nonexistent/model.gguf")

    def test_model_path_expansion(self):
        """Test that user path (~) is expanded."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            LlamaCppProvider(model_path="~/models/test.gguf")

        # Should have expanded the path (not contain ~)
        assert "~" not in str(exc_info.value) or "/Users/" in str(exc_info.value)


class TestLlamaCppProviderDefaults:
    """Tests for default values and constants."""

    def test_default_chat_format(self):
        """Test default chat format constant."""
        assert LlamaCppProvider.DEFAULT_CHAT_FORMAT == "chatml"


class TestLlamaCppProviderExceptions:
    """Tests for exception classes."""

    def test_model_not_found_error_inheritance(self):
        """Test ModelNotFoundError inherits from LLMProviderError."""
        assert issubclass(ModelNotFoundError, LLMProviderError)

    def test_generation_error_inheritance(self):
        """Test GenerationError inherits from LLMProviderError."""
        assert issubclass(GenerationError, LLMProviderError)

    def test_llm_provider_error_inheritance(self):
        """Test LLMProviderError inherits from Exception."""
        assert issubclass(LLMProviderError, Exception)

    def test_model_not_found_error_message(self):
        """Test ModelNotFoundError includes helpful message."""
        try:
            LlamaCppProvider(model_path="/nonexistent/model.gguf")
        except ModelNotFoundError as e:
            assert "Model not found" in str(e)
            assert "nonexistent" in str(e)


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_base_provider_context_manager(self):
        """Test context manager protocol is defined."""
        from mindcore.llm.base_provider import BaseLLMProvider

        # Check the abstract methods exist
        assert hasattr(BaseLLMProvider, "generate")
        assert hasattr(BaseLLMProvider, "is_available")
        assert hasattr(BaseLLMProvider, "name")
        assert hasattr(BaseLLMProvider, "close")
        assert hasattr(BaseLLMProvider, "__enter__")
        assert hasattr(BaseLLMProvider, "__exit__")
