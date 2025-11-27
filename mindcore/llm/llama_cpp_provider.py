"""
llama.cpp LLM Provider for Mindcore.

Provides CPU-optimized local inference using llama-cpp-python bindings.
Supports GGUF models for memory-efficient operation.
"""
import os
import time
from typing import List, Dict, Optional, Any

from .base_provider import (
    BaseLLMProvider,
    LLMResponse,
    LLMProviderError,
    ModelNotFoundError,
    GenerationError,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LlamaCppProvider(BaseLLMProvider):
    """
    llama.cpp provider for CPU-optimized local inference.

    Uses llama-cpp-python bindings for efficient CPU inference with
    quantized GGUF models. Ideal for metadata enrichment and context
    retrieval tasks without requiring GPU or external API calls.

    Features:
    - Pure CPU inference (no GPU required)
    - Memory-efficient quantized models (Q4_K_M recommended)
    - No network latency or API costs
    - Offline capable
    - Configurable thread count for optimal performance

    Example:
        >>> provider = LlamaCppProvider(
        ...     model_path="~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        ...     n_ctx=4096,
        ...     n_threads=8
        ... )
        >>> response = provider.generate([
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.content)
    """

    # Default chat format that works with most instruction-tuned models
    DEFAULT_CHAT_FORMAT = "chatml"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        """
        Initialize llama.cpp provider.

        Args:
            model_path: Path to GGUF model file. Supports ~ expansion.
            n_ctx: Context window size in tokens (default: 4096)
            n_threads: Number of CPU threads. None = auto-detect (uses all cores)
            n_gpu_layers: Number of layers to offload to GPU.
                         0 = pure CPU (default), -1 = offload all layers
            chat_format: Chat template format. None = auto-detect from model,
                        or use "chatml", "llama-2", "mistral-instruct", etc.
            verbose: Enable verbose llama.cpp logging
            **kwargs: Additional arguments passed to Llama constructor

        Raises:
            ModelNotFoundError: If model file doesn't exist
            LLMProviderError: If llama-cpp-python is not installed or model fails to load
        """
        # Expand user path
        self.model_path = os.path.expanduser(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.verbose = verbose
        self._extra_kwargs = kwargs
        self._llm = None
        self._model_name = os.path.basename(self.model_path)

        # Validate model path
        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Download a model with: mindcore download-model"
            )

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the llama.cpp model.

        Raises:
            LLMProviderError: If loading fails
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise LLMProviderError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python\n"
                "For GPU support: pip install llama-cpp-python[cuda] (NVIDIA) "
                "or pip install llama-cpp-python[metal] (Apple Silicon)"
            ) from e

        try:
            logger.info(
                f"Loading llama.cpp model: {self._model_name} "
                f"(ctx={self.n_ctx}, threads={self.n_threads}, gpu_layers={self.n_gpu_layers})"
            )

            load_kwargs = {
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": self.verbose,
                **self._extra_kwargs
            }

            # Only set chat_format if explicitly provided
            # Let llama-cpp-python auto-detect from model metadata otherwise
            if self.chat_format:
                load_kwargs["chat_format"] = self.chat_format

            self._llm = Llama(**load_kwargs)

            logger.info(f"Model loaded successfully: {self._model_name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise LLMProviderError(f"Failed to load llama.cpp model: {e}") from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate a response using llama.cpp.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            json_mode: If True, constrain output to valid JSON

        Returns:
            LLMResponse with generated content

        Raises:
            GenerationError: If generation fails
        """
        if not self.is_available():
            raise GenerationError("Model not loaded")

        start_time = time.time()

        try:
            # Build generation kwargs
            gen_kwargs = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Enable JSON mode if requested
            if json_mode:
                gen_kwargs["response_format"] = {"type": "json_object"}

            # Generate response
            response = self._llm.create_chat_completion(**gen_kwargs)

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Extract token usage if available
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens")

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Generation complete: {len(content)} chars, "
                f"{tokens_used or 'N/A'} tokens, {latency_ms:.0f}ms"
            )

            return LLMResponse(
                content=content,
                model=self._model_name,
                tokens_used=tokens_used,
                provider="llama.cpp",
                latency_ms=latency_ms,
                metadata={
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Generation failed after {latency_ms:.0f}ms: {e}")
            raise GenerationError(f"llama.cpp generation failed: {e}") from e

    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._llm is not None

    @property
    def name(self) -> str:
        """Provider name."""
        return "llama.cpp"

    @property
    def model_name(self) -> str:
        """Get the loaded model name."""
        return self._model_name

    def close(self) -> None:
        """Unload the model and free resources."""
        if self._llm is not None:
            logger.info(f"Unloading model: {self._model_name}")
            # llama-cpp-python handles cleanup via __del__
            self._llm = None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        return {
            "model_path": self.model_path,
            "model_name": self._model_name,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "is_loaded": self.is_available(),
        }

    def __repr__(self) -> str:
        status = "loaded" if self.is_available() else "not loaded"
        return f"LlamaCppProvider(model={self._model_name}, {status})"
