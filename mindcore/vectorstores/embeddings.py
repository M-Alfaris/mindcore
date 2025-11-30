"""
Embedding function implementations.

Provides adapters for various embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002)
- Sentence Transformers (local, no API required)
- Ollama (local, with many model options)
- Custom (bring your own embedding function)
"""
from typing import List, Optional, Callable, Any
import os

from .base import EmbeddingFunction
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddings(EmbeddingFunction):
    """
    OpenAI embeddings using the official API.

    Models:
    - text-embedding-3-small: 1536 dimensions, best price/performance
    - text-embedding-3-large: 3072 dimensions, highest quality
    - text-embedding-ada-002: 1536 dimensions, legacy

    Example:
        >>> embeddings = OpenAIEmbeddings(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
        >>> vector = embeddings.embed_query("Hello world")
        >>> len(vector)
        1536
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name
            base_url: Custom API base URL (for Azure or compatible APIs)
            dimensions: Override output dimensions (for models that support it)
            batch_size: Maximum texts per API call
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Run: pip install openai"
            )

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        self._model = model
        self._batch_size = batch_size

        # Determine dimensions
        if dimensions:
            self._dimensions = dimensions
        elif model in self.MODEL_DIMENSIONS:
            self._dimensions = self.MODEL_DIMENSIONS[model]
        else:
            self._dimensions = 1536  # Default

        # Initialize client
        kwargs = {"api_key": self._api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        logger.info(f"OpenAI embeddings initialized with model {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]

            # Clean texts (OpenAI doesn't like empty strings)
            batch = [t.replace("\n", " ").strip() or " " for t in batch]

            response = self._client.embeddings.create(
                input=batch,
                model=self._model
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions


class SentenceTransformerEmbeddings(EmbeddingFunction):
    """
    Local embeddings using sentence-transformers.

    No API required - runs entirely on your machine.

    Popular models:
    - all-MiniLM-L6-v2: 384 dim, fast, good quality
    - all-mpnet-base-v2: 768 dim, best quality
    - paraphrase-multilingual-MiniLM-L12-v2: 384 dim, multilingual

    Example:
        >>> embeddings = SentenceTransformerEmbeddings(
        ...     model_name="all-MiniLM-L6-v2"
        ... )
        >>> vector = embeddings.embed_query("Hello world")
        >>> len(vector)
        384
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize sentence-transformers embeddings.

        Args:
            model_name: Model name from HuggingFace or local path
            device: Device to run on ("cpu", "cuda", "mps")
            normalize_embeddings: Normalize vectors to unit length
            batch_size: Batch size for encoding
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._batch_size = batch_size

        # Load model
        self._model = SentenceTransformer(model_name, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

        logger.info(
            f"SentenceTransformer embeddings initialized: {model_name} "
            f"({self._dimensions} dimensions)"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions


class OllamaEmbeddings(EmbeddingFunction):
    """
    Local embeddings using Ollama.

    Requires Ollama to be installed and running locally.
    Supports any embedding model available in Ollama.

    Popular models:
    - nomic-embed-text: 768 dim, high quality
    - mxbai-embed-large: 1024 dim
    - all-minilm: 384 dim, fast

    Example:
        >>> embeddings = OllamaEmbeddings(
        ...     model="nomic-embed-text",
        ...     base_url="http://localhost:11434"
        ... )
        >>> vector = embeddings.embed_query("Hello world")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: Optional[int] = None
    ):
        """
        Initialize Ollama embeddings.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            dimensions: Expected embedding dimensions (auto-detected if not provided)
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx not installed. Run: pip install httpx"
            )

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

        # Auto-detect dimensions if not provided
        if dimensions:
            self._dimensions = dimensions
        else:
            # Get dimension by embedding a test string
            test_embedding = self._embed_single("test")
            self._dimensions = len(test_embedding)

        logger.info(
            f"Ollama embeddings initialized: {model} ({self._dimensions} dimensions)"
        )

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using Ollama API."""
        response = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []

        # Ollama doesn't have batch API, so we embed one at a time
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_single(text)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions


class CustomEmbeddings(EmbeddingFunction):
    """
    Custom embedding function wrapper.

    Wrap any embedding function with this adapter.

    Example:
        >>> def my_embed_fn(texts):
        ...     # Your custom logic
        ...     return [[0.1] * 384 for _ in texts]
        >>>
        >>> embeddings = CustomEmbeddings(
        ...     embed_fn=my_embed_fn,
        ...     dimension=384
        ... )
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]],
        dimension: int,
        query_embed_fn: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize custom embeddings.

        Args:
            embed_fn: Function that takes list of texts and returns list of embeddings
            dimension: Embedding dimension
            query_embed_fn: Optional separate function for query embedding
        """
        self._embed_fn = embed_fn
        self._query_embed_fn = query_embed_fn
        self._dimensions = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []
        return self._embed_fn(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if self._query_embed_fn:
            return self._query_embed_fn(text)
        return self._embed_fn([text])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions


# Factory function for creating embeddings
def create_embeddings(
    provider: str = "openai",
    **kwargs: Any
) -> EmbeddingFunction:
    """
    Factory function to create embedding functions.

    Args:
        provider: Provider name ("openai", "sentence_transformers", "ollama", "custom")
        **kwargs: Provider-specific arguments

    Returns:
        EmbeddingFunction instance

    Example:
        >>> # OpenAI
        >>> emb = create_embeddings("openai", model="text-embedding-3-small")
        >>>
        >>> # Local (no API)
        >>> emb = create_embeddings("sentence_transformers", model_name="all-MiniLM-L6-v2")
        >>>
        >>> # Ollama
        >>> emb = create_embeddings("ollama", model="nomic-embed-text")
    """
    providers = {
        "openai": OpenAIEmbeddings,
        "sentence_transformers": SentenceTransformerEmbeddings,
        "sentence-transformers": SentenceTransformerEmbeddings,
        "ollama": OllamaEmbeddings,
        "custom": CustomEmbeddings,
    }

    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Available: {list(providers.keys())}"
        )

    return providers[provider_lower](**kwargs)
