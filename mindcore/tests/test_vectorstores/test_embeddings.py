"""Tests for Embeddings implementations."""

import pytest

from mindcore.vectorstores.embeddings import (
    CustomEmbeddings,
    create_embeddings,
)


class TestCustomEmbeddings:
    """Tests for CustomEmbeddings class."""

    def test_initialization(self):
        """Test initialization with custom function."""

        def embed_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=2)

        assert embeddings.dimension == 2
        assert embeddings._embed_fn == embed_fn

    def test_embed_documents(self):
        """Test embedding documents with custom function."""

        def embed_fn(texts):
            return [[float(len(t))] * 3 for t in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=3)
        result = embeddings.embed_documents(["hi", "hello"])

        assert result == [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]]

    def test_embed_documents_empty(self):
        """Test embedding empty list."""

        def embed_fn(texts):
            return [[0.1] for _ in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=1)
        result = embeddings.embed_documents([])
        assert result == []

    def test_embed_query_uses_embed_fn(self):
        """Test embed_query uses embed_fn when no query_embed_fn."""

        def embed_fn(texts):
            return [[0.5, 0.5] for _ in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=2)
        result = embeddings.embed_query("test")

        assert result == [0.5, 0.5]

    def test_embed_query_uses_query_embed_fn(self):
        """Test embed_query uses query_embed_fn when provided."""

        def embed_fn(texts):
            return [[0.1, 0.1] for _ in texts]

        def query_embed_fn(text):
            return [0.9, 0.9]

        embeddings = CustomEmbeddings(
            embed_fn=embed_fn,
            dimension=2,
            query_embed_fn=query_embed_fn,
        )
        result = embeddings.embed_query("test")

        assert result == [0.9, 0.9]

    def test_dimension_property(self):
        """Test dimension property."""

        def embed_fn(texts):
            return [[0.1] * 384 for _ in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=384)
        assert embeddings.dimension == 384


class TestCreateEmbeddings:
    """Tests for create_embeddings factory function."""

    def test_create_custom(self):
        """Test creating custom embeddings."""

        def embed_fn(texts):
            return [[0.1] for _ in texts]

        embeddings = create_embeddings("custom", embed_fn=embed_fn, dimension=1)
        assert isinstance(embeddings, CustomEmbeddings)

    def test_create_unknown_provider(self):
        """Test error on unknown provider."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embeddings("unknown_provider")

    def test_available_providers_in_error(self):
        """Test error message lists available providers."""
        with pytest.raises(ValueError) as exc_info:
            create_embeddings("invalid")

        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "sentence_transformers" in error_msg or "sentence-transformers" in error_msg
        assert "ollama" in error_msg
        assert "custom" in error_msg


class TestEmbeddingFunctionInterface:
    """Tests for EmbeddingFunction abstract interface."""

    def test_embedding_function_abstract(self):
        """Test EmbeddingFunction cannot be instantiated directly."""
        from mindcore.vectorstores.base import EmbeddingFunction

        with pytest.raises(TypeError, match="abstract"):
            EmbeddingFunction()

    def test_custom_embeddings_implements_interface(self):
        """Test CustomEmbeddings properly implements EmbeddingFunction."""
        from mindcore.vectorstores.base import EmbeddingFunction

        def embed_fn(texts):
            return [[0.1] for _ in texts]

        embeddings = CustomEmbeddings(embed_fn=embed_fn, dimension=1)

        assert isinstance(embeddings, EmbeddingFunction)
        assert hasattr(embeddings, "embed_documents")
        assert hasattr(embeddings, "embed_query")
        assert hasattr(embeddings, "dimension")
