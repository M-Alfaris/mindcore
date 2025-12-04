"""Tests for VectorStore base classes."""

import pytest

from mindcore.vectorstores.base import (
    DistanceMetric,
    Document,
    EmbeddingFunction,
    SearchResult,
    VectorStore,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            page_content="Hello, world!",
            metadata={"author": "test"},
            id="doc_001",
        )

        assert doc.page_content == "Hello, world!"
        assert doc.metadata == {"author": "test"}
        assert doc.id == "doc_001"

    def test_document_default_values(self):
        """Test document with default values."""
        doc = Document(page_content="Content only")

        assert doc.page_content == "Content only"
        assert doc.metadata == {}
        assert doc.id is None

    def test_document_to_dict(self):
        """Test converting document to dictionary."""
        doc = Document(
            page_content="Test content",
            metadata={"key": "value"},
            id="doc_123",
        )

        result = doc.to_dict()

        assert result == {
            "page_content": "Test content",
            "metadata": {"key": "value"},
            "id": "doc_123",
        }

    def test_document_from_dict(self):
        """Test creating document from dictionary."""
        data = {
            "page_content": "From dict",
            "metadata": {"source": "test"},
            "id": "doc_456",
        }

        doc = Document.from_dict(data)

        assert doc.page_content == "From dict"
        assert doc.metadata == {"source": "test"}
        assert doc.id == "doc_456"

    def test_document_from_dict_minimal(self):
        """Test creating document from minimal dictionary."""
        data = {"page_content": "Minimal"}

        doc = Document.from_dict(data)

        assert doc.page_content == "Minimal"
        assert doc.metadata == {}
        assert doc.id is None


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        doc = Document(page_content="Result content")
        result = SearchResult(document=doc, score=0.95)

        assert result.document == doc
        assert result.score == 0.95

    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        doc = Document(
            page_content="Test",
            metadata={"key": "value"},
            id="doc_1",
        )
        result = SearchResult(document=doc, score=0.85)

        result_dict = result.to_dict()

        assert result_dict["score"] == 0.85
        assert result_dict["document"]["page_content"] == "Test"
        assert result_dict["document"]["metadata"] == {"key": "value"}


class TestDistanceMetric:
    """Tests for DistanceMetric enum."""

    def test_distance_metric_values(self):
        """Test distance metric enum values."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"
        assert DistanceMetric.INNER_PRODUCT.value == "inner_product"

    def test_distance_metric_string_behavior(self):
        """Test that DistanceMetric value is a string."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"


class TestEmbeddingFunction:
    """Tests for EmbeddingFunction abstract base class."""

    def test_embedding_function_is_abstract(self):
        """Test that EmbeddingFunction cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingFunction()

    def test_embedding_function_subclass(self):
        """Test that EmbeddingFunction can be subclassed."""

        class TestEmbeddings(EmbeddingFunction):
            def embed_documents(self, texts):
                return [[0.1, 0.2] for _ in texts]

            def embed_query(self, text):
                return [0.1, 0.2]

            @property
            def dimension(self):
                return 2

        embeddings = TestEmbeddings()
        assert embeddings.dimension == 2
        assert embeddings.embed_query("test") == [0.1, 0.2]
        assert embeddings.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


class TestVectorStore:
    """Tests for VectorStore abstract base class."""

    def test_vector_store_is_abstract(self):
        """Test that VectorStore cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            VectorStore()

    def test_vector_store_default_name(self):
        """Test default name attribute."""
        assert VectorStore.name == "base_vector_store"

    def test_vector_store_subclass(self):
        """Test that VectorStore can be subclassed."""

        class TestVectorStore(VectorStore):
            name = "test_store"

            def add_documents(self, documents, ids=None, **kwargs):
                return [doc.id or f"gen_{i}" for i, doc in enumerate(documents)]

            def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
                return [ids[i] if ids else f"text_{i}" for i in range(len(texts))]

            def similarity_search(self, query, k=4, filter=None, **kwargs):
                return []

            def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
                return []

        store = TestVectorStore()
        assert store.name == "test_store"

        # Test add_documents
        docs = [Document(page_content="Test", id="doc_1")]
        ids = store.add_documents(docs)
        assert ids == ["doc_1"]

        # Test add_texts
        text_ids = store.add_texts(["text1", "text2"])
        assert text_ids == ["text_0", "text_1"]

    def test_similarity_search_by_vector_default(self):
        """Test default similarity_search_by_vector raises NotImplementedError."""

        class MinimalVectorStore(VectorStore):
            def add_documents(self, documents, ids=None, **kwargs):
                return []

            def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
                return []

            def similarity_search(self, query, k=4, filter=None, **kwargs):
                return []

            def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
                return []

        store = MinimalVectorStore()

        # Default implementation should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            store.similarity_search_by_vector([0.1, 0.2, 0.3])


class TestVectorStoreIntegration:
    """Integration tests for VectorStore implementations."""

    @pytest.fixture
    def mock_embedding_function(self):
        """Create a mock embedding function."""

        class MockEmbeddings(EmbeddingFunction):
            def embed_documents(self, texts):
                # Return deterministic embeddings based on text length
                return [[float(len(t)) / 100.0] * 3 for t in texts]

            def embed_query(self, text):
                return [float(len(text)) / 100.0] * 3

            @property
            def dimension(self):
                return 3

        return MockEmbeddings()

    @pytest.fixture
    def in_memory_store(self, mock_embedding_function):
        """Create an in-memory vector store for testing."""

        class InMemoryVectorStore(VectorStore):
            name = "in_memory"

            def __init__(self, embedding_fn):
                self.embedding_fn = embedding_fn
                self.documents: list[Document] = []
                self.embeddings: list[list[float]] = []

            def add_documents(self, documents, ids=None, **kwargs):
                result_ids = []
                for i, doc in enumerate(documents):
                    doc_id = doc.id or (ids[i] if ids and i < len(ids) else f"doc_{len(self.documents)}")
                    doc.id = doc_id
                    self.documents.append(doc)
                    result_ids.append(doc_id)

                # Generate embeddings
                texts = [doc.page_content for doc in documents]
                new_embeddings = self.embedding_fn.embed_documents(texts)
                self.embeddings.extend(new_embeddings)

                return result_ids

            def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
                documents = []
                for i, text in enumerate(texts):
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    doc_id = ids[i] if ids and i < len(ids) else None
                    documents.append(Document(page_content=text, metadata=metadata, id=doc_id))
                return self.add_documents(documents)

            def similarity_search(self, query, k=4, filter=None, **kwargs):
                results = self.similarity_search_with_score(query, k, filter, **kwargs)
                return [r.document for r in results]

            def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
                if not self.documents:
                    return []

                query_embedding = self.embedding_fn.embed_query(query)

                # Simple cosine similarity
                results = []
                for i, (doc, emb) in enumerate(zip(self.documents, self.embeddings)):
                    # Apply filter if provided
                    if filter:
                        if not all(doc.metadata.get(key) == value for key, value in filter.items()):
                            continue

                    # Calculate dot product as simple similarity
                    score = sum(q * e for q, e in zip(query_embedding, emb))
                    results.append(SearchResult(document=doc, score=score))

                # Sort by score descending and return top k
                results.sort(key=lambda r: r.score, reverse=True)
                return results[:k]

            def similarity_search_by_vector(self, embedding, k=4, filter=None, **kwargs):
                if not self.documents:
                    return []

                results = []
                for doc, emb in zip(self.documents, self.embeddings):
                    if filter and not all(doc.metadata.get(key) == value for key, value in filter.items()):
                        continue
                    score = sum(q * e for q, e in zip(embedding, emb))
                    results.append((doc, score))

                results.sort(key=lambda r: r[1], reverse=True)
                return [doc for doc, _ in results[:k]]

        return InMemoryVectorStore(mock_embedding_function)

    def test_add_and_search(self, in_memory_store):
        """Test adding documents and searching."""
        docs = [
            Document(page_content="Short text", metadata={"type": "short"}),
            Document(page_content="This is a longer piece of text", metadata={"type": "long"}),
            Document(page_content="Medium length text here", metadata={"type": "medium"}),
        ]

        ids = in_memory_store.add_documents(docs)
        assert len(ids) == 3

        results = in_memory_store.similarity_search("longer text query", k=2)
        assert len(results) == 2

    def test_search_with_filter(self, in_memory_store):
        """Test searching with metadata filter."""
        docs = [
            Document(page_content="Document A", metadata={"category": "tech"}),
            Document(page_content="Document B", metadata={"category": "science"}),
            Document(page_content="Document C", metadata={"category": "tech"}),
        ]

        in_memory_store.add_documents(docs)

        results = in_memory_store.similarity_search(
            "query",
            k=10,
            filter={"category": "tech"},
        )

        assert len(results) == 2
        for doc in results:
            assert doc.metadata["category"] == "tech"

    def test_search_with_scores(self, in_memory_store):
        """Test searching with scores."""
        docs = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]

        in_memory_store.add_documents(docs)

        results = in_memory_store.similarity_search_with_score("Hello", k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_by_vector(self, in_memory_store, mock_embedding_function):
        """Test searching by vector directly."""
        docs = [
            Document(page_content="Test document one"),
            Document(page_content="Test document two"),
        ]

        in_memory_store.add_documents(docs)

        embedding = mock_embedding_function.embed_query("test query")
        results = in_memory_store.similarity_search_by_vector(embedding, k=2)

        assert len(results) == 2

    def test_add_texts(self, in_memory_store):
        """Test adding texts directly."""
        texts = ["Text one", "Text two", "Text three"]
        metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

        ids = in_memory_store.add_texts(texts, metadatas=metadatas)

        assert len(ids) == 3

        results = in_memory_store.similarity_search("query", k=10)
        assert len(results) == 3
