"""
Base classes for vector store integrations.

Inspired by LangChain's modular architecture, this provides a unified
interface for vector stores that can be used interchangeably.

All vector stores must inherit from VectorStore base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Sequence, TypeVar, Generic
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """
    A document with content and metadata.

    This is the standard document format used across all vector stores.
    Compatible with LangChain Document format for interoperability.
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"page_content": self.page_content, "metadata": self.metadata, "id": self.id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            page_content=data["page_content"], metadata=data.get("metadata", {}), id=data.get("id")
        )


@dataclass
class SearchResult:
    """
    Result from a similarity search.

    Contains the document and its similarity score.
    """

    document: Document
    score: float  # Higher = more similar (normalized 0-1 when possible)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"document": self.document.to_dict(), "score": self.score}


class DistanceMetric(str, Enum):
    """Supported distance/similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    INNER_PRODUCT = "inner_product"


class EmbeddingFunction(ABC):
    """
    Abstract base class for embedding functions.

    Embed documents and queries into vector representations.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        pass


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    All vector stores must implement this interface to be used
    interchangeably in the Mindcore context layer.

    Inspired by LangChain's VectorStore interface for compatibility.

    Example:
        >>> class MyVectorStore(VectorStore):
        ...     def add_documents(self, documents, **kwargs):
        ...         # Implementation
        ...         pass
        ...
        ...     def similarity_search(self, query, k=4, **kwargs):
        ...         # Implementation
        ...         pass
    """

    # Store name for identification
    name: str = "base_vector_store"

    @abstractmethod
    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            ids: Optional list of IDs for the documents
            **kwargs: Additional arguments for specific implementations

        Returns:
            List of IDs for the added documents
        """
        pass

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs
            **kwargs: Additional arguments

        Returns:
            List of IDs for the added texts
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of similar documents
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[SearchResult]:
        """
        Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of SearchResult with documents and scores
        """
        pass

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search by embedding vector directly.

        Override in subclass for optimized implementation.

        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of similar documents
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support similarity_search_by_vector"
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search.

        Optimizes for both relevance and diversity.

        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of candidates to fetch before reranking
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of documents optimized for relevance and diversity
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support max_marginal_relevance_search"
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete documents from the vector store.

        Args:
            ids: Optional list of IDs to delete
            filter: Optional metadata filter for deletion
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support delete")

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (may be fewer if some IDs not found)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support get_by_ids")

    @classmethod
    def from_documents(
        cls, documents: List[Document], embedding: EmbeddingFunction, **kwargs: Any
    ) -> "VectorStore":
        """
        Create a vector store from documents.

        Args:
            documents: List of documents to add
            embedding: Embedding function to use
            **kwargs: Additional arguments for initialization

        Returns:
            Initialized vector store with documents
        """
        raise NotImplementedError(f"{cls.__name__} does not support from_documents")

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: EmbeddingFunction,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        Create a vector store from texts.

        Args:
            texts: List of texts to add
            embedding: Embedding function to use
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments for initialization

        Returns:
            Initialized vector store with texts
        """
        raise NotImplementedError(f"{cls.__name__} does not support from_texts")

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None
    ) -> "VectorStoreRetriever":
        """
        Create a retriever from this vector store.

        Args:
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters

        Returns:
            VectorStoreRetriever instance
        """
        return VectorStoreRetriever(
            vectorstore=self, search_type=search_type, search_kwargs=search_kwargs or {}
        )

    async def aadd_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        """Async version of add_documents."""
        return self.add_documents(documents, ids, **kwargs)

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add_texts."""
        return self.add_texts(texts, metadatas, ids, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        """Async version of similarity_search."""
        return self.similarity_search(query, k, filter, **kwargs)

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[SearchResult]:
        """Async version of similarity_search_with_score."""
        return self.similarity_search_with_score(query, k, filter, **kwargs)

    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"


class VectorStoreRetriever:
    """
    Retriever that wraps a vector store.

    Provides a simple interface for retrieving relevant documents.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        search_type: str = "similarity",
        search_kwargs: Dict[str, Any] = None,
    ):
        """
        Initialize retriever.

        Args:
            vectorstore: The vector store to retrieve from
            search_type: Type of search to perform
            search_kwargs: Additional search parameters
        """
        self.vectorstore = vectorstore
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.

        Args:
            query: The query string

        Returns:
            List of relevant documents
        """
        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            results = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
            threshold = self.search_kwargs.get("score_threshold", 0.5)
            return [r.document for r in results if r.score >= threshold]
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        if self.search_type == "similarity":
            return await self.vectorstore.asimilarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            results = await self.vectorstore.asimilarity_search_with_score(
                query, **self.search_kwargs
            )
            threshold = self.search_kwargs.get("score_threshold", 0.5)
            return [r.document for r in results if r.score >= threshold]
        else:
            # Fall back to sync for unsupported async operations
            return self.get_relevant_documents(query)


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing and development.

    Uses numpy for vector operations if available, falls back to pure Python.
    """

    name = "in_memory"

    def __init__(
        self, embedding: EmbeddingFunction, metric: DistanceMetric = DistanceMetric.COSINE
    ):
        """
        Initialize in-memory vector store.

        Args:
            embedding: Embedding function to use
            metric: Distance metric for similarity
        """
        self.embedding = embedding
        self.metric = metric
        self._texts: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        doc_ids = ids or [doc.id for doc in documents]
        return self.add_texts(texts, metadatas, doc_ids, **kwargs)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the store."""
        import uuid

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Handle metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Store
        for i, (text, embedding, metadata, doc_id) in enumerate(
            zip(texts, embeddings, metadatas, ids)
        ):
            self._texts.append(text)
            self._embeddings.append(embedding)
            self._metadatas.append(metadata)
            self._ids.append(doc_id)

        logger.debug(f"Added {len(texts)} texts to in-memory store")
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents."""
        results = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [r.document for r in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[SearchResult]:
        """Search with scores."""
        if not self._embeddings:
            return []

        # Embed query
        query_embedding = self.embedding.embed_query(query)

        # Calculate similarities
        scores = []
        for i, embedding in enumerate(self._embeddings):
            # Apply filter if provided
            if filter:
                metadata = self._metadatas[i]
                if not self._matches_filter(metadata, filter):
                    continue

            score = self._calculate_similarity(query_embedding, embedding)
            scores.append((i, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for idx, score in scores[:k]:
            doc = Document(
                page_content=self._texts[idx], metadata=self._metadatas[idx], id=self._ids[idx]
            )
            results.append(SearchResult(document=doc, score=score))

        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """Delete documents."""
        if ids:
            indices_to_remove = [i for i, doc_id in enumerate(self._ids) if doc_id in ids]
        elif filter:
            indices_to_remove = [
                i for i, meta in enumerate(self._metadatas) if self._matches_filter(meta, filter)
            ]
        else:
            return False

        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self._texts[idx]
            del self._embeddings[idx]
            del self._metadatas[idx]
            del self._ids[idx]

        return True

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        results = []
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                results.append(
                    Document(
                        page_content=self._texts[idx], metadata=self._metadatas[idx], id=doc_id
                    )
                )
        return results

    @classmethod
    def from_documents(
        cls, documents: List[Document], embedding: EmbeddingFunction, **kwargs: Any
    ) -> "InMemoryVectorStore":
        """Create from documents."""
        store = cls(embedding=embedding, **kwargs)
        store.add_documents(documents)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: EmbeddingFunction,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "InMemoryVectorStore":
        """Create from texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate similarity between two vectors."""
        try:
            import numpy as np

            v1 = np.array(vec1)
            v2 = np.array(vec2)

            if self.metric == DistanceMetric.COSINE:
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            elif self.metric == DistanceMetric.DOT_PRODUCT:
                return float(np.dot(v1, v2))
            elif self.metric == DistanceMetric.EUCLIDEAN:
                # Convert distance to similarity (higher = more similar)
                return float(1 / (1 + np.linalg.norm(v1 - v2)))
            else:
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        except ImportError:
            # Pure Python fallback
            import math

            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))

            if self.metric == DistanceMetric.COSINE:
                return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
            elif self.metric == DistanceMetric.DOT_PRODUCT:
                return dot
            else:
                return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # Handle operators like {"$eq": value}, {"$in": [values]}
                for op, op_value in value.items():
                    if op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
                    elif op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$nin" and metadata[key] in op_value:
                        return False
                    elif op == "$gt" and not metadata[key] > op_value:
                        return False
                    elif op == "$gte" and not metadata[key] >= op_value:
                        return False
                    elif op == "$lt" and not metadata[key] < op_value:
                        return False
                    elif op == "$lte" and not metadata[key] <= op_value:
                        return False
            elif metadata[key] != value:
                return False
        return True
