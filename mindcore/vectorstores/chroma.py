"""Chroma vector store integration.

Chroma is an open-source embedding database with:
- Local persistence
- HTTP client support
- Built-in embedding functions
- Metadata filtering

Install: pip install chromadb
"""

import uuid
from typing import TYPE_CHECKING, Any

from mindcore.utils.logger import get_logger

from .base import DistanceMetric, Document, EmbeddingFunction, SearchResult, VectorStore


logger = get_logger(__name__)

if TYPE_CHECKING:
    import chromadb


class ChromaVectorStore(VectorStore):
    """Chroma vector store integration.

    Supports both local persistence and client/server mode.

    Example (local):
        >>> from mindcore.vectorstores import ChromaVectorStore
        >>> store = ChromaVectorStore(
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_db",
        ...     embedding=my_embedding_fn
        ... )
        >>> store.add_texts(["Hello world", "Goodbye world"])
        >>> results = store.similarity_search("Hello", k=1)

    Example (client mode):
        >>> store = ChromaVectorStore(
        ...     collection_name="my_docs",
        ...     host="localhost",
        ...     port=8000,
        ...     embedding=my_embedding_fn
        ... )
    """

    name = "chroma"

    def __init__(
        self,
        collection_name: str,
        embedding: EmbeddingFunction,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        distance_fn: DistanceMetric = DistanceMetric.COSINE,
        collection_metadata: dict[str, Any] | None = None,
        client: "chromadb.ClientAPI | None" = None,
    ):
        """Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            embedding: Embedding function to use
            persist_directory: Local directory for persistence (local mode)
            host: Chroma server host (client mode)
            port: Chroma server port (client mode)
            distance_fn: Distance metric for similarity
            collection_metadata: Additional collection metadata
            client: Pre-configured Chroma client (overrides other connection options)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.collection_name = collection_name
        self.embedding = embedding
        self.persist_directory = persist_directory
        self._distance_fn = distance_fn

        # Map distance metric to Chroma's format
        distance_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",
            DistanceMetric.INNER_PRODUCT: "ip",
        }
        chroma_distance = distance_map.get(distance_fn, "cosine")

        # Initialize client
        if client:
            self._client = client
        elif host and port:
            # Client/server mode
            self._client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to Chroma server at {host}:{port}")
        elif persist_directory:
            # Local persistent mode
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Using Chroma with persistence at {persist_directory}")
        else:
            # In-memory mode
            self._client = chromadb.Client()
            logger.info("Using Chroma in-memory mode")

        # Create or get collection
        metadata = collection_metadata or {}
        metadata["hnsw:space"] = chroma_distance

        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata=metadata
        )

        logger.info(f"Chroma collection '{collection_name}' initialized")

    def add_documents(
        self, documents: list[Document], ids: list[str] | None = None, **kwargs: Any
    ) -> list[str]:
        """Add documents to Chroma."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Use document IDs if available
        if ids is None:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]

        return self.add_texts(texts, metadatas, ids, **kwargs)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to Chroma."""
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Handle metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Chroma doesn't like None values in metadata
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {
                k: v for k, v in meta.items() if v is not None and not isinstance(v, (list, dict))
            }
            clean_metadatas.append(clean_meta)

        # Add to collection
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=clean_metadatas
        )

        logger.debug(f"Added {len(texts)} texts to Chroma collection")
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        """Search for similar documents."""
        results = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [r.document for r in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[SearchResult]:
        """Search with similarity scores."""
        # Embed query
        query_embedding = self.embedding.embed_query(query)

        # Convert filter to Chroma format
        where = self._convert_filter(filter) if filter else None

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                # Chroma returns distances, lower = more similar
                distance = results["distances"][0][i] if results["distances"] else 0
                score = self._distance_to_score(distance)

                doc = Document(
                    page_content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    id=doc_id,
                )
                search_results.append(SearchResult(document=doc, score=score))

        return search_results

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search by embedding vector."""
        where = self._convert_filter(filter) if filter else None

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas"],
        )

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    page_content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    id=doc_id,
                )
                documents.append(doc)

        return documents

    def delete(
        self,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete documents from Chroma."""
        try:
            if ids:
                self._collection.delete(ids=ids)
            elif filter:
                where = self._convert_filter(filter)
                self._collection.delete(where=where)
            else:
                return False

            logger.debug("Deleted documents from Chroma")
            return True
        except Exception as e:
            logger.exception(f"Failed to delete from Chroma: {e}")
            return False

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Get documents by IDs."""
        results = self._collection.get(ids=ids, include=["documents", "metadatas"])

        documents = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                doc = Document(
                    page_content=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                    id=doc_id,
                )
                documents.append(doc)

        return documents

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: EmbeddingFunction,
        collection_name: str = "mindcore_docs",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> "ChromaVectorStore":
        """Create Chroma store from documents."""
        store = cls(
            collection_name=collection_name,
            embedding=embedding,
            persist_directory=persist_directory,
            **kwargs,
        )
        store.add_documents(documents)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: EmbeddingFunction,
        metadatas: list[dict[str, Any]] | None = None,
        collection_name: str = "mindcore_docs",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> "ChromaVectorStore":
        """Create Chroma store from texts."""
        store = cls(
            collection_name=collection_name,
            embedding=embedding,
            persist_directory=persist_directory,
            **kwargs,
        )
        store.add_texts(texts, metadatas)
        return store

    def _convert_filter(self, filter: dict[str, Any]) -> dict[str, Any]:
        """Convert generic filter to Chroma where clause format."""
        if not filter:
            return {}

        # Chroma uses {"field": {"$eq": value}} format
        chroma_filter = {}
        for key, value in filter.items():
            if isinstance(value, dict):
                # Already in operator format
                chroma_filter[key] = value
            else:
                # Simple equality
                chroma_filter[key] = {"$eq": value}

        return chroma_filter

    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score (0-1, higher = more similar)."""
        if self._distance_fn == DistanceMetric.COSINE:
            # Cosine distance: 0 = identical, 2 = opposite
            return max(0, 1 - distance / 2)
        if self._distance_fn in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            # Inner product: higher = more similar (already a score)
            return distance
        # Euclidean: convert distance to similarity
        return 1 / (1 + distance)

    def health_check(self) -> bool:
        """Check if Chroma is accessible."""
        try:
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.exception(f"Chroma health check failed: {e}")
            return False

    def persist(self) -> None:
        """Persist data to disk (for local mode).

        Note: With PersistentClient, data is auto-persisted.
        """
        # PersistentClient auto-persists, but we keep this for API compatibility
        logger.debug("Chroma data persisted")

    @property
    def count(self) -> int:
        """Return number of documents in collection."""
        return self._collection.count()
