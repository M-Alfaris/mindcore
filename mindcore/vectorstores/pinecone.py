"""
Pinecone vector store integration.

Pinecone is a fully managed vector database with:
- Cloud-native scalability
- Real-time updates
- Metadata filtering
- Namespaces for multi-tenancy

Install: pip install pinecone-client
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import uuid
import time

from .base import (
    VectorStore, Document, SearchResult, EmbeddingFunction,
    DistanceMetric
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pinecone import Pinecone, Index


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector store integration.

    Supports both serverless and pod-based Pinecone indexes.

    Example:
        >>> from mindcore.vectorstores import PineconeVectorStore
        >>> store = PineconeVectorStore(
        ...     api_key="your-api-key",
        ...     index_name="my-index",
        ...     embedding=my_embedding_fn,
        ...     namespace="production"
        ... )
        >>> store.add_texts(["Hello world", "Goodbye world"])
        >>> results = store.similarity_search("Hello", k=1)

    Example (create index):
        >>> store = PineconeVectorStore.create_index(
        ...     api_key="your-api-key",
        ...     index_name="new-index",
        ...     embedding=my_embedding_fn,
        ...     dimension=1536,
        ...     metric="cosine"
        ... )
    """

    name = "pinecone"

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding: EmbeddingFunction,
        namespace: str = "",
        environment: Optional[str] = None,
        text_key: str = "text",
        index: Optional["Index"] = None
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding: Embedding function to use
            namespace: Namespace within the index (for multi-tenancy)
            environment: Pinecone environment (deprecated in newer SDK)
            text_key: Metadata key for storing document text
            index: Pre-configured Pinecone Index (overrides other options)
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client not installed. Run: pip install pinecone-client"
            )

        self.embedding = embedding
        self.namespace = namespace
        self.text_key = text_key
        self._index_name = index_name

        if index:
            self._index = index
        else:
            # Initialize Pinecone client (v3+ SDK)
            pc = Pinecone(api_key=api_key)
            self._index = pc.Index(index_name)

        logger.info(f"Connected to Pinecone index '{index_name}'")

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to Pinecone."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if ids is None:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]

        return self.add_texts(texts, metadatas, ids, **kwargs)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        **kwargs: Any
    ) -> List[str]:
        """Add texts to Pinecone."""
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

        # Prepare vectors for upsert
        vectors = []
        for i, (doc_id, embedding, text, metadata) in enumerate(
            zip(ids, embeddings, texts, metadatas)
        ):
            # Store text in metadata
            meta = dict(metadata)
            meta[self.text_key] = text

            # Pinecone requires flat metadata values
            clean_meta = self._flatten_metadata(meta)

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": clean_meta
            })

        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)

        logger.debug(f"Added {len(texts)} texts to Pinecone")
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents."""
        results = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [r.document for r in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """Search with similarity scores."""
        # Embed query
        query_embedding = self.embedding.embed_query(query)

        return self._search_by_vector_with_score(
            query_embedding, k, filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search by embedding vector."""
        results = self._search_by_vector_with_score(embedding, k, filter, **kwargs)
        return [r.document for r in results]

    def _search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> List[SearchResult]:
        """Internal search implementation."""
        # Query Pinecone
        results = self._index.query(
            vector=embedding,
            top_k=k,
            filter=filter,
            include_metadata=True,
            namespace=self.namespace
        )

        # Convert to SearchResult
        search_results = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})

            # Extract text from metadata
            text = metadata.pop(self.text_key, "")

            doc = Document(
                page_content=text,
                metadata=metadata,
                id=match["id"]
            )

            # Pinecone returns scores (higher = more similar)
            score = match.get("score", 0.0)
            search_results.append(SearchResult(document=doc, score=score))

        return search_results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        **kwargs: Any
    ) -> bool:
        """Delete documents from Pinecone."""
        try:
            if delete_all:
                self._index.delete(delete_all=True, namespace=self.namespace)
            elif ids:
                self._index.delete(ids=ids, namespace=self.namespace)
            elif filter:
                self._index.delete(filter=filter, namespace=self.namespace)
            else:
                return False

            logger.debug(f"Deleted documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Pinecone: {e}")
            return False

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        results = self._index.fetch(ids=ids, namespace=self.namespace)

        documents = []
        for doc_id, data in results.get("vectors", {}).items():
            metadata = data.get("metadata", {})
            text = metadata.pop(self.text_key, "")

            doc = Document(
                page_content=text,
                metadata=metadata,
                id=doc_id
            )
            documents.append(doc)

        return documents

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: EmbeddingFunction,
        api_key: str,
        index_name: str,
        namespace: str = "",
        **kwargs: Any
    ) -> "PineconeVectorStore":
        """Create Pinecone store from documents."""
        store = cls(
            api_key=api_key,
            index_name=index_name,
            embedding=embedding,
            namespace=namespace,
            **kwargs
        )
        store.add_documents(documents)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: EmbeddingFunction,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        index_name: str = "",
        namespace: str = "",
        **kwargs: Any
    ) -> "PineconeVectorStore":
        """Create Pinecone store from texts."""
        store = cls(
            api_key=api_key,
            index_name=index_name,
            embedding=embedding,
            namespace=namespace,
            **kwargs
        )
        store.add_texts(texts, metadatas)
        return store

    @classmethod
    def create_index(
        cls,
        api_key: str,
        index_name: str,
        embedding: EmbeddingFunction,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        spec_type: str = "serverless",
        **kwargs: Any
    ) -> "PineconeVectorStore":
        """
        Create a new Pinecone index and return a vector store.

        Args:
            api_key: Pinecone API key
            index_name: Name for the new index
            embedding: Embedding function (dimension auto-detected if not provided)
            dimension: Vector dimension (auto-detected from embedding if not provided)
            metric: Distance metric ("cosine", "euclidean", "dotproduct")
            cloud: Cloud provider ("aws", "gcp", "azure")
            region: Cloud region
            spec_type: "serverless" or "pod"
            **kwargs: Additional arguments

        Returns:
            PineconeVectorStore connected to the new index
        """
        try:
            from pinecone import Pinecone, ServerlessSpec, PodSpec
        except ImportError:
            raise ImportError(
                "pinecone-client not installed. Run: pip install pinecone-client"
            )

        # Auto-detect dimension from embedding
        if dimension is None:
            dimension = embedding.dimension

        pc = Pinecone(api_key=api_key)

        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]

        if index_name not in index_names:
            # Create index
            if spec_type == "serverless":
                spec = ServerlessSpec(cloud=cloud, region=region)
            else:
                spec = PodSpec(environment=region)

            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec
            )

            # Wait for index to be ready
            while not pc.describe_index(index_name).status.ready:
                time.sleep(1)

            logger.info(f"Created Pinecone index '{index_name}'")

        return cls(
            api_key=api_key,
            index_name=index_name,
            embedding=embedding,
            **kwargs
        )

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata for Pinecone (no nested dicts/lists)."""
        flat = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string if all strings
                if all(isinstance(v, str) for v in value):
                    flat[key] = ",".join(value)
                else:
                    flat[key] = str(value)
            elif isinstance(value, dict):
                # Flatten nested dict with dot notation
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat[f"{key}.{k}"] = v
            else:
                flat[key] = str(value)

        return flat

    def health_check(self) -> bool:
        """Check if Pinecone is accessible."""
        try:
            stats = self._index.describe_index_stats()
            return stats is not None
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return False

    @property
    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self._index.describe_index_stats()
