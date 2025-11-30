"""
PostgreSQL with pgvector extension integration.

pgvector adds vector similarity search to PostgreSQL with:
- Native PostgreSQL integration
- ACID transactions
- SQL-based filtering
- IVFFlat and HNSW indexes

Install: pip install psycopg[binary] pgvector
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import uuid
import json

from .base import (
    VectorStore, Document, SearchResult, EmbeddingFunction,
    DistanceMetric
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PGVectorStore(VectorStore):
    """
    PostgreSQL + pgvector vector store integration.

    Provides vector similarity search within PostgreSQL, allowing
    hybrid queries combining vector similarity with SQL conditions.

    Example:
        >>> from mindcore.vectorstores import PGVectorStore
        >>> store = PGVectorStore(
        ...     connection_string="postgresql://user:pass@localhost/db",
        ...     embedding=my_embedding_fn,
        ...     collection_name="documents"
        ... )
        >>> store.add_texts(["Hello world", "Goodbye world"])
        >>> results = store.similarity_search("Hello", k=1)

    Example (with existing connection):
        >>> import psycopg
        >>> conn = psycopg.connect(...)
        >>> store = PGVectorStore(
        ...     connection=conn,
        ...     embedding=my_embedding_fn
        ... )
    """

    name = "pgvector"

    def __init__(
        self,
        embedding: EmbeddingFunction,
        connection_string: Optional[str] = None,
        connection: Optional[Any] = None,
        collection_name: str = "mindcore_vectors",
        distance_strategy: DistanceMetric = DistanceMetric.COSINE,
        pre_delete_collection: bool = False,
        use_jsonb: bool = True
    ):
        """
        Initialize pgvector store.

        Args:
            embedding: Embedding function to use
            connection_string: PostgreSQL connection string
            connection: Pre-configured psycopg connection
            collection_name: Name of the table/collection
            distance_strategy: Distance metric for similarity
            pre_delete_collection: Drop existing table on init
            use_jsonb: Store metadata as JSONB (enables filtering)
        """
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError:
            raise ImportError(
                "psycopg and pgvector not installed. Run: pip install 'psycopg[binary]' pgvector"
            )

        self.embedding = embedding
        self.collection_name = collection_name
        self._distance_strategy = distance_strategy
        self._use_jsonb = use_jsonb

        # Initialize connection
        if connection:
            self._conn = connection
            self._owns_connection = False
        elif connection_string:
            self._conn = psycopg.connect(connection_string)
            self._owns_connection = True
        else:
            raise ValueError("Either connection_string or connection required")

        # Register pgvector extension
        register_vector(self._conn)

        # Ensure pgvector extension exists
        self._ensure_extension()

        # Create table if needed
        if pre_delete_collection:
            self._drop_table()
        self._create_table()

        logger.info(f"PGVector store initialized with table '{collection_name}'")

    def _ensure_extension(self) -> None:
        """Ensure pgvector extension is installed."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

    def _create_table(self) -> None:
        """Create the vectors table if it doesn't exist."""
        dimension = self.embedding.dimension

        metadata_col = "metadata JSONB" if self._use_jsonb else "metadata TEXT"

        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                embedding vector({dimension}),
                {metadata_col},
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """

        with self._conn.cursor() as cur:
            cur.execute(sql)
            self._conn.commit()

    def _drop_table(self) -> None:
        """Drop the vectors table."""
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            self._conn.commit()

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to pgvector."""
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
        **kwargs: Any
    ) -> List[str]:
        """Add texts to pgvector."""
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

        # Insert into database
        sql = f"""
            INSERT INTO {self.collection_name} (id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """

        with self._conn.cursor() as cur:
            for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                meta_value = json.dumps(metadata) if self._use_jsonb else str(metadata)
                cur.execute(sql, (doc_id, text, embedding, meta_value))

            self._conn.commit()

        logger.debug(f"Added {len(texts)} texts to pgvector")
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

        return self._search_by_vector_with_score(query_embedding, k, filter)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search by embedding vector."""
        results = self._search_by_vector_with_score(embedding, k, filter)
        return [r.document for r in results]

    def _search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Internal search implementation."""
        # Build distance expression based on strategy
        if self._distance_strategy == DistanceMetric.COSINE:
            distance_expr = f"embedding <=> %s::vector"
            score_expr = f"1 - (embedding <=> %s::vector)"
        elif self._distance_strategy == DistanceMetric.EUCLIDEAN:
            distance_expr = f"embedding <-> %s::vector"
            score_expr = f"1 / (1 + (embedding <-> %s::vector))"
        elif self._distance_strategy in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            distance_expr = f"embedding <#> %s::vector"
            score_expr = f"(embedding <#> %s::vector) * -1"  # Negate for similarity
        else:
            distance_expr = f"embedding <=> %s::vector"
            score_expr = f"1 - (embedding <=> %s::vector)"

        # Build WHERE clause from filter
        where_clause = ""
        filter_params = []
        if filter and self._use_jsonb:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            conditions.append(f"metadata->>'{key}' = %s")
                            filter_params.append(str(op_value))
                        elif op == "$ne":
                            conditions.append(f"metadata->>'{key}' != %s")
                            filter_params.append(str(op_value))
                        elif op == "$in":
                            placeholders = ",".join(["%s"] * len(op_value))
                            conditions.append(f"metadata->>'{key}' IN ({placeholders})")
                            filter_params.extend([str(v) for v in op_value])
                        elif op == "$gt":
                            conditions.append(f"(metadata->>'{key}')::numeric > %s")
                            filter_params.append(op_value)
                        elif op == "$gte":
                            conditions.append(f"(metadata->>'{key}')::numeric >= %s")
                            filter_params.append(op_value)
                        elif op == "$lt":
                            conditions.append(f"(metadata->>'{key}')::numeric < %s")
                            filter_params.append(op_value)
                        elif op == "$lte":
                            conditions.append(f"(metadata->>'{key}')::numeric <= %s")
                            filter_params.append(op_value)
                else:
                    conditions.append(f"metadata->>'{key}' = %s")
                    filter_params.append(str(value))

            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

        # Build query
        sql = f"""
            SELECT id, content, metadata, {score_expr} as score
            FROM {self.collection_name}
            {where_clause}
            ORDER BY {distance_expr}
            LIMIT %s
        """

        # Execute query
        params = [embedding, embedding] + filter_params + [embedding, k]

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        # Convert to SearchResult
        results = []
        for row in rows:
            doc_id, content, metadata_raw, score = row

            # Parse metadata
            if self._use_jsonb and isinstance(metadata_raw, str):
                metadata = json.loads(metadata_raw)
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = {}

            doc = Document(
                page_content=content,
                metadata=metadata,
                id=doc_id
            )
            results.append(SearchResult(document=doc, score=float(score)))

        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """Delete documents from pgvector."""
        try:
            with self._conn.cursor() as cur:
                if ids:
                    placeholders = ",".join(["%s"] * len(ids))
                    cur.execute(
                        f"DELETE FROM {self.collection_name} WHERE id IN ({placeholders})",
                        ids
                    )
                elif filter and self._use_jsonb:
                    conditions = []
                    params = []
                    for key, value in filter.items():
                        conditions.append(f"metadata->>'{key}' = %s")
                        params.append(str(value))

                    if conditions:
                        where = " AND ".join(conditions)
                        cur.execute(
                            f"DELETE FROM {self.collection_name} WHERE {where}",
                            params
                        )
                else:
                    return False

                self._conn.commit()

            logger.debug("Deleted documents from pgvector")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from pgvector: {e}")
            self._conn.rollback()
            return False

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        if not ids:
            return []

        placeholders = ",".join(["%s"] * len(ids))
        sql = f"""
            SELECT id, content, metadata
            FROM {self.collection_name}
            WHERE id IN ({placeholders})
        """

        with self._conn.cursor() as cur:
            cur.execute(sql, ids)
            rows = cur.fetchall()

        documents = []
        for row in rows:
            doc_id, content, metadata_raw = row

            if self._use_jsonb and isinstance(metadata_raw, str):
                metadata = json.loads(metadata_raw)
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = {}

            documents.append(Document(
                page_content=content,
                metadata=metadata,
                id=doc_id
            ))

        return documents

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: EmbeddingFunction,
        connection_string: str,
        collection_name: str = "mindcore_vectors",
        **kwargs: Any
    ) -> "PGVectorStore":
        """Create pgvector store from documents."""
        store = cls(
            connection_string=connection_string,
            embedding=embedding,
            collection_name=collection_name,
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
        connection_string: str = "",
        collection_name: str = "mindcore_vectors",
        **kwargs: Any
    ) -> "PGVectorStore":
        """Create pgvector store from texts."""
        store = cls(
            connection_string=connection_string,
            embedding=embedding,
            collection_name=collection_name,
            **kwargs
        )
        store.add_texts(texts, metadatas)
        return store

    def create_index(
        self,
        index_type: str = "ivfflat",
        lists: int = 100,
        probes: int = 10
    ) -> None:
        """
        Create an index for faster similarity search.

        Args:
            index_type: "ivfflat" or "hnsw"
            lists: Number of lists for IVFFlat (more = slower build, faster query)
            probes: Number of probes for IVFFlat queries
        """
        # Determine operator based on distance strategy
        if self._distance_strategy == DistanceMetric.COSINE:
            ops = "vector_cosine_ops"
        elif self._distance_strategy == DistanceMetric.EUCLIDEAN:
            ops = "vector_l2_ops"
        elif self._distance_strategy in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            ops = "vector_ip_ops"
        else:
            ops = "vector_cosine_ops"

        index_name = f"{self.collection_name}_{index_type}_idx"

        with self._conn.cursor() as cur:
            if index_type == "ivfflat":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.collection_name}
                    USING ivfflat (embedding {ops})
                    WITH (lists = {lists})
                """)
                # Set probes for queries
                cur.execute(f"SET ivfflat.probes = {probes}")
            elif index_type == "hnsw":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.collection_name}
                    USING hnsw (embedding {ops})
                """)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            self._conn.commit()

        logger.info(f"Created {index_type} index on {self.collection_name}")

    def health_check(self) -> bool:
        """Check if PostgreSQL connection is alive."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"pgvector health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the database connection if we own it."""
        if self._owns_connection and self._conn:
            self._conn.close()
            logger.debug("Closed pgvector connection")

    @property
    def count(self) -> int:
        """Return number of documents in collection."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            return cur.fetchone()[0]

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
