"""Knowledge Base - VectorDB integration with metadata filtering.

Provides semantic search over documents with support for:
- Multiple VectorDB backends (Chroma, Pinecone, pgvector)
- Metadata filtering (topics, categories, dates, etc.)
- Document management (add, update, delete)
- Vocabulary mapping for consistent metadata
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.logger import get_logger


if TYPE_CHECKING:
    from mindcore.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


@dataclass
class Document:
    """A document in the knowledge base.

    Documents are stored with embeddings for semantic search
    and metadata for filtering.
    """

    # Required fields
    content: str
    source: str  # Where this document came from

    # Optional identifiers
    doc_id: str | None = None
    url: str | None = None

    # Metadata for filtering (vocabulary-controlled)
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Additional metadata
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Custom metadata (flexible)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate doc_id if not provided."""
        if self.doc_id is None:
            # Hash content + source for unique ID
            hash_input = f"{self.content[:500]}:{self.source}"
            self.doc_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "topics": self.topics,
            "categories": self.categories,
            "tags": self.tags,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            doc_id=data.get("doc_id"),
            content=data["content"],
            source=data["source"],
            url=data.get("url"),
            topics=data.get("topics", []),
            categories=data.get("categories", []),
            tags=data.get("tags", []),
            title=data.get("title"),
            author=data.get("author"),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """A search result from the knowledge base."""

    document: Document
    similarity: float  # 0.0 to 1.0
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **self.document.to_dict(),
            "similarity": self.similarity,
            "rank": self.rank,
        }


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base."""

    # Vector store settings
    collection_name: str = "mindcore_knowledge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Search settings
    default_limit: int = 10
    min_similarity: float = 0.5

    # Chunking settings (for long documents)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Metadata validation
    validate_vocabulary: bool = True


class KnowledgeBase:
    """Knowledge base with VectorDB backend.

    Provides semantic search with metadata filtering,
    vocabulary-controlled tagging, and document management.

    Example:
        from mindcore.vectorstores import ChromaVectorStore
        from mindcore.context_lake import KnowledgeBase, Document

        vector_store = ChromaVectorStore(collection_name="knowledge")
        kb = KnowledgeBase(vector_store=vector_store)

        # Add documents
        kb.add(Document(
            content="How to reset your password...",
            source="help_docs",
            topics=["account", "password"],
            categories=["technical"],
        ))

        # Search
        results = kb.search(
            query="forgot my password",
            metadata_filter={"categories": ["technical"]},
        )
    """

    def __init__(
        self,
        vector_store: "BaseVectorStore",
        vocabulary: VocabularyManager | None = None,
        config: KnowledgeBaseConfig | None = None,
    ):
        """Initialize knowledge base.

        Args:
            vector_store: Vector store backend (Chroma, Pinecone, etc.)
            vocabulary: Vocabulary manager for metadata validation
            config: Configuration options
        """
        self.vector_store = vector_store
        self.vocabulary = vocabulary or get_vocabulary()
        self.config = config or KnowledgeBaseConfig()

        self._doc_count = 0

        logger.info(
            f"KnowledgeBase initialized: "
            f"collection={self.config.collection_name}, "
            f"validate_vocab={self.config.validate_vocabulary}"
        )

    def add(self, document: Document) -> str:
        """Add a document to the knowledge base.

        Args:
            document: Document to add

        Returns:
            Document ID
        """
        # Validate and normalize metadata
        if self.config.validate_vocabulary:
            document.topics = self.vocabulary.validate_topics(document.topics)
            document.categories = self.vocabulary.validate_categories(document.categories)

        # Chunk if too long
        if len(document.content) > self.config.chunk_size:
            chunks = self._chunk_document(document)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    doc_id=f"{document.doc_id}_chunk_{i}",
                    content=chunk,
                    source=document.source,
                    topics=document.topics,
                    categories=document.categories,
                    tags=document.tags,
                    title=document.title,
                    metadata={**document.metadata, "chunk_index": i, "parent_id": document.doc_id},
                )
                self._add_single(chunk_doc)

            self._doc_count += len(chunks)
            logger.info(f"Added document {document.doc_id} ({len(chunks)} chunks)")
            return document.doc_id

        self._add_single(document)
        self._doc_count += 1
        logger.info(f"Added document {document.doc_id}")
        return document.doc_id

    def _add_single(self, document: Document) -> None:
        """Add a single document/chunk to vector store."""
        metadata = {
            "doc_id": document.doc_id,
            "source": document.source,
            "topics": document.topics,
            "categories": document.categories,
            "tags": document.tags,
            "title": document.title or "",
            "created_at": document.created_at.isoformat() if document.created_at else "",
        }

        # Merge custom metadata
        metadata.update(document.metadata)

        self.vector_store.add(
            texts=[document.content],
            metadatas=[metadata],
            ids=[document.doc_id],
        )

    def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        ids = []
        for doc in documents:
            doc_id = self.add(doc)
            ids.append(doc_id)

        logger.info(f"Added batch of {len(documents)} documents")
        return ids

    def search(
        self,
        query: str,
        limit: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query (will be embedded)
            limit: Maximum results to return
            metadata_filter: Filter by metadata (topics, categories, etc.)
            min_similarity: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        limit = limit or self.config.default_limit
        min_similarity = min_similarity or self.config.min_similarity

        # Convert metadata filter to vector store format
        where_filter = None
        if metadata_filter:
            where_filter = self._build_where_filter(metadata_filter)

        # Query vector store
        results = self.vector_store.query(
            query_text=query,
            n_results=limit,
            where=where_filter,
        )

        # Convert to SearchResult objects
        search_results = []
        for i, (doc_data, score) in enumerate(results):
            # Skip low similarity results
            similarity = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
            if similarity < min_similarity:
                continue

            document = Document(
                doc_id=doc_data.get("doc_id", f"result_{i}"),
                content=doc_data.get("content", doc_data.get("text", "")),
                source=doc_data.get("source", "unknown"),
                topics=doc_data.get("topics", []),
                categories=doc_data.get("categories", []),
                tags=doc_data.get("tags", []),
                title=doc_data.get("title"),
                metadata=doc_data.get("metadata", {}),
            )

            search_results.append(SearchResult(
                document=document,
                similarity=similarity,
                rank=i + 1,
            ))

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Knowledge search complete: "
            f"query={query[:50]}..., "
            f"results={len(search_results)}, "
            f"latency={latency_ms:.0f}ms"
        )

        return search_results

    def _build_where_filter(self, metadata_filter: dict[str, Any]) -> dict[str, Any]:
        """Build vector store where filter from metadata filter.

        Handles different filter formats for different backends.
        """
        where = {}

        for key, value in metadata_filter.items():
            if isinstance(value, list):
                # List filter - any match
                where[key] = {"$in": value}
            elif isinstance(value, str):
                # Exact match
                where[key] = value
            elif isinstance(value, dict):
                # Pass through operator filters
                where[key] = value

        return where if where else None

    def _chunk_document(self, document: Document) -> list[str]:
        """Split document into chunks with overlap."""
        content = document.content
        chunks = []

        start = 0
        while start < len(content):
            end = start + self.config.chunk_size

            # Find a good break point (end of sentence or paragraph)
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + self.config.chunk_size // 2:
                    end = para_break

                # Or sentence break
                elif "." in content[start:end]:
                    sent_break = content.rfind(". ", start, end)
                    if sent_break > start + self.config.chunk_size // 2:
                        end = sent_break + 1

            chunks.append(content[start:end].strip())

            # Move start with overlap
            start = end - self.config.chunk_overlap

        return chunks

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted
        """
        try:
            # Also delete any chunks
            self.vector_store.delete(
                where={"$or": [
                    {"doc_id": doc_id},
                    {"parent_id": doc_id},
                ]}
            )
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete document {doc_id}: {e}")
            return False

    def update(self, document: Document) -> bool:
        """Update an existing document.

        Args:
            document: Document with updated content/metadata

        Returns:
            True if updated
        """
        document.updated_at = datetime.now(timezone.utc)

        # Delete old version
        self.delete(document.doc_id)

        # Add new version
        self.add(document)

        logger.info(f"Updated document {document.doc_id}")
        return True

    def get_by_topic(
        self, topic: str, limit: int = 10
    ) -> list[Document]:
        """Get documents by topic.

        Args:
            topic: Topic to filter by
            limit: Maximum results

        Returns:
            List of documents
        """
        results = self.vector_store.query(
            query_text="",  # Empty query for metadata-only search
            n_results=limit,
            where={"topics": {"$in": [topic]}},
        )

        return [
            Document.from_dict(doc_data)
            for doc_data, _ in results
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "document_count": self._doc_count,
            "collection_name": self.config.collection_name,
            "validate_vocabulary": self.config.validate_vocabulary,
        }
