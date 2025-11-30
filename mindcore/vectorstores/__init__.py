"""
Mindcore Vector Stores Package
==============================

Modular vector store integrations inspired by LangChain's architecture.
Provides a unified interface for different vector databases.

Architecture:
------------
- mindcore.vectorstores.base: Core abstractions (VectorStore, Document, EmbeddingFunction)
- mindcore.vectorstores.embeddings: Embedding providers (OpenAI, SentenceTransformers, Ollama)
- mindcore.vectorstores.chroma: Chroma integration
- mindcore.vectorstores.pinecone: Pinecone integration
- mindcore.vectorstores.pgvector: PostgreSQL + pgvector integration

Quick Start:
-----------
    >>> from mindcore.vectorstores import (
    ...     ChromaVectorStore,
    ...     OpenAIEmbeddings,
    ...     Document
    ... )
    >>>
    >>> # Create embeddings
    >>> embeddings = OpenAIEmbeddings()
    >>>
    >>> # Create vector store
    >>> store = ChromaVectorStore(
    ...     collection_name="my_docs",
    ...     embedding=embeddings,
    ...     persist_directory="./chroma_db"
    ... )
    >>>
    >>> # Add documents
    >>> docs = [
    ...     Document(page_content="Hello world", metadata={"source": "greeting"}),
    ...     Document(page_content="Goodbye world", metadata={"source": "farewell"})
    ... ]
    >>> store.add_documents(docs)
    >>>
    >>> # Search
    >>> results = store.similarity_search("Hello", k=1)
    >>> print(results[0].page_content)
    'Hello world'

Using Local Embeddings (No API Required):
----------------------------------------
    >>> from mindcore.vectorstores import (
    ...     InMemoryVectorStore,
    ...     SentenceTransformerEmbeddings
    ... )
    >>>
    >>> # Local embeddings (no API key needed)
    >>> embeddings = SentenceTransformerEmbeddings(
    ...     model_name="all-MiniLM-L6-v2"
    ... )
    >>>
    >>> # In-memory store for development
    >>> store = InMemoryVectorStore(embedding=embeddings)
"""

# Base classes
from .base import (
    VectorStore,
    Document,
    SearchResult,
    EmbeddingFunction,
    DistanceMetric,
    VectorStoreRetriever,
    InMemoryVectorStore,
)

# Embedding providers
from .embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    OllamaEmbeddings,
    CustomEmbeddings,
    create_embeddings,
)


# Lazy imports for optional dependencies
def get_chroma_store():
    """Get ChromaVectorStore (requires: pip install chromadb)."""
    from .chroma import ChromaVectorStore
    return ChromaVectorStore


def get_pinecone_store():
    """Get PineconeVectorStore (requires: pip install pinecone-client)."""
    from .pinecone import PineconeVectorStore
    return PineconeVectorStore


def get_pgvector_store():
    """Get PGVectorStore (requires: pip install 'psycopg[binary]' pgvector)."""
    from .pgvector import PGVectorStore
    return PGVectorStore


# Helper to create informative placeholder for missing dependencies
def _missing_dep_class(name: str, install_cmd: str):
    """Create a class that raises helpful error when instantiated."""
    class MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires additional dependencies. "
                f"Install with: {install_cmd}"
            )
        def __repr__(self):
            return f"<{name} - not installed>"
    MissingDependency.__name__ = name
    MissingDependency.__qualname__ = name
    return MissingDependency


# Try to import optional stores directly for convenience
try:
    from .chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = _missing_dep_class(
        "ChromaVectorStore",
        "pip install mindcore[chroma]  # or: pip install chromadb"
    )

try:
    from .pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = _missing_dep_class(
        "PineconeVectorStore",
        "pip install mindcore[pinecone]  # or: pip install pinecone-client"
    )

try:
    from .pgvector import PGVectorStore
except ImportError:
    PGVectorStore = _missing_dep_class(
        "PGVectorStore",
        "pip install mindcore[pgvector]  # or: pip install pgvector"
    )


__all__ = [
    # Base classes
    "VectorStore",
    "Document",
    "SearchResult",
    "EmbeddingFunction",
    "DistanceMetric",
    "VectorStoreRetriever",
    "InMemoryVectorStore",

    # Embedding providers
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "OllamaEmbeddings",
    "CustomEmbeddings",
    "create_embeddings",

    # Vector store implementations (may be None if deps not installed)
    "ChromaVectorStore",
    "PineconeVectorStore",
    "PGVectorStore",

    # Lazy loaders
    "get_chroma_store",
    "get_pinecone_store",
    "get_pgvector_store",
]
