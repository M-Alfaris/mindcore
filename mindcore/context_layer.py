"""Modular Context Layer.
=====================

A flexible, modular context assembly system that allows you to pick
which components to use. Inspired by LangChain's modular architecture.

Tiers:
------
1. **Basic** - Messages table + cache only (SQLite/PostgreSQL)
2. **Standard** - Basic + external connectors (orders, billing, etc.)
3. **Advanced** - Standard + vector store for semantic search
4. **Full** - All features enabled

Example (Basic - just messages):
    >>> layer = ContextLayer.basic(
    ...     database="sqlite:///mindcore.db"
    ... )

Example (With Vector Store):
    >>> layer = ContextLayer.with_vector_store(
    ...     database="sqlite:///mindcore.db",
    ...     vector_store=ChromaVectorStore(...),
    ... )

Example (Full Configuration):
    >>> layer = ContextLayer(
    ...     database=SQLiteManager("mindcore.db"),
    ...     cache=DiskCacheManager(),
    ...     vector_store=PineconeVectorStore(...),
    ...     connectors=ConnectorRegistry(),
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .utils.logger import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from .connectors import ConnectorRegistry
    from .vectorstores import EmbeddingFunction, VectorStore


class ContextLayerTier(str, Enum):
    """Context layer configuration tiers."""

    BASIC = "basic"  # Messages + cache only
    STANDARD = "standard"  # + external connectors
    ADVANCED = "advanced"  # + vector store
    FULL = "full"  # All features


@dataclass
class ContextLayerConfig:
    """Configuration for the context layer.

    Specifies which components are enabled and their settings.
    """

    # Core components (always required)
    database_url: str | None = None
    use_sqlite: bool = True
    sqlite_path: str = "mindcore.db"

    # Cache settings
    cache_enabled: bool = True
    cache_type: str = "disk"  # "memory" or "disk"
    cache_max_size: int = 1000
    cache_ttl: int = 3600
    cache_dir: str | None = None

    # Vector store settings
    vector_store_enabled: bool = False
    vector_store_type: str | None = None  # "chroma", "pinecone", "pgvector", "memory"
    vector_store_config: dict[str, Any] = field(default_factory=dict)

    # Embedding settings
    embedding_provider: str = "openai"  # "openai", "sentence_transformers", "ollama"
    embedding_config: dict[str, Any] = field(default_factory=dict)

    # External connectors settings
    connectors_enabled: bool = False
    connectors_config: dict[str, Any] = field(default_factory=dict)

    # LLM settings (for agents)
    llm_provider: str = "auto"  # "auto", "llama_cpp", "openai"

    @classmethod
    def basic(cls, sqlite_path: str = "mindcore.db") -> "ContextLayerConfig":
        """Create basic configuration (messages + cache only)."""
        return cls(
            use_sqlite=True,
            sqlite_path=sqlite_path,
            cache_enabled=True,
            cache_type="disk",
            vector_store_enabled=False,
            connectors_enabled=False,
        )

    @classmethod
    def standard(cls, sqlite_path: str = "mindcore.db") -> "ContextLayerConfig":
        """Create standard configuration (+ connectors)."""
        config = cls.basic(sqlite_path)
        config.connectors_enabled = True
        return config

    @classmethod
    def advanced(
        cls, sqlite_path: str = "mindcore.db", vector_store_type: str = "chroma"
    ) -> "ContextLayerConfig":
        """Create advanced configuration (+ vector store)."""
        config = cls.standard(sqlite_path)
        config.vector_store_enabled = True
        config.vector_store_type = vector_store_type
        return config

    @classmethod
    def full(
        cls, sqlite_path: str = "mindcore.db", vector_store_type: str = "chroma"
    ) -> "ContextLayerConfig":
        """Create full configuration (all features)."""
        return cls.advanced(sqlite_path, vector_store_type)


class ContextLayer:
    """Modular context layer that assembles context from multiple sources.

    The context layer is the core abstraction for retrieving relevant
    context for AI conversations. It can be configured to use different
    combinations of:

    1. **Message Store** - SQLite or PostgreSQL for conversation history
    2. **Cache** - In-memory or disk-based caching
    3. **Vector Store** - Semantic search (Chroma, Pinecone, pgvector)
    4. **External Connectors** - Business data (orders, billing, etc.)

    Example:
        >>> # Basic setup (messages only)
        >>> layer = ContextLayer.basic()
        >>>
        >>> # With vector search
        >>> layer = ContextLayer.with_vector_store(
        ...     vector_store_type="chroma",
        ...     persist_directory="./vectors"
        ... )
        >>>
        >>> # Get context
        >>> context = await layer.get_context(
        ...     user_id="user123",
        ...     query="What about my order?",
        ...     thread_id="thread456"
        ... )
    """

    def __init__(
        self,
        config: ContextLayerConfig | None = None,
        # Direct component injection
        database: Any | None = None,
        cache: Any | None = None,
        vector_store: "VectorStore | None" = None,
        embedding: "EmbeddingFunction | None" = None,
        connectors: "ConnectorRegistry | None" = None,
    ):
        """Initialize context layer.

        Args:
            config: Configuration object
            database: Pre-configured database manager
            cache: Pre-configured cache manager
            vector_store: Pre-configured vector store
            embedding: Pre-configured embedding function
            connectors: Pre-configured connector registry
        """
        self.config = config or ContextLayerConfig.basic()
        self._database = database
        self._cache = cache
        self._vector_store = vector_store
        self._embedding = embedding
        self._connectors = connectors

        # Initialize components if not provided
        if self._database is None:
            self._database = self._create_database()

        if self._cache is None and self.config.cache_enabled:
            self._cache = self._create_cache()

        if self._vector_store is None and self.config.vector_store_enabled:
            self._embedding = self._create_embedding()
            self._vector_store = self._create_vector_store()

        if self._connectors is None and self.config.connectors_enabled:
            self._connectors = self._create_connectors()

        self._log_configuration()

    def _create_database(self):
        """Create database manager based on config."""
        if self.config.use_sqlite:
            from .core import SQLiteManager

            return SQLiteManager(self.config.sqlite_path)
        from .core import DatabaseManager

        return DatabaseManager({"url": self.config.database_url})

    def _create_cache(self):
        """Create cache manager based on config."""
        if self.config.cache_type == "disk":
            from .core import DiskCacheManager

            return DiskCacheManager(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl,
                cache_dir=self.config.cache_dir,
            )
        from .core import CacheManager

        return CacheManager(max_size=self.config.cache_max_size, ttl_seconds=self.config.cache_ttl)

    def _create_embedding(self):
        """Create embedding function based on config."""
        from .vectorstores import create_embeddings

        return create_embeddings(
            provider=self.config.embedding_provider, **self.config.embedding_config
        )

    def _create_vector_store(self):
        """Create vector store based on config."""
        vs_type = self.config.vector_store_type
        vs_config = self.config.vector_store_config

        if vs_type == "memory":
            from .vectorstores import InMemoryVectorStore

            return InMemoryVectorStore(embedding=self._embedding)

        if vs_type == "chroma":
            from .vectorstores import get_chroma_store

            ChromaVectorStore = get_chroma_store()
            return ChromaVectorStore(
                embedding=self._embedding,
                collection_name=vs_config.get("collection_name", "mindcore"),
                persist_directory=vs_config.get("persist_directory"),
                **{
                    k: v
                    for k, v in vs_config.items()
                    if k not in ("collection_name", "persist_directory")
                },
            )

        if vs_type == "pinecone":
            from .vectorstores import get_pinecone_store

            PineconeVectorStore = get_pinecone_store()
            return PineconeVectorStore(
                embedding=self._embedding,
                api_key=vs_config.get("api_key"),
                index_name=vs_config.get("index_name"),
                namespace=vs_config.get("namespace", ""),
                **{
                    k: v
                    for k, v in vs_config.items()
                    if k not in ("api_key", "index_name", "namespace")
                },
            )

        if vs_type == "pgvector":
            from .vectorstores import get_pgvector_store

            PGVectorStore = get_pgvector_store()
            return PGVectorStore(
                embedding=self._embedding,
                connection_string=vs_config.get("connection_string"),
                collection_name=vs_config.get("collection_name", "mindcore_vectors"),
                **{
                    k: v
                    for k, v in vs_config.items()
                    if k not in ("connection_string", "collection_name")
                },
            )

        logger.warning(f"Unknown vector store type: {vs_type}")
        return None

    def _create_connectors(self):
        """Create connector registry based on config."""
        from .connectors import ConnectorRegistry

        return ConnectorRegistry()

    def _log_configuration(self):
        """Log the current configuration."""
        components = ["database"]
        if self._cache:
            components.append("cache")
        if self._vector_store:
            components.append(f"vector_store({self.config.vector_store_type})")
        if self._connectors:
            components.append("connectors")

        logger.info(f"ContextLayer initialized with: {', '.join(components)}")

    # Factory methods for common configurations

    @classmethod
    def basic(cls, sqlite_path: str = "mindcore.db", **kwargs) -> "ContextLayer":
        """Create a basic context layer with just messages and cache.

        Args:
            sqlite_path: Path to SQLite database
            **kwargs: Additional config options

        Returns:
            ContextLayer with basic configuration
        """
        config = ContextLayerConfig.basic(sqlite_path)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config=config)

    @classmethod
    def with_connectors(cls, sqlite_path: str = "mindcore.db", **kwargs) -> "ContextLayer":
        """Create context layer with external connectors enabled.

        Args:
            sqlite_path: Path to SQLite database
            **kwargs: Additional config options

        Returns:
            ContextLayer with connectors enabled
        """
        config = ContextLayerConfig.standard(sqlite_path)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config=config)

    @classmethod
    def with_vector_store(
        cls,
        vector_store_type: str = "chroma",
        sqlite_path: str = "mindcore.db",
        embedding_provider: str = "openai",
        **kwargs,
    ) -> "ContextLayer":
        """Create context layer with vector store for semantic search.

        Args:
            vector_store_type: Type of vector store ("chroma", "pinecone", "pgvector", "memory")
            sqlite_path: Path to SQLite database
            embedding_provider: Embedding provider ("openai", "sentence_transformers", "ollama")
            **kwargs: Additional config (vector_store_config, embedding_config, etc.)

        Returns:
            ContextLayer with vector store enabled
        """
        config = ContextLayerConfig.advanced(sqlite_path, vector_store_type)
        config.embedding_provider = embedding_provider

        # Handle nested configs
        if "vector_store_config" in kwargs:
            config.vector_store_config = kwargs.pop("vector_store_config")
        if "embedding_config" in kwargs:
            config.embedding_config = kwargs.pop("embedding_config")

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return cls(config=config)

    @classmethod
    def full(
        cls, vector_store_type: str = "chroma", sqlite_path: str = "mindcore.db", **kwargs
    ) -> "ContextLayer":
        """Create fully-featured context layer with all components.

        Args:
            vector_store_type: Type of vector store
            sqlite_path: Path to SQLite database
            **kwargs: Additional config options

        Returns:
            ContextLayer with all features enabled
        """
        config = ContextLayerConfig.full(sqlite_path, vector_store_type)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config=config)

    # Core functionality

    @property
    def database(self):
        """Get the database manager."""
        return self._database

    @property
    def cache(self):
        """Get the cache manager."""
        return self._cache

    @property
    def vector_store(self) -> "VectorStore | None":
        """Get the vector store (None if not enabled)."""
        return self._vector_store

    @property
    def connectors(self) -> "ConnectorRegistry | None":
        """Get the connector registry (None if not enabled)."""
        return self._connectors

    @property
    def has_vector_store(self) -> bool:
        """Check if vector store is available."""
        return self._vector_store is not None

    @property
    def has_connectors(self) -> bool:
        """Check if connectors are available."""
        return self._connectors is not None

    def get_recent_messages(self, user_id: str, thread_id: str, limit: int = 20) -> list[Any]:
        """Get recent messages from cache or database.

        Args:
            user_id: User identifier
            thread_id: Thread identifier
            limit: Maximum messages to return

        Returns:
            List of Message objects
        """
        # Try cache first
        if self._cache:
            cached = self._cache.get_recent_messages(user_id, thread_id, limit)
            if cached:
                return cached

        # Fall back to database
        return self._database.fetch_recent_messages(user_id, thread_id, limit)

    def search_messages(
        self,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        limit: int = 10,
        use_vector_search: bool = True,
    ) -> list[Any]:
        """Search messages using available methods.

        If vector store is available and use_vector_search=True,
        uses semantic search. Otherwise falls back to metadata search.

        Args:
            user_id: User identifier
            query: Search query
            thread_id: Optional thread filter
            limit: Maximum results
            use_vector_search: Whether to use vector search if available

        Returns:
            List of relevant messages/documents
        """
        # Use vector search if available
        if self._vector_store and use_vector_search:
            filter_dict = {"user_id": user_id}
            if thread_id:
                filter_dict["thread_id"] = thread_id

            return self._vector_store.similarity_search(query=query, k=limit, filter=filter_dict)

        # Fall back to database metadata search
        return self._database.search_by_relevance(user_id=user_id, thread_id=thread_id, limit=limit)

    async def get_external_context(
        self, user_id: str, topics: list[str], context: dict[str, Any]
    ) -> list[Any]:
        """Get context from external connectors.

        Args:
            user_id: User identifier
            topics: Topics to query for
            context: Additional context (extracted entities, etc.)

        Returns:
            List of ConnectorResult objects
        """
        if not self._connectors:
            return []

        return await self._connectors.lookup(user_id=user_id, topics=topics, context=context)

    async def get_context(
        self,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        max_messages: int = 20,
        max_vector_results: int = 5,
        topics: list[str] | None = None,
        include_external: bool = True,
    ) -> dict[str, Any]:
        """Get assembled context from all available sources.

        This is the main method for retrieving context for an AI conversation.
        It combines:
        1. Recent messages from the conversation
        2. Semantically similar content from vector store
        3. External data from connectors

        Args:
            user_id: User identifier
            query: The query/question to get context for
            thread_id: Optional thread identifier
            max_messages: Maximum recent messages to include
            max_vector_results: Maximum vector search results
            topics: Topics for external connector lookups
            include_external: Whether to include external connector data

        Returns:
            Dictionary containing context from all sources:
            {
                "recent_messages": [...],
                "semantic_matches": [...],
                "external_data": [...],
                "metadata": {...}
            }
        """
        context = {
            "recent_messages": [],
            "semantic_matches": [],
            "external_data": [],
            "metadata": {
                "user_id": user_id,
                "thread_id": thread_id,
                "query": query,
                "sources_used": [],
            },
        }

        # Get recent messages
        if thread_id:
            context["recent_messages"] = self.get_recent_messages(user_id, thread_id, max_messages)
            context["metadata"]["sources_used"].append("messages")

        # Get semantic matches from vector store
        if self._vector_store:
            context["semantic_matches"] = self.search_messages(
                user_id=user_id,
                query=query,
                thread_id=thread_id,
                limit=max_vector_results,
                use_vector_search=True,
            )
            context["metadata"]["sources_used"].append("vector_store")

        # Get external data from connectors
        if self._connectors and include_external and topics:
            external_results = await self.get_external_context(
                user_id=user_id, topics=topics, context={"query": query, "thread_id": thread_id}
            )
            context["external_data"] = external_results
            context["metadata"]["sources_used"].append("connectors")

        return context

    def add_to_vector_store(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str] | None:
        """Add texts to the vector store.

        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text

        Returns:
            List of IDs if vector store is enabled, None otherwise
        """
        if not self._vector_store:
            logger.warning("Vector store not enabled, cannot add texts")
            return None

        return self._vector_store.add_texts(texts, metadatas, ids)

    def register_connector(self, connector: Any) -> None:
        """Register an external connector.

        Args:
            connector: Connector instance to register
        """
        if not self._connectors:
            logger.warning("Connectors not enabled, cannot register")
            return

        self._connectors.register(connector)

    def health_check(self) -> dict[str, bool]:
        """Check health of all components.

        Returns:
            Dictionary mapping component names to health status
        """
        status = {"database": True}  # Assume healthy if we got this far

        if self._cache:
            status["cache"] = True  # In-memory/disk cache is always "healthy"

        if self._vector_store:
            status["vector_store"] = self._vector_store.health_check()

        if self._connectors:
            # This would need to be async in practice
            status["connectors"] = True

        return status

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if hasattr(self._database, "close"):
            self._database.close()

        if hasattr(self._cache, "close"):
            self._cache.close()

        if hasattr(self._vector_store, "close"):
            self._vector_store.close()

        logger.info("ContextLayer closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Convenience aliases
BasicContextLayer = ContextLayer.basic
VectorContextLayer = ContextLayer.with_vector_store
FullContextLayer = ContextLayer.full
