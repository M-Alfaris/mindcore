"""Knowledge Store - Unified Interface for Single or Multi-Agent Systems.
======================================================================

A simple, unified interface for storing and retrieving knowledge that
works for both single-agent and multi-agent deployments.

Design Philosophy:
-----------------
1. **Simple by default** - Works out of the box with zero configuration
2. **Scales when needed** - Add multi-agent features incrementally
3. **Progressive complexity** - Start simple, add features as you grow

Quick Start (Single Agent):
--------------------------
    >>> from mindcore import KnowledgeStore
    >>>
    >>> # Simplest possible setup - just works!
    >>> store = KnowledgeStore()
    >>>
    >>> # Store a message
    >>> store.add_message(
    ...     user_id="user123",
    ...     text="How do I reset my password?",
    ...     role="user"
    ... )
    >>>
    >>> # Get context
    >>> context = store.get_context(user_id="user123", query="password")

Growing to Multi-Agent:
----------------------
    >>> from mindcore import KnowledgeStore, AgentConfig
    >>>
    >>> # Enable multi-agent mode
    >>> store = KnowledgeStore(
    ...     multi_agent=True,
    ...     organization_id="my-company"
    ... )
    >>>
    >>> # Register agents
    >>> support_agent = store.register_agent(
    ...     name="Support Agent",
    ...     groups=["support-team"]
    ... )
    >>>
    >>> sales_agent = store.register_agent(
    ...     name="Sales Agent",
    ...     groups=["sales-team"]
    ... )
    >>>
    >>> # Store with agent context
    >>> store.add_message(
    ...     user_id="user123",
    ...     text="Customer asking about refund",
    ...     role="assistant",
    ...     agent_id=support_agent.agent_id,
    ...     visibility="shared",
    ...     sharing_groups=["support-team"]
    ... )
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .utils.logger import get_logger


logger = get_logger(__name__)


class StoreMode(str, Enum):
    """Operation mode for the knowledge store."""

    SIMPLE = "simple"  # Single agent, no access control
    MULTI_AGENT = "multi_agent"  # Multiple agents with access control


@dataclass
class AgentConfig:
    """Agent configuration for multi-agent mode.

    In simple mode, this is auto-generated with defaults.
    """

    agent_id: str
    name: str
    organization_id: str
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    api_key: str | None = None  # Only set on registration
    is_default: bool = False  # True for the auto-created simple mode agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "organization_id": self.organization_id,
            "groups": self.groups,
            "roles": self.roles,
            "is_default": self.is_default,
        }


@dataclass
class KnowledgeItem:
    """A piece of knowledge in the store.

    Can be a message, document, fact, or any other knowledge type.
    """

    item_id: str
    item_type: str  # "message", "document", "fact", "summary"
    content: str
    user_id: str | None = None
    thread_id: str | None = None

    # Multi-agent fields (optional in simple mode)
    agent_id: str | None = None
    visibility: str = "private"  # "private", "shared", "public"
    sharing_groups: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    # Vector embedding (if vector store enabled)
    embedding: list[float] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "content": self.content,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "visibility": self.visibility,
            "sharing_groups": self.sharing_groups,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class KnowledgeStore:
    """Unified knowledge store for single or multi-agent systems.

    This is the main entry point for storing and retrieving knowledge.
    It automatically handles:
    - Message storage and retrieval
    - Optional vector search
    - Optional multi-agent access control
    - Caching for performance

    Modes:
    ------
    1. **Simple Mode** (default): Single agent, no access control
       - Just works out of the box
       - All data belongs to a default agent
       - No permission checks

    2. **Multi-Agent Mode**: Multiple agents with privacy
       - Register multiple agents
       - Private, shared, and public visibility
       - Fine-grained access control
       - Knowledge sharing between agents
    """

    def __init__(
        self,
        # Database settings
        database_path: str = "mindcore.db",
        use_postgresql: bool = False,
        postgresql_url: str | None = None,
        # Multi-agent settings
        multi_agent: bool = False,
        organization_id: str = "default",
        # Vector store settings (optional)
        enable_vector_search: bool = False,
        vector_store_type: str = "memory",  # "memory", "chroma", "pinecone"
        vector_store_config: dict[str, Any] | None = None,
        embedding_provider: str = "openai",
        embedding_config: dict[str, Any] | None = None,
        # Cache settings
        enable_cache: bool = True,
        cache_type: str = "disk",  # "memory", "disk"
        # LLM settings (for enrichment)
        enable_enrichment: bool = True,
        llm_provider: str = "auto",
    ):
        """Initialize the knowledge store.

        Args:
            database_path: Path to SQLite database (ignored if use_postgresql=True)
            use_postgresql: Use PostgreSQL instead of SQLite
            postgresql_url: PostgreSQL connection URL
            multi_agent: Enable multi-agent mode with access control
            organization_id: Organization identifier (for multi-agent)
            enable_vector_search: Enable semantic search
            vector_store_type: Type of vector store
            vector_store_config: Vector store configuration
            embedding_provider: Embedding provider
            embedding_config: Embedding configuration
            enable_cache: Enable caching
            cache_type: Type of cache
            enable_enrichment: Enable metadata enrichment
            llm_provider: LLM provider for enrichment
        """
        self._mode = StoreMode.MULTI_AGENT if multi_agent else StoreMode.SIMPLE
        self._organization_id = organization_id
        self._enable_enrichment = enable_enrichment

        # Initialize components
        self._db = self._init_database(database_path, use_postgresql, postgresql_url)
        self._cache = self._init_cache(enable_cache, cache_type) if enable_cache else None
        self._vector_store = None
        self._embedding = None
        self._access_control = None
        self._enrichment_agent = None

        # Initialize vector search if enabled
        if enable_vector_search:
            self._embedding = self._init_embedding(embedding_provider, embedding_config or {})
            self._vector_store = self._init_vector_store(
                vector_store_type, vector_store_config or {}
            )

        # Initialize access control for multi-agent mode
        if self._mode == StoreMode.MULTI_AGENT:
            self._access_control = self._init_access_control()

        # Initialize enrichment agent if enabled
        if enable_enrichment:
            self._enrichment_agent = self._init_enrichment(llm_provider)

        # Create default agent for simple mode
        self._default_agent = self._create_default_agent()

        logger.info(
            f"KnowledgeStore initialized in {self._mode.value} mode "
            f"(vector_search={enable_vector_search}, enrichment={enable_enrichment})"
        )

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_database(self, path: str, use_pg: bool, pg_url: str | None):
        """Initialize database connection."""
        if use_pg and pg_url:
            from .core import DatabaseManager

            return DatabaseManager({"url": pg_url})
        from .core import SQLiteManager

        return SQLiteManager(path)

    def _init_cache(self, enabled: bool, cache_type: str):
        """Initialize cache."""
        if not enabled:
            return None
        if cache_type == "disk":
            from .core import DiskCacheManager

            return DiskCacheManager()
        from .core import CacheManager

        return CacheManager()

    def _init_embedding(self, provider: str, config: dict[str, Any]):
        """Initialize embedding function."""
        from .vectorstores import create_embeddings

        return create_embeddings(provider=provider, **config)

    def _init_vector_store(self, store_type: str, config: dict[str, Any]):
        """Initialize vector store."""
        from .vectorstores import InMemoryVectorStore

        if store_type == "memory":
            return InMemoryVectorStore(embedding=self._embedding)
        if store_type == "chroma":
            from .vectorstores import get_chroma_store

            ChromaVectorStore = get_chroma_store()
            return ChromaVectorStore(
                embedding=self._embedding,
                collection_name=config.get("collection_name", "mindcore"),
                persist_directory=config.get("persist_directory"),
            )
        if store_type == "pinecone":
            from .vectorstores import get_pinecone_store

            PineconeVectorStore = get_pinecone_store()
            return PineconeVectorStore(
                embedding=self._embedding,
                api_key=config.get("api_key"),
                index_name=config.get("index_name", "mindcore"),
            )
        return InMemoryVectorStore(embedding=self._embedding)

    def _init_access_control(self):
        """Initialize access control manager."""
        from .core.access_control import AccessControlManager

        return AccessControlManager(database=self._db)

    def _init_enrichment(self, llm_provider: str):
        """Initialize enrichment agent."""
        try:
            from .agents import EnrichmentAgent
            from .llm import create_provider

            provider = create_provider(llm_provider)
            return EnrichmentAgent(llm_provider=provider)
        except Exception as e:
            logger.warning(f"Could not initialize enrichment: {e}")
            return None

    def _create_default_agent(self) -> AgentConfig:
        """Create default agent for simple mode."""
        return AgentConfig(
            agent_id="default-agent",
            name="Default Agent",
            organization_id=self._organization_id,
            groups=["default"],
            roles=["default"],
            is_default=True,
        )

    # =========================================================================
    # Agent Management (Multi-Agent Mode)
    # =========================================================================

    def register_agent(
        self,
        name: str,
        groups: list[str] | None = None,
        roles: list[str] | None = None,
        agent_id: str | None = None,
    ) -> AgentConfig:
        """Register a new agent.

        Only available in multi-agent mode.

        Args:
            name: Human-readable agent name
            groups: Groups for shared access
            roles: Agent roles/capabilities
            agent_id: Optional specific agent ID

        Returns:
            AgentConfig with agent details and API key
        """
        if self._mode != StoreMode.MULTI_AGENT:
            logger.warning("register_agent called in simple mode - returning default agent")
            return self._default_agent

        if self._access_control is None:
            raise RuntimeError("Access control not initialized")

        agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"

        registration, api_key = self._access_control.register_agent(
            agent_id=agent_id,
            name=name,
            owner_id=self._organization_id,
            groups=groups or [],
            roles=roles or [],
        )

        config = AgentConfig(
            agent_id=registration.agent_id,
            name=registration.name,
            organization_id=self._organization_id,
            groups=registration.groups,
            roles=registration.roles,
            api_key=api_key,
        )

        logger.info(f"Registered agent: {name} ({agent_id})")
        return config

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get agent configuration by ID."""
        if self._mode != StoreMode.MULTI_AGENT:
            return self._default_agent if agent_id == "default-agent" else None

        if self._access_control:
            agent = self._access_control.get_agent(agent_id)
            if agent:
                return AgentConfig(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    organization_id=agent.owner_id,
                    groups=agent.groups,
                    roles=agent.roles,
                )
        return None

    # =========================================================================
    # Knowledge Storage
    # =========================================================================

    def add_message(
        self,
        user_id: str,
        text: str,
        role: str = "user",
        thread_id: str | None = None,
        session_id: str | None = None,
        # Multi-agent options (ignored in simple mode)
        agent_id: str | None = None,
        visibility: str = "private",
        sharing_groups: list[str] | None = None,
        # Additional metadata
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeItem:
        """Add a message to the knowledge store.

        This is the primary way to store conversation messages.

        Args:
            user_id: User identifier
            text: Message content
            role: Message role ("user", "assistant", "system")
            thread_id: Optional conversation thread ID
            session_id: Optional session ID
            agent_id: Agent that created this (multi-agent mode)
            visibility: "private", "shared", or "public" (multi-agent mode)
            sharing_groups: Groups to share with (multi-agent mode)
            metadata: Additional metadata

        Returns:
            KnowledgeItem representing the stored message
        """
        item_id = f"msg-{uuid.uuid4().hex[:12]}"
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        session_id = session_id or thread_id

        # Determine agent
        if self._mode == StoreMode.SIMPLE:
            agent_id = self._default_agent.agent_id
            visibility = "private"  # Simple mode = all private
            sharing_groups = []
        else:
            agent_id = agent_id or self._default_agent.agent_id

        # Create knowledge item
        item = KnowledgeItem(
            item_id=item_id,
            item_type="message",
            content=text,
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id,
            visibility=visibility,
            sharing_groups=sharing_groups or [],
            metadata={"role": role, "session_id": session_id, **(metadata or {})},
        )

        # Enrich metadata if enabled
        if self._enrichment_agent and self._enable_enrichment:
            try:
                enriched = self._enrichment_agent.process(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "session_id": session_id,
                        "role": role,
                        "text": text,
                    }
                )
                item.metadata["enrichment"] = enriched.metadata.to_dict()
            except Exception as e:
                logger.warning(f"Enrichment failed: {e}")

        # Store in database
        self._store_item(item)

        # Add to vector store if enabled
        if self._vector_store and self._embedding:
            self._add_to_vector_store(item)

        # Add to cache
        if self._cache:
            self._add_to_cache(item)

        # Create access policy in multi-agent mode
        if self._mode == StoreMode.MULTI_AGENT and self._access_control:
            from .core.access_control import KnowledgeVisibility

            vis_map = {
                "private": KnowledgeVisibility.PRIVATE,
                "shared": KnowledgeVisibility.SHARED,
                "public": KnowledgeVisibility.PUBLIC,
            }
            self._access_control.create_policy(
                resource_id=item_id,
                resource_type="message",
                owner_id=agent_id,
                owner_org=self._organization_id,
                visibility=vis_map.get(visibility, KnowledgeVisibility.PRIVATE),
                sharing_groups=sharing_groups or [],
            )

        logger.debug(f"Added message: {item_id} (visibility={visibility})")
        return item

    def add_document(
        self,
        content: str,
        title: str | None = None,
        source: str | None = None,
        agent_id: str | None = None,
        visibility: str = "public",
        sharing_groups: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeItem:
        """Add a document/knowledge base article.

        Documents are typically shared knowledge that multiple agents can access.

        Args:
            content: Document content
            title: Document title
            source: Source URL or reference
            agent_id: Agent that added this
            visibility: Visibility level
            sharing_groups: Groups to share with
            metadata: Additional metadata

        Returns:
            KnowledgeItem representing the document
        """
        item_id = f"doc-{uuid.uuid4().hex[:12]}"

        if self._mode == StoreMode.SIMPLE:
            agent_id = self._default_agent.agent_id
        else:
            agent_id = agent_id or self._default_agent.agent_id

        item = KnowledgeItem(
            item_id=item_id,
            item_type="document",
            content=content,
            agent_id=agent_id,
            visibility=visibility,
            sharing_groups=sharing_groups or [],
            metadata={"title": title, "source": source, **(metadata or {})},
        )

        self._store_item(item)

        if self._vector_store and self._embedding:
            self._add_to_vector_store(item)

        return item

    # =========================================================================
    # Knowledge Retrieval
    # =========================================================================

    def get_context(
        self,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        agent_id: str | None = None,
        max_messages: int = 20,
        max_semantic_results: int = 5,
        include_shared: bool = True,
        include_public: bool = True,
    ) -> dict[str, Any]:
        """Get relevant context for a query.

        Combines recent messages, semantic search, and shared knowledge.

        Args:
            user_id: User identifier
            query: Query to find context for
            thread_id: Optional thread filter
            agent_id: Agent requesting context (multi-agent mode)
            max_messages: Max recent messages to include
            max_semantic_results: Max semantic search results
            include_shared: Include shared knowledge (multi-agent)
            include_public: Include public knowledge

        Returns:
            Dictionary containing:
            - recent_messages: Recent conversation messages
            - semantic_matches: Semantically similar content
            - shared_knowledge: Shared team knowledge
            - public_knowledge: Public knowledge base
        """
        if self._mode == StoreMode.SIMPLE:
            agent_id = self._default_agent.agent_id
        else:
            agent_id = agent_id or self._default_agent.agent_id

        context = {
            "recent_messages": [],
            "semantic_matches": [],
            "shared_knowledge": [],
            "public_knowledge": [],
            "metadata": {
                "user_id": user_id,
                "thread_id": thread_id,
                "query": query,
                "agent_id": agent_id,
            },
        }

        # Get recent messages
        if thread_id:
            context["recent_messages"] = self._get_recent_messages(
                user_id, thread_id, agent_id, max_messages
            )

        # Semantic search
        if self._vector_store:
            context["semantic_matches"] = self._semantic_search(
                query, agent_id, max_semantic_results, include_shared, include_public
            )

        return context

    def search(
        self,
        query: str,
        agent_id: str | None = None,
        item_type: str | None = None,
        k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeItem]:
        """Search the knowledge store.

        Uses semantic search if vector store is enabled,
        otherwise falls back to metadata-based search.

        Args:
            query: Search query
            agent_id: Agent performing search (for access control)
            item_type: Filter by item type ("message", "document")
            k: Number of results
            filter_metadata: Additional metadata filters

        Returns:
            List of matching KnowledgeItems
        """
        if self._mode == StoreMode.SIMPLE:
            agent_id = self._default_agent.agent_id
        else:
            agent_id = agent_id or self._default_agent.agent_id

        if self._vector_store:
            return self._semantic_search(query, agent_id, k)
        return self._metadata_search(query, agent_id, item_type, k, filter_metadata)

    def get_message(self, message_id: str, agent_id: str | None = None) -> KnowledgeItem | None:
        """Get a specific message by ID."""
        # Access control check in multi-agent mode
        if self._mode == StoreMode.MULTI_AGENT and self._access_control:
            from .core.access_control import Permission

            if not self._access_control.can_access(
                agent_id or self._default_agent.agent_id, message_id, Permission.READ
            ):
                logger.warning(f"Access denied to message {message_id}")
                return None

        return self._load_item(message_id)

    # =========================================================================
    # Knowledge Sharing (Multi-Agent)
    # =========================================================================

    def share_with_agent(
        self,
        item_id: str,
        target_agent_id: str,
        sharing_agent_id: str | None = None,
        can_reshare: bool = False,
    ) -> bool:
        """Share a knowledge item with another agent.

        Only available in multi-agent mode.

        Args:
            item_id: Item to share
            target_agent_id: Agent to share with
            sharing_agent_id: Agent doing the sharing
            can_reshare: Allow the target to reshare

        Returns:
            True if sharing succeeded
        """
        if self._mode != StoreMode.MULTI_AGENT:
            logger.warning("share_with_agent called in simple mode")
            return False

        if not self._access_control:
            return False

        from .core.access_control import Permission

        permissions = {Permission.READ}
        if can_reshare:
            permissions.add(Permission.SHARE)

        return self._access_control.share_with_agent(
            resource_id=item_id,
            target_agent_id=target_agent_id,
            sharing_agent_id=sharing_agent_id or self._default_agent.agent_id,
            permissions=permissions,
        )

    def share_with_group(
        self, item_id: str, group_name: str, sharing_agent_id: str | None = None
    ) -> bool:
        """Share a knowledge item with a group.

        Args:
            item_id: Item to share
            group_name: Group to share with
            sharing_agent_id: Agent doing the sharing

        Returns:
            True if sharing succeeded
        """
        if self._mode != StoreMode.MULTI_AGENT:
            return False

        if not self._access_control:
            return False

        return self._access_control.share_with_group(
            resource_id=item_id,
            group_name=group_name,
            sharing_agent_id=sharing_agent_id or self._default_agent.agent_id,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _store_item(self, item: KnowledgeItem) -> None:
        """Store item in database."""
        # Convert to Message format for existing database
        from .core.schemas import Message, MessageMetadata, MessageRole

        if item.item_type == "message":
            role = item.metadata.get("role", "user")
            message = Message(
                message_id=item.item_id,
                user_id=item.user_id or "",
                thread_id=item.thread_id or "",
                session_id=item.metadata.get("session_id", item.thread_id or ""),
                role=MessageRole(role),
                raw_text=item.content,
                metadata=MessageMetadata(**(item.metadata.get("enrichment", {}))),
                created_at=item.created_at,
            )
            self._db.insert_message(message)

    def _load_item(self, item_id: str) -> KnowledgeItem | None:
        """Load item from database."""
        message = self._db.get_message_by_id(item_id)
        if message:
            return KnowledgeItem(
                item_id=message.message_id,
                item_type="message",
                content=message.raw_text,
                user_id=message.user_id,
                thread_id=message.thread_id,
                metadata=message.metadata.to_dict() if message.metadata else {},
                created_at=message.created_at,
            )
        return None

    def _add_to_vector_store(self, item: KnowledgeItem) -> None:
        """Add item to vector store."""
        if not self._vector_store:
            return

        from .vectorstores import Document

        doc = Document(
            page_content=item.content,
            metadata={
                "item_id": item.item_id,
                "item_type": item.item_type,
                "user_id": item.user_id,
                "thread_id": item.thread_id,
                "agent_id": item.agent_id,
                "visibility": item.visibility,
                **item.metadata,
            },
            id=item.item_id,
        )

        self._vector_store.add_documents([doc])

    def _add_to_cache(self, item: KnowledgeItem) -> None:
        """Add item to cache."""
        if not self._cache or item.item_type != "message":
            return

        from .core.schemas import Message, MessageRole

        message = Message(
            message_id=item.item_id,
            user_id=item.user_id or "",
            thread_id=item.thread_id or "",
            session_id=item.metadata.get("session_id", ""),
            role=MessageRole(item.metadata.get("role", "user")),
            raw_text=item.content,
            created_at=item.created_at,
        )
        self._cache.add_message(message)

    def _get_recent_messages(
        self, user_id: str, thread_id: str, agent_id: str, limit: int
    ) -> list[KnowledgeItem]:
        """Get recent messages for a thread."""
        # Try cache first
        if self._cache:
            messages = self._cache.get_recent_messages(user_id, thread_id, limit)
            if messages:
                return [
                    KnowledgeItem(
                        item_id=m.message_id,
                        item_type="message",
                        content=m.raw_text,
                        user_id=m.user_id,
                        thread_id=m.thread_id,
                        metadata={"role": m.role.value},
                        created_at=m.created_at,
                    )
                    for m in messages
                ]

        # Fall back to database
        messages = self._db.fetch_recent_messages(user_id, thread_id, limit)
        return [
            KnowledgeItem(
                item_id=m.message_id,
                item_type="message",
                content=m.raw_text,
                user_id=m.user_id,
                thread_id=m.thread_id,
                metadata={"role": m.role.value},
                created_at=m.created_at,
            )
            for m in messages
        ]

    def _semantic_search(
        self,
        query: str,
        agent_id: str,
        k: int,
        include_shared: bool = True,
        include_public: bool = True,
    ) -> list[KnowledgeItem]:
        """Perform semantic search."""
        if not self._vector_store:
            return []

        # Build filter based on visibility
        results = self._vector_store.similarity_search(query, k=k * 2)  # Get extra for filtering

        items = []
        for doc in results:
            # Access control in multi-agent mode
            if self._mode == StoreMode.MULTI_AGENT and self._access_control:
                from .core.access_control import Permission

                item_id = doc.metadata.get("item_id", doc.id)
                if not self._access_control.can_access(agent_id, item_id, Permission.READ):
                    continue

            items.append(
                KnowledgeItem(
                    item_id=doc.metadata.get("item_id", doc.id or ""),
                    item_type=doc.metadata.get("item_type", "unknown"),
                    content=doc.page_content,
                    user_id=doc.metadata.get("user_id"),
                    thread_id=doc.metadata.get("thread_id"),
                    agent_id=doc.metadata.get("agent_id"),
                    visibility=doc.metadata.get("visibility", "private"),
                    metadata=doc.metadata,
                )
            )

            if len(items) >= k:
                break

        return items

    def _metadata_search(
        self,
        query: str,
        agent_id: str,
        item_type: str | None,
        k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[KnowledgeItem]:
        """Search by metadata (fallback when no vector store)."""
        # This would use database text search
        # For now, return empty - implement based on database capabilities
        return []

    # =========================================================================
    # Properties and Status
    # =========================================================================

    @property
    def mode(self) -> StoreMode:
        """Get current operation mode."""
        return self._mode

    @property
    def is_multi_agent(self) -> bool:
        """Check if running in multi-agent mode."""
        return self._mode == StoreMode.MULTI_AGENT

    @property
    def has_vector_search(self) -> bool:
        """Check if vector search is enabled."""
        return self._vector_store is not None

    @property
    def default_agent(self) -> AgentConfig:
        """Get the default agent."""
        return self._default_agent

    def health_check(self) -> dict[str, bool]:
        """Check health of all components."""
        status = {
            "database": True,
            "cache": self._cache is not None,
            "vector_store": self._vector_store is not None,
            "enrichment": self._enrichment_agent is not None,
            "access_control": self._access_control is not None,
        }

        if self._vector_store:
            status["vector_store_healthy"] = self._vector_store.health_check()

        return status

    def close(self) -> None:
        """Close all connections."""
        if self._db and hasattr(self._db, "close"):
            self._db.close()
        if self._cache and hasattr(self._cache, "close"):
            self._cache.close()
        if self._vector_store and hasattr(self._vector_store, "close"):
            self._vector_store.close()

        logger.info("KnowledgeStore closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Convenience aliases for quick start
SimpleKnowledgeStore = KnowledgeStore
