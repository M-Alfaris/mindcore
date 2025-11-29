"""
Mindcore: Intelligent Memory and Context Management for AI Agents
===================================================================

Save 60-90% on token costs with intelligent memory management powered by
lightweight AI agents using local (llama.cpp) or cloud (OpenAI) LLMs.

Quick Start (Local LLM - No API Key Required):
---------------------------------------------
    # 1. Download a model first:
    #    mindcore download-model

    # 2. Set the model path:
    #    export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

    from mindcore import MindcoreClient

    client = MindcoreClient(use_sqlite=True)

    message = client.ingest_message({
        "user_id": "user123",
        "thread_id": "thread456",
        "session_id": "session789",
        "role": "user",
        "text": "Hello, how do I build AI agents?"
    })

    context = client.get_context(
        user_id="user123",
        thread_id="thread456",
        query="AI agent development"
    )

Features:
---------
- Local LLM inference with llama.cpp (CPU-optimized, no API costs)
- OpenAI fallback for reliability
- Automatic metadata enrichment (topics, sentiment, intent, etc.)
- Intelligent context assembly
- PostgreSQL or SQLite persistence
- In-memory caching for speed
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import threading
import atexit

# Version
__version__ = "0.2.0"
__author__ = "Mindcore Contributors"
__license__ = "MIT"

# Core classes
from .core import (
    ConfigLoader,
    DatabaseManager,
    SQLiteManager,
    CacheManager,
    Message,
    MessageMetadata,
    MessageRole,
    AssembledContext,
    ContextRequest,
    IngestRequest,
)

# AI Agents
from .agents import (
    BaseAgent,
    EnrichmentAgent as MetadataAgent,
    ContextAssemblerAgent as ContextAgent,
)

# LLM Providers
from .llm import (
    BaseLLMProvider,
    LLMResponse,
    LlamaCppProvider,
    OpenAIProvider,
    FallbackProvider,
    ProviderType,
    create_provider,
    get_provider_type,
)

# Utilities
from .utils import get_logger, generate_message_id, SecurityValidator, extract_query_hints

logger = get_logger(__name__)


def _get_sort_key(message: Message) -> float:
    """
    Get a sortable key for a message, handling timezone-naive and timezone-aware datetimes.

    Converts datetime to Unix timestamp (float) to avoid comparison issues between
    offset-naive and offset-aware datetimes.

    Args:
        message: Message object to get sort key for.

    Returns:
        Unix timestamp as float, or 0.0 if no valid datetime.
    """
    if message.created_at is None:
        return 0.0

    dt = message.created_at

    # Handle string datetimes (from SQLite)
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return 0.0

    # Ensure datetime is timezone-aware (assume UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


# Global instance with thread-safe initialization
_mindcore_instance: Optional['MindcoreClient'] = None
_instance_lock = threading.Lock()


class MindcoreClient:
    """
    Main Mindcore client for intelligent memory and context management.

    Uses LLM providers (llama.cpp or OpenAI) for:
    - Automatic metadata enrichment with MetadataAgent
    - Intelligent context assembly with ContextAgent
    - PostgreSQL or SQLite persistence with caching

    Usage:
        >>> from mindcore import MindcoreClient
        >>>
        >>> # Auto mode: uses llama.cpp if available, falls back to OpenAI
        >>> client = MindcoreClient(use_sqlite=True)
        >>>
        >>> # Force local LLM only
        >>> client = MindcoreClient(use_sqlite=True, llm_provider="llama_cpp")
        >>>
        >>> # Force OpenAI only
        >>> client = MindcoreClient(use_sqlite=True, llm_provider="openai")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_sqlite: bool = False,
        sqlite_path: str = "mindcore.db",
        llm_provider: Optional[str] = None
    ):
        """
        Initialize Mindcore client.

        Args:
            config_path: Optional path to config.yaml file.
            use_sqlite: If True, use SQLite instead of PostgreSQL.
            sqlite_path: Path to SQLite database file.
            llm_provider: LLM provider mode:
                - "auto" (default): llama.cpp primary, OpenAI fallback
                - "llama_cpp": Local inference only
                - "openai": Cloud inference only

        Raises:
            ValueError: If no LLM provider can be initialized

        Example:
            >>> client = MindcoreClient(use_sqlite=True)
            >>> client = MindcoreClient(use_sqlite=True, llm_provider="llama_cpp")
        """
        logger.info(f"Initializing Mindcore v{__version__}")

        # Load configuration
        self.config = ConfigLoader(config_path)
        self._use_sqlite = use_sqlite

        # Initialize database
        if use_sqlite:
            logger.info(f"Using SQLite database: {sqlite_path}")
            self.db = SQLiteManager(sqlite_path)
        else:
            db_config = self.config.get_database_config()
            self.db = DatabaseManager(db_config)
            self.db.initialize_schema()

        # Initialize cache
        cache_config = self.config.get_cache_config()
        self.cache = CacheManager(
            max_size=cache_config.get('max_size', 50),
            ttl_seconds=cache_config.get('ttl')
        )

        # Initialize LLM provider
        self._llm_provider = self._create_llm_provider(llm_provider)

        # Get generation defaults
        llm_config = self.config.get_llm_config()
        defaults = llm_config.get('defaults', {})

        # Initialize AI agents with the provider
        self.metadata_agent = MetadataAgent(
            llm_provider=self._llm_provider,
            temperature=defaults.get('temperature', 0.3),
            max_tokens=defaults.get('max_tokens_enrichment', 800)
        )

        self.context_agent = ContextAgent(
            llm_provider=self._llm_provider,
            temperature=defaults.get('temperature', 0.3),
            max_tokens=defaults.get('max_tokens_context', 1500)
        )

        db_type = "SQLite" if use_sqlite else "PostgreSQL"
        logger.info(
            f"Mindcore initialized with {db_type} and "
            f"{self._llm_provider.name} LLM provider"
        )

    def _create_llm_provider(self, provider_type_str: Optional[str]) -> BaseLLMProvider:
        """
        Create LLM provider based on configuration.

        Args:
            provider_type_str: Provider type string or None for config default

        Returns:
            Configured LLM provider

        Raises:
            ValueError: If no provider can be initialized
        """
        llm_config = self.config.get_llm_config()

        # Determine provider type
        if provider_type_str:
            provider_type = get_provider_type(provider_type_str)
        else:
            provider_type = get_provider_type(llm_config.get('provider', 'auto'))

        # Build provider configs
        llama_config = {}
        if llm_config['llama_cpp'].get('model_path'):
            llama_config = {
                'model_path': llm_config['llama_cpp']['model_path'],
                'n_ctx': llm_config['llama_cpp'].get('n_ctx', 4096),
                'n_threads': llm_config['llama_cpp'].get('n_threads'),
                'n_gpu_layers': llm_config['llama_cpp'].get('n_gpu_layers', 0),
                'chat_format': llm_config['llama_cpp'].get('chat_format'),
                'verbose': llm_config['llama_cpp'].get('verbose', False),
            }

        openai_config = {}
        if llm_config['openai'].get('api_key'):
            openai_config = {
                'api_key': llm_config['openai']['api_key'],
                'base_url': llm_config['openai'].get('base_url'),
                'model': llm_config['openai'].get('model', 'gpt-4o-mini'),
                'timeout': llm_config['openai'].get('timeout', 60),
                'max_retries': llm_config['openai'].get('max_retries', 3),
            }

        # Create provider
        try:
            return create_provider(
                provider_type=provider_type,
                llama_config=llama_config if llama_config else None,
                openai_config=openai_config if openai_config else None
            )
        except ValueError as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise ValueError(
                f"No LLM provider available. Configure either:\n"
                f"  - MINDCORE_LLAMA_MODEL_PATH for local inference, or\n"
                f"  - OPENAI_API_KEY for cloud inference\n"
                f"Original error: {e}"
            ) from e

    @property
    def llm_provider(self) -> BaseLLMProvider:
        """Get the active LLM provider."""
        return self._llm_provider

    @property
    def provider_name(self) -> str:
        """Get the name of the active LLM provider."""
        return self._llm_provider.name

    def ingest_message(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message with automatic metadata enrichment.

        The message will be:
        1. Validated for security
        2. Enriched with metadata (topics, sentiment, intent, etc.)
        3. Stored in database
        4. Cached for fast retrieval

        Args:
            message_dict: Dictionary containing:
                - user_id (str): User identifier
                - thread_id (str): Thread identifier
                - session_id (str): Session identifier
                - role (str): Message role (user, assistant, system, tool)
                - text (str): Message content
                - message_id (str, optional): Auto-generated if not provided

        Returns:
            Message: Enriched message object with metadata.

        Raises:
            ValueError: If validation fails.
        """
        # Validate message
        is_valid, error_msg = SecurityValidator.validate_message_dict(message_dict)
        if not is_valid:
            logger.error(f"Message validation failed: {error_msg}")
            raise ValueError(f"Invalid message: {error_msg}")

        # Enrich message with metadata
        message = self.metadata_agent.process(message_dict)

        # Store in database
        success = self.db.insert_message(message)
        if not success:
            logger.warning(f"Failed to store message {message.message_id} in database")

        # Add to cache
        self.cache.add_message(message)

        logger.info(f"Message {message.message_id} ingested successfully")
        return message

    def get_context(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        max_messages: int = 50
    ) -> AssembledContext:
        """
        Get intelligently assembled context for a query.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query or topic to find relevant context for.
            max_messages: Maximum messages to consider.

        Returns:
            AssembledContext with summarized context, key points, etc.

        Raises:
            ValueError: If validation fails.
        """
        # Validate query parameters
        is_valid, error_msg = SecurityValidator.validate_query_params(user_id, thread_id, query)
        if not is_valid:
            logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query parameters: {error_msg}")

        # Get messages from cache first
        cached_messages = self.cache.get_recent_messages(user_id, thread_id, limit=max_messages)

        # If cache doesn't have enough, fetch from database
        if len(cached_messages) < max_messages:
            db_messages = self.db.fetch_recent_messages(user_id, thread_id, limit=max_messages)

            # Merge messages, avoiding duplicates
            message_ids = {msg.message_id for msg in cached_messages}
            all_messages = list(cached_messages)

            for msg in db_messages:
                if msg.message_id not in message_ids:
                    all_messages.append(msg)
                    message_ids.add(msg.message_id)

            # Sort by created_at (most recent first) using timestamp to avoid timezone issues
            all_messages.sort(key=_get_sort_key, reverse=True)
            cached_messages = all_messages[:max_messages]

        logger.info(f"Retrieved {len(cached_messages)} messages for context assembly")

        # Assemble context
        context = self.context_agent.process(cached_messages, query)
        return context

    def get_relevant_context(
        self,
        user_id: str,
        query: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_messages: int = 20,
        min_importance: float = 0.3,
        include_current_thread: bool = True,
        use_session_metadata: bool = True
    ) -> AssembledContext:
        """
        Get context using relevance-based search on enriched metadata.

        This method uses fast SQL-based relevance scoring instead of fetching
        all messages. It uses the current session's topics/categories to find
        similar historical messages.

        Optimized for speed: ~50-100ms for relevance search vs ~4s for full LLM scan.

        Args:
            user_id: User identifier.
            query: Query to find relevant context for.
            thread_id: Optional thread filter (None = search all threads).
            session_id: Optional session filter.
            max_messages: Maximum relevant messages to retrieve.
            min_importance: Minimum importance score (0.0-1.0) to include.
            include_current_thread: If True and thread_id is set, always include
                                   recent messages from current thread.
            use_session_metadata: If True and thread_id is set, use session's
                                 aggregated topics/categories for search.

        Returns:
            AssembledContext with summarized context, key points, etc.

        Raises:
            ValueError: If validation fails.
        """
        # Validate query parameters
        is_valid, error_msg = SecurityValidator.validate_query_params(
            user_id, thread_id or "any", query
        )
        if not is_valid:
            logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query parameters: {error_msg}")

        # Get search criteria from session metadata or query hints
        topics_to_search = []
        categories_to_search = []
        intent_to_search = None

        # If we have a current thread, use its aggregated metadata
        if use_session_metadata and thread_id:
            session_meta = self.cache.get_session_metadata(user_id, thread_id)
            topics_to_search = session_meta.get('topics', [])
            categories_to_search = session_meta.get('categories', [])
            intents = session_meta.get('intents', [])
            if intents:
                intent_to_search = intents[0]  # Use most common intent

            logger.debug(f"Session metadata: topics={topics_to_search}, categories={categories_to_search}")

        # Also extract hints from the query itself
        hints = extract_query_hints(query)
        query_keywords = hints.get('keywords', [])
        query_intent = hints.get('intent_hint')
        query_categories = hints.get('category_hints', [])

        # Merge session metadata with query hints (query hints take priority for intent)
        all_topics = list(set(topics_to_search + query_keywords))
        all_categories = list(set(categories_to_search + query_categories))
        final_intent = query_intent or intent_to_search

        logger.debug(f"Search criteria: topics={all_topics[:5]}, categories={all_categories}, intent={final_intent}")

        # Search by relevance using enriched metadata
        relevant_messages = self.db.search_by_relevance(
            user_id=user_id,
            topics=all_topics if all_topics else None,
            categories=all_categories if all_categories else None,
            intent=final_intent,
            min_importance=min_importance,
            thread_id=None,  # Search across all threads for historical context
            session_id=session_id,
            limit=max_messages
        )

        # If current thread specified, also include recent cached messages
        if include_current_thread and thread_id:
            cached_messages = self.cache.get_recent_messages(
                user_id, thread_id, limit=10
            )
            # Merge, avoiding duplicates
            relevant_ids = {msg.message_id for msg in relevant_messages}
            for msg in cached_messages:
                if msg.message_id not in relevant_ids:
                    relevant_messages.append(msg)
                    relevant_ids.add(msg.message_id)

        # Sort by relevance (already sorted by DB) then recency
        relevant_messages.sort(key=_get_sort_key, reverse=True)
        relevant_messages = relevant_messages[:max_messages]

        logger.info(f"Retrieved {len(relevant_messages)} relevant messages for context assembly")

        # Assemble context (LLM summarizes pre-filtered messages)
        context = self.context_agent.process(relevant_messages, query)
        return context

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a single message by ID."""
        return self.db.get_message_by_id(message_id)

    def clear_cache(self, user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        """Clear message cache."""
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if hasattr(self, '_llm_provider') and self._llm_provider:
            self._llm_provider.close()
        self.db.close()
        logger.info("Mindcore client closed")


# Convenience alias
Mindcore = MindcoreClient


def initialize(
    config_path: Optional[str] = None,
    use_sqlite: bool = False,
    llm_provider: Optional[str] = None
) -> MindcoreClient:
    """
    Initialize global Mindcore instance (thread-safe singleton).

    Args:
        config_path: Optional path to config.yaml
        use_sqlite: Use SQLite instead of PostgreSQL
        llm_provider: LLM provider mode ("auto", "llama_cpp", "openai")

    Returns:
        MindcoreClient instance
    """
    global _mindcore_instance

    if _mindcore_instance is None:
        with _instance_lock:
            if _mindcore_instance is None:
                _mindcore_instance = MindcoreClient(
                    config_path=config_path,
                    use_sqlite=use_sqlite,
                    llm_provider=llm_provider
                )

    return _mindcore_instance


def get_client() -> MindcoreClient:
    """Get the global Mindcore client instance."""
    global _mindcore_instance

    if _mindcore_instance is None:
        with _instance_lock:
            if _mindcore_instance is None:
                _mindcore_instance = MindcoreClient()

    return _mindcore_instance


def _cleanup_on_exit():
    """Cleanup resources on program exit."""
    global _mindcore_instance
    if _mindcore_instance is not None:
        try:
            _mindcore_instance.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


atexit.register(_cleanup_on_exit)


# Async client (lazy import to avoid requiring async deps by default)
def get_async_client():
    """
    Get AsyncMindcoreClient class.

    Requires async dependencies: pip install mindcore[async]

    Returns:
        AsyncMindcoreClient class

    Example:
        AsyncMindcoreClient = get_async_client()
        async with AsyncMindcoreClient(use_sqlite=True) as client:
            message = await client.ingest_message(message_dict)
    """
    from .async_client import AsyncMindcoreClient
    return AsyncMindcoreClient


# Public API
__all__ = [
    # Version
    "__version__",

    # Main client
    "MindcoreClient",
    "Mindcore",
    "initialize",
    "get_client",
    "get_async_client",

    # AI Agents
    "MetadataAgent",
    "ContextAgent",
    "BaseAgent",

    # LLM Providers
    "BaseLLMProvider",
    "LLMResponse",
    "LlamaCppProvider",
    "OpenAIProvider",
    "FallbackProvider",
    "ProviderType",
    "create_provider",
    "get_provider_type",

    # Core data structures
    "Message",
    "MessageMetadata",
    "MessageRole",
    "AssembledContext",
    "ContextRequest",
    "IngestRequest",

    # Core managers
    "ConfigLoader",
    "DatabaseManager",
    "SQLiteManager",
    "CacheManager",
]
