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
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Persistent queue (survives crashes/restarts)
import persistqueue

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
    DiskCacheManager,
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
from .utils import get_logger, generate_message_id, SecurityValidator

# Schemas
from .core.schemas import MessageMetadata

# Retrieval Query Agent (LLM-powered query analysis)
from .agents import RetrievalQueryAgent, QueryIntent

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
        llm_provider: Optional[str] = None,
        persistent_cache: bool = True,
        cache_dir: Optional[str] = None
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
            persistent_cache: If True (default), use disk-backed cache that
                survives restarts. If False, use in-memory cache.
            cache_dir: Optional directory for persistent cache files.

        Raises:
            ValueError: If no LLM provider can be initialized

        Example:
            >>> client = MindcoreClient(use_sqlite=True)
            >>> client = MindcoreClient(use_sqlite=True, llm_provider="llama_cpp")
            >>> client = MindcoreClient(use_sqlite=True, persistent_cache=True)
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

        # Initialize cache (persistent or in-memory)
        cache_config = self.config.get_cache_config()
        if persistent_cache:
            self.cache = DiskCacheManager(
                max_size=cache_config.get('max_size', 50),
                ttl_seconds=cache_config.get('ttl'),
                cache_dir=cache_dir
            )
            logger.info("Using persistent disk-backed cache (diskcache)")
        else:
            self.cache = CacheManager(
                max_size=cache_config.get('max_size', 50),
                ttl_seconds=cache_config.get('ttl')
            )
            logger.info("Using in-memory cache (cachetools)")

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

        # Retrieval query agent (LLM-powered, no keyword matching)
        self.retrieval_query_agent = RetrievalQueryAgent(
            llm_provider=self._llm_provider,
            temperature=0.2,  # Lower for consistency
            max_tokens=500
        )

        # Background enrichment queue (persistent - survives crashes/restarts)
        # Uses SQLite-backed queue from persistqueue library
        self._queue_path = Path(tempfile.gettempdir()) / "mindcore_enrichment_queue"
        self._queue_path.mkdir(parents=True, exist_ok=True)
        self._enrichment_queue = persistqueue.SQLiteQueue(
            str(self._queue_path),
            auto_commit=True,
            multithreading=True
        )
        self._enrichment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindcore_enrichment")
        self._enrichment_running = True
        self._enrichment_future = self._enrichment_executor.submit(self._enrichment_worker)

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

    def ingest_message_fast(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message immediately without blocking on enrichment.

        The message is stored immediately with empty metadata, cached for the
        current session, and queued for background enrichment. This is ideal
        for real-time applications where response latency is critical.

        Enrichment happens asynchronously:
        1. Message stored with empty metadata (~0ms delay)
        2. Message added to cache (available for current session context)
        3. Background worker enriches and updates database

        Args:
            message_dict: Dictionary containing:
                - user_id (str): User identifier
                - thread_id (str): Thread identifier
                - session_id (str): Session identifier
                - role (str): Message role (user, assistant, system, tool)
                - text (str): Message content
                - message_id (str, optional): Auto-generated if not provided

        Returns:
            Message: Message object with empty metadata (will be enriched in background).

        Raises:
            ValueError: If validation fails.
        """
        # Validate message
        is_valid, error_msg = SecurityValidator.validate_message_dict(message_dict)
        if not is_valid:
            logger.error(f"Message validation failed: {error_msg}")
            raise ValueError(f"Invalid message: {error_msg}")

        # Create message with empty metadata (no LLM call)
        from .core.schemas import MessageRole
        message = Message(
            message_id=message_dict.get('message_id') or generate_message_id(),
            user_id=message_dict['user_id'],
            thread_id=message_dict['thread_id'],
            session_id=message_dict['session_id'],
            role=MessageRole(message_dict['role']),
            raw_text=message_dict['text'],
            metadata=MessageMetadata(),  # Empty metadata
        )

        # Store in database immediately
        success = self.db.insert_message(message)
        if not success:
            logger.warning(f"Failed to store message {message.message_id} in database")

        # Add to cache for immediate session context
        self.cache.add_message(message)

        # Queue for background enrichment with ALL IDs preserved
        # Using persistent queue - survives crashes/restarts
        enrichment_task = {
            'message_id': message.message_id,
            'user_id': message.user_id,
            'thread_id': message.thread_id,
            'session_id': message.session_id,
            'role': message.role.value,
            'text': message.raw_text,
        }
        self._enrichment_queue.put(enrichment_task)

        logger.info(f"Message {message.message_id} ingested fast (enrichment queued)")
        return message

    def _enrichment_worker(self) -> None:
        """Background worker that enriches messages from the persistent queue."""
        logger.info("Background enrichment worker started (persistent queue)")

        while self._enrichment_running:
            try:
                # Wait for task with timeout (allows graceful shutdown)
                # persistqueue uses get() with block=True by default
                try:
                    task = self._enrichment_queue.get(block=True, timeout=1.0)
                except Exception:
                    # persistqueue raises Empty on timeout
                    continue

                if task is None:
                    continue

                message_id = task.get('message_id')
                if not message_id:
                    logger.warning("Enrichment worker: task missing message_id")
                    self._enrichment_queue.task_done()
                    continue

                # Check if already enriched in database
                existing = self.db.get_message_by_id(message_id)
                if existing and existing.metadata.is_enriched:
                    logger.debug(f"Message {message_id} already enriched, skipping")
                    self._enrichment_queue.task_done()
                    continue

                # Enrich message using task data (all IDs preserved from queue)
                try:
                    enriched_message = self.metadata_agent.process({
                        'message_id': task['message_id'],
                        'user_id': task['user_id'],
                        'thread_id': task['thread_id'],
                        'session_id': task['session_id'],
                        'role': task['role'],
                        'text': task['text'],
                    })

                    # Update database with enriched metadata
                    self.db.update_message_metadata(
                        message_id=message_id,
                        metadata=enriched_message.metadata
                    )

                    # Update cache with enriched message (preserves all IDs)
                    self.cache.update_message(enriched_message)

                    logger.debug(f"Background enrichment completed for {message_id}")

                except Exception as e:
                    logger.error(f"Background enrichment failed for {message_id}: {e}")
                    # Mark as failed in database
                    failed_metadata = MessageMetadata(
                        enrichment_failed=True,
                        enrichment_error=str(e)
                    )
                    self.db.update_message_metadata(message_id, failed_metadata)

                self._enrichment_queue.task_done()

            except Exception as e:
                logger.error(f"Enrichment worker error: {e}")

        logger.info("Background enrichment worker stopped")

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
        use_session_metadata: bool = True,
        use_llm_query_analysis: bool = True
    ) -> AssembledContext:
        """
        Get context using relevance-based search on enriched metadata.

        Uses LLM-powered query analysis to understand semantic INTENT, not keywords.
        This correctly handles cases like "I don't need a refund" by NOT searching
        for refund-related messages.

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
            use_llm_query_analysis: If True, use LLM to analyze query intent.
                                   If False, only use session metadata.

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

        # Initialize search criteria
        topics_to_search = []
        categories_to_search = []
        intent_to_search = None

        # Get recent context for LLM query analysis
        recent_context = None
        if thread_id:
            cached_messages = self.cache.get_recent_messages(user_id, thread_id, limit=5)
            if cached_messages:
                recent_context = "\n".join([
                    f"[{msg.role}]: {msg.raw_text[:200]}"
                    for msg in cached_messages[:5]
                ])

        # Use LLM to analyze query intent (no keyword matching!)
        if use_llm_query_analysis:
            query_intent = self.retrieval_query_agent.analyze_query(
                query=query,
                recent_context=recent_context
            )
            topics_to_search = query_intent.topics
            categories_to_search = query_intent.categories
            intent_to_search = query_intent.intent

            logger.debug(
                f"LLM query analysis: topics={topics_to_search}, "
                f"categories={categories_to_search}, intent={intent_to_search}"
            )

        # Merge with session metadata if available
        if use_session_metadata and thread_id:
            session_meta = self.cache.get_session_metadata(user_id, thread_id)
            session_topics = session_meta.get('topics', [])
            session_categories = session_meta.get('categories', [])
            session_intents = session_meta.get('intents', [])

            # Merge (LLM analysis takes priority, session adds context)
            topics_to_search = list(set(topics_to_search + session_topics))
            categories_to_search = list(set(categories_to_search + session_categories))
            if not intent_to_search and session_intents:
                intent_to_search = session_intents[0]

            logger.debug(f"After session merge: topics={topics_to_search[:5]}, categories={categories_to_search}")

        # Search by relevance using enriched metadata
        relevant_messages = self.db.search_by_relevance(
            user_id=user_id,
            topics=topics_to_search if topics_to_search else None,
            categories=categories_to_search if categories_to_search else None,
            intent=intent_to_search,
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
        # Stop background enrichment worker
        if hasattr(self, '_enrichment_running'):
            self._enrichment_running = False
            if hasattr(self, '_enrichment_future'):
                try:
                    self._enrichment_future.result(timeout=5.0)  # Wait for worker to stop
                except Exception:
                    pass
            if hasattr(self, '_enrichment_executor'):
                self._enrichment_executor.shutdown(wait=True)
            logger.info("Background enrichment worker stopped")

        # Close persistent queue if exists
        if hasattr(self, '_enrichment_queue') and hasattr(self._enrichment_queue, 'close'):
            self._enrichment_queue.close()

        # Close cache (DiskCacheManager needs explicit close)
        if hasattr(self, 'cache') and hasattr(self.cache, 'close'):
            self.cache.close()

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
    "RetrievalQueryAgent",
    "QueryIntent",
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
