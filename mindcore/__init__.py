"""
Mindcore: Intelligent Memory and Context Management for AI Agents
===================================================================

Save 60-90% on token costs with intelligent memory management powered by
lightweight AI agents using local (llama.cpp) or cloud (OpenAI) LLMs.

Quick Start:
------------
    from mindcore import MindcoreClient

    client = MindcoreClient(use_sqlite=True)

    # Ingest message (fast async enrichment)
    message = client.ingest(
        user_id="user123",
        thread_id="thread456",
        session_id="session789",
        role="user",
        text="Hello, how do I build AI agents?"
    )

    # Get context (single LLM call with tools)
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
- Intelligent context assembly with tool calling
- VocabularyManager for controlled, extensible vocabulary
- PostgreSQL or SQLite persistence
- In-memory and disk-backed caching
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import threading
import atexit
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

# Persistent queue (survives crashes/restarts)
import persistqueue

# Version
__version__ = "0.3.0"
__author__ = "Mindcore Contributors"
__license__ = "MIT"

# Core classes
from .core import (
    ConfigLoader,
    DatabaseManager,
    SQLiteManager,
    CacheManager,
    DiskCacheManager,
    PreferencesManager,
    Message,
    MessageMetadata,
    MessageRole,
    AssembledContext,
    ContextRequest,
    IngestRequest,
    ThreadSummary,
    UserPreferences,
    # VocabularyManager
    VocabularyManager,
    VocabularySource,
    Intent,
    Sentiment,
    CommunicationStyle,
    EntityType,
    get_vocabulary,
)

# AI Agents
from .agents import (
    BaseAgent,
    EnrichmentAgent as MetadataAgent,
    SummarizationAgent,
    SmartContextAgent,
    ContextTools,
    TrivialMessageDetector,
    TrivialCategory,
    get_trivial_detector,
)

# Worker monitoring
from .core.worker_monitor import (
    WorkerMonitor,
    WorkerMetrics,
    WorkerStatus,
    get_worker_monitor,
)

# Adaptive preferences
from .core.adaptive_preferences import (
    AdaptivePreferencesLearner,
    AdaptiveConfig,
    get_adaptive_learner,
)

# Retention policy
from .core.retention_policy import (
    RetentionPolicyManager,
    RetentionConfig,
    MemoryTier,
    get_retention_policy,
)

# Cache invalidation
from .core.cache_invalidation import (
    CacheInvalidationManager,
    InvalidationReason,
    get_cache_invalidation,
)

# Multi-agent support
from .core.multi_agent import (
    MultiAgentConfig,
    MultiAgentManager,
    MemorySharingMode,
    AgentVisibility,
    AgentProfile,
    get_multi_agent_manager,
    configure_multi_agent,
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

logger = get_logger(__name__)


def _get_sort_key(message: Message) -> float:
    """Get a sortable key for a message, handling timezone issues."""
    if message.created_at is None:
        return 0.0

    dt = message.created_at

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return 0.0

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


# Global instance with thread-safe initialization
_mindcore_instance: Optional['MindcoreClient'] = None
_instance_lock = threading.Lock()


class MindcoreClient:
    """
    Main Mindcore client for intelligent memory and context management.

    Simplified API with:
    - Single ingestion flow (async background enrichment)
    - Single context retrieval (SmartContextAgent with tools)
    - VocabularyManager for controlled vocabulary

    Usage:
        >>> from mindcore import MindcoreClient
        >>>
        >>> client = MindcoreClient(use_sqlite=True)
        >>>
        >>> # Ingest message (instant, enriched in background)
        >>> message = client.ingest(
        ...     user_id="user123",
        ...     thread_id="thread456",
        ...     session_id="session789",
        ...     role="user",
        ...     text="How do I track my order?"
        ... )
        >>>
        >>> # Get context (single LLM call with tools)
        >>> context = client.get_context(
        ...     user_id="user123",
        ...     thread_id="thread456",
        ...     query="order tracking"
        ... )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_sqlite: bool = False,
        sqlite_path: str = "mindcore.db",
        llm_provider: Optional[str] = None,
        persistent_cache: bool = True,
        cache_dir: Optional[str] = None,
        vocabulary: Optional[VocabularyManager] = None,
        multi_agent_config: Optional[MultiAgentConfig] = None
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
            persistent_cache: If True (default), use disk-backed cache.
            cache_dir: Optional directory for persistent cache files.
            vocabulary: Optional VocabularyManager. If None, uses global instance.
            multi_agent_config: Optional MultiAgentConfig for shared memory mode.
                When enabled, agent_id is required for ingest() and get_context().

        Raises:
            ValueError: If no LLM provider can be initialized

        Example (multi-agent mode):
            >>> from mindcore import MindcoreClient, MultiAgentConfig, MemorySharingMode
            >>>
            >>> config = MultiAgentConfig(
            ...     enabled=True,
            ...     mode=MemorySharingMode.SHARED,
            ...     require_agent_id=True
            ... )
            >>> client = MindcoreClient(use_sqlite=True, multi_agent_config=config)
            >>>
            >>> # Register agents
            >>> client.register_agent("support_bot", "Support Agent", groups=["support"])
            >>> client.register_agent("sales_bot", "Sales Agent", groups=["sales"])
            >>>
            >>> # Ingest with agent_id
            >>> client.ingest(..., agent_id="support_bot")
            >>>
            >>> # Get context (can filter by agent_ids)
            >>> client.get_context(..., agent_id="support_bot")
        """
        logger.info(f"Initializing Mindcore v{__version__}")

        # Load configuration
        self.config = ConfigLoader(config_path)
        self._use_sqlite = use_sqlite

        # Initialize vocabulary (central to the system)
        self.vocabulary = vocabulary or get_vocabulary()

        # Initialize multi-agent manager
        self._multi_agent_config = multi_agent_config or MultiAgentConfig()
        self._multi_agent_manager = MultiAgentManager(self._multi_agent_config)
        if self._multi_agent_config.enabled:
            logger.info(
                f"Multi-agent mode enabled: {self._multi_agent_config.mode.value}, "
                f"require_agent_id={self._multi_agent_config.require_agent_id}"
            )

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

        # Initialize enrichment agent (for background enrichment)
        self._metadata_agent = MetadataAgent(
            llm_provider=self._llm_provider,
            temperature=defaults.get('temperature', 0.3),
            max_tokens=defaults.get('max_tokens_enrichment', 800),
            vocabulary=self.vocabulary
        )

        # SmartContextAgent is the PRIMARY context retrieval method
        self._smart_context_agent = None  # Lazy initialized

        # Background enrichment queue (persistent - survives crashes/restarts)
        self._queue_path = Path(tempfile.gettempdir()) / "mindcore_enrichment_queue"
        self._queue_path.mkdir(parents=True, exist_ok=True)
        self._enrichment_queue = persistqueue.SQLiteQueue(
            str(self._queue_path),
            auto_commit=True,
            multithreading=True
        )

        # Trivial message detector (skip LLM for greetings, fillers, etc.)
        self._trivial_detector = get_trivial_detector()

        # Worker monitoring
        self._worker_monitor = get_worker_monitor()
        self._worker_metrics = self._worker_monitor.register_worker("enrichment")

        # Preferences manager (for adaptive preferences)
        self._preferences_manager = PreferencesManager(self.db)

        # Adaptive preferences learner
        from .core.adaptive_preferences import AdaptivePreferencesLearner
        self._adaptive_learner = AdaptivePreferencesLearner(
            self._preferences_manager, self.db
        )

        # Retention policy manager
        self._retention_policy = RetentionPolicyManager(
            self.db,
            summarization_agent=None,  # Set later if needed
            config=RetentionConfig()
        )

        # Cache invalidation manager
        self._cache_invalidation = get_cache_invalidation()
        self._cache_invalidation.register_cache("messages", self.cache)

        self._enrichment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindcore_enrichment")
        self._enrichment_running = True
        self._enrichment_future = self._enrichment_executor.submit(self._enrichment_worker)

        db_type = "SQLite" if use_sqlite else "PostgreSQL"
        logger.info(
            f"Mindcore initialized with {db_type} and "
            f"{self._llm_provider.name} LLM provider"
        )

    def _create_llm_provider(self, provider_type_str: Optional[str]) -> BaseLLMProvider:
        """Create LLM provider based on configuration."""
        llm_config = self.config.get_llm_config()

        if provider_type_str:
            provider_type = get_provider_type(provider_type_str)
        else:
            provider_type = get_provider_type(llm_config.get('provider', 'auto'))

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

    # -------------------------------------------------------------------------
    # PRIMARY API: Ingestion
    # -------------------------------------------------------------------------

    def ingest(
        self,
        user_id: str,
        thread_id: str,
        session_id: str,
        role: str,
        text: str,
        message_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        visibility: Optional[str] = None,
        sharing_groups: Optional[List[str]] = None,
        **metadata
    ) -> Message:
        """
        Ingest a message with background enrichment.

        This is the PRIMARY ingestion method. Messages are stored immediately
        and enriched asynchronously in the background.

        Benefits:
        - Zero LLM latency on ingestion
        - Message immediately available for context
        - Persistent queue survives crashes
        - Enrichment happens automatically

        Args:
            user_id: User identifier
            thread_id: Thread/conversation identifier
            session_id: Session identifier
            role: Message role (user, assistant, system, tool)
            text: Message content
            message_id: Optional custom message ID
            agent_id: Agent identifier (required if multi-agent mode enabled with require_agent_id)
            visibility: Message visibility ("private", "shared", "public")
            sharing_groups: Groups that can access this message (for "shared" visibility)
            **metadata: Additional metadata to attach

        Returns:
            Message object (enrichment happens in background)

        Raises:
            ValueError: If validation fails or agent_id required but not provided

        Example:
            >>> # Single-agent mode
            >>> message = client.ingest(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     session_id="session789",
            ...     role="user",
            ...     text="How do I track my order #12345?"
            ... )

            >>> # Multi-agent mode
            >>> message = client.ingest(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     session_id="session789",
            ...     role="assistant",
            ...     text="I can help with that!",
            ...     agent_id="support_bot",
            ...     visibility="shared",
            ...     sharing_groups=["support", "orders"]
            ... )
        """
        # Validate agent_id in multi-agent mode
        is_valid, error_msg = self._multi_agent_manager.validate_agent_id(agent_id, for_write=True)
        if not is_valid:
            raise ValueError(error_msg)
        message_dict = {
            'user_id': user_id,
            'thread_id': thread_id,
            'session_id': session_id,
            'role': role,
            'text': text,
            'message_id': message_id,
        }

        # Validate message
        is_valid, error_msg = SecurityValidator.validate_message_dict(message_dict)
        if not is_valid:
            logger.error(f"Message validation failed: {error_msg}")
            raise ValueError(f"Invalid message: {error_msg}")

        # Resolve visibility and sharing groups for multi-agent mode
        resolved_visibility = visibility or self._multi_agent_manager.get_default_visibility(agent_id)
        resolved_groups = sharing_groups or self._multi_agent_manager.get_default_sharing_groups(agent_id)

        # Create message with empty metadata (no LLM call)
        message = Message(
            message_id=message_dict.get('message_id') or generate_message_id(),
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id,
            role=MessageRole(role),
            raw_text=text,
            metadata=MessageMetadata(),  # Empty metadata
            agent_id=agent_id,
            visibility=resolved_visibility,
            sharing_groups=resolved_groups,
        )

        # Store in database immediately
        success = self.db.insert_message(message)
        if not success:
            logger.warning(f"Failed to store message {message.message_id} in database")

        # Add to cache for immediate session context
        self.cache.add_message(message)

        # Queue for background enrichment
        enrichment_task = {
            'message_id': message.message_id,
            'user_id': message.user_id,
            'thread_id': message.thread_id,
            'session_id': message.session_id,
            'role': message.role.value,
            'text': message.raw_text,
        }
        self._enrichment_queue.put(enrichment_task)

        logger.debug(f"Message {message.message_id} ingested (enrichment queued)")
        return message

    def ingest_message(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message (legacy API).

        Deprecated: Use ingest() instead for cleaner API.
        """
        return self.ingest(
            user_id=message_dict['user_id'],
            thread_id=message_dict['thread_id'],
            session_id=message_dict['session_id'],
            role=message_dict['role'],
            text=message_dict['text'],
            message_id=message_dict.get('message_id')
        )

    def ingest_message_fast(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message (legacy API).

        Deprecated: Use ingest() instead. All ingestion is now fast by default.
        """
        return self.ingest_message(message_dict)

    def _enrichment_worker(self) -> None:
        """Background worker that enriches messages from the persistent queue."""
        logger.info("Background enrichment worker started (persistent queue)")
        self._worker_metrics.status = WorkerStatus.IDLE

        while self._enrichment_running:
            try:
                # Update queue size for monitoring
                try:
                    queue_size = self._enrichment_queue.qsize()
                    self._worker_monitor.update_queue_size("enrichment", queue_size)
                except Exception:
                    pass

                try:
                    task = self._enrichment_queue.get(block=True, timeout=1.0)
                except Exception:
                    continue

                if task is None:
                    continue

                message_id = task.get('message_id')
                if not message_id:
                    logger.warning("Enrichment worker: task missing message_id")
                    self._enrichment_queue.task_done()
                    continue

                # Check if already enriched
                existing = self.db.get_message_by_id(message_id)
                if existing and existing.metadata.is_enriched:
                    logger.debug(f"Message {message_id} already enriched, skipping")
                    self._enrichment_queue.task_done()
                    continue

                # Mark as processing
                self._worker_metrics.status = WorkerStatus.PROCESSING
                start_time = time.time()

                try:
                    text = task.get('text', '')

                    # Check if trivial message (skip LLM call)
                    trivial_result = self._trivial_detector.detect(text)

                    if trivial_result.is_trivial:
                        # Auto-enrich without LLM
                        enriched_message = self._trivial_detector.auto_enrich(
                            text=text,
                            user_id=task['user_id'],
                            thread_id=task['thread_id'],
                            session_id=task['session_id'],
                            role=task['role'],
                            message_id=message_id
                        )
                        self._worker_metrics.record_trivial_skip()
                        logger.debug(
                            f"Trivial message {message_id} auto-enriched "
                            f"(category={trivial_result.category.value})"
                        )
                    else:
                        # Full LLM enrichment
                        enriched_message = self._metadata_agent.process({
                            'message_id': task['message_id'],
                            'user_id': task['user_id'],
                            'thread_id': task['thread_id'],
                            'session_id': task['session_id'],
                            'role': task['role'],
                            'text': text,
                        })

                    # Update database
                    self.db.update_message_metadata(
                        message_id=message_id,
                        metadata=enriched_message.metadata
                    )

                    # Notify cache invalidation manager (updates cache and indexes)
                    self._cache_invalidation.notify_enrichment_complete(enriched_message)

                    # Learn from message metadata (adaptive preferences)
                    if not trivial_result.is_trivial:
                        self._adaptive_learner.process_message_metadata(
                            user_id=enriched_message.user_id,
                            metadata=enriched_message.metadata
                        )

                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    self._worker_metrics.record_processing(duration_ms)
                    logger.debug(f"Background enrichment completed for {message_id}")

                except Exception as e:
                    logger.error(f"Background enrichment failed for {message_id}: {e}")
                    self._worker_metrics.record_error(str(e))
                    failed_metadata = MessageMetadata(
                        enrichment_failed=True,
                        enrichment_error=str(e)
                    )
                    self.db.update_message_metadata(message_id, failed_metadata)

                self._enrichment_queue.task_done()
                self._worker_metrics.status = WorkerStatus.IDLE

            except Exception as e:
                logger.error(f"Enrichment worker error: {e}")
                self._worker_metrics.record_error(str(e))
                self._worker_metrics.status = WorkerStatus.ERROR

        self._worker_metrics.status = WorkerStatus.STOPPED
        logger.info("Background enrichment worker stopped")

    # -------------------------------------------------------------------------
    # PRIMARY API: Context Retrieval
    # -------------------------------------------------------------------------

    def get_context(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        additional_context: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        include_shared: bool = True,
        include_public: bool = True
    ) -> AssembledContext:
        """
        Get intelligently assembled context for a query.

        This is the PRIMARY context retrieval method. Uses SmartContextAgent
        with tool calling for efficient, intelligent context assembly.

        The agent decides which tools to use based on the query:
        - get_recent_messages: Current conversation context
        - search_history: Historical messages by topic
        - get_session_metadata: Session aggregated info
        - get_historical_summaries: Past conversation summaries
        - get_user_preferences: User preferences for personalization
        - lookup_external_data: External system data (orders, billing)

        Args:
            user_id: User identifier
            thread_id: Thread identifier
            query: Query to find relevant context for
            additional_context: Optional additional context to provide
            agent_id: Agent making the query (required if multi-agent mode with require_agent_id)
            agent_ids: Optional list of specific agent IDs to include context from
            include_shared: Include shared content from same groups (multi-agent mode)
            include_public: Include public content from all agents (multi-agent mode)

        Returns:
            AssembledContext with summarized context, key points, etc.

        Raises:
            ValueError: If validation fails or agent_id required but not provided

        Example:
            >>> # Single-agent mode
            >>> context = client.get_context(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     query="What did we discuss about billing?"
            ... )

            >>> # Multi-agent mode
            >>> context = client.get_context(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     query="What did we discuss about billing?",
            ...     agent_id="support_bot",
            ...     include_shared=True
            ... )
            >>> print(context.assembled_context)
        """
        # Validate agent_id in multi-agent mode
        is_valid, error_msg = self._multi_agent_manager.validate_agent_id(agent_id, for_write=False)
        if not is_valid:
            raise ValueError(error_msg)

        # Validate query parameters
        is_valid, error_msg = SecurityValidator.validate_query_params(user_id, thread_id, query)
        if not is_valid:
            logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query parameters: {error_msg}")

        # Build access filter for multi-agent mode
        access_filter = self._multi_agent_manager.get_access_filter(
            agent_id=agent_id,
            include_own=True,
            include_shared=include_shared,
            include_public=include_public
        )

        # If specific agent_ids provided, add them to filter
        if agent_ids:
            access_filter["specific_agent_ids"] = agent_ids

        agent = self._get_smart_context_agent()
        context = agent.process(
            query=query,
            user_id=user_id,
            thread_id=thread_id,
            additional_context=additional_context
        )

        logger.info(
            f"Context assembled: source={context.metadata.get('context_source', 'unknown')}, "
            f"confidence={context.metadata.get('confidence', 'unknown')}"
        )
        return context

    def _get_smart_context_agent(self) -> SmartContextAgent:
        """Get or create the SmartContextAgent (lazy initialization)."""
        if self._smart_context_agent is not None:
            return self._smart_context_agent

        # Create callbacks for the agent's tools
        def get_recent_messages(user_id: str, thread_id: str, limit: int) -> List[Message]:
            try:
                return self.cache.get_recent_messages(user_id, thread_id, limit)
            except Exception as e:
                logger.error(f"Error fetching recent messages: {e}")
                return []

        def search_history(
            user_id: str,
            thread_id: Optional[str],
            topics: List[str],
            categories: List[str],
            intent: Optional[str],
            limit: int
        ) -> List[Message]:
            try:
                return self.db.search_by_relevance(
                    user_id=user_id,
                    topics=topics if topics else None,
                    categories=categories if categories else None,
                    intent=intent,
                    thread_id=thread_id,
                    limit=limit
                )
            except Exception as e:
                logger.error(f"Error searching history: {e}")
                return []

        def get_session_metadata(user_id: str, thread_id: str) -> Dict[str, Any]:
            try:
                return self.cache.get_session_metadata(user_id, thread_id)
            except Exception as e:
                logger.error(f"Error fetching session metadata: {e}")
                return {"topics": [], "categories": [], "intents": [], "message_count": 0}

        tools = ContextTools(
            get_recent_messages=get_recent_messages,
            search_history=search_history,
            get_session_metadata=get_session_metadata
        )

        llm_config = self.config.get_llm_config()
        defaults = llm_config.get('defaults', {})

        self._smart_context_agent = SmartContextAgent(
            llm_provider=self._llm_provider,
            context_tools=tools,
            temperature=0.2,
            max_tokens=defaults.get('max_tokens_context', 1500),
            vocabulary=self.vocabulary,
            max_tool_rounds=3
        )

        logger.info("SmartContextAgent initialized with database/cache tools")
        return self._smart_context_agent

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_worker_health(self) -> Dict[str, Any]:
        """
        Get health and metrics for background workers.

        Returns:
            Health check result with worker status, metrics, and issues

        Example:
            >>> health = client.get_worker_health()
            >>> print(health['healthy'])
            True
            >>> print(health['workers']['enrichment']['metrics']['processed_count'])
            42
        """
        return self._worker_monitor.get_health()

    def get_enrichment_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the enrichment worker.

        Returns:
            Dictionary with enrichment worker metrics including:
            - processed_count: Total messages enriched
            - trivial_skip_count: Messages skipped (trivial detection)
            - error_count: Failed enrichments
            - avg_processing_time_ms: Average processing time
            - savings_from_trivial: LLM calls saved

        Example:
            >>> metrics = client.get_enrichment_metrics()
            >>> print(f"Processed: {metrics['processed_count']}")
            >>> print(f"LLM calls saved: {metrics['savings_from_trivial']['llm_calls_saved']}")
        """
        return self._worker_metrics.to_dict()

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a single message by ID."""
        return self.db.get_message_by_id(message_id)

    def get_recent_messages(
        self,
        user_id: str,
        thread_id: str,
        limit: int = 20
    ) -> List[Message]:
        """Get recent messages from cache/database."""
        cached = self.cache.get_recent_messages(user_id, thread_id, limit)
        if len(cached) >= limit:
            return cached[:limit]

        # Fetch from database if cache incomplete
        db_messages = self.db.fetch_recent_messages(user_id, thread_id, limit)
        message_ids = {msg.message_id for msg in cached}
        all_messages = list(cached)

        for msg in db_messages:
            if msg.message_id not in message_ids:
                all_messages.append(msg)

        all_messages.sort(key=_get_sort_key, reverse=True)
        return all_messages[:limit]

    def clear_cache(self, user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        """Clear message cache."""
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    def refresh_vocabulary(self) -> None:
        """Refresh vocabulary from external sources and update agents."""
        self.vocabulary.refresh_from_external()
        if self._smart_context_agent:
            self._smart_context_agent.refresh_vocabulary()
        if self._metadata_agent:
            self._metadata_agent.refresh_vocabulary()
        logger.info("Vocabulary refreshed across all agents")

    # -------------------------------------------------------------------------
    # Retention & Memory Management
    # -------------------------------------------------------------------------

    def run_tier_migration(
        self,
        user_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run tier migration for memory management.

        Moves messages between tiers (short-term → mid-term → long-term)
        based on age and importance.

        Args:
            user_id: Optional user to migrate (all users if None)
            dry_run: If True, report what would be done without migrating

        Returns:
            Migration result with counts

        Example:
            >>> result = client.run_tier_migration()
            >>> print(f"Migrated {result['messages_migrated']} messages")
        """
        result = self._retention_policy.run_migration(user_id, dry_run)
        return {
            "messages_migrated": result.messages_migrated,
            "threads_summarized": result.threads_summarized,
            "messages_deleted": result.messages_deleted,
            "errors": result.errors,
        }

    def get_decayed_importance(self, message: Message) -> float:
        """
        Get importance score with time-based decay.

        More recent messages have higher importance.

        Args:
            message: Message to calculate importance for

        Returns:
            Decayed importance (0.0 to 1.0)
        """
        return self._retention_policy.get_decayed_importance(message)

    def get_context_window(
        self,
        user_id: str,
        thread_id: str,
        max_messages: int = 50,
        min_importance: float = 0.1
    ) -> List[Message]:
        """
        Get optimized context window for a conversation.

        Selects messages based on recency, importance, and token budget.

        Args:
            user_id: User identifier
            thread_id: Thread identifier
            max_messages: Maximum messages to include
            min_importance: Minimum decayed importance to include

        Returns:
            List of messages optimized for context window
        """
        return self._retention_policy.get_context_window(
            user_id, thread_id, max_messages, min_importance=min_importance
        )

    # -------------------------------------------------------------------------
    # Multi-Agent Management
    # -------------------------------------------------------------------------

    @property
    def multi_agent_enabled(self) -> bool:
        """Check if multi-agent mode is enabled."""
        return self._multi_agent_manager.is_enabled

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: Optional[str] = None,
        sharing_groups: Optional[List[str]] = None,
        default_visibility: str = "private",
        can_read_public: bool = True,
        can_write_public: bool = False,
        **metadata
    ) -> AgentProfile:
        """
        Register an agent for multi-agent memory sharing.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            description: Optional agent description
            sharing_groups: Groups this agent belongs to for shared memory
            default_visibility: Default visibility for this agent's messages
                ("private", "shared", "public")
            can_read_public: Whether this agent can read public content
            can_write_public: Whether this agent can create public content
            **metadata: Additional agent metadata

        Returns:
            AgentProfile for the registered agent

        Example:
            >>> client.register_agent(
            ...     agent_id="support_bot",
            ...     name="Customer Support Agent",
            ...     sharing_groups=["support", "general"],
            ...     default_visibility="shared"
            ... )
        """
        visibility_enum = AgentVisibility(default_visibility)
        return self._multi_agent_manager.register_agent(
            agent_id=agent_id,
            name=name,
            description=description,
            sharing_groups=sharing_groups,
            default_visibility=visibility_enum,
            can_read_public=can_read_public,
            can_write_public=can_write_public,
            **metadata
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        return self._multi_agent_manager.unregister_agent(agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """
        Get an agent's profile.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentProfile or None if not found
        """
        return self._multi_agent_manager.get_agent(agent_id)

    def list_agents(self) -> List[AgentProfile]:
        """
        List all registered agents.

        Returns:
            List of AgentProfile objects
        """
        return self._multi_agent_manager.list_agents()

    def get_multi_agent_stats(self) -> Dict[str, Any]:
        """
        Get multi-agent statistics.

        Returns:
            Dictionary with multi-agent mode stats including:
            - enabled: Whether multi-agent mode is active
            - mode: Current sharing mode
            - registered_agents: Number of registered agents
            - sharing_groups: Number of sharing groups
            - agents: List of agent details
        """
        return self._multi_agent_manager.get_stats()

    # -------------------------------------------------------------------------
    # Adaptive Preferences
    # -------------------------------------------------------------------------

    def apply_learned_preferences(self, user_id: str) -> List[tuple]:
        """
        Apply learned preferences for a user.

        Updates user preferences based on patterns observed in their messages.

        Args:
            user_id: User identifier

        Returns:
            List of (field, action, value) tuples for updates applied
        """
        return self._adaptive_learner.apply_updates(user_id)

    def get_preference_signals(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of accumulated preference signals for a user.

        Args:
            user_id: User identifier

        Returns:
            Summary with signal counts and top signals by type
        """
        return self._adaptive_learner.get_signal_summary(user_id)

    def get_user_preferences(self, user_id: str):
        """
        Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences object
        """
        return self._preferences_manager.get_preferences(user_id)

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including invalidation metrics.

        Returns:
            Dictionary with cache and invalidation stats
        """
        cache_stats = self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}
        invalidation_stats = self._cache_invalidation.get_stats()

        return {
            "cache": cache_stats,
            "invalidation": invalidation_stats,
        }

    def invalidate_cache_by_topic(self, topic: str) -> int:
        """
        Invalidate all cached data related to a topic.

        Args:
            topic: Topic to invalidate

        Returns:
            Number of threads invalidated
        """
        return self._cache_invalidation.invalidate_by_topic(topic)

    def invalidate_stale_cache(self, max_age_seconds: int = 3600) -> int:
        """
        Invalidate cache entries older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries invalidated
        """
        return self._cache_invalidation.invalidate_stale(max_age_seconds)

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        # Stop background enrichment worker
        if hasattr(self, '_enrichment_running'):
            self._enrichment_running = False
            if hasattr(self, '_enrichment_future'):
                try:
                    self._enrichment_future.result(timeout=5.0)
                except Exception:
                    pass
            if hasattr(self, '_enrichment_executor'):
                self._enrichment_executor.shutdown(wait=True)
            logger.info("Background enrichment worker stopped")

        # Close persistent queue
        if hasattr(self, '_enrichment_queue') and hasattr(self._enrichment_queue, 'close'):
            self._enrichment_queue.close()

        # Close cache
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
    """Initialize global Mindcore instance (thread-safe singleton)."""
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


# Modular Context Layer
from .context_layer import (
    ContextLayer,
    ContextLayerConfig,
    ContextLayerTier,
    BasicContextLayer,
    VectorContextLayer,
    FullContextLayer,
)

# Knowledge Store (unified single/multi-agent interface)
from .knowledge_store import (
    KnowledgeStore,
    KnowledgeItem,
    AgentConfig,
    StoreMode,
    SimpleKnowledgeStore,
)


# Lazy imports for vector stores (optional dependencies)
def get_vector_stores():
    """Get vector store classes (requires optional dependencies)."""
    from . import vectorstores
    return vectorstores


def get_connectors():
    """Get connector classes."""
    from . import connectors
    return connectors


# Public API
__all__ = [
    # Version
    "__version__",

    # Main client
    "MindcoreClient",
    "Mindcore",
    "initialize",
    "get_client",

    # Modular Context Layer
    "ContextLayer",
    "ContextLayerConfig",
    "ContextLayerTier",
    "BasicContextLayer",
    "VectorContextLayer",
    "FullContextLayer",

    # Knowledge Store
    "KnowledgeStore",
    "SimpleKnowledgeStore",
    "KnowledgeItem",
    "AgentConfig",
    "StoreMode",

    # AI Agents
    "MetadataAgent",
    "SmartContextAgent",
    "ContextTools",
    "SummarizationAgent",
    "BaseAgent",
    "TrivialMessageDetector",
    "TrivialCategory",
    "get_trivial_detector",

    # VocabularyManager
    "VocabularyManager",
    "VocabularySource",
    "Intent",
    "Sentiment",
    "CommunicationStyle",
    "EntityType",
    "get_vocabulary",

    # Worker Monitoring
    "WorkerMonitor",
    "WorkerMetrics",
    "WorkerStatus",
    "get_worker_monitor",

    # Adaptive Preferences
    "AdaptivePreferencesLearner",
    "AdaptiveConfig",
    "get_adaptive_learner",

    # Retention Policy
    "RetentionPolicyManager",
    "RetentionConfig",
    "MemoryTier",
    "get_retention_policy",

    # Cache Invalidation
    "CacheInvalidationManager",
    "InvalidationReason",
    "get_cache_invalidation",

    # Multi-Agent Support
    "MultiAgentConfig",
    "MultiAgentManager",
    "MemorySharingMode",
    "AgentVisibility",
    "AgentProfile",
    "get_multi_agent_manager",
    "configure_multi_agent",

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
    "ThreadSummary",
    "UserPreferences",

    # Core managers
    "ConfigLoader",
    "DatabaseManager",
    "SQLiteManager",
    "CacheManager",
    "PreferencesManager",

    # Lazy imports
    "get_vector_stores",
    "get_connectors",
]
