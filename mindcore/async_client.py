"""Async Mindcore client for high-performance applications.

Provides async/await interface for non-blocking operations in asyncio applications.

Usage:
    from mindcore import AsyncMindcoreClient

    async def main():
        async with AsyncMindcoreClient(use_sqlite=True) as client:
            message = await client.ingest_message({
                "user_id": "user123",
                "thread_id": "thread456",
                "session_id": "session789",
                "role": "user",
                "text": "Hello, AI!"
            })

            context = await client.get_context(
                user_id="user123",
                thread_id="thread456",
                query="conversation context"
            )

    asyncio.run(main())
"""

import asyncio
import contextlib
from typing import Any

from .agents import (
    ContextAssemblerAgent as ContextAgent,
)
from .agents import (
    ContextTools,
    SmartContextAgent,
)
from .agents import (
    EnrichmentAgent as MetadataAgent,
)
from .core import (
    AssembledContext,
    CacheManager,
    ConfigLoader,
    DiskCacheManager,
    Message,
    MessageMetadata,
    MessageRole,
)
from .core.async_db import AsyncDatabaseManager, AsyncSQLiteManager
from .llm import (
    BaseLLMProvider,
    create_provider,
    get_provider_type,
)
from .utils import SecurityValidator, generate_message_id, get_logger


logger = get_logger(__name__)


class AsyncMindcoreClient:
    """Async Mindcore client for non-blocking operations.

    Ideal for:
    - FastAPI applications
    - High-concurrency web servers
    - Async microservices
    - Real-time applications

    Usage:
        async with AsyncMindcoreClient(use_sqlite=True) as client:
            message = await client.ingest_message(message_dict)
            context = await client.get_context(user_id, thread_id, query)
    """

    def __init__(
        self,
        config_path: str | None = None,
        use_sqlite: bool = False,
        sqlite_path: str = "mindcore.db",
        llm_provider: str | None = None,
        persistent_cache: bool = True,
        cache_dir: str | None = None,
    ):
        """Initialize async Mindcore client.

        Note: Call `await client.connect()` or use `async with` context manager
        to establish database connections.

        Args:
            config_path: Optional path to config.yaml file.
            use_sqlite: If True, use SQLite instead of PostgreSQL.
            sqlite_path: Path to SQLite database file.
            llm_provider: LLM provider mode ("auto", "llama_cpp", "openai").
            persistent_cache: If True (default), use disk-backed cache that
                survives restarts. If False, use in-memory cache.
            cache_dir: Optional directory for persistent cache files.
        """
        self.config = ConfigLoader(config_path)
        self._use_sqlite = use_sqlite
        self._sqlite_path = sqlite_path
        self._llm_provider_type = llm_provider
        self._persistent_cache = persistent_cache
        self._cache_dir = cache_dir
        self._connected = False

        # Will be initialized on connect
        self.db = None
        self.cache = None
        self._llm_provider = None
        self.metadata_agent = None
        self.context_agent = None
        self._smart_context_agent = None  # Lazy initialized

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize connections and agents.

        Must be called before using the client, or use `async with` context manager.
        """
        if self._connected:
            return

        logger.info("Initializing async Mindcore client")

        # Initialize database
        if self._use_sqlite:
            logger.info(f"Using async SQLite: {self._sqlite_path}")
            self.db = AsyncSQLiteManager(self._sqlite_path)
            await self.db.connect()
        else:
            db_config = self.config.get_database_config()
            self.db = AsyncDatabaseManager(db_config)
            await self.db.connect()

        # Initialize cache (persistent or in-memory)
        cache_config = self.config.get_cache_config()
        if self._persistent_cache:
            self.cache = DiskCacheManager(
                max_size=cache_config.get("max_size", 50),
                ttl_seconds=cache_config.get("ttl"),
                cache_dir=self._cache_dir,
            )
            logger.info("Using persistent disk-backed cache (diskcache)")
        else:
            self.cache = CacheManager(
                max_size=cache_config.get("max_size", 50), ttl_seconds=cache_config.get("ttl")
            )
            logger.info("Using in-memory cache (cachetools)")

        # Initialize LLM provider
        self._llm_provider = self._create_llm_provider(self._llm_provider_type)

        # Get generation defaults
        llm_config = self.config.get_llm_config()
        defaults = llm_config.get("defaults", {})

        # Initialize AI agents with the provider
        self.metadata_agent = MetadataAgent(
            llm_provider=self._llm_provider,
            temperature=defaults.get("temperature", 0.3),
            max_tokens=defaults.get("max_tokens_enrichment", 800),
        )

        self.context_agent = ContextAgent(
            llm_provider=self._llm_provider,
            temperature=defaults.get("temperature", 0.3),
            max_tokens=defaults.get("max_tokens_context", 1500),
        )

        # Background enrichment queue
        self._enrichment_queue: asyncio.Queue = asyncio.Queue()
        self._enrichment_task: asyncio.Task | None = None

        self._connected = True
        db_type = "SQLite" if self._use_sqlite else "PostgreSQL"
        logger.info(
            f"Async Mindcore initialized with {db_type} and {self._llm_provider.name} LLM provider"
        )

    def _create_llm_provider(self, provider_type_str: str | None) -> BaseLLMProvider:
        """Create LLM provider based on configuration."""
        llm_config = self.config.get_llm_config()

        if provider_type_str:
            provider_type = get_provider_type(provider_type_str)
        else:
            provider_type = get_provider_type(llm_config.get("provider", "auto"))

        llama_config = {}
        if llm_config["llama_cpp"].get("model_path"):
            llama_config = {
                "model_path": llm_config["llama_cpp"]["model_path"],
                "n_ctx": llm_config["llama_cpp"].get("n_ctx", 4096),
                "n_threads": llm_config["llama_cpp"].get("n_threads"),
                "n_gpu_layers": llm_config["llama_cpp"].get("n_gpu_layers", 0),
                "chat_format": llm_config["llama_cpp"].get("chat_format"),
                "verbose": llm_config["llama_cpp"].get("verbose", False),
            }

        openai_config = {}
        if llm_config["openai"].get("api_key"):
            openai_config = {
                "api_key": llm_config["openai"]["api_key"],
                "base_url": llm_config["openai"].get("base_url"),
                "model": llm_config["openai"].get("model", "gpt-4o-mini"),
                "timeout": llm_config["openai"].get("timeout", 60),
                "max_retries": llm_config["openai"].get("max_retries", 3),
            }

        try:
            return create_provider(
                provider_type=provider_type,
                llama_config=llama_config if llama_config else None,
                openai_config=openai_config if openai_config else None,
            )
        except ValueError as e:
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
        return self._llm_provider.name if self._llm_provider else "not_initialized"

    async def ingest_message(self, message_dict: dict[str, Any]) -> Message:
        """Ingest a message with automatic metadata enrichment.

        Args:
            message_dict: Dictionary containing message data.

        Returns:
            Message: Enriched message object with metadata.

        Raises:
            ValueError: If validation fails.
            RuntimeError: If client not connected.
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")

        # Validate message
        is_valid, error_msg = SecurityValidator.validate_message_dict(message_dict)
        if not is_valid:
            logger.error(f"Message validation failed: {error_msg}")
            raise ValueError(f"Invalid message: {error_msg}")

        # Enrich message with metadata (sync operation - LLM call)
        message = self.metadata_agent.process(message_dict)

        # Store in database (async)
        success = await self.db.insert_message(message)
        if not success:
            logger.warning(f"Failed to store message {message.message_id} in database")

        # Add to cache (sync - thread-safe)
        self.cache.add_message(message)

        logger.info(f"Message {message.message_id} ingested successfully (async)")
        return message

    async def ingest_message_fast(self, message_dict: dict[str, Any]) -> Message:
        """Ingest a message immediately without blocking on enrichment.

        The message is stored immediately with empty metadata, cached for the
        current session, and queued for background enrichment. This is ideal
        for real-time applications where response latency is critical.

        Enrichment happens asynchronously:
        1. Message stored with empty metadata (~0ms delay)
        2. Message added to cache (available for current session context)
        3. Background task enriches and updates database

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
            RuntimeError: If client not connected.
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")

        # Validate message
        is_valid, error_msg = SecurityValidator.validate_message_dict(message_dict)
        if not is_valid:
            logger.error(f"Message validation failed: {error_msg}")
            raise ValueError(f"Invalid message: {error_msg}")

        # Create message with empty metadata (no LLM call)
        message = Message(
            message_id=message_dict.get("message_id") or generate_message_id(),
            user_id=message_dict["user_id"],
            thread_id=message_dict["thread_id"],
            session_id=message_dict["session_id"],
            role=MessageRole(message_dict["role"]),
            raw_text=message_dict["text"],
            metadata=MessageMetadata(),  # Empty metadata
        )

        # Store in database immediately (async)
        success = await self.db.insert_message(message)
        if not success:
            logger.warning(f"Failed to store message {message.message_id} in database")

        # Add to cache for immediate session context (sync - thread-safe)
        self.cache.add_message(message)

        # Queue for background enrichment with ALL IDs preserved
        enrichment_task = {
            "message_id": message.message_id,
            "user_id": message.user_id,
            "thread_id": message.thread_id,
            "session_id": message.session_id,
            "role": message.role.value,
            "text": message.raw_text,
        }
        await self._enrichment_queue.put(enrichment_task)

        # Start enrichment worker if not running
        if self._enrichment_task is None or self._enrichment_task.done():
            self._enrichment_task = asyncio.create_task(self._enrichment_worker())

        logger.info(f"Message {message.message_id} ingested fast (enrichment queued, async)")
        return message

    async def _enrichment_worker(self) -> None:
        """Background async task that enriches messages from the queue."""
        logger.info("Async background enrichment worker started")

        while self._connected:
            try:
                # Wait for task with timeout (allows graceful shutdown)
                try:
                    task = await asyncio.wait_for(self._enrichment_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if queue is empty and we can exit
                    if self._enrichment_queue.empty():
                        break
                    continue

                if task is None:
                    continue

                message_id = task.get("message_id")
                if not message_id:
                    logger.warning("Enrichment worker: task missing message_id")
                    continue

                # Check if already enriched in database
                existing = await self.db.get_message_by_id(message_id)
                if existing and existing.metadata.is_enriched:
                    logger.debug(f"Message {message_id} already enriched, skipping")
                    continue

                # Enrich message using task data (all IDs preserved from queue)
                try:
                    enriched_message = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda t=task: self.metadata_agent.process(
                            {
                                "message_id": t["message_id"],
                                "user_id": t["user_id"],
                                "thread_id": t["thread_id"],
                                "session_id": t["session_id"],
                                "role": t["role"],
                                "text": t["text"],
                            }
                        ),
                    )

                    # Update database with enriched metadata
                    await self.db.update_message_metadata(
                        message_id=message_id, metadata=enriched_message.metadata
                    )

                    # Update cache with enriched message (preserves all IDs)
                    self.cache.update_message(enriched_message)

                    logger.debug(f"Background enrichment completed for {message_id} (async)")

                except Exception as e:
                    logger.exception(f"Background enrichment failed for {message_id}: {e}")
                    # Mark as failed in database
                    failed_metadata = MessageMetadata(
                        enrichment_failed=True, enrichment_error=str(e)
                    )
                    await self.db.update_message_metadata(message_id, failed_metadata)

            except Exception as e:
                logger.exception(f"Async enrichment worker error: {e}")

        logger.info("Async background enrichment worker stopped")

    async def get_context(
        self, user_id: str, thread_id: str, query: str, max_messages: int = 50
    ) -> AssembledContext:
        """Get intelligently assembled context for a query.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query or topic to find relevant context for.
            max_messages: Maximum messages to consider.

        Returns:
            AssembledContext with summarized context, key points, etc.

        Raises:
            ValueError: If validation fails.
            RuntimeError: If client not connected.
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")

        # Validate query parameters
        is_valid, error_msg = SecurityValidator.validate_query_params(user_id, thread_id, query)
        if not is_valid:
            logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query parameters: {error_msg}")

        # Get messages from cache first
        cached_messages = self.cache.get_recent_messages(user_id, thread_id, limit=max_messages)

        # If cache doesn't have enough, fetch from database (async)
        if len(cached_messages) < max_messages:
            db_messages = await self.db.fetch_recent_messages(
                user_id, thread_id, limit=max_messages
            )

            # Merge messages, avoiding duplicates
            message_ids = {msg.message_id for msg in cached_messages}
            all_messages = list(cached_messages)

            for msg in db_messages:
                if msg.message_id not in message_ids:
                    all_messages.append(msg)
                    message_ids.add(msg.message_id)

            # Sort by created_at (most recent first)
            all_messages.sort(key=lambda m: m.created_at or m.message_id, reverse=True)
            cached_messages = all_messages[:max_messages]

        logger.info(f"Retrieved {len(cached_messages)} messages for context assembly (async)")

        # Assemble context (sync operation - LLM call)
        return self.context_agent.process(cached_messages, query)

    def _get_smart_context_agent(self) -> SmartContextAgent:
        """Get or create the SmartContextAgent (lazy initialization).

        Creates ContextTools callbacks that connect the agent to
        cache and database for fetching context data.
        """
        if self._smart_context_agent is not None:
            return self._smart_context_agent

        # Create callbacks for the agent's tools with exception handling
        # Note: These are sync callbacks - SmartContextAgent handles sync operations
        # The async database calls are wrapped with asyncio.run
        def get_recent_messages(user_id: str, thread_id: str, limit: int) -> list:
            try:
                return self.cache.get_recent_messages(user_id, thread_id, limit)
            except Exception as e:
                logger.exception(f"Error fetching recent messages: {e}")
                return []

        def search_history(
            user_id: str,
            thread_id: str | None,
            topics: list,
            categories: list,
            intent: str | None,
            limit: int,
        ) -> list:
            # This runs in a thread pool executor (from get_context_smart),
            # so we can safely create a new event loop for the async DB call
            try:
                coro = self.db.search_by_relevance(
                    user_id=user_id,
                    topics=topics if topics else None,
                    categories=categories if categories else None,
                    intent=intent,
                    thread_id=thread_id,
                    limit=limit,
                )
                # Create new event loop for this thread (executor thread)
                try:
                    asyncio.get_running_loop()
                    # If we're in an async context, we need to handle differently
                    # This shouldn't happen since we run agent in executor
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, coro)
                        return future.result(timeout=30)  # 30 second timeout
                except RuntimeError:
                    # No running loop - we're in the executor thread, safe to use asyncio.run
                    return asyncio.run(coro)
            except Exception as e:
                logger.exception(f"Error searching history: {e}")
                return []

        def get_session_metadata(user_id: str, thread_id: str) -> dict:
            try:
                return self.cache.get_session_metadata(user_id, thread_id)
            except Exception as e:
                logger.exception(f"Error fetching session metadata: {e}")
                return {"topics": [], "categories": [], "intents": [], "message_count": 0}

        tools = ContextTools(
            get_recent_messages=get_recent_messages,
            search_history=search_history,
            get_session_metadata=get_session_metadata,
        )

        llm_config = self.config.get_llm_config()
        defaults = llm_config.get("defaults", {})

        self._smart_context_agent = SmartContextAgent(
            llm_provider=self._llm_provider,
            context_tools=tools,
            temperature=0.2,  # Low for consistency
            max_tokens=defaults.get("max_tokens_context", 1500),
            max_tool_rounds=3,
        )

        logger.info("SmartContextAgent initialized with database/cache tools (async client)")
        return self._smart_context_agent

    async def get_context_smart(
        self, user_id: str, thread_id: str, query: str, additional_context: str | None = None
    ) -> AssembledContext:
        """Get context using single LLM call with tool calling (optimized latency).

        This method uses SmartContextAgent which combines query analysis and
        context assembly into a single LLM call with tools. The agent
        intelligently decides what context to fetch using:
        - get_recent_messages: Recent conversation context
        - search_history: Historical messages by topics/categories/intent
        - get_session_metadata: Session aggregated metadata

        This reduces latency from 2 LLM calls to 1 while maintaining
        intelligent context selection.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query to find relevant context for.
            additional_context: Optional context to provide to the agent.

        Returns:
            AssembledContext with summarized context, key points, etc.

        Raises:
            ValueError: If validation fails.
            RuntimeError: If client not connected.

        Example:
            >>> context = await client.get_context_smart(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     query="What did we discuss about billing last time?"
            ... )
            >>> print(context.summary)
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")

        # Validate query parameters
        is_valid, error_msg = SecurityValidator.validate_query_params(user_id, thread_id, query)
        if not is_valid:
            logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query parameters: {error_msg}")

        agent = self._get_smart_context_agent()

        # Run agent (sync) in executor to not block event loop
        loop = asyncio.get_running_loop()
        context = await loop.run_in_executor(
            None,
            lambda: agent.process(
                query=query,
                user_id=user_id,
                thread_id=thread_id,
                additional_context=additional_context,
            ),
        )

        logger.info(
            f"SmartContextAgent assembled context (async): "
            f"source={context.metadata.get('context_source', 'unknown')}, "
            f"confidence={context.metadata.get('confidence', 'unknown')}"
        )
        return context

    async def get_message(self, message_id: str) -> Message | None:
        """Get a single message by ID."""
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")
        return await self.db.get_message_by_id(message_id)

    def clear_cache(self, user_id: str | None = None, thread_id: str | None = None) -> None:
        """Clear message cache."""
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        # Stop background enrichment task
        self._connected = False
        if (
            hasattr(self, "_enrichment_task")
            and self._enrichment_task
            and not self._enrichment_task.done()
        ):
            self._enrichment_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._enrichment_task
            logger.info("Background enrichment task cancelled")

        # Close cache (DiskCacheManager needs explicit close)
        if hasattr(self, "cache") and self.cache and hasattr(self.cache, "close"):
            self.cache.close()

        if self._llm_provider:
            self._llm_provider.close()
        if self.db:
            await self.db.close()
        logger.info("Async Mindcore client closed")


# Convenience alias
AsyncMindcore = AsyncMindcoreClient
