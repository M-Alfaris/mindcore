"""
Async Mindcore client for high-performance applications.

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
from typing import Optional, Dict, Any, List

from .core import (
    ConfigLoader,
    CacheManager,
    Message,
    MessageMetadata,
    AssembledContext,
)
from .core.async_db import AsyncSQLiteManager, AsyncDatabaseManager
from .agents import (
    EnrichmentAgent as MetadataAgent,
    ContextAssemblerAgent as ContextAgent,
)
from .llm import (
    BaseLLMProvider,
    ProviderType,
    create_provider,
    get_provider_type,
)
from .utils import get_logger, SecurityValidator

logger = get_logger(__name__)


class AsyncMindcoreClient:
    """
    Async Mindcore client for non-blocking operations.

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
        config_path: Optional[str] = None,
        use_sqlite: bool = False,
        sqlite_path: str = "mindcore.db",
        llm_provider: Optional[str] = None
    ):
        """
        Initialize async Mindcore client.

        Note: Call `await client.connect()` or use `async with` context manager
        to establish database connections.

        Args:
            config_path: Optional path to config.yaml file.
            use_sqlite: If True, use SQLite instead of PostgreSQL.
            sqlite_path: Path to SQLite database file.
            llm_provider: LLM provider mode ("auto", "llama_cpp", "openai").
        """
        self.config = ConfigLoader(config_path)
        self._use_sqlite = use_sqlite
        self._sqlite_path = sqlite_path
        self._llm_provider_type = llm_provider
        self._connected = False

        # Will be initialized on connect
        self.db = None
        self.cache = None
        self._llm_provider = None
        self.metadata_agent = None
        self.context_agent = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Initialize connections and agents.

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

        # Initialize cache (sync - uses thread-safe in-memory cache)
        cache_config = self.config.get_cache_config()
        self.cache = CacheManager(
            max_size=cache_config.get('max_size', 50),
            ttl_seconds=cache_config.get('ttl')
        )

        # Initialize LLM provider
        self._llm_provider = self._create_llm_provider(self._llm_provider_type)

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

        self._connected = True
        db_type = "SQLite" if self._use_sqlite else "PostgreSQL"
        logger.info(
            f"Async Mindcore initialized with {db_type} and "
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

    async def ingest_message(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message with automatic metadata enrichment.

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

    async def get_context(
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
            db_messages = await self.db.fetch_recent_messages(user_id, thread_id, limit=max_messages)

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
        context = self.context_agent.process(cached_messages, query)
        return context

    async def get_message(self, message_id: str) -> Optional[Message]:
        """Get a single message by ID."""
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' or call connect() first.")
        return await self.db.get_message_by_id(message_id)

    def clear_cache(self, user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        """Clear message cache."""
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self._llm_provider:
            self._llm_provider.close()
        if self.db:
            await self.db.close()
        self._connected = False
        logger.info("Async Mindcore client closed")


# Convenience alias
AsyncMindcore = AsyncMindcoreClient
