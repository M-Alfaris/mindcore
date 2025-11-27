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
from .utils import get_logger, generate_message_id, SecurityValidator

logger = get_logger(__name__)

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

            # Sort by created_at (most recent first)
            all_messages.sort(key=lambda m: m.created_at or m.message_id, reverse=True)
            cached_messages = all_messages[:max_messages]

        logger.info(f"Retrieved {len(cached_messages)} messages for context assembly")

        # Assemble context
        context = self.context_agent.process(cached_messages, query)
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


# Public API
__all__ = [
    # Version
    "__version__",

    # Main client
    "MindcoreClient",
    "Mindcore",
    "initialize",
    "get_client",

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
