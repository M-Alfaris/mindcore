"""
Mindcore: Intelligent Memory and Context Management for AI Agents
===================================================================

Save 60-90% on token costs with intelligent memory management powered by
lightweight AI agents.

Quick Start:
-----------
    from mindcore import MindcoreClient

    # Initialize
    client = MindcoreClient()

    # Ingest messages with automatic metadata enrichment
    message = client.ingest_message({
        "user_id": "user123",
        "thread_id": "thread456",
        "session_id": "session789",
        "role": "user",
        "text": "Hello, how do I build AI agents?"
    })

    # Get intelligent context
    context = client.get_context(
        user_id="user123",
        thread_id="thread456",
        query="AI agent development"
    )

Framework Integration:
--------------------
    from mindcore.integrations import LangChainIntegration

    integration = LangChainIntegration(client)
    memory = integration.as_langchain_memory("user123", "thread456", "session789")

Features:
---------
- 🤖 Two lightweight AI agents (MetadataAgent, ContextAgent)
- 💾 PostgreSQL + in-memory caching
- 🔒 Production-grade security
- 💰 60-90% cost savings vs traditional approaches
- 🔌 LangChain, LlamaIndex, custom AI integrations
"""

from typing import Optional, Dict, Any, List
import threading
import atexit

# Version
__version__ = "0.1.0"
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

# AI Agents (with clean aliases)
from .agents import (
    BaseAgent,
    EnrichmentAgent as MetadataAgent,  # Clean name
    ContextAssemblerAgent as ContextAgent,  # Clean name
)

# Main client
from .utils import get_logger, generate_message_id, SecurityValidator

logger = get_logger(__name__)

# Global instance with thread-safe initialization
_mindcore_instance: Optional['MindcoreClient'] = None
_instance_lock = threading.Lock()


class MindcoreClient:
    """
    Main Mindcore client for intelligent memory and context management.

    The MindcoreClient provides:
    - Automatic metadata enrichment with MetadataAgent (GPT-4o-mini)
    - Intelligent context assembly with ContextAgent (GPT-4o-mini)
    - PostgreSQL or SQLite persistence with caching
    - 60-90% cost savings vs traditional memory management

    Usage:
        >>> from mindcore import MindcoreClient
        >>>
        >>> # Standard usage (requires PostgreSQL)
        >>> client = MindcoreClient()
        >>>
        >>> # Local development with SQLite (no PostgreSQL needed!)
        >>> client = MindcoreClient(use_sqlite=True)
        >>>
        >>> # Ingest message
        >>> msg = client.ingest_message({
        ...     "user_id": "user123",
        ...     "thread_id": "thread456",
        ...     "session_id": "session789",
        ...     "role": "user",
        ...     "text": "Hello!"
        ... })
        >>>
        >>> # Get context
        >>> context = client.get_context(
        ...     user_id="user123",
        ...     thread_id="thread456",
        ...     query="conversation history"
        ... )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_sqlite: bool = False,
        sqlite_path: str = "mindcore.db"
    ):
        """
        Initialize Mindcore client.

        Args:
            config_path: Optional path to config.yaml file.
                        If not provided, uses default locations or environment variables.
            use_sqlite: If True, use SQLite instead of PostgreSQL.
                       Perfect for local development and testing.
            sqlite_path: Path to SQLite database file (default: "mindcore.db").
                        Use ":memory:" for in-memory database.

        Example:
            >>> client = MindcoreClient()  # Use default config (PostgreSQL)
            >>> client = MindcoreClient(use_sqlite=True)  # Use SQLite for local dev
            >>> client = MindcoreClient(use_sqlite=True, sqlite_path=":memory:")  # In-memory
            >>> client = MindcoreClient("path/to/config.yaml")  # Custom config
        """
        logger.info(f"Initializing Mindcore v{__version__}")

        # Load configuration
        self.config = ConfigLoader(config_path)
        self._use_sqlite = use_sqlite

        # Initialize database (SQLite or PostgreSQL)
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
            ttl_seconds=cache_config.get('ttl')  # None means no TTL
        )

        # Initialize AI agents
        openai_config = self.config.get_openai_config()
        api_key = openai_config.get('api_key')

        if not api_key:
            logger.warning(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or add to config.yaml"
            )

        # Metadata enrichment agent
        self.metadata_agent = MetadataAgent(
            api_key=api_key,
            model=openai_config.get('model', 'gpt-4o-mini'),
            temperature=openai_config.get('temperature', 0.3)
        )

        # Context assembly agent
        self.context_agent = ContextAgent(
            api_key=api_key,
            model=openai_config.get('model', 'gpt-4o-mini'),
            temperature=openai_config.get('temperature', 0.3)
        )

        # Legacy aliases for backward compatibility
        self.enrichment_agent = self.metadata_agent
        self.context_assembler = self.context_agent

        db_type = "SQLite" if use_sqlite else "PostgreSQL"
        logger.info(f"Mindcore initialized successfully with {db_type}")

    def ingest_message(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message with automatic metadata enrichment.

        The message will be:
        1. Validated for security
        2. Enriched with metadata (topics, sentiment, intent, etc.)
        3. Stored in PostgreSQL
        4. Cached for fast retrieval

        Args:
            message_dict: Dictionary containing:
                - user_id (str): User identifier
                - thread_id (str): Conversation thread identifier
                - session_id (str): Session identifier
                - role (str): Message role (user, assistant, system, tool)
                - text (str): Message content
                - message_id (str, optional): Auto-generated if not provided

        Returns:
            Message: Enriched message object with metadata.

        Raises:
            ValueError: If validation fails or required fields are missing.

        Example:
            >>> message = client.ingest_message({
            ...     "user_id": "user123",
            ...     "thread_id": "thread456",
            ...     "session_id": "session789",
            ...     "role": "user",
            ...     "text": "What are best practices for AI agents?"
            ... })
            >>> print(message.metadata.topics)
            ['AI', 'best practices', 'agents']
        """
        # Validate message with security checks
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

        The ContextAgent will:
        1. Retrieve recent messages from cache and database
        2. Analyze relevance to the query using metadata
        3. Summarize and compress relevant information
        4. Return structured context ready for LLM injection

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query or topic to find relevant context for.
            max_messages: Maximum messages to consider (default: 50).

        Returns:
            AssembledContext: Object containing:
                - assembled_context (str): Summarized relevant context
                - key_points (List[str]): Key points from history
                - relevant_message_ids (List[str]): IDs of relevant messages
                - metadata (Dict): Topics, sentiment, importance

        Raises:
            ValueError: If validation fails.

        Example:
            >>> context = client.get_context(
            ...     user_id="user123",
            ...     thread_id="thread456",
            ...     query="AI agent memory management"
            ... )
            >>> print(context.assembled_context)
            'User previously discussed implementing memory systems...'
            >>> print(context.key_points)
            ['Use vector databases', 'Implement caching', 'Add metadata']
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

            # Merge messages from cache and database, avoiding duplicates
            message_ids = {msg.message_id for msg in cached_messages}
            all_messages = list(cached_messages)

            for msg in db_messages:
                if msg.message_id not in message_ids:
                    all_messages.append(msg)
                    message_ids.add(msg.message_id)

            # Sort by created_at to ensure chronological order (most recent first)
            all_messages.sort(key=lambda m: m.created_at or m.message_id, reverse=True)

            # Limit to max_messages
            cached_messages = all_messages[:max_messages]

        logger.info(f"Retrieved {len(cached_messages)} messages for context assembly")

        # Assemble context using AI agent
        context = self.context_agent.process(cached_messages, query)

        return context

    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a single message by ID.

        Args:
            message_id: Message identifier.

        Returns:
            Message object or None if not found.

        Example:
            >>> message = client.get_message("msg_abc123")
            >>> if message:
            ...     print(message.raw_text)
        """
        return self.db.get_message_by_id(message_id)

    def clear_cache(self, user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        """
        Clear message cache.

        Args:
            user_id: Optional user ID to clear specific thread.
            thread_id: Optional thread ID to clear specific thread.
                      If both provided, clears that specific thread.
                      If neither provided, clears entire cache.

        Example:
            >>> client.clear_cache("user123", "thread456")  # Clear specific thread
            >>> client.clear_cache()  # Clear all cache
        """
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    def close(self) -> None:
        """
        Close all connections and cleanup resources.

        Call this when shutting down your application.

        Example:
            >>> client = MindcoreClient()
            >>> # ... use client ...
            >>> client.close()
        """
        self.db.close()
        logger.info("Mindcore client closed")


# Convenience alias (backward compatibility)
Mindcore = MindcoreClient


def initialize(config_path: Optional[str] = None) -> MindcoreClient:
    """
    Initialize global Mindcore instance (thread-safe singleton pattern).

    Args:
        config_path: Optional path to config.yaml file.

    Returns:
        MindcoreClient instance.

    Example:
        >>> from mindcore import initialize
        >>> client = initialize()
    """
    global _mindcore_instance

    if _mindcore_instance is None:
        with _instance_lock:
            # Double-check locking pattern
            if _mindcore_instance is None:
                _mindcore_instance = MindcoreClient(config_path)

    return _mindcore_instance


def get_client() -> MindcoreClient:
    """
    Get the global Mindcore client instance (thread-safe).

    Returns:
        MindcoreClient instance (auto-initializes if needed).

    Example:
        >>> from mindcore import get_client
        >>> client = get_client()
    """
    global _mindcore_instance

    if _mindcore_instance is None:
        with _instance_lock:
            # Double-check locking pattern
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


# Register cleanup handler
atexit.register(_cleanup_on_exit)


# Legacy function names (backward compatibility)
initialize_mindcore = initialize
get_mindcore_instance = get_client


# Public API - what users can import
__all__ = [
    # Version
    "__version__",

    # Main client
    "MindcoreClient",
    "Mindcore",  # Alias
    "initialize",
    "get_client",

    # AI Agents (clean names)
    "MetadataAgent",
    "ContextAgent",
    "BaseAgent",

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

    # Legacy (backward compatibility)
    "initialize_mindcore",
    "get_mindcore_instance",
]
