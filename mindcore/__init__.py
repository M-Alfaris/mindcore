"""
Mindcore: Intelligent memory and context management for AI agents.

Mindcore provides two lightweight AI agents powered by GPT-4o-mini:
1. Metadata Enrichment Agent - Enriches messages with intelligent metadata
2. Context Assembly Agent - Retrieves and summarizes relevant historical context
"""
from typing import Optional, Dict, Any

from .core import (
    ConfigLoader,
    DatabaseManager,
    CacheManager,
    Message,
    MessageMetadata,
    AssembledContext,
)
from .agents import EnrichmentAgent, ContextAssemblerAgent
from .utils import get_logger, validate_message_dict, generate_message_id

logger = get_logger(__name__)

__version__ = "0.1.0"
__all__ = ["Mindcore", "get_mindcore_instance"]

# Global instance
_mindcore_instance: Optional['Mindcore'] = None


class Mindcore:
    """
    Main Mindcore class for intelligent memory and context management.

    This class provides:
    - Message ingestion with automatic metadata enrichment
    - Context assembly for historical information retrieval
    - PostgreSQL persistence
    - In-memory caching for fast access
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Mindcore framework.

        Args:
            config_path: Optional path to config.yaml file.
        """
        logger.info(f"Initializing Mindcore v{__version__}")

        # Load configuration
        self.config = ConfigLoader(config_path)

        # Initialize database
        db_config = self.config.get_database_config()
        self.db = DatabaseManager(db_config)
        self.db.initialize_schema()

        # Initialize cache
        cache_config = self.config.get_cache_config()
        self.cache = CacheManager(max_size=cache_config.get('max_size', 50))

        # Initialize AI agents
        openai_config = self.config.get_openai_config()
        api_key = openai_config.get('api_key')

        if not api_key:
            logger.warning("OpenAI API key not found in config. Set OPENAI_API_KEY environment variable or add to config.yaml")

        self.enrichment_agent = EnrichmentAgent(
            api_key=api_key,
            model=openai_config.get('model', 'gpt-4o-mini'),
            temperature=openai_config.get('temperature', 0.3)
        )

        self.context_agent = ContextAssemblerAgent(
            api_key=api_key,
            model=openai_config.get('model', 'gpt-4o-mini'),
            temperature=openai_config.get('temperature', 0.3)
        )

        logger.info("Mindcore initialized successfully")

    def ingest_message(self, message_dict: Dict[str, Any]) -> Message:
        """
        Ingest a message: enrich with metadata and store.

        Args:
            message_dict: Dictionary containing message fields:
                - user_id: str
                - thread_id: str
                - session_id: str
                - role: str
                - text: str
                - message_id: str (optional)

        Returns:
            Enriched Message object.
        """
        # Validate message
        if not validate_message_dict(message_dict):
            raise ValueError("Invalid message dictionary. Required fields: user_id, thread_id, session_id, role, text")

        # Enrich message
        message = self.enrichment_agent.process(message_dict)

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
        Get assembled context for a query.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query or topic for context assembly.
            max_messages: Maximum number of messages to consider.

        Returns:
            AssembledContext object with summarized context.
        """
        # Get messages from cache first
        cached_messages = self.cache.get_recent_messages(user_id, thread_id, limit=max_messages)

        # If cache doesn't have enough, fetch from database
        if len(cached_messages) < max_messages:
            db_messages = self.db.fetch_recent_messages(user_id, thread_id, limit=max_messages)

            # Merge (cache is already most recent)
            message_ids = {msg.message_id for msg in cached_messages}
            for msg in db_messages:
                if msg.message_id not in message_ids and len(cached_messages) < max_messages:
                    cached_messages.append(msg)

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
        """
        return self.db.get_message_by_id(message_id)

    def clear_cache(self, user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            user_id: Optional user ID to clear specific thread.
            thread_id: Optional thread ID to clear specific thread.
        """
        if user_id and thread_id:
            self.cache.clear_thread(user_id, thread_id)
        else:
            self.cache.clear_all()

    def close(self) -> None:
        """Close all connections and cleanup."""
        self.db.close()
        logger.info("Mindcore closed")


def initialize_mindcore(config_path: Optional[str] = None) -> Mindcore:
    """
    Initialize global Mindcore instance.

    Args:
        config_path: Optional path to config.yaml file.

    Returns:
        Mindcore instance.
    """
    global _mindcore_instance

    if _mindcore_instance is None:
        _mindcore_instance = Mindcore(config_path)

    return _mindcore_instance


def get_mindcore_instance() -> Mindcore:
    """
    Get the global Mindcore instance.

    Returns:
        Mindcore instance.

    Raises:
        RuntimeError: If Mindcore has not been initialized.
    """
    global _mindcore_instance

    if _mindcore_instance is None:
        # Auto-initialize with default config
        _mindcore_instance = Mindcore()

    return _mindcore_instance
