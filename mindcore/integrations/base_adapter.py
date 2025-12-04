"""Base adapter class for framework integrations."""

from abc import ABC, abstractmethod
from typing import Any

from mindcore.core.schemas import AssembledContext, Message
from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


class BaseAdapter(ABC):
    """Base adapter for integrating Mindcore with AI frameworks.

    Provides standard interface for:
    - Message conversion
    - Context injection
    - Callback integration
    """

    def __init__(self, mindcore_instance):
        """Initialize adapter.

        Args:
            mindcore_instance: Instance of Mindcore.
        """
        self.mindcore = mindcore_instance

    @abstractmethod
    def format_message_for_ingestion(self, framework_message: Any) -> dict[str, Any]:
        """Convert framework-specific message to Mindcore format.

        Args:
            framework_message: Message in framework format.

        Returns:
            Message dict for Mindcore ingestion.
        """

    @abstractmethod
    def inject_context_into_prompt(self, context: AssembledContext, existing_prompt: str) -> str:
        """Inject assembled context into framework prompt.

        Args:
            context: Assembled context from Mindcore.
            existing_prompt: Existing prompt template.

        Returns:
            Enhanced prompt with context.
        """

    def ingest_conversation(
        self, messages: list[Any], user_id: str, thread_id: str, session_id: str
    ) -> list[Message]:
        """Ingest a conversation history.

        Args:
            messages: List of framework messages.
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.

        Returns:
            List of enriched Message objects.
        """
        enriched_messages = []

        for msg in messages:
            try:
                # Convert to Mindcore format
                msg_dict = self.format_message_for_ingestion(msg)

                # Add IDs
                msg_dict["user_id"] = user_id
                msg_dict["thread_id"] = thread_id
                msg_dict["session_id"] = session_id

                # Ingest
                enriched_msg = self.mindcore.ingest_message(msg_dict)
                enriched_messages.append(enriched_msg)

            except Exception as e:
                # Log but continue
                logger.warning(f"Failed to ingest message: {e}")
                continue

        return enriched_messages

    def get_enhanced_context(
        self, user_id: str, thread_id: str, query: str, max_messages: int = 50
    ) -> AssembledContext:
        """Get assembled context.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            query: Query or topic.
            max_messages: Max messages to consider.

        Returns:
            AssembledContext object.
        """
        return self.mindcore.get_context(user_id, thread_id, query, max_messages)
