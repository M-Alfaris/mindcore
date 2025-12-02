"""
LlamaIndex integration adapter for Mindcore.

Provides seamless integration with LlamaIndex 0.10+:
- Chat history integration
- Context injection
- Custom memory backend

Compatible with:
- llama-index >= 0.10.0
- llama-index-core >= 0.10.0
"""

from typing import Dict, Any, List, Optional
from .base_adapter import BaseAdapter
from ..core.schemas import AssembledContext
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Check for LlamaIndex at import time
_LLAMAINDEX_AVAILABLE = False
_LLAMAINDEX_CORE_AVAILABLE = False

try:
    import llama_index.core

    _LLAMAINDEX_CORE_AVAILABLE = True
    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        import llama_index

        _LLAMAINDEX_AVAILABLE = True
    except ImportError:
        pass


def require_llamaindex():
    """Check if LlamaIndex is available, raise helpful error if not."""
    if not _LLAMAINDEX_AVAILABLE:
        raise ImportError("LlamaIndex is not installed. Install with: pip install llama-index")


class LlamaIndexAdapter(BaseAdapter):
    """
    Adapter for integrating Mindcore with LlamaIndex 0.10+.

    Usage:
        from llama_index.core.chat_engine import SimpleChatEngine
        from mindcore import MindcoreClient
        from mindcore.integrations import LlamaIndexAdapter

        mindcore = MindcoreClient()
        adapter = LlamaIndexAdapter(mindcore)

        # Ingest chat messages
        adapter.ingest_llamaindex_conversation(
            messages=[{"role": "user", "content": "Hello!"}],
            user_id="user123",
            thread_id="thread456",
            session_id="session789"
        )

        # Get context
        context = adapter.get_enhanced_context(
            user_id="user123",
            thread_id="thread456",
            query="chat history"
        )
    """

    def format_message_for_ingestion(self, framework_message: Any) -> Dict[str, Any]:
        """
        Convert LlamaIndex message to Mindcore format.

        Args:
            framework_message: LlamaIndex ChatMessage or dict.

        Returns:
            Message dict for Mindcore ingestion.
        """
        # LlamaIndex uses ChatMessage objects or dicts
        if isinstance(framework_message, dict):
            return {
                "role": framework_message.get("role", "user"),
                "text": framework_message.get("content", ""),
            }

        # If it's a ChatMessage object
        try:
            return {
                "role": getattr(framework_message, "role", "user"),
                "text": getattr(framework_message, "content", str(framework_message)),
            }
        except Exception:
            # Fallback
            return {"role": "user", "text": str(framework_message)}

    def inject_context_into_prompt(self, context: AssembledContext, existing_prompt: str) -> str:
        """
        Inject assembled context into LlamaIndex prompt.

        Args:
            context: Assembled context from Mindcore.
            existing_prompt: Existing system prompt.

        Returns:
            Enhanced prompt with historical context.
        """
        context_section = []

        if context.assembled_context:
            context_section.append("Historical Context:")
            context_section.append(context.assembled_context)
            context_section.append("")

        if context.key_points:
            context_section.append("Key Points:")
            for point in context.key_points:
                context_section.append(f"- {point}")
            context_section.append("")

        if not context_section:
            return existing_prompt

        context_text = "\n".join(context_section)

        # Prepend context to prompt
        return f"{context_text}\n{existing_prompt}"

    def ingest_llamaindex_conversation(
        self, messages: List[Any], user_id: str, thread_id: str, session_id: str
    ) -> List:
        """
        Ingest LlamaIndex conversation messages.

        Args:
            messages: List of LlamaIndex ChatMessage objects or dicts.
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.

        Returns:
            List of enriched Message objects.
        """
        return self.ingest_conversation(messages, user_id, thread_id, session_id)

    def create_chat_memory(
        self, user_id: str, thread_id: str, session_id: str, max_messages: int = 50
    ):
        """
        Create a LlamaIndex-compatible chat memory.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.
            max_messages: Maximum messages to return.

        Returns:
            Chat memory object.
        """

        class MindcoreChatMemory:
            """LlamaIndex-compatible chat memory backed by Mindcore."""

            def __init__(self, adapter, user_id, thread_id, session_id, max_messages):
                self.adapter = adapter
                self.user_id = user_id
                self.thread_id = thread_id
                self.session_id = session_id
                self.max_messages = max_messages

            def get_messages(self):
                """Get chat messages."""
                messages = self.adapter.mindcore.cache.get_recent_messages(
                    self.user_id, self.thread_id, limit=self.max_messages
                )

                # Convert to LlamaIndex format
                return [{"role": msg.role.value, "content": msg.raw_text} for msg in messages]

            def add_message(self, role: str, content: str):
                """Add a message."""
                self.adapter.mindcore.ingest_message(
                    {
                        "user_id": self.user_id,
                        "thread_id": self.thread_id,
                        "session_id": self.session_id,
                        "role": role,
                        "text": content,
                    }
                )

            def reset(self):
                """Clear memory."""
                self.adapter.mindcore.clear_cache(self.user_id, self.thread_id)

        return MindcoreChatMemory(self, user_id, thread_id, session_id, max_messages)
