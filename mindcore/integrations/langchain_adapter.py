"""
LangChain integration adapter for Mindcore.

Provides seamless integration with LangChain:
- Automatic message ingestion via callbacks
- Context injection into prompts
- Memory interface compatibility
"""
from typing import Dict, Any, List, Optional
from .base_adapter import BaseAdapter
from ..core.schemas import AssembledContext
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LangChainAdapter(BaseAdapter):
    """
    Adapter for integrating Mindcore with LangChain.

    Usage:
        from langchain.chat_models import ChatOpenAI
        from mindcore import Mindcore
        from mindcore.adapters import LangChainAdapter

        mindcore = Mindcore()
        adapter = LangChainAdapter(mindcore)

        # Ingest LangChain messages
        from langchain.schema import HumanMessage, AIMessage

        messages = [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!")
        ]

        adapter.ingest_langchain_conversation(
            messages=messages,
            user_id="user123",
            thread_id="thread456",
            session_id="session789"
        )

        # Get context and inject into prompt
        context = adapter.get_enhanced_context(
            user_id="user123",
            thread_id="thread456",
            query="previous conversation"
        )

        enhanced_prompt = adapter.inject_context_into_prompt(
            context=context,
            existing_prompt="You are a helpful assistant."
        )
    """

    def format_message_for_ingestion(self, framework_message: Any) -> Dict[str, Any]:
        """
        Convert LangChain message to Mindcore format.

        Args:
            framework_message: LangChain message object (HumanMessage, AIMessage, SystemMessage).

        Returns:
            Message dict for Mindcore ingestion.
        """
        # Detect message type
        message_type = type(framework_message).__name__

        role_mapping = {
            'HumanMessage': 'user',
            'AIMessage': 'assistant',
            'SystemMessage': 'system',
            'ChatMessage': 'user',
            'FunctionMessage': 'tool',
            'ToolMessage': 'tool',
        }

        role = role_mapping.get(message_type, 'user')

        return {
            'role': role,
            'text': framework_message.content
        }

    def inject_context_into_prompt(self, context: AssembledContext, existing_prompt: str) -> str:
        """
        Inject assembled context into LangChain prompt.

        Args:
            context: Assembled context from Mindcore.
            existing_prompt: Existing system prompt.

        Returns:
            Enhanced prompt with historical context.
        """
        context_section = []

        if context.assembled_context:
            context_section.append("\n## Historical Context")
            context_section.append(context.assembled_context)

        if context.key_points:
            context_section.append("\n## Key Points from Previous Conversations")
            for i, point in enumerate(context.key_points, 1):
                context_section.append(f"{i}. {point}")

        if not context_section:
            return existing_prompt

        context_text = "\n".join(context_section)

        # Insert context before the main prompt
        enhanced = f"{existing_prompt}\n{context_text}\n"

        return enhanced

    def ingest_langchain_conversation(
        self,
        messages: List[Any],
        user_id: str,
        thread_id: str,
        session_id: str
    ) -> List:
        """
        Ingest LangChain conversation messages.

        Args:
            messages: List of LangChain message objects.
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.

        Returns:
            List of enriched Message objects.
        """
        return self.ingest_conversation(messages, user_id, thread_id, session_id)

    def create_langchain_callback(self, user_id: str, thread_id: str, session_id: str):
        """
        Create a LangChain callback handler for automatic ingestion.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.

        Returns:
            Callback handler instance.

        Note:
            This requires langchain to be installed.
            Usage:
                callback = adapter.create_langchain_callback("user123", "thread456", "session789")
                llm = ChatOpenAI(callbacks=[callback])
        """
        try:
            from langchain.callbacks.base import BaseCallbackHandler

            class MindcoreCallback(BaseCallbackHandler):
                """Callback handler that ingests messages into Mindcore."""

                def __init__(self, adapter, user_id, thread_id, session_id):
                    self.adapter = adapter
                    self.user_id = user_id
                    self.thread_id = thread_id
                    self.session_id = session_id

                def on_llm_end(self, response, **kwargs):
                    """Ingest LLM output."""
                    try:
                        for generation in response.generations:
                            for output in generation:
                                self.adapter.mindcore.ingest_message({
                                    'user_id': self.user_id,
                                    'thread_id': self.thread_id,
                                    'session_id': self.session_id,
                                    'role': 'assistant',
                                    'text': output.text
                                })
                    except Exception as e:
                        logger.error(f"Mindcore callback error during LLM end: {e}")

                def on_chat_model_start(self, serialized, messages, **kwargs):
                    """Ingest user messages."""
                    try:
                        for msg_list in messages:
                            for msg in msg_list:
                                msg_dict = self.adapter.format_message_for_ingestion(msg)
                                msg_dict.update({
                                    'user_id': self.user_id,
                                    'thread_id': self.thread_id,
                                    'session_id': self.session_id
                                })
                                self.adapter.mindcore.ingest_message(msg_dict)
                    except Exception as e:
                        logger.error(f"Mindcore callback error during chat model start: {e}")

            return MindcoreCallback(self, user_id, thread_id, session_id)

        except ImportError:
            raise ImportError("LangChain is not installed. Install with: pip install langchain")

    def as_langchain_memory(self, user_id: str, thread_id: str, session_id: str):
        """
        Create a LangChain-compatible memory interface.

        Args:
            user_id: User identifier.
            thread_id: Thread identifier.
            session_id: Session identifier.

        Returns:
            LangChain BaseChatMessageHistory compatible object.

        Note:
            This provides a read-only memory interface for LangChain.
        """
        try:
            from langchain.schema import BaseChatMessageHistory, HumanMessage, AIMessage

            class MindcoreMemory(BaseChatMessageHistory):
                """LangChain memory interface for Mindcore."""

                def __init__(self, adapter, user_id, thread_id, session_id):
                    self.adapter = adapter
                    self.user_id = user_id
                    self.thread_id = thread_id
                    self.session_id = session_id

                @property
                def messages(self):
                    """Get messages from Mindcore."""
                    # Fetch from cache/db
                    msgs = self.adapter.mindcore.cache.get_recent_messages(
                        self.user_id, self.thread_id
                    )

                    # Convert to LangChain format
                    lc_messages = []
                    for msg in msgs:
                        if msg.role.value == 'user':
                            lc_messages.append(HumanMessage(content=msg.raw_text))
                        elif msg.role.value == 'assistant':
                            lc_messages.append(AIMessage(content=msg.raw_text))

                    return lc_messages

                def add_message(self, message):
                    """Add message to Mindcore."""
                    msg_dict = self.adapter.format_message_for_ingestion(message)
                    msg_dict.update({
                        'user_id': self.user_id,
                        'thread_id': self.thread_id,
                        'session_id': self.session_id
                    })
                    self.adapter.mindcore.ingest_message(msg_dict)

                def clear(self):
                    """Clear memory (cache only)."""
                    self.adapter.mindcore.clear_cache(self.user_id, self.thread_id)

            return MindcoreMemory(self, user_id, thread_id, session_id)

        except ImportError:
            raise ImportError("LangChain is not installed. Install with: pip install langchain")
