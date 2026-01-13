"""LangChain integration for Mindcore.

Provides memory adapters that integrate Mindcore with LangChain chains and agents.
Updated for LangChain v0.3.x (2025) patterns.

This module provides two integration approaches:

1. MindcoreChatMessageHistory - Implements BaseChatMessageHistory for use with
   RunnableWithMessageHistory and LangGraph. This is the recommended approach
   for LangChain v0.3.x.

2. MindcoreMemory - Legacy memory interface for backwards compatibility with
   older LangChain patterns.

Example (Modern - LangChain v0.3.x):
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_openai import ChatOpenAI
    from mindcore.integrations import MindcoreChatMessageHistory

    def get_session_history(session_id: str):
        return MindcoreChatMessageHistory(
            storage="postgresql://localhost/mindcore",
            session_id=session_id,
            user_id="user_123",
        )

    chain_with_history = RunnableWithMessageHistory(
        llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

Example (Legacy):
    from langchain.chains import ConversationChain
    from mindcore.integrations import MindcoreMemory

    memory = MindcoreMemory(
        storage="postgresql://localhost/mindcore",
        user_id="user_123",
    )
    chain = ConversationChain(llm=llm, memory=memory)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

from mindcore import Mindcore


# Type stubs - avoid hard dependencies
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from mindcore.flr import Memory


class BaseMessageLike(ABC):
    """Protocol for message-like objects when LangChain is not installed."""

    @property
    @abstractmethod
    def type(self) -> str:
        """Message type (human, ai, system)."""
        ...

    @property
    @abstractmethod
    def content(self) -> str:
        """Message content."""
        ...


class MindcoreChatMessageHistory:
    """LangChain v0.3.x compatible chat message history backed by Mindcore.

    Implements the BaseChatMessageHistory interface for use with
    RunnableWithMessageHistory and LangGraph persistence.

    This is the recommended integration for LangChain v0.3.x (2025+).

    Benefits:
    - PostgreSQL-first with deterministic scoring
    - Session aggregates for hierarchical retrieval
    - Cross-agent memory sharing
    - Full audit trail
    - Compatible with LangGraph checkpointers

    Example:
        from langchain_core.runnables.history import RunnableWithMessageHistory

        def get_session_history(session_id: str):
            return MindcoreChatMessageHistory(
                storage="postgresql://localhost/mindcore",
                session_id=session_id,
                user_id="user_123",
            )

        chain = RunnableWithMessageHistory(
            llm,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    """

    def __init__(
        self,
        storage: str = "sqlite:///mindcore.db",
        session_id: str = "default",
        user_id: str = "default",
        agent_id: str | None = None,
    ):
        """Initialize Mindcore chat message history.

        Args:
            storage: Mindcore storage connection string
            session_id: Session identifier for message grouping
            user_id: User identifier for memory isolation
            agent_id: Optional agent ID for multi-agent scenarios
        """
        self._mindcore = Mindcore(storage=storage)
        self._session_id = session_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._message_cache: list[dict[str, str]] = []

    @property
    def messages(self) -> list[Any]:
        """Retrieve all messages from this session.

        Returns:
            List of BaseMessage objects (or dicts if LangChain not installed)
        """
        # Try to use LangChain message types if available
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            memories = self._mindcore.search(
                user_id=self._user_id,
                topics=["chat", "conversation"],
                memory_types=["episodic"],
                limit=1000,
            )

            # Filter by session
            session_memories = [m for m in memories if m.session_id == self._session_id]

            # Sort by turn_index to maintain order
            session_memories.sort(key=lambda m: m.turn_index)

            messages = []
            for mem in session_memories:
                content = mem.content
                role = mem.message_role or "human"

                if role in ("human", "user"):
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                # Parse from content prefix as fallback
                elif content.startswith("Human: "):
                    messages.append(HumanMessage(content=content[7:]))
                elif content.startswith("AI: "):
                    messages.append(AIMessage(content=content[4:]))
                elif content.startswith("System: "):
                    messages.append(SystemMessage(content=content[8:]))
                else:
                    messages.append(HumanMessage(content=content))

            return messages

        except ImportError:
            # Fallback to dict format if LangChain not installed
            return self._get_messages_as_dicts()

    def _get_messages_as_dicts(self) -> list[dict[str, str]]:
        """Get messages as plain dicts (fallback when LangChain not installed)."""
        memories = self._mindcore.search(
            user_id=self._user_id,
            topics=["chat", "conversation"],
            memory_types=["episodic"],
            limit=1000,
        )

        session_memories = [m for m in memories if m.session_id == self._session_id]
        session_memories.sort(key=lambda m: m.turn_index)

        messages = []
        for mem in session_memories:
            role = mem.message_role or "human"
            messages.append({"role": role, "content": mem.content})

        return messages

    async def aget_messages(self) -> list[Any]:
        """Async variant of messages property.

        Returns:
            List of BaseMessage objects
        """
        # For now, delegate to sync version
        # TODO: Implement true async when Mindcore supports async operations
        return self.messages

    def add_messages(self, messages: Sequence[Any]) -> None:
        """Add messages to the history.

        This is the preferred method for adding multiple messages efficiently.

        Args:
            messages: Sequence of BaseMessage objects to add
        """
        current_count = len(self.messages)

        for i, message in enumerate(messages):
            # Extract role and content from message
            if hasattr(message, "type"):
                role = message.type  # LangChain message
                content = message.content
            elif isinstance(message, dict):
                role = message.get("role", "human")
                content = message.get("content", "")
            else:
                role = "human"
                content = str(message)

            # Map LangChain types to standard roles
            role_mapping = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
            }
            normalized_role = role_mapping.get(role, role)

            # Store in Mindcore
            self._mindcore.store(
                content=content,
                memory_type="episodic",
                user_id=self._user_id,
                topics=["chat", "conversation"],
                importance=0.6 if normalized_role == "assistant" else 0.5,
                session_id=self._session_id,
                agent_id=self._agent_id,
                message_role=normalized_role,
                turn_index=current_count + i,
            )

    async def aadd_messages(self, messages: Sequence[Any]) -> None:
        """Async variant for adding messages.

        Args:
            messages: Sequence of BaseMessage objects to add
        """
        # For now, delegate to sync version
        self.add_messages(messages)

    def add_message(self, message: Any) -> None:
        """Add a single message to the history.

        Args:
            message: BaseMessage object to add
        """
        self.add_messages([message])

    def add_user_message(self, message: str) -> None:
        """Convenience method to add a human message.

        Args:
            message: Message content string
        """
        try:
            from langchain_core.messages import HumanMessage

            self.add_messages([HumanMessage(content=message)])
        except ImportError:
            self.add_messages([{"role": "human", "content": message}])

    def add_ai_message(self, message: str) -> None:
        """Convenience method to add an AI message.

        Args:
            message: Message content string
        """
        try:
            from langchain_core.messages import AIMessage

            self.add_messages([AIMessage(content=message)])
        except ImportError:
            self.add_messages([{"role": "ai", "content": message}])

    def clear(self) -> None:
        """Clear all messages from this session."""
        memories = self._mindcore.search(
            user_id=self._user_id,
            memory_types=["episodic"],
            limit=10000,
        )

        for mem in memories:
            if mem.session_id == self._session_id:
                try:
                    self._mindcore.delete(mem.memory_id)
                except Exception:
                    pass

    async def aclear(self) -> None:
        """Async variant for clearing messages."""
        self.clear()


class MindcoreMemory:
    """Legacy LangChain memory interface backed by Mindcore.

    This class provides backwards compatibility with older LangChain
    patterns. For new implementations, prefer MindcoreChatMessageHistory
    with RunnableWithMessageHistory.

    Example:
        memory = MindcoreMemory(
            storage="postgresql://localhost/mindcore",
            user_id="user_123",
            session_id="session_abc",
        )

        # Use with ConversationChain (legacy pattern)
        chain = ConversationChain(llm=llm, memory=memory)
    """

    # LangChain memory interface attributes
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    def __init__(
        self,
        storage: str = "sqlite:///mindcore.db",
        user_id: str = "default",
        session_id: str | None = None,
        agent_id: str | None = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        k: int = 10,
    ):
        """Initialize Mindcore memory for LangChain.

        Args:
            storage: Mindcore storage connection string
            user_id: User identifier for memory isolation
            session_id: Optional session ID for conversation grouping
            agent_id: Optional agent ID for multi-agent scenarios
            memory_key: Key to use for memory in chain context
            input_key: Key for input in save_context
            output_key: Key for output in save_context
            return_messages: Return as message objects (for chat models)
            k: Number of memories to retrieve
        """
        self._mindcore = Mindcore(storage=storage)
        self._user_id = user_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._k = k

        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables (LangChain interface)."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory variables for the chain (LangChain interface).

        Retrieves relevant memories from Mindcore based on the current input.

        Args:
            inputs: Current chain inputs (may contain query hints)

        Returns:
            Dict with memory_key containing formatted memory string or messages
        """
        # Extract query from inputs if available
        query = inputs.get(self.input_key, "")

        if not query:
            # No query context, return recent memories
            memories = self._mindcore.search(
                user_id=self._user_id,
                memory_types=["episodic"],
                limit=self._k,
            )
        else:
            # Query-based retrieval
            result = self._mindcore.recall(
                query=str(query),
                user_id=self._user_id,
                agent_id=self._agent_id,
                limit=self._k,
            )
            memories = result.memories if hasattr(result, "memories") else []

        # Format memories
        if self.return_messages:
            return {self.memory_key: self._format_as_messages(memories)}
        return {self.memory_key: self._format_as_string(memories)}

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Save context from a conversation turn (LangChain interface).

        Stores the input/output pair as memories in Mindcore.

        Args:
            inputs: Input dict containing user message
            outputs: Output dict containing assistant response
        """
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")

        # Store user input as memory
        if input_str:
            self._mindcore.store(
                content=input_str,
                memory_type="episodic",
                user_id=self._user_id,
                topics=["conversation"],
                importance=0.5,
                session_id=self._session_id,
                agent_id=self._agent_id,
                message_role="user",
            )

        # Store assistant output as memory
        if output_str:
            self._mindcore.store(
                content=output_str,
                memory_type="episodic",
                user_id=self._user_id,
                topics=["conversation"],
                importance=0.6,
                session_id=self._session_id,
                agent_id=self._agent_id,
                message_role="assistant",
            )

    def clear(self) -> None:
        """Clear memory (LangChain interface).

        Note: This only clears the current session if session_id is set.
        For full memory deletion, use the Mindcore API directly.
        """
        if self._session_id:
            memories = self._mindcore.search(
                user_id=self._user_id,
                memory_types=["episodic"],
                limit=1000,
            )
            for mem in memories:
                if mem.session_id == self._session_id:
                    try:
                        self._mindcore.delete(mem.memory_id)
                    except Exception:
                        pass

    def _format_as_string(self, memories: list[Memory]) -> str:
        """Format memories as a string for buffer memory."""
        if not memories:
            return ""

        lines = []
        for mem in memories:
            role = getattr(mem, "message_role", None) or "User"
            content = mem.content
            lines.append(f"{role.capitalize()}: {content}")

        return "\n".join(lines)

    def _format_as_messages(self, memories: list[Memory]) -> list[dict[str, str]]:
        """Format memories as message objects for chat models."""
        messages = []
        for mem in memories:
            role = getattr(mem, "message_role", None) or "user"
            content = mem.content

            # Map to standard roles
            role_mapping = {"human": "user", "ai": "assistant"}
            role = role_mapping.get(role, role)

            messages.append({"role": role, "content": content})

        return messages

    # Additional convenience methods

    def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        topics: list[str] | None = None,
        importance: float = 0.5,
    ) -> str:
        """Add a memory directly.

        Args:
            content: Memory content
            memory_type: Type of memory
            topics: Relevant topics
            importance: Importance score 0-1

        Returns:
            Memory ID
        """
        return self._mindcore.store(
            content=content,
            memory_type=memory_type,
            user_id=self._user_id,
            topics=topics or [],
            importance=importance,
            session_id=self._session_id,
            agent_id=self._agent_id,
        )

    def search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories with a query.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching memories
        """
        result = self._mindcore.recall(
            query=query,
            user_id=self._user_id,
            agent_id=self._agent_id,
            limit=limit,
        )
        return result.memories if hasattr(result, "memories") else []
