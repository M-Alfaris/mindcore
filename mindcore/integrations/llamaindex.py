"""LlamaIndex integration for Mindcore.

Provides memory adapters that integrate Mindcore with LlamaIndex chat engines
and agents. Updated for LlamaIndex 2025 patterns.

This module provides two integration approaches:

1. MindcoreMemoryBlock - Implements BaseMemoryBlock for use with the new
   LlamaIndex Memory class. This is the recommended approach for 2025+.

2. MindcoreIndexMemory - Legacy memory interface for backwards compatibility
   with older ChatMemoryBuffer patterns (deprecated in LlamaIndex).

Example (Modern - Memory class with memory blocks):
    from llama_index.core.memory import Memory
    from mindcore.integrations import MindcoreMemoryBlock

    memory = Memory.from_defaults(
        session_id="my_session",
        token_limit=40000,
        memory_blocks=[
            MindcoreMemoryBlock(
                storage="postgresql://localhost/mindcore",
                user_id="user_123",
                priority=1,
            ),
        ],
    )

    agent = FunctionAgent.from_tools(
        tools=[...],
        memory=memory,
    )

Example (Legacy):
    from mindcore.integrations import MindcoreIndexMemory

    memory = MindcoreIndexMemory(
        storage="postgresql://localhost/mindcore",
        user_id="user_123",
    )

    chat_engine = index.as_chat_engine(
        memory=memory,
        chat_mode="context",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mindcore import Mindcore


if TYPE_CHECKING:
    from mindcore.flr import Memory


@dataclass
class MindcoreMemoryBlock:
    """LlamaIndex 2025 compatible memory block backed by Mindcore.

    Implements the BaseMemoryBlock interface for use with the new
    LlamaIndex Memory class and agent memory system.

    This is the recommended integration for LlamaIndex 2025+.

    Benefits:
    - PostgreSQL-first with deterministic scoring
    - Semantic search for relevant context retrieval
    - Cross-agent memory sharing
    - Full audit trail
    - Priority-based memory truncation

    Example:
        from llama_index.core.memory import Memory

        memory = Memory.from_defaults(
            session_id="my_session",
            memory_blocks=[
                MindcoreMemoryBlock(
                    storage="postgresql://localhost/mindcore",
                    user_id="user_123",
                    priority=1,
                ),
            ],
        )

    Attributes:
        priority: Priority level for truncation (0 = never truncate)
    """

    storage: str = "sqlite:///mindcore.db"
    user_id: str = "default"
    session_id: str | None = None
    priority: int = 1  # 0 = never truncate, 1+ = truncation order
    token_limit: int = 4000
    tokenizer_fn: Callable[[str], list] | None = None
    _mindcore: Mindcore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize Mindcore connection."""
        self._mindcore = Mindcore(storage=self.storage)

    @property
    def mindcore(self) -> Mindcore:
        """Get Mindcore instance."""
        if self._mindcore is None:
            self._mindcore = Mindcore(storage=self.storage)
        return self._mindcore

    async def _aget(
        self,
        messages: list[dict[str, str]] | None = None,
        **block_kwargs: Any,
    ) -> str:
        """Async retrieval of relevant memories.

        This is called by the Memory class to get context for the agent.

        Args:
            messages: Recent chat messages for context
            **block_kwargs: Additional keyword arguments

        Returns:
            Formatted string of relevant memories
        """
        # Extract query from recent messages if available
        query = ""
        if messages:
            # Use the last user message as query context
            for msg in reversed(messages):
                role = msg.get("role", "")
                if role in ("user", "human"):
                    query = msg.get("content", "")
                    break

        if query:
            # Semantic search based on query
            result = self.mindcore.recall(
                query=query,
                user_id=self.user_id,
                limit=10,
            )
            memories = result.memories if hasattr(result, "memories") else []
        else:
            # Return recent memories
            memories = self.mindcore.search(
                user_id=self.user_id,
                memory_types=["episodic", "semantic"],
                limit=10,
            )

        # Filter by session if set
        if self.session_id:
            memories = [m for m in memories if m.session_id == self.session_id]

        # Format memories as context string
        if not memories:
            return ""

        lines = ["Relevant context from memory:"]
        for mem in memories:
            lines.append(f"- {mem.content}")

        return "\n".join(lines)

    async def _aput(self, messages: list[dict[str, str]]) -> None:
        """Async storage of messages to memory.

        This is called by the Memory class to persist messages.

        Args:
            messages: Messages to store
        """
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if content:
                self.mindcore.store(
                    content=content,
                    memory_type="episodic",
                    user_id=self.user_id,
                    topics=["chat", role],
                    importance=0.6 if role in ("assistant", "ai") else 0.5,
                    session_id=self.session_id,
                )

    async def atruncate(self, content: str, tokens_to_truncate: int) -> str:
        """Truncate content when memory exceeds token limit.

        Args:
            content: Current content string
            tokens_to_truncate: Number of tokens to remove

        Returns:
            Truncated content string
        """
        if not content:
            return ""

        # Simple truncation: remove from the beginning (oldest context)
        lines = content.split("\n")

        # Estimate tokens per line (rough: 1 token per 4 chars)
        tokens_removed = 0
        truncated_lines = []

        for line in reversed(lines):
            line_tokens = len(line) // 4
            if tokens_removed < tokens_to_truncate:
                tokens_removed += line_tokens
            else:
                truncated_lines.insert(0, line)

        return "\n".join(truncated_lines)

    def get(
        self,
        messages: list[dict[str, str]] | None = None,
        **block_kwargs: Any,
    ) -> str:
        """Sync retrieval of relevant memories.

        Args:
            messages: Recent chat messages for context
            **block_kwargs: Additional keyword arguments

        Returns:
            Formatted string of relevant memories
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._aget(messages, **block_kwargs))

    def put(self, messages: list[dict[str, str]]) -> None:
        """Sync storage of messages to memory.

        Args:
            messages: Messages to store
        """
        import asyncio

        asyncio.get_event_loop().run_until_complete(self._aput(messages))


class MindcoreIndexMemory:
    """Legacy LlamaIndex memory interface backed by Mindcore.

    This class provides backwards compatibility with older LlamaIndex
    ChatMemoryBuffer patterns. For new implementations, prefer
    MindcoreMemoryBlock with the Memory class.

    Example:
        memory = MindcoreIndexMemory(
            storage="postgresql://localhost/mindcore",
            user_id="user_123",
        )

        chat_engine = index.as_chat_engine(memory=memory)
    """

    def __init__(
        self,
        storage: str = "sqlite:///mindcore.db",
        user_id: str = "default",
        session_id: str | None = None,
        token_limit: int = 4000,
    ):
        """Initialize Mindcore memory for LlamaIndex.

        Args:
            storage: Mindcore storage connection string
            user_id: User identifier for memory isolation
            session_id: Optional session ID for conversation grouping
            token_limit: Approximate token limit for context window
        """
        self._mindcore = Mindcore(storage=storage)
        self._user_id = user_id
        self._session_id = session_id
        self._token_limit = token_limit
        self._chat_history: list[dict[str, str]] = []

    def put(self, message: dict[str, str]) -> str:
        """Store a chat message (LlamaIndex interface).

        Args:
            message: Message dict with 'role' and 'content' keys

        Returns:
            Memory ID
        """
        role = message.get("role", "user")
        content = message.get("content", "")

        # Add to local chat history
        self._chat_history.append(message)

        # Store in Mindcore
        return self._mindcore.store(
            content=content,
            memory_type="episodic",
            user_id=self._user_id,
            topics=["chat", role],
            importance=0.6 if role == "assistant" else 0.5,
            session_id=self._session_id,
        )

    def put_messages(self, messages: list[dict[str, str]]) -> None:
        """Store multiple chat messages (LlamaIndex 2025 interface).

        Args:
            messages: List of message dicts to store
        """
        for msg in messages:
            self.put(msg)

    def get(self, limit: int = 10) -> list[dict[str, str]]:
        """Get recent chat messages (LlamaIndex interface).

        Args:
            limit: Maximum messages to retrieve

        Returns:
            List of message dicts with 'role' and 'content'
        """
        # First check local history
        if self._chat_history:
            return self._chat_history[-limit:]

        # Fall back to Mindcore search
        memories = self._mindcore.search(
            user_id=self._user_id,
            topics=["chat"],
            memory_types=["episodic"],
            limit=limit,
        )

        return self._memories_to_messages(memories)

    def get_all(self) -> list[dict[str, str]]:
        """Get all chat messages for the session (LlamaIndex interface).

        Returns:
            List of all message dicts
        """
        if self._chat_history:
            return self._chat_history.copy()

        memories = self._mindcore.search(
            user_id=self._user_id,
            topics=["chat"],
            memory_types=["episodic"],
            limit=1000,
        )

        return self._memories_to_messages(memories)

    def reset(self) -> None:
        """Clear chat history (LlamaIndex interface)."""
        self._chat_history.clear()

        if self._session_id:
            # Clear session-specific memories
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

    def set_context(self, context: str) -> str:
        """Set context for the conversation.

        Args:
            context: Context string to remember

        Returns:
            Memory ID
        """
        return self._mindcore.store(
            content=context,
            memory_type="semantic",
            user_id=self._user_id,
            topics=["context"],
            importance=0.8,
            session_id=self._session_id,
        )

    def _memories_to_messages(self, memories: list[Memory]) -> list[dict[str, str]]:
        """Convert Mindcore memories to LlamaIndex message format."""
        messages = []
        for mem in memories:
            content = mem.content

            # Check for stored role in topics
            role = "user"
            if "assistant" in mem.topics:
                role = "assistant"
            elif "system" in mem.topics:
                role = "system"

            # Parse role from content prefix as fallback
            if content.startswith("User: "):
                role = "user"
                content = content[6:]
            elif content.startswith("Assistant: "):
                role = "assistant"
                content = content[11:]
            elif content.startswith("System: "):
                role = "system"
                content = content[8:]

            messages.append({"role": role, "content": content})

        return messages

    # Additional LlamaIndex-compatible methods

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Get chat history as property."""
        return self.get_all()

    def add_user_message(self, content: str) -> str:
        """Add a user message.

        Args:
            content: Message content

        Returns:
            Memory ID
        """
        return self.put({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> str:
        """Add an assistant message.

        Args:
            content: Message content

        Returns:
            Memory ID
        """
        return self.put({"role": "assistant", "content": content})

    def search_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[Memory]:
        """Search for relevant context based on query.

        This is useful for context-aware chat engines that need
        to retrieve relevant memories beyond just chat history.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        result = self._mindcore.recall(
            query=query,
            user_id=self._user_id,
            limit=limit,
        )

        return result.memories if hasattr(result, "memories") else []

    def get_context_string(
        self,
        query: str | None = None,
        include_chat_history: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """Get formatted context string for LLM.

        Args:
            query: Optional query for relevance-based retrieval
            include_chat_history: Include recent chat messages
            max_tokens: Override default token limit

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self._token_limit
        parts = []
        char_limit = max_tokens * 4  # Rough chars-to-tokens

        # Add relevant memories if query provided
        if query:
            memories = self.search_context(query, limit=5)
            if memories:
                parts.append("Relevant context:")
                for mem in memories:
                    parts.append(f"- {mem.content}")
                parts.append("")

        # Add chat history
        if include_chat_history:
            messages = self.get(limit=20)
            if messages:
                parts.append("Chat history:")
                for msg in messages:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    parts.append(f"{role}: {content}")

        result = "\n".join(parts)

        # Truncate if too long
        if len(result) > char_limit:
            result = result[:char_limit] + "..."

        return result
