"""LlamaIndex integration for Mindcore.

Provides memory adapters that integrate Mindcore with LlamaIndex chat engines
and agents.

Example:
    from llama_index.core import VectorStoreIndex
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

from typing import Any

from mindcore import Mindcore


class MindcoreIndexMemory:
    """LlamaIndex-compatible memory backed by Mindcore.

    Implements a memory interface compatible with LlamaIndex chat engines
    while delegating all storage and retrieval to Mindcore's FLR/CLST protocols.

    Benefits over LlamaIndex's built-in memory:
    - PostgreSQL-first with deterministic scoring
    - No vector embeddings required (faster, cheaper)
    - Session aggregates for hierarchical retrieval
    - Cross-agent memory sharing

    Example:
        memory = MindcoreIndexMemory(
            storage="postgresql://localhost/mindcore",
            user_id="user_123",
        )

        # Use with chat engine
        chat_engine = index.as_chat_engine(memory=memory)

        # Or use directly
        memory.put("user_123", {"role": "user", "content": "Hello"})
        messages = memory.get_all()
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
            content=f"{role.capitalize()}: {content}",
            memory_type="episodic",
            user_id=self._user_id,
            topics=["chat", role],
            importance=0.6 if role == "assistant" else 0.5,
            session_id=self._session_id,
        )

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
                if mem.get("session_id") == self._session_id:
                    try:
                        self._mindcore.delete(mem["memory_id"])
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

    def _memories_to_messages(self, memories: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Convert Mindcore memories to LlamaIndex message format."""
        messages = []
        for mem in memories:
            content = mem.get("content", "")

            # Parse role from content prefix
            if content.startswith("User: "):
                messages.append({"role": "user", "content": content[6:]})
            elif content.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": content[11:]})
            elif content.startswith("System: "):
                messages.append({"role": "system", "content": content[8:]})
            else:
                # Default to user message
                messages.append({"role": "user", "content": content})

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
    ) -> list[dict[str, Any]]:
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
                    parts.append(f"- {mem.get('content', '')}")
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
