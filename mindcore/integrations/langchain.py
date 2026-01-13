"""LangChain integration for Mindcore.

Provides memory adapters that integrate Mindcore with LangChain chains and agents.

Example:
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from mindcore.integrations import MindcoreMemory

    memory = MindcoreMemory(
        storage="postgresql://localhost/mindcore",
        user_id="user_123",
    )

    chain = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory,
    )

    response = chain.predict(input="Hello!")
"""

from __future__ import annotations

from typing import Any

from mindcore import Mindcore


class MindcoreMemory:
    """LangChain-compatible memory backed by Mindcore.

    Implements the LangChain memory interface while delegating all
    storage and retrieval to Mindcore's FLR/CLST protocols.

    Benefits over LangChain's built-in memory:
    - PostgreSQL-first with deterministic scoring
    - Session aggregates for hierarchical retrieval
    - Cross-agent memory sharing
    - Full audit trail

    Example:
        memory = MindcoreMemory(
            storage="postgresql://localhost/mindcore",
            user_id="user_123",
            session_id="session_abc",
        )

        # Use with ConversationChain
        chain = ConversationChain(llm=llm, memory=memory)

        # Or use directly
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        variables = memory.load_memory_variables({})
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
                content=f"User: {input_str}",
                memory_type="episodic",
                user_id=self._user_id,
                topics=["conversation"],
                importance=0.5,
                session_id=self._session_id,
                agent_id=self._agent_id,
            )

        # Store assistant output as memory
        if output_str:
            self._mindcore.store(
                content=f"Assistant: {output_str}",
                memory_type="episodic",
                user_id=self._user_id,
                topics=["conversation"],
                importance=0.5,
                session_id=self._session_id,
                agent_id=self._agent_id,
            )

    def clear(self) -> None:
        """Clear memory (LangChain interface).

        Note: This only clears the current session if session_id is set.
        For full memory deletion, use the Mindcore API directly.
        """
        # Clear session-specific memories if session_id is set
        if self._session_id:
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

    def _format_as_string(self, memories: list[dict[str, Any]]) -> str:
        """Format memories as a string for buffer memory."""
        if not memories:
            return ""

        lines = []
        for mem in memories:
            content = mem.get("content", "")
            lines.append(content)

        return "\n".join(lines)

    def _format_as_messages(self, memories: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Format memories as message objects for chat models."""
        messages = []
        for mem in memories:
            content = mem.get("content", "")
            # Parse "User: " or "Assistant: " prefix
            if content.startswith("User: "):
                messages.append({"role": "user", "content": content[6:]})
            elif content.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": content[11:]})
            else:
                messages.append({"role": "system", "content": content})

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
    ) -> list[dict[str, Any]]:
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
