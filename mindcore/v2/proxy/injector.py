"""Memory injection for Mindcore Proxy.

Injects relevant memories into Claude Code sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..flr import FLR, Memory


class InjectionStrategy(str, Enum):
    """Strategy for injecting memories into sessions."""

    PREPEND = "prepend"  # Add memories at start of context
    APPEND = "append"  # Add memories at end of context
    INTERLEAVE = "interleave"  # Mix memories with conversation
    SYSTEM = "system"  # Inject as system message


@dataclass
class InjectionResult:
    """Result of a memory injection operation.

    Attributes:
        injected_count: Number of memories injected
        total_tokens: Estimated token count of injected content
        memories: The memories that were injected
        formatted_content: The formatted injection content
    """

    injected_count: int = 0
    total_tokens: int = 0
    memories: list[Any] = field(default_factory=list)
    formatted_content: str = ""


class MemoryInjector:
    """Injects relevant memories from FLR into sessions.

    Uses the FLR recall mechanism to find relevant memories
    and formats them for injection into Claude Code sessions.
    """

    # Default templates for formatting memories
    DEFAULT_MEMORY_TEMPLATE = "- {content}"
    DEFAULT_WRAPPER_TEMPLATE = """<relevant_memories>
The following information may be relevant to this session:

{memories}
</relevant_memories>"""

    def __init__(
        self,
        flr: FLR,
        strategy: InjectionStrategy = InjectionStrategy.SYSTEM,
        max_memories: int = 5,
        max_tokens: int = 1000,
        memory_template: str | None = None,
        wrapper_template: str | None = None,
    ) -> None:
        """Initialize the memory injector.

        Args:
            flr: FLR instance for memory recall
            strategy: How to inject memories
            max_memories: Maximum number of memories to inject
            max_tokens: Maximum token budget for injection
            memory_template: Template for formatting each memory
            wrapper_template: Template for wrapping all memories
        """
        self.flr = flr
        self.strategy = strategy
        self.max_memories = max_memories
        self.max_tokens = max_tokens
        self.memory_template = memory_template or self.DEFAULT_MEMORY_TEMPLATE
        self.wrapper_template = wrapper_template or self.DEFAULT_WRAPPER_TEMPLATE

    def inject(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
    ) -> InjectionResult:
        """Inject relevant memories for a query.

        Args:
            query: Query to find relevant memories for
            user_id: User ID to recall memories for
            agent_id: Optional agent ID
            attention_hints: Optional topics to focus on

        Returns:
            InjectionResult with formatted memories
        """
        # Recall relevant memories
        recall_result = self.flr.recall(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            attention_hints=attention_hints or [],
            limit=self.max_memories,
        )

        if not recall_result.memories:
            return InjectionResult()

        # Format memories
        formatted_memories = []
        total_tokens = 0

        for memory in recall_result.memories:
            formatted = self.memory_template.format(
                content=memory.content,
                memory_type=memory.memory_type,
                topics=", ".join(memory.topics) if memory.topics else "",
            )

            # Estimate tokens (rough: ~4 chars per token)
            estimated_tokens = len(formatted) // 4

            if total_tokens + estimated_tokens > self.max_tokens:
                break

            formatted_memories.append(formatted)
            total_tokens += estimated_tokens

        if not formatted_memories:
            return InjectionResult()

        # Wrap memories
        memories_text = "\n".join(formatted_memories)
        wrapped = self.wrapper_template.format(memories=memories_text)

        return InjectionResult(
            injected_count=len(formatted_memories),
            total_tokens=total_tokens,
            memories=recall_result.memories[:len(formatted_memories)],
            formatted_content=wrapped,
        )

    def inject_for_messages(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
    ) -> InjectionResult:
        """Inject memories based on recent messages.

        Analyzes recent messages to build a query and find
        relevant memories.

        Args:
            messages: Recent session messages
            user_id: User ID
            agent_id: Optional agent ID

        Returns:
            InjectionResult with formatted memories
        """
        if not messages:
            return InjectionResult()

        # Build query from recent user messages
        user_messages = [
            m.get("content", "")
            for m in messages[-5:]
            if m.get("role") == "user"
        ]

        if not user_messages:
            return InjectionResult()

        query = " ".join(user_messages)[:500]

        # Extract topics from messages as attention hints
        attention_hints = self._extract_hints(messages)

        return self.inject(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            attention_hints=attention_hints,
        )

    def format_for_strategy(
        self,
        messages: list[dict[str, Any]],
        injection: InjectionResult,
    ) -> list[dict[str, Any]]:
        """Format messages with injection based on strategy.

        Args:
            messages: Original messages
            injection: Injection result

        Returns:
            Messages with memories injected according to strategy
        """
        if not injection.formatted_content:
            return messages

        if self.strategy == InjectionStrategy.SYSTEM:
            # Add as system message at the start
            return [
                {"role": "system", "content": injection.formatted_content},
                *messages,
            ]

        elif self.strategy == InjectionStrategy.PREPEND:
            # Prepend to first user message
            result = []
            prepended = False
            for msg in messages:
                if not prepended and msg.get("role") == "user":
                    result.append({
                        **msg,
                        "content": f"{injection.formatted_content}\n\n{msg['content']}",
                    })
                    prepended = True
                else:
                    result.append(msg)
            return result

        elif self.strategy == InjectionStrategy.APPEND:
            # Append to last user message
            result = list(messages)
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    result[i] = {
                        **result[i],
                        "content": f"{result[i]['content']}\n\n{injection.formatted_content}",
                    }
                    break
            return result

        else:  # INTERLEAVE
            # Just use system message approach
            return [
                {"role": "system", "content": injection.formatted_content},
                *messages,
            ]

    def _extract_hints(self, messages: list[dict[str, Any]]) -> list[str]:
        """Extract attention hints from messages.

        Args:
            messages: Session messages

        Returns:
            List of topic hints
        """
        hints = set()

        tech_keywords = [
            "python", "javascript", "typescript", "react", "vue",
            "api", "database", "sql", "file", "test", "error",
            "bug", "feature", "deploy", "config", "docker",
        ]

        for msg in messages[-10:]:
            content = msg.get("content", "").lower()
            for keyword in tech_keywords:
                if keyword in content:
                    hints.add(keyword)

        return list(hints)[:5]
