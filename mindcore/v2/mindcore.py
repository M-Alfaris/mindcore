"""Mindcore v2 - Universal Memory Layer for AI Agents.

A modern memory layer that provides:
- FLR (Fast Learning Recall) for inference-time memory access
- CLST (Cognitive Long-term Storage Transfer) for durable storage
- Structured output integration with LLM JSON schemas
- Auto-extraction of memories from conversations
- Multi-agent support with access control
- MCP and REST API interfaces

Example:
    from mindcore.v2 import Mindcore

    # Initialize
    memory = Mindcore(storage="sqlite:///memory.db")

    # Store memory
    memory.store(
        content="User prefers email communication",
        memory_type="preference",
        user_id="user123",
        topics=["communication"],
    )

    # Recall relevant memories
    result = memory.recall(
        query="How should I contact the user?",
        user_id="user123",
    )

    # Get JSON schema for LLM structured output
    schema = memory.get_json_schema()
"""

from __future__ import annotations

from typing import Any, Callable

from .access import AccessController, Permission
from .clst import CLST, CompressionStrategy
from .extraction import MemoryExtractor
from .flr import FLR, Memory, RecallResult
from .storage import SQLiteStorage, BaseStorage
from .vocabulary import VocabularySchema, DEFAULT_VOCABULARY


class Mindcore:
    """Universal memory layer for AI agents.

    Integrates FLR, CLST, and all supporting components into a
    unified, easy-to-use interface.

    Example:
        # Simple usage
        memory = Mindcore()
        memory.store("User likes Python", "preference", "user123", ["programming"])
        result = memory.recall("programming preferences", "user123")

        # With vocabulary
        vocab = VocabularySchema(
            version="1.0.0",
            topics=["support", "billing", "product"],
        )
        memory = Mindcore(vocabulary=vocab)

        # Multi-agent
        memory = Mindcore(enable_multi_agent=True)
        memory.register_agent("support_bot", "Support Agent", teams=["support"])
    """

    def __init__(
        self,
        storage: str | BaseStorage = "sqlite:///mindcore.db",
        vocabulary: VocabularySchema | None = None,
        enable_multi_agent: bool = False,
        auto_extract_llm: Callable[[str], str] | None = None,
    ):
        """Initialize Mindcore.

        Args:
            storage: Storage backend or connection string
                - "sqlite:///path.db" for SQLite
                - BaseStorage instance for custom backends
            vocabulary: Vocabulary schema for metadata control
            enable_multi_agent: Enable multi-agent access control
            auto_extract_llm: LLM function for enhanced auto-extraction
        """
        # Initialize storage
        if isinstance(storage, str):
            if storage.startswith("sqlite:///"):
                db_path = storage[10:]  # Remove "sqlite:///"
                self._storage = SQLiteStorage(db_path)
            else:
                # Default to SQLite
                self._storage = SQLiteStorage(storage)
        else:
            self._storage = storage

        # Initialize vocabulary
        self._vocabulary = vocabulary or DEFAULT_VOCABULARY

        # Initialize access controller
        self._access_controller = AccessController() if enable_multi_agent else None

        # Initialize FLR and CLST
        self._flr = FLR(storage=self._storage)
        self._clst = CLST(storage=self._storage, vocabulary=self._vocabulary)

        # Initialize extractor
        self._extractor = MemoryExtractor(
            vocabulary=self._vocabulary,
            auto_extract_llm=auto_extract_llm,
        )

    # === Core Memory Operations ===

    def store(
        self,
        content: str,
        memory_type: str,
        user_id: str,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        importance: float = 0.5,
        entities: list[str] | None = None,
        access_level: str = "private",
        agent_id: str | None = None,
    ) -> str:
        """Store a memory.

        Args:
            content: Memory content
            memory_type: Type (episodic, semantic, preference, etc.)
            user_id: User identifier
            topics: Relevant topics
            categories: Categories
            importance: Importance score 0-1
            entities: Extracted entities
            access_level: Access level for multi-agent
            agent_id: Agent storing the memory

        Returns:
            Memory ID
        """
        memory = Memory(
            memory_id="",
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=agent_id,
            topics=topics or [],
            categories=categories or [],
            importance=importance,
            entities=entities or [],
            access_level=access_level,
            vocabulary_version=self._vocabulary.version,
        )

        return self._clst.store(memory)

    def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """Fast recall of relevant memories.

        Uses FLR for optimized retrieval with scoring.

        Args:
            query: Query or current context
            user_id: User identifier
            agent_id: Agent requesting (for access control)
            attention_hints: Topics to prioritize
            memory_types: Filter by memory types
            limit: Max memories to return

        Returns:
            RecallResult with scored memories
        """
        return self._flr.query(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            attention_hints=attention_hints,
            memory_types=memory_types,
            limit=limit,
        )

    def search(
        self,
        user_id: str,
        query: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Search memories with filters.

        Uses CLST for comprehensive search.

        Args:
            user_id: User identifier
            query: Text search query
            topics: Filter by topics
            categories: Filter by categories
            memory_types: Filter by memory types
            limit: Max results

        Returns:
            List of matching memories
        """
        return self._clst.search(
            query=query,
            user_id=user_id,
            topics=topics,
            categories=categories,
            memory_types=memory_types,
            limit=limit,
        )

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        return self._clst.retrieve(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        return self._clst.delete(memory_id)

    def reinforce(self, memory_id: str, signal: float) -> None:
        """Reinforce a memory with a learning signal.

        Positive signals increase future recall probability.
        Negative signals decrease it.

        Args:
            memory_id: Memory to reinforce
            signal: Signal from -1.0 to +1.0
        """
        self._flr.reinforce(memory_id, signal)

    # === Extraction ===

    def extract_from_response(
        self,
        llm_response: dict[str, Any],
        user_id: str,
        agent_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        auto_store: bool = True,
    ) -> list[Memory]:
        """Extract memories from LLM response.

        Combines structured output parsing and auto-extraction.

        Args:
            llm_response: LLM structured output
            user_id: User identifier
            agent_id: Agent identifier
            messages: Conversation messages for auto-extraction
            auto_store: Automatically store extracted memories

        Returns:
            List of extracted memories
        """
        result = self._extractor.extract_from_response(
            llm_response=llm_response,
            user_id=user_id,
            agent_id=agent_id,
            messages=messages,
        )

        if auto_store:
            for memory in result.memories:
                self._clst.store(memory, validate=False)

        return result.memories

    def auto_extract(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
        auto_store: bool = True,
    ) -> list[Memory]:
        """Auto-extract memories from conversation.

        Mem0-style extraction of implicit memories.

        Args:
            messages: Conversation messages
            user_id: User identifier
            agent_id: Agent identifier
            auto_store: Automatically store extracted memories

        Returns:
            List of extracted memories
        """
        result = self._extractor.auto_extract(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
        )

        if auto_store:
            for memory in result.memories:
                self._clst.store(memory, validate=False)

        return result.memories

    # === Vocabulary ===

    def get_json_schema(self, include_response: bool = True) -> dict[str, Any]:
        """Get JSON schema for LLM structured output.

        Args:
            include_response: Include response field

        Returns:
            JSON Schema dict
        """
        return self._vocabulary.to_json_schema(include_response=include_response)

    def get_vocabulary_instructions(self) -> str:
        """Get vocabulary instructions for LLM prompts."""
        return self._vocabulary.to_prompt_instructions()

    def validate_memory(self, memory_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate memory data against vocabulary.

        Args:
            memory_data: Memory dict to validate

        Returns:
            (is_valid, list of errors)
        """
        return self._vocabulary.validate(memory_data)

    # === Multi-Agent ===

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        teams: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register an agent for multi-agent access control.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable name
            description: Agent description
            teams: Team memberships

        Returns:
            Agent profile dict
        """
        if not self._access_controller:
            raise RuntimeError("Multi-agent not enabled. Initialize with enable_multi_agent=True")

        profile = self._access_controller.register_agent(
            agent_id=agent_id,
            name=name,
            description=description,
            teams=teams,
        )
        return profile.to_dict()

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if not self._access_controller:
            return False
        return self._access_controller.unregister_agent(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents."""
        if not self._access_controller:
            return []
        return [a.to_dict() for a in self._access_controller.list_agents()]

    # === CLST Operations ===

    def compress(
        self,
        user_id: str,
        older_than_days: int = 30,
        strategy: str = "summarize",
    ) -> dict[str, Any]:
        """Compress old memories.

        Args:
            user_id: User whose memories to compress
            older_than_days: Only compress memories older than this
            strategy: Compression strategy (summarize, merge, deduplicate)

        Returns:
            Compression result dict
        """
        from datetime import timedelta

        try:
            strategy_enum = CompressionStrategy(strategy)
        except ValueError:
            strategy_enum = CompressionStrategy.SUMMARIZE

        result = self._clst.compress(
            user_id=user_id,
            older_than=timedelta(days=older_than_days),
            strategy=strategy_enum,
        )

        return {
            "original_count": result.original_count,
            "compressed_count": result.compressed_count,
            "compression_ratio": result.compression_ratio,
            "removed_count": len(result.removed_memory_ids),
        }

    def sync(
        self,
        source_agent: str,
        target_agent: str,
        user_id: str,
        memory_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync memories between agents.

        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            user_id: User context
            memory_types: Types to sync

        Returns:
            Sync result dict
        """
        result = self._clst.sync(
            source_agent=source_agent,
            target_agent=target_agent,
            user_id=user_id,
            memory_types=memory_types,
        )

        return {
            "memories_transferred": result.memories_transferred,
            "conflicts_resolved": result.conflicts_resolved,
            "errors": result.errors,
        }

    def migrate_vocabulary(
        self,
        from_version: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Migrate memories to current vocabulary version.

        Args:
            from_version: Source vocabulary version
            user_id: Optional user filter

        Returns:
            Migration result dict
        """
        result = self._clst.migrate(
            from_version=from_version,
            user_id=user_id,
        )

        return {
            "from_version": result.from_version,
            "to_version": result.to_version,
            "memories_migrated": result.memories_migrated,
            "memories_failed": result.memories_failed,
            "errors": result.errors[:10],  # Limit errors shown
        }

    # === Server ===

    def get_mcp_server(self):
        """Get MCP server instance for native LLM integration."""
        from .server import MCPServer

        return MCPServer(
            flr=self._flr,
            clst=self._clst,
            vocabulary=self._vocabulary,
            access_controller=self._access_controller,
        )

    def serve_rest(self, host: str = "0.0.0.0", port: int = 8000):
        """Start REST API server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        from .server import run_server

        run_server(
            flr=self._flr,
            clst=self._clst,
            vocabulary=self._vocabulary,
            access_controller=self._access_controller,
            host=host,
            port=port,
        )

    # === Stats ===

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        return {
            "vocabulary_version": self._vocabulary.version,
            "multi_agent_enabled": self._access_controller is not None,
            "flr": self._flr.get_stats(),
            "clst": self._clst.get_stats(),
            "access": self._access_controller.get_stats() if self._access_controller else None,
        }

    def close(self) -> None:
        """Close all connections."""
        self._flr.flush_reinforcements()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
