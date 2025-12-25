"""Mindcore v2 - Universal Memory Layer for AI Agents.

A modern memory layer that provides:
- FLR (Fast Learning Recall) for inference-time memory access
- CLST (Cognitive Long-term Storage Transfer) for durable storage
- Structured output integration with LLM JSON schemas
- Multi-agent support with access control
- MCP and REST API interfaces

IMPORTANT: Direct Structured Output Required
-------------------------------------------
Your AI agent must output structured metadata directly. Mindcore does NOT
extract memories from unstructured responses. Configure your LLM to return:

{
    "response": "Your response to the user...",
    "memories_to_store": [
        {
            "content": "Memory content",
            "memory_type": "preference|episodic|semantic|...",
            "topics": ["topic1", "topic2"],
            "importance": 0.8
        }
    ]
}

Then store directly using mindcore.store() - no extraction layer needed.

Example:
    from mindcore.v2 import Mindcore

    # Initialize
    memory = Mindcore(storage="sqlite:///memory.db")

    # Store memory directly from LLM structured output
    for mem in llm_response["memories_to_store"]:
        memory.store(
            content=mem["content"],
            memory_type=mem["memory_type"],
            user_id="user123",
            topics=mem.get("topics", []),
            importance=mem.get("importance", 0.5),
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

from typing import Any

from .access import AccessController
from .clst import CLST, CompressionStrategy
from .exceptions import (
    MultiAgentNotEnabledError,
)
from .flr import FLR, Memory, RecallResult
from .storage import BaseStorage, SQLiteStorage
from .vocabulary import DEFAULT_VOCABULARY, VocabularySchema


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
        retention_policy: dict[str, Any] | None = None,
    ):
        """Initialize Mindcore.

        Args:
            storage: Storage backend or connection string
                - "sqlite:///path.db" for SQLite
                - "postgresql://..." for PostgreSQL
                - BaseStorage instance for custom backends
            vocabulary: Vocabulary schema for metadata control
            enable_multi_agent: Enable multi-agent access control
            retention_policy: Optional retention policy config:
                {
                    "episodic": {"max_age_days": 730},     # 2 years
                    "preference": {"max_age_days": None},  # Forever
                    "working": {"max_age_days": 1},        # 1 day
                    "default_max_age_days": 365,           # Default
                }
        """
        # Initialize storage
        if isinstance(storage, str):
            if storage.startswith("sqlite:///"):
                db_path = storage[10:]  # Remove "sqlite:///"
                self._storage = SQLiteStorage(db_path)
            elif storage.startswith(("postgresql://", "postgres://")):
                from mindcore.v2.storage.postgres import PostgresStorage

                self._storage = PostgresStorage(storage)
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
        # Pass access_controller as agent_registry for team-based access control
        self._flr = FLR(storage=self._storage, agent_registry=self._access_controller)
        self._clst = CLST(storage=self._storage, vocabulary=self._vocabulary)

        # Initialize retention policy if provided
        self._retention_policy = None
        if retention_policy:
            self._init_retention_policy(retention_policy)

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
            vocabulary_version=self._vocabulary.schema.version,
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

    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Args:
            memory_id: Memory identifier

        Raises:
            MemoryNotFoundError: If memory doesn't exist
        """
        self._clst.delete(memory_id)

    def reinforce(self, memory_id: str, signal: float) -> None:
        """Reinforce a memory with a learning signal.

        Positive signals increase future recall probability.
        Negative signals decrease it.

        Args:
            memory_id: Memory to reinforce
            signal: Signal from -1.0 to +1.0
        """
        self._flr.reinforce(memory_id, signal)

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
        return self._vocabulary.validate_memory(memory_data)

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

        Raises:
            MultiAgentNotEnabledError: If multi-agent mode is not enabled
        """
        if not self._access_controller:
            raise MultiAgentNotEnabledError

        profile = self._access_controller.register_agent(
            agent_id=agent_id,
            name=name,
            description=description,
            teams=teams,
        )
        return profile.to_dict()

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            MultiAgentNotEnabledError: If multi-agent mode is not enabled
            AgentNotFoundError: If agent doesn't exist
        """
        if not self._access_controller:
            raise MultiAgentNotEnabledError
        self._access_controller.unregister_agent(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents.

        Returns:
            List of agent profile dicts

        Raises:
            MultiAgentNotEnabledError: If multi-agent mode is not enabled
        """
        if not self._access_controller:
            raise MultiAgentNotEnabledError
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
        create_checkpoints: bool = True,
    ) -> dict[str, Any]:
        """Migrate memories to current vocabulary version.

        Args:
            from_version: Source vocabulary version
            user_id: Optional user filter
            create_checkpoints: Whether to create rollback checkpoints

        Returns:
            Migration result dict with checkpoint info for potential rollback

        Raises:
            ValueError: If no migration path exists
        """
        result = self._clst.migrate(
            from_version=from_version,
            user_id=user_id,
            create_checkpoints=create_checkpoints,
        )

        # Store result for potential rollback
        self._last_migration_result = result

        return {
            "from_version": result.from_version,
            "to_version": result.to_version,
            "memories_migrated": result.memories_migrated,
            "memories_failed": result.memories_failed,
            "errors": result.errors[:10],  # Limit errors shown
            "can_rollback": result.can_rollback,
            "checkpoint_count": len(result.checkpoints),
        }

    def rollback_vocabulary_migration(self) -> dict[str, Any]:
        """Rollback the last vocabulary migration.

        Uses checkpoints from the most recent migrate_vocabulary() call.

        Returns:
            Rollback result dict

        Raises:
            ValueError: If no migration to rollback or checkpoints unavailable
        """
        if not hasattr(self, "_last_migration_result") or not self._last_migration_result:
            raise ValueError("No migration to rollback. Run migrate_vocabulary() first.")

        result = self._clst.rollback_migration(self._last_migration_result)

        # Clear the stored migration result
        self._last_migration_result = None

        return {
            "from_version": result.from_version,
            "to_version": result.to_version,
            "memories_rolled_back": result.memories_migrated,
            "memories_failed": result.memories_failed,
            "errors": result.errors[:10],
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
            "vocabulary_version": self._vocabulary.schema.version,
            "multi_agent_enabled": self._access_controller is not None,
            "flr": self._flr.get_stats(),
            "clst": self._clst.get_stats(),
            "access": self._access_controller.get_stats() if self._access_controller else None,
        }

    # === GDPR/CCPA Compliance ===

    def gdpr_export(self, user_id: str) -> dict[str, Any]:
        """Export all user data for GDPR compliance (Article 15 - Right of Access).

        Args:
            user_id: User identifier

        Returns:
            Dict containing all user data in exportable format
        """
        from .enterprise.compliance import ComplianceManager

        compliance = ComplianceManager(self._storage)
        result = compliance.export_user_data(user_id)
        return result.to_dict()

    def gdpr_delete(self, user_id: str) -> dict[str, Any]:
        """Delete all user data for GDPR compliance (Article 17 - Right to Erasure).

        WARNING: This operation is irreversible.

        Args:
            user_id: User identifier

        Returns:
            Dict with deletion confirmation and statistics
        """
        from .enterprise.compliance import ComplianceManager

        compliance = ComplianceManager(self._storage)

        # Also invalidate cache if using smart cache
        if hasattr(self._flr, "invalidate_user_cache"):
            try:
                self._flr.invalidate_user_cache(user_id)
            except Exception:
                pass

        result = compliance.delete_user_data(user_id, clear_cache=True)
        return result.to_dict()

    def gdpr_anonymize(
        self,
        user_id: str,
        strategy: str = "pseudonymize",
    ) -> dict[str, Any]:
        """Anonymize user data while preserving analytics value.

        Args:
            user_id: User identifier
            strategy: Anonymization strategy:
                - "pseudonymize": Replace user_id with random ID
                - "hash": Hash user_id deterministically
                - "redact": Remove PII from content
                - "aggregate": Keep only metadata

        Returns:
            Dict with anonymization result and new user ID
        """
        from .enterprise.compliance import AnonymizationStrategy, ComplianceManager

        compliance = ComplianceManager(self._storage)

        try:
            strategy_enum = AnonymizationStrategy(strategy)
        except ValueError:
            strategy_enum = AnonymizationStrategy.PSEUDONYMIZE

        result = compliance.anonymize_user_data(user_id, strategy=strategy_enum)
        return result.to_dict()

    def get_user_data_summary(self, user_id: str) -> dict[str, Any]:
        """Get summary of data held for a user (for data access requests).

        Args:
            user_id: User identifier

        Returns:
            Summary dict with counts and date ranges
        """
        from .enterprise.compliance import ComplianceManager

        compliance = ComplianceManager(self._storage)
        return compliance.get_user_data_summary(user_id)

    def _init_retention_policy(self, policy_config: dict[str, Any]) -> None:
        """Initialize retention policy from config dict.

        Args:
            policy_config: Retention policy configuration
        """
        from .enterprise.compliance import RetentionPolicy

        # Convert user-friendly format to RetentionPolicy
        memory_type_policies = {}
        default_max_age = policy_config.get("default_max_age_days", 365)

        for key, value in policy_config.items():
            if key == "default_max_age_days":
                continue
            if isinstance(value, dict):
                memory_type_policies[key] = value.get("max_age_days")
            elif isinstance(value, int):
                memory_type_policies[key] = value
            elif value is None:
                memory_type_policies[key] = None

        self._retention_policy = RetentionPolicy(
            memory_type_policies=memory_type_policies,
            default_max_age_days=default_max_age,
        )

    def enforce_retention(
        self,
        user_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Enforce retention policy by deleting expired memories.

        Call this periodically (e.g., daily cron job) to clean up old data.

        Args:
            user_id: Optional user ID to limit enforcement to
            dry_run: If True, only count without deleting

        Returns:
            Dict with enforcement results
        """
        from .enterprise.compliance import ComplianceManager

        compliance = ComplianceManager(
            self._storage,
            retention_policy=self._retention_policy,
        )
        result = compliance.enforce_retention(user_id=user_id, dry_run=dry_run)
        return result.to_dict()

    def get_retention_status(self, user_id: str | None = None) -> dict[str, Any]:
        """Check what would be affected by retention enforcement.

        Args:
            user_id: Optional user ID to check

        Returns:
            Dict with affected memory counts by type
        """
        from .enterprise.compliance import ComplianceManager

        compliance = ComplianceManager(
            self._storage,
            retention_policy=self._retention_policy,
        )
        return compliance.check_retention_status(user_id=user_id)

    def close(self) -> None:
        """Close all connections."""
        self._flr.flush_reinforcements()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
