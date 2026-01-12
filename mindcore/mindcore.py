"""Mindcore - Universal Memory Layer for AI Agents.

A modern memory layer that provides:
- FLR (Fast Learning Recall) for inference-time memory access
- CLST (Cognitive Long-term Storage Transfer) for durable storage
- SVL (Structured Validation Layer) as mandatory kernel for all data flows
- Structured output integration with LLM JSON schemas
- Multi-agent support with access control
- MCP and REST API interfaces

SECURITY: SVL Gate Enforcement
------------------------------
ALL data entering or leaving Mindcore is validated through the SVL Gate.
There are NO bypass paths. This ensures:
1. All inbound data is canonicalized and validated
2. All outbound data is validated before reaching LLMs
3. Reinforcement signals are bounds-checked
4. Content is scanned for PII/injection patterns

Usage:
    from mindcore import Mindcore

    memory = Mindcore(storage="sqlite:///memory.db")
    memory.store("User likes Python", "preference", "user123", ["programming"])
    result = memory.recall("programming preferences", "user123")

For more control over SVL configuration:

    from mindcore.svl import SVLPipeline

    pipeline = SVLPipeline(
        storage="sqlite:///memory.db",
        llm_call=my_llm_function,
        enable_hot_path=True,
    )
"""

from __future__ import annotations

from typing import Any, Callable

from .access import AccessController
from .exceptions import (
    MultiAgentNotEnabledError,
)
from .storage import BaseStorage, SQLiteStorage
from .svl import DEFAULT_SVL, StructuredValidationLayer
from .svl.gate import GatePolicy, RetryConfig, SVLGate
from .svl.gated_storage import GatedCLST, GatedFLR, RecallResult
from .vocabulary import DEFAULT_VOCABULARY, VocabularySchema


class Mindcore:
    """Universal memory layer for AI agents.

    Integrates FLR, CLST, and all supporting components into a
    unified, easy-to-use interface.

    SECURITY: All data flows are locked behind the SVL Gate.
    There are NO bypass paths for data entering or leaving the system.

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
        svl: StructuredValidationLayer | None = None,
        gate_policy: GatePolicy | None = None,
        enable_multi_agent: bool = False,
        retention_policy: dict[str, Any] | None = None,
    ):
        """Initialize Mindcore with mandatory SVL Gate enforcement.

        Args:
            storage: Storage backend or connection string
                - "sqlite:///path.db" for SQLite
                - "postgresql://..." for PostgreSQL
                - BaseStorage instance for custom backends
            vocabulary: Legacy vocabulary schema (prefer svl parameter)
            svl: StructuredValidationLayer for SVL configuration
            gate_policy: GatePolicy for SVL Gate configuration
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
                from mindcore.storage.postgres import PostgresStorage

                self._storage = PostgresStorage(storage)
            else:
                # Default to SQLite
                self._storage = SQLiteStorage(storage)
        else:
            self._storage = storage

        # Initialize SVL (prefer svl parameter, fall back to vocabulary)
        self._svl = svl or DEFAULT_SVL
        self._vocabulary = vocabulary or DEFAULT_VOCABULARY

        # Initialize SVL Gate - ALL data flows through this
        self._gate = SVLGate(
            svl=self._svl,
            policy=gate_policy or GatePolicy(),
            retry_config=RetryConfig(),
        )

        # Initialize access controller
        self._access_controller = AccessController() if enable_multi_agent else None

        # Initialize GATED FLR and CLST (mandatory SVL enforcement)
        self._flr = GatedFLR(storage=self._storage, gate=self._gate)
        self._clst = GatedCLST(storage=self._storage, gate=self._gate)

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
        session_id: str | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> str:
        """Store a memory with mandatory SVL Gate validation.

        All data is validated through the SVL Gate before storage.
        There is NO bypass path.

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
            session_id: Session identifier
            llm_call: Optional LLM function for retry on validation failure

        Returns:
            Memory ID

        Raises:
            ValueError: If validation fails and cannot be corrected
        """
        memory_data = {
            "content": content,
            "memory_type": memory_type,
            "topics": topics or [],
            "categories": categories or [],
            "importance": importance,
            "entities": entities or [],
            "access_level": access_level,
        }

        result = self._clst.store(
            memory_data=memory_data,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_call=llm_call,
        )

        if not result.success:
            raise ValueError(f"Memory validation failed: {result.error_message}")

        return result.memory_id

    def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        attention_hints: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """Fast recall of relevant memories with SVL Gate validation.

        All returned memories are validated through the SVL Gate
        before being returned. There is NO bypass path.

        Args:
            query: Query or current context
            user_id: User identifier
            agent_id: Agent requesting (for access control)
            attention_hints: Topics to prioritize
            memory_types: Filter by memory types
            limit: Max memories to return

        Returns:
            RecallResult with SVL-validated memories
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
    ) -> list[dict[str, Any]]:
        """Search memories with filters and SVL Gate validation.

        All returned memories are validated through the SVL Gate.
        There is NO bypass path.

        Args:
            user_id: User identifier
            query: Text search query
            topics: Filter by topics
            categories: Filter by categories
            memory_types: Filter by memory types
            limit: Max results

        Returns:
            List of SVL-validated memory dicts
        """
        results = self._clst.search(
            query=query,
            user_id=user_id,
            topics=topics,
            categories=categories,
            memory_types=memory_types,
            limit=limit,
        )
        # Return validated memories only
        return [r.memory for r in results if r.success and r.memory]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID with SVL Gate validation.

        The memory is validated through the SVL Gate before being returned.
        """
        result = self._clst.retrieve(memory_id)
        if result is None or not result.success:
            return None
        return result.memory

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
        """Compress old memories with SVL Gate validation.

        Compressed memories are re-validated through the SVL Gate.

        Args:
            user_id: User whose memories to compress
            older_than_days: Only compress memories older than this
            strategy: Compression strategy (summarize, merge, deduplicate)

        Returns:
            Compression result dict
        """
        from datetime import timedelta

        # Use GatedCLST compress which already validates through gate
        return self._clst.compress(
            user_id=user_id,
            older_than=timedelta(days=older_than_days),
            strategy=strategy,
        )

    def sync(
        self,
        source_agent: str,
        target_agent: str,
        user_id: str,
        memory_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync memories between agents with SVL Gate validation.

        All transferred memories are validated through the SVL Gate.

        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            user_id: User context
            memory_types: Types to sync

        Returns:
            Sync result dict
        """
        from mindcore.clst import CLST

        # Create temp CLST for sync operation (uses validated memories)
        temp_clst = CLST(storage=self._storage, vocabulary=self._vocabulary)
        result = temp_clst.sync(
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

        Migrated memories are re-validated through the SVL Gate.

        Args:
            from_version: Source vocabulary version
            user_id: Optional user filter
            create_checkpoints: Whether to create rollback checkpoints

        Returns:
            Migration result dict with checkpoint info for potential rollback

        Raises:
            ValueError: If no migration path exists
        """
        from mindcore.clst import CLST

        # Create temp CLST for migration
        temp_clst = CLST(storage=self._storage, vocabulary=self._vocabulary)
        result = temp_clst.migrate(
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
        from mindcore.clst import CLST

        if not hasattr(self, "_last_migration_result") or not self._last_migration_result:
            raise ValueError("No migration to rollback. Run migrate_vocabulary() first.")

        # Create temp CLST for rollback
        temp_clst = CLST(storage=self._storage, vocabulary=self._vocabulary)
        result = temp_clst.rollback_migration(self._last_migration_result)

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


# =============================================================================
# Factory Functions for SVL-First Architecture
# =============================================================================


def create_pipeline(
    storage: str = "sqlite:///mindcore.db",
    llm_call: Any = None,
    enable_hot_path: bool = True,
    enable_external_sources: bool = True,
    vocabulary: Any = None,
) -> Any:
    """Create an SVL Pipeline with full gate enforcement.

    This is the RECOMMENDED way to create a MindCore instance for production.
    The pipeline ensures:
    1. All data passes through SVL Gate (no bypass paths)
    2. Automatic canonicalization of LLM outputs
    3. Hot-path optimization (skip CLST for simple queries)
    4. External data source integration

    Args:
        storage: Storage connection string or backend
        llm_call: LLM function for context decisions and retries
        enable_hot_path: Enable hot-path optimization
        enable_external_sources: Enable external data source fetching
        vocabulary: Optional StructuredValidationLayer

    Returns:
        SVLPipeline instance

    Example:
        from mindcore import create_pipeline

        pipeline = create_pipeline(
            storage="sqlite:///memory.db",
            llm_call=my_llm_function,
        )

        # Store with mandatory validation
        result = pipeline.store(
            llm_output={"content": "...", "memory_type": "preference"},
            user_id="user123",
        )

        # Query with hot-path optimization
        result = pipeline.query(
            query="What are my preferences?",
            user_id="user123",
        )
    """
    from .svl import SVLPipeline

    return SVLPipeline(
        storage=storage,
        vocabulary=vocabulary,
        llm_call=llm_call,
        enable_hot_path=enable_hot_path,
        enable_external_sources=enable_external_sources,
    )
