"""CLST - Cognitive Long-term Storage Transfer.

A protocol for moving, syncing, and compressing long-term memory between
AI agents or between an agent and its external memory vault.

CLST handles:
- Long-term memory storage
- Memory compression and consolidation
- Cross-agent memory sync
- Memory transfer between instances
- Vocabulary version migrations
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Generator
from enum import Enum

if TYPE_CHECKING:
    from ..storage.base import BaseStorage
    from ..vocabulary import VocabularySchema

from ..flr import Memory


class CompressionStrategy(str, Enum):
    """Memory compression strategies."""

    SUMMARIZE = "summarize"    # LLM-based summarization
    MERGE = "merge"            # Merge similar memories
    DEDUPLICATE = "deduplicate"  # Remove duplicates
    EXTRACT = "extract"        # Extract key facts


class SyncDirection(str, Enum):
    """Direction for memory sync."""

    PUSH = "push"      # Push to target
    PULL = "pull"      # Pull from source
    BIDIRECTIONAL = "bidirectional"  # Sync both ways


@dataclass
class CompressionResult:
    """Result of memory compression."""

    original_count: int
    compressed_count: int
    strategy: CompressionStrategy
    compressed_memories: list[Memory]
    removed_memory_ids: list[str]
    compression_ratio: float
    latency_ms: float


@dataclass
class SyncResult:
    """Result of memory sync operation."""

    source_agent: str
    target_agent: str
    direction: SyncDirection
    memories_transferred: int
    conflicts_resolved: int
    errors: list[str]
    latency_ms: float


@dataclass
class TransferManifest:
    """Manifest for memory transfer between instances."""

    transfer_id: str
    source_instance: str
    target_instance: str
    memory_count: int
    total_size_bytes: int
    vocabulary_version: str
    created_at: datetime
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "source_instance": self.source_instance,
            "target_instance": self.target_instance,
            "memory_count": self.memory_count,
            "total_size_bytes": self.total_size_bytes,
            "vocabulary_version": self.vocabulary_version,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }


@dataclass
class MigrationResult:
    """Result of vocabulary migration."""

    from_version: str
    to_version: str
    memories_migrated: int
    memories_failed: int
    errors: list[str]
    latency_ms: float


class CLST:
    """Cognitive Long-term Storage Transfer - Cold path memory storage.

    Handles durable storage, compression, cross-agent sync, and transfers.

    Example:
        clst = CLST(storage=storage, vocabulary=vocab)

        # Store memory
        memory_id = clst.store(memory)

        # Compress old memories
        result = clst.compress(
            user_id="user123",
            older_than=days(30),
            strategy=CompressionStrategy.SUMMARIZE,
        )

        # Sync between agents
        result = clst.sync(
            source_agent="support_bot",
            target_agent="sales_bot",
            memory_types=["semantic"],
        )

        # Transfer to another instance
        manifest = clst.transfer(
            memories=memories,
            destination="backup.mindcore.io",
        )
    """

    def __init__(
        self,
        storage: BaseStorage,
        vocabulary: VocabularySchema | None = None,
        compression_llm: callable | None = None,
    ):
        """Initialize CLST.

        Args:
            storage: Storage backend
            vocabulary: Vocabulary schema for validation
            compression_llm: Optional LLM function for summarization
        """
        self.storage = storage
        self.vocabulary = vocabulary
        self.compression_llm = compression_llm

    def store(
        self,
        memory: Memory,
        validate: bool = True,
    ) -> str:
        """Store a memory in long-term storage.

        Args:
            memory: Memory to store
            validate: Validate against vocabulary

        Returns:
            Memory ID
        """
        # Validate against vocabulary if provided
        if validate and self.vocabulary:
            is_valid, errors = self.vocabulary.validate(memory.to_dict())
            if not is_valid:
                raise ValueError(f"Memory validation failed: {errors}")

        # Tag with vocabulary version
        if self.vocabulary:
            memory.vocabulary_version = self.vocabulary.version

        # Store
        return self.storage.store(memory)

    def store_batch(
        self,
        memories: list[Memory],
        validate: bool = True,
    ) -> list[str]:
        """Store multiple memories.

        Args:
            memories: Memories to store
            validate: Validate against vocabulary

        Returns:
            List of memory IDs
        """
        memory_ids = []
        for memory in memories:
            try:
                memory_id = self.store(memory, validate=validate)
                memory_ids.append(memory_id)
            except ValueError:
                # Skip invalid memories, log in production
                pass
        return memory_ids

    def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        return self.storage.get(memory_id)

    def search(
        self,
        query: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        memory_types: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Search memories with filters.

        Args:
            query: Text search query
            user_id: Filter by user
            agent_id: Filter by agent
            topics: Filter by topics
            categories: Filter by categories
            memory_types: Filter by memory types
            start_date: Filter by creation date (start)
            end_date: Filter by creation date (end)
            limit: Max results

        Returns:
            List of matching memories
        """
        return self.storage.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            topics=topics,
            categories=categories,
            memory_types=memory_types,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        return self.storage.delete(memory_id)

    def compress(
        self,
        user_id: str,
        older_than: timedelta | None = None,
        memory_types: list[str] | None = None,
        strategy: CompressionStrategy = CompressionStrategy.SUMMARIZE,
        min_memories: int = 10,
    ) -> CompressionResult:
        """Compress old memories.

        Reduces storage by merging, summarizing, or deduplicating memories.

        Args:
            user_id: User whose memories to compress
            older_than: Only compress memories older than this
            memory_types: Only compress these memory types
            strategy: Compression strategy
            min_memories: Minimum memories required to trigger compression

        Returns:
            CompressionResult with details
        """
        start_time = time.time()

        # Get candidate memories
        end_date = None
        if older_than:
            end_date = datetime.now(timezone.utc) - older_than

        candidates = self.search(
            user_id=user_id,
            memory_types=memory_types or ["episodic"],
            end_date=end_date,
            limit=1000,
        )

        if len(candidates) < min_memories:
            return CompressionResult(
                original_count=len(candidates),
                compressed_count=len(candidates),
                strategy=strategy,
                compressed_memories=[],
                removed_memory_ids=[],
                compression_ratio=1.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Apply compression strategy
        if strategy == CompressionStrategy.DEDUPLICATE:
            compressed, removed = self._deduplicate(candidates)
        elif strategy == CompressionStrategy.MERGE:
            compressed, removed = self._merge_similar(candidates)
        elif strategy == CompressionStrategy.SUMMARIZE:
            compressed, removed = self._summarize(candidates, user_id)
        elif strategy == CompressionStrategy.EXTRACT:
            compressed, removed = self._extract_facts(candidates, user_id)
        else:
            compressed, removed = candidates, []

        # Delete removed memories
        for memory_id in removed:
            self.storage.delete(memory_id)

        # Store compressed memories
        for memory in compressed:
            if memory.memory_id not in [c.memory_id for c in candidates]:
                self.store(memory, validate=False)

        latency = (time.time() - start_time) * 1000

        return CompressionResult(
            original_count=len(candidates),
            compressed_count=len(compressed),
            strategy=strategy,
            compressed_memories=compressed,
            removed_memory_ids=removed,
            compression_ratio=len(compressed) / len(candidates) if candidates else 1.0,
            latency_ms=latency,
        )

    def sync(
        self,
        source_agent: str,
        target_agent: str,
        user_id: str,
        memory_types: list[str] | None = None,
        direction: SyncDirection = SyncDirection.PUSH,
        conflict_resolution: str = "source_wins",
    ) -> SyncResult:
        """Sync memories between agents.

        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            user_id: User context
            memory_types: Types to sync
            direction: Sync direction
            conflict_resolution: How to handle conflicts

        Returns:
            SyncResult with details
        """
        start_time = time.time()
        errors = []
        transferred = 0
        conflicts = 0

        # Get source memories that are shareable
        source_memories = self.search(
            user_id=user_id,
            agent_id=source_agent,
            memory_types=memory_types,
            limit=1000,
        )

        # Filter to shared/team access level
        shareable = [
            m for m in source_memories
            if m.access_level in ["shared", "team", "global"]
        ]

        # Get existing target memories for conflict detection
        target_memories = self.search(
            user_id=user_id,
            agent_id=target_agent,
            memory_types=memory_types,
            limit=1000,
        )
        target_contents = {m.content: m for m in target_memories}

        # Transfer memories
        for memory in shareable:
            # Check for conflict (same content)
            if memory.content in target_contents:
                conflicts += 1
                if conflict_resolution == "skip":
                    continue
                elif conflict_resolution == "target_wins":
                    continue
                # source_wins: overwrite

            # Create copy for target agent
            synced = Memory(
                memory_id=self._generate_id(f"sync_{target_agent}_{memory.memory_id}"),
                content=memory.content,
                memory_type=memory.memory_type,
                user_id=user_id,
                agent_id=target_agent,
                topics=memory.topics.copy(),
                categories=memory.categories.copy(),
                sentiment=memory.sentiment,
                importance=memory.importance,
                entities=memory.entities.copy(),
                access_level="private",  # Synced copies are private to target
                vocabulary_version=memory.vocabulary_version,
            )

            try:
                self.store(synced, validate=False)
                transferred += 1
            except Exception as e:
                errors.append(f"Failed to sync memory {memory.memory_id}: {e}")

        latency = (time.time() - start_time) * 1000

        return SyncResult(
            source_agent=source_agent,
            target_agent=target_agent,
            direction=direction,
            memories_transferred=transferred,
            conflicts_resolved=conflicts,
            errors=errors,
            latency_ms=latency,
        )

    def transfer(
        self,
        memories: list[Memory],
        destination: str,
        include_embeddings: bool = False,
    ) -> TransferManifest:
        """Create a transfer package for memories.

        Args:
            memories: Memories to transfer
            destination: Target instance identifier
            include_embeddings: Include embedding vectors

        Returns:
            TransferManifest with transfer details
        """
        transfer_id = self._generate_id(f"transfer_{destination}_{time.time()}")

        # Serialize memories
        serialized = []
        for memory in memories:
            data = memory.to_dict()
            if not include_embeddings:
                data.pop("embedding", None)
            serialized.append(data)

        # Calculate size and checksum
        json_data = json.dumps(serialized)
        size_bytes = len(json_data.encode("utf-8"))
        checksum = hashlib.sha256(json_data.encode()).hexdigest()

        # Store transfer data (in real implementation, send to destination)
        self.storage.store_transfer(transfer_id, serialized)

        return TransferManifest(
            transfer_id=transfer_id,
            source_instance="local",
            target_instance=destination,
            memory_count=len(memories),
            total_size_bytes=size_bytes,
            vocabulary_version=self.vocabulary.version if self.vocabulary else "unknown",
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

    def receive_transfer(
        self,
        transfer_id: str,
        validate: bool = True,
    ) -> list[Memory]:
        """Receive a transfer from another instance.

        Args:
            transfer_id: Transfer identifier
            validate: Validate memories against vocabulary

        Returns:
            List of received memories
        """
        # Retrieve transfer data
        serialized = self.storage.get_transfer(transfer_id)
        if not serialized:
            raise ValueError(f"Transfer {transfer_id} not found")

        memories = []
        for data in serialized:
            memory = Memory.from_dict(data)

            # Migrate vocabulary if needed
            if self.vocabulary and memory.vocabulary_version != self.vocabulary.version:
                data = self.vocabulary.migrate_memory(data, memory.vocabulary_version)
                memory = Memory.from_dict(data)
                memory.vocabulary_version = self.vocabulary.version

            if validate and self.vocabulary:
                is_valid, errors = self.vocabulary.validate(memory.to_dict())
                if not is_valid:
                    continue  # Skip invalid memories

            memories.append(memory)

        return memories

    def migrate(
        self,
        from_version: str,
        user_id: str | None = None,
        batch_size: int = 100,
    ) -> MigrationResult:
        """Migrate memories to current vocabulary version.

        Args:
            from_version: Source vocabulary version
            user_id: Optional user filter
            batch_size: Batch size for processing

        Returns:
            MigrationResult with details
        """
        start_time = time.time()

        if not self.vocabulary:
            raise ValueError("Vocabulary required for migration")

        if from_version not in self.vocabulary.migrations:
            raise ValueError(f"No migration path from {from_version}")

        migrated = 0
        failed = 0
        errors = []

        # Stream memories with old version
        for batch in self._stream_memories_by_version(from_version, user_id, batch_size):
            for memory in batch:
                try:
                    # Migrate
                    data = memory.to_dict()
                    migrated_data = self.vocabulary.migrate_memory(data, from_version)

                    # Update in storage
                    updated_memory = Memory.from_dict(migrated_data)
                    updated_memory.vocabulary_version = self.vocabulary.version
                    self.storage.update(updated_memory)

                    migrated += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"Failed to migrate {memory.memory_id}: {e}")

        latency = (time.time() - start_time) * 1000

        return MigrationResult(
            from_version=from_version,
            to_version=self.vocabulary.version,
            memories_migrated=migrated,
            memories_failed=failed,
            errors=errors[:100],  # Limit error list
            latency_ms=latency,
        )

    def _deduplicate(
        self,
        memories: list[Memory],
    ) -> tuple[list[Memory], list[str]]:
        """Remove duplicate memories."""
        seen_content = {}
        unique = []
        removed = []

        for memory in memories:
            # Simple content hash
            content_hash = hashlib.md5(memory.content.encode()).hexdigest()

            if content_hash in seen_content:
                # Keep the one with higher importance/reinforcement
                existing = seen_content[content_hash]
                if memory.importance > existing.importance:
                    removed.append(existing.memory_id)
                    seen_content[content_hash] = memory
                    unique = [m for m in unique if m.memory_id != existing.memory_id]
                    unique.append(memory)
                else:
                    removed.append(memory.memory_id)
            else:
                seen_content[content_hash] = memory
                unique.append(memory)

        return unique, removed

    def _merge_similar(
        self,
        memories: list[Memory],
    ) -> tuple[list[Memory], list[str]]:
        """Merge memories with similar topics."""
        # Group by topic set
        groups: dict[str, list[Memory]] = {}
        for memory in memories:
            key = ",".join(sorted(memory.topics)) or "none"
            if key not in groups:
                groups[key] = []
            groups[key].append(memory)

        merged = []
        removed = []

        for topic_key, group in groups.items():
            if len(group) < 3:
                # Keep as-is
                merged.extend(group)
            else:
                # Merge into single memory
                contents = [m.content for m in group]
                combined_content = " | ".join(contents[:10])  # Limit length

                merged_memory = Memory(
                    memory_id=self._generate_id(f"merged_{topic_key}"),
                    content=f"[Merged from {len(group)} memories] {combined_content}",
                    memory_type="semantic",  # Merged becomes semantic
                    user_id=group[0].user_id,
                    agent_id=group[0].agent_id,
                    topics=group[0].topics,
                    categories=list(set(c for m in group for c in m.categories)),
                    importance=max(m.importance for m in group),
                    entities=list(set(e for m in group for e in m.entities)),
                    access_level=group[0].access_level,
                )
                merged.append(merged_memory)
                removed.extend([m.memory_id for m in group])

        return merged, removed

    def _summarize(
        self,
        memories: list[Memory],
        user_id: str,
    ) -> tuple[list[Memory], list[str]]:
        """Summarize memories using LLM."""
        if not self.compression_llm:
            # Fallback to merge
            return self._merge_similar(memories)

        # Group by time period (weekly)
        weekly_groups: dict[str, list[Memory]] = {}
        for memory in memories:
            if memory.created_at:
                week_key = memory.created_at.strftime("%Y-W%W")
            else:
                week_key = "unknown"
            if week_key not in weekly_groups:
                weekly_groups[week_key] = []
            weekly_groups[week_key].append(memory)

        summarized = []
        removed = []

        for week_key, group in weekly_groups.items():
            if len(group) < 5:
                summarized.extend(group)
                continue

            # Create summary prompt
            contents = "\n".join([f"- {m.content}" for m in group[:20]])
            prompt = f"Summarize these memories into 2-3 key points:\n{contents}"

            try:
                summary = self.compression_llm(prompt)

                summary_memory = Memory(
                    memory_id=self._generate_id(f"summary_{week_key}"),
                    content=f"[Summary for {week_key}] {summary}",
                    memory_type="semantic",
                    user_id=user_id,
                    agent_id=group[0].agent_id,
                    topics=list(set(t for m in group for t in m.topics))[:5],
                    categories=list(set(c for m in group for c in m.categories)),
                    importance=0.7,  # Summaries are important
                    access_level=group[0].access_level,
                )
                summarized.append(summary_memory)
                removed.extend([m.memory_id for m in group])
            except Exception:
                # Keep original on failure
                summarized.extend(group)

        return summarized, removed

    def _extract_facts(
        self,
        memories: list[Memory],
        user_id: str,
    ) -> tuple[list[Memory], list[str]]:
        """Extract key facts from memories."""
        if not self.compression_llm:
            return self._deduplicate(memories)

        # Batch memories
        contents = "\n".join([f"- {m.content}" for m in memories[:50]])
        prompt = f"Extract key facts from these memories as a bullet list:\n{contents}"

        try:
            facts = self.compression_llm(prompt)

            fact_memory = Memory(
                memory_id=self._generate_id("facts"),
                content=f"[Extracted facts] {facts}",
                memory_type="semantic",
                user_id=user_id,
                agent_id=memories[0].agent_id if memories else None,
                topics=list(set(t for m in memories for t in m.topics))[:10],
                importance=0.8,
                access_level="shared",
            )

            return [fact_memory], [m.memory_id for m in memories]
        except Exception:
            return self._deduplicate(memories)

    def _stream_memories_by_version(
        self,
        version: str,
        user_id: str | None,
        batch_size: int,
    ) -> Generator[list[Memory], None, None]:
        """Stream memories by vocabulary version."""
        offset = 0
        while True:
            batch = self.storage.search_by_version(
                version=version,
                user_id=user_id,
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            yield batch
            offset += batch_size

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def get_stats(self) -> dict[str, Any]:
        """Get CLST statistics."""
        return {
            "vocabulary_version": self.vocabulary.version if self.vocabulary else None,
            "storage_stats": self.storage.get_stats() if hasattr(self.storage, "get_stats") else {},
        }
