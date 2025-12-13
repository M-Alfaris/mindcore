"""Memory Extraction - Structured output parsing.

Parses memories from LLM structured output responses.
The main agent is responsible for extracting memories as part of its response.

No fallbacks - fails hard on errors.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..flr import Memory
from ..vocabulary import VocabularySchema


@dataclass
class ExtractionResult:
    """Result of memory extraction."""

    memories: list[Memory]
    extraction_latency_ms: float


class MemoryExtractor:
    """Extract memories from LLM structured output.

    The main agent outputs structured JSON with memories_to_store.
    This extractor validates and converts them to Memory objects.

    Fails hard on validation errors - no fallbacks, no silent failures.

    Example:
        extractor = MemoryExtractor(vocabulary=vocab)

        # Parse from LLM structured output
        result = extractor.extract(
            output=llm_response,
            user_id="user123",
        )

        # If validation fails, raises ValueError
    """

    def __init__(self, vocabulary: VocabularySchema):
        """Initialize extractor.

        Args:
            vocabulary: Vocabulary schema for validation (required)

        Raises:
            ValueError: If vocabulary is None
        """
        if vocabulary is None:
            raise ValueError("Vocabulary is required for extraction")
        self.vocabulary = vocabulary

    def extract(
        self,
        output: dict[str, Any],
        user_id: str,
        agent_id: str | None = None,
    ) -> ExtractionResult:
        """Extract memories from LLM structured output.

        Expects output in format:
        {
            "response": "...",
            "memories_to_store": [
                {"content": "...", "memory_type": "...", ...}
            ]
        }

        Args:
            output: Structured output from LLM
            user_id: User identifier
            agent_id: Agent identifier

        Returns:
            ExtractionResult with parsed memories

        Raises:
            TypeError: If output is not a dict
            ValueError: If memories fail validation
            KeyError: If required fields are missing
        """
        start_time = time.time()

        if not isinstance(output, dict):
            raise TypeError(f"Expected dict output, got {type(output).__name__}")

        memories_data = output.get("memories_to_store", [])

        if not isinstance(memories_data, list):
            raise TypeError(
                f"memories_to_store must be a list, got {type(memories_data).__name__}"
            )

        memories = []

        for i, mem_data in enumerate(memories_data):
            # Validate type
            if not isinstance(mem_data, dict):
                raise TypeError(
                    f"Memory {i}: expected dict, got {type(mem_data).__name__}"
                )

            # Validate required fields
            if "content" not in mem_data:
                raise KeyError(f"Memory {i}: missing required field 'content'")
            if "memory_type" not in mem_data:
                raise KeyError(f"Memory {i}: missing required field 'memory_type'")

            # Validate against vocabulary
            is_valid, errors = self.vocabulary.validate(mem_data)
            if not is_valid:
                raise ValueError(f"Memory {i} validation failed: {errors}")

            # Create Memory object
            memory = Memory(
                memory_id=mem_data.get("id", f"mem_{uuid.uuid4().hex[:12]}"),
                content=mem_data["content"],
                memory_type=mem_data["memory_type"],
                user_id=user_id,
                agent_id=agent_id,
                topics=mem_data.get("topics", []),
                categories=mem_data.get("categories", []),
                sentiment=mem_data.get("sentiment", "neutral"),
                importance=mem_data.get("importance", 0.5),
                entities=mem_data.get("entities", []),
                access_level=mem_data.get("access_level", "private"),
                vocabulary_version=self.vocabulary.version,
            )
            memories.append(memory)

        latency = (time.time() - start_time) * 1000

        return ExtractionResult(
            memories=memories,
            extraction_latency_ms=latency,
        )

    def validate_output(self, output: dict[str, Any]) -> None:
        """Validate that output matches expected schema.

        Call this before extract() to validate the full output structure.

        Args:
            output: LLM output to validate

        Raises:
            TypeError: If types are wrong
            KeyError: If required fields missing
            ValueError: If values are invalid
        """
        if not isinstance(output, dict):
            raise TypeError(f"Output must be dict, got {type(output).__name__}")

        if "response" not in output:
            raise KeyError("Output missing required field 'response'")

        if "memories_to_store" in output:
            memories = output["memories_to_store"]

            if not isinstance(memories, list):
                raise TypeError(
                    f"memories_to_store must be list, got {type(memories).__name__}"
                )

            for i, mem in enumerate(memories):
                if not isinstance(mem, dict):
                    raise TypeError(f"Memory {i}: must be dict")
                if "content" not in mem:
                    raise KeyError(f"Memory {i}: missing 'content'")
                if "memory_type" not in mem:
                    raise KeyError(f"Memory {i}: missing 'memory_type'")

                # Validate against vocabulary
                is_valid, errors = self.vocabulary.validate(mem)
                if not is_valid:
                    raise ValueError(f"Memory {i}: {errors}")
