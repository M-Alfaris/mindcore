"""Memory Extraction - Structured output and auto-extraction.

Handles two modes of memory extraction:
1. Structured output: Parse memories from LLM JSON responses
2. Auto-extraction: Mem0-style implicit memory extraction from conversations
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..flr import Memory
from ..vocabulary import VocabularySchema, MemoryType


@dataclass
class ExtractionResult:
    """Result of memory extraction."""

    memories: list[Memory]
    source: str  # "structured" or "auto"
    extraction_latency_ms: float
    errors: list[str] = field(default_factory=list)


class MemoryExtractor:
    """Extract memories from LLM outputs and conversations.

    Supports two extraction modes:
    1. Structured: Parse memories from JSON structured output
    2. Auto: Extract implicit memories from conversation text

    Example:
        extractor = MemoryExtractor(vocabulary=vocab)

        # From structured LLM output
        memories = extractor.from_structured_output(llm_response)

        # Auto-extract from conversation
        memories = extractor.auto_extract(messages, user_id="user123")
    """

    # Patterns for auto-extraction
    PREFERENCE_PATTERNS = [
        r"(?:i |my )(?:prefer|like|want|love|hate|dislike)s?\s+(.+)",
        r"(?:i'd rather|i would rather)\s+(.+)",
        r"(?:please |always |never )(.+)",
        r"(?:don't|do not)\s+(.+)",
    ]

    ENTITY_PATTERNS = [
        r"(?:my |the )(?:order|order number|tracking|reference)(?:\s*(?:#|number|:))?\s*([A-Z0-9-]+)",
        r"(?:my |the )(?:account|user|customer)(?:\s*(?:#|id|:))?\s*([A-Z0-9-]+)",
        r"(?:email|e-mail)(?:\s*(?:is|:))?\s*([\w\.-]+@[\w\.-]+)",
        r"(?:phone|tel|mobile)(?:\s*(?:is|:))?\s*([\d\s\-\+\(\)]+)",
    ]

    FACT_PATTERNS = [
        r"(?:i am|i'm|i work|i live|i have)\s+(.+)",
        r"(?:my name is|call me)\s+(.+)",
        r"(?:i've been|i was|i used to)\s+(.+)",
    ]

    def __init__(
        self,
        vocabulary: VocabularySchema | None = None,
        auto_extract_llm: Callable[[str], str] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize extractor.

        Args:
            vocabulary: Vocabulary schema for validation
            auto_extract_llm: Optional LLM function for enhanced extraction
            min_confidence: Minimum confidence for auto-extracted memories
        """
        self.vocabulary = vocabulary
        self.auto_extract_llm = auto_extract_llm
        self.min_confidence = min_confidence

    def from_structured_output(
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
        """
        import time
        start_time = time.time()

        memories = []
        errors = []

        memories_data = output.get("memories_to_store", [])

        for i, mem_data in enumerate(memories_data):
            try:
                # Validate against vocabulary
                if self.vocabulary:
                    is_valid, validation_errors = self.vocabulary.validate(mem_data)
                    if not is_valid:
                        errors.append(f"Memory {i}: {validation_errors}")
                        continue

                # Create Memory object
                memory = Memory(
                    memory_id=mem_data.get("id", f"mem_{uuid.uuid4().hex[:12]}"),
                    content=mem_data["content"],
                    memory_type=mem_data.get("memory_type", "episodic"),
                    user_id=user_id,
                    agent_id=agent_id,
                    topics=mem_data.get("topics", []),
                    categories=mem_data.get("categories", []),
                    sentiment=mem_data.get("sentiment", "neutral"),
                    importance=mem_data.get("importance", 0.5),
                    entities=mem_data.get("entities", []),
                    access_level=mem_data.get("access_level", "private"),
                    vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
                )
                memories.append(memory)

            except Exception as e:
                errors.append(f"Memory {i}: {str(e)}")

        latency = (time.time() - start_time) * 1000

        return ExtractionResult(
            memories=memories,
            source="structured",
            extraction_latency_ms=latency,
            errors=errors,
        )

    def auto_extract(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
        use_llm: bool = True,
    ) -> ExtractionResult:
        """Auto-extract implicit memories from conversation.

        Mem0-style extraction that identifies:
        - User preferences
        - Entities (order IDs, emails, etc.)
        - Facts about the user
        - Procedural information

        Args:
            messages: Conversation messages
            user_id: User identifier
            agent_id: Agent identifier
            use_llm: Use LLM for enhanced extraction

        Returns:
            ExtractionResult with extracted memories
        """
        import time
        start_time = time.time()

        memories = []
        errors = []

        # Extract from user messages only
        user_messages = [
            m for m in messages
            if m.get("role") == "user"
        ]

        for msg in user_messages:
            content = msg.get("content", "")

            # Rule-based extraction
            rule_memories = self._extract_with_rules(content, user_id, agent_id)
            memories.extend(rule_memories)

        # LLM-enhanced extraction
        if use_llm and self.auto_extract_llm and user_messages:
            try:
                llm_memories = self._extract_with_llm(user_messages, user_id, agent_id)
                memories.extend(llm_memories)
            except Exception as e:
                errors.append(f"LLM extraction failed: {e}")

        # Deduplicate
        memories = self._deduplicate_memories(memories)

        latency = (time.time() - start_time) * 1000

        return ExtractionResult(
            memories=memories,
            source="auto",
            extraction_latency_ms=latency,
            errors=errors,
        )

    def _extract_with_rules(
        self,
        content: str,
        user_id: str,
        agent_id: str | None,
    ) -> list[Memory]:
        """Extract memories using rule-based patterns."""
        memories = []
        content_lower = content.lower()

        # Extract preferences
        for pattern in self.PREFERENCE_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                memory = Memory(
                    memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                    content=f"User preference: {match.strip()}",
                    memory_type="preference",
                    user_id=user_id,
                    agent_id=agent_id,
                    topics=self._infer_topics(match),
                    importance=0.7,
                    access_level="shared",
                    vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
                )
                memories.append(memory)

        # Extract entities
        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                entity_type = self._classify_entity(pattern)
                memory = Memory(
                    memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                    content=f"User {entity_type}: {match.strip()}",
                    memory_type="entity",
                    user_id=user_id,
                    agent_id=agent_id,
                    topics=self._infer_topics(match),
                    entities=[match.strip()],
                    importance=0.8,
                    access_level="private",
                    vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
                )
                memories.append(memory)

        # Extract facts
        for pattern in self.FACT_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out too short matches
                    memory = Memory(
                        memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                        content=f"User fact: {match.strip()}",
                        memory_type="semantic",
                        user_id=user_id,
                        agent_id=agent_id,
                        topics=self._infer_topics(match),
                        importance=0.6,
                        access_level="shared",
                        vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
                    )
                    memories.append(memory)

        return memories

    def _extract_with_llm(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None,
    ) -> list[Memory]:
        """Extract memories using LLM."""
        if not self.auto_extract_llm:
            return []

        # Build prompt
        conversation = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages[-10:]  # Last 10 messages
        ])

        prompt = f"""Analyze this conversation and extract key memories to store.
Focus on:
1. User preferences and settings
2. Important facts about the user
3. Entities mentioned (order IDs, emails, etc.)
4. Procedural knowledge shared

Conversation:
{conversation}

Return a JSON array of memories, each with:
- content: The memory content
- memory_type: One of [episodic, semantic, procedural, preference, entity]
- topics: List of relevant topics
- importance: 0-1 score

Only extract genuinely important information worth remembering."""

        try:
            response = self.auto_extract_llm(prompt)

            # Parse JSON from response
            import json
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []

            extracted = json.loads(json_match.group())

            memories = []
            for item in extracted:
                if item.get("importance", 0) < self.min_confidence:
                    continue

                memory = Memory(
                    memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                    content=item.get("content", ""),
                    memory_type=item.get("memory_type", "semantic"),
                    user_id=user_id,
                    agent_id=agent_id,
                    topics=item.get("topics", []),
                    importance=item.get("importance", 0.5),
                    access_level="shared",
                    vocabulary_version=self.vocabulary.version if self.vocabulary else "1.0.0",
                )
                memories.append(memory)

            return memories

        except Exception:
            return []

    def _infer_topics(self, text: str) -> list[str]:
        """Infer topics from text content."""
        if not self.vocabulary:
            return []

        text_lower = text.lower()
        matched_topics = []

        for topic in self.vocabulary.topics:
            if topic.lower() in text_lower:
                matched_topics.append(topic)

        # Keyword-based inference
        keyword_topics = {
            "order": ["order", "orders", "shipping", "delivery"],
            "billing": ["billing", "payment", "invoice", "charge", "refund"],
            "account": ["account", "login", "password", "profile"],
            "support": ["help", "issue", "problem", "support"],
            "product": ["product", "feature", "service"],
        }

        for topic, keywords in keyword_topics.items():
            if any(kw in text_lower for kw in keywords):
                if topic in self.vocabulary.topics and topic not in matched_topics:
                    matched_topics.append(topic)

        return matched_topics[:5]  # Limit to 5 topics

    def _classify_entity(self, pattern: str) -> str:
        """Classify entity type based on pattern."""
        if "order" in pattern:
            return "order_id"
        elif "account" in pattern or "user" in pattern or "customer" in pattern:
            return "account_id"
        elif "email" in pattern:
            return "email"
        elif "phone" in pattern or "tel" in pattern or "mobile" in pattern:
            return "phone"
        return "entity"

    def _deduplicate_memories(self, memories: list[Memory]) -> list[Memory]:
        """Remove duplicate memories."""
        seen_content = set()
        unique = []

        for memory in memories:
            # Normalize content for comparison
            normalized = memory.content.lower().strip()
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique.append(memory)

        return unique

    def extract_from_response(
        self,
        llm_response: Any,
        user_id: str,
        agent_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ExtractionResult:
        """Combined extraction from LLM response.

        Tries structured output first, then auto-extraction.

        Args:
            llm_response: Raw LLM response (dict or string)
            user_id: User identifier
            agent_id: Agent identifier
            messages: Optional conversation messages for auto-extraction

        Returns:
            Combined ExtractionResult
        """
        import time
        start_time = time.time()

        all_memories = []
        all_errors = []

        # Try structured extraction
        if isinstance(llm_response, dict):
            structured_result = self.from_structured_output(
                llm_response, user_id, agent_id
            )
            all_memories.extend(structured_result.memories)
            all_errors.extend(structured_result.errors)

        # Auto-extract from messages if provided
        if messages:
            auto_result = self.auto_extract(
                messages, user_id, agent_id, use_llm=True
            )
            all_memories.extend(auto_result.memories)
            all_errors.extend(auto_result.errors)

        # Deduplicate combined results
        all_memories = self._deduplicate_memories(all_memories)

        latency = (time.time() - start_time) * 1000

        return ExtractionResult(
            memories=all_memories,
            source="combined",
            extraction_latency_ms=latency,
            errors=all_errors,
        )
