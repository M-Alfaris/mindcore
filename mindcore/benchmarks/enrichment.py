"""Dataset Enrichment Pipeline - Full SVL Metadata Generation.

Generates COMPLETE SVL-compliant metadata for benchmark data using LLM.
This ensures benchmark data is identical to production data flow:

    Raw Content → LLM Extraction → EnforcedMetadata → SVL Gate → CLST/FLR

Required Metadata Fields (from EnforcedMetadata):
    - message_id: Unique message identifier
    - user_id: User identifier
    - session_id: Session identifier (for CLST clustering)
    - agent_id: Agent identifier (optional)
    - thread_id: Thread identifier for multi-thread conversations
    - topics: List of topics from SVL vocabulary
    - categories: List of categories from SVL vocabulary
    - entities: Named entities extracted from content
    - message_type: statement, question, command, etc.
    - message_intent: provide_info, ask_question, express_preference, etc.
    - importance: 0.0-1.0 score
    - confidence: 0.0-1.0 extraction confidence
    - urgency: low, medium, high
    - sentiment: positive, negative, neutral
    - emotional_classification: happy, sad, angry, neutral, etc.
    - temporal_qualifier: past, present, future, always
    - domain_label: Domain classification
    - memory_type: episodic, semantic, preference, procedural, working
    - access_level: private, shared, public
    - created_at: Timestamp

Usage:
    from mindcore.benchmarks.enrichment import DatasetEnrichmentPipeline

    # With OpenAI
    pipeline = DatasetEnrichmentPipeline(
        llm_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Enrich raw data with full metadata
    enriched = pipeline.enrich_dataset(raw_data)

    # Store through SVL pipeline
    for memory in enriched.memories:
        svl_pipeline.store(
            llm_output=memory.to_llm_output(),
            user_id=memory.user_id,
            session_id=memory.session_id,
        )
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from mindcore.benchmarks.datasets import (
    BenchmarkDataset,
    ConversationSession,
    ConversationTurn,
    DatasetType,
    MemoryTestCase,
)


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers for metadata extraction."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For testing without API calls


# Valid values from SVL vocabulary
VALID_MEMORY_TYPES = ["preference", "semantic", "episodic", "working", "procedural"]
VALID_MESSAGE_TYPES = ["statement", "question", "command", "response", "greeting", "farewell"]
VALID_MESSAGE_INTENTS = [
    "provide_info",
    "ask_question",
    "request_action",
    "express_preference",
    "share_experience",
    "give_feedback",
    "clarify",
    "acknowledge",
]
VALID_SENTIMENTS = ["positive", "negative", "neutral", "mixed"]
VALID_URGENCIES = ["low", "medium", "high", "critical"]
VALID_ACCESS_LEVELS = ["private", "shared", "public"]
VALID_TEMPORAL_QUALIFIERS = ["past", "present", "future", "always", "never", "sometimes"]
VALID_EMOTIONAL_CLASSIFICATIONS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "neutral",
    "anxious",
    "excited",
]


@dataclass
class FullMetadata:
    """Complete SVL-enforced metadata structure.

    This matches the EnforcedMetadata class in svl/enforced_metadata.py
    """

    # Required identifiers
    message_id: str
    user_id: str
    session_id: str
    agent_id: str | None = None
    thread_id: str | None = None

    # SVL-enforced classifications
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    message_type: str = "statement"
    message_intent: str = "provide_info"

    # Scores
    importance: float = 0.5
    confidence: float = 0.8
    urgency: str = "medium"

    # Additional SVL fields
    sentiment: str = "neutral"
    emotional_classification: str = "neutral"
    temporal_qualifier: str | None = None
    domain_label: str | None = None

    # Memory classification
    memory_type: str = "episodic"
    access_level: str = "private"

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for SVL Gate."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "topics": self.topics,
            "categories": self.categories,
            "entities": self.entities,
            "message_type": self.message_type,
            "message_intent": self.message_intent,
            "importance": self.importance,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "emotional_classification": self.emotional_classification,
            "temporal_qualifier": self.temporal_qualifier,
            "domain_label": self.domain_label,
            "memory_type": self.memory_type,
            "access_level": self.access_level,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EnrichedMemory:
    """A memory enriched with FULL SVL-compliant metadata."""

    # Content
    content: str

    # Full metadata
    metadata: FullMetadata

    # Original context for reference
    original_role: str = "user"
    original_turn_index: int = 0

    # Enrichment info
    enriched_by: str = ""
    extraction_confidence: float = 0.0

    @property
    def user_id(self) -> str:
        return self.metadata.user_id

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def message_id(self) -> str:
        return self.metadata.message_id

    @property
    def memory_type(self) -> str:
        return self.metadata.memory_type

    def to_llm_output(self) -> dict[str, Any]:
        """Convert to LLM output format for SVLPipeline.store()."""
        return {
            "content": self.content,
            **self.metadata.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "original_role": self.original_role,
            "original_turn_index": self.original_turn_index,
            "enriched_by": self.enriched_by,
            "extraction_confidence": self.extraction_confidence,
        }


@dataclass
class EnrichedDataset:
    """A dataset enriched with full SVL-compliant metadata."""

    name: str
    source: str
    memories: list[EnrichedMemory] = field(default_factory=list)
    test_cases: list[MemoryTestCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.memories)

    def to_benchmark_dataset(self) -> BenchmarkDataset:
        """Convert to BenchmarkDataset for benchmark runner."""
        sessions_by_id: dict[str, ConversationSession] = {}

        for mem in self.memories:
            sid = mem.session_id
            if sid not in sessions_by_id:
                sessions_by_id[sid] = ConversationSession(
                    session_id=sid,
                    user_id=mem.user_id,
                    started_at=mem.metadata.created_at,
                )

            session = sessions_by_id[sid]
            session.turns.append(
                ConversationTurn(
                    role=mem.original_role,
                    content=mem.content,
                    timestamp=mem.metadata.created_at,
                    metadata=mem.metadata.to_dict(),
                )
            )

        return BenchmarkDataset(
            name=self.name,
            dataset_type=DatasetType.MULTI_SESSION,
            description=f"Enriched dataset from {self.source} with full SVL metadata",
            sessions=list(sessions_by_id.values()),
            test_cases=self.test_cases,
            metadata={
                **self.metadata,
                "source": self.source,
                "enriched": True,
                "size": self.size,
                "has_full_metadata": True,
            },
        )

    def save(self, path: str) -> None:
        """Save enriched dataset to JSON file."""
        data = {
            "name": self.name,
            "source": self.source,
            "memories": [m.to_dict() for m in self.memories],
            "test_cases": [
                {
                    "test_id": tc.test_id,
                    "query": tc.query,
                    "expected_memories": tc.expected_memories,
                    "context": tc.context,
                    "ground_truth": tc.ground_truth,
                }
                for tc in self.test_cases
            ],
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> EnrichedDataset:
        """Load enriched dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        memories = []
        for m in data["memories"]:
            meta_data = m.get("metadata", {})
            created_at = meta_data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            else:
                created_at = datetime.now(timezone.utc)

            metadata = FullMetadata(
                message_id=meta_data.get("message_id", f"msg_{uuid.uuid4().hex[:12]}"),
                user_id=meta_data.get("user_id", ""),
                session_id=meta_data.get("session_id", ""),
                agent_id=meta_data.get("agent_id"),
                thread_id=meta_data.get("thread_id"),
                topics=meta_data.get("topics", []),
                categories=meta_data.get("categories", []),
                entities=meta_data.get("entities", []),
                message_type=meta_data.get("message_type", "statement"),
                message_intent=meta_data.get("message_intent", "provide_info"),
                importance=meta_data.get("importance", 0.5),
                confidence=meta_data.get("confidence", 0.8),
                urgency=meta_data.get("urgency", "medium"),
                sentiment=meta_data.get("sentiment", "neutral"),
                emotional_classification=meta_data.get("emotional_classification", "neutral"),
                temporal_qualifier=meta_data.get("temporal_qualifier"),
                domain_label=meta_data.get("domain_label"),
                memory_type=meta_data.get("memory_type", "episodic"),
                access_level=meta_data.get("access_level", "private"),
                created_at=created_at,
            )

            memories.append(
                EnrichedMemory(
                    content=m["content"],
                    metadata=metadata,
                    original_role=m.get("original_role", "user"),
                    original_turn_index=m.get("original_turn_index", 0),
                    enriched_by=m.get("enriched_by", ""),
                    extraction_confidence=m.get("extraction_confidence", 0.0),
                )
            )

        test_cases = [
            MemoryTestCase(
                test_id=tc["test_id"],
                query=tc["query"],
                expected_memories=tc.get("expected_memories", []),
                context=tc.get("context", {}),
                ground_truth=tc.get("ground_truth"),
            )
            for tc in data.get("test_cases", [])
        ]

        return cls(
            name=data["name"],
            source=data["source"],
            memories=memories,
            test_cases=test_cases,
            metadata=data.get("metadata", {}),
        )


# Full extraction prompt for LLM - generates all required metadata
FULL_EXTRACTION_PROMPT = """You are a metadata extraction assistant for an AI memory system.
Your task is to extract COMPLETE structured metadata from user messages.

USER MESSAGE:
{content}

CONTEXT:
- User ID: {user_id}
- Session ID: {session_id}
- Message Index: {turn_index}

Extract ALL of the following fields in JSON format:

{{
    "memory_type": "<REQUIRED: one of: preference, semantic, episodic, working, procedural>",
    "message_type": "<REQUIRED: one of: statement, question, command, response, greeting, farewell>",
    "message_intent": "<REQUIRED: one of: provide_info, ask_question, request_action, express_preference, share_experience, give_feedback, clarify, acknowledge>",
    "topics": ["<REQUIRED: 1-3 relevant topics>"],
    "categories": ["<REQUIRED: 1-2 categories like personal, work, technology, entertainment>"],
    "entities": ["<named entities: people, places, products, etc.>"],
    "importance": <REQUIRED: float 0.0-1.0>,
    "confidence": <REQUIRED: float 0.0-1.0, your confidence in this extraction>,
    "urgency": "<REQUIRED: one of: low, medium, high, critical>",
    "sentiment": "<REQUIRED: one of: positive, negative, neutral, mixed>",
    "emotional_classification": "<REQUIRED: one of: happy, sad, angry, fearful, surprised, disgusted, neutral, anxious, excited>",
    "temporal_qualifier": "<one of: past, present, future, always, never, sometimes, or null>",
    "domain_label": "<domain like: technology, health, finance, entertainment, or null>"
}}

Classification Rules:
- memory_type:
  - "preference" = user preferences, likes, dislikes, choices
  - "semantic" = facts, knowledge, definitions
  - "episodic" = events, experiences, stories
  - "working" = current context, temporary info
  - "procedural" = how-to, instructions, processes

- message_intent:
  - "express_preference" = stating likes/dislikes
  - "provide_info" = sharing information
  - "share_experience" = telling about events
  - "ask_question" = asking something
  - "request_action" = asking for something to be done

- importance: 0.8+ for preferences/key facts, 0.5 default, 0.3 for casual

Return ONLY valid JSON, no other text."""


class DatasetEnrichmentPipeline:
    """Pipeline for enriching raw data with FULL SVL-compliant metadata.

    Generates all required fields that SVL Gate expects, ensuring
    proper flow through SVL → CLST → FLR.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        vocabulary_topics: list[str] | None = None,
        vocabulary_categories: list[str] | None = None,
    ):
        """Initialize the enrichment pipeline.

        Args:
            llm_provider: Provider to use ("openai", "anthropic", "local")
            api_key: API key for the provider
            model: Model to use (defaults to gpt-4o-mini for openai)
            vocabulary_topics: Valid topics for SVL vocabulary
            vocabulary_categories: Valid categories for SVL vocabulary
        """
        self.provider = LLMProvider(llm_provider)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

        # Default vocabulary constraints
        self.valid_topics = vocabulary_topics or [
            "benchmark",
            "test",
            "preferences",
            "personal",
            "work",
            "entertainment",
            "health",
            "technology",
            "travel",
            "food",
            "shopping",
            "social",
            "programming",
            "communication",
            "settings",
        ]
        self.valid_categories = vocabulary_categories or [
            "user_preference",
            "general",
            "work",
            "personal",
            "system",
            "technology",
            "lifestyle",
            "communication",
        ]

        # Setup LLM call function
        self._llm_call = self._setup_llm()

    def _setup_llm(self) -> Callable[[str], str]:
        """Setup the LLM call function based on provider."""
        if self.provider == LLMProvider.LOCAL:
            return self._mock_llm_call

        if self.provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)
                model = self.model or "gpt-4o-mini"

                def call_openai(prompt: str) -> str:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content or ""

                return call_openai

            except ImportError:
                logger.warning("OpenAI not installed, using local mock")
                return self._mock_llm_call

        elif self.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=self.api_key)
                model = self.model or "claude-3-5-haiku-20241022"

                def call_anthropic(prompt: str) -> str:
                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text

                return call_anthropic

            except ImportError:
                logger.warning("Anthropic not installed, using local mock")
                return self._mock_llm_call

        return self._mock_llm_call

    def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM call for testing without API.

        Generates realistic metadata based on content analysis.
        """
        content_lower = prompt.lower()

        # Determine memory type and intent
        if any(
            word in content_lower for word in ["prefer", "like", "love", "hate", "favorite", "want"]
        ):
            memory_type = "preference"
            message_intent = "express_preference"
            importance = 0.8
        elif any(
            word in content_lower
            for word in ["remember", "yesterday", "last week", "when i", "once"]
        ):
            memory_type = "episodic"
            message_intent = "share_experience"
            importance = 0.6
        elif "?" in content_lower:
            memory_type = "working"
            message_intent = "ask_question"
            importance = 0.5
        else:
            memory_type = "semantic"
            message_intent = "provide_info"
            importance = 0.5

        # Determine message type
        if "?" in content_lower:
            message_type = "question"
        elif any(word in content_lower for word in ["please", "can you", "could you"]):
            message_type = "command"
        else:
            message_type = "statement"

        # Determine sentiment
        if any(word in content_lower for word in ["love", "great", "amazing", "happy", "like"]):
            sentiment = "positive"
            emotional = "happy"
        elif any(word in content_lower for word in ["hate", "bad", "terrible", "sad", "angry"]):
            sentiment = "negative"
            emotional = "sad"
        else:
            sentiment = "neutral"
            emotional = "neutral"

        # Determine topics
        topics = ["benchmark"]
        if (
            "dark mode" in content_lower
            or "light mode" in content_lower
            or "theme" in content_lower
        ):
            topics.append("preferences")
            topics.append("settings")
        if any(word in content_lower for word in ["python", "javascript", "programming", "code"]):
            topics.append("technology")
            topics.append("programming")
        if any(word in content_lower for word in ["work", "remote", "office", "meeting"]):
            topics.append("work")
        if any(word in content_lower for word in ["travel", "trip", "vacation"]):
            topics.append("travel")

        # Determine categories
        categories = ["general"]
        if memory_type == "preference":
            categories = ["user_preference"]
        elif "work" in topics:
            categories = ["work"]
        elif "technology" in topics:
            categories = ["technology"]

        # Extract entities (simple pattern matching for mock)
        entities = []
        if "python" in content_lower:
            entities.append("Python")
        if "javascript" in content_lower:
            entities.append("JavaScript")
        if "dark mode" in content_lower:
            entities.append("dark mode")
        if "light mode" in content_lower:
            entities.append("light mode")

        # Temporal qualifier
        temporal = None
        if any(word in content_lower for word in ["always", "usually", "normally"]):
            temporal = "always"
        elif any(word in content_lower for word in ["yesterday", "last", "ago"]):
            temporal = "past"
        elif any(word in content_lower for word in ["now", "currently", "today"]):
            temporal = "present"
        elif any(word in content_lower for word in ["will", "going to", "tomorrow"]):
            temporal = "future"

        # Domain
        domain = None
        if "technology" in topics or "programming" in topics:
            domain = "technology"
        elif "health" in content_lower:
            domain = "health"

        return json.dumps(
            {
                "memory_type": memory_type,
                "message_type": message_type,
                "message_intent": message_intent,
                "topics": list(set(topics))[:3],
                "categories": list(set(categories))[:2],
                "entities": entities,
                "importance": importance,
                "confidence": 0.85,
                "urgency": "medium",
                "sentiment": sentiment,
                "emotional_classification": emotional,
                "temporal_qualifier": temporal,
                "domain_label": domain,
            }
        )

    def extract_full_metadata(
        self,
        content: str,
        user_id: str,
        session_id: str,
        turn_index: int = 0,
    ) -> dict[str, Any]:
        """Extract FULL metadata from content using LLM.

        Returns all fields required by SVL EnforcedMetadata.
        """
        prompt = FULL_EXTRACTION_PROMPT.format(
            content=content,
            user_id=user_id,
            session_id=session_id,
            turn_index=turn_index,
        )

        try:
            response = self._llm_call(prompt)

            # Parse JSON response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            metadata = json.loads(response.strip())

            # Validate and normalize ALL fields
            return self._normalize_metadata(metadata)

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            # Return safe defaults with ALL required fields
            return {
                "memory_type": "episodic",
                "message_type": "statement",
                "message_intent": "provide_info",
                "topics": ["benchmark"],
                "categories": ["general"],
                "entities": [],
                "importance": 0.5,
                "confidence": 0.0,  # Low confidence for fallback
                "urgency": "medium",
                "sentiment": "neutral",
                "emotional_classification": "neutral",
                "temporal_qualifier": None,
                "domain_label": None,
            }

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize all metadata fields to valid SVL values."""
        return {
            "memory_type": self._validate_enum(
                metadata.get("memory_type", "episodic"),
                VALID_MEMORY_TYPES,
                "episodic",
            ),
            "message_type": self._validate_enum(
                metadata.get("message_type", "statement"),
                VALID_MESSAGE_TYPES,
                "statement",
            ),
            "message_intent": self._validate_enum(
                metadata.get("message_intent", "provide_info"),
                VALID_MESSAGE_INTENTS,
                "provide_info",
            ),
            "topics": self._normalize_list(metadata.get("topics", []), self.valid_topics, 3),
            "categories": self._normalize_list(
                metadata.get("categories", []), self.valid_categories, 2
            ),
            "entities": metadata.get("entities", [])[:5],  # Max 5 entities
            "importance": max(0.0, min(1.0, float(metadata.get("importance", 0.5)))),
            "confidence": max(0.0, min(1.0, float(metadata.get("confidence", 0.8)))),
            "urgency": self._validate_enum(
                metadata.get("urgency", "medium"),
                VALID_URGENCIES,
                "medium",
            ),
            "sentiment": self._validate_enum(
                metadata.get("sentiment", "neutral"),
                VALID_SENTIMENTS,
                "neutral",
            ),
            "emotional_classification": self._validate_enum(
                metadata.get("emotional_classification", "neutral"),
                VALID_EMOTIONAL_CLASSIFICATIONS,
                "neutral",
            ),
            "temporal_qualifier": self._validate_enum(
                metadata.get("temporal_qualifier"),
                VALID_TEMPORAL_QUALIFIERS,
                None,
            ),
            "domain_label": metadata.get("domain_label"),
        }

    def _validate_enum(
        self, value: str | None, valid_values: list[str], default: str | None
    ) -> str | None:
        """Validate enum value against valid options."""
        if value is None:
            return default
        value = str(value).lower().strip()
        if value in valid_values:
            return value
        return default

    def _normalize_list(self, items: list, valid_items: list[str], max_items: int) -> list[str]:
        """Normalize list to valid SVL items."""
        normalized = []
        for item in items[:max_items]:
            item = str(item).lower().strip()
            if item in valid_items:
                normalized.append(item)
            else:
                # Try to find close match
                for valid in valid_items:
                    if valid in item or item in valid:
                        normalized.append(valid)
                        break
                else:
                    normalized.append("benchmark")  # Fallback

        return list(set(normalized)) or ["benchmark"]

    def enrich_memory(
        self,
        content: str,
        user_id: str,
        session_id: str,
        turn_index: int = 0,
        original_role: str = "user",
        timestamp: datetime | None = None,
        agent_id: str | None = None,
    ) -> EnrichedMemory:
        """Enrich a single memory with FULL SVL metadata.

        Args:
            content: Memory content text
            user_id: User identifier
            session_id: Session identifier
            turn_index: Turn index in session
            original_role: Original speaker role
            timestamp: Memory timestamp
            agent_id: Agent identifier (optional)

        Returns:
            EnrichedMemory with complete SVL-compliant metadata
        """
        # Generate unique message_id
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        # Extract metadata from LLM
        extracted = self.extract_full_metadata(content, user_id, session_id, turn_index)

        # Build full metadata
        metadata = FullMetadata(
            message_id=message_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            thread_id=None,
            topics=extracted["topics"],
            categories=extracted["categories"],
            entities=extracted["entities"],
            message_type=extracted["message_type"],
            message_intent=extracted["message_intent"],
            importance=extracted["importance"],
            confidence=extracted["confidence"],
            urgency=extracted["urgency"],
            sentiment=extracted["sentiment"],
            emotional_classification=extracted["emotional_classification"],
            temporal_qualifier=extracted["temporal_qualifier"],
            domain_label=extracted["domain_label"],
            memory_type=extracted["memory_type"],
            access_level="private",
            created_at=timestamp or datetime.now(timezone.utc),
        )

        return EnrichedMemory(
            content=content,
            metadata=metadata,
            original_role=original_role,
            original_turn_index=turn_index,
            enriched_by=self.provider.value,
            extraction_confidence=extracted["confidence"],
        )

    def enrich_dataset(
        self,
        raw_data: list[dict[str, Any]],
        source_name: str = "custom",
    ) -> EnrichedDataset:
        """Enrich a list of raw data items with FULL metadata.

        Args:
            raw_data: List of dicts with at least 'content' and 'user_id'
            source_name: Name of the data source

        Returns:
            EnrichedDataset ready for benchmarking with SVL
        """
        enriched_memories = []

        for i, item in enumerate(raw_data):
            memory = self.enrich_memory(
                content=item["content"],
                user_id=item.get("user_id", "user_001"),
                session_id=item.get("session_id", f"session_{uuid.uuid4().hex[:8]}"),
                turn_index=item.get("turn_index", i),
                original_role=item.get("role", "user"),
                timestamp=item.get("timestamp"),
                agent_id=item.get("agent_id"),
            )
            enriched_memories.append(memory)

        return EnrichedDataset(
            name=f"enriched_{source_name}",
            source=source_name,
            memories=enriched_memories,
            metadata={
                "enriched_at": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider.value,
                "total_items": len(enriched_memories),
                "has_full_metadata": True,
            },
        )


def create_sample_enriched_dataset() -> EnrichedDataset:
    """Create a sample enriched dataset with FULL metadata for testing.

    This demonstrates the proper format for benchmark data that
    passes through SVL validation.
    """
    pipeline = DatasetEnrichmentPipeline(llm_provider="local")

    sample_data = [
        {
            "content": "I prefer dark mode for all my applications.",
            "user_id": "user_001",
            "session_id": "session_001",
            "role": "user",
        },
        {
            "content": "Python is my favorite programming language.",
            "user_id": "user_001",
            "session_id": "session_001",
            "role": "user",
        },
        {
            "content": "I work remotely and prefer async communication.",
            "user_id": "user_001",
            "session_id": "session_002",
            "role": "user",
        },
        {
            "content": "Last week I attended a conference about AI safety.",
            "user_id": "user_002",
            "session_id": "session_003",
            "role": "user",
        },
        {
            "content": "I like light mode for reading documentation.",
            "user_id": "user_002",
            "session_id": "session_003",
            "role": "user",
        },
    ]

    return pipeline.enrich_dataset(sample_data, source_name="sample")


if __name__ == "__main__":
    # Create and save sample dataset
    dataset = create_sample_enriched_dataset()
    dataset.save("sample_enriched.json")
    print(f"Created sample enriched dataset with {dataset.size} memories")
    print()

    # Show full metadata for first memory
    for mem in dataset.memories[:2]:
        print(f"Content: {mem.content}")
        print("Full Metadata:")
        for key, value in mem.metadata.to_dict().items():
            print(f"  {key}: {value}")
        print()
