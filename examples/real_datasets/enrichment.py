"""Metadata enrichment pipeline for dataset segments.

This module uses LLMs to generate full SVL-compliant metadata for
raw conversation data from datasets like LoCoMo, Persona-Chat, and MultiWOZ.

The enrichment process:
1. Takes raw conversation turns from downloaded datasets
2. Uses LLM to extract entities, topics, intents, sentiment, etc.
3. Generates full EnforcedMetadata that SVL can validate
4. Produces EnrichedMemory objects ready for PostgreSQL storage
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from examples.real_datasets.downloader import ConversationSession, ConversationTurn, DatasetSegment
from examples.real_datasets.postgres_store import EnrichedMemory


logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """Configuration for the enrichment pipeline."""

    # LLM provider settings
    llm_provider: str = "openai"  # openai, anthropic, local
    api_key: str | None = None
    model: str | None = None  # Provider default if not specified

    # Enrichment behavior
    batch_size: int = 10  # Memories to enrich per LLM call
    max_retries: int = 3
    temperature: float = 0.0  # Deterministic

    # SVL vocabulary (provided by caller)
    vocabulary_topics: list[str] = field(default_factory=list)
    vocabulary_categories: list[str] = field(default_factory=list)

    # Default values for missing metadata
    default_importance: float = 0.5
    default_confidence: float = 0.8


class DatasetMetadataEnricher:
    """Enriches dataset conversations with SVL-compliant metadata.

    Uses LLM to extract:
    - Topics and categories from SVL vocabulary
    - Entities mentioned in the content
    - Message type and intent
    - Sentiment and emotional classification
    - Importance and urgency scores
    - Memory type classification
    """

    def __init__(self, config: EnrichmentConfig | None = None):
        """Initialize the enricher.

        Args:
            config: Enrichment configuration
        """
        self.config = config or EnrichmentConfig()
        self._client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Setup LLM client based on provider."""
        provider = self.config.llm_provider.lower()

        if provider == "openai":
            self._setup_openai()
        elif provider == "anthropic":
            self._setup_anthropic()
        elif provider == "local":
            # Use rule-based extraction
            self._client = None
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _setup_openai(self) -> None:
        """Setup OpenAI client."""
        try:
            import openai

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client = openai.OpenAI(api_key=api_key)
                self._provider = "openai"
            else:
                logger.warning("No OpenAI API key, falling back to local extraction")
                self._client = None
        except ImportError:
            logger.warning("OpenAI not installed, falling back to local extraction")
            self._client = None

    def _setup_anthropic(self) -> None:
        """Setup Anthropic client."""
        try:
            import anthropic

            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
                self._provider = "anthropic"
            else:
                logger.warning("No Anthropic API key, falling back to local extraction")
                self._client = None
        except ImportError:
            logger.warning("Anthropic not installed, falling back to local extraction")
            self._client = None

    def enrich_segment(
        self,
        segment: DatasetSegment,
        progress_callback: callable | None = None,
    ) -> list[EnrichedMemory]:
        """Enrich all conversations in a dataset segment.

        Args:
            segment: DatasetSegment to enrich
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of EnrichedMemory objects
        """
        all_memories = []
        total_turns = segment.total_turns
        processed = 0

        for session in segment.sessions:
            session_memories = self.enrich_session(
                session=session,
                dataset_name=segment.dataset_name,
            )
            all_memories.extend(session_memories)

            processed += len(session.turns)
            if progress_callback:
                progress_callback(processed, total_turns)

        logger.info(
            f"Enriched {len(all_memories)} memories from "
            f"{len(segment.sessions)} sessions in {segment.dataset_name}"
        )

        return all_memories

    def enrich_session(
        self,
        session: ConversationSession,
        dataset_name: str = "",
    ) -> list[EnrichedMemory]:
        """Enrich a single conversation session.

        Args:
            session: ConversationSession to enrich
            dataset_name: Source dataset name

        Returns:
            List of EnrichedMemory objects
        """
        memories = []

        # Collect user turns for batch enrichment
        user_turns = [t for t in session.turns if t.role == "user"]

        if self._client:
            # Use LLM for enrichment
            enriched_data = self._batch_enrich_llm(
                turns=user_turns,
                session=session,
                dataset_name=dataset_name,
            )
        else:
            # Use rule-based extraction
            enriched_data = self._batch_enrich_local(
                turns=user_turns,
                session=session,
                dataset_name=dataset_name,
            )

        for turn, metadata in zip(user_turns, enriched_data, strict=False):
            memory = EnrichedMemory(
                content=turn.content,
                memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                user_id=session.user_id,
                session_id=session.session_id,
                message_id=f"msg_{uuid.uuid4().hex[:12]}",
                topics=metadata.get("topics", []),
                categories=metadata.get("categories", []),
                entities=metadata.get("entities", []),
                message_type=metadata.get("message_type", "statement"),
                message_intent=metadata.get("message_intent", "provide_info"),
                importance=metadata.get("importance", self.config.default_importance),
                confidence=metadata.get("confidence", self.config.default_confidence),
                urgency=metadata.get("urgency", "medium"),
                sentiment=metadata.get("sentiment", "neutral"),
                emotional_classification=metadata.get("emotional_classification", "neutral"),
                temporal_qualifier=metadata.get("temporal_qualifier"),
                domain_label=metadata.get("domain_label", session.domain),
                memory_type=metadata.get("memory_type", "episodic"),
                access_level="private",
                dataset_name=dataset_name,
                turn_index=turn.turn_index,
            )
            memories.append(memory)

        return memories

    def _batch_enrich_llm(
        self,
        turns: list[ConversationTurn],
        session: ConversationSession,
        dataset_name: str,
    ) -> list[dict[str, Any]]:
        """Batch enrich turns using LLM.

        Args:
            turns: Conversation turns to enrich
            session: Parent session
            dataset_name: Source dataset

        Returns:
            List of metadata dicts
        """
        if not turns:
            return []

        # Build prompt with all turns
        prompt = self._build_enrichment_prompt(turns, session)

        try:
            if self._provider == "openai":
                response = self._call_openai(prompt)
            elif self._provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return self._batch_enrich_local(turns, session, dataset_name)

            # Parse response
            metadata_list = self._parse_llm_response(response, len(turns))
            return metadata_list

        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}, falling back to local")
            return self._batch_enrich_local(turns, session, dataset_name)

    def _build_enrichment_prompt(
        self,
        turns: list[ConversationTurn],
        session: ConversationSession,
    ) -> str:
        """Build the enrichment prompt for LLM."""
        # Format turns
        turns_text = "\n".join([f'{i+1}. "{turn.content}"' for i, turn in enumerate(turns)])

        # Format vocabulary
        topics_text = (
            ", ".join(self.config.vocabulary_topics[:50])
            or "general, personal, work, technology, lifestyle"
        )
        categories_text = (
            ", ".join(self.config.vocabulary_categories[:30]) or "general, user_preference, system"
        )

        # Persona context if available
        persona_context = ""
        if session.persona:
            persona_context = f"\nUser persona traits: {', '.join(session.persona)}"

        return f"""Extract SVL-compliant metadata for each user message.

## Context
Dataset: {session.metadata.get('dataset', 'unknown')}
User ID: {session.user_id}
Session ID: {session.session_id}{persona_context}

## User Messages to Analyze
{turns_text}

## Available Vocabulary

Topics (choose 1-3 per message):
{topics_text}

Categories (choose 1-2 per message):
{categories_text}

Message Types: query, command, statement, feedback, response, clarification, suggestion, confirmation
Message Intents: ask_question, request_action, provide_info, give_feedback, greeting, farewell, express_preference, share_experience
Memory Types: episodic, semantic, procedural, preference, entity, working
Sentiments: positive, negative, neutral, mixed
Urgency: critical, high, medium, low, informational
Emotional: neutral, joy, sadness, anger, fear, surprise, trust, anticipation

## Required Output Format

Return a JSON array with one object per message. Each object must have:
{{
    "topics": ["topic1", "topic2"],
    "categories": ["category1"],
    "entities": ["extracted entity names"],
    "message_type": "statement|query|command|...",
    "message_intent": "provide_info|ask_question|...",
    "importance": 0.0-1.0,
    "confidence": 0.0-1.0,
    "urgency": "medium",
    "sentiment": "neutral",
    "emotional_classification": "neutral",
    "temporal_qualifier": null or "past_event|current|future_plan|recurring",
    "memory_type": "episodic|semantic|preference|..."
}}

Rules:
1. Topics and categories MUST be from the vocabulary provided
2. Extract ALL entities mentioned (names, tools, places, etc.)
3. Importance: 0.1=trivial, 0.5=normal, 0.8=important, 0.95=critical
4. Preference statements should have memory_type="preference"
5. Facts about tools/concepts should have memory_type="semantic"
6. Personal experiences should have memory_type="episodic"

Respond with valid JSON array only, no other text."""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        model = self.config.model or "gpt-4o-mini"

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        model = self.config.model or "claude-3-haiku-20240307"

        response = self._client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _parse_llm_response(
        self,
        response: str,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        """Parse LLM response into metadata list."""
        try:
            # Try direct parse
            data = json.loads(response)

            # Handle wrapped response
            if isinstance(data, dict):
                if "messages" in data:
                    data = data["messages"]
                elif "metadata" in data:
                    data = data["metadata"]
                elif "results" in data:
                    data = data["results"]
                else:
                    # Single object, wrap in list
                    data = [data]

            if not isinstance(data, list):
                data = [data]

            # Pad or truncate to expected count
            while len(data) < expected_count:
                data.append(self._default_metadata())
            data = data[:expected_count]

            # Validate and clean each entry
            return [self._validate_metadata(m) for m in data]

        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    while len(data) < expected_count:
                        data.append(self._default_metadata())
                    return [self._validate_metadata(m) for m in data[:expected_count]]
                except json.JSONDecodeError:
                    pass

            # Fall back to defaults
            logger.warning("Could not parse LLM response, using defaults")
            return [self._default_metadata() for _ in range(expected_count)]

    def _validate_metadata(self, metadata: dict) -> dict:
        """Validate and clean metadata dict."""
        valid = self._default_metadata()

        # Copy valid fields
        if "topics" in metadata and isinstance(metadata["topics"], list):
            valid["topics"] = [t for t in metadata["topics"] if isinstance(t, str)][:5]

        if "categories" in metadata and isinstance(metadata["categories"], list):
            valid["categories"] = [c for c in metadata["categories"] if isinstance(c, str)][:3]

        if "entities" in metadata and isinstance(metadata["entities"], list):
            valid["entities"] = [e for e in metadata["entities"] if isinstance(e, str)]

        for field in [
            "message_type",
            "message_intent",
            "sentiment",
            "emotional_classification",
            "urgency",
            "memory_type",
        ]:
            if field in metadata and isinstance(metadata[field], str):
                valid[field] = metadata[field]

        for field in ["importance", "confidence"]:
            if field in metadata:
                try:
                    val = float(metadata[field])
                    valid[field] = max(0.0, min(1.0, val))
                except (TypeError, ValueError):
                    pass

        if metadata.get("temporal_qualifier"):
            valid["temporal_qualifier"] = str(metadata["temporal_qualifier"])

        return valid

    def _default_metadata(self) -> dict:
        """Get default metadata values."""
        return {
            "topics": ["general"],
            "categories": ["general"],
            "entities": [],
            "message_type": "statement",
            "message_intent": "provide_info",
            "importance": self.config.default_importance,
            "confidence": self.config.default_confidence,
            "urgency": "medium",
            "sentiment": "neutral",
            "emotional_classification": "neutral",
            "temporal_qualifier": None,
            "memory_type": "episodic",
        }

    def _batch_enrich_local(
        self,
        turns: list[ConversationTurn],
        session: ConversationSession,
        dataset_name: str,
    ) -> list[dict[str, Any]]:
        """Rule-based enrichment when LLM is unavailable.

        Uses keyword matching, patterns, and heuristics to extract
        metadata from conversation turns.
        """
        results = []

        for turn in turns:
            metadata = self._enrich_single_local(turn, session)
            results.append(metadata)

        return results

    def _enrich_single_local(
        self,
        turn: ConversationTurn,
        session: ConversationSession,
    ) -> dict[str, Any]:
        """Rule-based enrichment for a single turn."""
        content = turn.content.lower()
        metadata = self._default_metadata()

        # Extract topics based on keywords
        topic_keywords = {
            "technology": [
                "python",
                "javascript",
                "code",
                "programming",
                "software",
                "api",
                "database",
            ],
            "work": ["work", "job", "project", "meeting", "team", "deadline", "office"],
            "personal": ["i ", "my ", "me ", "myself", "prefer", "like", "love", "hate"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "destination"],
            "food": ["food", "eat", "restaurant", "cook", "meal", "lunch", "dinner"],
            "health": ["health", "exercise", "workout", "doctor", "medicine", "sleep"],
            "entertainment": ["movie", "music", "game", "book", "show", "watch", "play"],
            "communication": ["email", "message", "call", "chat", "slack", "notification"],
            "settings": ["dark mode", "light mode", "setting", "preference", "configure"],
        }

        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in content for kw in keywords):
                detected_topics.append(topic)

        if detected_topics:
            metadata["topics"] = detected_topics[:3]
        elif session.persona:
            # Use persona as topic hint
            for trait in session.persona[:2]:
                metadata["topics"].append(trait.split()[0].lower())

        # Determine message type and intent
        if "?" in turn.content:
            metadata["message_type"] = "query"
            metadata["message_intent"] = "ask_question"
        elif any(cmd in content for cmd in ["please", "can you", "could you", "would you"]):
            metadata["message_type"] = "command"
            metadata["message_intent"] = "request_action"
        elif any(
            pref in content for pref in ["i prefer", "i like", "i love", "i hate", "favorite"]
        ):
            metadata["message_type"] = "statement"
            metadata["message_intent"] = "express_preference"
            metadata["memory_type"] = "preference"
            metadata["importance"] = 0.7
        elif any(exp in content for exp in ["i am", "i work", "i have", "i do"]):
            metadata["message_type"] = "statement"
            metadata["message_intent"] = "share_experience"
            metadata["memory_type"] = "semantic"
        elif any(greet in content for greet in ["hi", "hello", "hey"]):
            metadata["message_type"] = "statement"
            metadata["message_intent"] = "greeting"
            metadata["importance"] = 0.2

        # Determine sentiment
        positive_words = [
            "love",
            "great",
            "excellent",
            "amazing",
            "good",
            "wonderful",
            "perfect",
            "happy",
        ]
        negative_words = ["hate", "terrible", "awful", "bad", "horrible", "frustrated", "annoyed"]

        pos_count = sum(1 for w in positive_words if w in content)
        neg_count = sum(1 for w in negative_words if w in content)

        if pos_count > neg_count:
            metadata["sentiment"] = "positive"
            metadata["emotional_classification"] = "joy"
        elif neg_count > pos_count:
            metadata["sentiment"] = "negative"
            metadata["emotional_classification"] = "frustration"

        # Extract entities (simple NER)
        entities = []

        # Capitalized words (potential names/proper nouns)
        words = turn.content.split()
        for word in words:
            if (
                word[0].isupper()
                and len(word) > 1
                and word not in ["I", "I'm", "I've", "Hi", "Hello", "Hey"]
            ):
                clean = word.strip(".,!?\"'")
                if clean and len(clean) > 1:
                    entities.append(clean)

        # Known tools/technologies
        tech_entities = [
            "Python",
            "JavaScript",
            "TypeScript",
            "React",
            "FastAPI",
            "Django",
            "PostgreSQL",
            "MongoDB",
            "Redis",
            "Docker",
            "Kubernetes",
            "AWS",
            "Figma",
            "Notion",
            "Slack",
            "GitHub",
            "VSCode",
            "Jupyter",
        ]
        for tech in tech_entities:
            if tech.lower() in content:
                entities.append(tech)

        metadata["entities"] = list(set(entities))[:10]

        # Determine urgency
        if any(urg in content for urg in ["urgent", "asap", "immediately", "critical"]):
            metadata["urgency"] = "high"
        elif any(urg in content for urg in ["soon", "quickly", "priority"]):
            metadata["urgency"] = "medium"

        # Temporal qualifiers
        if any(t in content for t in ["yesterday", "last week", "before", "used to"]):
            metadata["temporal_qualifier"] = "past_event"
        elif any(t in content for t in ["tomorrow", "next week", "will", "going to"]):
            metadata["temporal_qualifier"] = "future_plan"
        elif any(t in content for t in ["always", "usually", "every"]):
            metadata["temporal_qualifier"] = "recurring"

        # Categories based on domain
        if session.domain:
            metadata["categories"] = [session.domain, "task_oriented"]
        elif "prefer" in content or "like" in content:
            metadata["categories"] = ["user_preference"]

        return metadata


class EnrichmentPipeline:
    """Complete pipeline for downloading, enriching, and storing datasets."""

    def __init__(
        self,
        postgres_dsn: str,
        enrichment_config: EnrichmentConfig | None = None,
    ):
        """Initialize the pipeline.

        Args:
            postgres_dsn: PostgreSQL connection string
            enrichment_config: Enrichment configuration
        """
        from examples.real_datasets.downloader import DatasetDownloader
        from examples.real_datasets.postgres_store import PostgresDatasetStore

        self.downloader = DatasetDownloader()
        self.enricher = DatasetMetadataEnricher(enrichment_config)
        self.store = PostgresDatasetStore(dsn=postgres_dsn)

    def run(
        self,
        datasets: list[str] = ["locomo", "persona_chat", "multiwoz"],
        max_sessions_per_dataset: int = 50,
        recreate_schema: bool = False,
    ) -> dict[str, Any]:
        """Run the complete enrichment pipeline.

        Args:
            datasets: List of datasets to process
            max_sessions_per_dataset: Max sessions per dataset
            recreate_schema: Drop and recreate schema

        Returns:
            Pipeline results
        """
        results = {
            "datasets_processed": [],
            "total_memories": 0,
            "total_sessions": 0,
            "errors": [],
        }

        # Setup database
        if recreate_schema:
            self.store.drop_schema(cascade=True)
        self.store.create_schema()

        # Process each dataset
        for dataset_name in datasets:
            try:
                logger.info(f"Processing dataset: {dataset_name}")

                # Download
                if dataset_name == "locomo":
                    segment = self.downloader.download_locomo(max_sessions_per_dataset)
                elif dataset_name == "persona_chat":
                    segment = self.downloader.download_persona_chat(max_sessions_per_dataset)
                elif dataset_name == "multiwoz":
                    segment = self.downloader.download_multiwoz(max_sessions_per_dataset)
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue

                # Enrich
                memories = self.enricher.enrich_segment(segment)

                # Store sessions
                for session in segment.sessions:
                    self.store.store_session(
                        session_id=session.session_id,
                        user_id=session.user_id,
                        dataset_name=segment.dataset_name,
                        persona=session.persona,
                        domain=session.domain,
                        total_turns=len(session.turns),
                        metadata=session.metadata,
                    )

                # Store memories
                self.store.store_memories(memories)

                results["datasets_processed"].append(
                    {
                        "name": segment.dataset_name,
                        "sessions": len(segment.sessions),
                        "memories": len(memories),
                    }
                )
                results["total_memories"] += len(memories)
                results["total_sessions"] += len(segment.sessions)

                logger.info(
                    f"Completed {dataset_name}: "
                    f"{len(segment.sessions)} sessions, {len(memories)} memories"
                )

            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                results["errors"].append({"dataset": dataset_name, "error": str(e)})

        self.store.close()
        return results
