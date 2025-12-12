"""Deterministic Context Agent with 2-Call Architecture.

Replaces the multi-round tool-calling approach with a more predictable:
1. Extract intent (ONE LLM call)
2. Parallel data fetching (NO LLM, just queries)
3. Generate context (ONE LLM call)

Benefits:
- Predictable latency (exactly 2 LLM calls)
- Predictable cost (no variable tool-calling rounds)
- Easier to debug (explicit flow)
- Parallel data fetching for lower latency
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mindcore.core.schemas import AssembledContext, Message, ThreadSummary, UserPreferences
from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.logger import LogCategory, get_logger

from .base_agent import BaseAgent


if TYPE_CHECKING:
    from mindcore.llm import BaseLLMProvider

logger = get_logger(__name__, category=LogCategory.CONTEXT)


@dataclass
class QueryIntent:
    """Extracted intent from user query."""

    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    intent_type: str | None = None
    time_range: str = "recent"  # recent, historical, all
    needs_external_data: bool = False
    needs_user_preferences: bool = False
    needs_summaries: bool = False
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class FetchedData:
    """Container for all fetched data from parallel queries."""

    recent_messages: list[Message] = field(default_factory=list)
    historical_messages: list[Message] = field(default_factory=list)
    summaries: list[ThreadSummary] = field(default_factory=list)
    user_preferences: UserPreferences | None = None
    external_data: list[Any] = field(default_factory=list)
    session_metadata: dict[str, Any] = field(default_factory=dict)
    fetch_latency_ms: float = 0.0
    fetch_errors: list[str] = field(default_factory=list)


@dataclass
class ContextToolCallbacks:
    """Callbacks for fetching context data.

    All callbacks should be synchronous (the agent handles parallelization).
    """

    get_recent_messages: Callable  # (user_id, thread_id, limit) -> list[Message]
    search_history: Callable  # (user_id, thread_id, topics, categories, intent, limit) -> list[Message]
    get_session_metadata: Callable  # (user_id, thread_id) -> dict
    get_historical_summaries: Callable | None = None  # (user_id, topics, limit) -> list[ThreadSummary]
    get_user_preferences: Callable | None = None  # (user_id) -> UserPreferences
    lookup_external_data: Callable | None = None  # (user_id, topics, context) -> list[Any]


class DeterministicContextAgent(BaseAgent):
    """Deterministic 2-call context agent.

    Architecture:
    ```
    Query → Extract Intent (1 LLM call)
          → Parallel Data Fetch (0 LLM calls, just queries)
          → Generate Context (1 LLM call)
    ```

    This provides:
    - Exactly 2 LLM calls (predictable cost)
    - Parallel data fetching (lower latency)
    - Explicit, debuggable flow

    Example:
        >>> agent = DeterministicContextAgent(
        ...     llm_provider=provider,
        ...     callbacks=ContextToolCallbacks(
        ...         get_recent_messages=lambda uid, tid, limit: [...],
        ...         search_history=lambda uid, tid, topics, cats, intent, limit: [...],
        ...         get_session_metadata=lambda uid, tid: {...},
        ...     )
        ... )
        >>> context = agent.process(
        ...     query="What did we discuss about billing?",
        ...     user_id="user123",
        ...     thread_id="thread456"
        ... )
    """

    # Fast path patterns for common queries (skip LLM intent extraction)
    FAST_PATH_PATTERNS = [
        (r"(?:my |the )?order[s]?(?:\s|$|#|\d)", ["orders", "tracking"], True),
        (r"(?:my |the )?bill(?:ing)?|invoice|payment|subscription", ["billing", "payment"], True),
        (r"(?:my |the )?delivery|shipping|track", ["delivery", "tracking"], True),
        (r"(?:my |the )?account|login|password|profile", ["account"], False),
        (r"last time|previously|before|earlier|remember when", [], False),  # Historical query
        (r"help|how (?:do|can) I|what is", ["help"], False),
    ]

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        callbacks: ContextToolCallbacks,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        vocabulary: VocabularyManager | None = None,
        parallel_workers: int = 4,
    ):
        """Initialize deterministic context agent.

        Args:
            llm_provider: LLM provider instance
            callbacks: Data fetching callbacks
            temperature: Temperature for LLM calls
            max_tokens: Maximum tokens per response
            vocabulary: VocabularyManager for constraints
            parallel_workers: Number of parallel fetch workers
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.callbacks = callbacks
        self.vocabulary = vocabulary or get_vocabulary()
        self.parallel_workers = parallel_workers

        self._intent_system_prompt = self._create_intent_prompt()
        self._context_system_prompt = self._create_context_prompt()

    def _create_intent_prompt(self) -> str:
        """Create system prompt for intent extraction."""
        vocab_prompt = self.vocabulary.to_prompt_list()

        return f"""You are a query intent extraction agent. Analyze the user's query and extract structured intent.

{vocab_prompt}

Return a JSON object with these fields:
{{
    "topics": ["1-3 relevant topics from Available Topics"],
    "categories": ["1-2 categories from Available Categories"],
    "intent_type": "one intent from Available Intents",
    "time_range": "recent|historical|all",
    "needs_external_data": true/false,
    "needs_user_preferences": true/false,
    "needs_summaries": true/false,
    "extracted_entities": {{"order_id": "...", "date": "...", etc.}},
    "confidence": 0.0-1.0
}}

Guidelines:
- "time_range": "recent" for current conversation, "historical" for past references, "all" for comprehensive
- "needs_external_data": true if query mentions orders, billing, subscriptions, deliveries
- "needs_user_preferences": true if personalization would help the response
- "needs_summaries": true if query references older conversations
- Extract any IDs, dates, or specific entities mentioned
- Be concise. Return ONLY valid JSON."""

    def _create_context_prompt(self) -> str:
        """Create system prompt for context generation."""
        return """You are a context assembly agent. Given fetched data, create a concise context summary.

Return a JSON object:
{
    "relevant_context": "Concise summary of relevant information",
    "key_points": ["Important point 1", "Important point 2"],
    "context_source": "recent|historical|summaries|preferences|external|multiple",
    "confidence": "high|medium|low"
}

Guidelines:
- Summarize only what's relevant to the query
- Be concise - the main AI doesn't need raw data
- Highlight key facts that help answer the query
- If external data (orders, billing) is present, include specific details
- Return ONLY valid JSON."""

    def process(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        additional_context: str | None = None,
    ) -> AssembledContext:
        """Assemble context using deterministic 2-call architecture.

        Flow:
        1. Try fast path (pattern matching, no LLM)
        2. If no fast path, extract intent (LLM call 1)
        3. Parallel data fetching (no LLM)
        4. Generate context (LLM call 2)

        Args:
            query: User's query
            user_id: User identifier
            thread_id: Thread identifier
            additional_context: Optional additional context

        Returns:
            AssembledContext with relevant information
        """
        start_time = time.time()
        llm_calls = 0

        logger.debug(
            "DeterministicContextAgent processing",
            query_preview=query[:100],
            user_id=user_id,
            thread_id=thread_id,
        )

        # Step 1: Try fast path first (no LLM call)
        intent = self._try_fast_path(query)

        if intent is None:
            # Step 2: Extract intent (LLM call 1)
            intent = self._extract_intent(query, additional_context)
            llm_calls += 1

        logger.debug(
            "Intent extracted",
            topics=intent.topics,
            time_range=intent.time_range,
            needs_external=intent.needs_external_data,
            fast_path=llm_calls == 0,
        )

        # Step 3: Parallel data fetching (no LLM calls)
        fetched_data = self._fetch_data_parallel(
            user_id=user_id,
            thread_id=thread_id,
            intent=intent,
        )

        # Step 4: Generate context (LLM call 2)
        context = self._generate_context(
            query=query,
            intent=intent,
            fetched_data=fetched_data,
            additional_context=additional_context,
        )
        llm_calls += 1

        total_latency = (time.time() - start_time) * 1000

        # Add metadata
        context.metadata["llm_calls"] = llm_calls
        context.metadata["total_latency_ms"] = total_latency
        context.metadata["fetch_latency_ms"] = fetched_data.fetch_latency_ms
        context.metadata["fast_path_used"] = llm_calls == 1
        context.metadata["intent_confidence"] = intent.confidence

        logger.info(
            f"Context assembled: llm_calls={llm_calls}, "
            f"latency={total_latency:.0f}ms, "
            f"fast_path={llm_calls == 1}"
        )

        return context

    def _try_fast_path(self, query: str) -> QueryIntent | None:
        """Try to extract intent without LLM using pattern matching.

        Returns QueryIntent if fast path succeeds, None otherwise.
        """
        query_lower = query.lower()

        for pattern, topics, needs_external in self.FAST_PATH_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Validate topics against vocabulary
                validated_topics = self.vocabulary.validate_topics(topics)

                # Check for historical markers
                is_historical = bool(
                    re.search(r"last time|previously|before|earlier|remember", query_lower)
                )

                intent = QueryIntent(
                    topics=validated_topics if validated_topics else ["general"],
                    categories=["support"] if needs_external else ["general"],
                    time_range="historical" if is_historical else "recent",
                    needs_external_data=needs_external,
                    needs_user_preferences=False,
                    needs_summaries=is_historical,
                    confidence=0.7,  # Fast path has moderate confidence
                )

                # Extract entities (order IDs, etc.)
                intent.extracted_entities = self._extract_entities_fast(query)

                logger.debug(f"Fast path matched: pattern={pattern}")
                return intent

        return None

    def _extract_entities_fast(self, query: str) -> dict[str, Any]:
        """Extract common entities from query using regex."""
        entities = {}

        # Order IDs (various formats)
        order_match = re.search(r"(?:order\s*(?:#|number|id)?:?\s*)([A-Z0-9-]{5,20})", query, re.I)
        if order_match:
            entities["order_id"] = order_match.group(1)

        # Invoice IDs
        invoice_match = re.search(r"(?:invoice\s*(?:#|number|id)?:?\s*)([A-Z0-9-]{5,20})", query, re.I)
        if invoice_match:
            entities["invoice_id"] = invoice_match.group(1)

        # Dates (basic patterns)
        date_match = re.search(
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", query
        )
        if date_match:
            entities["date"] = date_match.group(1)

        return entities

    def _extract_intent(self, query: str, additional_context: str | None) -> QueryIntent:
        """Extract intent using LLM (call 1 of 2)."""
        user_content = f"Query: {query}"
        if additional_context:
            user_content += f"\nAdditional context: {additional_context}"

        messages = [
            {"role": "system", "content": self._intent_system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self._call_llm(messages, json_mode=True)
            data = self._parse_json_response(response)

            # Validate against vocabulary
            raw_topics = data.get("topics", [])
            raw_categories = data.get("categories", [])

            validated_topics = self.vocabulary.validate_topics(raw_topics)
            validated_categories = self.vocabulary.validate_categories(raw_categories)

            return QueryIntent(
                topics=validated_topics if validated_topics else ["general"],
                categories=validated_categories if validated_categories else ["general"],
                intent_type=data.get("intent_type"),
                time_range=data.get("time_range", "recent"),
                needs_external_data=data.get("needs_external_data", False),
                needs_user_preferences=data.get("needs_user_preferences", False),
                needs_summaries=data.get("needs_summaries", False),
                extracted_entities=data.get("extracted_entities", {}),
                confidence=data.get("confidence", 0.5),
            )

        except Exception as e:
            logger.warning(f"Intent extraction failed: {e}, using defaults")
            return QueryIntent(
                topics=["general"],
                categories=["general"],
                time_range="recent",
                confidence=0.0,
            )

    def _fetch_data_parallel(
        self,
        user_id: str,
        thread_id: str,
        intent: QueryIntent,
    ) -> FetchedData:
        """Fetch all needed data in parallel (no LLM calls)."""
        start_time = time.time()
        result = FetchedData()

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {}

            # Always fetch recent messages
            futures["recent"] = executor.submit(
                self._safe_fetch,
                "recent_messages",
                self.callbacks.get_recent_messages,
                user_id,
                thread_id,
                10,
            )

            # Always fetch session metadata
            futures["metadata"] = executor.submit(
                self._safe_fetch,
                "session_metadata",
                self.callbacks.get_session_metadata,
                user_id,
                thread_id,
            )

            # Conditionally fetch historical messages
            if intent.time_range in ("historical", "all"):
                futures["historical"] = executor.submit(
                    self._safe_fetch,
                    "historical_messages",
                    self.callbacks.search_history,
                    user_id,
                    None,  # Search across all threads
                    intent.topics,
                    intent.categories,
                    intent.intent_type,
                    20,
                )

            # Conditionally fetch summaries
            if intent.needs_summaries and self.callbacks.get_historical_summaries:
                futures["summaries"] = executor.submit(
                    self._safe_fetch,
                    "summaries",
                    self.callbacks.get_historical_summaries,
                    user_id,
                    intent.topics if intent.topics else None,
                    5,
                )

            # Conditionally fetch user preferences
            if intent.needs_user_preferences and self.callbacks.get_user_preferences:
                futures["preferences"] = executor.submit(
                    self._safe_fetch,
                    "user_preferences",
                    self.callbacks.get_user_preferences,
                    user_id,
                )

            # Conditionally fetch external data
            if intent.needs_external_data and self.callbacks.lookup_external_data:
                futures["external"] = executor.submit(
                    self._safe_fetch,
                    "external_data",
                    self.callbacks.lookup_external_data,
                    user_id,
                    intent.topics,
                    intent.extracted_entities,
                )

            # Collect results
            for future_name, future in futures.items():
                try:
                    data_type, data = future.result(timeout=10.0)

                    if data_type == "recent_messages":
                        result.recent_messages = data or []
                    elif data_type == "historical_messages":
                        result.historical_messages = data or []
                    elif data_type == "session_metadata":
                        result.session_metadata = data or {}
                    elif data_type == "summaries":
                        result.summaries = data or []
                    elif data_type == "user_preferences":
                        result.user_preferences = data
                    elif data_type == "external_data":
                        result.external_data = data or []

                except Exception as e:
                    error_msg = f"Failed to fetch {future_name}: {e}"
                    logger.warning(error_msg)
                    result.fetch_errors.append(error_msg)

        result.fetch_latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Parallel fetch complete",
            recent_count=len(result.recent_messages),
            historical_count=len(result.historical_messages),
            summaries_count=len(result.summaries),
            has_preferences=result.user_preferences is not None,
            external_count=len(result.external_data),
            latency_ms=result.fetch_latency_ms,
            errors=len(result.fetch_errors),
        )

        return result

    def _safe_fetch(self, data_type: str, callback: callable, *args) -> tuple[str, Any]:
        """Safely execute a fetch callback."""
        try:
            data = callback(*args)
            return (data_type, data)
        except Exception as e:
            logger.warning(f"Fetch error for {data_type}: {e}")
            return (data_type, None)

    def _generate_context(
        self,
        query: str,
        intent: QueryIntent,
        fetched_data: FetchedData,
        additional_context: str | None,
    ) -> AssembledContext:
        """Generate final context using LLM (call 2 of 2)."""
        # Format fetched data for LLM
        data_summary = self._format_fetched_data(fetched_data)

        user_content = f"""Query: {query}

Extracted Intent:
- Topics: {', '.join(intent.topics)}
- Time range: {intent.time_range}
- Entities: {json.dumps(intent.extracted_entities) if intent.extracted_entities else 'None'}

Fetched Data:
{data_summary}"""

        if additional_context:
            user_content += f"\n\nAdditional Context: {additional_context}"

        messages = [
            {"role": "system", "content": self._context_system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self._call_llm(messages, json_mode=True)
            data = self._parse_json_response(response)

            return AssembledContext(
                assembled_context=data.get("relevant_context", ""),
                key_points=data.get("key_points", []),
                relevant_message_ids=[msg.message_id for msg in fetched_data.recent_messages[:5]],
                metadata={
                    "context_source": data.get("context_source", "unknown"),
                    "confidence": data.get("confidence", "medium"),
                    "query": query,
                    "topics_used": intent.topics,
                },
            )

        except Exception as e:
            logger.warning(f"Context generation failed: {e}")

            # Fallback: return basic context
            return self._create_fallback_context(query, fetched_data)

    def _format_fetched_data(self, data: FetchedData) -> str:
        """Format fetched data for LLM consumption."""
        parts = []

        # Recent messages
        if data.recent_messages:
            parts.append("=== Recent Messages ===")
            for msg in data.recent_messages[:10]:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                text_preview = msg.raw_text[:300] if msg.raw_text else ""
                parts.append(f"[{role}]: {text_preview}")

        # Historical messages
        if data.historical_messages:
            parts.append("\n=== Historical Messages ===")
            for msg in data.historical_messages[:10]:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                text_preview = msg.raw_text[:200] if msg.raw_text else ""
                topics = ", ".join(msg.metadata.topics) if msg.metadata.topics else "none"
                parts.append(f"[{role}] (topics: {topics}): {text_preview}")

        # Summaries
        if data.summaries:
            parts.append("\n=== Historical Summaries ===")
            for summary in data.summaries[:5]:
                parts.append(f"Thread {summary.thread_id}: {summary.summary[:300]}")

        # User preferences
        if data.user_preferences:
            parts.append("\n=== User Preferences ===")
            prefs = data.user_preferences
            if prefs.preferred_name:
                parts.append(f"Name: {prefs.preferred_name}")
            if prefs.interests:
                parts.append(f"Interests: {', '.join(prefs.interests)}")
            if prefs.communication_style != "balanced":
                parts.append(f"Style: {prefs.communication_style}")

        # External data
        if data.external_data:
            parts.append("\n=== External Data ===")
            for item in data.external_data[:5]:
                if hasattr(item, "to_context_string"):
                    parts.append(item.to_context_string())
                else:
                    parts.append(str(item)[:500])

        # Session metadata
        if data.session_metadata:
            parts.append("\n=== Session Info ===")
            if data.session_metadata.get("topics"):
                parts.append(f"Session topics: {', '.join(data.session_metadata['topics'][:5])}")
            if data.session_metadata.get("message_count"):
                parts.append(f"Message count: {data.session_metadata['message_count']}")

        return "\n".join(parts) if parts else "No data fetched."

    def _create_fallback_context(self, query: str, data: FetchedData) -> AssembledContext:
        """Create fallback context when LLM generation fails."""
        context_parts = []

        if data.recent_messages:
            context_parts.append(
                f"Recent conversation has {len(data.recent_messages)} messages."
            )

        if data.external_data:
            context_parts.append(f"External data available: {len(data.external_data)} items.")

        return AssembledContext(
            assembled_context=" ".join(context_parts) if context_parts else "Unable to assemble context.",
            key_points=[],
            relevant_message_ids=[msg.message_id for msg in data.recent_messages[:3]],
            metadata={
                "context_source": "fallback",
                "confidence": "low",
                "query": query,
                "error": "Context generation failed",
            },
        )

    def refresh_vocabulary(self) -> None:
        """Refresh vocabulary and update prompts."""
        self.vocabulary.refresh_from_external()
        self._intent_system_prompt = self._create_intent_prompt()
        logger.info("DeterministicContextAgent vocabulary refreshed")
