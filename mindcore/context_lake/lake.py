"""Context Lake - Unified context aggregation layer.

Aggregates context from multiple sources:
- PostgreSQL (structured message history)
- VectorDB (semantic knowledge base)
- Data Connectors (external systems)
- API Listeners (real-time data)

The Context Lake provides a single interface for the DeterministicContextAgent
to query all context sources in parallel and receive a unified result.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from mindcore.core.schemas import AssembledContext, Message, ThreadSummary, UserPreferences
from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.logger import get_logger


if TYPE_CHECKING:
    from mindcore.connectors import ConnectorRegistry
    from mindcore.core import DatabaseManager, SQLiteManager

    from .api_listener import APIListenerRegistry
    from .knowledge_base import KnowledgeBase

logger = get_logger(__name__)


class ContextSource(str, Enum):
    """Sources of context data."""

    MESSAGES = "messages"  # PostgreSQL message history
    KNOWLEDGE_BASE = "knowledge_base"  # VectorDB semantic search
    CONNECTORS = "connectors"  # External data connectors
    API_LISTENERS = "api_listeners"  # Real-time API data
    SUMMARIES = "summaries"  # Thread summaries
    PREFERENCES = "preferences"  # User preferences


@dataclass
class ContextQuery:
    """Query parameters for the Context Lake.

    Specifies what context to retrieve and from which sources.
    """

    # Required identifiers
    user_id: str
    query: str

    # Optional identifiers (for scoping)
    thread_id: str | None = None
    session_id: str | None = None

    # Filter parameters (mapped to PostgreSQL columns)
    topics: list[str] | None = None
    categories: list[str] | None = None
    intent: str | None = None
    sentiment: str | None = None
    min_confidence: float | None = None
    min_importance: float | None = None

    # Time filters
    time_range: str = "recent"  # recent, day, week, month, all
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Source toggles
    include_messages: bool = True
    include_knowledge_base: bool = True
    include_external_data: bool = True
    include_api_data: bool = True
    include_summaries: bool = True
    include_preferences: bool = True

    # Limits
    max_messages: int = 20
    max_knowledge_results: int = 5
    max_external_items: int = 10

    # Extracted entities (from query analysis)
    entities: dict[str, Any] = field(default_factory=dict)

    def get_time_range(self) -> tuple[datetime | None, datetime | None]:
        """Get start and end dates based on time_range."""
        if self.start_date and self.end_date:
            return (self.start_date, self.end_date)

        now = datetime.now(timezone.utc)

        if self.time_range == "recent":
            return (now - timedelta(hours=24), now)
        elif self.time_range == "day":
            return (now - timedelta(days=1), now)
        elif self.time_range == "week":
            return (now - timedelta(weeks=1), now)
        elif self.time_range == "month":
            return (now - timedelta(days=30), now)
        else:  # all
            return (None, None)


@dataclass
class ContextResult:
    """Result from Context Lake query.

    Contains data from all queried sources.
    """

    # Source data
    messages: list[Message] = field(default_factory=list)
    knowledge_results: list[dict[str, Any]] = field(default_factory=list)
    external_data: list[dict[str, Any]] = field(default_factory=list)
    api_data: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[ThreadSummary] = field(default_factory=list)
    preferences: UserPreferences | None = None

    # Metadata
    sources_queried: list[ContextSource] = field(default_factory=list)
    sources_with_data: list[ContextSource] = field(default_factory=list)
    total_items: int = 0
    query_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    # Aggregated topics and categories from all sources
    aggregated_topics: list[str] = field(default_factory=list)
    aggregated_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages_count": len(self.messages),
            "knowledge_results_count": len(self.knowledge_results),
            "external_data_count": len(self.external_data),
            "api_data_count": len(self.api_data),
            "summaries_count": len(self.summaries),
            "has_preferences": self.preferences is not None,
            "sources_queried": [s.value for s in self.sources_queried],
            "sources_with_data": [s.value for s in self.sources_with_data],
            "total_items": self.total_items,
            "query_latency_ms": self.query_latency_ms,
            "errors": self.errors,
            "aggregated_topics": self.aggregated_topics,
            "aggregated_categories": self.aggregated_categories,
        }


@dataclass
class ContextLakeConfig:
    """Configuration for Context Lake."""

    # Parallel fetching
    max_workers: int = 6
    fetch_timeout: float = 10.0

    # PostgreSQL query settings
    message_limit: int = 50
    include_low_confidence: bool = False
    min_confidence_threshold: float = 0.3

    # Knowledge base settings
    similarity_threshold: float = 0.7
    knowledge_limit: int = 10

    # Connector settings
    connector_timeout: float = 5.0

    # API listener settings
    api_cache_ttl: int = 60  # seconds

    # Context generation
    max_context_tokens: int = 4000
    summarize_long_context: bool = True


class ContextLake:
    """Unified context aggregation layer.

    The Context Lake is the central hub that:
    1. Receives queries from the DeterministicContextAgent
    2. Queries all configured sources in parallel
    3. Aggregates and deduplicates results
    4. Returns a unified ContextResult

    Example:
        lake = ContextLake(
            db_manager=postgres_manager,
            knowledge_base=kb,
            connector_registry=connectors,
            api_registry=api_listeners,
        )

        result = lake.query(ContextQuery(
            user_id="user123",
            query="What's my order status?",
            topics=["orders"],
            include_external_data=True,
        ))
    """

    def __init__(
        self,
        db_manager: "DatabaseManager | SQLiteManager",
        knowledge_base: "KnowledgeBase | None" = None,
        connector_registry: "ConnectorRegistry | None" = None,
        api_registry: "APIListenerRegistry | None" = None,
        vocabulary: VocabularyManager | None = None,
        config: ContextLakeConfig | None = None,
    ):
        """Initialize Context Lake.

        Args:
            db_manager: PostgreSQL or SQLite database manager
            knowledge_base: Optional knowledge base with VectorDB
            connector_registry: Optional connector registry for external data
            api_registry: Optional API listener registry
            vocabulary: Vocabulary manager for topic/category mapping
            config: Configuration options
        """
        self.db = db_manager
        self.knowledge_base = knowledge_base
        self.connectors = connector_registry
        self.api_registry = api_registry
        self.vocabulary = vocabulary or get_vocabulary()
        self.config = config or ContextLakeConfig()

        logger.info(
            f"ContextLake initialized: "
            f"kb={knowledge_base is not None}, "
            f"connectors={connector_registry is not None}, "
            f"api_listeners={api_registry is not None}"
        )

    def query(self, query: ContextQuery) -> ContextResult:
        """Query the Context Lake for relevant context.

        Queries all enabled sources in parallel and aggregates results.

        Args:
            query: Query parameters specifying what context to retrieve

        Returns:
            ContextResult with data from all sources
        """
        start_time = time.time()
        result = ContextResult()

        # Determine which sources to query
        sources_to_query = []
        if query.include_messages:
            sources_to_query.append(ContextSource.MESSAGES)
        if query.include_knowledge_base and self.knowledge_base:
            sources_to_query.append(ContextSource.KNOWLEDGE_BASE)
        if query.include_external_data and self.connectors:
            sources_to_query.append(ContextSource.CONNECTORS)
        if query.include_api_data and self.api_registry:
            sources_to_query.append(ContextSource.API_LISTENERS)
        if query.include_summaries:
            sources_to_query.append(ContextSource.SUMMARIES)
        if query.include_preferences:
            sources_to_query.append(ContextSource.PREFERENCES)

        result.sources_queried = sources_to_query

        # Query all sources in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for source in sources_to_query:
                if source == ContextSource.MESSAGES:
                    futures[source] = executor.submit(self._fetch_messages, query)
                elif source == ContextSource.KNOWLEDGE_BASE:
                    futures[source] = executor.submit(self._fetch_knowledge, query)
                elif source == ContextSource.CONNECTORS:
                    futures[source] = executor.submit(self._fetch_external, query)
                elif source == ContextSource.API_LISTENERS:
                    futures[source] = executor.submit(self._fetch_api_data, query)
                elif source == ContextSource.SUMMARIES:
                    futures[source] = executor.submit(self._fetch_summaries, query)
                elif source == ContextSource.PREFERENCES:
                    futures[source] = executor.submit(self._fetch_preferences, query)

            # Collect results
            for source, future in futures.items():
                try:
                    data = future.result(timeout=self.config.fetch_timeout)
                    self._add_source_data(result, source, data)

                    if data:
                        result.sources_with_data.append(source)

                except Exception as e:
                    error_msg = f"Failed to fetch {source.value}: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

        # Aggregate topics and categories
        result.aggregated_topics = self._aggregate_topics(result)
        result.aggregated_categories = self._aggregate_categories(result)

        # Calculate totals
        result.total_items = (
            len(result.messages)
            + len(result.knowledge_results)
            + len(result.external_data)
            + len(result.api_data)
            + len(result.summaries)
            + (1 if result.preferences else 0)
        )

        result.query_latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"ContextLake query complete: "
            f"items={result.total_items}, "
            f"sources={len(result.sources_with_data)}/{len(result.sources_queried)}, "
            f"latency={result.query_latency_ms:.0f}ms"
        )

        return result

    def _fetch_messages(self, query: ContextQuery) -> list[Message]:
        """Fetch messages from PostgreSQL."""
        start_date, end_date = query.get_time_range()

        # Build filter parameters
        filters = {
            "user_id": query.user_id,
            "limit": query.max_messages,
        }

        if query.thread_id:
            filters["thread_id"] = query.thread_id

        if query.topics:
            filters["topics"] = query.topics

        if query.categories:
            filters["categories"] = query.categories

        if query.intent:
            filters["intent"] = query.intent

        if query.min_confidence:
            filters["min_confidence"] = query.min_confidence
        elif not self.config.include_low_confidence:
            filters["min_confidence"] = self.config.min_confidence_threshold

        if start_date:
            filters["start_date"] = start_date

        if end_date:
            filters["end_date"] = end_date

        # Query database
        try:
            messages = self.db.search_messages(**filters)
            return messages
        except Exception as e:
            logger.warning(f"Message fetch failed: {e}")
            return []

    def _fetch_knowledge(self, query: ContextQuery) -> list[dict[str, Any]]:
        """Fetch from knowledge base (VectorDB)."""
        if not self.knowledge_base:
            return []

        try:
            # Build metadata filters
            metadata_filter = {}
            if query.topics:
                metadata_filter["topics"] = query.topics
            if query.categories:
                metadata_filter["categories"] = query.categories

            results = self.knowledge_base.search(
                query=query.query,
                limit=query.max_knowledge_results,
                metadata_filter=metadata_filter if metadata_filter else None,
                min_similarity=self.config.similarity_threshold,
            )

            return [r.to_dict() for r in results]

        except Exception as e:
            logger.warning(f"Knowledge base fetch failed: {e}")
            return []

    def _fetch_external(self, query: ContextQuery) -> list[dict[str, Any]]:
        """Fetch from data connectors."""
        if not self.connectors:
            return []

        results = []

        try:
            # Get relevant connectors based on topics
            relevant_connectors = self.connectors.get_connectors_for_topics(
                query.topics or []
            )

            for connector in relevant_connectors:
                try:
                    data = connector.fetch(
                        user_id=query.user_id,
                        context=query.entities,
                    )
                    if data:
                        results.extend(data[:query.max_external_items])
                except Exception as e:
                    logger.warning(f"Connector {connector.name} failed: {e}")

        except Exception as e:
            logger.warning(f"External data fetch failed: {e}")

        return results

    def _fetch_api_data(self, query: ContextQuery) -> list[dict[str, Any]]:
        """Fetch from API listeners."""
        if not self.api_registry:
            return []

        results = []

        try:
            # Get relevant listeners based on topics
            relevant_listeners = self.api_registry.get_listeners_for_topics(
                query.topics or []
            )

            for listener in relevant_listeners:
                try:
                    data = listener.get_cached_data(query.user_id)
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.warning(f"API listener {listener.name} failed: {e}")

        except Exception as e:
            logger.warning(f"API data fetch failed: {e}")

        return results

    def _fetch_summaries(self, query: ContextQuery) -> list[ThreadSummary]:
        """Fetch thread summaries."""
        try:
            summaries = self.db.get_thread_summaries(
                user_id=query.user_id,
                topics=query.topics,
                limit=5,
            )
            return summaries
        except Exception as e:
            logger.warning(f"Summaries fetch failed: {e}")
            return []

    def _fetch_preferences(self, query: ContextQuery) -> UserPreferences | None:
        """Fetch user preferences."""
        try:
            return self.db.get_user_preferences(query.user_id)
        except Exception as e:
            logger.warning(f"Preferences fetch failed: {e}")
            return None

    def _add_source_data(
        self, result: ContextResult, source: ContextSource, data: Any
    ) -> None:
        """Add fetched data to result."""
        if source == ContextSource.MESSAGES:
            result.messages = data or []
        elif source == ContextSource.KNOWLEDGE_BASE:
            result.knowledge_results = data or []
        elif source == ContextSource.CONNECTORS:
            result.external_data = data or []
        elif source == ContextSource.API_LISTENERS:
            result.api_data = data or []
        elif source == ContextSource.SUMMARIES:
            result.summaries = data or []
        elif source == ContextSource.PREFERENCES:
            result.preferences = data

    def _aggregate_topics(self, result: ContextResult) -> list[str]:
        """Aggregate topics from all sources."""
        topics = set()

        for msg in result.messages:
            if msg.metadata and msg.metadata.topics:
                topics.update(msg.metadata.topics)

        for summary in result.summaries:
            if summary.topics:
                topics.update(summary.topics)

        for kb_result in result.knowledge_results:
            if kb_result.get("topics"):
                topics.update(kb_result["topics"])

        return list(topics)[:20]  # Limit to top 20

    def _aggregate_categories(self, result: ContextResult) -> list[str]:
        """Aggregate categories from all sources."""
        categories = set()

        for msg in result.messages:
            if msg.metadata and msg.metadata.categories:
                categories.update(msg.metadata.categories)

        for summary in result.summaries:
            if summary.categories:
                categories.update(summary.categories)

        return list(categories)[:10]

    def get_condensed_context(
        self,
        result: ContextResult,
        query: str,
        max_length: int | None = None,
    ) -> str:
        """Generate condensed context string for the main agent.

        Args:
            result: ContextResult from query()
            query: Original user query
            max_length: Maximum context length in characters

        Returns:
            Condensed context string
        """
        max_length = max_length or self.config.max_context_tokens * 4  # ~4 chars per token

        parts = []

        # User preferences (high priority, always include)
        if result.preferences:
            pref_str = result.preferences.to_context_string()
            if pref_str:
                parts.append(f"[User Preferences]\n{pref_str}")

        # External data (high priority for specific queries)
        if result.external_data:
            external_strs = []
            for item in result.external_data[:5]:
                if hasattr(item, "to_context_string"):
                    external_strs.append(item.to_context_string())
                else:
                    external_strs.append(str(item)[:200])
            if external_strs:
                parts.append(f"[External Data]\n" + "\n".join(external_strs))

        # API data
        if result.api_data:
            api_strs = [str(item)[:200] for item in result.api_data[:3]]
            if api_strs:
                parts.append(f"[Real-time Data]\n" + "\n".join(api_strs))

        # Knowledge base results
        if result.knowledge_results:
            kb_strs = []
            for r in result.knowledge_results[:3]:
                content = r.get("content", r.get("text", ""))[:300]
                kb_strs.append(content)
            if kb_strs:
                parts.append(f"[Knowledge Base]\n" + "\n".join(kb_strs))

        # Summaries
        if result.summaries:
            summary_strs = [s.to_context_string()[:200] for s in result.summaries[:3]]
            if summary_strs:
                parts.append(f"[Historical Summaries]\n" + "\n".join(summary_strs))

        # Recent messages (lower priority, fill remaining space)
        if result.messages:
            msg_strs = []
            for msg in result.messages[:10]:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                text = msg.raw_text[:200] if msg.raw_text else ""
                msg_strs.append(f"[{role}]: {text}")
            if msg_strs:
                parts.append(f"[Recent Messages]\n" + "\n".join(msg_strs))

        # Join and truncate
        context = "\n\n".join(parts)

        if len(context) > max_length:
            context = context[: max_length - 100] + "\n\n[Context truncated...]"

        return context

    def get_status(self) -> dict[str, Any]:
        """Get Context Lake status."""
        return {
            "sources_available": {
                "messages": True,
                "knowledge_base": self.knowledge_base is not None,
                "connectors": self.connectors is not None,
                "api_listeners": self.api_registry is not None,
            },
            "config": {
                "max_workers": self.config.max_workers,
                "fetch_timeout": self.config.fetch_timeout,
                "min_confidence": self.config.min_confidence_threshold,
            },
        }
