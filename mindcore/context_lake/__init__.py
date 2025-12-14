"""Context Lake - Unified context aggregation for AI agents.

The Context Lake is the central hub that aggregates context from multiple sources:
- PostgreSQL (structured messages with metadata filtering)
- VectorDB (semantic knowledge base search)
- Data Connectors (external data with vocabulary mapping)
- API Listeners (real-time service data)

Usage:
    from mindcore.context_lake import ContextLake, ContextQuery

    lake = ContextLake(
        postgres_manager=db_manager,
        vector_store=chroma_store,
        connectors=connector_registry,
    )

    # Query the context lake
    context = lake.query(ContextQuery(
        user_id="user123",
        thread_id="thread456",
        query="What about my order status?",
        include_knowledge_base=True,
        include_external_data=True,
    ))

    # Get condensed context for the main agent
    condensed = lake.get_condensed_context(context)
"""

from .api_listener import (
    APIListener,
    APIListenerConfig,
    APIListenerRegistry,
    EventHandler,
    WebhookListener,
)
from .knowledge_base import (
    Document,
    KnowledgeBase,
    KnowledgeBaseConfig,
    SearchResult,
)
from .lake import (
    ContextLake,
    ContextLakeConfig,
    ContextQuery,
    ContextResult,
    ContextSource,
)


__all__ = [
    "APIListener",
    "APIListenerConfig",
    "APIListenerRegistry",
    "ContextLake",
    "ContextLakeConfig",
    "ContextQuery",
    "ContextResult",
    "ContextSource",
    "Document",
    "EventHandler",
    "KnowledgeBase",
    "KnowledgeBaseConfig",
    "SearchResult",
    "WebhookListener",
]
