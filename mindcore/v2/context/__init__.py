"""Context module - Unified context assembly for MindCore.

The ContextGateway provides a single entry point for building LLM context,
orchestrating FLR (hot path), CLST (cold path), and SVL (data sources).

Key Features:
- build_context(): Full hierarchical retrieval
- build_context_with_decision(): LLM-driven context decision

Example:
    from mindcore.v2.context import ContextGateway
    from mindcore.v2.svl import ContextDecision, HistoricalContextNeeded

    gateway = ContextGateway(
        storage=postgres_storage,
        svl=shared_vocabulary_layer,
    )

    # Option 1: Full hierarchical retrieval
    context = gateway.build_context(
        query="What about my order?",
        user_id="user_123",
        session_id="session_abc",
        attention_hints=["orders"],
    )

    # Option 2: LLM-driven context decision
    decision = ContextDecision(
        historical_context_needed=HistoricalContextNeeded.FALSE,
        suggested_topics=["orders"],
    )
    context = gateway.build_context_with_decision(
        query="Hello!",
        context_decision=decision,
        user_id="user_123",
        session_id="session_abc",
    )

    # Get formatted context for LLM
    llm_context = context.to_llm_context()

    # After LLM response, record it with metadata
    response_meta = gateway.record_response(
        query_metadata=context.query_metadata,
        response_text="Your order #12345 shipped yesterday.",
        memories_to_store=llm_output.memories_to_store,
    )
"""

from .gateway import (
    ContextGateway,
    ContextResult,
    QueryMetadata,
    ResponseMetadata,
)

__all__ = [
    "ContextGateway",
    "ContextResult",
    "QueryMetadata",
    "ResponseMetadata",
]
