"""
AI agents for Mindcore framework.
"""

from .base_agent import BaseAgent, AgentInitializationError, APICallError
from .enrichment_agent import EnrichmentAgent
from .context_assembler_agent import ContextAssemblerAgent
from .retrieval_query_agent import RetrievalQueryAgent, QueryIntent
from .smart_context_agent import SmartContextAgent, ContextTools
from .summarization_agent import SummarizationAgent
from .trivial_detector import (
    TrivialMessageDetector,
    TrivialCategory,
    TrivialMatch,
    get_trivial_detector,
    reset_trivial_detector,
)

__all__ = [
    "BaseAgent",
    "EnrichmentAgent",
    "ContextAssemblerAgent",
    "RetrievalQueryAgent",
    "QueryIntent",
    "SmartContextAgent",
    "ContextTools",
    "SummarizationAgent",
    "AgentInitializationError",
    "APICallError",
    # Trivial message detection
    "TrivialMessageDetector",
    "TrivialCategory",
    "TrivialMatch",
    "get_trivial_detector",
    "reset_trivial_detector",
]
