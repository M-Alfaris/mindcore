"""
Framework integrations for seamless Mindcore integration with popular AI frameworks.

This module provides ready-to-use integrations for:
- LangChain 0.1+/1.x (via langchain-core)
- LlamaIndex 0.10+
- Custom AI systems

Usage:
    from mindcore import MindcoreClient
    from mindcore.integrations import LangChainIntegration

    client = MindcoreClient()
    integration = LangChainIntegration(client)
"""

from .langchain_adapter import LangChainAdapter as LangChainIntegration
from .llamaindex_adapter import LlamaIndexAdapter as LlamaIndexIntegration
from .base_adapter import BaseAdapter as BaseIntegration

__all__ = [
    "BaseIntegration",
    "LangChainIntegration",
    "LlamaIndexIntegration",
]
