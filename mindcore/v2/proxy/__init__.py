"""Mindcore Proxy - Claude Code Integration.

A proxy layer that captures reasoning from Claude Code sessions
and injects relevant memories into new sessions using FLR.

Features:
- Session reasoning capture and extraction
- Automatic memory injection from FLR
- Context management (auto-summarize at limits)
- Team sync support via CLST

Example:
    # Start the proxy
    from mindcore.v2.proxy import MindcoreProxy

    proxy = MindcoreProxy(
        storage="sqlite:///mindcore.db",
        port=8080,
    )
    proxy.start()

    # Or via CLI:
    # mindcore proxy --port 8080
"""

from .session import Session, SessionManager, SessionState
from .extractor import ReasoningExtractor, ExtractedReasoning, ReasoningType
from .injector import MemoryInjector, InjectionStrategy
from .server import MindcoreProxy, ProxyConfig

__all__ = [
    # Session management
    "Session",
    "SessionManager",
    "SessionState",
    # Reasoning extraction
    "ReasoningExtractor",
    "ExtractedReasoning",
    "ReasoningType",
    # Memory injection
    "MemoryInjector",
    "InjectionStrategy",
    # Proxy server
    "MindcoreProxy",
    "ProxyConfig",
]
