"""FLR - Fast Learning Recall.

A protocol for rapid retrieval, inference-time memory access, and short-term
contextual recall among AI agents.
"""

from .recall import (
    FLR,
    ContextWindow,
    Memory,
    RecallResult,
)


__all__ = [
    "FLR",
    "ContextWindow",
    "Memory",
    "RecallResult",
]
