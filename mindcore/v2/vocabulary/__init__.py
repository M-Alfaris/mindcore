"""Vocabulary module - DEPRECATED, use mindcore.v2.svl instead.

This module is provided for backwards compatibility only.
All functionality has been consolidated into the Shared Vocabulary Layer (SVL).

Migration guide:
    # Old way
    from mindcore.v2.vocabulary import VocabularySchema, DEFAULT_VOCABULARY

    # New way (recommended)
    from mindcore.v2.svl import SharedVocabularyLayer, DEFAULT_SVL

The VocabularySchema class is now an alias for SharedVocabularyLayer.
"""

import warnings

from mindcore.v2.svl import DEFAULT_SVL as DEFAULT_VOCABULARY

# Re-export from SVL for backwards compatibility
from mindcore.v2.svl import AccessLevel, FieldSchema, MemoryType, Migration, Sentiment
from mindcore.v2.svl import SharedVocabularyLayer as VocabularySchema


def __getattr__(name: str):
    """Emit deprecation warning for legacy imports."""
    if name in ("VocabularySchema", "DEFAULT_VOCABULARY"):
        warnings.warn(
            f"{name} is deprecated, use SharedVocabularyLayer/DEFAULT_SVL from mindcore.v2.svl instead",
            DeprecationWarning,
            stacklevel=2,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_VOCABULARY",
    "AccessLevel",
    "FieldSchema",
    "MemoryType",
    "Migration",
    "Sentiment",
    "VocabularySchema",
]
