"""SVL User Sources - Unified data source configuration.

This folder contains all data source definitions for SVL. Sources can be defined as:

1. Python modules with @source decorators (recommended for complex logic)
2. YAML/JSON configuration files (recommended for simple sources)

Folder Structure:
    user_sources/
    ├── __init__.py          # This file - auto-discovery entry point
    ├── config.yaml          # Simple sources (SQL, API, MCP)
    ├── topics/              # Sources organized by term type
    │   ├── orders.py        # Order-related sources
    │   └── products.py      # Product-related sources
    ├── categories/          # Category-based sources
    │   └── support.py       # Support category sources
    └── custom/              # Any custom organization
        └── integrations.py  # Third-party integrations

Usage:
    from mindcore.svl import SharedVocabularyLayer

    svl = SharedVocabularyLayer()

    # Auto-discover and register all sources from this folder
    count, errors = svl.discover_sources()

    # Or specify custom path
    count, errors = svl.discover_sources("/path/to/my/sources")
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mindcore.svl.registry import SourceDefinition


def discover_sources(
    sources_path: str | Path | None = None,
) -> tuple[list["SourceDefinition"], list[tuple[str, Exception]]]:
    """Discover all sources in this folder.

    Args:
        sources_path: Optional custom path (defaults to this folder)

    Returns:
        Tuple of (discovered sources, errors)
    """
    # Import here to avoid circular imports
    from mindcore.svl.registry import SourceDiscovery

    if sources_path is None:
        sources_path = Path(__file__).parent

    discovery = SourceDiscovery(sources_path)
    sources = discovery.discover()
    errors = discovery.get_errors()

    return sources, errors


# Export for convenience
__all__ = [
    "discover_sources",
]
