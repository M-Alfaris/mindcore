"""
Base classes for external connectors.

Connectors provide READ-ONLY access to external business systems.
They map conversation topics to external data sources.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectorResult:
    """
    Result from an external connector lookup.

    Contains the fetched data, source information, and cache metadata.
    """
    data: Dict[str, Any]  # The fetched data
    source: str  # Connector name
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cache_ttl: int = 300  # How long to cache (seconds)
    error: Optional[str] = None  # Error message if failed

    @property
    def success(self) -> bool:
        """Check if lookup was successful."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "cache_ttl": self.cache_ttl,
            "error": self.error,
            "success": self.success
        }

    def to_context_string(self) -> str:
        """Format result for inclusion in AI context."""
        if self.error:
            return f"[{self.source}] Error: {self.error}"

        if not self.data:
            return f"[{self.source}] No data found."

        parts = [f"[{self.source}] Data:"]
        for key, value in self.data.items():
            if isinstance(value, list):
                if len(value) > 0:
                    parts.append(f"  {key}: {len(value)} items")
                    # Show first few items
                    for item in value[:3]:
                        if isinstance(item, dict):
                            item_str = ", ".join(f"{k}: {v}" for k, v in list(item.items())[:4])
                            parts.append(f"    - {item_str}")
                        else:
                            parts.append(f"    - {item}")
                    if len(value) > 3:
                        parts.append(f"    ... and {len(value) - 3} more")
            elif isinstance(value, dict):
                parts.append(f"  {key}:")
                for k, v in list(value.items())[:5]:
                    parts.append(f"    {k}: {v}")
            else:
                parts.append(f"  {key}: {value}")

        return "\n".join(parts)


class BaseConnector(ABC):
    """
    Abstract base class for external system connectors.

    Connectors provide READ-ONLY access to external systems.
    They map topics to external data sources and extract relevant
    entities from conversation text.

    IMPORTANT: Connectors must NEVER write to external systems.
    All operations must be read-only for security.

    Implementing a Custom Connector:
    --------------------------------
    1. Subclass BaseConnector
    2. Set `topics` list for topics this connector handles
    3. Set `name` for identification
    4. Implement `lookup()` for data fetching
    5. Implement `extract_entities()` for entity extraction
    6. Optionally override `can_handle()` for custom topic matching

    Example:
        >>> class MyConnector(BaseConnector):
        ...     topics = ["my_topic", "related_topic"]
        ...     name = "my_connector"
        ...
        ...     async def lookup(self, user_id, context):
        ...         # Fetch data from your system
        ...         data = await fetch_from_my_system(user_id)
        ...         return ConnectorResult(data=data, source=self.name)
        ...
        ...     def extract_entities(self, text):
        ...         # Extract relevant IDs, dates, etc.
        ...         return {"my_id": extract_id(text)}
    """

    # Topic(s) this connector handles
    topics: List[str] = []

    # Human-readable name
    name: str = "base_connector"

    # Cache TTL in seconds (default 5 minutes)
    cache_ttl: int = 300

    # Whether this connector is enabled
    enabled: bool = True

    @abstractmethod
    async def lookup(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> ConnectorResult:
        """
        Fetch data from external system.

        This method must be READ-ONLY. Never modify external data.

        Args:
            user_id: The user making the request (for filtering data)
            context: Extracted context containing:
                - entities: Extracted entities (IDs, dates, etc.)
                - query: Original user query
                - topics: Matched topics

        Returns:
            ConnectorResult with fetched data or error

        Example:
            >>> async def lookup(self, user_id, context):
            ...     order_id = context.get("order_id")
            ...     orders = await self.db.fetch_orders(user_id, order_id)
            ...     return ConnectorResult(
            ...         data={"orders": orders},
            ...         source=self.name
            ...     )
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract relevant entities from text for lookup.

        Identifies IDs, dates, amounts, and other relevant
        information that can be used to query external systems.

        Args:
            text: User message or query text

        Returns:
            Dictionary of extracted entities

        Example:
            >>> def extract_entities(self, text):
            ...     entities = {}
            ...     # Extract order IDs like #12345 or ORD-12345
            ...     order_match = re.search(r'#?(\\d{4,})|ORD-(\\d+)', text)
            ...     if order_match:
            ...         entities["order_id"] = order_match.group(1) or order_match.group(2)
            ...     return entities
        """
        pass

    def can_handle(self, topics: List[str]) -> bool:
        """
        Check if this connector handles any of the given topics.

        Override for custom matching logic (e.g., regex, synonyms).

        Args:
            topics: List of topics from conversation

        Returns:
            True if connector can handle any topic
        """
        if not self.enabled:
            return False
        return bool(set(self.topics) & set(topics))

    def _extract_dates(self, text: str) -> Dict[str, Any]:
        """
        Helper to extract date-related information from text.

        Args:
            text: Text to extract dates from

        Returns:
            Dictionary with date-related entities
        """
        entities = {}

        # ISO dates (2024-03-15)
        iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        if iso_dates:
            entities["dates"] = iso_dates

        # Relative dates
        text_lower = text.lower()
        if "today" in text_lower:
            entities["relative_date"] = "today"
        elif "yesterday" in text_lower:
            entities["relative_date"] = "yesterday"
        elif "last week" in text_lower:
            entities["relative_date"] = "last_week"
        elif "last month" in text_lower:
            entities["relative_date"] = "last_month"
        elif "this month" in text_lower:
            entities["relative_date"] = "this_month"

        return entities

    def _extract_amounts(self, text: str) -> Dict[str, Any]:
        """
        Helper to extract monetary amounts from text.

        Args:
            text: Text to extract amounts from

        Returns:
            Dictionary with amount-related entities
        """
        entities = {}

        # Currency amounts ($100, $99.99, 100 USD)
        amount_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*(?:USD|EUR|GBP)',
        ]

        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["amounts"] = [float(m) for m in matches]
                break

        return entities

    async def health_check(self) -> bool:
        """
        Check if the connector is healthy and can connect.

        Override to implement actual health check logic.

        Returns:
            True if healthy, False otherwise
        """
        return self.enabled

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, topics={self.topics})>"
