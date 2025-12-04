"""Connector Registry for managing external data connectors.

The registry provides a central place to register, configure,
and invoke connectors based on conversation topics.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from mindcore.utils.logger import get_logger

from .base import BaseConnector, ConnectorResult


logger = get_logger(__name__)


class ConnectorRegistry:
    """Registry of external data connectors.

    Manages connector registration, topic mapping, and coordinated
    lookups across multiple connectors.

    Example:
        >>> from mindcore.connectors import ConnectorRegistry, OrdersConnector
        >>>
        >>> registry = ConnectorRegistry()
        >>>
        >>> # Register connectors
        >>> registry.register(OrdersConnector(db_url="..."))
        >>> registry.register(BillingConnector(api_key="..."))
        >>>
        >>> # Lookup data based on topics
        >>> results = await registry.lookup(
        ...     user_id="user123",
        ...     topics=["orders", "billing"],
        ...     context={"order_id": "12345"}
        ... )
        >>>
        >>> for result in results:
        ...     print(f"{result.source}: {result.data}")
    """

    def __init__(self, cache_enabled: bool = True, default_cache_ttl: int = 300):
        """Initialize connector registry.

        Args:
            cache_enabled: Whether to cache connector results
            default_cache_ttl: Default cache TTL in seconds
        """
        self._connectors: dict[str, BaseConnector] = {}
        self._topic_map: dict[str, BaseConnector] = {}
        self._cache: dict[str, tuple] = {}  # (result, expiry_time)
        self._cache_enabled = cache_enabled
        self._default_cache_ttl = default_cache_ttl

        logger.info("ConnectorRegistry initialized")

    def register(self, connector: BaseConnector) -> None:
        """Register a connector.

        Args:
            connector: Connector instance to register

        Raises:
            ValueError: If connector with same name already exists
        """
        if connector.name in self._connectors:
            raise ValueError(f"Connector '{connector.name}' already registered")

        self._connectors[connector.name] = connector

        for topic in connector.topics:
            if topic in self._topic_map:
                logger.warning(
                    f"Topic '{topic}' already mapped to {self._topic_map[topic].name}, "
                    f"overwriting with {connector.name}"
                )
            self._topic_map[topic] = connector

        logger.info(f"Registered connector: {connector.name} (topics: {connector.topics})")

    def unregister(self, name: str) -> bool:
        """Unregister a connector by name.

        Args:
            name: Connector name to unregister

        Returns:
            True if connector was unregistered, False if not found
        """
        if name not in self._connectors:
            return False

        connector = self._connectors.pop(name)

        # Remove from topic map
        for topic in connector.topics:
            if topic in self._topic_map and self._topic_map[topic].name == name:
                del self._topic_map[topic]

        logger.info(f"Unregistered connector: {name}")
        return True

    def get_connector(self, name: str) -> BaseConnector | None:
        """Get a connector by name.

        Args:
            name: Connector name

        Returns:
            Connector instance or None
        """
        return self._connectors.get(name)

    def get_connector_for_topic(self, topic: str) -> BaseConnector | None:
        """Get the connector that handles a specific topic.

        Args:
            topic: Topic name

        Returns:
            Connector instance or None
        """
        return self._topic_map.get(topic)

    def list_connectors(self) -> list[dict[str, Any]]:
        """List all registered connectors.

        Returns:
            List of connector info dicts
        """
        return [
            {"name": c.name, "topics": c.topics, "enabled": c.enabled, "cache_ttl": c.cache_ttl}
            for c in self._connectors.values()
        ]

    def list_topics(self) -> list[str]:
        """List all registered topics.

        Returns:
            List of topic names
        """
        return list(self._topic_map.keys())

    async def lookup(
        self, user_id: str, topics: list[str], context: dict[str, Any], timeout: float = 10.0
    ) -> list[ConnectorResult]:
        """Lookup data from all relevant connectors.

        Invokes connectors that match the given topics and returns
        their results. Connectors are invoked in parallel.

        Args:
            user_id: User identifier (for filtering data)
            topics: Topics mentioned in conversation
            context: Extracted entities and context
            timeout: Timeout in seconds for all lookups

        Returns:
            List of ConnectorResult from matching connectors
        """
        results = []
        seen_connectors = set()
        tasks = []

        for topic in topics:
            connector = self._topic_map.get(topic)
            if connector and connector.name not in seen_connectors and connector.enabled:
                seen_connectors.add(connector.name)

                # Check cache first
                cache_key = self._get_cache_key(connector.name, user_id, context)
                cached = self._get_cached(cache_key)
                if cached:
                    logger.debug(f"Cache hit for {connector.name}")
                    results.append(cached)
                    continue

                # Create lookup task
                tasks.append(self._lookup_with_connector(connector, user_id, context, cache_key))

        if tasks:
            try:
                # Run all lookups in parallel with timeout
                task_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
                )

                for result in task_results:
                    if isinstance(result, Exception):
                        logger.error(f"Connector lookup failed: {result}")
                        results.append(
                            ConnectorResult(data={}, source="unknown", error=str(result))
                        )
                    else:
                        results.append(result)

            except asyncio.TimeoutError:
                logger.exception(f"Connector lookups timed out after {timeout}s")
                results.append(
                    ConnectorResult(
                        data={}, source="timeout", error=f"Lookups timed out after {timeout}s"
                    )
                )

        return results

    async def _lookup_with_connector(
        self, connector: BaseConnector, user_id: str, context: dict[str, Any], cache_key: str
    ) -> ConnectorResult:
        """Perform lookup with a single connector.

        Args:
            connector: Connector to use
            user_id: User identifier
            context: Context for lookup
            cache_key: Cache key for result

        Returns:
            ConnectorResult
        """
        try:
            logger.debug(f"Looking up data from {connector.name} for user {user_id}")
            result = await connector.lookup(user_id, context)

            # Cache result
            if self._cache_enabled and result.success:
                ttl = result.cache_ttl or self._default_cache_ttl
                self._set_cached(cache_key, result, ttl)

            return result

        except Exception as e:
            logger.exception(f"Connector {connector.name} failed: {e}")
            return ConnectorResult(data={}, source=connector.name, error=str(e))

    def extract_entities_for_topics(self, text: str, topics: list[str]) -> dict[str, Any]:
        """Extract entities using connectors that handle given topics.

        Args:
            text: Text to extract entities from
            topics: Topics to find relevant connectors

        Returns:
            Combined entities from all relevant connectors
        """
        entities = {}
        seen_connectors = set()

        for topic in topics:
            connector = self._topic_map.get(topic)
            if connector and connector.name not in seen_connectors:
                seen_connectors.add(connector.name)
                try:
                    connector_entities = connector.extract_entities(text)
                    # Merge entities (connector-specific keys avoid conflicts)
                    for key, value in connector_entities.items():
                        if key in entities:
                            # Merge if both are lists
                            if isinstance(entities[key], list) and isinstance(value, list):
                                entities[key].extend(value)
                            elif isinstance(entities[key], list):
                                entities[key].append(value)
                            else:
                                entities[key] = [entities[key], value]
                        else:
                            entities[key] = value
                except Exception as e:
                    logger.warning(f"Entity extraction failed for {connector.name}: {e}")

        return entities

    def _get_cache_key(self, connector_name: str, user_id: str, context: dict[str, Any]) -> str:
        """Generate cache key for a lookup."""
        # Use sorted context keys for consistent hashing
        context_str = str(sorted(context.items()))
        return f"{connector_name}:{user_id}:{hash(context_str)}"

    def _get_cached(self, cache_key: str) -> ConnectorResult | None:
        """Get cached result if not expired."""
        if not self._cache_enabled:
            return None

        cached = self._cache.get(cache_key)
        if cached:
            result, expiry = cached
            if datetime.now(timezone.utc).timestamp() < expiry:
                return result
            # Expired, remove from cache
            del self._cache[cache_key]

        return None

    def _set_cached(self, cache_key: str, result: ConnectorResult, ttl: int) -> None:
        """Cache a result."""
        if not self._cache_enabled:
            return

        expiry = datetime.now(timezone.utc).timestamp() + ttl
        self._cache[cache_key] = (result, expiry)

    def clear_cache(self, connector_name: str | None = None) -> int:
        """Clear cached results.

        Args:
            connector_name: If provided, only clear cache for this connector

        Returns:
            Number of cache entries cleared
        """
        if connector_name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{connector_name}:")]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
        count = len(self._cache)
        self._cache.clear()
        return count

    async def health_check(self) -> dict[str, bool]:
        """Check health of all registered connectors.

        Returns:
            Dict mapping connector names to health status
        """
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception as e:
                logger.exception(f"Health check failed for {name}: {e}")
                results[name] = False
        return results

    def get_connectors_for_topics(self, topics: list[str]) -> list[BaseConnector]:
        """Get all connectors that handle any of the given topics.

        Used by Context Lake for parallel data fetching.

        Args:
            topics: List of topics to match

        Returns:
            List of matching connectors (deduplicated)
        """
        seen = set()
        connectors = []

        for topic in topics:
            connector = self._topic_map.get(topic)
            if connector and connector.name not in seen and connector.enabled:
                seen.add(connector.name)
                connectors.append(connector)

        return connectors

    def fetch(
        self,
        user_id: str,
        context: dict[str, Any],
        topics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous fetch for Context Lake integration.

        Wraps the async lookup in an event loop for sync contexts.

        Args:
            user_id: User identifier
            context: Context with extracted entities
            topics: Optional topics to filter connectors

        Returns:
            List of fetched data dictionaries
        """
        if topics is None:
            topics = list(self._topic_map.keys())

        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.lookup(user_id, topics, context)
                    )
                    results = future.result(timeout=15.0)
            else:
                results = loop.run_until_complete(
                    self.lookup(user_id, topics, context)
                )
        except RuntimeError:
            # No event loop, create one
            results = asyncio.run(self.lookup(user_id, topics, context))

        return [r.to_dict() for r in results if r.success]

    def get_vocabulary_topics(self) -> list[str]:
        """Get all topics registered with vocabulary from connectors.

        Returns:
            List of connector-registered topics
        """
        from mindcore.core.vocabulary import get_vocabulary
        vocabulary = get_vocabulary()
        return vocabulary.get_connector_topics()

    def get_status(self) -> dict[str, Any]:
        """Get registry status for monitoring.

        Returns:
            Status dictionary with connector info
        """
        return {
            "connectors": self.list_connectors(),
            "topics": self.list_topics(),
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
        }
