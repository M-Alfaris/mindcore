"""API Listeners for real-time external service data.

Provides infrastructure to:
- Listen for webhook events from external services
- Poll APIs for updates
- Cache and serve real-time data to the Context Lake
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class APIListenerConfig:
    """Configuration for an API listener."""

    name: str
    topics: list[str]  # Topics this listener provides data for
    categories: list[str] = field(default_factory=list)

    # Polling settings
    poll_interval: int = 60  # seconds
    enabled: bool = True

    # Cache settings
    cache_ttl: int = 300  # seconds
    max_cache_entries: int = 1000

    # Authentication (optional)
    api_key: str | None = None
    api_secret: str | None = None
    base_url: str | None = None


@dataclass
class CachedData:
    """Cached data from an API listener."""

    data: dict[str, Any]
    fetched_at: datetime
    expires_at: datetime
    source: str

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at


class EventHandler:
    """Handler for webhook events.

    Processes incoming webhook events and stores relevant data.
    """

    def __init__(self, name: str, vocabulary: VocabularyManager | None = None):
        """Initialize event handler.

        Args:
            name: Handler name
            vocabulary: Vocabulary for topic/category validation
        """
        self.name = name
        self.vocabulary = vocabulary or get_vocabulary()
        self._handlers: dict[str, list[Callable]] = {}

    def register(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type.

        Args:
            event_type: Event type to handle
            handler: Callback function(event_data) -> dict | None
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    def process(self, event_type: str, event_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Process an event.

        Args:
            event_type: Type of event
            event_data: Event payload

        Returns:
            List of processed data items
        """
        handlers = self._handlers.get(event_type, [])
        results = []

        for handler in handlers:
            try:
                result = handler(event_data)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

        return results


class APIListener(ABC):
    """Base class for API listeners.

    Subclass to implement listeners for specific services
    (Stripe, Shopify, Salesforce, etc.)
    """

    def __init__(
        self,
        config: APIListenerConfig,
        vocabulary: VocabularyManager | None = None,
    ):
        """Initialize listener.

        Args:
            config: Listener configuration
            vocabulary: Vocabulary manager
        """
        self.config = config
        self.name = config.name
        self.vocabulary = vocabulary or get_vocabulary()

        # Cache
        self._cache: dict[str, CachedData] = {}
        self._cache_lock = threading.Lock()

        # Polling
        self._poll_thread: threading.Thread | None = None
        self._running = False

    @abstractmethod
    def fetch(self, user_id: str, **kwargs) -> dict[str, Any] | None:
        """Fetch data for a user.

        Args:
            user_id: User identifier
            **kwargs: Additional parameters

        Returns:
            Fetched data or None
        """

    @abstractmethod
    def process_webhook(self, event_data: dict[str, Any]) -> dict[str, Any] | None:
        """Process a webhook event.

        Args:
            event_data: Webhook payload

        Returns:
            Processed data or None
        """

    def get_cached_data(self, user_id: str) -> dict[str, Any] | None:
        """Get cached data for a user.

        Args:
            user_id: User identifier

        Returns:
            Cached data or None if expired/missing
        """
        cache_key = self._make_cache_key(user_id)

        with self._cache_lock:
            cached = self._cache.get(cache_key)

            if cached and not cached.is_expired:
                return cached.data

        # Fetch fresh data
        try:
            data = self.fetch(user_id)
            if data:
                self._set_cache(user_id, data)
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch data for {user_id}: {e}")
            return None

    def _set_cache(self, user_id: str, data: dict[str, Any]) -> None:
        """Set cache entry."""
        cache_key = self._make_cache_key(user_id)
        now = datetime.now(timezone.utc)

        with self._cache_lock:
            # Evict if at capacity
            if len(self._cache) >= self.config.max_cache_entries:
                self._evict_oldest()

            self._cache[cache_key] = CachedData(
                data=data,
                fetched_at=now,
                expires_at=now + timedelta(seconds=self.config.cache_ttl),
                source=self.name,
            )

    def _make_cache_key(self, user_id: str) -> str:
        """Create cache key."""
        return f"{self.name}:{user_id}"

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self._cache:
            return

        # Sort by fetch time and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].fetched_at,
        )
        to_remove = max(1, len(sorted_keys) // 10)

        for key in sorted_keys[:to_remove]:
            del self._cache[key]

    def start_polling(self) -> None:
        """Start background polling."""
        if self._running or not self.config.enabled:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"api_listener_{self.name}",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(f"Started polling for {self.name}")

    def stop_polling(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        logger.info(f"Stopped polling for {self.name}")

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                self._poll_all_cached()
            except Exception as e:
                logger.warning(f"Poll error for {self.name}: {e}")

            time.sleep(self.config.poll_interval)

    def _poll_all_cached(self) -> None:
        """Refresh all cached entries."""
        with self._cache_lock:
            user_ids = [k.split(":")[1] for k in self._cache.keys()]

        for user_id in user_ids:
            try:
                data = self.fetch(user_id)
                if data:
                    self._set_cache(user_id, data)
            except Exception as e:
                logger.debug(f"Failed to refresh {user_id}: {e}")


# Import timedelta for cache expiration
from datetime import timedelta


class WebhookListener(APIListener):
    """Webhook-based API listener.

    Receives and processes webhook events from external services.
    """

    def __init__(
        self,
        config: APIListenerConfig,
        event_handler: EventHandler | None = None,
        vocabulary: VocabularyManager | None = None,
    ):
        """Initialize webhook listener.

        Args:
            config: Listener configuration
            event_handler: Custom event handler
            vocabulary: Vocabulary manager
        """
        super().__init__(config, vocabulary)
        self.event_handler = event_handler or EventHandler(config.name, vocabulary)
        self._event_buffer: dict[str, list[dict]] = {}  # user_id -> events

    def fetch(self, user_id: str, **kwargs) -> dict[str, Any] | None:
        """Get buffered events for a user."""
        events = self._event_buffer.get(user_id, [])

        if not events:
            return None

        return {
            "source": self.name,
            "user_id": user_id,
            "events": events[-10:],  # Last 10 events
            "event_count": len(events),
        }

    def process_webhook(self, event_data: dict[str, Any]) -> dict[str, Any] | None:
        """Process incoming webhook event."""
        event_type = event_data.get("type", event_data.get("event_type", "unknown"))
        user_id = self._extract_user_id(event_data)

        if not user_id:
            logger.debug(f"No user_id in webhook event: {event_type}")
            return None

        # Process with handler
        results = self.event_handler.process(event_type, event_data)

        # Buffer event
        if user_id not in self._event_buffer:
            self._event_buffer[user_id] = []

        processed_event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": event_data,
            "processed": results,
        }

        self._event_buffer[user_id].append(processed_event)

        # Keep buffer bounded
        if len(self._event_buffer[user_id]) > 100:
            self._event_buffer[user_id] = self._event_buffer[user_id][-50:]

        # Update cache
        self._set_cache(user_id, self.fetch(user_id))

        logger.debug(f"Processed webhook: {event_type} for {user_id}")
        return processed_event

    def _extract_user_id(self, event_data: dict[str, Any]) -> str | None:
        """Extract user ID from event data.

        Override for service-specific extraction.
        """
        # Try common patterns
        for key in ["user_id", "customer_id", "customer", "user", "account_id"]:
            if key in event_data:
                value = event_data[key]
                if isinstance(value, dict):
                    return value.get("id")
                return str(value)

        # Check nested data
        data = event_data.get("data", {})
        if isinstance(data, dict):
            for key in ["user_id", "customer_id", "customer", "user"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, dict):
                        return value.get("id")
                    return str(value)

        return None


class PollingAPIListener(APIListener):
    """Polling-based API listener.

    Periodically polls an API endpoint for updates.
    """

    def __init__(
        self,
        config: APIListenerConfig,
        fetch_fn: Callable[[str], dict[str, Any] | None],
        vocabulary: VocabularyManager | None = None,
    ):
        """Initialize polling listener.

        Args:
            config: Listener configuration
            fetch_fn: Function to fetch data for a user_id
            vocabulary: Vocabulary manager
        """
        super().__init__(config, vocabulary)
        self._fetch_fn = fetch_fn

    def fetch(self, user_id: str, **kwargs) -> dict[str, Any] | None:
        """Fetch data using the configured function."""
        try:
            return self._fetch_fn(user_id)
        except Exception as e:
            logger.warning(f"Fetch failed for {user_id}: {e}")
            return None

    def process_webhook(self, event_data: dict[str, Any]) -> dict[str, Any] | None:
        """Polling listeners don't process webhooks."""
        return None


class APIListenerRegistry:
    """Registry of API listeners.

    Manages multiple listeners and routes queries to appropriate ones.
    """

    def __init__(self, vocabulary: VocabularyManager | None = None):
        """Initialize registry.

        Args:
            vocabulary: Vocabulary manager
        """
        self.vocabulary = vocabulary or get_vocabulary()
        self._listeners: dict[str, APIListener] = {}
        self._topic_index: dict[str, list[str]] = {}  # topic -> listener names

    def register(self, listener: APIListener) -> None:
        """Register a listener.

        Args:
            listener: Listener to register
        """
        self._listeners[listener.name] = listener

        # Index by topics
        for topic in listener.config.topics:
            if topic not in self._topic_index:
                self._topic_index[topic] = []
            self._topic_index[topic].append(listener.name)

        logger.info(
            f"Registered API listener: {listener.name} "
            f"(topics: {listener.config.topics})"
        )

    def unregister(self, name: str) -> None:
        """Unregister a listener by name."""
        if name in self._listeners:
            listener = self._listeners[name]

            # Remove from topic index
            for topic in listener.config.topics:
                if topic in self._topic_index:
                    self._topic_index[topic] = [
                        n for n in self._topic_index[topic] if n != name
                    ]

            # Stop polling
            listener.stop_polling()

            del self._listeners[name]
            logger.info(f"Unregistered API listener: {name}")

    def get(self, name: str) -> APIListener | None:
        """Get a listener by name."""
        return self._listeners.get(name)

    def get_listeners_for_topics(self, topics: list[str]) -> list[APIListener]:
        """Get listeners that provide data for given topics.

        Args:
            topics: Topics to match

        Returns:
            List of matching listeners
        """
        listener_names = set()

        for topic in topics:
            if topic in self._topic_index:
                listener_names.update(self._topic_index[topic])

        return [
            self._listeners[name]
            for name in listener_names
            if name in self._listeners
        ]

    def get_all_data(self, user_id: str) -> dict[str, dict[str, Any]]:
        """Get data from all listeners for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict of listener_name -> data
        """
        results = {}

        for name, listener in self._listeners.items():
            try:
                data = listener.get_cached_data(user_id)
                if data:
                    results[name] = data
            except Exception as e:
                logger.warning(f"Failed to get data from {name}: {e}")

        return results

    def process_webhook(
        self, listener_name: str, event_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Route webhook to specific listener.

        Args:
            listener_name: Target listener name
            event_data: Webhook payload

        Returns:
            Processed data or None
        """
        listener = self._listeners.get(listener_name)
        if not listener:
            logger.warning(f"Unknown listener: {listener_name}")
            return None

        return listener.process_webhook(event_data)

    def start_all_polling(self) -> None:
        """Start polling for all listeners."""
        for listener in self._listeners.values():
            listener.start_polling()

    def stop_all_polling(self) -> None:
        """Stop polling for all listeners."""
        for listener in self._listeners.values():
            listener.stop_polling()

    def get_status(self) -> dict[str, Any]:
        """Get registry status."""
        return {
            "listeners": list(self._listeners.keys()),
            "topic_index": dict(self._topic_index),
            "count": len(self._listeners),
        }
