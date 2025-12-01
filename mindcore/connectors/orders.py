"""
Orders Connector - Read-only access to order/purchase systems.

Provides context about user orders, deliveries, and purchase history.
"""
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from .base import BaseConnector, ConnectorResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OrdersConnector(BaseConnector):
    """
    Read-only connector for orders/purchase systems.

    Maps topics like "orders", "delivery", "shipping" to order data.
    Extracts order IDs, dates, and product names from conversation.

    IMPORTANT: This connector is READ-ONLY. It never modifies order data.

    Topics are automatically registered with VocabularyManager for
    consistent vocabulary across the system.

    Configuration:
    -------------
    The connector can be configured to work with different backends:
    - PostgreSQL/MySQL database (via db_url)
    - REST API (via api_url)
    - Custom callback function (via lookup_fn)

    Example with Database:
        >>> connector = OrdersConnector(
        ...     db_url="postgresql://readonly:pass@orders-db/orders",
        ...     user_id_column="customer_id",
        ...     table_name="orders"
        ... )

    Example with REST API:
        >>> connector = OrdersConnector(
        ...     api_url="https://api.example.com/orders",
        ...     api_key="your-api-key"
        ... )

    Example with Custom Function:
        >>> async def my_lookup(user_id, context):
        ...     # Your custom logic
        ...     return {"orders": [...]}
        ...
        >>> connector = OrdersConnector(lookup_fn=my_lookup)
    """

    topics = ["orders", "order", "purchase", "delivery", "shipping", "tracking"]
    name = "orders"
    cache_ttl = 300  # 5 minutes
    _topics_registered = False

    def __init__(
        self,
        # Database config
        db_url: Optional[str] = None,
        user_id_column: str = "user_id",
        table_name: str = "orders",
        # API config
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        # Custom function
        lookup_fn: Optional[callable] = None,
        # General config
        max_results: int = 10,
        enabled: bool = True
    ):
        """
        Initialize orders connector.

        Args:
            db_url: Database connection URL (read-only user recommended)
            user_id_column: Column name that maps to Mindcore user_id
            table_name: Orders table name
            api_url: REST API base URL for orders
            api_key: API key for authentication
            lookup_fn: Custom async function for lookups
            max_results: Maximum orders to return
            enabled: Whether connector is enabled
        """
        self.db_url = db_url
        self.user_id_column = user_id_column
        self.table_name = table_name
        self.api_url = api_url
        self.api_key = api_key
        self.lookup_fn = lookup_fn
        self.max_results = max_results
        self.enabled = enabled
        self._db_pool = None

        # Register topics with VocabularyManager
        super().__init__()

    async def lookup(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> ConnectorResult:
        """
        Fetch orders for a user.

        Args:
            user_id: User identifier
            context: Contains extracted entities like order_id, dates

        Returns:
            ConnectorResult with orders data
        """
        try:
            # Use custom function if provided
            if self.lookup_fn:
                data = await self.lookup_fn(user_id, context)
                return ConnectorResult(
                    data=data,
                    source=self.name,
                    cache_ttl=self.cache_ttl
                )

            # Use API if configured
            if self.api_url:
                return await self._lookup_via_api(user_id, context)

            # Use database if configured
            if self.db_url:
                return await self._lookup_via_db(user_id, context)

            # No backend configured - return mock data for demo
            return await self._lookup_mock(user_id, context)

        except Exception as e:
            logger.error(f"Orders lookup failed for user {user_id}: {e}")
            return ConnectorResult(
                data={},
                source=self.name,
                error=str(e)
            )

    async def _lookup_via_api(self, user_id: str, context: Dict[str, Any]) -> ConnectorResult:
        """Lookup orders via REST API."""
        try:
            import aiohttp

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            params = {"user_id": user_id, "limit": self.max_results}

            # Add filters from context
            if context.get("order_id"):
                params["order_id"] = context["order_id"]
            if context.get("date_from"):
                params["date_from"] = context["date_from"]

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ConnectorResult(
                            data={"orders": data.get("orders", []), "count": data.get("count", 0)},
                            source=self.name,
                            cache_ttl=self.cache_ttl
                        )
                    else:
                        return ConnectorResult(
                            data={},
                            source=self.name,
                            error=f"API returned status {response.status}"
                        )

        except ImportError:
            return ConnectorResult(
                data={},
                source=self.name,
                error="aiohttp not installed. Run: pip install aiohttp"
            )
        except Exception as e:
            return ConnectorResult(
                data={},
                source=self.name,
                error=f"API error: {str(e)}"
            )

    async def _lookup_via_db(self, user_id: str, context: Dict[str, Any]) -> ConnectorResult:
        """Lookup orders via database."""
        try:
            import asyncpg

            if self._db_pool is None:
                self._db_pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)

            query = f"""
                SELECT order_id, status, total, items, created_at, updated_at
                FROM {self.table_name}
                WHERE {self.user_id_column} = $1
            """
            params = [user_id]
            param_count = 1

            # Add filters
            if context.get("order_id"):
                param_count += 1
                query += f" AND order_id = ${param_count}"
                params.append(context["order_id"])

            if context.get("date_from"):
                param_count += 1
                query += f" AND created_at >= ${param_count}"
                params.append(context["date_from"])

            query += f" ORDER BY created_at DESC LIMIT {self.max_results}"

            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            orders = [dict(row) for row in rows]

            return ConnectorResult(
                data={"orders": orders, "count": len(orders)},
                source=self.name,
                cache_ttl=self.cache_ttl
            )

        except ImportError:
            return ConnectorResult(
                data={},
                source=self.name,
                error="asyncpg not installed. Run: pip install asyncpg"
            )
        except Exception as e:
            return ConnectorResult(
                data={},
                source=self.name,
                error=f"Database error: {str(e)}"
            )

    async def _lookup_mock(self, user_id: str, context: Dict[str, Any]) -> ConnectorResult:
        """
        Return mock data for demonstration purposes.

        In production, configure db_url, api_url, or lookup_fn.
        """
        logger.warning(f"OrdersConnector using mock data - configure a backend for production")

        order_id = context.get("order_id", "ORD-12345")
        now = datetime.now(timezone.utc)

        mock_orders = [
            {
                "order_id": order_id,
                "status": "delivered",
                "total": 99.99,
                "items": ["Product A", "Product B"],
                "created_at": (now - timedelta(days=5)).isoformat(),
                "tracking_number": "1Z999AA10123456784"
            },
            {
                "order_id": "ORD-12344",
                "status": "shipped",
                "total": 49.99,
                "items": ["Product C"],
                "created_at": (now - timedelta(days=2)).isoformat(),
                "tracking_number": "1Z999AA10123456785"
            }
        ]

        # Filter by order_id if provided
        if context.get("order_id"):
            mock_orders = [o for o in mock_orders if o["order_id"] == context["order_id"]]

        return ConnectorResult(
            data={
                "orders": mock_orders,
                "count": len(mock_orders),
                "note": "This is mock data. Configure a backend for production."
            },
            source=self.name,
            cache_ttl=60  # Shorter TTL for mock data
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract order-related entities from text.

        Extracts:
        - Order IDs (#12345, ORD-12345, order 12345)
        - Tracking numbers
        - Dates

        Args:
            text: User message text

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract order IDs
        order_patterns = [
            r'#(\d{4,})',  # #12345
            r'ORD-?(\d+)',  # ORD-12345 or ORD12345
            r'order[:\s#]*(\d{4,})',  # order 12345, order: 12345
            r'order\s+number[:\s]*(\d+)',  # order number 12345
        ]

        for pattern in order_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["order_id"] = matches[0]
                break

        # Extract tracking numbers (common carriers)
        tracking_patterns = [
            r'1Z[0-9A-Z]{16}',  # UPS
            r'\d{12,22}',  # FedEx, USPS (generic)
        ]

        for pattern in tracking_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["tracking_number"] = matches[0]
                break

        # Extract dates
        date_entities = self._extract_dates(text)
        entities.update(date_entities)

        return entities

    async def health_check(self) -> bool:
        """Check if connector can connect to backend."""
        if self.lookup_fn:
            return True

        if self.api_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.head(self.api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        return resp.status < 500
            except Exception:
                return False

        if self.db_url:
            try:
                import asyncpg
                conn = await asyncpg.connect(self.db_url, timeout=5)
                await conn.close()
                return True
            except Exception:
                return False

        # Mock mode is always healthy
        return True
