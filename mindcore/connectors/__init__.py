"""External Connectors for Mindcore.

Provides read-only access to external business systems like orders,
billing, CRM, etc. Connectors map conversation topics to external
data sources for enriched context.

Example:
    >>> from mindcore.connectors import ConnectorRegistry, OrdersConnector
    >>>
    >>> # Create and configure connector
    >>> orders = OrdersConnector(
    ...     db_url="postgresql://readonly:pass@orders-db/orders",
    ...     user_id_column="customer_id"
    ... )
    >>>
    >>> # Register with Mindcore
    >>> registry = ConnectorRegistry()
    >>> registry.register(orders)
    >>>
    >>> # Lookup data for a user
    >>> results = await registry.lookup(
    ...     user_id="user123",
    ...     topics=["orders", "delivery"],
    ...     context={"order_id": "12345"}
    ... )
"""

from .base import BaseConnector, ConnectorResult
from .billing import BillingConnector
from .orders import OrdersConnector
from .registry import ConnectorRegistry


__all__ = [
    "BaseConnector",
    "BillingConnector",
    "ConnectorRegistry",
    "ConnectorResult",
    "OrdersConnector",
]
