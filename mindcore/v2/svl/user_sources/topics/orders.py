"""Order-related data sources.

This module demonstrates how to create data sources for order-related topics
using the @source decorator. Both sync and async patterns are shown.

Example usage in your agent:
    # The sources are automatically discovered and registered
    results = svl.fetch_for_topics(["orders", "order_details"], context={
        "user_id": "user_123",
        "order_id": "order_456"
    })
"""

from mindcore.v2.svl import TriggerCondition
from mindcore.v2.svl.registry import source


@source(
    term="orders",
    term_type="topic",
    description="Fetch user's recent orders from database",
    trigger=TriggerCondition.ON_QUERY,
    cache_ttl=60,
    priority=10,
    tags=["ecommerce", "database"],
)
async def get_user_orders(context: dict) -> list[dict]:
    """Fetch recent orders for a user.

    Args:
        context: Must contain 'user_id'

    Returns:
        List of order dictionaries
    """
    user_id = context.get("user_id")
    if not user_id:
        return []

    # Example: Replace with your actual database query
    # This is a placeholder that returns mock data
    #
    # Real implementation might look like:
    # async with db.connection() as conn:
    #     return await conn.fetch(
    #         "SELECT * FROM orders WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10",
    #         user_id
    #     )

    return [
        {
            "order_id": f"ord_{user_id}_001",
            "user_id": user_id,
            "status": "delivered",
            "total": 99.99,
            "items_count": 3,
        },
        {
            "order_id": f"ord_{user_id}_002",
            "user_id": user_id,
            "status": "processing",
            "total": 149.50,
            "items_count": 2,
        },
    ]


@source(
    term="order_details",
    term_type="topic",
    description="Fetch detailed order information including items and shipping",
    trigger=TriggerCondition.ON_DEMAND,
    cache_ttl=30,
    priority=5,
)
async def get_order_details(context: dict) -> dict | None:
    """Fetch detailed information for a specific order.

    Args:
        context: Must contain 'order_id', optionally 'user_id' for verification

    Returns:
        Order details dictionary or None if not found
    """
    order_id = context.get("order_id")
    if not order_id:
        return None

    # Example: Replace with your actual implementation
    # Real implementation might aggregate data from multiple sources:
    #
    # order = await db.fetch_order(order_id)
    # items = await db.fetch_order_items(order_id)
    # shipping = await shipping_api.get_tracking(order["tracking_id"])
    #
    # return {
    #     "order": order,
    #     "items": items,
    #     "shipping": shipping,
    # }

    return {
        "order_id": order_id,
        "status": "processing",
        "created_at": "2024-01-15T10:30:00Z",
        "items": [
            {"name": "Widget A", "quantity": 2, "price": 29.99},
            {"name": "Widget B", "quantity": 1, "price": 49.99},
        ],
        "shipping": {
            "carrier": "UPS",
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "2024-01-20",
        },
        "total": 109.97,
    }


@source(
    term="order_history",
    term_type="topic",
    description="Fetch complete order history with pagination",
    trigger=TriggerCondition.ON_DEMAND,
    cache_ttl=120,
)
def get_order_history(context: dict) -> dict:
    """Fetch paginated order history.

    This is a sync function example - useful when calling sync-only APIs.

    Args:
        context: Contains 'user_id', optional 'page' and 'page_size'

    Returns:
        Paginated order history
    """
    user_id = context.get("user_id")
    page = context.get("page", 1)
    page_size = context.get("page_size", 20)

    # Example: Replace with actual database query
    # offset = (page - 1) * page_size
    # orders = db.query(
    #     "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
    #     user_id, page_size, offset
    # )
    # total = db.query_one("SELECT COUNT(*) FROM orders WHERE user_id = ?", user_id)

    return {
        "user_id": user_id,
        "page": page,
        "page_size": page_size,
        "total": 42,
        "orders": [
            {"order_id": f"ord_{i}", "status": "completed", "total": 50 + i * 10}
            for i in range((page - 1) * page_size, min(page * page_size, 42))
        ],
    }
