"""Support category data sources.

Sources for customer support-related data aggregation.
"""

from mindcore.svl import TriggerCondition
from mindcore.svl.registry import source


@source(
    term="support",
    term_type="category",
    description="Aggregate support context for a user",
    trigger=TriggerCondition.ON_QUERY,
    cache_ttl=30,
    priority=20,
)
async def get_support_context(context: dict) -> dict:
    """Aggregate support-related context for a user.

    Combines ticket history, account status, and recent interactions
    to provide comprehensive context for support conversations.

    Args:
        context: Contains 'user_id'

    Returns:
        Aggregated support context
    """
    user_id = context.get("user_id")
    if not user_id:
        return {}

    # Example: Replace with actual data fetching
    # In production, this would fetch from multiple sources:
    #
    # tickets = await ticket_service.get_recent(user_id, limit=5)
    # account = await account_service.get_status(user_id)
    # interactions = await crm.get_recent_interactions(user_id)
    #
    # return {
    #     "tickets": tickets,
    #     "account": account,
    #     "interactions": interactions,
    #     "sentiment_score": calculate_sentiment(interactions),
    # }

    return {
        "user_id": user_id,
        "open_tickets": 2,
        "recent_tickets": [
            {"id": "TKT-001", "subject": "Billing question", "status": "open"},
            {"id": "TKT-002", "subject": "Feature request", "status": "resolved"},
        ],
        "account_status": "active",
        "subscription_tier": "premium",
        "customer_since": "2023-01-15",
        "lifetime_value": 1250.00,
        "sentiment_score": 0.7,
        "preferred_channel": "chat",
    }


@source(
    term="ticket_details",
    term_type="category",
    trigger=TriggerCondition.ON_DEMAND,
    cache_ttl=10,
)
async def get_ticket_details(context: dict) -> dict | None:
    """Fetch detailed ticket information.

    Args:
        context: Contains 'ticket_id'

    Returns:
        Ticket details or None
    """
    ticket_id = context.get("ticket_id")
    if not ticket_id:
        return None

    # Example: Replace with actual implementation
    return {
        "ticket_id": ticket_id,
        "subject": "Billing question",
        "status": "open",
        "priority": "normal",
        "created_at": "2024-01-10T14:30:00Z",
        "messages": [
            {
                "from": "customer",
                "text": "I have a question about my last invoice.",
                "timestamp": "2024-01-10T14:30:00Z",
            },
            {
                "from": "agent",
                "text": "I'd be happy to help. Can you provide your invoice number?",
                "timestamp": "2024-01-10T14:35:00Z",
            },
        ],
        "tags": ["billing", "invoice"],
    }
