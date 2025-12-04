"""Billing Connector - Read-only access to billing/payment systems.

Provides context about invoices, payments, subscriptions, and charges.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from mindcore.utils.logger import get_logger

from .base import BaseConnector, ConnectorResult


logger = get_logger(__name__)


class BillingConnector(BaseConnector):
    """Read-only connector for billing/payment systems.

    Maps topics like "billing", "payment", "invoice", "subscription"
    to billing data. Extracts invoice numbers, amounts, and dates.

    IMPORTANT: This connector is READ-ONLY. It never modifies billing data.
    Refunds, cancellations, and other write operations must go through
    proper business channels.

    Topics are automatically registered with VocabularyManager for
    consistent vocabulary across the system.

    Configuration:
    -------------
    The connector can be configured to work with different backends:
    - Stripe API (via stripe_api_key)
    - Custom REST API (via api_url)
    - Database (via db_url)
    - Custom callback function (via lookup_fn)

    Example with Stripe:
        >>> connector = BillingConnector(
        ...     stripe_api_key="sk_live_...",
        ...     stripe_customer_id_field="stripe_id"  # field in your user model
        ... )

    Example with Custom API:
        >>> connector = BillingConnector(
        ...     api_url="https://billing.example.com/api",
        ...     api_key="your-api-key"
        ... )
    """

    topics = ["billing", "payment", "invoice", "subscription", "charge", "refund", "receipt"]
    name = "billing"
    cache_ttl = 300  # 5 minutes
    _topics_registered = False

    def __init__(
        self,
        # Stripe config
        stripe_api_key: str | None = None,
        # API config
        api_url: str | None = None,
        api_key: str | None = None,
        # Database config
        db_url: str | None = None,
        user_id_column: str = "user_id",
        # Custom function
        lookup_fn: callable | None = None,
        # General config
        max_results: int = 10,
        enabled: bool = True,
    ):
        """Initialize billing connector.

        Args:
            stripe_api_key: Stripe API key (read-only operations only)
            api_url: Custom billing API URL
            api_key: API key for custom API
            db_url: Database connection URL
            user_id_column: Column name for user ID mapping
            lookup_fn: Custom async lookup function
            max_results: Maximum results to return
            enabled: Whether connector is enabled
        """
        self.stripe_api_key = stripe_api_key
        self.api_url = api_url
        self.api_key = api_key
        self.db_url = db_url
        self.user_id_column = user_id_column
        self.lookup_fn = lookup_fn
        self.max_results = max_results
        self.enabled = enabled

        # Register topics with VocabularyManager
        super().__init__()

    async def lookup(self, user_id: str, context: dict[str, Any]) -> ConnectorResult:
        """Fetch billing information for a user.

        Args:
            user_id: User identifier
            context: Contains extracted entities like invoice_id, amounts

        Returns:
            ConnectorResult with billing data
        """
        try:
            # Use custom function if provided
            if self.lookup_fn:
                data = await self.lookup_fn(user_id, context)
                return ConnectorResult(data=data, source=self.name, cache_ttl=self.cache_ttl)

            # Use Stripe if configured
            if self.stripe_api_key:
                return await self._lookup_via_stripe(user_id, context)

            # Use custom API if configured
            if self.api_url:
                return await self._lookup_via_api(user_id, context)

            # Use database if configured
            if self.db_url:
                return await self._lookup_via_db(user_id, context)

            # No backend configured - return mock data
            return await self._lookup_mock(user_id, context)

        except Exception as e:
            logger.exception(f"Billing lookup failed for user {user_id}: {e}")
            return ConnectorResult(data={}, source=self.name, error=str(e))

    async def _lookup_via_stripe(self, user_id: str, context: dict[str, Any]) -> ConnectorResult:
        """Lookup billing via Stripe API."""
        try:
            import stripe

            stripe.api_key = self.stripe_api_key

            # Note: In production, you'd map user_id to Stripe customer ID
            # This assumes user_id IS the Stripe customer ID or you have a mapping
            customer_id = context.get("stripe_customer_id", user_id)

            data = {"invoices": [], "subscriptions": [], "charges": []}

            # Get recent invoices (read-only)
            invoices = stripe.Invoice.list(customer=customer_id, limit=self.max_results)
            data["invoices"] = [
                {
                    "invoice_id": inv.id,
                    "amount": inv.amount_due / 100,  # Convert from cents
                    "status": inv.status,
                    "created": datetime.fromtimestamp(inv.created).isoformat(),
                    "due_date": (
                        datetime.fromtimestamp(inv.due_date).isoformat() if inv.due_date else None
                    ),
                }
                for inv in invoices.data
            ]

            # Get active subscriptions (read-only)
            subscriptions = stripe.Subscription.list(customer=customer_id, limit=self.max_results)
            data["subscriptions"] = [
                {
                    "subscription_id": sub.id,
                    "status": sub.status,
                    "plan": sub.plan.nickname if sub.plan else "Unknown",
                    "amount": sub.plan.amount / 100 if sub.plan else 0,
                    "current_period_end": datetime.fromtimestamp(
                        sub.current_period_end
                    ).isoformat(),
                }
                for sub in subscriptions.data
            ]

            return ConnectorResult(data=data, source=self.name, cache_ttl=self.cache_ttl)

        except ImportError:
            return ConnectorResult(
                data={}, source=self.name, error="stripe not installed. Run: pip install stripe"
            )
        except Exception as e:
            return ConnectorResult(data={}, source=self.name, error=f"Stripe error: {e!s}")

    async def _lookup_via_api(self, user_id: str, context: dict[str, Any]) -> ConnectorResult:
        """Lookup billing via custom REST API."""
        try:
            import aiohttp

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            params = {"user_id": user_id, "limit": self.max_results}

            # Add filters from context
            if context.get("invoice_id"):
                params["invoice_id"] = context["invoice_id"]

            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    self.api_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    return ConnectorResult(data=data, source=self.name, cache_ttl=self.cache_ttl)
                return ConnectorResult(
                    data={},
                    source=self.name,
                    error=f"API returned status {response.status}",
                )

        except ImportError:
            return ConnectorResult(
                data={}, source=self.name, error="aiohttp not installed. Run: pip install aiohttp"
            )
        except Exception as e:
            return ConnectorResult(data={}, source=self.name, error=f"API error: {e!s}")

    async def _lookup_via_db(self, user_id: str, context: dict[str, Any]) -> ConnectorResult:
        """Lookup billing via database."""
        try:
            import asyncpg

            pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)

            # Query invoices
            query = f"""
                SELECT invoice_id, amount, status, created_at, due_date
                FROM invoices
                WHERE {self.user_id_column} = $1
                ORDER BY created_at DESC
                LIMIT {self.max_results}
            """

            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_id)

            invoices = [dict(row) for row in rows]
            await pool.close()

            return ConnectorResult(
                data={"invoices": invoices, "count": len(invoices)},
                source=self.name,
                cache_ttl=self.cache_ttl,
            )

        except ImportError:
            return ConnectorResult(
                data={}, source=self.name, error="asyncpg not installed. Run: pip install asyncpg"
            )
        except Exception as e:
            return ConnectorResult(data={}, source=self.name, error=f"Database error: {e!s}")

    async def _lookup_mock(self, user_id: str, context: dict[str, Any]) -> ConnectorResult:
        """Return mock data for demonstration."""
        logger.warning("BillingConnector using mock data - configure a backend for production")

        now = datetime.now(timezone.utc)

        mock_data = {
            "invoices": [
                {
                    "invoice_id": "INV-2024-001",
                    "amount": 99.00,
                    "status": "paid",
                    "created_at": (now - timedelta(days=30)).isoformat(),
                    "paid_at": (now - timedelta(days=28)).isoformat(),
                },
                {
                    "invoice_id": "INV-2024-002",
                    "amount": 99.00,
                    "status": "pending",
                    "created_at": now.isoformat(),
                    "due_date": (now + timedelta(days=30)).isoformat(),
                },
            ],
            "subscriptions": [
                {
                    "subscription_id": "SUB-001",
                    "plan": "Pro Monthly",
                    "status": "active",
                    "amount": 99.00,
                    "next_billing": (now + timedelta(days=15)).isoformat(),
                }
            ],
            "payment_methods": [
                {
                    "type": "card",
                    "last4": "4242",
                    "brand": "Visa",
                    "exp_month": 12,
                    "exp_year": 2025,
                }
            ],
            "note": "This is mock data. Configure a backend for production.",
        }

        # Filter by invoice_id if provided
        if context.get("invoice_id"):
            mock_data["invoices"] = [
                i for i in mock_data["invoices"] if i["invoice_id"] == context["invoice_id"]
            ]

        return ConnectorResult(
            data=mock_data,
            source=self.name,
            cache_ttl=60,  # Shorter TTL for mock data
        )

    def extract_entities(self, text: str) -> dict[str, Any]:
        """Extract billing-related entities from text.

        Extracts:
        - Invoice numbers (INV-123, invoice 123)
        - Amounts ($99.99, 99 USD)
        - Dates
        - Subscription mentions

        Args:
            text: User message text

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract invoice numbers
        invoice_patterns = [
            r"INV-?(\d+)",  # INV-123 or INV123
            r"invoice[:\s#]*(\d+)",  # invoice 123
            r"receipt[:\s#]*(\d+)",  # receipt 123
        ]

        for pattern in invoice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["invoice_id"] = f"INV-{matches[0]}"
                break

        # Extract amounts
        amount_entities = self._extract_amounts(text)
        entities.update(amount_entities)

        # Extract dates
        date_entities = self._extract_dates(text)
        entities.update(date_entities)

        # Check for subscription-related keywords
        text_lower = text.lower()
        if any(
            word in text_lower
            for word in ["subscription", "subscribe", "plan", "upgrade", "downgrade"]
        ):
            entities["query_type"] = "subscription"
        elif any(word in text_lower for word in ["refund", "cancel", "charge back"]):
            entities["query_type"] = "refund_inquiry"
        elif any(word in text_lower for word in ["invoice", "receipt", "bill"]):
            entities["query_type"] = "invoice"
        elif any(word in text_lower for word in ["payment", "pay", "charge"]):
            entities["query_type"] = "payment"

        return entities

    async def health_check(self) -> bool:
        """Check if connector can connect to backend."""
        if self.lookup_fn:
            return True

        if self.stripe_api_key:
            try:
                import stripe

                stripe.api_key = self.stripe_api_key
                # Simple API check
                stripe.Account.retrieve()
                return True
            except Exception:
                return False

        if self.api_url:
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.head(self.api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp,
                ):
                    return resp.status < 500
            except Exception:
                return False

        # Mock mode is always healthy
        return True
