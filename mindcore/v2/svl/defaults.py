"""SVL Default Configuration - Convention over configuration for external sources.

This module provides sensible defaults and auto-configuration for mapping
topics to external data sources, minimizing boilerplate and configuration work.

Design Philosophy:
- Convention over configuration: "orders" topic → "orders" table by default
- Sensible defaults that work for 80% of use cases
- One-line database setup for common patterns
- Easy overrides when customization is needed

Example - Minimal Configuration:
    from mindcore.v2.svl import SVLPipeline, DefaultSourceConfig

    # One-line setup: auto-maps topics to same-named tables
    pipeline = create_pipeline(storage=storage, vocabulary=vocab)
    pipeline.auto_configure_database(
        connection_string="postgresql://localhost/mydb",
        topics=["orders", "products", "users"],  # Maps to orders, products, users tables
    )

    # That's it! Now queries for "orders" topic automatically fetch from orders table

Example - With Customization:
    # Override defaults for specific topics
    pipeline.auto_configure_database(
        connection_string="postgresql://localhost/mydb",
        topics=["orders", "products"],
        overrides={
            "orders": {
                "table": "customer_orders",  # Different table name
                "query_template": "SELECT * FROM customer_orders WHERE user_id = :user_id AND status = 'active'",
            },
        },
    )

Example - Auto-Discovery from Database Schema:
    # Automatically discover tables and map matching topics
    discovered = pipeline.auto_discover_tables(
        connection_string="postgresql://localhost/mydb",
        schema="public",
    )
    print(f"Auto-mapped {discovered} topics to tables")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sources import (
    DataSource,
    SourceRegistry,
    SourceType,
    TableSource,
    TriggerCondition,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Default Configuration
# =============================================================================


class NamingConvention(str, Enum):
    """Naming conventions for topic-to-table mapping."""

    EXACT = "exact"  # topic "orders" → table "orders"
    SNAKE_CASE = "snake_case"  # topic "userOrders" → table "user_orders"
    PLURAL = "plural"  # topic "order" → table "orders"
    SINGULAR = "singular"  # topic "orders" → table "order"


class ParamPattern(str, Enum):
    """Common parameter patterns for filtering."""

    USER_ID = "user_id"  # Filter by user_id
    SESSION_ID = "session_id"  # Filter by session_id
    AGENT_ID = "agent_id"  # Filter by agent_id
    CREATED_AT = "created_at"  # Filter by creation time


@dataclass
class DefaultSourceConfig:
    """Global default configuration for all auto-configured sources.

    Customize this to change defaults for your entire application.

    Example:
        config = DefaultSourceConfig(
            cache_ttl_seconds=120,  # 2 min cache instead of 1 min
            default_limit=50,  # Return 50 rows instead of 100
            naming_convention=NamingConvention.SNAKE_CASE,
        )
        pipeline.auto_configure_database(..., config=config)
    """

    # Cache settings
    cache_ttl_seconds: int = 60

    # Query settings
    default_limit: int = 100
    timeout_seconds: int = 30

    # Naming convention
    naming_convention: NamingConvention = NamingConvention.EXACT

    # Default parameters to include in queries (when available in context)
    default_params: list[ParamPattern] = field(
        default_factory=lambda: [ParamPattern.USER_ID]
    )

    # Trigger condition
    trigger: TriggerCondition = TriggerCondition.ON_QUERY

    # Whether sources are enabled by default
    enabled: bool = True

    # Query template pattern
    # Use {table}, {params}, {limit} as placeholders
    query_pattern: str = "SELECT * FROM {table} WHERE {params} LIMIT {limit}"

    # Fallback query when no params match context
    fallback_query_pattern: str = "SELECT * FROM {table} ORDER BY created_at DESC LIMIT {limit}"

    def get_table_name(self, topic: str) -> str:
        """Convert topic name to table name based on naming convention."""
        if self.naming_convention == NamingConvention.EXACT:
            return topic
        elif self.naming_convention == NamingConvention.SNAKE_CASE:
            return self._to_snake_case(topic)
        elif self.naming_convention == NamingConvention.PLURAL:
            return self._to_plural(topic)
        elif self.naming_convention == NamingConvention.SINGULAR:
            return self._to_singular(topic)
        return topic

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.lower())
        return "".join(result)

    def _to_plural(self, name: str) -> str:
        """Simple pluralization."""
        if name.endswith("s"):
            return name
        if name.endswith("y"):
            return name[:-1] + "ies"
        return name + "s"

    def _to_singular(self, name: str) -> str:
        """Simple singularization."""
        if name.endswith("ies"):
            return name[:-3] + "y"
        if name.endswith("s") and not name.endswith("ss"):
            return name[:-1]
        return name


# Default configuration instance
DEFAULT_CONFIG = DefaultSourceConfig()


# =============================================================================
# Smart Table Source
# =============================================================================


@dataclass
class SmartTableSource(TableSource):
    """A TableSource that automatically builds queries based on context.

    Unlike regular TableSource, this one:
    - Auto-generates queries if query_template is empty
    - Dynamically adjusts query based on available context params
    - Falls back gracefully when params aren't available

    Example:
        source = SmartTableSource(
            name="orders",
            connection_string="postgresql://...",
            table="orders",
        )
        # Automatically generates:
        # - "SELECT * FROM orders WHERE user_id = :user_id LIMIT 100" (if user_id in context)
        # - "SELECT * FROM orders ORDER BY created_at DESC LIMIT 100" (fallback)
    """

    # Parameters to look for in context
    context_params: list[str] = field(default_factory=lambda: ["user_id"])

    # Fallback behavior
    allow_fallback: bool = True  # Use fallback query when no params match
    fallback_order_by: str = "created_at DESC"

    def _build_dynamic_query(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build query dynamically based on available context.

        Returns:
            Tuple of (query_string, params_dict)
        """
        # Check which context params are available
        available_params = {}
        where_clauses = []

        for param in self.context_params:
            if param in context:
                available_params[param] = context[param]
                where_clauses.append(f"{param} = :{param}")

        # Also check param_mapping
        for context_key, sql_param in self.param_mapping.items():
            if context_key in context:
                available_params[sql_param] = context[context_key]
                where_clauses.append(f"{sql_param} = :{sql_param}")

        # Build query
        safe_table = self._validate_identifier(self.table)
        safe_limit = int(self.limit)
        if safe_limit < 1 or safe_limit > 10000:
            safe_limit = 100

        if where_clauses:
            where_clause = " AND ".join(where_clauses)
            query = f"SELECT * FROM {safe_table} WHERE {where_clause} LIMIT {safe_limit}"
        elif self.allow_fallback:
            query = f"SELECT * FROM {safe_table} ORDER BY {self.fallback_order_by} LIMIT {safe_limit}"
            available_params = {}
        else:
            raise ValueError(
                f"No matching context params for {self.table}. "
                f"Required one of: {self.context_params}"
            )

        return query, available_params

    def _do_fetch(self, context: dict[str, Any]) -> Any:
        """Execute query with dynamic params."""
        import time
        from .sources import FetchResult

        start = time.time()

        try:
            # Use explicit template if provided, otherwise build dynamically
            if self.query_template:
                return super()._do_fetch(context)

            query, params = self._build_dynamic_query(context)

            # Execute based on connection type
            if self.connection_string.startswith("postgresql"):
                data = self._fetch_postgres(query, params)
            elif self.connection_string.startswith("sqlite"):
                data = self._fetch_sqlite(query, params)
            else:
                raise ValueError(f"Unsupported database: {self.connection_string}")

            latency = (time.time() - start) * 1000
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=data,
                success=True,
                latency_ms=latency,
                metadata={"query": query, "params": params, "dynamic": True},
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            from .sources import FetchResult
            return FetchResult(
                source_name=self.name,
                source_type=self.source_type,
                data=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )


# =============================================================================
# Auto-Configuration Functions
# =============================================================================


def create_smart_sources(
    connection_string: str,
    topics: list[str],
    config: DefaultSourceConfig | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, SmartTableSource]]:
    """Create SmartTableSources for a list of topics.

    Args:
        connection_string: Database connection string
        topics: List of topic names to map
        config: Optional configuration (uses defaults if None)
        overrides: Optional per-topic overrides

    Returns:
        List of (topic, SmartTableSource) tuples

    Example:
        sources = create_smart_sources(
            "postgresql://localhost/mydb",
            ["orders", "products", "users"],
            overrides={
                "orders": {"table": "customer_orders"},
            },
        )
    """
    config = config or DEFAULT_CONFIG
    overrides = overrides or {}
    sources = []

    for topic in topics:
        # Get table name from convention
        table = config.get_table_name(topic)

        # Apply overrides
        override = overrides.get(topic, {})

        source = SmartTableSource(
            name=override.get("name", f"{topic}_source"),
            description=override.get("description", f"Auto-configured source for {topic}"),
            connection_string=connection_string,
            table=override.get("table", table),
            query_template=override.get("query_template", ""),
            param_mapping=override.get("param_mapping", {}),
            limit=override.get("limit", config.default_limit),
            timeout_seconds=override.get("timeout", config.timeout_seconds),
            enabled=override.get("enabled", config.enabled),
            cache_ttl_seconds=override.get("cache_ttl", config.cache_ttl_seconds),
            trigger=override.get("trigger", config.trigger),
            context_params=[p.value for p in config.default_params],
            allow_fallback=override.get("allow_fallback", True),
        )

        sources.append((topic, source))

    return sources


def auto_configure_registry(
    registry: SourceRegistry,
    connection_string: str,
    topics: list[str],
    config: DefaultSourceConfig | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Auto-configure a SourceRegistry with smart sources.

    Args:
        registry: The registry to configure
        connection_string: Database connection string
        topics: List of topic names to map
        config: Optional configuration
        overrides: Optional per-topic overrides

    Returns:
        Number of sources configured

    Example:
        count = auto_configure_registry(
            registry,
            "postgresql://localhost/mydb",
            ["orders", "products"],
        )
    """
    sources = create_smart_sources(connection_string, topics, config, overrides)

    for topic, source in sources:
        registry.map(term=topic, source=source, term_type="topic")

    return len(sources)


def discover_tables(
    connection_string: str,
    schema: str = "public",
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Discover tables from a database schema.

    Args:
        connection_string: Database connection string
        schema: Schema to query (default: public)
        exclude_patterns: Table name patterns to exclude (e.g., ["_backup", "tmp_"])

    Returns:
        List of table names

    Example:
        tables = discover_tables("postgresql://localhost/mydb")
        # Returns: ["orders", "products", "users", ...]
    """
    exclude_patterns = exclude_patterns or ["_backup", "_temp", "_tmp", "_old", "_archive"]
    tables = []

    if connection_string.startswith("postgresql"):
        tables = _discover_postgres_tables(connection_string, schema)
    elif connection_string.startswith("sqlite"):
        tables = _discover_sqlite_tables(connection_string)
    else:
        raise ValueError(f"Unsupported database for discovery: {connection_string}")

    # Filter excluded patterns
    filtered = []
    for table in tables:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in table.lower():
                exclude = True
                break
        if not exclude:
            filtered.append(table)

    return filtered


def _discover_postgres_tables(connection_string: str, schema: str) -> list[str]:
    """Discover tables from PostgreSQL."""
    try:
        import psycopg
    except ImportError:
        raise ImportError("psycopg required: pip install 'psycopg[binary]'")

    with psycopg.connect(connection_string) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        return [row[0] for row in cur.fetchall()]


def _discover_sqlite_tables(connection_string: str) -> list[str]:
    """Discover tables from SQLite."""
    import sqlite3

    db_path = connection_string.replace("sqlite:///", "")
    if db_path == ":memory:":
        return []

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall() if not row[0].startswith("sqlite_")]
    finally:
        conn.close()


# =============================================================================
# Common Topic Presets
# =============================================================================


# Common e-commerce topics with their table conventions
ECOMMERCE_TOPICS = {
    "orders": {"table": "orders", "params": ["user_id", "status"]},
    "products": {"table": "products", "params": ["category", "status"]},
    "customers": {"table": "customers", "params": ["user_id", "email"]},
    "cart": {"table": "shopping_cart", "params": ["user_id", "session_id"]},
    "payments": {"table": "payments", "params": ["user_id", "order_id"]},
    "shipping": {"table": "shipments", "params": ["order_id", "tracking_number"]},
    "reviews": {"table": "product_reviews", "params": ["product_id", "user_id"]},
    "inventory": {"table": "inventory", "params": ["product_id", "warehouse_id"]},
}

# Common CRM topics
CRM_TOPICS = {
    "contacts": {"table": "contacts", "params": ["user_id", "email"]},
    "leads": {"table": "leads", "params": ["status", "assigned_to"]},
    "opportunities": {"table": "opportunities", "params": ["stage", "owner_id"]},
    "accounts": {"table": "accounts", "params": ["user_id", "industry"]},
    "activities": {"table": "activities", "params": ["contact_id", "type"]},
    "tasks": {"table": "tasks", "params": ["assigned_to", "status"]},
    "notes": {"table": "notes", "params": ["related_to", "created_by"]},
}

# Common support topics
SUPPORT_TOPICS = {
    "tickets": {"table": "support_tickets", "params": ["user_id", "status"]},
    "messages": {"table": "ticket_messages", "params": ["ticket_id"]},
    "agents": {"table": "support_agents", "params": ["team_id", "status"]},
    "knowledge_base": {"table": "kb_articles", "params": ["category", "status"]},
    "feedback": {"table": "customer_feedback", "params": ["user_id", "rating"]},
}


def get_preset_topics(preset: str) -> dict[str, dict[str, Any]]:
    """Get topic configuration for a preset domain.

    Args:
        preset: Preset name ("ecommerce", "crm", "support")

    Returns:
        Dict of topic configs

    Example:
        topics = get_preset_topics("ecommerce")
        pipeline.auto_configure_database(
            "postgresql://...",
            topics=list(topics.keys()),
            overrides=topics,
        )
    """
    presets = {
        "ecommerce": ECOMMERCE_TOPICS,
        "crm": CRM_TOPICS,
        "support": SUPPORT_TOPICS,
    }
    return presets.get(preset, {})


def create_preset_sources(
    connection_string: str,
    preset: str,
    config: DefaultSourceConfig | None = None,
) -> list[tuple[str, SmartTableSource]]:
    """Create sources from a preset domain.

    Args:
        connection_string: Database connection string
        preset: Preset name
        config: Optional configuration

    Returns:
        List of (topic, source) tuples

    Example:
        sources = create_preset_sources("postgresql://...", "ecommerce")
    """
    preset_config = get_preset_topics(preset)
    if not preset_config:
        raise ValueError(f"Unknown preset: {preset}. Available: ecommerce, crm, support")

    return create_smart_sources(
        connection_string=connection_string,
        topics=list(preset_config.keys()),
        config=config,
        overrides=preset_config,
    )


# =============================================================================
# Quick Setup Functions
# =============================================================================


def quick_setup_database(
    connection_string: str,
    topics: list[str] | None = None,
    preset: str | None = None,
    auto_discover: bool = False,
    config: DefaultSourceConfig | None = None,
) -> SourceRegistry:
    """One-function database setup for most common cases.

    Args:
        connection_string: Database connection string
        topics: Explicit list of topics to map
        preset: Use a preset ("ecommerce", "crm", "support")
        auto_discover: Auto-discover tables from database
        config: Optional configuration

    Returns:
        Configured SourceRegistry

    Example - Explicit topics:
        registry = quick_setup_database(
            "postgresql://localhost/mydb",
            topics=["orders", "products", "users"],
        )

    Example - Preset:
        registry = quick_setup_database(
            "postgresql://localhost/mydb",
            preset="ecommerce",
        )

    Example - Auto-discover:
        registry = quick_setup_database(
            "postgresql://localhost/mydb",
            auto_discover=True,
        )
    """
    registry = SourceRegistry()

    if preset:
        # Use preset configuration
        sources = create_preset_sources(connection_string, preset, config)
        for topic, source in sources:
            registry.map(term=topic, source=source, term_type="topic")
        logger.info(f"Configured {len(sources)} sources from '{preset}' preset")

    elif auto_discover:
        # Discover tables from database
        tables = discover_tables(connection_string)
        sources = create_smart_sources(connection_string, tables, config)
        for topic, source in sources:
            registry.map(term=topic, source=source, term_type="topic")
        logger.info(f"Auto-discovered and configured {len(sources)} sources")

    elif topics:
        # Explicit topics
        sources = create_smart_sources(connection_string, topics, config)
        for topic, source in sources:
            registry.map(term=topic, source=source, term_type="topic")
        logger.info(f"Configured {len(sources)} sources for topics")

    else:
        raise ValueError(
            "Must provide one of: topics, preset, or auto_discover=True"
        )

    return registry


__all__ = [
    # Configuration
    "DefaultSourceConfig",
    "DEFAULT_CONFIG",
    "NamingConvention",
    "ParamPattern",
    # Smart sources
    "SmartTableSource",
    # Auto-configuration
    "create_smart_sources",
    "auto_configure_registry",
    "discover_tables",
    # Presets
    "ECOMMERCE_TOPICS",
    "CRM_TOPICS",
    "SUPPORT_TOPICS",
    "get_preset_topics",
    "create_preset_sources",
    # Quick setup
    "quick_setup_database",
]
