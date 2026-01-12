"""Domain Sources - PostgreSQL-centric data source management.

This module provides a thin Python API layer over PostgreSQL functions
for managing domain data sources (tables, APIs, etc.).

The heavy lifting happens in PostgreSQL:
- Source configuration stored in `domain_source_configs` table
- Query execution via `execute_table_source()` function
- Full audit trail in `domain_source_audit_log` table
- Preference extraction in `user_preferences` table

Python is just the connection layer.

Example:
    from mindcore.svl.domain_sources import DomainSourceManager

    manager = DomainSourceManager(connection_string="postgresql://...")

    # Register a source (writes to PostgreSQL)
    manager.register_source(
        name="orders",
        source_type="table",
        topics=["orders", "purchases"],
        query_template="SELECT * FROM orders WHERE user_id = :user_id LIMIT 10",
    )

    # Fetch data for topics (calls PostgreSQL function)
    result = manager.fetch_for_topics(
        topics=["orders"],
        user_id="user_123",
    )

    # Get audit trail
    audit = manager.get_audit_summary(user_id="user_123")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DomainSourceType(str, Enum):
    """Types of domain data sources."""

    TABLE = "table"
    API = "api"
    FUNCTION = "function"
    MCP = "mcp"


@dataclass
class SourceFetchResult:
    """Result from fetching data from a domain source."""

    source_name: str
    success: bool
    data: list[dict[str, Any]] | None = None
    error: str | None = None
    rows: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    topics_matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "rows": self.rows,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "topics_matched": self.topics_matched,
        }


@dataclass
class PreferenceExtraction:
    """Extracted preference from a message."""

    preference_type: str  # 'communication', 'product', 'ui', etc.
    preference_key: str
    preference_value: Any
    confidence: float = 1.0
    source_text: str | None = None


@dataclass
class AuditSummary:
    """Summary of domain source operations."""

    source_name: str
    total_calls: int
    successful_calls: int
    avg_latency_ms: float
    cache_hit_rate: float


class DomainSourceManager:
    """PostgreSQL-centric domain source manager.

    This class provides a thin Python API over PostgreSQL functions
    for managing domain data sources.

    Architecture:
        Python (this class) → PostgreSQL Functions → Data

    All heavy lifting happens in PostgreSQL:
        - get_sources_for_topics() - Find sources for topics
        - execute_table_source() - Execute queries with audit logging
        - get_user_preferences() - Retrieve preferences
        - upsert_preference() - Store preferences with versioning
    """

    def __init__(self, connection_string: str):
        """Initialize domain source manager.

        Args:
            connection_string: PostgreSQL connection string
        """
        self._connection_string = connection_string
        self._pool = None

    def _get_connection(self):
        """Get a database connection."""
        try:
            import psycopg
        except ImportError:
            raise ImportError("psycopg required: pip install 'psycopg[binary]'")

        return psycopg.connect(self._connection_string)

    # =========================================================================
    # SOURCE REGISTRATION
    # =========================================================================

    def register_source(
        self,
        name: str,
        source_type: str | DomainSourceType,
        topics: list[str],
        *,
        description: str = "",
        table_name: str | None = None,
        query_template: str | None = None,
        param_mapping: dict[str, str] | None = None,
        api_url: str | None = None,
        api_method: str = "GET",
        api_headers: dict[str, str] | None = None,
        function_name: str | None = None,
        cache_ttl_seconds: int = 60,
        priority: int = 0,
    ) -> str:
        """Register a domain source in PostgreSQL.

        Args:
            name: Unique source name
            source_type: 'table', 'api', 'function', or 'mcp'
            topics: List of topics this source handles
            description: Human-readable description
            table_name: For table sources, the table name
            query_template: SQL query with :param placeholders
            param_mapping: Map context keys to SQL params
            api_url: For API sources, the endpoint URL
            api_method: HTTP method for API sources
            api_headers: Headers for API sources
            function_name: For function sources, the PostgreSQL function name
            cache_ttl_seconds: How long to cache results
            priority: Higher priority sources checked first

        Returns:
            Source ID (UUID)
        """
        if isinstance(source_type, DomainSourceType):
            source_type = source_type.value

        logger.info(
            "Registering domain source",
            extra={
                "source_name": name,
                "source_type": source_type,
                "topics": topics,
            },
        )

        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO domain_source_configs (
                    name, source_type, description, topics,
                    table_name, query_template, param_mapping,
                    api_url, api_method, api_headers,
                    function_name, cache_ttl_seconds, priority
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (name) DO UPDATE SET
                    source_type = EXCLUDED.source_type,
                    description = EXCLUDED.description,
                    topics = EXCLUDED.topics,
                    table_name = EXCLUDED.table_name,
                    query_template = EXCLUDED.query_template,
                    param_mapping = EXCLUDED.param_mapping,
                    api_url = EXCLUDED.api_url,
                    api_method = EXCLUDED.api_method,
                    api_headers = EXCLUDED.api_headers,
                    function_name = EXCLUDED.function_name,
                    cache_ttl_seconds = EXCLUDED.cache_ttl_seconds,
                    priority = EXCLUDED.priority,
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    name,
                    source_type,
                    description,
                    topics,
                    table_name,
                    query_template,
                    json.dumps(param_mapping or {}),
                    api_url,
                    api_method,
                    json.dumps(api_headers or {}),
                    function_name,
                    cache_ttl_seconds,
                    priority,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            return str(result[0]) if result else ""

    def unregister_source(self, name: str) -> bool:
        """Remove a domain source.

        Args:
            name: Source name to remove

        Returns:
            True if source was removed
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM domain_source_configs WHERE name = %s",
                (name,),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_sources(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """List all registered domain sources.

        Args:
            enabled_only: If True, only return enabled sources

        Returns:
            List of source configurations
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            query = "SELECT * FROM domain_source_configs"
            if enabled_only:
                query += " WHERE enabled = true"
            query += " ORDER BY priority DESC, name"

            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def fetch_for_topics(
        self,
        topics: list[str],
        user_id: str,
        params: dict[str, Any] | None = None,
    ) -> list[SourceFetchResult]:
        """Fetch data from all sources that handle the given topics.

        This calls PostgreSQL's get_sources_for_topics() and execute_table_source()
        functions for full audit logging.

        Args:
            topics: List of topics to fetch data for
            user_id: User ID for parameterized queries
            params: Additional parameters for queries

        Returns:
            List of fetch results from each matched source
        """
        logger.info(
            "Fetching data for topics",
            extra={"topics": topics, "user_id": user_id},
        )

        results = []
        params = params or {}

        with self._get_connection() as conn, conn.cursor() as cur:
            # Get sources for topics (PostgreSQL function)
            cur.execute(
                "SELECT * FROM get_sources_for_topics(%s)",
                (topics,),
            )

            sources = cur.fetchall()
            source_columns = [desc[0] for desc in cur.description]

            for source_row in sources:
                source = dict(zip(source_columns, source_row, strict=False))
                source_name = source["source_name"]
                source_type = source["source_type"]
                matched_topics = source.get("matched_topics", [])

                logger.debug(
                    "Executing source",
                    extra={
                        "source_name": source_name,
                        "source_type": source_type,
                        "matched_topics": matched_topics,
                    },
                )

                if source_type == "table":
                    # Use PostgreSQL function for execution and audit logging
                    cur.execute(
                        "SELECT execute_table_source(%s, %s, %s)",
                        (source_name, user_id, json.dumps(params)),
                    )
                    result_json = cur.fetchone()[0]

                    results.append(
                        SourceFetchResult(
                            source_name=source_name,
                            success=result_json.get("success", False),
                            data=result_json.get("data"),
                            error=result_json.get("error"),
                            rows=result_json.get("rows", 0),
                            latency_ms=result_json.get("latency_ms", 0),
                            topics_matched=matched_topics,
                        )
                    )

                elif source_type == "function":
                    # Call PostgreSQL function directly
                    function_name = source.get("function_name")
                    if function_name:
                        try:
                            cur.execute(
                                f"SELECT {function_name}(%s)",  # noqa: S608
                                (user_id,),
                            )
                            data = cur.fetchone()[0]
                            results.append(
                                SourceFetchResult(
                                    source_name=source_name,
                                    success=True,
                                    data=[data] if isinstance(data, dict) else data,
                                    rows=1,
                                    topics_matched=matched_topics,
                                )
                            )
                        except Exception as e:
                            results.append(
                                SourceFetchResult(
                                    source_name=source_name,
                                    success=False,
                                    error=str(e),
                                    topics_matched=matched_topics,
                                )
                            )

                # API sources would be handled differently (HTTP call)
                # For now, we focus on PostgreSQL-native sources

        return results

    # =========================================================================
    # PREFERENCE MANAGEMENT
    # =========================================================================

    def get_user_preferences(
        self,
        user_id: str,
        preference_type: str | None = None,
    ) -> dict[str, Any]:
        """Get active preferences for a user.

        Calls PostgreSQL's get_user_preferences() function.

        Args:
            user_id: User ID
            preference_type: Optional filter by type

        Returns:
            Dictionary of preferences
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT get_user_preferences(%s, %s)",
                (user_id, preference_type),
            )
            result = cur.fetchone()
            return result[0] if result else {}

    def store_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_key: str,
        preference_value: Any,
        confidence: float = 1.0,
        source_memory_id: str | None = None,
    ) -> str:
        """Store a user preference with versioning.

        Calls PostgreSQL's upsert_preference() function.

        Args:
            user_id: User ID
            preference_type: Type of preference ('communication', 'ui', etc.)
            preference_key: Preference key
            preference_value: Preference value (will be stored as JSONB)
            confidence: Confidence score (0-1)
            source_memory_id: ID of memory this was extracted from

        Returns:
            Preference ID
        """
        logger.info(
            "Storing preference",
            extra={
                "user_id": user_id,
                "preference_type": preference_type,
                "preference_key": preference_key,
            },
        )

        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT upsert_preference(%s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    preference_type,
                    preference_key,
                    json.dumps(preference_value),
                    confidence,
                    source_memory_id,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            return str(result[0]) if result else ""

    def extract_preferences_from_text(
        self,
        text: str,
        user_id: str,
    ) -> list[PreferenceExtraction]:
        """Extract preferences from text using pattern matching.

        This is a simple rule-based extractor. For better results,
        use LLM-based extraction via MetadataExtractor.

        Args:
            text: Text to extract preferences from
            user_id: User ID for context

        Returns:
            List of extracted preferences
        """
        preferences = []
        text_lower = text.lower()

        # Simple pattern matching for common preferences
        patterns = [
            # Communication preferences
            (r"prefer\s+(email|phone|chat|sms)", "communication", "channel"),
            (r"(don't|do not)\s+call\s+me", "communication", "no_calls"),
            (r"(prefer|want)\s+(brief|detailed)\s+(response|answer)", "communication", "response_length"),

            # UI preferences
            (r"(prefer|want|like)\s+(dark|light)\s+mode", "ui", "theme"),
            (r"(large|small)\s+(font|text)", "ui", "font_size"),

            # Product preferences
            (r"(prefer|favorite|like)\s+(brand|product):\s*(\w+)", "product", "favorite_brand"),
            (r"(allergic|allergy)\s+to\s+(\w+)", "product", "allergies"),

            # Time preferences
            (r"(best|preferred)\s+time.*?(morning|afternoon|evening|night)", "scheduling", "preferred_time"),
        ]

        import re

        for pattern, pref_type, pref_key in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1) if len(match.groups()) >= 1 else True
                preferences.append(
                    PreferenceExtraction(
                        preference_type=pref_type,
                        preference_key=pref_key,
                        preference_value=value,
                        confidence=0.8,  # Rule-based = 80% confidence
                        source_text=match.group(0),
                    )
                )

        return preferences

    # =========================================================================
    # AUDIT & TRACEABILITY
    # =========================================================================

    def get_audit_summary(
        self,
        user_id: str,
        since_hours: int = 24,
    ) -> list[AuditSummary]:
        """Get audit summary for a user.

        Calls PostgreSQL's get_audit_summary() function.

        Args:
            user_id: User ID
            since_hours: How many hours to look back

        Returns:
            List of audit summaries by source
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM get_audit_summary(%s, NOW() - INTERVAL '%s hours')",
                (user_id, since_hours),
            )

            return [
                AuditSummary(
                    source_name=row[0],
                    total_calls=row[1],
                    successful_calls=row[2],
                    avg_latency_ms=float(row[3]) if row[3] else 0.0,
                    cache_hit_rate=float(row[4]) if row[4] else 0.0,
                )
                for row in cur.fetchall()
            ]

    def get_audit_log(
        self,
        user_id: str | None = None,
        source_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get detailed audit log entries.

        Args:
            user_id: Optional filter by user
            source_name: Optional filter by source
            limit: Maximum entries to return

        Returns:
            List of audit log entries
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            query = "SELECT * FROM domain_source_audit_log WHERE 1=1"
            params = []

            if user_id:
                query += " AND user_id = %s"
                params.append(user_id)
            if source_name:
                query += " AND source_name = %s"
                params.append(source_name)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]

    # =========================================================================
    # SCHEMA SETUP
    # =========================================================================

    def setup_schema(self) -> bool:
        """Create domain sources schema if not exists.

        Runs the domain_sources.sql schema file.

        Returns:
            True if schema was created/updated successfully
        """
        import importlib.resources

        try:
            schema_path = (
                importlib.resources.files("mindcore.storage.schema") / "domain_sources.sql"
            )
            with importlib.resources.as_file(schema_path) as path:
                sql = path.read_text()
        except Exception:
            # Fallback: read from file directly
            from pathlib import Path

            schema_file = Path(__file__).parent.parent / "storage" / "schema" / "domain_sources.sql"
            sql = schema_file.read_text()

        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()

        logger.info("Domain sources schema created successfully")
        return True
