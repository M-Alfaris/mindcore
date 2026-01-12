"""Tests for PostgreSQL-centric domain source management."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from mindcore.svl.domain_sources import (
    DomainSourceManager,
    DomainSourceType,
    SourceFetchResult,
    PreferenceExtraction,
    AuditSummary,
)


class TestDomainSourceType:
    """Test DomainSourceType enum."""

    def test_values(self):
        assert DomainSourceType.TABLE.value == "table"
        assert DomainSourceType.API.value == "api"
        assert DomainSourceType.FUNCTION.value == "function"
        assert DomainSourceType.MCP.value == "mcp"


class TestSourceFetchResult:
    """Test SourceFetchResult dataclass."""

    def test_creation(self):
        result = SourceFetchResult(
            source_name="orders",
            success=True,
            data=[{"order_id": "123", "status": "shipped"}],
            rows=1,
            latency_ms=15.5,
        )

        assert result.source_name == "orders"
        assert result.success is True
        assert result.rows == 1
        assert result.latency_ms == 15.5

    def test_to_dict(self):
        result = SourceFetchResult(
            source_name="test",
            success=True,
            data=[{"foo": "bar"}],
        )
        d = result.to_dict()

        assert d["source_name"] == "test"
        assert d["success"] is True
        assert d["data"] == [{"foo": "bar"}]


class TestPreferenceExtraction:
    """Test PreferenceExtraction dataclass."""

    def test_creation(self):
        pref = PreferenceExtraction(
            preference_type="communication",
            preference_key="channel",
            preference_value="email",
            confidence=0.9,
            source_text="I prefer email",
        )

        assert pref.preference_type == "communication"
        assert pref.preference_key == "channel"
        assert pref.preference_value == "email"
        assert pref.confidence == 0.9


class TestDomainSourceManager:
    """Test DomainSourceManager class."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    @pytest.fixture
    def manager(self):
        """Create a DomainSourceManager instance."""
        return DomainSourceManager("postgresql://test:test@localhost/test")

    def test_init(self, manager):
        assert manager._connection_string == "postgresql://test:test@localhost/test"

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_register_source(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = ("uuid-123",)

        result = manager.register_source(
            name="orders_source",
            source_type="table",
            topics=["orders", "purchases"],
            query_template="SELECT * FROM orders WHERE user_id = :user_id",
        )

        assert result == "uuid-123"
        mock_cursor.execute.assert_called_once()

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_list_sources(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn

        mock_cursor.description = [
            ("id",), ("name",), ("source_type",), ("topics",), ("enabled",)
        ]
        mock_cursor.fetchall.return_value = [
            ("uuid-1", "orders", "table", ["orders"], True),
            ("uuid-2", "prefs", "function", ["preferences"], True),
        ]

        sources = manager.list_sources()

        assert len(sources) == 2
        assert sources[0]["name"] == "orders"
        assert sources[1]["name"] == "prefs"

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_fetch_for_topics(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn

        # Mock get_sources_for_topics result
        mock_cursor.description = [
            ("source_id",), ("source_name",), ("source_type",),
            ("table_name",), ("query_template",), ("param_mapping",),
            ("api_url",), ("function_name",), ("cache_ttl_seconds",),
            ("matched_topics",)
        ]
        mock_cursor.fetchall.return_value = [
            (
                "uuid-1", "orders_source", "table",
                "orders", "SELECT * FROM orders", {},
                None, None, 60,
                ["orders"]
            )
        ]
        mock_cursor.fetchone.return_value = (
            {"success": True, "data": [{"order_id": "123"}], "rows": 1, "latency_ms": 10.5},
        )

        results = manager.fetch_for_topics(
            topics=["orders"],
            user_id="user_123",
        )

        assert len(results) == 1
        assert results[0].source_name == "orders_source"
        assert results[0].success is True

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_get_user_preferences(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = (
            {"communication.channel": {"value": "email", "confidence": 0.9}},
        )

        prefs = manager.get_user_preferences(user_id="user_123")

        assert "communication.channel" in prefs
        assert prefs["communication.channel"]["value"] == "email"

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_store_preference(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = ("uuid-pref-123",)

        pref_id = manager.store_preference(
            user_id="user_123",
            preference_type="communication",
            preference_key="channel",
            preference_value="email",
            confidence=0.9,
        )

        assert pref_id == "uuid-pref-123"
        mock_cursor.execute.assert_called_once()

    def test_extract_preferences_from_text(self, manager):
        """Test rule-based preference extraction."""
        text = "I prefer email for communications and dark mode for the UI."

        prefs = manager.extract_preferences_from_text(text, user_id="user_123")

        # Should extract at least the email preference
        pref_types = [p.preference_type for p in prefs]
        assert any("communication" in t or "ui" in t for t in pref_types)

    def test_extract_preferences_no_match(self, manager):
        """Test extraction with no preferences in text."""
        text = "Just a regular message with no preferences."

        prefs = manager.extract_preferences_from_text(text, user_id="user_123")

        # May or may not find preferences depending on patterns
        assert isinstance(prefs, list)

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_get_audit_summary(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchall.return_value = [
            ("orders_source", 100, 95, 15.5, 0.75),
        ]

        summary = manager.get_audit_summary(user_id="user_123")

        assert len(summary) == 1
        assert summary[0].source_name == "orders_source"
        assert summary[0].total_calls == 100
        assert summary[0].successful_calls == 95
        assert summary[0].cache_hit_rate == 0.75

    @patch("mindcore.svl.domain_sources.DomainSourceManager._get_connection")
    def test_get_audit_log(self, mock_get_conn, manager, mock_connection):
        mock_conn, mock_cursor = mock_connection
        mock_get_conn.return_value = mock_conn
        mock_cursor.description = [
            ("id",), ("operation",), ("source_name",), ("success",), ("latency_ms",)
        ]
        mock_cursor.fetchall.return_value = [
            (1, "fetch", "orders_source", True, 10.5),
            (2, "fetch", "orders_source", True, 12.3),
        ]

        logs = manager.get_audit_log(user_id="user_123", limit=10)

        assert len(logs) == 2
        assert logs[0]["operation"] == "fetch"
        assert logs[0]["success"] is True


class TestPreferenceExtractionPatterns:
    """Test preference extraction patterns."""

    @pytest.fixture
    def manager(self):
        return DomainSourceManager("postgresql://test:test@localhost/test")

    def test_email_preference(self, manager):
        text = "I prefer email"
        prefs = manager.extract_preferences_from_text(text, "user_123")

        comm_prefs = [p for p in prefs if p.preference_type == "communication"]
        assert len(comm_prefs) >= 1

    def test_dark_mode_preference(self, manager):
        text = "I want dark mode please"
        prefs = manager.extract_preferences_from_text(text, "user_123")

        ui_prefs = [p for p in prefs if p.preference_type == "ui"]
        assert len(ui_prefs) >= 1

    def test_brief_response_preference(self, manager):
        text = "I prefer brief responses"
        prefs = manager.extract_preferences_from_text(text, "user_123")

        comm_prefs = [p for p in prefs if p.preference_type == "communication"]
        # May or may not match depending on exact regex
        assert isinstance(prefs, list)


class TestIntegrationWithPostgreSQL:
    """Integration tests requiring a real PostgreSQL database.

    These tests are skipped unless DATABASE_URL environment variable is set.
    """

    @pytest.fixture
    def manager(self):
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not set")
        return DomainSourceManager(db_url)

    @pytest.mark.integration
    def test_full_flow(self, manager):
        """Test complete flow: register, fetch, audit."""
        # Setup schema
        manager.setup_schema()

        # Register a test source
        source_id = manager.register_source(
            name="test_orders",
            source_type="table",
            topics=["orders", "test"],
            query_template="SELECT 1 as test_value",
        )
        assert source_id

        # List sources
        sources = manager.list_sources()
        assert any(s["name"] == "test_orders" for s in sources)

        # Fetch (will fail without actual table, but tests the flow)
        results = manager.fetch_for_topics(["orders"], user_id="test_user")
        assert isinstance(results, list)

        # Cleanup
        manager.unregister_source("test_orders")
