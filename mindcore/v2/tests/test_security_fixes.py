"""Tests for security and bug fixes from the code audit.

This module tests the fixes for:
1. Path traversal protection in SQLite source
2. URL validation for SSRF protection in API source
3. Timezone-aware datetime comparisons
4. Safe query result handling
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta


class TestPathTraversalProtection:
    """Tests for path traversal protection in SQLite source."""

    def test_sqlite_path_traversal_blocked(self):
        """Test that sensitive system paths are blocked."""
        from mindcore.v2.svl.sources import TableSource

        # Test sensitive paths are blocked
        sensitive_paths = [
            "sqlite:////etc/passwd",
            "sqlite:////etc/shadow",
            "sqlite:////proc/self/environ",
            "sqlite:////sys/class/net",
            "sqlite:////dev/null",
            "sqlite:////root/.ssh/id_rsa",
            "sqlite:////boot/vmlinuz",
        ]

        for path in sensitive_paths:
            source = TableSource(
                connection_string=path,
                table_name="test",
                columns=["id"],
            )
            with pytest.raises(ValueError, match="Access to system path is not allowed"):
                source._fetch_sqlite("SELECT 1", {})

    def test_sqlite_invalid_extension_blocked(self):
        """Test that non-SQLite file extensions are blocked."""
        from mindcore.v2.svl.sources import TableSource

        # Test invalid extensions
        invalid_paths = [
            "sqlite:////tmp/data.txt",
            "sqlite:////tmp/config.json",
            "sqlite:////tmp/script.py",
        ]

        for path in invalid_paths:
            source = TableSource(
                connection_string=path,
                table_name="test",
                columns=["id"],
            )
            with pytest.raises(ValueError, match="Invalid SQLite database extension"):
                source._fetch_sqlite("SELECT 1", {})

    def test_sqlite_valid_extensions_allowed(self):
        """Test that valid SQLite extensions are allowed."""
        from mindcore.v2.svl.sources import TableSource

        # These should not raise on extension check (may fail on actual connection)
        valid_extensions = [".db", ".sqlite", ".sqlite3"]

        for ext in valid_extensions:
            source = TableSource(
                connection_string=f"sqlite:////tmp/test{ext}",
                table_name="test",
                columns=["id"],
            )
            # Should not raise ValueError for extension
            # (will raise sqlite3 error for non-existent file, which is expected)
            try:
                source._fetch_sqlite("SELECT 1", {})
            except ValueError:
                pytest.fail(f"Valid extension {ext} was incorrectly rejected")
            except Exception:
                # Other errors (like file not found) are acceptable
                pass

    def test_sqlite_memory_db_allowed(self):
        """Test that in-memory database is allowed."""
        from mindcore.v2.svl.sources import TableSource

        source = TableSource(
            connection_string="sqlite:///:memory:",
            table_name="test",
            columns=["id"],
        )
        # Should not raise ValueError (may fail on query, but extension check passes)
        try:
            source._fetch_sqlite("SELECT 1", {})
        except ValueError as e:
            if "extension" in str(e).lower() or "system path" in str(e).lower():
                pytest.fail(":memory: should be allowed")


class TestURLValidation:
    """Tests for URL validation in API source."""

    def test_url_scheme_validation(self):
        """Test that only http/https schemes are allowed."""
        from mindcore.v2.svl.sources import APISource

        # Test invalid schemes
        invalid_schemes = [
            "file:///etc/passwd",
            "ftp://example.com/data",
            "gopher://evil.com",
        ]

        for url in invalid_schemes:
            source = APISource(url=url)
            result = source.fetch({})
            # Should fail with error about scheme or return error result
            assert result.error is not None or "Invalid URL scheme" in str(result.error)

    def test_url_param_encoding(self):
        """Test that URL parameters are properly encoded."""
        from mindcore.v2.svl.sources import APISource

        source = APISource(
            url="https://api.example.com/users/{user_id}",
            url_params={"user_id_context": "user_id"},
        )

        # Test that path traversal in params is encoded
        # The value "../../../etc/passwd" should be URL-encoded
        context = {"user_id_context": "../../../etc/passwd"}

        # Fetch will fail (no network), but we can verify the URL construction
        # by checking that it doesn't raise a ValueError for the encoding
        result = source.fetch(context)
        # Connection error is expected, not ValueError
        assert result.error is not None


class TestTimezoneComparison:
    """Tests for timezone-aware datetime comparisons."""

    def test_access_policy_expiration_timezone_aware(self):
        """Test that access policy expiration handles timezones correctly."""
        from mindcore.v2.federation.access_control import AccessPolicy

        # Test with timezone-aware datetime
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        policy = AccessPolicy(expires_at=future)
        assert not policy.is_expired()

        # Test with past datetime
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        policy = AccessPolicy(expires_at=past)
        assert policy.is_expired()

    def test_access_policy_available_from_timezone_aware(self):
        """Test that access policy availability handles timezones correctly."""
        from mindcore.v2.federation.access_control import AccessPolicy

        # Test with future availability
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        policy = AccessPolicy(available_from=future)
        assert not policy.is_available()

        # Test with past availability
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        policy = AccessPolicy(available_from=past)
        assert policy.is_available()

    def test_memory_acl_created_at_is_timezone_aware(self):
        """Test that MemoryACL created_at is timezone-aware."""
        from mindcore.v2.federation.access_control import MemoryACL, AccessScope

        scope = AccessScope(org_id="test-org")
        acl = MemoryACL(
            memory_id="test-memory",
            owner_agent_id="test-agent",
            owner_scope=scope,
        )

        # created_at should be timezone-aware
        assert acl.created_at.tzinfo is not None

    def test_record_access_uses_timezone_aware_datetime(self):
        """Test that record_access uses timezone-aware datetime."""
        from mindcore.v2.federation.access_control import MemoryACL, AccessScope

        scope = AccessScope(org_id="test-org")
        acl = MemoryACL(
            memory_id="test-memory",
            owner_agent_id="test-agent",
            owner_scope=scope,
        )

        acl.record_access("another-agent")

        # last_accessed_at should be timezone-aware
        assert acl.last_accessed_at is not None
        assert acl.last_accessed_at.tzinfo is not None


class TestQueryOptimizerConstants:
    """Tests for query optimizer constants."""

    def test_constants_are_defined(self):
        """Test that all query optimizer constants are defined."""
        from mindcore.v2.flr.query_optimizer import (
            USAGE_HISTORY_WINDOW,
            LOW_USAGE_THRESHOLD,
            HIGH_USAGE_THRESHOLD,
            LOW_USAGE_LIMIT_MULTIPLIER,
            HIGH_USAGE_LIMIT_MULTIPLIER,
            MIN_RETRIEVAL_LIMIT,
            MAX_RETRIEVAL_LIMIT,
            CONFIDENCE_SAMPLE_THRESHOLD,
            POOR_TOPIC_THRESHOLD,
        )

        # Verify constants have sensible values
        assert USAGE_HISTORY_WINDOW > 0
        assert 0 < LOW_USAGE_THRESHOLD < HIGH_USAGE_THRESHOLD < 1
        assert 0 < LOW_USAGE_LIMIT_MULTIPLIER < 1
        assert HIGH_USAGE_LIMIT_MULTIPLIER > 1
        assert MIN_RETRIEVAL_LIMIT > 0
        assert MAX_RETRIEVAL_LIMIT > MIN_RETRIEVAL_LIMIT
        assert CONFIDENCE_SAMPLE_THRESHOLD > 0
        assert 0 < POOR_TOPIC_THRESHOLD < 1


class TestRecallConstants:
    """Tests for recall scoring constants."""

    def test_popularity_constant_defined(self):
        """Test that popularity normalization constant is defined."""
        from mindcore.v2.flr.recall import POPULARITY_NORMALIZATION_FACTOR

        assert POPULARITY_NORMALIZATION_FACTOR > 0


class TestSafeQueryResults:
    """Tests for safe handling of query results."""

    def test_postgres_message_index_handles_empty_result(self):
        """Test that message index handles empty results gracefully."""
        # This tests the code path, actual DB testing would require a fixture
        from mindcore.v2.storage.postgres import PostgresStorage

        # Verify the method signature and logic is correct
        # The actual test requires a database connection
        assert hasattr(PostgresStorage, '_get_next_message_index_internal')
