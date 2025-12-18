"""Tests for enterprise features.

Tests observability, rate limiting, audit logging, and encryption
modules.
"""

import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest


class TestRateLimiting:
    """Tests for rate limiting module."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter."""
        pytest.importorskip("limits")
        from mindcore.v2.enterprise import RateLimiter

        return RateLimiter(limit="10/minute")

    def test_is_allowed(self, rate_limiter):
        """Test basic rate limiting."""
        # First request should be allowed
        assert rate_limiter.is_allowed("user123", "store") is True

    def test_rate_limit_exceeded(self, rate_limiter):
        """Test that rate limit is enforced."""
        # Exhaust the limit
        for _ in range(10):
            rate_limiter.is_allowed("user_exceed", "store")

        # Next request should be denied
        assert rate_limiter.is_allowed("user_exceed", "store") is False

    def test_different_users_independent(self, rate_limiter):
        """Test that different users have independent limits."""
        # Exhaust limit for user1
        for _ in range(10):
            rate_limiter.is_allowed("user1", "store")

        # user2 should still be allowed
        assert rate_limiter.is_allowed("user2", "store") is True

    def test_check_without_consuming(self, rate_limiter):
        """Test checking limit without consuming quota."""
        # Check should return True
        assert rate_limiter.check("user_check", "store") is True

        # After checking 10 times (not consuming), is_allowed should still work
        for _ in range(10):
            rate_limiter.check("user_check", "store")

        # This consumes quota - should still work
        assert rate_limiter.is_allowed("user_check", "store") is True

    def test_get_remaining(self, rate_limiter):
        """Test getting remaining quota."""
        # Use 3 requests
        for _ in range(3):
            rate_limiter.is_allowed("user_remaining", "store")

        remaining = rate_limiter.get_remaining("user_remaining", "store")
        # window_stats.remaining from limits library is the remaining available
        # So after 3 uses out of 10, remaining should be 7
        assert remaining == 7

    def test_context_manager(self, rate_limiter):
        """Test rate limiting context manager."""
        from mindcore.v2.enterprise import RateLimitExceeded

        # Should work within limit
        with rate_limiter.limit("user_ctx", "store"):
            pass

        # Exhaust limit
        for _ in range(9):  # Already used 1
            rate_limiter.is_allowed("user_ctx", "store")

        # Should raise
        with pytest.raises(RateLimitExceeded), rate_limiter.limit("user_ctx", "store"):
            pass

    def test_operation_specific_limits(self):
        """Test operation-specific rate limits."""
        pytest.importorskip("limits")
        from mindcore.v2.enterprise import RateLimitConfig, RateLimiter

        config = RateLimitConfig(
            default_limit="100/minute",
            operation_limits={
                "store": "5/minute",
                "recall": "50/minute",
            },
        )
        limiter = RateLimiter(config=config)

        # Exhaust store limit
        for _ in range(5):
            limiter.is_allowed("user_op", "store")

        # Store should be limited
        assert limiter.is_allowed("user_op", "store") is False

        # Recall should still work
        assert limiter.is_allowed("user_op", "recall") is True

    def test_get_stats(self, rate_limiter):
        """Test getting detailed stats."""
        # Use some quota
        rate_limiter.is_allowed("user_stats", "store")
        rate_limiter.is_allowed("user_stats", "store")

        stats = rate_limiter.get_stats("user_stats", "store")

        assert "limit" in stats
        assert "remaining" in stats
        assert "used" in stats
        assert stats["limit"] == 10


class TestEncryption:
    """Tests for encryption module."""

    @pytest.fixture
    def encryptor(self):
        """Create a field encryptor."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import EncryptionConfig, FieldEncryptor

        key = FieldEncryptor.generate_key()
        config = EncryptionConfig(key=key)
        return FieldEncryptor(config)

    def test_encrypt_decrypt(self, encryptor):
        """Test basic encryption and decryption."""
        plaintext = "sensitive user data"
        encrypted = encryptor.encrypt(plaintext)

        assert encrypted != plaintext
        assert encrypted.startswith("enc:v1:")

        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext

    def test_is_encrypted(self, encryptor):
        """Test encrypted detection."""
        plaintext = "not encrypted"
        encrypted = encryptor.encrypt("sensitive")

        assert encryptor.is_encrypted(encrypted) is True
        assert encryptor.is_encrypted(plaintext) is False

    def test_encrypt_memory(self, encryptor):
        """Test encrypting memory dict."""
        memory = {
            "memory_id": "mem_123",
            "content": "sensitive user preference",
            "topics": ["preferences"],
            "user_id": "user123",
        }

        encrypted = encryptor.encrypt_memory(memory)

        # Content should be encrypted
        assert encryptor.is_encrypted(encrypted["content"])

        # Other fields should not be encrypted
        assert encrypted["memory_id"] == "mem_123"
        assert encrypted["topics"] == ["preferences"]

        # Metadata should be added
        assert encrypted["_encrypted"] is True

    def test_decrypt_memory(self, encryptor):
        """Test decrypting memory dict."""
        memory = {
            "memory_id": "mem_123",
            "content": "sensitive data",
            "topics": ["test"],
        }

        encrypted = encryptor.encrypt_memory(memory)
        decrypted = encryptor.decrypt_memory(encrypted)

        assert decrypted["content"] == "sensitive data"
        assert "_encrypted" not in decrypted

    def test_generate_key(self):
        """Test key generation."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import FieldEncryptor

        key1 = FieldEncryptor.generate_key()
        key2 = FieldEncryptor.generate_key()

        # Keys should be unique
        assert key1 != key2

        # Keys should be valid base64
        import base64

        base64.urlsafe_b64decode(key1)

    def test_password_based_key_derivation(self):
        """Test key derivation from password."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import EncryptionConfig, FieldEncryptor

        config = EncryptionConfig(
            password="strong-password-123",  # noqa: S106
            salt="unique-salt-value",
            kdf_iterations=100_000,  # Reduced for testing
        )

        encryptor = FieldEncryptor(config)

        plaintext = "test data"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext

    def test_key_from_env(self):
        """Test loading key from environment."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import EncryptionConfig, FieldEncryptor

        key = FieldEncryptor.generate_key()
        os.environ["TEST_ENCRYPTION_KEY"] = key

        try:
            config = EncryptionConfig(key_from_env="TEST_ENCRYPTION_KEY")
            encryptor = FieldEncryptor(config)

            encrypted = encryptor.encrypt("test")
            decrypted = encryptor.decrypt(encrypted)
            assert decrypted == "test"
        finally:
            del os.environ["TEST_ENCRYPTION_KEY"]

    def test_empty_value_handling(self, encryptor):
        """Test handling of empty values."""
        assert encryptor.encrypt("") == ""
        assert encryptor.decrypt("") == ""
        assert encryptor.is_encrypted("") is False


class TestKeyRotation:
    """Tests for key rotation."""

    def test_key_rotation(self):
        """Test rotating encryption keys."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import (
            EncryptionConfig,
            FieldEncryptor,
            KeyRotator,
        )

        # Generate old and new keys
        old_key = FieldEncryptor.generate_key()
        new_key = FieldEncryptor.generate_key()

        # Encrypt with old key
        old_encryptor = FieldEncryptor(EncryptionConfig(key=old_key))
        memory = {"content": "sensitive data", "topics": ["test"]}
        encrypted = old_encryptor.encrypt_memory(memory)

        # Rotate to new key
        rotator = KeyRotator(keys=[old_key, new_key], primary_key_index=1)
        rotated = rotator.rotate_memory(encrypted)

        # Decrypt with new key should work
        new_encryptor = FieldEncryptor(EncryptionConfig(key=new_key))
        decrypted = new_encryptor.decrypt_memory(rotated)

        assert decrypted["content"] == "sensitive data"

    def test_verify_rotation(self):
        """Test rotation verification."""
        pytest.importorskip("cryptography")
        from mindcore.v2.enterprise import (
            EncryptionConfig,
            FieldEncryptor,
            KeyRotator,
        )

        old_key = FieldEncryptor.generate_key()
        new_key = FieldEncryptor.generate_key()

        old_encryptor = FieldEncryptor(EncryptionConfig(key=old_key))
        memory = {"content": "test data"}
        encrypted = old_encryptor.encrypt_memory(memory)

        rotator = KeyRotator(keys=[old_key, new_key], primary_key_index=1)
        rotated = rotator.rotate_memory(encrypted)

        assert rotator.verify_rotation(encrypted, rotated) is True


class TestAuditLogging:
    """Tests for audit logging module."""

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger."""
        from mindcore.v2.enterprise import AuditConfig, AuditLogger

        config = AuditConfig(output="stdout", json_format=False)
        return AuditLogger(config)

    def test_log_store(self, audit_logger, capsys):
        """Test logging store operations."""
        audit_logger.log_store(
            user_id="user123",
            memory_id="mem_abc",
            memory_type="semantic",
        )

        captured = capsys.readouterr()
        assert "memory.store" in captured.out

    def test_log_recall(self, audit_logger, capsys):
        """Test logging recall operations."""
        audit_logger.log_recall(
            user_id="user123",
            query_hash="sha256:abc123",
            memories_returned=5,
        )

        captured = capsys.readouterr()
        assert "memory.recall" in captured.out

    def test_log_access(self, audit_logger, capsys):
        """Test logging access events."""
        audit_logger.log_access(
            user_id="user123",
            resource="memory:mem_abc",
            action="read",
            granted=True,
        )

        captured = capsys.readouterr()
        assert "access.granted" in captured.out

    def test_log_access_denied(self, audit_logger, capsys):
        """Test logging access denied events."""
        audit_logger.log_access(
            user_id="user123",
            resource="memory:mem_secret",
            action="read",
            granted=False,
            reason="insufficient permissions",
        )

        captured = capsys.readouterr()
        assert "access.denied" in captured.out

    def test_context_manager(self, audit_logger, capsys):
        """Test audit context manager."""
        with audit_logger.context(request_id="req_123", ip_address="192.168.1.1"):
            audit_logger.log_store(
                user_id="user123",
                memory_id="mem_abc",
                memory_type="semantic",
            )

        captured = capsys.readouterr()
        assert "req_123" in captured.out

    def test_redaction(self):
        """Test sensitive data redaction."""
        from mindcore.v2.enterprise import AuditConfig, AuditLogger

        config = AuditConfig(
            output="stdout",
            json_format=False,
            redact_patterns=["password", "secret"],
        )
        logger = AuditLogger(config)

        # The redaction happens internally
        data = {"password": "secret123", "user": "test"}
        redacted = logger._redact_sensitive(data)

        assert redacted["password"] == "[REDACTED]"  # noqa: S105
        assert redacted["user"] == "test"

    def test_file_output(self):
        """Test file-based audit logging."""
        from mindcore.v2.enterprise import AuditConfig, AuditLogger

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            config = AuditConfig(
                output="file",
                file_path=log_path,
                json_format=True,
                rotation_enabled=False,
            )
            logger = AuditLogger(config)

            logger.log_store(
                user_id="user123",
                memory_id="mem_abc",
                memory_type="semantic",
            )

            logger.close()

            with open(log_path) as f:
                content = f.read()
                assert "memory.store" in content
        finally:
            os.unlink(log_path)


class TestObservability:
    """Tests for observability module."""

    def test_metrics_import(self):
        """Test that metrics can be imported."""
        pytest.importorskip("opentelemetry")
        from mindcore.v2.enterprise import MindcoreMetrics, ObservabilityConfig

        config = ObservabilityConfig(
            service_name="test-service",
            console_export=False,  # Don't actually export
        )

        metrics = MindcoreMetrics(config)
        assert metrics is not None

    def test_tracer_import(self):
        """Test that tracer can be imported."""
        pytest.importorskip("opentelemetry")
        from mindcore.v2.enterprise import MindcoreTracer, ObservabilityConfig

        config = ObservabilityConfig(
            service_name="test-service",
            console_export=False,
        )

        tracer = MindcoreTracer(config)
        assert tracer is not None

    def test_trace_operation(self):
        """Test tracing an operation."""
        pytest.importorskip("opentelemetry")
        from mindcore.v2.enterprise import MindcoreTracer, ObservabilityConfig

        config = ObservabilityConfig(
            service_name="test-service",
            console_export=False,
        )
        tracer = MindcoreTracer(config)

        with tracer.trace_operation("test_op", user_id="user123") as span:
            span.set_attribute("test_attr", "value")

        # Should complete without error

    def test_config_attributes(self):
        """Test configuration resource attributes."""
        from mindcore.v2.enterprise import ObservabilityConfig

        config = ObservabilityConfig(
            service_name="my-service",
            service_version="1.0.0",
            environment="production",
            custom_attributes={"region": "us-east-1"},
        )

        attrs = config.get_resource_attributes()

        assert attrs["service.name"] == "my-service"
        assert attrs["service.version"] == "1.0.0"
        assert attrs["deployment.environment"] == "production"
        assert attrs["region"] == "us-east-1"


class TestEnterpriseImports:
    """Test that enterprise module exports are correct."""

    def test_all_exports(self):
        """Test that all expected classes are exported."""
        from mindcore.v2.enterprise import (
            # Audit
            AuditConfig,
            AuditEvent,
            AuditEventType,
            # Encryption
            EncryptionConfig,
            EncryptionError,
            MetricType,
            # Observability
            ObservabilityConfig,
            RateLimitBackend,
            # Rate Limiting
            RateLimitConfig,
            RateLimitExceeded,
            SpanKind,
        )

        # All imports should work
        assert ObservabilityConfig is not None
        assert RateLimitConfig is not None
        assert AuditConfig is not None
        assert EncryptionConfig is not None
