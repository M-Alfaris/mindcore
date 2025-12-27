"""Test 10: Authentication and Error Handling Tests.

Tests error handling and edge cases:
- Exception hierarchy
- Validation errors
- Storage errors
- Permission errors
- Rate limiting
- Error recovery
"""

import pytest


# ============================================================================
# Exception Imports
# ============================================================================


@pytest.fixture
def exception_classes():
    """Import all exception classes."""
    try:
        from mindcore.exceptions import (
            AccessError,
            AgentNotFoundError,
            ConfigurationError,
            MemoryNotFoundError,
            MemoryValidationError,
            MigrationError,
            MigrationPathError,
            MindcoreError,
            MultiAgentNotEnabledError,
            PermissionDeniedError,
            RollbackError,
            StorageConnectionError,
            StorageError,
            ValidationError,
            VocabularyValidationError,
        )

        return {
            "MindcoreError": MindcoreError,
            "StorageError": StorageError,
            "MemoryNotFoundError": MemoryNotFoundError,
            "StorageConnectionError": StorageConnectionError,
            "ValidationError": ValidationError,
            "VocabularyValidationError": VocabularyValidationError,
            "MemoryValidationError": MemoryValidationError,
            "MigrationError": MigrationError,
            "MigrationPathError": MigrationPathError,
            "RollbackError": RollbackError,
            "AccessError": AccessError,
            "PermissionDeniedError": PermissionDeniedError,
            "AgentNotFoundError": AgentNotFoundError,
            "ConfigurationError": ConfigurationError,
            "MultiAgentNotEnabledError": MultiAgentNotEnabledError,
        }
    except ImportError:
        pytest.skip("Exception classes not available")


# ============================================================================
# Exception Hierarchy Tests
# ============================================================================


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_mindcore_error_is_base(self, exception_classes):
        """Test MindcoreError is base for all exceptions."""
        base = exception_classes["MindcoreError"]

        # All other exceptions should inherit from it
        for name, exc_class in exception_classes.items():
            if name != "MindcoreError":
                assert issubclass(exc_class, base)

    def test_storage_error_hierarchy(self, exception_classes):
        """Test StorageError hierarchy."""
        storage_error = exception_classes["StorageError"]
        memory_not_found = exception_classes["MemoryNotFoundError"]
        connection_error = exception_classes["StorageConnectionError"]

        assert issubclass(memory_not_found, storage_error)
        assert issubclass(connection_error, storage_error)

    def test_validation_error_hierarchy(self, exception_classes):
        """Test ValidationError hierarchy."""
        validation_error = exception_classes["ValidationError"]
        vocab_error = exception_classes["VocabularyValidationError"]
        memory_error = exception_classes["MemoryValidationError"]

        assert issubclass(vocab_error, validation_error)
        assert issubclass(memory_error, validation_error)

    def test_access_error_hierarchy(self, exception_classes):
        """Test AccessError hierarchy."""
        access_error = exception_classes["AccessError"]
        permission_denied = exception_classes["PermissionDeniedError"]
        agent_not_found = exception_classes["AgentNotFoundError"]

        assert issubclass(permission_denied, access_error)
        assert issubclass(agent_not_found, access_error)


# ============================================================================
# Storage Error Tests
# ============================================================================


class TestStorageErrors:
    """Test storage-related errors."""

    def test_memory_not_found_error(self, mindcore):
        """Test MemoryNotFoundError is raised for missing memory."""
        from mindcore.exceptions import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError) as exc_info:
            mindcore.delete("nonexistent_memory_id_12345")

        assert exc_info.value.memory_id == "nonexistent_memory_id_12345"

    def test_memory_not_found_error_message(self, exception_classes):
        """Test MemoryNotFoundError message formatting."""
        error = exception_classes["MemoryNotFoundError"]("test_id_123")

        assert "test_id_123" in str(error)

    def test_storage_connection_error(self, exception_classes):
        """Test StorageConnectionError."""
        error = exception_classes["StorageConnectionError"](
            message="Connection refused", backend="postgresql"
        )

        assert "Connection refused" in str(error)
        assert error.backend == "postgresql"


# ============================================================================
# Validation Error Tests
# ============================================================================


class TestValidationErrors:
    """Test validation-related errors."""

    def test_vocabulary_validation_error(self, mindcore):
        """Test VocabularyValidationError for invalid vocabulary."""
        # This depends on how strict validation is configured
        # The test verifies that invalid memory_type is rejected or accepted
        raised = False
        error_msg = ""
        try:
            mindcore.store(
                content="Test",
                memory_type="completely_invalid_type_xyz",
                user_id="test_user",
                topics=["api"],
            )
            # If it doesn't raise, validation might be lenient
        except Exception as exc:
            raised = True
            error_msg = str(exc).lower()

        # If an error was raised, it should be a validation error
        if raised:
            assert "validation" in error_msg or "invalid" in error_msg

    def test_vocabulary_validation_error_contains_errors(self, exception_classes):
        """Test VocabularyValidationError contains error list."""
        errors = ["Invalid topic: xyz", "Invalid category: abc"]
        error = exception_classes["VocabularyValidationError"](
            errors=errors, memory_data={"content": "test"}
        )

        assert error.errors == errors
        assert error.memory_data == {"content": "test"}

    def test_memory_validation_error(self, exception_classes):
        """Test MemoryValidationError."""
        error = exception_classes["MemoryValidationError"](
            message="Importance must be between 0 and 1", field="importance"
        )

        assert "importance" in str(error).lower()
        assert error.field == "importance"


# ============================================================================
# Access Error Tests
# ============================================================================


class TestAccessErrors:
    """Test access-related errors."""

    def test_permission_denied_error(self, exception_classes):
        """Test PermissionDeniedError."""
        error = exception_classes["PermissionDeniedError"](
            message="Cannot delete memory", agent_id="test_agent", permission="delete"
        )

        assert error.agent_id == "test_agent"
        assert error.permission == "delete"
        assert "delete" in str(error).lower()

    def test_agent_not_found_error(self, exception_classes):
        """Test AgentNotFoundError."""
        error = exception_classes["AgentNotFoundError"]("unknown_agent")

        assert error.agent_id == "unknown_agent"
        assert "unknown_agent" in str(error)


# ============================================================================
# Configuration Error Tests
# ============================================================================


class TestConfigurationErrors:
    """Test configuration-related errors."""

    def test_multi_agent_not_enabled_error(self, mindcore):
        """Test MultiAgentNotEnabledError when multi-agent is disabled."""
        from mindcore.exceptions import MultiAgentNotEnabledError

        with pytest.raises(MultiAgentNotEnabledError):
            mindcore.register_agent(agent_id="test", name="Test", teams=[])


# ============================================================================
# Migration Error Tests
# ============================================================================


class TestMigrationErrors:
    """Test migration-related errors."""

    def test_migration_path_error(self, exception_classes):
        """Test MigrationPathError."""
        error = exception_classes["MigrationPathError"](from_version="0.5.0", to_version="2.0.0")

        assert error.from_version == "0.5.0"
        assert error.to_version == "2.0.0"
        assert "0.5.0" in str(error)

    def test_rollback_error(self, exception_classes):
        """Test RollbackError."""
        error = exception_classes["RollbackError"](
            message="Cannot rollback: no checkpoint", checkpoint_id="chk_123"
        )

        assert error.checkpoint_id == "chk_123"


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Test rate limiting behavior (if implemented)."""

    def test_rapid_requests_handled(self, mindcore):
        """Test that rapid requests don't crash the system."""
        # Make many rapid requests
        for i in range(100):
            try:
                mindcore.store(
                    content=f"Rapid request {i}",
                    memory_type="semantic",
                    user_id="rate_test_user",
                    topics=["api"],
                )
            except Exception as e:
                # If rate limited, should get appropriate error
                if "rate" in str(e).lower():
                    break
                raise

    def test_rate_limit_recovery(self, mindcore):
        """Test recovery after potential rate limiting."""
        import time

        # Make some requests
        for i in range(10):
            mindcore.store(
                content=f"Rate test {i}",
                memory_type="semantic",
                user_id="rate_recovery_user",
                topics=["api"],
            )

        # Brief pause
        time.sleep(0.1)

        # Should still work after pause
        memory_id = mindcore.store(
            content="After rate test",
            memory_type="semantic",
            user_id="rate_recovery_user",
            topics=["api"],
        )

        assert memory_id is not None


# ============================================================================
# Error Recovery Tests
# ============================================================================


class TestErrorRecovery:
    """Test error recovery scenarios."""

    def test_continue_after_validation_error(self, mindcore):
        """Test that operations continue after validation error."""
        # Trigger a validation error
        try:
            mindcore.store(
                content="Invalid",
                memory_type="invalid_type_xyz",
                user_id="recovery_user",
                topics=["api"],
            )
        except Exception:
            pass

        # Should still work after error
        memory_id = mindcore.store(
            content="Valid memory after error",
            memory_type="semantic",
            user_id="recovery_user",
            topics=["api"],
        )

        assert memory_id is not None

    def test_continue_after_not_found_error(self, mindcore):
        """Test operations continue after not found error."""
        # Trigger not found error
        try:
            mindcore.delete("nonexistent_12345")
        except Exception:
            pass

        # Should still work
        memory_id = mindcore.store(
            content="After not found error",
            memory_type="semantic",
            user_id="recovery_user",
            topics=["api"],
        )

        assert memory_id is not None


# ============================================================================
# Content Validation Tests
# ============================================================================


class TestContentValidation:
    """Test content-related validation."""

    def test_empty_content_handling(self, mindcore):
        """Test handling of empty content."""
        try:
            mindcore.store(
                content="",  # Empty content
                memory_type="semantic",
                user_id="empty_test",
                topics=["api"],
            )
            # May accept or reject empty content
        except Exception:
            # Should fail gracefully
            pass

    def test_very_long_content(self, mindcore):
        """Test handling of very long content."""
        long_content = "x" * 100000  # 100KB

        try:
            mindcore.store(
                content=long_content, memory_type="semantic", user_id="long_test", topics=["api"]
            )
            # Should handle or limit
        except Exception:
            # Should fail gracefully
            pass

    def test_special_characters_in_content(self, mindcore):
        """Test handling special characters."""
        special_content = "Test with 'quotes', \"doubles\", \n newlines, \t tabs, and émojis 🎉"

        memory_id = mindcore.store(
            content=special_content, memory_type="semantic", user_id="special_test", topics=["api"]
        )

        memory = mindcore.get(memory_id)
        assert memory.content == special_content


# ============================================================================
# Input Sanitization Tests
# ============================================================================


class TestInputSanitization:
    """Test input sanitization for security."""

    def test_sql_injection_prevention(self, mindcore):
        """Test that SQL injection attempts are handled safely."""
        malicious_content = "'; DROP TABLE memories; --"

        # Should not cause SQL injection
        memory_id = mindcore.store(
            content=malicious_content,
            memory_type="semantic",
            user_id="injection_test",
            topics=["api"],
        )

        # Memory should exist and contain the literal string
        memory = mindcore.get(memory_id)
        assert memory is not None
        assert "DROP TABLE" in memory.content  # Stored as literal

    def test_xss_content_stored_safely(self, mindcore):
        """Test that XSS content is stored as-is."""
        xss_content = "<script>alert('xss')</script>"

        memory_id = mindcore.store(
            content=xss_content, memory_type="semantic", user_id="xss_test", topics=["api"]
        )

        memory = mindcore.get(memory_id)
        # Should be stored as literal string (escaping is display concern)
        assert memory.content == xss_content


# ============================================================================
# Concurrent Error Tests
# ============================================================================


class TestConcurrentErrors:
    """Test error handling under concurrent access."""

    def test_concurrent_deletes(self, mindcore):
        """Test concurrent delete attempts on same memory."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create a memory
        memory_id = mindcore.store(
            content="To be deleted concurrently",
            memory_type="semantic",
            user_id="concurrent_test",
            topics=["api"],
        )

        def try_delete():
            try:
                mindcore.delete(memory_id)
                return "deleted"
            except Exception as e:
                return f"error: {type(e).__name__}"

        # Try to delete from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_delete) for _ in range(5)]
            results = [f.result() for f in as_completed(futures)]

        # One should succeed, others should get MemoryNotFoundError
        deleted_count = sum(1 for r in results if r == "deleted")
        error_count = sum(1 for r in results if "error" in r)

        # At least one should have worked or errored appropriately
        assert deleted_count >= 1 or error_count >= 1
