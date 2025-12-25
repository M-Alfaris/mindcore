"""Tests for GDPR/CCPA Compliance Tools."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindcore.v2 import Mindcore
from mindcore.v2.enterprise.compliance import (
    AnonymizationStrategy,
    ComplianceManager,
    RetentionPolicy,
)
from mindcore.v2.flr import Memory
from mindcore.v2.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    """Create a temporary SQLite storage for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    storage.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def compliance(storage):
    """Create ComplianceManager instance."""
    return ComplianceManager(storage)


@pytest.fixture
def sample_memories(storage):
    """Create sample memories for a user."""
    memories = []
    for i in range(5):
        memory = Memory(
            memory_id=f"mem_{i}",
            content=f"Test content {i} - user@example.com",
            memory_type="preference" if i < 2 else "episodic",
            user_id="user_123",
            topics=["test", f"topic_{i}"],
            importance=0.5 + (i * 0.1),
            entities=["John Doe", "user@example.com"],
        )
        storage.store(memory)
        memories.append(memory)
    return memories


class TestGDPRExport:
    """Test GDPR data export functionality."""

    def test_export_user_data(self, compliance, sample_memories):
        """Test exporting all user data."""
        result = compliance.export_user_data("user_123")

        assert result.user_id == "user_123"
        assert result.memory_count == 5
        assert len(result.memories) == 5
        assert result.export_id.startswith("export_")

    def test_export_nonexistent_user(self, compliance):
        """Test exporting data for user with no memories."""
        result = compliance.export_user_data("nonexistent")

        assert result.memory_count == 0
        assert len(result.memories) == 0

    def test_export_to_json(self, compliance, sample_memories):
        """Test exporting to JSON format."""
        result = compliance.export_user_data("user_123")
        json_output = result.to_json()

        assert "user_123" in json_output
        assert "memories" in json_output
        assert "export_info" in json_output

    def test_export_without_metadata(self, compliance, sample_memories):
        """Test export with minimal fields."""
        result = compliance.export_user_data(
            "user_123",
            include_metadata=False,
        )

        # Should only have essential fields
        first_memory = result.memories[0]
        assert "memory_id" in first_memory
        assert "content" in first_memory
        assert "memory_type" in first_memory
        assert "topics" not in first_memory

    def test_export_without_system_fields(self, compliance, sample_memories):
        """Test export without internal system fields."""
        result = compliance.export_user_data(
            "user_123",
            include_system_fields=False,
        )

        first_memory = result.memories[0]
        assert "reinforcement_score" not in first_memory
        assert "vocabulary_version" not in first_memory


class TestGDPRDelete:
    """Test GDPR data deletion functionality."""

    def test_delete_user_data(self, compliance, storage, sample_memories):
        """Test deleting all user data."""
        # Verify data exists
        assert len(storage.search(user_id="user_123")) == 5

        result = compliance.delete_user_data("user_123")

        assert result.user_id == "user_123"
        assert result.memories_deleted == 5
        assert result.deletion_id.startswith("delete_")
        assert result.verification_token != ""

        # Verify data is gone
        assert len(storage.search(user_id="user_123")) == 0

    def test_delete_nonexistent_user(self, compliance):
        """Test deleting data for user with no memories."""
        result = compliance.delete_user_data("nonexistent")

        assert result.memories_deleted == 0

    def test_delete_with_verification(self, compliance, sample_memories):
        """Test deletion with verification."""
        result = compliance.delete_user_data("user_123", verify=True)

        assert result.verification_token != ""
        assert len(result.verification_token) == 16


class TestAnonymization:
    """Test data anonymization functionality."""

    def test_pseudonymize(self, compliance, storage, sample_memories):
        """Test pseudonymization strategy."""
        result = compliance.anonymize_user_data(
            "user_123",
            strategy=AnonymizationStrategy.PSEUDONYMIZE,
        )

        assert result.memories_anonymized == 5
        assert result.anonymized_user_id.startswith("anon_")
        assert result.strategy == AnonymizationStrategy.PSEUDONYMIZE

        # Verify original user has no data
        assert len(storage.search(user_id="user_123")) == 0

        # Verify anonymized user has the data
        assert len(storage.search(user_id=result.anonymized_user_id)) == 5

    def test_hash_anonymization(self, compliance, storage, sample_memories):
        """Test hash-based anonymization."""
        result = compliance.anonymize_user_data(
            "user_123",
            strategy=AnonymizationStrategy.HASH,
        )

        assert result.memories_anonymized == 5
        assert result.strategy == AnonymizationStrategy.HASH

    def test_redact_pii(self, compliance, storage, sample_memories):
        """Test PII redaction strategy."""
        result = compliance.anonymize_user_data(
            "user_123",
            strategy=AnonymizationStrategy.REDACT,
        )

        assert result.memories_anonymized == 5

        # Verify PII is redacted
        anonymized = storage.search(user_id=result.anonymized_user_id)
        for memory in anonymized:
            assert "user@example.com" not in memory.content
            assert "[REDACTED]" in memory.content
            assert len(memory.entities) == 0

    def test_aggregate_anonymization(self, compliance, storage, sample_memories):
        """Test aggregation strategy."""
        result = compliance.anonymize_user_data(
            "user_123",
            strategy=AnonymizationStrategy.AGGREGATE,
        )

        assert result.memories_anonymized == 5

        # Verify content is aggregated
        anonymized = storage.search(user_id=result.anonymized_user_id)
        for memory in anonymized:
            assert "[AGGREGATED:" in memory.content


class TestRetentionPolicy:
    """Test retention policy functionality."""

    def test_retention_policy_config(self):
        """Test retention policy configuration."""
        policy = RetentionPolicy(
            memory_type_policies={
                "episodic": 730,
                "preference": None,
                "working": 1,
            },
            default_max_age_days=365,
        )

        assert policy.get_max_age("episodic") == 730
        assert policy.get_max_age("preference") is None
        assert policy.get_max_age("working") == 1
        assert policy.get_max_age("unknown") == 365  # Default

    def test_cutoff_date_calculation(self):
        """Test cutoff date calculation."""
        policy = RetentionPolicy(
            memory_type_policies={"working": 1},
            default_max_age_days=30,
        )

        cutoff = policy.get_cutoff_date("working")
        assert cutoff is not None

        # Should be approximately 1 day ago
        expected = datetime.now(timezone.utc) - timedelta(days=1)
        assert abs((cutoff - expected).total_seconds()) < 60

    def test_no_expiration(self):
        """Test memory types with no expiration."""
        policy = RetentionPolicy(
            memory_type_policies={"preference": None},
        )

        assert policy.get_cutoff_date("preference") is None


class TestRetentionEnforcement:
    """Test retention policy enforcement."""

    def test_enforce_retention(self, storage):
        """Test enforcing retention policy."""
        # Create old memories
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(3):
            memory = Memory(
                memory_id=f"old_{i}",
                content=f"Old content {i}",
                memory_type="working",
                user_id="user_123",
                created_at=old_date,
            )
            storage.store(memory)

        # Create recent memories
        for i in range(2):
            memory = Memory(
                memory_id=f"new_{i}",
                content=f"New content {i}",
                memory_type="working",
                user_id="user_123",
            )
            storage.store(memory)

        policy = RetentionPolicy(
            memory_type_policies={"working": 30},
        )
        compliance = ComplianceManager(storage, retention_policy=policy)

        result = compliance.enforce_retention()

        assert result.memories_deleted == 3
        assert result.memories_by_type.get("working") == 3

        # Verify old memories are gone
        remaining = storage.search(user_id="user_123")
        assert len(remaining) == 2

    def test_enforce_retention_dry_run(self, storage):
        """Test dry run of retention enforcement."""
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(3):
            memory = Memory(
                memory_id=f"old_{i}",
                content=f"Old content {i}",
                memory_type="working",
                user_id="user_123",
                created_at=old_date,
            )
            storage.store(memory)

        policy = RetentionPolicy(
            memory_type_policies={"working": 30},
        )
        compliance = ComplianceManager(storage, retention_policy=policy)

        result = compliance.enforce_retention(dry_run=True)

        assert result.memories_deleted == 3

        # Verify memories still exist (dry run)
        remaining = storage.search(user_id="user_123")
        assert len(remaining) == 3

    def test_no_policy_enforcement(self, storage):
        """Test enforcement with no policy configured."""
        compliance = ComplianceManager(storage)

        result = compliance.enforce_retention()

        assert result.memories_deleted == 0
        assert "No retention policy" in result.errors[0]


class TestUserDataSummary:
    """Test user data summary functionality."""

    def test_get_summary(self, compliance, sample_memories):
        """Test getting user data summary."""
        summary = compliance.get_user_data_summary("user_123")

        assert summary["user_id"] == "user_123"
        assert summary["total_memories"] == 5
        assert "memories_by_type" in summary
        assert "memories_by_topic" in summary
        assert summary["oldest_memory"] is not None

    def test_empty_user_summary(self, compliance):
        """Test summary for user with no data."""
        summary = compliance.get_user_data_summary("nonexistent")

        assert summary["total_memories"] == 0


class TestRetentionStatus:
    """Test retention status checking."""

    def test_check_status(self, storage):
        """Test checking retention status."""
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(5):
            memory = Memory(
                memory_id=f"old_{i}",
                content=f"Content {i}",
                memory_type="working",
                user_id="user_123",
                created_at=old_date,
            )
            storage.store(memory)

        policy = RetentionPolicy(
            memory_type_policies={"working": 30},
        )
        compliance = ComplianceManager(storage, retention_policy=policy)

        status = compliance.check_retention_status()

        assert status["status"] == "ok"
        assert status["memories_affected"] == 5


class TestComplianceEvents:
    """Test compliance event callbacks."""

    def test_event_callback(self, storage, sample_memories):
        """Test that events are emitted."""
        events = []

        def on_event(event_type, subject, data):
            events.append((event_type, subject, data))

        compliance = ComplianceManager(storage, on_event=on_event)

        compliance.export_user_data("user_123")
        compliance.delete_user_data("user_123")

        assert len(events) >= 2
        event_types = [e[0].value for e in events]
        assert "data_export" in event_types
        assert "data_delete" in event_types


class TestMindcoreIntegration:
    """Test Mindcore integration with compliance features."""

    def test_gdpr_export_via_mindcore(self):
        """Test GDPR export through Mindcore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            # Store some memories
            for i in range(3):
                mc.store(
                    content=f"Test memory {i}",
                    memory_type="preference",
                    user_id="user_123",
                )

            # Export
            export = mc.gdpr_export("user_123")

            assert export["export_info"]["memory_count"] == 3
            assert len(export["memories"]) == 3

            mc.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_gdpr_delete_via_mindcore(self):
        """Test GDPR delete through Mindcore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            # Store memories
            for i in range(3):
                mc.store(
                    content=f"Test memory {i}",
                    memory_type="preference",
                    user_id="user_123",
                )

            # Delete
            result = mc.gdpr_delete("user_123")

            assert result["memories_deleted"] == 3

            # Verify deletion
            search_result = mc.search(user_id="user_123")
            assert len(search_result) == 0

            mc.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_gdpr_anonymize_via_mindcore(self):
        """Test GDPR anonymize through Mindcore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            # Store memories
            for i in range(3):
                mc.store(
                    content=f"Test memory {i}",
                    memory_type="preference",
                    user_id="user_123",
                )

            # Anonymize
            result = mc.gdpr_anonymize("user_123", strategy="pseudonymize")

            assert result["memories_anonymized"] == 3
            assert result["anonymized_user_id"].startswith("anon_")

            mc.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_retention_policy_via_mindcore(self):
        """Test retention policy through Mindcore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(
                storage=f"sqlite:///{db_path}",
                retention_policy={
                    "working": {"max_age_days": 1},
                    "preference": {"max_age_days": None},  # Forever
                    "default_max_age_days": 30,
                },
            )

            # Store memories
            mc.store(
                content="Working memory",
                memory_type="working",
                user_id="user_123",
            )
            mc.store(
                content="Preference",
                memory_type="preference",
                user_id="user_123",
            )

            # Check retention status
            status = mc.get_retention_status()

            assert status["status"] == "ok"
            assert "policy" in status

            mc.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_user_data_summary_via_mindcore(self):
        """Test user data summary through Mindcore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mc = Mindcore(storage=f"sqlite:///{db_path}")

            # Store memories (using valid topics from default vocabulary)
            for i in range(5):
                mc.store(
                    content=f"Test memory {i}",
                    memory_type="preference" if i < 2 else "episodic",
                    user_id="user_123",
                    topics=["order"],  # Valid topic from default vocabulary
                )

            summary = mc.get_user_data_summary("user_123")

            assert summary["total_memories"] == 5
            assert summary["memories_by_type"]["preference"] == 2
            assert summary["memories_by_type"]["episodic"] == 3

            mc.close()
        finally:
            Path(db_path).unlink(missing_ok=True)
