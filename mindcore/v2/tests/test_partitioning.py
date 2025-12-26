"""Tests for Storage Partitioning - Time-based table partitioning.

Tests cover:
- PartitionInterval enum
- PartitionInfo and PartitioningStatus dataclasses
- PartitionManager: setup, creation, archival, status

Uses mocking since actual PostgreSQL is not available in test environment.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from mindcore.v2.storage.partitioning import (
    PartitionInfo,
    PartitioningStatus,
    PartitionInterval,
    PartitionManager,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockCursor:
    """Mock psycopg cursor."""

    def __init__(self):
        self._results = []
        self._result_index = 0
        self.rowcount = 1

    def execute(self, sql, params=None):
        return self

    def fetchone(self):
        if self._results and self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return None

    def fetchall(self):
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnection:
    """Mock psycopg connection."""

    def __init__(self):
        self._cursor = MockCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnectionPool:
    """Mock psycopg_pool ConnectionPool."""

    def __init__(self):
        self._connection = MockConnection()

    def connection(self):
        return self

    def __enter__(self):
        return self._connection

    def __exit__(self, *args):
        pass


class MockPostgresStorage:
    """Mock PostgresStorage for testing PartitionManager."""

    def __init__(self):
        self._pool = MockConnectionPool()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create mocked PostgresStorage."""
    return MockPostgresStorage()


@pytest.fixture
def partition_manager(mock_storage):
    """Create PartitionManager with mocked storage."""
    return PartitionManager(mock_storage)


# =============================================================================
# PartitionInterval Enum Tests
# =============================================================================


class TestPartitionInterval:
    """Tests for PartitionInterval enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert PartitionInterval.DAILY.value == "daily"
        assert PartitionInterval.WEEKLY.value == "weekly"
        assert PartitionInterval.MONTHLY.value == "monthly"
        assert PartitionInterval.QUARTERLY.value == "quarterly"
        assert PartitionInterval.YEARLY.value == "yearly"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert PartitionInterval("daily") == PartitionInterval.DAILY
        assert PartitionInterval("monthly") == PartitionInterval.MONTHLY


# =============================================================================
# PartitionInfo Tests
# =============================================================================


class TestPartitionInfo:
    """Tests for PartitionInfo dataclass."""

    def test_create_partition_info(self):
        """Test creating partition info."""
        now = datetime.now(timezone.utc)
        info = PartitionInfo(
            name="memories_p2025_01",
            start_date=now,
            end_date=now + timedelta(days=30),
            row_count=1000,
            size_bytes=1024 * 1024,
            size_pretty="1 MB",
        )

        assert info.name == "memories_p2025_01"
        assert info.row_count == 1000
        assert info.size_pretty == "1 MB"


# =============================================================================
# PartitioningStatus Tests
# =============================================================================


class TestPartitioningStatus:
    """Tests for PartitioningStatus dataclass."""

    def test_create_status_not_partitioned(self):
        """Test creating status for non-partitioned table."""
        status = PartitioningStatus(
            is_partitioned=False,
            interval=None,
            partitions=[],
            total_partitions=0,
            total_rows=0,
            total_size_bytes=0,
            total_size_pretty="0 bytes",
            oldest_partition=None,
            newest_partition=None,
        )

        assert status.is_partitioned is False
        assert status.interval is None
        assert status.total_partitions == 0

    def test_create_status_partitioned(self):
        """Test creating status for partitioned table."""
        now = datetime.now(timezone.utc)
        partition = PartitionInfo(
            name="memories_p2025_01",
            start_date=now,
            end_date=now + timedelta(days=30),
            row_count=1000,
            size_bytes=1024 * 1024,
            size_pretty="1 MB",
        )

        status = PartitioningStatus(
            is_partitioned=True,
            interval=PartitionInterval.MONTHLY,
            partitions=[partition],
            total_partitions=1,
            total_rows=1000,
            total_size_bytes=1024 * 1024,
            total_size_pretty="1 MB",
            oldest_partition="memories_p2025_01",
            newest_partition="memories_p2025_01",
        )

        assert status.is_partitioned is True
        assert status.interval == PartitionInterval.MONTHLY
        assert status.total_partitions == 1
        assert len(status.partitions) == 1


# =============================================================================
# PartitionManager Initialization Tests
# =============================================================================


class TestPartitionManagerInit:
    """Tests for PartitionManager initialization."""

    def test_init_stores_pool(self, partition_manager, mock_storage):
        """Test initialization stores pool reference."""
        assert partition_manager._pool is mock_storage._pool

    def test_partition_prefix(self, partition_manager):
        """Test partition prefix constant."""
        assert partition_manager.PARTITION_PREFIX == "memories_p"


# =============================================================================
# Setup Partitioning Tests
# =============================================================================


class TestSetupPartitioning:
    """Tests for setup_partitioning method."""

    def test_setup_already_partitioned(self, partition_manager, mock_storage):
        """Test setup when table is already partitioned."""
        # Mock returns 'p' for partitioned table
        mock_storage._pool._connection._cursor._results = [("p",)]

        result = partition_manager.setup_partitioning()

        assert result is True

    def test_setup_with_string_interval(self, partition_manager, mock_storage):
        """Test setup with string interval."""
        # First return None (not partitioned), then succeed
        mock_storage._pool._connection._cursor._results = [(None,)]

        result = partition_manager.setup_partitioning(interval="monthly")

        assert result is True

    def test_setup_with_enum_interval(self, partition_manager, mock_storage):
        """Test setup with PartitionInterval enum."""
        mock_storage._pool._connection._cursor._results = [(None,)]

        result = partition_manager.setup_partitioning(interval=PartitionInterval.WEEKLY)

        assert result is True

    def test_setup_migrate_existing(self, partition_manager, mock_storage):
        """Test setup with migration."""
        mock_storage._pool._connection._cursor._results = [(None,)]

        result = partition_manager.setup_partitioning(
            interval=PartitionInterval.MONTHLY,
            migrate_existing=True,
        )

        assert result is True


# =============================================================================
# Create Future Partitions Tests
# =============================================================================


class TestCreateFuturePartitions:
    """Tests for create_future_partitions method."""

    def test_create_monthly_partitions(self, partition_manager, mock_storage):
        """Test creating monthly partitions."""
        # Mock _get_interval to return MONTHLY
        partition_manager._get_interval = lambda: PartitionInterval.MONTHLY

        created = partition_manager.create_future_partitions(months_ahead=3)

        # Should attempt to create partitions
        assert isinstance(created, list)

    def test_create_weekly_partitions(self, partition_manager, mock_storage):
        """Test creating weekly partitions."""
        partition_manager._get_interval = lambda: PartitionInterval.WEEKLY

        created = partition_manager.create_future_partitions(months_ahead=2)

        assert isinstance(created, list)

    def test_create_daily_partitions(self, partition_manager, mock_storage):
        """Test creating daily partitions."""
        partition_manager._get_interval = lambda: PartitionInterval.DAILY

        created = partition_manager.create_future_partitions(months_ahead=1)

        assert isinstance(created, list)

    def test_create_quarterly_partitions(self, partition_manager, mock_storage):
        """Test creating quarterly partitions."""
        partition_manager._get_interval = lambda: PartitionInterval.QUARTERLY

        created = partition_manager.create_future_partitions(months_ahead=2)

        assert isinstance(created, list)

    def test_create_yearly_partitions(self, partition_manager, mock_storage):
        """Test creating yearly partitions."""
        partition_manager._get_interval = lambda: PartitionInterval.YEARLY

        created = partition_manager.create_future_partitions(months_ahead=1)

        assert isinstance(created, list)

    def test_create_partitions_no_interval(self, partition_manager, mock_storage):
        """Test creating partitions when not partitioned."""
        partition_manager._get_interval = lambda: None

        created = partition_manager.create_future_partitions(months_ahead=3)

        assert created == []


# =============================================================================
# Archive Partitions Tests
# =============================================================================


class TestArchivePartitions:
    """Tests for archive_partitions method."""

    def test_archive_old_partitions(self, partition_manager, mock_storage):
        """Test archiving old partitions."""
        # Mock partition list
        old_date = datetime.now(timezone.utc) - timedelta(days=400)
        bound_expr = f"FOR VALUES FROM ('{old_date.isoformat()}') TO ('{(old_date + timedelta(days=30)).isoformat()}')"
        mock_storage._pool._connection._cursor._results = [
            ("memories_p2024_01", bound_expr),
        ]

        archived = partition_manager.archive_partitions(
            older_than_months=12,
            detach_only=True,
        )

        assert isinstance(archived, list)

    def test_archive_drop_partitions(self, partition_manager, mock_storage):
        """Test dropping old partitions."""
        old_date = datetime.now(timezone.utc) - timedelta(days=400)
        bound_expr = f"FOR VALUES FROM ('{old_date.isoformat()}') TO ('{(old_date + timedelta(days=30)).isoformat()}')"
        mock_storage._pool._connection._cursor._results = [
            ("memories_p2024_01", bound_expr),
        ]

        archived = partition_manager.archive_partitions(
            older_than_months=12,
            detach_only=False,
        )

        assert isinstance(archived, list)

    def test_archive_invalid_bound_format(self, partition_manager, mock_storage):
        """Test archiving with invalid bound format is skipped."""
        mock_storage._pool._connection._cursor._results = [
            ("memories_p2024_01", "INVALID BOUND FORMAT"),
        ]

        archived = partition_manager.archive_partitions(older_than_months=12)

        # Should skip invalid partitions
        assert archived == []


# =============================================================================
# Get Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_not_partitioned(self, partition_manager, mock_storage):
        """Test getting status for non-partitioned table."""
        mock_storage._pool._connection._cursor._results = [("r",)]  # regular table

        status = partition_manager.get_status()

        assert status.is_partitioned is False
        assert status.interval is None

    def test_get_status_partitioned(self, partition_manager, mock_storage):
        """Test getting status for partitioned table."""
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)

        bound_expr = f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"

        # Setup mock to return partitioned table and partition info
        call_count = [0]

        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return ("p",)  # partitioned
            if call_count[0] == 2:
                return ("monthly",)  # interval from _get_interval
            return ("1 MB",)  # total size pretty

        def mock_fetchall():
            return [
                (
                    "memories_p2025_01",
                    bound_expr,
                    1000,
                    1024 * 1024,
                    "1 MB",
                ),
            ]

        mock_storage._pool._connection._cursor.fetchone = mock_fetchone
        mock_storage._pool._connection._cursor.fetchall = mock_fetchall

        status = partition_manager.get_status()

        assert status.is_partitioned is True


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_interval_exists(self, partition_manager, mock_storage):
        """Test getting interval when configured."""
        mock_storage._pool._connection._cursor._results = [("monthly",)]

        interval = partition_manager._get_interval()

        assert interval == PartitionInterval.MONTHLY

    def test_get_interval_not_exists(self, partition_manager, mock_storage):
        """Test getting interval when not configured."""
        mock_storage._pool._connection._cursor._results = []

        interval = partition_manager._get_interval()

        assert interval is None

    def test_get_interval_exception(self, partition_manager, mock_storage):
        """Test getting interval handles exceptions."""

        def raise_error():
            raise RuntimeError("Database error")

        mock_storage._pool.connection = raise_error

        interval = partition_manager._get_interval()

        assert interval is None


# =============================================================================
# Partition Creation Helper Tests
# =============================================================================


class TestPartitionCreationHelpers:
    """Tests for partition creation helper methods."""

    def test_create_monthly_partition(self, partition_manager, mock_storage):
        """Test creating monthly partition."""
        target = datetime(2025, 6, 15, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_monthly_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025_06"

    def test_create_monthly_partition_december(self, partition_manager, mock_storage):
        """Test creating December partition (edge case)."""
        target = datetime(2025, 12, 15, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_monthly_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025_12"

    def test_create_weekly_partition(self, partition_manager, mock_storage):
        """Test creating weekly partition."""
        target = datetime(2025, 6, 15, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_weekly_partition(
                conn.cursor(),
                target,
            )

        assert result is not None
        assert result.startswith("memories_p2025_w")

    def test_create_daily_partition(self, partition_manager, mock_storage):
        """Test creating daily partition."""
        target = datetime(2025, 6, 15, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_daily_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025_06_15"

    def test_create_quarterly_partition(self, partition_manager, mock_storage):
        """Test creating quarterly partition."""
        target = datetime(2025, 6, 15, tzinfo=timezone.utc)  # Q2

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_quarterly_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025_q2"

    def test_create_quarterly_partition_q4(self, partition_manager, mock_storage):
        """Test creating Q4 partition (edge case)."""
        target = datetime(2025, 11, 15, tzinfo=timezone.utc)  # Q4

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_quarterly_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025_q4"

    def test_create_yearly_partition(self, partition_manager, mock_storage):
        """Test creating yearly partition."""
        target = datetime(2025, 6, 15, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_yearly_partition(
                conn.cursor(),
                target,
            )

        assert result == "memories_p2025"

    def test_create_partition_base(self, partition_manager, mock_storage):
        """Test base partition creation."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 1, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_partition(
                conn.cursor(),
                "memories_p2025_01",
                start,
                end,
            )

        assert result == "memories_p2025_01"

    def test_create_partition_handles_exception(self, partition_manager, mock_storage):
        """Test partition creation handles exceptions."""

        def raise_error(*args, **kwargs):
            raise RuntimeError("Partition exists")

        mock_storage._pool._connection._cursor.execute = raise_error

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 1, tzinfo=timezone.utc)

        with mock_storage._pool.connection() as conn:
            result = partition_manager._create_partition(
                conn.cursor(),
                "memories_p2025_01",
                start,
                end,
            )

        # Should return None on error
        assert result is None


# =============================================================================
# Partition Range Creation Tests
# =============================================================================


class TestPartitionRangeCreation:
    """Tests for _create_partition_range method."""

    def test_create_partition_range_monthly(self, partition_manager, mock_storage):
        """Test creating range of monthly partitions."""
        with mock_storage._pool.connection() as conn:
            partition_manager._create_partition_range(
                conn.cursor(),
                PartitionInterval.MONTHLY,
                months_back=1,
                months_ahead=1,
            )

        # Should not raise

    def test_create_partition_range_weekly(self, partition_manager, mock_storage):
        """Test creating range of weekly partitions."""
        with mock_storage._pool.connection() as conn:
            partition_manager._create_partition_range(
                conn.cursor(),
                PartitionInterval.WEEKLY,
                months_back=0,
                months_ahead=1,
            )

    def test_create_partition_range_daily(self, partition_manager, mock_storage):
        """Test creating range of daily partitions."""
        with mock_storage._pool.connection() as conn:
            partition_manager._create_partition_range(
                conn.cursor(),
                PartitionInterval.DAILY,
                months_back=0,
                months_ahead=1,
            )


# =============================================================================
# Index Creation Tests
# =============================================================================


class TestIndexCreation:
    """Tests for _create_partition_indexes method."""

    def test_create_partition_indexes(self, partition_manager, mock_storage):
        """Test creating indexes on partitioned table."""
        with mock_storage._pool.connection() as conn:
            partition_manager._create_partition_indexes(conn.cursor())

        # Should not raise

    def test_create_partition_indexes_handles_existing(self, partition_manager, mock_storage):
        """Test creating indexes handles existing indexes."""
        call_count = [0]

        def execute_with_error(sql, *args):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("Index already exists")

        mock_storage._pool._connection._cursor.execute = execute_with_error

        with mock_storage._pool.connection() as conn:
            # Should not raise, errors are caught
            partition_manager._create_partition_indexes(conn.cursor())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
