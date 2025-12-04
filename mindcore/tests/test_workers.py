"""Tests for distributed workers module."""

import pytest

from mindcore.workers.celery_app import CeleryConfig, is_celery_available
from mindcore.workers.enrichment_worker import EnrichmentTask
from mindcore.workers.worker_manager import WorkerBackend, WorkerConfig


class TestCeleryConfig:
    """Tests for CeleryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CeleryConfig()

        assert config.broker_url == "redis://localhost:6379/0"
        assert config.task_default_queue == "mindcore"
        assert config.task_acks_late is True
        assert config.task_time_limit == 300

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CeleryConfig(
            broker_url="redis://redis.example.com:6379/0",
            result_backend="redis://redis.example.com:6379/1",
            task_default_queue="custom_queue",
            worker_concurrency=8,
        )

        assert config.broker_url == "redis://redis.example.com:6379/0"
        assert config.task_default_queue == "custom_queue"
        assert config.worker_concurrency == 8

    def test_to_celery_config(self):
        """Test conversion to Celery config dict."""
        config = CeleryConfig(
            broker_url="redis://localhost:6379/0",
            task_time_limit=600,
        )

        celery_dict = config.to_celery_config()

        assert celery_dict["broker_url"] == "redis://localhost:6379/0"
        assert celery_dict["task_time_limit"] == 600
        assert celery_dict["task_acks_late"] is True


class TestEnrichmentTask:
    """Tests for EnrichmentTask."""

    def test_default_values(self):
        """Test default task values."""
        task = EnrichmentTask(
            message_id="msg123",
            user_id="user456",
            thread_id="thread789",
            session_id="session123",
            role="user",
            text="Hello, how are you?",
        )

        assert task.message_id == "msg123"
        assert task.priority == 5
        assert task.retry_count == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        task = EnrichmentTask(
            message_id="msg123",
            user_id="user456",
            thread_id="thread789",
            session_id="session123",
            role="user",
            text="Hello",
            priority=3,
        )

        data = task.to_dict()

        assert data["message_id"] == "msg123"
        assert data["priority"] == 3
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "message_id": "msg123",
            "user_id": "user456",
            "thread_id": "thread789",
            "session_id": "session123",
            "role": "user",
            "text": "Hello",
            "priority": 2,
            "retry_count": 1,
        }

        task = EnrichmentTask.from_dict(data)

        assert task.message_id == "msg123"
        assert task.priority == 2
        assert task.retry_count == 1


class TestWorkerConfig:
    """Tests for WorkerConfig."""

    def test_default_backend(self):
        """Test default backend is thread pool."""
        config = WorkerConfig()

        assert config.backend == WorkerBackend.THREAD
        assert config.workers == 4

    def test_celery_config(self):
        """Test Celery backend configuration."""
        config = WorkerConfig(
            backend=WorkerBackend.CELERY,
            celery_broker="redis://localhost:6379/0",
            celery_result_backend="redis://localhost:6379/1",
        )

        assert config.backend == WorkerBackend.CELERY
        assert config.celery_broker == "redis://localhost:6379/0"

    def test_to_dict(self):
        """Test serialization to dict."""
        config = WorkerConfig(
            backend=WorkerBackend.THREAD,
            workers=8,
        )

        data = config.to_dict()

        assert data["backend"] == "thread"
        assert data["workers"] == 8


class TestWorkerBackend:
    """Tests for WorkerBackend enum."""

    def test_values(self):
        """Test enum values."""
        assert WorkerBackend.THREAD.value == "thread"
        assert WorkerBackend.CELERY.value == "celery"


class TestCeleryAvailability:
    """Tests for Celery availability check."""

    def test_is_celery_available(self):
        """Test Celery availability check."""
        # This will return True if celery is installed, False otherwise
        result = is_celery_available()
        assert isinstance(result, bool)


class TestWorkerManagerThreadBackend:
    """Tests for WorkerManager with thread backend."""

    def test_thread_backend_status(self):
        """Test getting status with thread backend."""
        from unittest.mock import MagicMock

        # Create mock client
        mock_client = MagicMock()
        mock_client.enrichment_worker_count = 4
        mock_client.get_worker_health.return_value = {"healthy": True}
        mock_client._enrichment_queue.qsize.return_value = 10

        from mindcore.workers import WorkerConfig, WorkerManager

        config = WorkerConfig(backend=WorkerBackend.THREAD, workers=4)

        manager = WorkerManager.__new__(WorkerManager)
        manager.client = mock_client
        manager.config = config
        manager._initialized = True
        manager._celery_worker = None

        status = manager.get_status()

        assert status["backend"] == "thread"
        assert status["queue_depth"] == 10

    def test_submit_to_thread_backend(self):
        """Test submitting task to thread backend."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_queue = MagicMock()
        mock_client._enrichment_queue = mock_queue

        from mindcore.workers import WorkerConfig, WorkerManager

        config = WorkerConfig(backend=WorkerBackend.THREAD)

        manager = WorkerManager.__new__(WorkerManager)
        manager.client = mock_client
        manager.config = config
        manager._initialized = True
        manager._celery_worker = None

        task_data = {
            "message_id": "msg123",
            "user_id": "user456",
            "thread_id": "thread789",
            "session_id": "session123",
            "role": "user",
            "text": "Hello",
        }

        manager.submit_enrichment(task_data)

        mock_queue.put.assert_called_once_with(task_data)


class TestWorkerManagerHealth:
    """Tests for WorkerManager health checking."""

    def test_healthy_status(self):
        """Test healthy status detection."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client._enrichment_queue.qsize.return_value = 100
        mock_client.get_worker_health.return_value = {"healthy": True}

        from mindcore.workers import WorkerConfig, WorkerManager

        config = WorkerConfig(backend=WorkerBackend.THREAD, max_queue_size=10000)

        manager = WorkerManager.__new__(WorkerManager)
        manager.client = mock_client
        manager.config = config
        manager._initialized = True
        manager._celery_worker = None

        health = manager.get_health()

        assert health["healthy"] is True
        assert health["queue_depth"] == 100

    def test_unhealthy_queue_depth(self):
        """Test unhealthy detection with high queue depth."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client._enrichment_queue.qsize.return_value = 9000  # 90% of limit
        mock_client.get_worker_health.return_value = {"healthy": True}

        from mindcore.workers import WorkerConfig, WorkerManager

        config = WorkerConfig(backend=WorkerBackend.THREAD, max_queue_size=10000)

        manager = WorkerManager.__new__(WorkerManager)
        manager.client = mock_client
        manager.config = config
        manager._initialized = True
        manager._celery_worker = None

        health = manager.get_health()

        assert health["healthy"] is False
        assert len(health["issues"]) > 0
        assert health["issues"][0]["type"] == "high_queue_depth"
