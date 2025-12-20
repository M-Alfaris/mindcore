"""Pytest Configuration and Fixtures for Mindcore Testing.

This module provides shared fixtures and configuration for all tests.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import pytest


# Add mindcore to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Configuration
# ============================================================================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "mindcore")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mindcore_test")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mindcore_test")

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


# ============================================================================
# Helper Functions
# ============================================================================


def is_postgres_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            connect_timeout=5,
        )
        conn.close()
        return True
    except Exception:
        return False


# Skip markers
requires_postgres = pytest.mark.skipif(
    not is_postgres_available(), reason="PostgreSQL not available"
)


# ============================================================================
# Fixtures: Storage
# ============================================================================


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def sqlite_storage(temp_db_path):
    """Provide a SQLite storage instance."""
    from mindcore.v2.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(temp_db_path)
    yield storage
    storage.close()


@pytest.fixture
def postgres_storage():
    """Provide a PostgreSQL storage instance (if available)."""
    if not is_postgres_available():
        pytest.skip("PostgreSQL not available")

    from mindcore.v2.storage.postgres import PostgresStorage

    storage = PostgresStorage(POSTGRES_URL)
    yield storage
    storage.close()


# ============================================================================
# Fixtures: Mindcore Instances
# ============================================================================


@pytest.fixture
def mindcore(temp_db_path):
    """Provide a basic Mindcore instance with SQLite."""
    from mindcore.v2 import Mindcore

    mc = Mindcore(storage=f"sqlite:///{temp_db_path}")
    yield mc
    mc.close()


@pytest.fixture
def mindcore_multi_agent(temp_db_path):
    """Provide a Mindcore instance with multi-agent enabled."""
    from mindcore.v2 import Mindcore

    mc = Mindcore(storage=f"sqlite:///{temp_db_path}", enable_multi_agent=True)
    yield mc
    mc.close()


@pytest.fixture
def mindcore_postgres():
    """Provide a Mindcore instance with PostgreSQL."""
    if not is_postgres_available():
        pytest.skip("PostgreSQL not available")

    from mindcore.v2 import Mindcore

    mc = Mindcore(storage=POSTGRES_URL)
    yield mc
    mc.close()


# ============================================================================
# Fixtures: SVL Components
# ============================================================================


@pytest.fixture
def default_svl():
    """Provide the default SharedVocabularyLayer."""
    from mindcore.v2.svl import SharedVocabularyLayer

    return SharedVocabularyLayer()


@pytest.fixture
def custom_svl():
    """Provide a custom SharedVocabularyLayer with domains."""
    from mindcore.v2.svl import SharedVocabularyLayer, SVLSchema

    schema = SVLSchema(
        version="1.0.0",
        topics=["test_topic", "custom_topic", "api", "billing"],
        categories=["test_category", "custom_category", "support"],
        subcategories={},
        memory_types=["episodic", "semantic", "preference", "entity", "procedural"],
        sentiments=["positive", "negative", "neutral"],
        access_levels=["private", "team", "shared", "global"],
        domains=["customer_service"],
        custom_fields=[],
        migrations={},
        description="Custom test vocabulary",
    )

    return SharedVocabularyLayer(schema=schema)


# ============================================================================
# Fixtures: FLR and CLST
# ============================================================================


@pytest.fixture
def flr(sqlite_storage):
    """Provide an FLR instance."""
    from mindcore.v2.flr import FLR

    return FLR(storage=sqlite_storage)


@pytest.fixture
def clst(sqlite_storage, default_svl):
    """Provide a CLST instance."""
    from mindcore.v2.clst import CLST

    return CLST(storage=sqlite_storage, vocabulary=default_svl)


# ============================================================================
# Fixtures: Sample Data
# ============================================================================


@pytest.fixture
def sample_memory_data() -> dict:
    """Provide sample memory data for testing."""
    return {
        "content": "User prefers dark mode for the interface",
        "memory_type": "preference",
        "user_id": "test_user_001",
        "topics": ["settings"],
        "categories": ["account"],
        "importance": 0.8,
        "entities": ["dark mode"],
        "access_level": "private",
    }


@pytest.fixture
def sample_memories() -> list[dict]:
    """Provide multiple sample memories."""
    return [
        {
            "content": "User prefers Python for backend development",
            "memory_type": "preference",
            "user_id": "test_user_001",
            "topics": ["api", "integration"],
            "categories": ["technical"],
            "importance": 0.9,
        },
        {
            "content": "User reported bug with login flow yesterday",
            "memory_type": "episodic",
            "user_id": "test_user_001",
            "topics": ["bug", "issue"],
            "categories": ["support"],
            "importance": 0.7,
        },
        {
            "content": "Company uses OAuth2 for authentication",
            "memory_type": "semantic",
            "user_id": "test_user_001",
            "topics": ["api", "documentation"],
            "categories": ["technical"],
            "importance": 0.6,
        },
        {
            "content": "Billing contact is John Smith at billing@company.com",
            "memory_type": "entity",
            "user_id": "test_user_001",
            "topics": ["billing", "account"],
            "categories": ["billing"],
            "importance": 0.75,
            "entities": ["John Smith", "billing@company.com"],
        },
        {
            "content": "To reset password: 1. Click forgot password 2. Enter email 3. Check inbox 4. Click link",
            "memory_type": "procedural",
            "user_id": "test_user_001",
            "topics": ["help", "account"],
            "categories": ["support"],
            "importance": 0.5,
        },
    ]


@pytest.fixture
def sample_agents() -> list[dict]:
    """Provide sample agent configurations."""
    return [
        {
            "agent_id": "support_bot",
            "name": "Support Agent",
            "description": "Handles customer support",
            "teams": ["support", "general"],
        },
        {
            "agent_id": "sales_bot",
            "name": "Sales Agent",
            "description": "Handles sales inquiries",
            "teams": ["sales"],
        },
        {
            "agent_id": "tech_bot",
            "name": "Technical Agent",
            "description": "Handles technical issues",
            "teams": ["support", "engineering"],
        },
    ]


# ============================================================================
# Fixtures: Demo Data Files
# ============================================================================


@pytest.fixture
def demo_data_dir() -> Path:
    """Provide path to demo data directory."""
    return Path(__file__).parent.parent / "demo_data"


@pytest.fixture
def memories_data(demo_data_dir) -> dict:
    """Load memories from demo data."""
    with open(demo_data_dir / "memories.json") as f:
        return json.load(f)


@pytest.fixture
def agents_data(demo_data_dir) -> dict:
    """Load agents from demo data."""
    with open(demo_data_dir / "agents.json") as f:
        return json.load(f)


@pytest.fixture
def vocabularies_data(demo_data_dir) -> dict:
    """Load vocabularies from demo data."""
    with open(demo_data_dir / "vocabularies.json") as f:
        return json.load(f)


# ============================================================================
# Fixtures: Timing and Dates
# ============================================================================


@pytest.fixture
def old_date() -> datetime:
    """Provide a date 60 days in the past."""
    return datetime.now() - timedelta(days=60)


@pytest.fixture
def recent_date() -> datetime:
    """Provide a date from yesterday."""
    return datetime.now() - timedelta(days=1)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "postgres: marks tests that require PostgreSQL")
    config.addinivalue_line("markers", "integration: marks integration tests")
