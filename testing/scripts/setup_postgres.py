#!/usr/bin/env python3
"""PostgreSQL Setup Script for Mindcore Testing.

This script:
1. Waits for PostgreSQL container to be ready
2. Creates the mindcore schema
3. Validates connectivity
"""

import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "mindcore")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mindcore_test")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mindcore_test")

CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def print_status(message: str, status: str = "info"):
    """Print status message with optional rich formatting."""
    if RICH_AVAILABLE:
        console = Console()
        colors = {"info": "blue", "success": "green", "error": "red", "warning": "yellow"}
        console.print(f"[{colors.get(status, 'white')}]{message}[/]")
    else:
        print(f"[{status.upper()}] {message}")


def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(  # nosec B603 B607
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def start_postgres_container() -> bool:
    """Start PostgreSQL container using docker-compose."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"

    if not compose_file.exists():
        print_status(f"docker-compose.yml not found at {compose_file}", "error")
        return False

    print_status("Starting PostgreSQL container...", "info")

    try:
        result = subprocess.run(  # nosec B603 B607
            ["docker-compose", "-f", str(compose_file), "up", "-d", "postgres"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            print_status(f"Failed to start container: {result.stderr}", "error")
            return False

        print_status("Container started successfully", "success")
        return True

    except subprocess.SubprocessError as e:
        print_status(f"Docker command failed: {e}", "error")
        return False


def wait_for_postgres(max_attempts: int = 30, delay: float = 2.0) -> bool:
    """Wait for PostgreSQL to be ready."""
    print_status(f"Waiting for PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}...", "info")

    try:
        import psycopg2
    except ImportError:
        print_status("psycopg2 not installed. Install with: pip install psycopg2-binary", "error")
        return False

    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                database=POSTGRES_DB,
                connect_timeout=5,
            )
            conn.close()
            print_status(f"PostgreSQL is ready! (attempt {attempt + 1})", "success")
            return True

        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            continue

    print_status(f"PostgreSQL not ready after {max_attempts} attempts", "error")
    return False


def create_schema() -> bool:
    """Create mindcore schema tables."""
    try:
        import psycopg2  # noqa: F401  - availability check
    except ImportError:
        print_status("psycopg2 not installed", "error")
        return False

    print_status("Creating mindcore schema...", "info")

    # The schema will be created automatically by mindcore storage
    # We just validate connectivity here
    try:
        from mindcore.storage.postgres import PostgresStorage

        storage = PostgresStorage(CONNECTION_STRING)
        stats = storage.get_stats()
        storage.close()

        print_status(f"Schema created. Stats: {stats}", "success")
        return True

    except ImportError:
        print_status("mindcore not installed. Schema will be created on first use.", "warning")
        return True
    except Exception as e:
        print_status(f"Schema creation failed: {e}", "error")
        return False


def validate_connection() -> bool:
    """Validate full mindcore connection to PostgreSQL."""
    print_status("Validating mindcore connection...", "info")

    try:
        from mindcore import Mindcore

        mc = Mindcore(storage=CONNECTION_STRING)

        # Store a test memory
        memory_id = mc.store(
            content="Test memory for PostgreSQL validation",
            memory_type="semantic",
            user_id="test_user",
            topics=["api"],
            importance=0.5,
        )

        # Recall it
        result = mc.recall(query="test memory PostgreSQL", user_id="test_user")

        # Clean up
        mc.delete(memory_id)
        mc.close()

        if len(result.memories) > 0:
            print_status("Connection validated successfully!", "success")
            return True
        print_status("Validation failed: could not recall test memory", "error")
        return False

    except ImportError:
        print_status("mindcore not installed. Run: pip install mindcore", "warning")
        return True
    except Exception as e:
        print_status(f"Validation failed: {e}", "error")
        return False


def main():
    """Main setup flow."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(
            Panel.fit(
                "[bold blue]Mindcore PostgreSQL Setup[/bold blue]\n"
                f"Host: {POSTGRES_HOST}:{POSTGRES_PORT}\n"
                f"Database: {POSTGRES_DB}",
                title="Configuration",
            )
        )
    else:
        print("=" * 50)
        print("Mindcore PostgreSQL Setup")
        print(f"Host: {POSTGRES_HOST}:{POSTGRES_PORT}")
        print(f"Database: {POSTGRES_DB}")
        print("=" * 50)

    # Step 1: Check Docker
    if not check_docker_running():
        print_status("Docker is not running. Please start Docker first.", "error")
        sys.exit(1)

    # Step 2: Start container
    if not start_postgres_container():
        print_status("Failed to start PostgreSQL container", "error")
        sys.exit(1)

    # Step 3: Wait for PostgreSQL
    if not wait_for_postgres():
        print_status("PostgreSQL connection timed out", "error")
        sys.exit(1)

    # Step 4: Create schema
    if not create_schema():
        print_status("Schema creation failed", "error")
        sys.exit(1)

    # Step 5: Validate
    if not validate_connection():
        print_status("Connection validation failed", "error")
        sys.exit(1)

    print_status("\nPostgreSQL setup complete! Ready for testing.", "success")
    print_status(f"\nConnection string: {CONNECTION_STRING}", "info")


if __name__ == "__main__":
    main()
