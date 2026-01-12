#!/usr/bin/env python3
"""Mindcore SAGE CLI.

Usage:
    mindcore init [--postgres URL]    Initialize database
    mindcore check [--postgres URL]   Check database connection
    mindcore version                  Show version

Examples:
    # Initialize with SQLite (development)
    mindcore init

    # Initialize with PostgreSQL
    mindcore init --postgres postgresql://user:pass@localhost/mindcore

    # Check connection
    mindcore check --postgres postgresql://localhost/mindcore
"""

from __future__ import annotations

import argparse
import sys


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize database schema."""

    if args.postgres:
        return init_postgres(args.postgres)
    else:
        return init_sqlite(args.sqlite or "mindcore.db")


def init_sqlite(db_path: str) -> int:
    """Initialize SQLite database."""

    print(f"Initializing SQLite database: {db_path}")

    try:
        from mindcore.v2.storage import SQLiteStorage

        storage = SQLiteStorage(db_path)
        print(f"  ✓ Database created: {db_path}")
        print(f"  ✓ Schema initialized")
        print()
        print("Next steps:")
        print(f"  from mindcore.v2.storage import SQLiteStorage")
        print(f"  storage = SQLiteStorage('{db_path}')")
        return 0

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1


def init_postgres(connection_string: str) -> int:
    """Initialize PostgreSQL database."""

    print(f"Initializing PostgreSQL database...")

    try:
        from mindcore.v2.storage import PostgresStorage

        # Connect
        print(f"  Connecting to database...")
        storage = PostgresStorage(connection_string)
        print(f"  ✓ Connected")

        # Initialize schema
        print(f"  Initializing schema...")
        storage.initialize_full_schema()
        print(f"  ✓ Schema created (tables, indexes, functions, triggers)")

        # Verify
        print(f"  Verifying...")
        storage.initialize_functions()
        print(f"  ✓ Functions loaded")

        print()
        print("Database ready! Next steps:")
        print()
        print("  from mindcore.v2.storage import PostgresStorage")
        print(f"  storage = PostgresStorage('{_redact_password(connection_string)}')")
        print()
        print("Or use the SVL pipeline:")
        print()
        print("  from mindcore.v2.svl import SVLPipeline")
        print("  pipeline = SVLPipeline(storage=storage)")
        return 0

    except ImportError:
        print("  ✗ psycopg not installed")
        print("  Run: pip install 'psycopg[binary,pool]'")
        return 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Check database connection and schema."""

    if args.postgres:
        return check_postgres(args.postgres)
    else:
        return check_sqlite(args.sqlite or "mindcore.db")


def check_sqlite(db_path: str) -> int:
    """Check SQLite database."""

    import os

    if not os.path.exists(db_path):
        print(f"✗ Database not found: {db_path}")
        print(f"  Run: mindcore init")
        return 1

    try:
        from mindcore.v2.storage import SQLiteStorage

        storage = SQLiteStorage(db_path)
        count = storage.count()
        print(f"✓ SQLite database OK")
        print(f"  Path: {db_path}")
        print(f"  Memories: {count}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def check_postgres(connection_string: str) -> int:
    """Check PostgreSQL connection and schema."""

    print("Checking PostgreSQL connection...")

    try:
        from mindcore.v2.storage import PostgresStorage

        storage = PostgresStorage(connection_string)

        # Check connection
        print("  ✓ Connection OK")

        # Check tables
        with storage._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('memories', 'sessions')
            """)
            tables = [row[0] for row in cur.fetchall()]

        if "memories" in tables and "sessions" in tables:
            print("  ✓ Schema OK (memories, sessions tables exist)")
        else:
            print("  ⚠ Schema incomplete")
            print(f"    Found: {tables}")
            print("    Run: mindcore init --postgres ...")

        # Check functions
        with storage._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT routine_name FROM information_schema.routines
                WHERE routine_schema = 'public'
                AND routine_name = 'sage_score'
            """)
            functions = [row[0] for row in cur.fetchall()]

        if "sage_score" in functions:
            print("  ✓ Functions OK (sage_score exists)")
        else:
            print("  ⚠ Functions missing")
            print("    Run: mindcore init --postgres ...")

        # Count memories
        count = storage.count()
        print(f"  Memories: {count}")

        return 0

    except ImportError:
        print("  ✗ psycopg not installed")
        print("  Run: pip install 'psycopg[binary,pool]'")
        return 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Show version."""

    try:
        from mindcore import __version__

        print(f"mindcore {__version__}")
    except ImportError:
        print("mindcore (version unknown)")

    print()
    print("SAGE - Structured Augmented Generation Engine")
    print("PostgreSQL-first memory platform for AI agents")
    return 0


def _redact_password(url: str) -> str:
    """Redact password from connection string for display."""

    import re

    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)


def main() -> int:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        prog="mindcore",
        description="Mindcore SAGE - Structured Augmented Generation Engine",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument(
        "--postgres", metavar="URL", help="PostgreSQL connection string"
    )
    init_parser.add_argument(
        "--sqlite", metavar="PATH", help="SQLite database path (default: mindcore.db)"
    )
    init_parser.set_defaults(func=cmd_init)

    # check command
    check_parser = subparsers.add_parser("check", help="Check database connection")
    check_parser.add_argument(
        "--postgres", metavar="URL", help="PostgreSQL connection string"
    )
    check_parser.add_argument("--sqlite", metavar="PATH", help="SQLite database path")
    check_parser.set_defaults(func=cmd_check)

    # version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
