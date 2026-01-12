#!/usr/bin/env python3
"""SAGE Quick Start Example.

This example shows the simplest way to get started with Mindcore SAGE.

Requirements:
    pip install mindcore

For PostgreSQL (recommended for production):
    docker-compose up -d  # See docker-compose.yml in this directory

For SQLite (development only):
    No setup needed - just run this script!
"""

from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.flr import Memory

# =============================================================================
# OPTION 1: SQLite (Zero Setup - Great for Testing)
# =============================================================================

def quickstart_sqlite():
    """Quickstart with SQLite - no database setup needed."""

    print("=== SAGE Quickstart (SQLite) ===\n")

    # 1. Create storage (in-memory for this example)
    storage = SQLiteStorage(":memory:")

    # 2. Store a memory
    memory = Memory(
        content="User prefers dark mode and brief responses",
        memory_type="preference",
        user_id="user_123",
        topics=["settings", "ui"],
        categories=["preferences"],
        importance=0.8,
        sentiment="neutral",
    )

    memory_id = storage.store(memory)
    print(f"Stored memory: {memory_id}")

    # 3. Search memories
    results = storage.search(
        user_id="user_123",
        query="user preferences",
        limit=5,
    )

    print(f"\nFound {len(results)} memories:")
    for mem in results:
        print(f"  - [{mem.importance:.1f}] {mem.content[:50]}...")

    # 4. Reinforce a memory (positive feedback)
    storage.update_reinforcement(memory_id, signal=0.3)
    print(f"\nReinforced memory with +0.3 signal")

    # 5. Get memory back
    retrieved = storage.get(memory_id)
    print(f"Retrieved: {retrieved.content}")
    print(f"Reinforcement score: {retrieved.reinforcement_score}")

    print("\n✓ SQLite quickstart complete!")


# =============================================================================
# OPTION 2: PostgreSQL (Production Ready)
# =============================================================================

def quickstart_postgres():
    """Quickstart with PostgreSQL - requires running database.

    Run: docker-compose up -d
    """

    print("=== SAGE Quickstart (PostgreSQL) ===\n")

    try:
        from mindcore.v2.storage import PostgresStorage
    except ImportError:
        print("Install psycopg: pip install 'psycopg[binary,pool]'")
        return

    # 1. Connect to PostgreSQL
    # Default docker-compose credentials
    connection_string = "postgresql://mindcore:mindcore@localhost:5432/mindcore"

    try:
        storage = PostgresStorage(connection_string)
    except Exception as e:
        print(f"Could not connect to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker-compose up -d")
        return

    # 2. Initialize schema (first time only)
    print("Initializing database schema...")
    storage.initialize_full_schema()
    print("✓ Schema initialized")

    # 3. Store a memory
    memory = Memory(
        content="Customer asked about order #12345 shipping status",
        memory_type="episodic",
        user_id="user_456",
        session_id="session_abc",
        topics=["orders", "shipping"],
        categories=["support"],
        importance=0.7,
        sentiment="neutral",
    )

    memory_id = storage.store(memory)
    print(f"\nStored memory: {memory_id}")

    # 4. SAGE scored search (scoring happens in PostgreSQL!)
    print("\nSearching with SAGE scoring...")
    results = storage.search_scored(
        user_id="user_456",
        query="order shipping",
        topics=["orders"],
        limit=10,
    )

    print(f"Found {len(results)} memories:")
    for mem, score in results:
        print(f"  - [score={score:.3f}] {mem.content[:50]}...")

    # 5. Find relevant sessions
    sessions = storage.find_relevant_sessions(
        user_id="user_456",
        topics=["orders"],
        limit=5,
    )

    print(f"\nFound {len(sessions)} relevant sessions")

    print("\n✓ PostgreSQL quickstart complete!")


# =============================================================================
# OPTION 3: Full SAGE Pipeline with SVL
# =============================================================================

def quickstart_sage_pipeline():
    """Full SAGE pipeline with SVL kernel validation."""

    print("=== SAGE Pipeline Quickstart ===\n")

    from mindcore.v2.storage import SQLiteStorage
    from mindcore.v2.svl import SharedVocabularyLayer, SVLPipeline

    # 1. Create storage
    storage = SQLiteStorage(":memory:")

    # 2. Create SVL vocabulary (the kernel)
    svl = SharedVocabularyLayer(domains=["customer_service"])

    # 3. Create pipeline
    pipeline = SVLPipeline(
        storage=storage,
        vocabulary=svl,
        use_simple_flr=True,  # Deterministic hot path
    )

    print("✓ Pipeline created with SVL kernel")

    # 4. Store via pipeline (SVL validates metadata)
    result = pipeline.store(
        llm_output={
            "content": "User wants to cancel their subscription",
            "memory_type": "episodic",
            "topics": ["billing", "cancellation"],
            "categories": ["support"],
            "importance": 0.8,
            "sentiment": "negative",
        },
        user_id="user_789",
        session_id="session_xyz",
    )

    print(f"Stored via pipeline: {result.memory_id}")

    # 5. Query via pipeline
    query_result = pipeline.query(
        query="subscription cancellation",
        user_id="user_789",
        limit=5,
    )

    print(f"\nQuery returned {len(query_result.memories)} memories")
    print(f"CLST needed: {query_result.clst_decision.needs_clst}")

    # 6. Get pipeline stats
    stats = pipeline.get_stats()
    print(f"\nPipeline stats:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Hot path ratio: {stats['hot_path_ratio']:.1%}")

    print("\n✓ SAGE pipeline quickstart complete!")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("MINDCORE SAGE - Quick Start Examples")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "sqlite":
            quickstart_sqlite()
        elif mode == "postgres":
            quickstart_postgres()
        elif mode == "pipeline":
            quickstart_sage_pipeline()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python quickstart.py [sqlite|postgres|pipeline]")
    else:
        # Run SQLite example by default (zero setup)
        quickstart_sqlite()
        print("\n" + "-" * 60)
        print("Try other examples:")
        print("  python quickstart.py postgres   # PostgreSQL + SAGE scoring")
        print("  python quickstart.py pipeline   # Full SVL pipeline")
