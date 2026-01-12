"""Benchmark: Search Comparison (Python vs SQL-based ranking).

This benchmark compares:
1. Python-side scoring (word overlap, topic matching)
2. SQL-based scoring (pg_trgm similarity, rank_memory function)
3. BM25 hybrid search (if ParadeDB available)

Requirements:
- PostgreSQL with pg_trgm extension for SQL benchmarks
- ParadeDB pg_search for BM25 benchmarks
- Run schema/extensions.sql and schema/ranking_functions.sql first

Usage:
    # Run with default settings
    python -m mindcore.benchmarks.search_comparison

    # Run with custom iterations
    python -m mindcore.benchmarks.search_comparison --iterations 100

    # Run against specific database
    python -m mindcore.benchmarks.search_comparison --database-url postgresql://user:pass@localhost/mindcore
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mindcore.flr import Memory


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    results_count: int
    error_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "results_count": self.results_count,
            "error_count": self.error_count,
            "notes": self.notes,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: list[BenchmarkResult] = field(default_factory=list)
    memory_count: int = 0
    capabilities: dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark Suite: {self.name}",
            f"Description: {self.description}",
            f"Timestamp: {self.timestamp}",
            f"Memories in database: {self.memory_count}",
            "",
            "Available Capabilities:",
        ]

        for cap, available in self.capabilities.items():
            status = "Yes" if available else "No"
            lines.append(f"  {cap}: {status}")

        lines.append("")
        lines.append("Results:")
        lines.append("-" * 80)

        # Get baseline (basic_search) for comparison
        baseline = next((r for r in self.results if r.name == "basic_search (ts_rank)"), None)

        for result in self.results:
            speedup = ""
            if baseline and result.name != baseline.name and baseline.avg_time_ms > 0:
                ratio = baseline.avg_time_ms / result.avg_time_ms
                if ratio > 1:
                    speedup = f" ({ratio:.1f}x faster than basic)"
                elif ratio < 1:
                    speedup = f" ({1/ratio:.1f}x slower than basic)"

            lines.append(
                f"{result.name}:\n"
                f"  Avg: {result.avg_time_ms:.3f}ms{speedup}\n"
                f"  Min/Max: {result.min_time_ms:.3f}ms / {result.max_time_ms:.3f}ms\n"
                f"  StdDev: {result.std_dev_ms:.3f}ms\n"
                f"  Results: {result.results_count} per query\n"
                f"  Errors: {result.error_count}"
            )
            if result.notes:
                lines.append(f"  Notes: {result.notes}")
            lines.append("")

        return "\n".join(lines)


def create_test_memories(storage, count: int = 1000) -> list[str]:
    """Create test memories for benchmarking.

    Creates memories with various topics for realistic search scenarios.
    """
    memory_ids = []
    topics_pool = [
        ["orders", "shipping", "delivery"],
        ["orders", "returns", "refunds"],
        ["account", "settings", "profile"],
        ["billing", "payment", "invoices"],
        ["support", "help", "issues"],
        ["products", "catalog", "inventory"],
        ["preferences", "notifications", "email"],
    ]

    for i in range(count):
        topic_set = topics_pool[i % len(topics_pool)]
        memory = Memory(
            memory_id=f"bench_mem_{i}",
            content=f"This is test memory {i} about {', '.join(topic_set)}. "
            f"The user asked about their order status and shipping information.",
            memory_type="episodic",
            user_id="benchmark_user",
            topics=topic_set,
            categories=["benchmark"],
            importance=0.3 + (i % 7) * 0.1,
            access_count=i % 20,
            reinforcement_score=((i % 10) - 5) / 10,
        )
        try:
            memory_id = storage.store(memory)
            memory_ids.append(memory_id)
        except Exception:
            pass  # Skip duplicates

    return memory_ids


def benchmark_search_ranked(
    storage,
    query: str,
    user_id: str,
    attention_hints: list[str],
    iterations: int,
) -> BenchmarkResult:
    """Benchmark SQL-based search_ranked."""
    times = []
    results_count = 0
    errors = 0

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            results = storage.search_ranked(
                query=query,
                user_id=user_id,
                attention_hints=attention_hints,
                limit=50,
            )
            results_count = len(results)
        except Exception:
            errors += 1
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="search_ranked (pg_trgm + rank_memory)",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        results_count=results_count,
        error_count=errors,
        notes="Trigram similarity + SQL rank_memory function",
    )


def benchmark_search_bm25(
    storage,
    query: str,
    user_id: str,
    attention_hints: list[str],
    iterations: int,
) -> BenchmarkResult:
    """Benchmark BM25 hybrid search."""
    times = []
    results_count = 0
    errors = 0

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            results = storage.search_bm25(
                query=query,
                user_id=user_id,
                attention_hints=attention_hints,
                limit=50,
            )
            results_count = len(results)
        except Exception:
            errors += 1
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="search_bm25 (ParadeDB)",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        results_count=results_count,
        error_count=errors,
        notes="BM25 + custom ranking signals",
    )


def benchmark_basic_search(
    storage,
    query: str,
    user_id: str,
    topics: list[str],
    iterations: int,
) -> BenchmarkResult:
    """Benchmark basic ts_rank search (baseline)."""
    times = []
    results_count = 0
    errors = 0

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            results = storage.search(
                query=query,
                user_id=user_id,
                topics=topics,
                limit=50,
            )
            results_count = len(results)
        except Exception:
            errors += 1
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="basic_search (ts_rank)",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        results_count=results_count,
        error_count=errors,
        notes="PostgreSQL ts_rank full-text search",
    )


def run_benchmarks(
    database_url: str,
    iterations: int = 50,
    memory_count: int = 1000,
    create_test_data: bool = True,
) -> BenchmarkSuite:
    """Run the complete benchmark suite.

    Compares SQL-based search methods:
    1. Basic search (ts_rank)
    2. search_ranked (pg_trgm + rank_memory)
    3. search_bm25 (ParadeDB)
    """
    from mindcore.storage.postgres import PostgresStorage

    print("Connecting to database...")
    storage = PostgresStorage(database_url)

    capabilities = storage.search_capabilities
    print(f"Search capabilities: {capabilities}")

    # Create test data if needed
    if create_test_data:
        print(f"Creating {memory_count} test memories...")
        create_test_memories(storage, memory_count)

    # Get memory count
    stats = storage.get_stats()
    actual_memory_count = stats.get("memory_count", memory_count)

    # Benchmark parameters
    query = "order status shipping delivery"
    attention_hints = ["orders", "shipping"]
    user_id = "benchmark_user"

    suite = BenchmarkSuite(
        name="Search Comparison",
        description=f"Comparing SQL search methods with {actual_memory_count} memories, {iterations} iterations",
        memory_count=actual_memory_count,
        capabilities=capabilities,
    )

    # Run benchmarks
    print(f"\nRunning benchmarks ({iterations} iterations each)...")

    # 1. Basic search (always available)
    print("  [1/3] Basic search (ts_rank)...")
    suite.results.append(
        benchmark_basic_search(storage, query, user_id, attention_hints, iterations)
    )

    # 2. SQL-based search_ranked (requires pg_trgm + rank_memory)
    if capabilities.get("sql_memory_ranking"):
        print("  [2/3] search_ranked (pg_trgm + rank_memory)...")
        suite.results.append(
            benchmark_search_ranked(storage, query, user_id, attention_hints, iterations)
        )
    else:
        print("  [2/3] search_ranked - SKIPPED (extensions not available)")

    # 3. BM25 search (requires ParadeDB)
    if capabilities.get("bm25_search"):
        print("  [3/3] search_bm25 (ParadeDB)...")
        suite.results.append(
            benchmark_search_bm25(storage, query, user_id, attention_hints, iterations)
        )
    else:
        print("  [3/3] search_bm25 - SKIPPED (ParadeDB not available)")

    storage.close()
    return suite


def main():
    """Run the search comparison benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark search methods (Python vs SQL-based ranking)"
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "postgresql://localhost/mindcore"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50)",
    )
    parser.add_argument(
        "--memory-count",
        type=int,
        default=1000,
        help="Number of test memories to create (default: 1000)",
    )
    parser.add_argument(
        "--skip-create-data",
        action="store_true",
        help="Skip creating test data (use existing data)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    try:
        suite = run_benchmarks(
            database_url=args.database_url,
            iterations=args.iterations,
            memory_count=args.memory_count,
            create_test_data=not args.skip_create_data,
        )

        if args.output == "json":
            import json

            output = {
                "name": suite.name,
                "description": suite.description,
                "timestamp": suite.timestamp,
                "memory_count": suite.memory_count,
                "capabilities": suite.capabilities,
                "results": [r.to_dict() for r in suite.results],
            }
            print(json.dumps(output, indent=2))
        else:
            print("\n" + "=" * 80)
            print(suite.summary())

    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
