#!/usr/bin/env python3
"""Runner script for real dataset benchmarking.

This script provides a CLI interface for running the complete
dataset benchmarking pipeline.

Usage:
    # Run with default settings (local enrichment)
    python -m examples.real_datasets.run_benchmark

    # Run with OpenAI enrichment
    python -m examples.real_datasets.run_benchmark --llm openai --api-key $OPENAI_API_KEY

    # Run specific datasets only
    python -m examples.real_datasets.run_benchmark --datasets locomo,persona_chat

    # Custom PostgreSQL connection
    python -m examples.real_datasets.run_benchmark --postgres-dsn "postgresql://user:pass@host:5432/db"

    # Recreate database schema (clean start)
    python -m examples.real_datasets.run_benchmark --recreate-schema

Requirements:
    - PostgreSQL database running locally (or specify --postgres-dsn)
    - Python packages: psycopg[binary] or psycopg2-binary
    - Optional: datasets (HuggingFace), openai or anthropic for LLM enrichment
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_postgres_connection(dsn: str) -> bool:
    """Check if PostgreSQL is accessible."""
    try:
        try:
            import psycopg

            conn = psycopg.connect(dsn)
        except ImportError:
            import psycopg2 as psycopg

            conn = psycopg.connect(dsn)

        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print("You can create the database with:")
        print("  createdb mindcore_datasets")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run real dataset benchmarking pipeline for Mindcore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with local (rule-based) enrichment
  python -m examples.real_datasets.run_benchmark

  # Run with OpenAI GPT-4 enrichment
  python -m examples.real_datasets.run_benchmark --llm openai --api-key sk-xxx

  # Run specific datasets
  python -m examples.real_datasets.run_benchmark --datasets locomo,persona_chat

  # Clean start with schema recreation
  python -m examples.real_datasets.run_benchmark --recreate-schema
        """,
    )

    # Database settings
    parser.add_argument(
        "--postgres-dsn",
        default=os.environ.get("POSTGRES_DSN", "postgresql://localhost:5432/mindcore_datasets"),
        help="PostgreSQL connection string (default: postgresql://localhost:5432/mindcore_datasets)",
    )
    parser.add_argument(
        "--schema",
        default="datasets",
        help="PostgreSQL schema name (default: datasets)",
    )
    parser.add_argument(
        "--recreate-schema",
        action="store_true",
        help="Drop and recreate the schema (clean start)",
    )

    # Dataset settings
    parser.add_argument(
        "--datasets",
        default="locomo,persona_chat,multiwoz",
        help="Comma-separated list of datasets to process (default: locomo,persona_chat,multiwoz)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=50,
        help="Maximum sessions per dataset (default: 50)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum turns per session (default: 30)",
    )

    # LLM enrichment settings
    parser.add_argument(
        "--llm",
        choices=["local", "openai", "anthropic"],
        default="local",
        help="LLM provider for metadata enrichment (default: local)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for LLM provider (or set OPENAI_API_KEY/ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--model",
        help="LLM model to use (e.g., gpt-4o-mini, claude-3-haiku-20240307)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default="./benchmark_output",
        help="Output directory for results (default: ./benchmark_output)",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Don't save intermediate JSON files",
    )

    # Pipeline control
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download datasets, don't enrich or store",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests after storing",
    )

    # General
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't run pipeline",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print banner
    print("\n" + "=" * 60)
    print("MINDCORE REAL DATASET BENCHMARK")
    print("=" * 60)

    # Check prerequisites
    print("\nChecking prerequisites...")

    # Check PostgreSQL
    print(f"  PostgreSQL: {args.postgres_dsn[:30]}...")
    if not check_postgres_connection(args.postgres_dsn):
        print("\n[FAIL] PostgreSQL not accessible")
        sys.exit(1)
    print("  PostgreSQL: OK")

    # Check HuggingFace datasets
    try:
        import datasets

        print("  HuggingFace datasets: OK")
    except ImportError:
        print("  HuggingFace datasets: Not installed (will use sample data)")
        print("    Install with: pip install datasets")

    # Check LLM provider
    if args.llm == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  OpenAI: No API key (use --api-key or set OPENAI_API_KEY)")
            args.llm = "local"
        else:
            try:
                import openai

                print("  OpenAI: OK")
            except ImportError:
                print("  OpenAI: Not installed (pip install openai)")
                args.llm = "local"

    elif args.llm == "anthropic":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  Anthropic: No API key (use --api-key or set ANTHROPIC_API_KEY)")
            args.llm = "local"
        else:
            try:
                import anthropic

                print("  Anthropic: OK")
            except ImportError:
                print("  Anthropic: Not installed (pip install anthropic)")
                args.llm = "local"

    if args.llm == "local":
        print("  Enrichment: Using local rule-based extraction")

    if args.check_only:
        print("\nPrerequisite check complete.")
        sys.exit(0)

    # Parse datasets
    datasets_list = [d.strip() for d in args.datasets.split(",")]
    print(f"\nDatasets to process: {', '.join(datasets_list)}")

    # Import pipeline
    from examples.real_datasets.pipeline import DatasetPipeline, PipelineConfig

    # Create config
    config = PipelineConfig(
        postgres_dsn=args.postgres_dsn,
        schema_name=args.schema,
        recreate_schema=args.recreate_schema,
        datasets=datasets_list,
        max_sessions_per_dataset=args.max_sessions,
        max_turns_per_session=args.max_turns,
        llm_provider=args.llm,
        llm_api_key=args.api_key,
        llm_model=args.model,
        output_dir=args.output_dir,
        save_intermediate=not args.no_save_intermediate,
    )

    # Create and run pipeline
    pipeline = DatasetPipeline(config)

    print("\nStarting pipeline...")
    print("-" * 40)

    if args.download_only:
        # Only download
        pipeline.download()
        print("\nDownload complete. Use --help to see options for enrichment and storage.")
    # Run complete pipeline
    elif args.skip_tests:
        pipeline.download()
        pipeline.enrich()
        pipeline.store_data()
    else:
        pipeline.run()

    # Print summary
    pipeline.print_summary()

    # Return exit code based on test results
    if pipeline._result and pipeline._result.tests_total > 0:
        pass_rate = pipeline._result.tests_passed / pipeline._result.tests_total
        if pass_rate < 0.8:
            print(f"\n[WARNING] Pass rate below 80%: {pass_rate:.1%}")
            sys.exit(1)

    print("\nResults saved to:", args.output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
