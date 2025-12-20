#!/usr/bin/env python3
"""Test Orchestrator for Mindcore Testing.

This script runs all tests in the proper order with comprehensive reporting.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class TestResult:
    """Result of a test run."""

    name: str
    passed: int
    failed: int
    skipped: int
    duration: float
    error: str | None = None

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.error is None


# Test configuration
TESTS = [
    {"file": "test_01_storage.py", "name": "Storage Layer", "requires_postgres": False},
    {"file": "test_02_single_agent.py", "name": "Single Agent", "requires_postgres": False},
    {"file": "test_03_multi_agent.py", "name": "Multi-Agent", "requires_postgres": False},
    {"file": "test_04_flr_clst_flow.py", "name": "FLR/CLST Flow", "requires_postgres": False},
    {"file": "test_05_svl_domains.py", "name": "SVL Domains", "requires_postgres": False},
    {"file": "test_06_svl_sources.py", "name": "SVL Sources", "requires_postgres": False},
    {"file": "test_07_rest_api.py", "name": "REST API", "requires_postgres": False},
    {"file": "test_08_mcp_server.py", "name": "MCP Server", "requires_postgres": False},
    {"file": "test_09_rbac.py", "name": "RBAC", "requires_postgres": False},
    {"file": "test_10_auth_errors.py", "name": "Auth & Errors", "requires_postgres": False},
    {"file": "test_11_integration.py", "name": "Integration", "requires_postgres": True},
]

TESTS_DIR = Path(__file__).parent.parent / "tests"


def print_status(message: str, status: str = "info"):
    """Print status message."""
    if RICH_AVAILABLE:
        console = Console()
        colors = {"info": "blue", "success": "green", "error": "red", "warning": "yellow"}
        console.print(f"[{colors.get(status, 'white')}]{message}[/]")
    else:
        print(f"[{status.upper()}] {message}")


def check_postgres_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "mindcore"),
            password=os.getenv("POSTGRES_PASSWORD", "mindcore_test"),
            database=os.getenv("POSTGRES_DB", "mindcore_test"),
            connect_timeout=5,
        )
        conn.close()
        return True
    except Exception:
        return False


def run_test(test_file: str) -> TestResult:
    """Run a single test file and return results."""
    test_path = TESTS_DIR / test_file

    if not test_path.exists():
        return TestResult(
            name=test_file,
            passed=0,
            failed=0,
            skipped=0,
            duration=0,
            error=f"Test file not found: {test_path}",
        )

    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per test file
            cwd=str(TESTS_DIR.parent),
            check=False,
        )

        duration = time.time() - start_time

        # Parse output for counts
        output = result.stdout + result.stderr

        passed = 0
        failed = 0
        skipped = 0

        # Look for pytest summary line like "5 passed, 1 failed, 2 skipped"
        for line in output.split("\n"):
            if "passed" in line or "failed" in line or "skipped" in line:
                if "passed" in line:
                    try:
                        passed = int(line.split("passed")[0].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass
                if "failed" in line:
                    try:
                        failed = int(line.split("failed")[0].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass
                if "skipped" in line:
                    try:
                        skipped = int(line.split("skipped")[0].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass

        # If we couldn't parse, count PASSED/FAILED in output
        if passed == 0 and failed == 0:
            passed = output.count("PASSED")
            failed = output.count("FAILED")
            skipped = output.count("SKIPPED")

        error_msg = None
        if result.returncode != 0 and failed == 0:
            # Test run failed but not due to test failures
            error_msg = output[-500:] if len(output) > 500 else output

        return TestResult(
            name=test_file,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=duration,
            error=error_msg,
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=test_file,
            passed=0,
            failed=0,
            skipped=0,
            duration=300,
            error="Test timed out after 5 minutes",
        )
    except Exception as e:
        return TestResult(
            name=test_file,
            passed=0,
            failed=0,
            skipped=0,
            duration=time.time() - start_time,
            error=str(e),
        )


def run_all_tests(skip_postgres: bool = False, test_filter: str | None = None) -> list[TestResult]:
    """Run all tests and collect results."""
    results = []
    postgres_available = check_postgres_available()

    if not postgres_available and not skip_postgres:
        print_status("PostgreSQL not available. Some tests will be skipped.", "warning")

    tests_to_run = TESTS
    if test_filter:
        tests_to_run = [t for t in TESTS if test_filter.lower() in t["name"].lower()]

    for test_config in tests_to_run:
        test_file = test_config["file"]
        test_name = test_config["name"]

        # Skip PostgreSQL tests if not available
        if test_config.get("requires_postgres") and not postgres_available:
            results.append(
                TestResult(
                    name=test_file,
                    passed=0,
                    failed=0,
                    skipped=1,
                    duration=0,
                    error="PostgreSQL not available",
                )
            )
            print_status(f"Skipping {test_name} (requires PostgreSQL)", "warning")
            continue

        print_status(f"Running {test_name}...", "info")
        result = run_test(test_file)
        results.append(result)

        if result.success:
            print_status(
                f"  {test_name}: {result.passed} passed in {result.duration:.2f}s", "success"
            )
        else:
            print_status(f"  {test_name}: {result.failed} failed, {result.passed} passed", "error")
            if result.error:
                print_status(f"  Error: {result.error[:200]}", "error")

    return results


def show_summary(results: list[TestResult]):
    """Display test summary."""
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration for r in results)

    if RICH_AVAILABLE:
        console = Console()

        # Results table
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Passed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Skipped", style="yellow")
        table.add_column("Duration", style="blue")
        table.add_column("Status")

        for result in results:
            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
            table.add_row(
                result.name,
                str(result.passed),
                str(result.failed),
                str(result.skipped),
                f"{result.duration:.2f}s",
                status,
            )

        # Summary row
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold green]{total_passed}[/bold green]",
            f"[bold red]{total_failed}[/bold red]",
            f"[bold yellow]{total_skipped}[/bold yellow]",
            f"[bold]{total_duration:.2f}s[/bold]",
            "",
        )

        console.print(table)

        # Final summary
        if total_failed == 0:
            console.print(
                Panel(
                    f"[bold green]All tests passed![/bold green]\n"
                    f"{total_passed} tests in {total_duration:.2f}s",
                    title="Success",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]{total_failed} tests failed[/bold red]\n"
                    f"{total_passed} passed, {total_skipped} skipped",
                    title="Failure",
                )
            )
    else:
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        for result in results:
            status = "PASS" if result.success else "FAIL"
            print(
                f"{result.name:40} {status:6} {result.passed:3}p {result.failed:3}f {result.duration:.2f}s"
            )
        print("-" * 60)
        print(f"{'TOTAL':40} {'':6} {total_passed:3}p {total_failed:3}f {total_duration:.2f}s")
        print("=" * 60)

        if total_failed == 0:
            print(f"\nSUCCESS: All {total_passed} tests passed!")
        else:
            print(f"\nFAILURE: {total_failed} tests failed, {total_passed} passed")


def main():
    """Main test orchestration flow."""
    import argparse

    parser = argparse.ArgumentParser(description="Run mindcore tests")
    parser.add_argument(
        "--skip-postgres", action="store_true", help="Skip tests requiring PostgreSQL"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter tests by name (e.g., 'storage', 'agent')"
    )
    parser.add_argument(
        "--setup-postgres", action="store_true", help="Run PostgreSQL setup before tests"
    )
    parser.add_argument("--load-data", action="store_true", help="Load demo data before tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print_status("=" * 50, "info")
    print_status("Mindcore Test Suite", "info")
    print_status("=" * 50, "info")

    # Optional: Setup PostgreSQL
    if args.setup_postgres:
        print_status("\nSetting up PostgreSQL...", "info")
        setup_script = Path(__file__).parent / "setup_postgres.py"
        subprocess.run([sys.executable, str(setup_script)], check=False)

    # Optional: Load demo data
    if args.load_data:
        print_status("\nLoading demo data...", "info")
        load_script = Path(__file__).parent / "load_demo_data.py"
        subprocess.run([sys.executable, str(load_script), "--multi-agent"], check=False)

    # Run tests
    print_status("\nRunning tests...\n", "info")
    results = run_all_tests(skip_postgres=args.skip_postgres, test_filter=args.filter)

    # Show summary
    print_status("", "info")
    show_summary(results)

    # Exit with appropriate code
    total_failed = sum(r.failed for r in results)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
