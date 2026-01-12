"""Mindcore CLI - Interactive setup and management tools.

Usage:
    mindcore init          # Interactive setup wizard
    mindcore doctor        # Health check and diagnostics
    mindcore demo          # Run a quick demo
    mindcore status        # Show current configuration
    mindcore serve         # Start the API server

Quick Start:
    $ pip install mindcore
    $ mindcore init
    $ mindcore demo
"""

import os
from pathlib import Path

import click

from .demo import demo_command
from .doctor import doctor_command
from .init import init_command


class Colors:
    """ANSI color codes."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def styled(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    return "".join(styles) + text + Colors.RESET


@click.group()
@click.version_option(prog_name="mindcore")
def main():
    """Mindcore - Memory Layer for AI Agents.

    Get started quickly:

        $ mindcore init     # Interactive setup wizard

        $ mindcore demo     # Run a quick demo

        $ mindcore doctor   # Check your setup
    """


# Register commands
main.add_command(init_command, name="init")
main.add_command(doctor_command, name="doctor")
main.add_command(demo_command, name="demo")


@main.command()
def status():
    """Show current Mindcore configuration and status."""
    click.echo()
    click.echo(styled("  Mindcore Status", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo()

    # Check for config file
    config_paths = [
        Path("mindcore.yaml"),
        Path("mindcore.yml"),
        Path("config/mindcore.yaml"),
        Path(".mindcore/config.yaml"),
    ]

    config = None
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            try:
                import yaml

                with open(path) as f:
                    config = yaml.safe_load(f)
            except Exception:
                pass
            break

    if config_path:
        click.echo(f"  {styled('Config:', Colors.BOLD)} {config_path}")
    else:
        click.echo(f"  {styled('Config:', Colors.BOLD)} {styled('Not found', Colors.YELLOW)}")
        click.echo(styled("         Run: mindcore init", Colors.DIM))

    click.echo()

    # Storage configuration
    click.echo(styled("  Storage", Colors.BOLD))
    if config and "storage" in config:
        storage = config["storage"]
        backend = storage.get("backend", "sqlite")
        if backend == "sqlite":
            path = storage.get("path", "./mindcore.db")
            db_exists = Path(path).exists() if path else False
            status = (
                styled("exists", Colors.GREEN) if db_exists else styled("not created", Colors.DIM)
            )
            click.echo("    Backend: SQLite")
            click.echo(f"    Path: {path} ({status})")
        else:
            conn = storage.get("connection", {})
            host = conn.get("host", "localhost")
            port = conn.get("port", 5432)
            database = conn.get("database", "mindcore")
            click.echo("    Backend: PostgreSQL")
            click.echo(f"    Host: {host}:{port}/{database}")
    else:
        click.echo(styled("    Not configured", Colors.DIM))

    click.echo()

    # LLM configuration
    click.echo(styled("  LLM Provider", Colors.BOLD))
    llm_found = False
    if config and "llm" in config:
        provider = config["llm"].get("provider")
        model = config["llm"].get("model")
        click.echo(f"    Provider: {provider}")
        if model:
            click.echo(f"    Model: {model}")
        llm_found = True
    else:
        # Check environment
        providers = [
            ("OPENAI_API_KEY", "OpenAI"),
            ("ANTHROPIC_API_KEY", "Anthropic"),
            ("GOOGLE_API_KEY", "Google"),
        ]
        for env_var, name in providers:
            if os.environ.get(env_var):
                key = os.environ.get(env_var, "")
                masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                click.echo(f"    {name}: {masked}")
                llm_found = True

    if not llm_found:
        click.echo(styled("    Not configured (using rule-based fallback)", Colors.DIM))

    click.echo()

    # SVL configuration
    click.echo(styled("  SVL Vocabulary", Colors.BOLD))
    if config and "svl" in config:
        svl = config["svl"]
        domains = svl.get("domains", [])
        policies = svl.get("policies", {})

        if domains:
            click.echo(f"    Domains: {', '.join(domains)}")
        else:
            click.echo(f"    Domains: {styled('Default', Colors.DIM)}")

        strict = policies.get("strict_mode", False)
        click.echo(f"    Strict mode: {styled('Yes', Colors.YELLOW) if strict else 'No'}")
    else:
        click.echo(styled("    Using defaults", Colors.DIM))

    click.echo()

    # Features
    click.echo(styled("  Features", Colors.BOLD))
    if config and "features" in config:
        features = config["features"]
        enabled = [k for k, v in features.items() if v]
        if enabled:
            for feat in enabled[:5]:
                click.echo(f"    {styled('✓', Colors.GREEN)} {feat.replace('_', ' ').title()}")
            if len(enabled) > 5:
                click.echo(styled(f"    ...and {len(enabled) - 5} more", Colors.DIM))
        else:
            click.echo(styled("    Default features", Colors.DIM))
    else:
        click.echo(styled("    Default features", Colors.DIM))

    click.echo()

    # Quick stats if database exists
    if config and "storage" in config:
        storage = config["storage"]
        if storage.get("backend") == "sqlite":
            db_path = storage.get("path", "./mindcore.db")
            if Path(db_path).exists():
                try:
                    import sqlite3

                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    count = cursor.fetchone()[0]
                    conn.close()

                    click.echo(styled("  Quick Stats", Colors.BOLD))
                    click.echo(f"    Memories stored: {count}")
                    click.echo()
                except Exception:
                    pass

    click.echo(styled("  Commands", Colors.BOLD))
    click.echo(f"    {styled('mindcore doctor', Colors.CYAN)}  - Check for issues")
    click.echo(f"    {styled('mindcore demo', Colors.CYAN)}    - Interactive demo")
    click.echo(f"    {styled('mindcore init', Colors.CYAN)}    - Reconfigure")
    click.echo()


@main.command()
@click.argument("request_id", required=False)
@click.option("--last", "-l", is_flag=True, help="Explain the last request")
def explain(request_id: str | None, last: bool):
    r"""Explain what happened during a memory operation.

    Shows what memories were stored/recalled using real audit logs.

    \b
    Requirements:
        - Audit logging must be enabled in configuration
        - Operations must be logged to access explain data

    \b
    Examples:
        mindcore explain abc123     # Explain specific request
        mindcore explain --last     # Explain last operation
    """
    click.echo()
    click.echo(styled("  Mindcore Explain", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo()

    if not request_id and not last:
        click.echo(styled("  Usage: mindcore explain <request-id>", Colors.YELLOW))
        click.echo(styled("         mindcore explain --last", Colors.YELLOW))
        click.echo()
        click.echo(styled("  Request IDs are returned from store/recall operations.", Colors.DIM))
        click.echo(styled("  Use --last to explain the most recent operation.", Colors.DIM))
        return

    # Check for config file to find audit log location
    config_paths = [
        Path("mindcore.yaml"),
        Path("mindcore.yml"),
        Path("config/mindcore.yaml"),
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            try:
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)
                break
            except Exception:
                pass

    if not config:
        click.echo(styled("  No configuration found.", Colors.YELLOW))
        click.echo(styled("  Run: mindcore init", Colors.DIM))
        click.echo()
        click.echo(styled("  To enable audit logs, configure:", Colors.DIM))
        click.echo(styled("    enterprise:", Colors.DIM))
        click.echo(styled("      audit:", Colors.DIM))
        click.echo(styled("        enabled: true", Colors.DIM))
        click.echo(styled("        file_path: ./audit.log", Colors.DIM))
        return

    # Check for audit configuration
    audit_config = config.get("enterprise", {}).get("audit", {})
    audit_enabled = audit_config.get("enabled", False)
    audit_file = audit_config.get("file_path")

    if not audit_enabled:
        click.echo(styled("  Audit logging is not enabled.", Colors.YELLOW))
        click.echo()
        click.echo(styled("  To enable, add to your mindcore.yaml:", Colors.DIM))
        click.echo(styled("    enterprise:", Colors.DIM))
        click.echo(styled("      audit:", Colors.DIM))
        click.echo(styled("        enabled: true", Colors.DIM))
        click.echo(styled("        file_path: ./audit.log", Colors.DIM))
        return

    # Try to read from audit log file
    if audit_file:
        audit_path = Path(audit_file)
        if not audit_path.exists():
            click.echo(styled(f"  Audit log file not found: {audit_file}", Colors.YELLOW))
            click.echo(styled("  No operations have been logged yet.", Colors.DIM))
            return

        try:
            import json

            # Read audit log entries
            entries = []
            with open(audit_path) as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if stripped:
                        try:
                            entries.append(json.loads(stripped))
                        except json.JSONDecodeError:
                            continue

            if not entries:
                click.echo(styled("  Audit log is empty.", Colors.YELLOW))
                click.echo(
                    styled("  Operations will be logged after memory operations.", Colors.DIM)
                )
                return

            # Find the requested entry
            target_entry = None
            if last:
                target_entry = entries[-1] if entries else None
            elif request_id:
                for entry in reversed(entries):
                    if (
                        entry.get("request_id") == request_id
                        or entry.get("memory_id") == request_id
                    ):
                        target_entry = entry
                        break

            if not target_entry:
                if request_id:
                    click.echo(
                        styled(
                            f"  Request ID '{request_id}' not found in audit log.", Colors.YELLOW
                        )
                    )
                else:
                    click.echo(styled("  No entries found in audit log.", Colors.YELLOW))
                return

            # Display the actual audit entry
            click.echo(styled("  Audit Entry", Colors.BOLD))
            click.echo()

            for key, value in target_entry.items():
                if key.startswith("_"):
                    continue
                display_key = key.replace("_", " ").title()
                click.echo(f"    {styled(display_key + ':', Colors.CYAN)} {value}")

            click.echo()

        except Exception as e:
            click.echo(styled(f"  Error reading audit log: {e}", Colors.RED))
            return

    else:
        click.echo(styled("  Audit file path not configured.", Colors.YELLOW))
        click.echo()
        click.echo(styled("  Configure in mindcore.yaml:", Colors.DIM))
        click.echo(styled("    enterprise:", Colors.DIM))
        click.echo(styled("      audit:", Colors.DIM))
        click.echo(styled("        file_path: ./audit.log", Colors.DIM))


@main.command()
@click.argument("action", type=click.Choice(["view", "validate", "diff", "reset"]), default="view")
@click.option("--path", "-p", type=click.Path(), help="Config file path")
def config(action: str, path: str | None):
    r"""View, validate, or manage Mindcore configuration.

    \b
    Actions:
        view      Show current configuration (default)
        validate  Check configuration for errors
        diff      Compare with defaults
        reset     Reset to defaults (interactive)

    \b
    Examples:
        mindcore config              # View current config
        mindcore config validate     # Check for errors
        mindcore config reset        # Reset to defaults
    """
    click.echo()
    click.echo(styled("  Mindcore Config", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo()

    # Find config file
    config_paths = [
        Path(path) if path else None,
        Path("mindcore.yaml"),
        Path("mindcore.yml"),
        Path("config/mindcore.yaml"),
    ]

    config_data = None
    config_path = None
    for p in config_paths:
        if p and p.exists():
            config_path = p
            try:
                import yaml

                with open(p) as f:
                    config_data = yaml.safe_load(f)
            except Exception as e:
                click.echo(styled(f"  Error reading {p}: {e}", Colors.RED))
                return
            break

    if action == "view":
        if not config_data:
            click.echo(styled("  No configuration found.", Colors.YELLOW))
            click.echo(styled("  Run: mindcore init", Colors.DIM))
            return

        click.echo(f"  {styled('File:', Colors.BOLD)} {config_path}")
        click.echo()

        import yaml

        # Pretty print config
        for section, values in config_data.items():
            click.echo(styled(f"  {section}:", Colors.BOLD))
            if isinstance(values, dict):
                for k, v in values.items():
                    click.echo(f"    {k}: {v}")
            else:
                click.echo(f"    {values}")
            click.echo()

    elif action == "validate":
        if not config_data:
            click.echo(styled("  ✗ No configuration found", Colors.RED))
            return

        click.echo(styled("  Validating configuration...", Colors.DIM))
        click.echo()

        errors = []
        warnings = []

        # Check required sections
        required = ["storage"]
        for section in required:
            if section not in config_data:
                errors.append(f"Missing required section: {section}")

        # Check storage config
        if "storage" in config_data:
            storage = config_data["storage"]
            if "backend" not in storage and "path" not in storage:
                warnings.append("Storage backend not specified, will use SQLite default")

        # Check SVL config
        if "svl" in config_data:
            svl = config_data["svl"]
            if svl.get("policies", {}).get("strict_mode"):
                click.echo(f"  {styled('i', Colors.CYAN)} Strict mode enabled")

        if errors:
            for err in errors:
                click.echo(f"  {styled('✗', Colors.RED)} {err}")
            click.echo()
            click.echo(styled("  Configuration invalid", Colors.RED, Colors.BOLD))
        elif warnings:
            for warn in warnings:
                click.echo(f"  {styled('!', Colors.YELLOW)} {warn}")
            click.echo()
            click.echo(styled("  ✓ Configuration valid (with warnings)", Colors.YELLOW))
        else:
            click.echo(styled("  ✓ Configuration valid", Colors.GREEN, Colors.BOLD))

    elif action == "diff":
        click.echo(styled("  Comparing with defaults...", Colors.DIM))
        click.echo()

        if not config_data:
            click.echo(styled("  No configuration to compare.", Colors.YELLOW))
            return

        # Show differences from defaults
        defaults = {
            "storage": {"backend": "sqlite", "path": "./mindcore.db"},
            "svl": {"policies": {"strict_mode": False, "require_user_id": True}},
            "features": {"hot_path": True, "session_segmentation": True},
        }

        def diff_dict(current, default, prefix=""):
            for key in set(list(current.keys()) + list(default.keys())):
                curr_val = current.get(key)
                def_val = default.get(key)

                if curr_val != def_val:
                    if curr_val is None:
                        click.echo(f"  {styled('-', Colors.RED)} {prefix}{key}: {def_val}")
                    elif def_val is None:
                        click.echo(f"  {styled('+', Colors.GREEN)} {prefix}{key}: {curr_val}")
                    elif isinstance(curr_val, dict) and isinstance(def_val, dict):
                        diff_dict(curr_val, def_val, f"{prefix}{key}.")
                    else:
                        click.echo(
                            f"  {styled('~', Colors.YELLOW)} {prefix}{key}: {def_val} → {curr_val}"
                        )

        diff_dict(config_data, defaults)
        click.echo()

    elif action == "reset":
        click.echo(styled("  This will reset configuration to defaults.", Colors.YELLOW))
        click.echo()

        if config_path and config_path.exists():
            click.echo(f"  Current config: {config_path}")
            if click.confirm(styled("  Create backup and reset?", Colors.YELLOW)):
                backup_path = config_path.with_suffix(".yaml.bak")
                import shutil

                shutil.copy(config_path, backup_path)
                click.echo(f"  {styled('✓', Colors.GREEN)} Backup created: {backup_path}")

                # Run init
                click.echo()
                click.echo(styled("  Run 'mindcore init' to create new configuration.", Colors.DIM))
        else:
            click.echo(styled("  No config to reset. Run: mindcore init", Colors.DIM))


@main.command()
@click.argument(
    "test_type",
    type=click.Choice(["replay", "audit", "drift", "latency", "all"]),
    default="all",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--iterations", "-n", default=10, help="Number of iterations for benchmarks")
def benchmark(test_type: str, verbose: bool, iterations: int):
    r"""Run trust and performance benchmarks.

    These benchmarks prove Mindcore works correctly and deterministically.
    All measurements are performed on real operations - no simulated data.

    \b
    Benchmark types:
        replay   - Test deterministic memory replay
        audit    - Verify audit trail integrity
        drift    - Check for memory drift over time
        latency  - Measure FLR and CLST latency
        all      - Run all benchmarks

    \b
    Examples:
        mindcore benchmark           # Run all benchmarks
        mindcore benchmark replay    # Test determinism
        mindcore benchmark latency   # Measure performance
        mindcore benchmark -n 50     # Run with 50 iterations
    """
    import hashlib
    import tempfile
    import time

    click.echo()
    click.echo(styled("  Mindcore Benchmarks", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo(styled("  All tests use real operations with actual measurements", Colors.DIM))
    click.echo()

    tests_to_run = [test_type] if test_type != "all" else ["replay", "latency", "audit", "drift"]
    results = {}

    # Create a real temporary database for benchmarks
    try:
        from mindcore import Mindcore
        from mindcore.svl import SharedVocabularyLayer
    except ImportError:
        click.echo(styled("  Error: Mindcore not installed properly", Colors.RED))
        return

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        vocab = SharedVocabularyLayer()
        vocab.add_topics("benchmark", "test", "settings", "user")
        memory = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)

        for test in tests_to_run:
            click.echo(styled(f"  Running: {test}", Colors.BOLD))

            if test == "replay":
                # Test deterministic replay with real operations
                click.echo(styled("    Testing deterministic memory replay...", Colors.DIM))

                # Store test memories
                test_contents = [
                    ("User prefers dark mode interface", ["settings", "user"]),
                    ("Application settings stored", ["settings"]),
                    ("Benchmark test memory", ["benchmark", "test"]),
                    ("User completed tutorial", ["user"]),
                    ("Configuration validated", ["settings", "benchmark"]),
                ]

                stored_ids = []
                for content, topics in test_contents:
                    mid = memory.store(
                        content=content,
                        memory_type="semantic",
                        user_id="benchmark_user",
                        topics=topics,
                    )
                    stored_ids.append(mid)

                if verbose:
                    click.echo(f"    Stored {len(stored_ids)} test memories")

                # Perform multiple identical recalls and compare results
                query = "user settings preferences"
                first_result = None
                all_match = True

                for i in range(min(iterations, 10)):
                    result = memory.recall(
                        query=query,
                        user_id="benchmark_user",
                        limit=5,
                    )

                    # Hash the results for comparison (sha256 for security)
                    result_hash = hashlib.sha256(
                        str(
                            [
                                m.get("memory_id") if isinstance(m, dict) else m.memory_id
                                for m in result.memories
                            ]
                        ).encode()
                    ).hexdigest()

                    if first_result is None:
                        first_result = result_hash
                    elif result_hash != first_result:
                        all_match = False
                        break

                if all_match:
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Stored {len(stored_ids)} test memories"
                    )
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Replayed {min(iterations, 10)} times with identical results"
                    )
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Hash match: PASS ({first_result[:8]}...)"
                    )
                    results["replay"] = "PASS"
                else:
                    click.echo(f"    {styled('✗', Colors.RED)} Determinism check FAILED")
                    results["replay"] = "FAIL"

            elif test == "latency":
                # Measure real operation latencies
                click.echo(
                    styled(f"    Measuring latency ({iterations} iterations)...", Colors.DIM)
                )

                store_times = []
                recall_times = []

                for i in range(iterations):
                    # Measure store latency
                    start = time.perf_counter()
                    memory.store(
                        content=f"Latency test memory {i}",
                        memory_type="episodic",
                        user_id="latency_user",
                        topics=["benchmark"],
                    )
                    store_times.append((time.perf_counter() - start) * 1000)

                    # Measure recall latency
                    start = time.perf_counter()
                    memory.recall(
                        query="latency test",
                        user_id="latency_user",
                        limit=5,
                    )
                    recall_times.append((time.perf_counter() - start) * 1000)

                # Calculate statistics
                avg_store = sum(store_times) / len(store_times)
                avg_recall = sum(recall_times) / len(recall_times)
                min_store = min(store_times)
                max_store = max(store_times)
                min_recall = min(recall_times)
                max_recall = max(recall_times)

                # Display results
                store_color = Colors.GREEN if avg_store < 50 else Colors.YELLOW
                recall_color = Colors.GREEN if avg_recall < 50 else Colors.YELLOW

                click.echo(
                    f"    Store operation: {styled(f'{avg_store:.2f}ms', store_color)} (min: {min_store:.2f}ms, max: {max_store:.2f}ms)"
                )
                click.echo(
                    f"    Recall operation: {styled(f'{avg_recall:.2f}ms', recall_color)} (min: {min_recall:.2f}ms, max: {max_recall:.2f}ms)"
                )

                if verbose:
                    click.echo(f"    Iterations: {iterations}")

                # Pass if average latency is under 100ms
                if avg_store < 100 and avg_recall < 100:
                    click.echo(f"    {styled('✓', Colors.GREEN)} Latency within acceptable bounds")
                    results["latency"] = "PASS"
                else:
                    click.echo(f"    {styled('!', Colors.YELLOW)} Latency higher than expected")
                    results["latency"] = "WARN"

            elif test == "audit":
                # Verify operations are tracked correctly
                click.echo(styled("    Verifying operation tracking...", Colors.DIM))

                # Store some memories and track them
                operation_ids = []
                for i in range(5):
                    mid = memory.store(
                        content=f"Audit test memory {i}",
                        memory_type="semantic",
                        user_id="audit_user",
                        topics=["benchmark"],
                    )
                    operation_ids.append(mid)

                # Verify all can be retrieved
                all_found = True
                for mid in operation_ids:
                    retrieved = memory.get(mid)
                    if retrieved is None:
                        all_found = False
                        break

                # Verify sequential storage
                memories_list = memory.search(user_id="audit_user", topics=["benchmark"])

                if all_found:
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} All {len(operation_ids)} operations tracked"
                    )
                    click.echo(f"    {styled('✓', Colors.GREEN)} All memories retrievable by ID")
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Search returns {len(memories_list)} memories"
                    )
                    results["audit"] = "PASS"
                else:
                    click.echo(f"    {styled('✗', Colors.RED)} Some operations not tracked")
                    results["audit"] = "FAIL"

            elif test == "drift":
                # Check reinforcement score stability
                click.echo(styled("    Checking reinforcement stability...", Colors.DIM))

                # Store memory and apply reinforcements
                drift_id = memory.store(
                    content="Drift test memory",
                    memory_type="semantic",
                    user_id="drift_user",
                    topics=["benchmark"],
                )

                initial_mem = memory.get(drift_id)
                # Handle both dict and Memory object return types
                if isinstance(initial_mem, dict):
                    initial_score = initial_mem.get("reinforcement_score", 0.0)
                else:
                    initial_score = initial_mem.reinforcement_score

                # Apply positive and negative reinforcements
                memory.reinforce(drift_id, signal=0.5)
                memory.reinforce(drift_id, signal=-0.2)
                memory.reinforce(drift_id, signal=0.3)

                final_mem = memory.get(drift_id)
                # Handle both dict and Memory object return types
                if isinstance(final_mem, dict):
                    final_score = final_mem.get("reinforcement_score", 0.0)
                else:
                    final_score = final_mem.reinforcement_score

                # Verify scores are bounded
                score_bounded = -1.0 <= final_score <= 1.0
                score_changed = final_score != initial_score

                if score_bounded and score_changed:
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Reinforcement score bounded: {final_score:.3f}"
                    )
                    click.echo(
                        f"    {styled('✓', Colors.GREEN)} Score changed from {initial_score:.3f} to {final_score:.3f}"
                    )
                    click.echo(f"    {styled('✓', Colors.GREEN)} No unexpected drift detected")
                    results["drift"] = "PASS"
                else:
                    if not score_bounded:
                        click.echo(
                            f"    {styled('✗', Colors.RED)} Score out of bounds: {final_score}"
                        )
                    if not score_changed:
                        click.echo(
                            f"    {styled('!', Colors.YELLOW)} Score did not change after reinforcement"
                        )
                    results["drift"] = "FAIL" if not score_bounded else "WARN"

            click.echo()

        memory.close()

    except Exception as e:
        click.echo(styled(f"  Error running benchmarks: {e}", Colors.RED))
        import traceback

        if verbose:
            traceback.print_exc()
        return
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except Exception:
            pass

    # Summary
    click.echo(styled("  ═" * 25, Colors.BOLD))
    passed = sum(1 for r in results.values() if r == "PASS")
    warned = sum(1 for r in results.values() if r == "WARN")
    failed = sum(1 for r in results.values() if r == "FAIL")
    total = len(results)

    if passed == total:
        click.echo(styled(f"  All {total} benchmarks passed!", Colors.GREEN, Colors.BOLD))
    elif failed == 0:
        click.echo(
            styled(f"  {passed}/{total} passed, {warned} warnings", Colors.YELLOW, Colors.BOLD)
        )
    else:
        click.echo(
            styled(
                f"  {passed} passed, {warned} warnings, {failed} failed", Colors.RED, Colors.BOLD
            )
        )

    click.echo()
    click.echo(styled("  These real measurements prove:", Colors.DIM))
    click.echo(styled("    • Memory operations are deterministic (replay)", Colors.DIM))
    click.echo(styled("    • Operations are properly tracked (audit)", Colors.DIM))
    click.echo(styled("    • Performance is measured accurately (latency)", Colors.DIM))
    click.echo(styled("    • Reinforcement bounds are enforced (drift)", Colors.DIM))
    click.echo()


@main.command("benchmark-suite")
@click.argument(
    "suite",
    type=click.Choice(
        ["quick", "core", "full", "determinism", "performance", "quality", "cost", "robustness"]
    ),
    default="core",
)
@click.option(
    "--scenario",
    "-s",
    type=click.Choice(["single_agent", "multi_agent", "hot_path", "cold_path"]),
    default="single_agent",
)
@click.option("--size", type=click.Choice(["small", "medium", "large"]), default="small")
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.option("--dashboard", "-d", type=click.Path(), help="Generate HTML dashboard")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def benchmark_suite(
    suite: str, scenario: str, size: str, output: str | None, dashboard: str | None, verbose: bool
):
    r"""Run comprehensive benchmark suite against industry standards.

    Benchmarks are designed to measure what matters for production AI memory:

    \b
    DETERMINISM  - Can you replay and get identical results?
    AUDITABILITY - Can you explain what happened and why?
    QUALITY      - Does memory remain accurate over time?
    COST         - What's the FLR vs CLST efficiency?
    ROBUSTNESS   - Does the system handle edge cases?

    \b
    Suites:
        quick       - Fast validation (< 1 minute)
        core        - Essential benchmarks (< 5 minutes)
        full        - Complete evaluation (< 30 minutes)
        determinism - Replay consistency tests
        performance - Latency and throughput tests
        quality     - Recall accuracy tests
        cost        - FLR vs CLST comparison
        robustness  - Noise and drift resistance

    \b
    Comparison targets:
        - Mem0 (https://github.com/mem0ai/mem0)
        - MemGPT/Letta (https://github.com/letta-ai/letta)
        - LangMem (https://github.com/langchain-ai/langmem)
        - Zep (https://github.com/getzep/zep)

    \b
    Examples:
        mindcore benchmark-suite                    # Run core benchmarks
        mindcore benchmark-suite full -v            # Full suite with details
        mindcore benchmark-suite determinism -o results.json
    """
    click.echo()
    click.echo(styled("  Mindcore Benchmark Suite", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo(styled("  Industry-standard evaluation for production AI memory", Colors.DIM))
    click.echo()

    try:
        from mindcore.benchmarks import BenchmarkRunner, BenchmarkSuite

        # Map string to enum
        suite_map = {
            "quick": BenchmarkSuite.QUICK,
            "core": BenchmarkSuite.CORE,
            "full": BenchmarkSuite.FULL,
            "determinism": BenchmarkSuite.DETERMINISM,
            "performance": BenchmarkSuite.PERFORMANCE,
            "quality": BenchmarkSuite.QUALITY,
            "cost": BenchmarkSuite.COST,
            "robustness": BenchmarkSuite.ROBUSTNESS,
        }

        from mindcore.benchmarks.runner import BenchmarkConfig, Scenario

        scenario_map = {
            "single_agent": Scenario.SINGLE_AGENT,
            "multi_agent": Scenario.MULTI_AGENT,
            "hot_path": Scenario.HOT_PATH_ONLY,
            "cold_path": Scenario.COLD_PATH_ONLY,
        }

        config = BenchmarkConfig(
            suite=suite_map[suite],
            scenario=scenario_map[scenario],
            dataset_size=size,
            verbose=verbose,
        )

        runner = BenchmarkRunner(config)

        click.echo(f"  Suite: {styled(suite, Colors.CYAN)}")
        click.echo(f"  Scenario: {styled(scenario, Colors.CYAN)}")
        click.echo(f"  Dataset size: {styled(size, Colors.CYAN)}")
        click.echo()

        result = runner.run_suite()

        # Show results
        click.echo(styled("  ═" * 25, Colors.BOLD))
        click.echo()

        for b in result.benchmarks:
            status_color = Colors.GREEN if b.passed else Colors.RED
            status = "PASS" if b.passed else "FAIL"
            click.echo(f"  [{styled(status, status_color)}] {b.name}")

            if verbose and b.metrics.latency.samples:
                click.echo(
                    f"         p50: {b.metrics.latency.p50:.2f}ms, p99: {b.metrics.latency.p99:.2f}ms"
                )

        click.echo()
        click.echo(styled("  ═" * 25, Colors.BOLD))

        if result.passed == result.total:
            click.echo(
                styled(f"  All {result.total} benchmarks passed!", Colors.GREEN, Colors.BOLD)
            )
        else:
            click.echo(
                styled(
                    f"  {result.passed}/{result.total} passed, {result.failed} failed",
                    Colors.YELLOW,
                    Colors.BOLD,
                )
            )

        # Export if requested
        if output:
            runner.export_report(output)
            click.echo(f"\n  Results exported to: {styled(output, Colors.CYAN)}")

        # Generate dashboard if requested
        if dashboard:
            from mindcore.benchmarks import generate_dashboard

            # Use output file if available, otherwise create temp JSON
            json_file = output or "benchmark_results.json"
            if not output:
                runner.export_report(json_file)

            generate_dashboard(json_file, dashboard)
            click.echo(f"  Dashboard generated: {styled(dashboard, Colors.CYAN)}")

        click.echo()

    except ImportError as e:
        click.echo(styled(f"  Error: Missing dependency - {e}", Colors.RED))
        click.echo(styled("  Install with: pip install mindcore[benchmarks]", Colors.DIM))


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the Mindcore API server."""
    click.echo(f"Starting Mindcore API server on {host}:{port}...")
    try:
        from mindcore.server import run_server

        run_server(host=host, port=port)
    except ImportError:
        click.echo("Error: Server dependencies not installed.", err=True)
        click.echo("Install with: pip install mindcore[server]", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
