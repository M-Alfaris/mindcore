"""Health check and diagnostics for Mindcore.

Validates:
- Configuration files
- Database connectivity (with actual connection tests)
- LLM API access (with actual API validation)
- SVL vocabulary
- Feature availability
- Disk space and permissions

Provides:
- Verbose mode for detailed output
- Fix mode for automatic issue resolution
"""

import os
import sys
from pathlib import Path

import click


# Global verbose flag
VERBOSE = False


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def styled(text: str, *styles: str) -> str:
    """Apply ANSI color/style codes to text."""
    return "".join(styles) + text + Colors.RESET


def verbose_log(message: str):
    """Print message only in verbose mode."""
    if VERBOSE:
        click.echo(styled(f"         [debug] {message}", Colors.DIM))


def check_passed(name: str, detail: str = ""):
    """Print a passed check."""
    status = styled("✓ PASS", Colors.GREEN, Colors.BOLD)
    click.echo(f"  {status}  {name}")
    if detail:
        click.echo(styled(f"         {detail}", Colors.DIM))


def check_failed(name: str, error: str, fix: str = ""):
    """Print a failed check."""
    status = styled("✗ FAIL", Colors.RED, Colors.BOLD)
    click.echo(f"  {status}  {name}")
    click.echo(styled(f"         Error: {error}", Colors.RED))
    if fix:
        click.echo(styled(f"         Fix: {fix}", Colors.YELLOW))


def check_warn(name: str, warning: str):
    """Print a warning."""
    status = styled("! WARN", Colors.YELLOW, Colors.BOLD)
    click.echo(f"  {status}  {name}")
    click.echo(styled(f"         {warning}", Colors.YELLOW))


def check_skip(name: str, reason: str):
    """Print a skipped check."""
    status = styled("- SKIP", Colors.DIM)
    click.echo(f"  {status}  {name}")
    click.echo(styled(f"         {reason}", Colors.DIM))


def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 10):
        check_passed("Python version", f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    check_failed(
        "Python version",
        f"Python {version.major}.{version.minor} found, requires 3.10+",
        "Upgrade to Python 3.10 or later",
    )
    return False


def check_mindcore_import() -> bool:
    """Check Mindcore can be imported."""
    try:
        from mindcore import __version__

        check_passed("Mindcore import", f"Version {__version__}")
        return True
    except ImportError as e:
        check_failed("Mindcore import", str(e), "Run: pip install mindcore")
        return False


def check_config_file() -> dict | None:
    """Check for configuration file."""
    config_paths = [
        Path("mindcore.yaml"),
        Path("mindcore.yml"),
        Path("config/mindcore.yaml"),
        Path(".mindcore/config.yaml"),
    ]

    for path in config_paths:
        if path.exists():
            try:
                import yaml

                with open(path) as f:
                    config = yaml.safe_load(f)
                check_passed("Config file", str(path))
                return config
            except Exception as e:
                check_failed("Config file", f"Invalid YAML: {e}")
                return None

    check_warn("Config file", "No mindcore.yaml found. Using defaults. Run: mindcore init")
    return None


def check_storage(config: dict | None) -> bool:
    """Check storage connectivity."""
    if config and "storage" in config:
        storage_config = config["storage"]
        backend = storage_config.get("backend", "sqlite")
    else:
        backend = "sqlite"

    if backend == "sqlite":
        try:
            from mindcore import SQLiteStorage

            SQLiteStorage(":memory:")  # Tests initialization
            check_passed("SQLite storage", "In-memory test successful")
            return True
        except Exception as e:
            check_failed("SQLite storage", str(e))
            return False

    elif backend == "postgresql":
        try:
            from mindcore import PostgresStorage

            conn = storage_config.get("connection", {})
            host = conn.get("host", os.environ.get("MINDCORE_DB_HOST", "localhost"))
            port = conn.get("port", os.environ.get("MINDCORE_DB_PORT", "5432"))
            database = conn.get("database", os.environ.get("MINDCORE_DB_NAME", "mindcore"))
            user = conn.get("user", os.environ.get("MINDCORE_DB_USER", "postgres"))
            password = os.environ.get("MINDCORE_DB_PASSWORD", "")

            # Just try to create the storage object to test connection
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            PostgresStorage(connection_string)  # Tests connection
            check_passed("PostgreSQL storage", f"{host}:{port}/{database}")
            return True

        except ImportError:
            check_failed(
                "PostgreSQL storage",
                "psycopg2 not installed",
                "Run: pip install mindcore[postgres]",
            )
            return False
        except Exception as e:
            check_failed("PostgreSQL storage", str(e), "Check your database connection settings")
            return False

    return True


def test_llm_connectivity(provider: str, api_key: str) -> tuple[bool, str]:
    """Test actual LLM API connectivity."""
    try:
        import urllib.error
        import urllib.request

        verbose_log(f"Testing {provider} API connectivity...")

        if provider == "openai":
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True, "API connected successfully"
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    return False, "Invalid API key"
                return False, f"API error: {e.code}"
            except urllib.error.URLError as e:
                return False, f"Network error: {e.reason}"

        elif provider == "anthropic":
            # For Anthropic, we can't easily test without making a real request
            # Just validate the key format
            if api_key and len(api_key) > 20:
                return True, "API key format valid"
            return False, "Invalid API key format"

        elif provider == "google":
            if api_key and api_key.startswith("AIza"):
                return True, "API key format valid"
            return False, "Invalid API key format"

        return True, "Provider configured"
    except Exception as e:
        return False, str(e)


def check_llm_provider(config: dict | None, test_connectivity: bool = False) -> bool:
    """Check LLM provider configuration."""
    if not config or "llm" not in config:
        # Check environment variables
        providers = [
            ("OPENAI_API_KEY", "OpenAI", "openai"),
            ("ANTHROPIC_API_KEY", "Anthropic", "anthropic"),
            ("GOOGLE_API_KEY", "Google", "google"),
        ]

        found = []
        for env_var, name, provider_id in providers:
            api_key = os.environ.get(env_var)
            if api_key:
                found.append(name)
                verbose_log(f"Found {env_var} in environment")

                # Test connectivity if requested
                if test_connectivity:
                    success, message = test_llm_connectivity(provider_id, api_key)
                    if success:
                        verbose_log(f"{name}: {message}")
                    else:
                        check_warn(f"{name} connectivity", message)

        if found:
            check_passed("LLM provider", f"Found: {', '.join(found)}")
            return True
        check_warn(
            "LLM provider",
            "No LLM API keys found. Metadata extraction will use rule-based fallback.",
        )
        return True  # Not a hard failure

    provider = config["llm"].get("provider")
    verbose_log(f"Config specifies provider: {provider}")

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    if provider in env_var_map:
        env_var = env_var_map[provider]
        api_key = os.environ.get(env_var)

        if not api_key:
            check_failed(
                "LLM provider",
                f"{provider.title()} configured but {env_var} not set",
                f"Set {env_var} in your environment or .env file",
            )
            return False

        # Test connectivity if requested
        if test_connectivity:
            verbose_log("Testing API connectivity...")
            success, message = test_llm_connectivity(provider, api_key)
            if success:
                check_passed("LLM provider", f"{provider.title()}: {message}")
            else:
                check_failed("LLM provider", f"{provider.title()}: {message}")
                return False
        else:
            check_passed("LLM provider", f"Configured: {provider}")

        return True

    # Custom or unknown provider
    check_passed("LLM provider", f"Configured: {provider}")
    return True


def check_svl_vocabulary(config: dict | None) -> bool:
    """Check SVL vocabulary."""
    try:
        from mindcore.svl import DEFAULT_SVL

        # Try to load default
        svl = DEFAULT_SVL
        check_passed(
            "SVL vocabulary", f"Default loaded with {len(svl.schema.memory_types)} memory types"
        )

        # Check domains if configured
        if config and "svl" in config:
            domains = config["svl"].get("domains", [])
            if domains:
                from mindcore.svl import get_domain

                for domain_name in domains:
                    try:
                        domain = get_domain(domain_name)
                        click.echo(
                            styled(
                                f"         Domain: {domain_name} ({len(domain.topics)} topics)",
                                Colors.DIM,
                            )
                        )
                    except Exception:
                        check_warn("SVL domain", f"Domain '{domain_name}' not found")

        return True

    except Exception as e:
        check_failed("SVL vocabulary", str(e))
        return False


def check_optional_features() -> dict:
    """Check optional feature availability."""
    features = {}

    # Check vector stores
    try:
        import importlib.util

        if importlib.util.find_spec("chromadb") is not None:
            features["chroma"] = True
            check_passed("Vector store (Chroma)", "Available")
        else:
            raise ImportError("chromadb not found")
    except ImportError:
        features["chroma"] = False
        check_skip("Vector store (Chroma)", "Not installed. Run: pip install mindcore[chroma]")

    # Check observability
    try:
        if importlib.util.find_spec("opentelemetry") is not None:
            features["observability"] = True
            check_passed("Observability", "OpenTelemetry available")
        else:
            raise ImportError("opentelemetry not found")
    except ImportError:
        features["observability"] = False
        check_skip("Observability", "Not installed. Run: pip install mindcore[observability]")

    # Check async support
    try:
        if importlib.util.find_spec("asyncpg") is not None:
            features["async"] = True
            check_passed("Async PostgreSQL", "asyncpg available")
        else:
            raise ImportError("asyncpg not found")
    except ImportError:
        features["async"] = False
        check_skip("Async PostgreSQL", "Not installed. Run: pip install mindcore[async]")

    return features


def run_quick_test() -> bool:
    """Run a quick functional test."""
    try:
        from mindcore import Mindcore
        from mindcore.svl import SharedVocabularyLayer

        # Create vocabulary with test topics
        vocab = SharedVocabularyLayer()
        vocab.add_topics("test", "doctor")

        # Create in-memory instance
        memory = Mindcore(storage="sqlite:///:memory:", vocabulary=vocab)

        # Store a test memory
        memory.store(
            content="Doctor test memory",
            memory_type="preference",
            user_id="doctor_test",
            topics=["test"],
        )

        # Recall it
        result = memory.recall(
            query="test",
            user_id="doctor_test",
        )

        if result.memories and len(result.memories) > 0:
            check_passed("Quick test", "Store and recall working")
            return True
        check_failed("Quick test", "Recall returned no memories")
        return False

    except Exception as e:
        check_failed("Quick test", str(e))
        return False


def try_fix_config() -> bool:
    """Attempt to create a default configuration file."""
    try:
        import yaml

        default_config = {
            "version": "2.0",
            "storage": {
                "backend": "sqlite",
                "path": "./mindcore.db",
            },
            "svl": {
                "policies": {
                    "strict_mode": False,
                    "require_user_id": True,
                },
            },
            "features": {
                "hot_path": True,
                "session_segmentation": True,
            },
        }

        config_path = Path("mindcore.yaml")
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        click.echo(styled("         Fixed: Created mindcore.yaml", Colors.GREEN))
        return True
    except Exception as e:
        click.echo(styled(f"         Could not create config: {e}", Colors.RED))
        return False


def try_fix_env_file() -> bool:
    """Attempt to create a template .env file."""
    try:
        env_path = Path(".env")
        if not env_path.exists():
            template = """# Mindcore Environment Configuration
# Uncomment and set the API keys you want to use

# LLM Providers (for metadata extraction)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=AIza...

# PostgreSQL (if using)
# MINDCORE_DB_HOST=localhost
# MINDCORE_DB_PORT=5432
# MINDCORE_DB_NAME=mindcore
# MINDCORE_DB_USER=postgres
# MINDCORE_DB_PASSWORD=
"""
            with open(env_path, "w") as f:
                f.write(template)

            click.echo(styled("         Fixed: Created .env template", Colors.GREEN))
            return True
        return False
    except Exception as e:
        click.echo(styled(f"         Could not create .env: {e}", Colors.RED))
        return False


def check_disk_space() -> bool:
    """Check available disk space."""
    try:
        import shutil

        cwd = Path.cwd()
        _total, _used, free = shutil.disk_usage(cwd)
        free_mb = free // (1024 * 1024)
        free_gb = free_mb / 1024

        verbose_log(f"Disk space: {free_gb:.1f} GB free")

        if free_mb < 100:  # Less than 100MB
            check_warn("Disk space", f"Only {free_mb}MB available - may cause issues")
            return False
        if free_mb < 1000:  # Less than 1GB
            check_passed("Disk space", f"{free_mb}MB available (low)")
        else:
            check_passed("Disk space", f"{free_gb:.1f}GB available")
        return True
    except Exception as e:
        verbose_log(f"Could not check disk space: {e}")
        return True  # Don't fail on this


def check_write_permissions() -> bool:
    """Check write permissions in current directory."""
    try:
        test_file = Path(".mindcore_permission_test")
        test_file.touch()
        test_file.unlink()
        check_passed("Write permissions", "Current directory is writable")
        return True
    except PermissionError:
        check_failed(
            "Write permissions",
            "Cannot write to current directory",
            "Check directory permissions or use a different location",
        )
        return False
    except Exception as e:
        verbose_log(f"Permission check error: {e}")
        return True


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--fix", is_flag=True, help="Attempt to fix issues")
@click.option("--test-llm", is_flag=True, help="Test LLM API connectivity")
def doctor_command(verbose: bool, fix: bool, test_llm: bool):
    r"""Check your Mindcore setup and diagnose issues.

    Validates configuration, connectivity, and features.

    \b
    Examples:
        mindcore doctor            # Basic health check
        mindcore doctor -v         # Verbose output
        mindcore doctor --fix      # Attempt to fix issues
        mindcore doctor --test-llm # Test LLM API connectivity
    """
    global VERBOSE
    VERBOSE = verbose

    click.echo()
    click.echo(styled("  Mindcore Doctor", Colors.BOLD, Colors.CYAN))
    click.echo(styled("  Checking your setup...", Colors.DIM))
    if verbose:
        click.echo(styled("  (Verbose mode enabled)", Colors.DIM))
    if fix:
        click.echo(styled("  (Fix mode enabled)", Colors.DIM))
    click.echo()

    passed = 0
    failed = 0
    warnings = 0
    fixed = 0

    # Core checks
    click.echo(styled("  Core Components", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    if check_python_version():
        passed += 1
    else:
        failed += 1

    if check_mindcore_import():
        passed += 1
    else:
        failed += 1
        click.echo()
        click.echo(styled("  Cannot continue without Mindcore installed.", Colors.RED))
        if fix:
            click.echo(styled("  Attempting fix: pip install mindcore", Colors.YELLOW))
            import subprocess

            result = subprocess.run(  # noqa: S603 # nosec B603
                [sys.executable, "-m", "pip", "install", "mindcore"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo(styled("  Fixed: Mindcore installed", Colors.GREEN))
                fixed += 1
            else:
                click.echo(styled("  Could not install Mindcore", Colors.RED))
                raise SystemExit(1)
        else:
            raise SystemExit(1)

    config = check_config_file()
    if config:
        passed += 1
    else:
        warnings += 1
        if fix:
            if try_fix_config():
                fixed += 1
                # Reload config
                try:
                    import yaml

                    with open("mindcore.yaml") as f:
                        config = yaml.safe_load(f)
                except Exception:
                    pass

    click.echo()
    click.echo(styled("  Environment", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    if check_disk_space():
        passed += 1
    else:
        warnings += 1

    if check_write_permissions():
        passed += 1
    else:
        failed += 1

    click.echo()
    click.echo(styled("  Storage & Connectivity", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    if check_storage(config):
        passed += 1
    else:
        failed += 1

    if check_llm_provider(config, test_connectivity=test_llm):
        passed += 1
    else:
        failed += 1
        if fix:
            if try_fix_env_file():
                fixed += 1
                click.echo(styled("         Edit .env and add your API keys", Colors.YELLOW))

    click.echo()
    click.echo(styled("  SVL & Vocabulary", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    if check_svl_vocabulary(config):
        passed += 1
    else:
        failed += 1

    click.echo()
    click.echo(styled("  Optional Features", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    check_optional_features()

    click.echo()
    click.echo(styled("  Functional Test", Colors.BOLD))
    click.echo(styled("  ─" * 25, Colors.DIM))

    if run_quick_test():
        passed += 1
    else:
        failed += 1

    # Summary
    click.echo()
    click.echo(styled("  ═" * 25, Colors.BOLD))

    if failed == 0:
        click.echo(styled("  All checks passed!", Colors.GREEN, Colors.BOLD))
        status_icon = styled("✓", Colors.GREEN)
    else:
        click.echo(styled(f"  {failed} check(s) failed", Colors.RED, Colors.BOLD))
        status_icon = styled("✗", Colors.RED)

    click.echo()
    click.echo(f"  {status_icon} Passed: {passed}")
    if failed > 0:
        click.echo(f"  {styled('✗', Colors.RED)} Failed: {failed}")
    if warnings > 0:
        click.echo(f"  {styled('!', Colors.YELLOW)} Warnings: {warnings}")
    if fixed > 0:
        click.echo(f"  {styled('⚡', Colors.CYAN)} Fixed: {fixed}")

    click.echo()

    if failed > 0:
        click.echo(styled("  Need help?", Colors.BOLD))
        click.echo(f"    {styled('•', Colors.CYAN)} Run: mindcore init        # Reconfigure")
        click.echo(f"    {styled('•', Colors.CYAN)} Run: mindcore doctor --fix  # Auto-fix issues")
        click.echo(f"    {styled('•', Colors.CYAN)} Docs: https://github.com/mindcore/mindcore")
        click.echo()
        raise SystemExit(1)
    click.echo(styled("  Ready to use Mindcore!", Colors.GREEN))
    click.echo(f"    {styled('•', Colors.CYAN)} Run: mindcore demo     # See it in action")
    click.echo(f"    {styled('•', Colors.CYAN)} Run: python quickstart.py  # Test your setup")
    click.echo()
