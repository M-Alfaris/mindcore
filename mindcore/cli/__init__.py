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
