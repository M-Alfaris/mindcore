"""Mindcore CLI - Interactive setup and management tools.

Usage:
    mindcore init          # Interactive setup wizard
    mindcore doctor        # Health check and diagnostics
    mindcore demo          # Run a quick demo
    mindcore serve         # Start the API server

Quick Start:
    $ pip install mindcore
    $ mindcore init
    $ mindcore demo
"""

import click

from .demo import demo_command
from .doctor import doctor_command
from .init import init_command


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
