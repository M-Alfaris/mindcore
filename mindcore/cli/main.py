"""
Mindcore CLI - Main command-line interface.

Provides commands for:
- Downloading and managing GGUF models
- Checking system status
- Configuration management
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click package is required for CLI. Install with: pip install click")
    sys.exit(1)

from .models import (
    RECOMMENDED_MODELS,
    DEFAULT_MODEL,
    get_model_info,
    get_default_models_dir,
    format_model_table,
)


def get_version() -> str:
    """Get Mindcore version."""
    try:
        from mindcore import __version__

        return __version__
    except ImportError:
        return "unknown"


@click.group()
@click.version_option(version=get_version(), prog_name="mindcore")
def cli():
    """
    Mindcore CLI - Intelligent Memory Management for AI Agents

    Download models, check status, and manage your Mindcore installation.

    \b
    Quick Start:
      mindcore download-model              # Download default model
      export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
      python -c "from mindcore import MindcoreClient; print('Ready!')"
    """
    pass


@cli.command("download-model")
@click.option(
    "--model",
    "-m",
    type=click.Choice(list(RECOMMENDED_MODELS.keys())),
    default=DEFAULT_MODEL,
    help=f"Model to download (default: {DEFAULT_MODEL})",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory to save model (default: ~/.mindcore/models)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing model file")
def download_model(model: str, output_dir: Optional[str], force: bool):
    """
    Download a GGUF model for local inference.

    \b
    Examples:
      mindcore download-model                    # Download default (Llama 3.2 3B)
      mindcore download-model -m qwen2.5-3b     # Download Qwen 2.5 3B
      mindcore download-model -o ./models       # Custom output directory
    """
    model_info = get_model_info(model)
    if not model_info:
        click.echo(f"Error: Unknown model '{model}'", err=True)
        sys.exit(1)

    # Determine output directory
    if output_dir:
        models_dir = Path(output_dir)
    else:
        models_dir = Path(get_default_models_dir())

    # Create directory if needed
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / model_info.filename

    # Check if file exists
    if output_path.exists() and not force:
        click.echo(f"Model already exists: {output_path}")
        click.echo(f"Use --force to overwrite")
        click.echo()
        click.echo("To use this model, set:")
        click.echo(f"  export MINDCORE_LLAMA_MODEL_PATH={output_path}")
        return

    click.echo(f"Downloading {model_info.name}...")
    click.echo(f"  URL: {model_info.url}")
    click.echo(f"  Size: ~{model_info.size_gb} GB")
    click.echo(f"  Destination: {output_path}")
    click.echo()

    # Download with progress
    try:
        _download_with_progress(model_info.url, output_path)
    except KeyboardInterrupt:
        click.echo("\nDownload cancelled.")
        if output_path.exists():
            output_path.unlink()
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError downloading model: {e}", err=True)
        if output_path.exists():
            output_path.unlink()
        sys.exit(1)

    click.echo()
    click.echo(click.style("Download complete!", fg="green", bold=True))
    click.echo()
    click.echo("To use this model, set the environment variable:")
    click.echo(click.style(f"  export MINDCORE_LLAMA_MODEL_PATH={output_path}", fg="cyan"))
    click.echo()
    click.echo("Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)")


def _download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    try:
        import urllib.request
        import shutil

        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with click.progressbar(
                length=total_size,
                label="Downloading",
                show_eta=True,
                show_percent=True,
            ) as bar:
                with open(output_path, "wb") as out_file:
                    downloaded = 0
                    chunk_size = 1024 * 1024  # 1MB chunks

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        bar.update(len(chunk))

    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download: {e}")


@cli.command("list-models")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
def list_models(verbose: bool):
    """
    List available models for download.

    \b
    Examples:
      mindcore list-models           # Show model table
      mindcore list-models -v        # Show detailed info
    """
    if verbose:
        for key, model in RECOMMENDED_MODELS.items():
            default_marker = " (DEFAULT)" if key == DEFAULT_MODEL else ""
            click.echo(click.style(f"\n{key}{default_marker}", fg="cyan", bold=True))
            click.echo(f"  Name: {model.name}")
            click.echo(f"  Description: {model.description}")
            click.echo(f"  Base Model: {model.base_model}")
            click.echo(f"  Size: {model.size_gb} GB")
            click.echo(f"  Min RAM: {model.min_ram_gb} GB")
            click.echo(f"  Quantization: {model.quantization}")
            click.echo(f"  Recommended for: {', '.join(model.recommended_for)}")
            click.echo(f"  URL: {model.url}")
    else:
        click.echo(format_model_table())

    click.echo()
    click.echo(f"Default model: {click.style(DEFAULT_MODEL, fg='green')}")
    click.echo()
    click.echo("Download with: mindcore download-model -m <model-key>")


@cli.command("status")
def status():
    """
    Check Mindcore installation and configuration status.

    Shows:
    - Installed version
    - Configured LLM providers
    - Model availability
    - Database configuration
    """
    click.echo(click.style("Mindcore Status", fg="cyan", bold=True))
    click.echo("=" * 50)

    # Version
    version = get_version()
    click.echo(f"Version: {version}")
    click.echo()

    # LLM Provider Configuration
    click.echo(click.style("LLM Providers:", bold=True))

    # Check llama.cpp
    llama_path = os.getenv("MINDCORE_LLAMA_MODEL_PATH")
    if llama_path:
        path = Path(llama_path).expanduser()
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            click.echo(f"  llama.cpp: {click.style('Configured', fg='green')}")
            click.echo(f"    Model: {path.name}")
            click.echo(f"    Size: {size_mb:.1f} MB")
        else:
            click.echo(f"  llama.cpp: {click.style('Model not found', fg='yellow')}")
            click.echo(f"    Path: {llama_path}")
    else:
        click.echo(f"  llama.cpp: {click.style('Not configured', fg='yellow')}")
        click.echo("    Set MINDCORE_LLAMA_MODEL_PATH to enable")

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("MINDCORE_OPENAI_BASE_URL")
    if openai_key:
        if openai_base:
            click.echo(f"  OpenAI-compatible: {click.style('Configured', fg='green')}")
            click.echo(f"    Base URL: {openai_base}")
        else:
            click.echo(f"  OpenAI: {click.style('Configured', fg='green')}")
            click.echo(f"    API Key: {openai_key[:8]}...")
    else:
        click.echo(f"  OpenAI: {click.style('Not configured', fg='yellow')}")
        click.echo("    Set OPENAI_API_KEY to enable")

    click.echo()

    # Check installed models
    click.echo(click.style("Installed Models:", bold=True))
    models_dir = Path(get_default_models_dir())
    if models_dir.exists():
        gguf_files = list(models_dir.glob("*.gguf"))
        if gguf_files:
            for model_path in gguf_files:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                click.echo(f"  {model_path.name} ({size_mb:.1f} MB)")
        else:
            click.echo("  No models found in ~/.mindcore/models")
    else:
        click.echo("  Models directory not created yet")
        click.echo("  Run: mindcore download-model")

    click.echo()

    # Quick start hints
    if not llama_path and not openai_key:
        click.echo(click.style("Quick Start:", fg="yellow", bold=True))
        click.echo("  1. Download a model:")
        click.echo("     mindcore download-model")
        click.echo()
        click.echo("  2. Set the model path:")
        click.echo(
            "     export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        )
        click.echo()
        click.echo("  Or use OpenAI as fallback:")
        click.echo("     export OPENAI_API_KEY=sk-...")


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(show: bool):
    """
    Show or manage Mindcore configuration.

    \b
    Examples:
      mindcore config --show     # Show current config
    """
    if show:
        try:
            from mindcore.core import ConfigLoader

            loader = ConfigLoader()
            llm_config = loader.get_llm_config()

            click.echo(click.style("Current Configuration:", fg="cyan", bold=True))
            click.echo(f"Config file: {loader.config_path}")
            click.echo()

            click.echo("LLM Settings:")
            click.echo(f"  Provider: {llm_config['provider']}")
            click.echo()

            click.echo("  llama.cpp:")
            click.echo(f"    model_path: {llm_config['llama_cpp']['model_path'] or 'Not set'}")
            click.echo(f"    n_ctx: {llm_config['llama_cpp']['n_ctx']}")
            click.echo(f"    n_gpu_layers: {llm_config['llama_cpp']['n_gpu_layers']}")
            click.echo()

            click.echo("  OpenAI:")
            api_key = llm_config["openai"]["api_key"]
            click.echo(f"    api_key: {api_key[:8] + '...' if api_key else 'Not set'}")
            click.echo(f"    base_url: {llm_config['openai']['base_url'] or 'Default (OpenAI)'}")
            click.echo(f"    model: {llm_config['openai']['model']}")
            click.echo()

            click.echo("  Defaults:")
            click.echo(f"    temperature: {llm_config['defaults']['temperature']}")
            click.echo(
                f"    max_tokens_enrichment: {llm_config['defaults']['max_tokens_enrichment']}"
            )
            click.echo(f"    max_tokens_context: {llm_config['defaults']['max_tokens_context']}")

        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("Use --show to display current configuration")
        click.echo("Edit config.yaml directly for changes")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
