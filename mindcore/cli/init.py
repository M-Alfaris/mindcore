"""Interactive onboarding wizard for Mindcore.

Guides users through:
1. Persona selection (developer, startup, enterprise, etc.)
2. Storage setup (SQLite, PostgreSQL)
3. LLM provider configuration
4. SVL vocabulary selection
5. Policy configuration
6. Optional integrations
"""

import os
from pathlib import Path

import click


# ANSI colors for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def styled(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    return "".join(styles) + text + Colors.RESET


def print_banner():
    """Print the Mindcore welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ███╗██╗███╗   ██╗██████╗  ██████╗ ██████╗ ██████╗   ║
    ║   ████╗ ████║██║████╗  ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗  ║
    ║   ██╔████╔██║██║██╔██╗ ██║██║  ██║██║     ██║   ██║██████╔╝  ║
    ║   ██║╚██╔╝██║██║██║╚██╗██║██║  ██║██║     ██║   ██║██╔══██╗  ║
    ║   ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝╚██████╗╚██████╔╝██║  ██║  ║
    ║   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝  ║
    ║                                                              ║
    ║              Memory Layer for AI Agents                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    click.echo(styled(banner, Colors.CYAN, Colors.BOLD))


def print_step(step: int, total: int, title: str):
    """Print a step header."""
    click.echo()
    click.echo(styled(f"  Step {step}/{total}: {title}", Colors.BOLD, Colors.BLUE))
    click.echo(styled("  " + "─" * 50, Colors.DIM))


PERSONAS = {
    "1": {
        "name": "Solo Developer",
        "desc": "Personal projects, experimentation, learning",
        "storage": "sqlite",
        "features": ["basic"],
        "complexity": "minimal",
    },
    "2": {
        "name": "Freelancer / Consultant",
        "desc": "Building AI solutions for clients",
        "storage": "sqlite",
        "features": ["basic", "multi_agent"],
        "complexity": "simple",
    },
    "3": {
        "name": "Startup",
        "desc": "Fast iteration, scale later",
        "storage": "postgresql",
        "features": ["basic", "multi_agent", "api"],
        "complexity": "standard",
    },
    "4": {
        "name": "AI Research Team / Lab",
        "desc": "Experimentation, flexible architecture",
        "storage": "postgresql",
        "features": ["basic", "multi_agent", "federation", "observability"],
        "complexity": "advanced",
    },
    "5": {
        "name": "AI Team (Mid-size Business)",
        "desc": "Production AI systems, existing infrastructure",
        "storage": "postgresql",
        "features": ["basic", "multi_agent", "api", "enterprise"],
        "complexity": "advanced",
    },
    "6": {
        "name": "Enterprise AI Team",
        "desc": "Compliance, security, scale, governance",
        "storage": "postgresql",
        "features": ["all"],
        "complexity": "full",
    },
    "7": {
        "name": "C-Suite Evaluation",
        "desc": "Quick demo for technical evaluation",
        "storage": "sqlite",
        "features": ["demo"],
        "complexity": "demo",
    },
}

DOMAINS = {
    "1": {"name": "Customer Service", "domains": ["customer_service"]},
    "2": {"name": "E-commerce", "domains": ["ecommerce"]},
    "3": {"name": "SaaS / Software", "domains": ["saas"]},
    "4": {"name": "Healthcare", "domains": ["healthcare"]},
    "5": {"name": "Finance / Banking", "domains": ["finance"]},
    "6": {"name": "Education", "domains": ["education"]},
    "7": {"name": "HR / Recruiting", "domains": ["hr"]},
    "8": {"name": "Custom / General", "domains": []},
}


def select_persona() -> dict:
    """Let user select their persona."""
    click.echo()
    click.echo(styled("  Who are you?", Colors.BOLD))
    click.echo()

    for key, persona in PERSONAS.items():
        click.echo(
            f"    {styled(key, Colors.CYAN, Colors.BOLD)}) {styled(persona['name'], Colors.BOLD)}"
        )
        click.echo(styled(f"       {persona['desc']}", Colors.DIM))

    click.echo()
    choice = click.prompt(
        styled("  Select your role", Colors.YELLOW),
        type=click.Choice(list(PERSONAS.keys())),
        default="1",
    )

    selected = PERSONAS[choice]
    click.echo()
    click.echo(styled(f"  ✓ Selected: {selected['name']}", Colors.GREEN))
    return selected


def select_domain() -> dict:
    """Let user select their domain/industry."""
    click.echo()
    click.echo(styled("  What domain will your AI agent work in?", Colors.BOLD))
    click.echo()

    for key, domain in DOMAINS.items():
        click.echo(f"    {styled(key, Colors.CYAN, Colors.BOLD)}) {domain['name']}")

    click.echo()
    choice = click.prompt(
        styled("  Select domain", Colors.YELLOW),
        type=click.Choice(list(DOMAINS.keys())),
        default="8",
    )

    selected = DOMAINS[choice]
    click.echo(styled(f"  ✓ Selected: {selected['name']}", Colors.GREEN))
    return selected


def configure_storage(persona: dict) -> dict:
    """Configure storage based on persona."""
    recommended = persona["storage"]

    click.echo()
    click.echo(styled("  Storage Configuration", Colors.BOLD))
    click.echo()

    if recommended == "sqlite":
        click.echo(
            f"    {styled('1', Colors.CYAN, Colors.BOLD)}) SQLite {styled('(Recommended for you)', Colors.GREEN)}"
        )
        click.echo(styled("       Zero setup, great for development", Colors.DIM))
        click.echo(f"    {styled('2', Colors.CYAN, Colors.BOLD)}) PostgreSQL")
        click.echo(styled("       Production-ready, requires setup", Colors.DIM))
    else:
        click.echo(f"    {styled('1', Colors.CYAN, Colors.BOLD)}) SQLite")
        click.echo(styled("       Zero setup, great for development", Colors.DIM))
        click.echo(
            f"    {styled('2', Colors.CYAN, Colors.BOLD)}) PostgreSQL {styled('(Recommended for you)', Colors.GREEN)}"
        )
        click.echo(styled("       Production-ready, scalable", Colors.DIM))

    click.echo()
    choice = click.prompt(
        styled("  Select storage", Colors.YELLOW),
        type=click.Choice(["1", "2"]),
        default="1" if recommended == "sqlite" else "2",
    )

    if choice == "1":
        db_path = click.prompt(styled("  SQLite file path", Colors.YELLOW), default="./mindcore.db")
        config = {"type": "sqlite", "path": db_path}
        click.echo(styled(f"  ✓ SQLite configured: {db_path}", Colors.GREEN))
    else:
        click.echo()
        click.echo(styled("  PostgreSQL connection:", Colors.DIM))
        host = click.prompt("    Host", default="localhost")
        port = click.prompt("    Port", default="5432")
        database = click.prompt("    Database", default="mindcore")
        user = click.prompt("    User", default="postgres")
        password = click.prompt("    Password", hide_input=True, default="")

        config = {
            "type": "postgresql",
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        click.echo(styled(f"  ✓ PostgreSQL configured: {host}:{port}/{database}", Colors.GREEN))

    return config


def configure_llm() -> dict:
    """Configure LLM provider."""
    click.echo()
    click.echo(styled("  LLM Provider (for metadata extraction)", Colors.BOLD))
    click.echo()
    click.echo(f"    {styled('1', Colors.CYAN, Colors.BOLD)}) OpenAI (GPT-4)")
    click.echo(f"    {styled('2', Colors.CYAN, Colors.BOLD)}) Anthropic (Claude)")
    click.echo(f"    {styled('3', Colors.CYAN, Colors.BOLD)}) Google (Gemini)")
    click.echo(f"    {styled('4', Colors.CYAN, Colors.BOLD)}) Custom / Local LLM")
    click.echo(f"    {styled('5', Colors.CYAN, Colors.BOLD)}) Skip (configure later)")

    click.echo()
    choice = click.prompt(
        styled("  Select provider", Colors.YELLOW),
        type=click.Choice(["1", "2", "3", "4", "5"]),
        default="5",
    )

    config = {"provider": None, "api_key": None, "model": None}

    if choice == "1":
        config["provider"] = "openai"
        config["model"] = "gpt-4o-mini"
        api_key = click.prompt(
            "    OpenAI API Key", hide_input=True, default=os.environ.get("OPENAI_API_KEY", "")
        )
        if api_key:
            config["api_key"] = api_key
            click.echo(styled("  ✓ OpenAI configured", Colors.GREEN))
        else:
            click.echo(styled("  ! API key not set, add OPENAI_API_KEY to .env", Colors.YELLOW))
    elif choice == "2":
        config["provider"] = "anthropic"
        config["model"] = "claude-3-haiku-20240307"
        api_key = click.prompt(
            "    Anthropic API Key",
            hide_input=True,
            default=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        if api_key:
            config["api_key"] = api_key
            click.echo(styled("  ✓ Anthropic configured", Colors.GREEN))
        else:
            click.echo(styled("  ! API key not set, add ANTHROPIC_API_KEY to .env", Colors.YELLOW))
    elif choice == "3":
        config["provider"] = "google"
        config["model"] = "gemini-2.0-flash"
        api_key = click.prompt(
            "    Google API Key", hide_input=True, default=os.environ.get("GOOGLE_API_KEY", "")
        )
        if api_key:
            config["api_key"] = api_key
            click.echo(styled("  ✓ Google configured", Colors.GREEN))
        else:
            click.echo(styled("  ! API key not set, add GOOGLE_API_KEY to .env", Colors.YELLOW))
    elif choice == "4":
        config["provider"] = "custom"
        click.echo(styled("  ✓ Will use custom LLM (configure in mindcore.yaml)", Colors.GREEN))
    else:
        click.echo(styled("  ✓ Skipped LLM configuration", Colors.DIM))

    return config


def configure_policies(persona: dict) -> dict:
    """Configure SVL policies based on persona."""
    complexity = persona.get("complexity", "minimal")

    click.echo()
    click.echo(styled("  Policy Configuration", Colors.BOLD))
    click.echo()

    if complexity in ["minimal", "simple", "demo"]:
        click.echo(styled("  Using sensible defaults (recommended for your setup)", Colors.DIM))
        click.echo()

        config = {
            "strict_mode": False,
            "require_user_id": True,
            "max_content_length": 10000,
            "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
        }

        click.echo(styled("  ✓ Default policies applied", Colors.GREEN))
        return config

    # Advanced configuration for enterprise/research
    click.echo("  Configure validation policies:")
    click.echo()

    strict = click.confirm(
        styled("    Enable strict mode?", Colors.YELLOW)
        + styled(" (reject invalid metadata)", Colors.DIM),
        default=complexity == "full",
    )

    require_topics = click.confirm(
        styled("    Require topics on all memories?", Colors.YELLOW), default=False
    )

    max_length = click.prompt(
        styled("    Max content length", Colors.YELLOW), default=10000, type=int
    )

    config = {
        "strict_mode": strict,
        "require_user_id": True,
        "require_topics": require_topics,
        "max_content_length": max_length,
        "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
    }

    click.echo()
    click.echo(styled("  ✓ Custom policies configured", Colors.GREEN))
    return config


def generate_config_files(
    persona: dict,
    domain: dict,
    storage: dict,
    llm: dict,
    policies: dict,
    output_dir: Path,
):
    """Generate configuration files."""
    import yaml

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate mindcore.yaml
    config = {
        "version": "2.0",
        "storage": {},
        "svl": {
            "domains": domain.get("domains", []),
            "policies": policies,
        },
        "features": {
            "hot_path": True,
            "session_segmentation": True,
        },
    }

    if storage["type"] == "sqlite":
        config["storage"]["backend"] = "sqlite"
        config["storage"]["path"] = storage["path"]
    else:
        config["storage"]["backend"] = "postgresql"
        config["storage"]["connection"] = {
            "host": storage["host"],
            "port": int(storage["port"]),
            "database": storage["database"],
            "user": storage["user"],
        }

    if llm["provider"]:
        config["llm"] = {
            "provider": llm["provider"],
            "model": llm["model"],
        }

    # Add feature flags based on persona
    features = persona.get("features", [])
    if "multi_agent" in features:
        config["features"]["multi_agent"] = True
    if "federation" in features:
        config["features"]["federation"] = True
    if "enterprise" in features or "all" in features:
        config["features"]["audit_logging"] = True
        config["features"]["encryption"] = True
        config["features"]["rate_limiting"] = True
    if "observability" in features or "all" in features:
        config["features"]["observability"] = True

    config_path = output_dir / "mindcore.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Generate .env file
    env_lines = [
        "# Mindcore Environment Configuration",
        f"# Generated for: {persona['name']}",
        "",
    ]

    if storage["type"] == "postgresql" and storage.get("password"):
        env_lines.append(f"MINDCORE_DB_PASSWORD={storage['password']}")

    if llm.get("api_key"):
        provider = llm["provider"].upper()
        env_lines.append(f"{provider}_API_KEY={llm['api_key']}")

    env_path = output_dir / ".env"
    with open(env_path, "w") as f:
        f.write("\n".join(env_lines))

    # Generate quick start script
    script = generate_quickstart_script(storage, domain)
    script_path = output_dir / "quickstart.py"
    with open(script_path, "w") as f:
        f.write(script)

    return config_path, env_path, script_path


def generate_quickstart_script(storage: dict, domain: dict) -> str:
    """Generate a quickstart Python script."""
    if storage["type"] == "sqlite":
        storage_line = f'storage="{storage["path"]}"'
    else:
        storage_line = f'storage="postgresql://{storage["user"]}@{storage["host"]}:{storage["port"]}/{storage["database"]}"'

    domains_import = ""
    domains_init = ""
    vocab_setup = ""
    if domain.get("domains"):
        domain_name = domain["domains"][0].upper() + "_DOMAIN"
        domains_import = f"\nfrom mindcore.svl import {domain_name}, SharedVocabularyLayer"
        vocab_setup = f"""
    # Set up vocabulary with your domain
    vocab = SharedVocabularyLayer()
    vocab.add_domain({domain_name})
    vocab.add_topics("preferences", "settings")  # Add custom topics as needed
"""
        domains_init = ", vocabulary=vocab"
    else:
        domains_import = "\nfrom mindcore.svl import SharedVocabularyLayer"
        vocab_setup = """
    # Set up vocabulary with custom topics
    vocab = SharedVocabularyLayer()
    vocab.add_topics("preferences", "settings")  # Add your topics here
"""
        domains_init = ", vocabulary=vocab"

    return f'''#!/usr/bin/env python3
"""Mindcore Quick Start - Your AI memory layer is ready!

Run this script to verify your setup:
    python quickstart.py
"""

from mindcore import Mindcore{domains_import}


def main():
{vocab_setup}
    # Initialize Mindcore with your configured storage
    memory = Mindcore({storage_line}{domains_init})

    print("Mindcore initialized successfully!")
    print()

    # Store a test memory
    memory_id = memory.store(
        content="User prefers dark mode and concise responses",
        memory_type="preference",
        user_id="demo_user",
        topics=["preferences"],
    )
    print(f"Stored memory: {{memory_id}}")

    # Recall the memory
    result = memory.recall(
        query="user preferences",
        user_id="demo_user",
        limit=5,
    )

    print(f"Recalled {{len(result.memories)}} memories:")
    for mem in result.memories:
        print(f"  - {{mem.content[:50]}}...")

    print()
    print("Your Mindcore setup is working!")
    print()
    print("Next steps:")
    print("  1. Integrate with your AI agent")
    print("  2. Configure SVL vocabulary for your domain")
    print("  3. Set up external data sources")
    print()
    print("Documentation: https://github.com/mindcore/mindcore")


if __name__ == "__main__":
    main()
'''


def print_summary(
    persona: dict,
    config_path: Path,
    env_path: Path,
    script_path: Path,
):
    """Print setup summary and next steps."""
    click.echo()
    click.echo(styled("  " + "═" * 50, Colors.GREEN))
    click.echo(styled("  Setup Complete!", Colors.GREEN, Colors.BOLD))
    click.echo(styled("  " + "═" * 50, Colors.GREEN))
    click.echo()

    click.echo(styled("  Generated files:", Colors.BOLD))
    click.echo(f"    {styled('•', Colors.CYAN)} {config_path}")
    click.echo(f"    {styled('•', Colors.CYAN)} {env_path}")
    click.echo(f"    {styled('•', Colors.CYAN)} {script_path}")
    click.echo()

    click.echo(styled("  Quick Start:", Colors.BOLD))
    click.echo()
    click.echo(styled(f"    python {script_path}", Colors.CYAN))
    click.echo()

    click.echo(styled("  Or use in your code:", Colors.BOLD))
    click.echo()
    click.echo(styled("    from mindcore import Mindcore", Colors.DIM))
    click.echo(styled('    memory = Mindcore("sqlite:///mindcore.db")', Colors.DIM))
    click.echo(
        styled(
            '    memory.store(content="...", memory_type="preference", user_id="user1")', Colors.DIM
        )
    )
    click.echo()

    click.echo(styled("  Need help?", Colors.BOLD))
    click.echo(
        f"    {styled('•', Colors.CYAN)} Run {styled('mindcore doctor', Colors.YELLOW)} to check your setup"
    )
    click.echo(
        f"    {styled('•', Colors.CYAN)} Run {styled('mindcore demo', Colors.YELLOW)} for an interactive demo"
    )
    click.echo(f"    {styled('•', Colors.CYAN)} Docs: https://github.com/mindcore/mindcore")
    click.echo()


@click.command()
@click.option("--quick", is_flag=True, help="Quick setup with minimal prompts")
@click.option(
    "--output", "-o", type=click.Path(), default=".", help="Output directory for config files"
)
def init_command(quick: bool, output: str):
    r"""Interactive setup wizard for Mindcore.

    Guides you through configuring:

    \b
    - Storage (SQLite or PostgreSQL)
    - LLM provider for metadata extraction
    - SVL vocabulary for your domain
    - Validation policies

    Creates mindcore.yaml, .env, and quickstart.py in your project.
    """
    import importlib.util

    if importlib.util.find_spec("yaml") is None:
        click.echo("Error: PyYAML required. Install with: pip install pyyaml", err=True)
        raise SystemExit(1)

    output_dir = Path(output).resolve()

    print_banner()

    click.echo(styled("  Welcome to Mindcore Setup!", Colors.BOLD))
    click.echo(styled("  Let's configure your AI memory layer.", Colors.DIM))
    click.echo()

    total_steps = 5

    if quick:
        # Quick mode - minimal prompts
        print_step(1, 2, "Quick Setup")
        persona = PERSONAS["1"]
        domain = DOMAINS["8"]

        print_step(2, 2, "Storage")
        storage = {"type": "sqlite", "path": "./mindcore.db"}
        click.echo(styled("  ✓ Using SQLite (./mindcore.db)", Colors.GREEN))

        llm = {"provider": None, "api_key": None, "model": None}
        policies = {
            "strict_mode": False,
            "require_user_id": True,
            "max_content_length": 10000,
            "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
        }
    else:
        # Full interactive mode
        print_step(1, total_steps, "Who You Are")
        persona = select_persona()

        print_step(2, total_steps, "Your Domain")
        domain = select_domain()

        print_step(3, total_steps, "Storage Setup")
        storage = configure_storage(persona)

        print_step(4, total_steps, "LLM Provider")
        llm = configure_llm()

        print_step(5, total_steps, "Policies")
        policies = configure_policies(persona)

    # Generate config files
    click.echo()
    click.echo(styled("  Generating configuration files...", Colors.DIM))

    config_path, env_path, script_path = generate_config_files(
        persona=persona,
        domain=domain,
        storage=storage,
        llm=llm,
        policies=policies,
        output_dir=output_dir,
    )

    print_summary(persona, config_path, env_path, script_path)
