"""Interactive onboarding wizard for Mindcore.

Streamlined 4-step setup (~2 minutes):
1. What are you building? (determines all defaults)
2. Storage setup (SQLite/PostgreSQL with testing)
3. Rules & governance (safe presets)
4. Confirmation (what was created, where configs live, how to undo)

Design principles:
- Prove safety (nothing scary happens)
- Prove usefulness (memory actually works)
- Prove reversibility (can undo everything)
- Avoid architectural overwhelm
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


# What are you building? (Primary determinant of all defaults)
PROJECT_TYPES = {
    "1": {
        "name": "Single AI Agent",
        "desc": "Dev/prototype, personal assistant, chatbot",
        "storage": "sqlite",
        "clst": False,  # No cold storage needed
        "audit": False,
        "strict": False,
        "complexity": "minimal",
    },
    "2": {
        "name": "Multi-Agent System",
        "desc": "Multiple agents sharing memory",
        "storage": "sqlite",
        "clst": True,  # Cross-agent needs CLST
        "audit": False,
        "strict": False,
        "complexity": "standard",
    },
    "3": {
        "name": "Enterprise AI Platform",
        "desc": "Production system, team collaboration",
        "storage": "postgresql",
        "clst": True,
        "audit": True,
        "strict": True,
        "complexity": "advanced",
    },
    "4": {
        "name": "Regulated / Gov / Healthcare",
        "desc": "Compliance required, full audit trail",
        "storage": "postgresql",
        "clst": True,
        "audit": True,
        "strict": True,
        "compliance": True,
        "complexity": "full",
    },
}

# Sector selection (optional, for pre-configured rules)
SECTORS = {
    "1": {"name": "General", "domains": [], "default": True},
    "2": {"name": "Finance / Banking", "domains": ["finance"], "retention_days": 2555},
    "3": {"name": "Healthcare", "domains": ["healthcare"], "hipaa": True},
    "4": {"name": "Government", "domains": ["government"], "audit_required": True},
    "5": {"name": "E-commerce", "domains": ["ecommerce"]},
    "6": {"name": "Customer Service", "domains": ["customer_service"]},
}

# Governance presets (keep it simple)
GOVERNANCE_PRESETS = {
    "1": {
        "name": "Safe defaults",
        "desc": "Recommended - sensible settings for most cases",
        "config": {
            "strict_mode": False,
            "require_user_id": True,
            "long_term_memory": True,
            "preference_promotion": "conservative",
            "audit_level": "minimal",
        },
    },
    "2": {
        "name": "Strict mode",
        "desc": "Production-ready with full validation",
        "config": {
            "strict_mode": True,
            "require_user_id": True,
            "long_term_memory": True,
            "preference_promotion": "strict",
            "audit_level": "standard",
        },
    },
    "3": {
        "name": "Compliance mode",
        "desc": "Full audit trail, regulatory ready",
        "config": {
            "strict_mode": True,
            "require_user_id": True,
            "long_term_memory": True,
            "preference_promotion": "strict",
            "audit_level": "full",
            "retention_policy": True,
        },
    },
}

# Legacy compatibility - map old structures to new
PERSONAS = {
    "1": {"name": "Solo Developer", "storage": "sqlite", "complexity": "minimal"},
    "2": {"name": "Startup", "storage": "sqlite", "complexity": "simple"},
    "3": {"name": "Enterprise", "storage": "postgresql", "complexity": "full"},
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

INTEGRATION_MODES = {
    "1": {"name": "New Project", "full_setup": True},
    "2": {"name": "Existing AI Agent", "show_integration_guide": True},
    "3": {"name": "Partial Integration", "minimal": True},
    "4": {"name": "Evaluation / Demo", "demo": True},
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


def select_integration_mode() -> dict:
    """Let user select how they want to integrate Mindcore."""
    click.echo()
    click.echo(styled("  How will you use Mindcore?", Colors.BOLD))
    click.echo()

    for key, mode in INTEGRATION_MODES.items():
        click.echo(
            f"    {styled(key, Colors.CYAN, Colors.BOLD)}) {styled(mode['name'], Colors.BOLD)}"
        )
        click.echo(styled(f"       {mode['desc']}", Colors.DIM))

    click.echo()
    choice = click.prompt(
        styled("  Select mode", Colors.YELLOW),
        type=click.Choice(list(INTEGRATION_MODES.keys())),
        default="1",
    )

    selected = INTEGRATION_MODES[choice]
    click.echo()
    click.echo(styled(f"  ✓ Mode: {selected['name']}", Colors.GREEN))
    return selected


def test_sqlite_connection(path: str) -> tuple[bool, str]:
    """Test SQLite connection and write permissions."""
    import sqlite3
    from pathlib import Path

    try:
        db_path = Path(path)
        parent = db_path.parent

        # Check if parent directory exists or can be created
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                return False, f"Cannot create directory: {parent}"

        # Test connection
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS _mindcore_test (id INTEGER)")
        cursor.execute("DROP TABLE _mindcore_test")
        conn.close()

        return True, "Connection successful"
    except sqlite3.Error as e:
        return False, f"SQLite error: {e}"
    except Exception as e:
        return False, str(e)


def test_postgres_connection(
    host: str, port: str, database: str, user: str, password: str
) -> tuple[bool, str]:
    """Test PostgreSQL connection."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=host,
            port=int(port),
            database=database,
            user=user,
            password=password,
            connect_timeout=5,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        conn.close()

        return True, f"Connected to PostgreSQL ({version.split(',')[0]})"
    except ImportError:
        return False, "psycopg2 not installed. Run: pip install mindcore[postgres]"
    except Exception as e:
        error_msg = str(e)
        if "password authentication failed" in error_msg:
            return False, "Authentication failed - check username/password"
        if "could not connect to server" in error_msg or "Connection refused" in error_msg:
            return False, f"Cannot connect to {host}:{port} - is PostgreSQL running?"
        if "does not exist" in error_msg:
            return False, f"Database '{database}' does not exist"
        return False, error_msg


def test_llm_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Test LLM API key validity."""
    if not api_key:
        return False, "No API key provided"

    try:
        if provider == "openai":
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True, "OpenAI API key valid"
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    return False, "Invalid API key"
                return False, f"API error: {e.code}"
            except urllib.error.URLError:
                return False, "Network error - cannot reach OpenAI API"

        elif provider == "anthropic":
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                data=b'{"model":"claude-3-haiku-20240307","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}',
            )
            try:
                with urllib.request.urlopen(req, timeout=10):
                    return True, "Anthropic API key valid"
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    return False, "Invalid API key"
                if e.code == 400:
                    # Bad request but auth worked
                    return True, "Anthropic API key valid"
                return False, f"API error: {e.code}"
            except urllib.error.URLError:
                return False, "Network error - cannot reach Anthropic API"

        elif provider == "google":
            # Google uses different auth, just check format
            if api_key.startswith("AIza") and len(api_key) > 30:
                return True, "Google API key format valid (not verified)"
            return False, "Invalid Google API key format"

        return True, "Key format accepted"
    except Exception as e:
        return False, f"Validation error: {e}"


def check_existing_files(output_dir: Path) -> list[Path]:
    """Check for existing configuration files that would be overwritten."""
    files_to_check = ["mindcore.yaml", ".env", "quickstart.py"]
    existing = []

    for filename in files_to_check:
        filepath = output_dir / filename
        if filepath.exists():
            existing.append(filepath)

    return existing


def configure_storage(persona: dict) -> dict:
    """Configure storage based on persona with connection testing."""
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
        # SQLite configuration with testing
        while True:
            db_path = click.prompt(
                styled("  SQLite file path", Colors.YELLOW), default="./mindcore.db"
            )

            click.echo(styled("  Testing connection...", Colors.DIM))
            success, message = test_sqlite_connection(db_path)

            if success:
                config = {"type": "sqlite", "path": db_path}
                click.echo(styled(f"  ✓ SQLite configured: {db_path}", Colors.GREEN))
                break
            click.echo(styled(f"  ✗ Error: {message}", Colors.RED))
            retry = click.confirm(styled("  Try a different path?", Colors.YELLOW), default=True)
            if not retry:
                click.echo(styled("  Using path anyway (will retry on first use)", Colors.YELLOW))
                config = {"type": "sqlite", "path": db_path}
                break
    else:
        # PostgreSQL configuration with testing
        while True:
            click.echo()
            click.echo(styled("  PostgreSQL connection:", Colors.DIM))
            click.echo(styled("  (Leave blank to use environment variables)", Colors.DIM))
            click.echo()

            host = click.prompt("    Host", default=os.environ.get("MINDCORE_DB_HOST", "localhost"))
            port = click.prompt("    Port", default=os.environ.get("MINDCORE_DB_PORT", "5432"))
            database = click.prompt(
                "    Database", default=os.environ.get("MINDCORE_DB_NAME", "mindcore")
            )
            user = click.prompt("    User", default=os.environ.get("MINDCORE_DB_USER", "postgres"))
            password = click.prompt(
                "    Password", hide_input=True, default=os.environ.get("MINDCORE_DB_PASSWORD", "")
            )

            click.echo()
            click.echo(styled("  Testing connection...", Colors.DIM))
            success, message = test_postgres_connection(host, port, database, user, password)

            if success:
                config = {
                    "type": "postgresql",
                    "host": host,
                    "port": port,
                    "database": database,
                    "user": user,
                    "password": password,
                }
                click.echo(styled(f"  ✓ {message}", Colors.GREEN))
                break
            click.echo(styled(f"  ✗ {message}", Colors.RED))
            click.echo()

            # Provide helpful suggestions
            if "psycopg2 not installed" in message:
                click.echo(styled("  To install PostgreSQL support:", Colors.YELLOW))
                click.echo(styled("    pip install mindcore[postgres]", Colors.CYAN))
                click.echo()
            elif "Connection refused" in message or "Cannot connect" in message:
                click.echo(styled("  Troubleshooting:", Colors.YELLOW))
                click.echo(styled("    1. Is PostgreSQL running?", Colors.DIM))
                click.echo(styled("    2. Check host/port settings", Colors.DIM))
                click.echo(
                    styled(
                        f"    3. Try: psql -h {host} -p {port} -U {user} -d {database}", Colors.DIM
                    )
                )
                click.echo()
            elif "does not exist" in message:
                click.echo(styled("  To create the database:", Colors.YELLOW))
                click.echo(
                    styled(f"    createdb -h {host} -p {port} -U {user} {database}", Colors.CYAN)
                )
                click.echo()

            action = click.prompt(
                styled("  What would you like to do?", Colors.YELLOW),
                type=click.Choice(["retry", "sqlite", "continue"]),
                default="retry",
            )

            if action == "retry":
                continue
            if action == "sqlite":
                click.echo(styled("  Switching to SQLite...", Colors.DIM))
                db_path = click.prompt(
                    styled("  SQLite file path", Colors.YELLOW), default="./mindcore.db"
                )
                config = {"type": "sqlite", "path": db_path}
                click.echo(styled(f"  ✓ SQLite configured: {db_path}", Colors.GREEN))
                break
            click.echo(styled("  Continuing without connection test", Colors.YELLOW))
            config = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "user": user,
                "password": password,
            }
            break

    return config


def configure_llm(integration_mode: dict | None = None) -> dict:
    """Configure LLM provider with validation."""
    click.echo()
    click.echo(styled("  LLM Provider Configuration", Colors.BOLD))
    click.echo()

    # Check if user has existing agent
    if integration_mode and integration_mode.get("show_integration_guide"):
        click.echo(styled("  Since you have an existing AI agent, you have options:", Colors.DIM))
        click.echo()
        click.echo(
            f"    {styled('1', Colors.CYAN, Colors.BOLD)}) Use my agent's LLM {styled('(Recommended)', Colors.GREEN)}"
        )
        click.echo(
            styled("       Mindcore will use your existing LLM for metadata extraction", Colors.DIM)
        )
        click.echo(f"    {styled('2', Colors.CYAN, Colors.BOLD)}) Configure separate LLM")
        click.echo(
            styled("       Use a different LLM for Mindcore's metadata extraction", Colors.DIM)
        )
        click.echo(f"    {styled('3', Colors.CYAN, Colors.BOLD)}) No LLM (rule-based only)")
        click.echo(styled("       Use rule-based extraction (works but less accurate)", Colors.DIM))

        click.echo()
        mode_choice = click.prompt(
            styled("  Select option", Colors.YELLOW),
            type=click.Choice(["1", "2", "3"]),
            default="1",
        )

        if mode_choice == "1":
            click.echo()
            click.echo(styled("  ✓ Will use your agent's LLM", Colors.GREEN))
            click.echo()
            click.echo(styled("  Integration code:", Colors.DIM))
            click.echo(styled("    # In your agent code:", Colors.CYAN))
            click.echo(styled("    from mindcore import Mindcore", Colors.CYAN))
            click.echo(styled("    memory = Mindcore(", Colors.CYAN))
            click.echo(styled("        storage='sqlite:///mindcore.db',", Colors.CYAN))
            click.echo(
                styled(
                    "        llm_client=your_llm_client  # Pass your existing client", Colors.CYAN
                )
            )
            click.echo(styled("    )", Colors.CYAN))
            return {"provider": "agent", "api_key": None, "model": None, "use_agent_llm": True}
        if mode_choice == "3":
            click.echo(styled("  ✓ Using rule-based extraction only", Colors.GREEN))
            return {"provider": None, "api_key": None, "model": None}
        # else fall through to normal LLM configuration

    click.echo(styled("  LLM is used for intelligent metadata extraction.", Colors.DIM))
    click.echo(
        styled("  Without it, Mindcore uses rule-based fallback (less accurate).", Colors.DIM)
    )
    click.echo()

    click.echo(f"    {styled('1', Colors.CYAN, Colors.BOLD)}) OpenAI (GPT-4o-mini)")
    click.echo(f"    {styled('2', Colors.CYAN, Colors.BOLD)}) Anthropic (Claude 3 Haiku)")
    click.echo(f"    {styled('3', Colors.CYAN, Colors.BOLD)}) Google (Gemini 2.0)")
    click.echo(
        f"    {styled('4', Colors.CYAN, Colors.BOLD)}) Custom / Local LLM (Ollama, vLLM, etc.)"
    )
    click.echo(f"    {styled('5', Colors.CYAN, Colors.BOLD)}) Skip for now")

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
        existing_key = os.environ.get("OPENAI_API_KEY", "")

        if existing_key:
            click.echo(styled("  Found OPENAI_API_KEY in environment", Colors.DIM))
            use_existing = click.confirm(styled("  Use this key?", Colors.YELLOW), default=True)
            if use_existing:
                api_key = existing_key
            else:
                api_key = click.prompt("    Enter new OpenAI API Key", hide_input=True)
        else:
            api_key = click.prompt(
                "    OpenAI API Key",
                hide_input=True,
                default="",
            )

        if api_key:
            click.echo(styled("  Validating API key...", Colors.DIM))
            success, message = test_llm_api_key("openai", api_key)
            if success:
                config["api_key"] = api_key
                click.echo(styled(f"  ✓ {message}", Colors.GREEN))
            else:
                click.echo(styled(f"  ✗ {message}", Colors.RED))
                proceed = click.confirm(styled("  Continue anyway?", Colors.YELLOW), default=False)
                if proceed:
                    config["api_key"] = api_key
                else:
                    config["provider"] = None
                    click.echo(styled("  Skipped LLM configuration", Colors.DIM))
        else:
            click.echo(styled("  ! No API key provided", Colors.YELLOW))
            click.echo(styled("  Add OPENAI_API_KEY to .env file later", Colors.DIM))

    elif choice == "2":
        config["provider"] = "anthropic"
        config["model"] = "claude-3-haiku-20240307"
        existing_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if existing_key:
            click.echo(styled("  Found ANTHROPIC_API_KEY in environment", Colors.DIM))
            use_existing = click.confirm(styled("  Use this key?", Colors.YELLOW), default=True)
            if use_existing:
                api_key = existing_key
            else:
                api_key = click.prompt("    Enter new Anthropic API Key", hide_input=True)
        else:
            api_key = click.prompt(
                "    Anthropic API Key",
                hide_input=True,
                default="",
            )

        if api_key:
            click.echo(styled("  Validating API key...", Colors.DIM))
            success, message = test_llm_api_key("anthropic", api_key)
            if success:
                config["api_key"] = api_key
                click.echo(styled(f"  ✓ {message}", Colors.GREEN))
            else:
                click.echo(styled(f"  ✗ {message}", Colors.RED))
                proceed = click.confirm(styled("  Continue anyway?", Colors.YELLOW), default=False)
                if proceed:
                    config["api_key"] = api_key
                else:
                    config["provider"] = None
                    click.echo(styled("  Skipped LLM configuration", Colors.DIM))
        else:
            click.echo(styled("  ! No API key provided", Colors.YELLOW))
            click.echo(styled("  Add ANTHROPIC_API_KEY to .env file later", Colors.DIM))

    elif choice == "3":
        config["provider"] = "google"
        config["model"] = "gemini-2.0-flash"
        existing_key = os.environ.get("GOOGLE_API_KEY", "")

        if existing_key:
            click.echo(styled("  Found GOOGLE_API_KEY in environment", Colors.DIM))
            use_existing = click.confirm(styled("  Use this key?", Colors.YELLOW), default=True)
            if use_existing:
                api_key = existing_key
            else:
                api_key = click.prompt("    Enter new Google API Key", hide_input=True)
        else:
            api_key = click.prompt(
                "    Google API Key",
                hide_input=True,
                default="",
            )

        if api_key:
            success, message = test_llm_api_key("google", api_key)
            if success:
                config["api_key"] = api_key
                click.echo(styled(f"  ✓ {message}", Colors.GREEN))
            else:
                click.echo(styled(f"  ✗ {message}", Colors.YELLOW))
                config["api_key"] = api_key  # Still save it
        else:
            click.echo(styled("  ! No API key provided", Colors.YELLOW))
            click.echo(styled("  Add GOOGLE_API_KEY to .env file later", Colors.DIM))

    elif choice == "4":
        config["provider"] = "custom"
        click.echo()
        click.echo(styled("  Custom LLM Configuration", Colors.BOLD))
        click.echo(styled("  Mindcore supports OpenAI-compatible APIs.", Colors.DIM))
        click.echo()

        base_url = click.prompt(
            "    API Base URL",
            default="http://localhost:11434/v1",
        )
        model = click.prompt(
            "    Model name",
            default="llama3.2",
        )

        config["provider"] = "custom"
        config["model"] = model
        config["base_url"] = base_url
        click.echo()
        click.echo(styled("  ✓ Custom LLM configured", Colors.GREEN))
        click.echo(
            styled("  Make sure your LLM server is running before using Mindcore", Colors.DIM)
        )

    else:
        click.echo()
        click.echo(styled("  ✓ Skipped LLM configuration", Colors.DIM))
        click.echo(styled("  Mindcore will use rule-based metadata extraction", Colors.DIM))
        click.echo(styled("  You can add an LLM later in mindcore.yaml", Colors.DIM))

    return config


def configure_policies(persona: dict) -> dict:
    """Configure SVL policies based on persona."""
    complexity = persona.get("complexity", "minimal")

    click.echo()
    click.echo(styled("  Policy Configuration", Colors.BOLD))
    click.echo()
    click.echo(styled("  Policies control how memories are validated and stored.", Colors.DIM))
    click.echo()

    # Policy presets
    POLICY_PRESETS = {
        "1": {
            "name": "Relaxed",
            "desc": "Accept most data, minimal validation",
            "config": {
                "strict_mode": False,
                "require_user_id": True,
                "require_topics": False,
                "max_content_length": 50000,
                "allowed_memory_types": [
                    "preference",
                    "semantic",
                    "episodic",
                    "procedural",
                    "skill",
                    "fact",
                ],
            },
        },
        "2": {
            "name": "Balanced",
            "desc": "Sensible defaults for most use cases",
            "config": {
                "strict_mode": False,
                "require_user_id": True,
                "require_topics": False,
                "max_content_length": 10000,
                "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
            },
        },
        "3": {
            "name": "Strict",
            "desc": "Enforce all validation rules",
            "config": {
                "strict_mode": True,
                "require_user_id": True,
                "require_topics": True,
                "max_content_length": 5000,
                "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
            },
        },
        "4": {
            "name": "Enterprise",
            "desc": "Full validation with audit requirements",
            "config": {
                "strict_mode": True,
                "require_user_id": True,
                "require_topics": True,
                "require_source": True,
                "max_content_length": 10000,
                "audit_all_operations": True,
                "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
            },
        },
        "5": {
            "name": "Custom",
            "desc": "Configure each policy individually",
            "config": None,
        },
    }

    # Determine recommended preset based on complexity
    if complexity in ["minimal", "demo"]:
        recommended = "1"
    elif complexity in ["simple", "standard"]:
        recommended = "2"
    elif complexity == "advanced":
        recommended = "3"
    elif complexity == "full":
        recommended = "4"
    else:
        recommended = "2"

    for key, preset in POLICY_PRESETS.items():
        rec_tag = styled(" (Recommended)", Colors.GREEN) if key == recommended else ""
        click.echo(
            f"    {styled(key, Colors.CYAN, Colors.BOLD)}) {styled(preset['name'], Colors.BOLD)}{rec_tag}"
        )
        click.echo(styled(f"       {preset['desc']}", Colors.DIM))

    click.echo()
    choice = click.prompt(
        styled("  Select policy preset", Colors.YELLOW),
        type=click.Choice(list(POLICY_PRESETS.keys())),
        default=recommended,
    )

    if choice != "5":
        config = POLICY_PRESETS[choice]["config"].copy()
        click.echo()
        click.echo(styled(f"  ✓ {POLICY_PRESETS[choice]['name']} policies applied", Colors.GREEN))
        return config

    # Custom configuration
    click.echo()
    click.echo(styled("  Custom Policy Configuration", Colors.BOLD))
    click.echo()

    strict = click.confirm(
        styled("    Strict mode?", Colors.YELLOW)
        + styled(" (reject memories with invalid metadata)", Colors.DIM),
        default=False,
    )

    require_user_id = click.confirm(
        styled("    Require user_id?", Colors.YELLOW)
        + styled(" (all memories must have a user ID)", Colors.DIM),
        default=True,
    )

    require_topics = click.confirm(
        styled("    Require topics?", Colors.YELLOW)
        + styled(" (all memories must have at least one topic)", Colors.DIM),
        default=False,
    )

    max_length = click.prompt(
        styled("    Max content length", Colors.YELLOW),
        default=10000,
        type=int,
    )

    click.echo()
    click.echo(styled("  Available memory types:", Colors.DIM))
    all_types = [
        "preference",
        "semantic",
        "episodic",
        "procedural",
        "skill",
        "fact",
        "relationship",
        "goal",
    ]
    for i, mtype in enumerate(all_types, 1):
        click.echo(styled(f"    {i}. {mtype}", Colors.DIM))

    click.echo()
    types_input = click.prompt(
        styled("    Allowed types (comma-separated numbers, or 'all')", Colors.YELLOW),
        default="1,2,3,4",
    )

    if types_input.lower() == "all":
        allowed_types = all_types
    else:
        try:
            indices = [int(x.strip()) - 1 for x in types_input.split(",")]
            allowed_types = [all_types[i] for i in indices if 0 <= i < len(all_types)]
        except (ValueError, IndexError):
            allowed_types = ["preference", "semantic", "episodic", "procedural"]
            click.echo(styled("  Using default memory types", Colors.YELLOW))

    # Custom memory types
    click.echo()
    add_custom = click.confirm(
        styled("    Add custom memory types?", Colors.YELLOW),
        default=False,
    )

    if add_custom:
        custom_types = click.prompt(
            styled("    Enter custom types (comma-separated)", Colors.YELLOW),
            default="",
        )
        if custom_types:
            for raw_type in custom_types.split(","):
                clean_type = raw_type.strip().lower().replace(" ", "_")
                if clean_type and clean_type not in allowed_types:
                    allowed_types.append(clean_type)
                    click.echo(styled(f"    Added: {clean_type}", Colors.GREEN))

    config = {
        "strict_mode": strict,
        "require_user_id": require_user_id,
        "require_topics": require_topics,
        "max_content_length": max_length,
        "allowed_memory_types": allowed_types,
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
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files without asking")
def init_command(quick: bool, output: str, force: bool):
    r"""Interactive setup wizard for Mindcore.

    Guides you through configuring:

    \b
    - Integration mode (new project, existing agent, evaluation)
    - Storage (SQLite or PostgreSQL) with connection testing
    - LLM provider for metadata extraction with API validation
    - SVL vocabulary for your domain
    - Validation policies

    Creates mindcore.yaml, .env, and quickstart.py in your project.
    """
    import importlib.util

    if importlib.util.find_spec("yaml") is None:
        click.echo("Error: PyYAML required. Install with: pip install pyyaml", err=True)
        raise SystemExit(1)

    output_dir = Path(output).resolve()

    # Check for existing files
    if not force:
        existing_files = check_existing_files(output_dir)
        if existing_files:
            click.echo()
            click.echo(styled("  ⚠ Existing configuration files found:", Colors.YELLOW))
            for f in existing_files:
                click.echo(styled(f"    • {f}", Colors.DIM))
            click.echo()

            overwrite = click.confirm(
                styled("  Overwrite these files?", Colors.YELLOW),
                default=False,
            )
            if not overwrite:
                click.echo()
                click.echo(styled("  Setup cancelled. Use --force to overwrite.", Colors.DIM))
                click.echo(styled("  Or specify a different output directory with -o", Colors.DIM))
                raise SystemExit(0)
            click.echo()

    print_banner()

    click.echo(styled("  Welcome to Mindcore Setup!", Colors.BOLD))
    click.echo(styled("  Let's configure your AI memory layer.", Colors.DIM))
    click.echo()

    integration_mode = None

    if quick:
        # Quick mode - minimal prompts
        total_steps = 2
        print_step(1, total_steps, "Quick Setup")
        persona = PERSONAS["1"]
        domain = DOMAINS["8"]
        integration_mode = INTEGRATION_MODES["4"]  # Evaluation mode

        print_step(2, total_steps, "Storage")
        storage = {"type": "sqlite", "path": "./mindcore.db"}
        click.echo(styled("  ✓ Using SQLite (./mindcore.db)", Colors.GREEN))

        llm = {"provider": None, "api_key": None, "model": None}
        policies = {
            "strict_mode": False,
            "require_user_id": True,
            "require_topics": False,
            "max_content_length": 10000,
            "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
        }
    else:
        # Full interactive mode
        total_steps = 6

        print_step(1, total_steps, "Who You Are")
        persona = select_persona()

        print_step(2, total_steps, "Integration Mode")
        integration_mode = select_integration_mode()

        # Adjust flow based on integration mode
        if integration_mode.get("demo"):
            # Demo mode - skip to quick setup
            click.echo(styled("  Setting up for quick evaluation...", Colors.DIM))
            domain = DOMAINS["8"]
            storage = {"type": "sqlite", "path": "./mindcore.db"}
            click.echo(styled("  ✓ Using SQLite (./mindcore.db)", Colors.GREEN))
            llm = {"provider": None, "api_key": None, "model": None}
            policies = {
                "strict_mode": False,
                "require_user_id": True,
                "require_topics": False,
                "max_content_length": 10000,
                "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
            }
        elif integration_mode.get("minimal"):
            # Minimal mode - just storage
            print_step(3, total_steps, "Storage Setup")
            storage = configure_storage(persona)

            domain = DOMAINS["8"]
            llm = {"provider": None, "api_key": None, "model": None}
            policies = {
                "strict_mode": False,
                "require_user_id": True,
                "require_topics": False,
                "max_content_length": 10000,
                "allowed_memory_types": ["preference", "semantic", "episodic", "procedural"],
            }

            click.echo()
            click.echo(
                styled("  ✓ Minimal setup selected - skipping domain, LLM, policies", Colors.DIM)
            )
        else:
            # Full setup
            print_step(3, total_steps, "Your Domain")
            domain = select_domain()

            print_step(4, total_steps, "Storage Setup")
            storage = configure_storage(persona)

            print_step(5, total_steps, "LLM Provider")
            llm = configure_llm(integration_mode)

            print_step(6, total_steps, "Policies")
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

    # Show integration-specific guidance
    if integration_mode and integration_mode.get("show_integration_guide"):
        click.echo(styled("  Integration Guide for Existing Agents:", Colors.BOLD))
        click.echo()
        click.echo(styled("  Add to your agent's initialization:", Colors.DIM))
        click.echo(styled("    from mindcore import Mindcore", Colors.CYAN))
        click.echo(styled("    memory = Mindcore('sqlite:///mindcore.db')", Colors.CYAN))
        click.echo()
        click.echo(styled("  Before LLM calls - get context:", Colors.DIM))
        click.echo(
            styled("    context = memory.recall(query=user_message, user_id=user_id)", Colors.CYAN)
        )
        click.echo(styled("    system_prompt += format_context(context.memories)", Colors.CYAN))
        click.echo()
        click.echo(styled("  After LLM responses - store learnings:", Colors.DIM))
        click.echo(
            styled(
                "    memory.store(content=info, memory_type='semantic', user_id=user_id)",
                Colors.CYAN,
            )
        )
        click.echo()
