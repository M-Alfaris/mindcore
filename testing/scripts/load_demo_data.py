#!/usr/bin/env python3
"""Demo Data Loader for Mindcore Testing.

This script loads sample data from demo_data/ into mindcore storage
for testing purposes.
"""

import json
import sys
from pathlib import Path
from typing import Any


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Paths
DEMO_DATA_DIR = Path(__file__).parent.parent / "demo_data"
MEMORIES_FILE = DEMO_DATA_DIR / "memories.json"
AGENTS_FILE = DEMO_DATA_DIR / "agents.json"
VOCABULARIES_FILE = DEMO_DATA_DIR / "vocabularies.json"


def print_status(message: str, status: str = "info"):
    """Print status message."""
    if RICH_AVAILABLE:
        console = Console()
        colors = {"info": "blue", "success": "green", "error": "red", "warning": "yellow"}
        console.print(f"[{colors.get(status, 'white')}]{message}[/]")
    else:
        print(f"[{status.upper()}] {message}")


def load_json(file_path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def load_memories(mc, data: dict) -> int:
    """Load user memories into mindcore."""
    count = 0

    # Load per-user memories
    for user_data in data.get("user_memories", []):
        user_id = user_data["user_id"]

        for memory in user_data.get("memories", []):
            try:
                mc.store(
                    content=memory["content"],
                    memory_type=memory.get("memory_type", "semantic"),
                    user_id=user_id,
                    topics=memory.get("topics", []),
                    categories=memory.get("categories", []),
                    importance=memory.get("importance", 0.5),
                    entities=memory.get("entities", []),
                    access_level=memory.get("access_level", "private"),
                )
                count += 1
            except Exception as e:
                print_status(f"Failed to store memory for {user_id}: {e}", "warning")

    # Load cross-user/global memories
    for memory in data.get("cross_user_memories", []):
        try:
            mc.store(
                content=memory["content"],
                memory_type=memory.get("memory_type", "semantic"),
                user_id="system",  # Global memories use system user
                topics=memory.get("topics", []),
                categories=memory.get("categories", []),
                importance=memory.get("importance", 0.5),
                access_level=memory.get("access_level", "global"),
            )
            count += 1
        except Exception as e:
            print_status(f"Failed to store global memory: {e}", "warning")

    return count


def load_agents(mc, data: dict) -> int:
    """Load multi-agent configurations."""
    count = 0

    for agent in data.get("agents", []):
        try:
            mc.register_agent(
                agent_id=agent["id"],
                name=agent["name"],
                description=agent.get("description", ""),
                teams=agent.get("teams", []),
            )
            count += 1
        except Exception as e:
            print_status(f"Failed to register agent {agent['id']}: {e}", "warning")

    return count


def setup_vocabulary(data: dict):
    """Setup custom vocabulary from domain configs."""
    try:
        from mindcore.v2.svl import SharedVocabularyLayer

        vocab = SharedVocabularyLayer()

        for domain in data.get("domains", []):
            # Add topics
            for topic in domain.get("topics", []):
                vocab.add_topics(topic)

            # Add categories
            for category in domain.get("categories", []):
                vocab.add_categories(category)

            # Add subcategories
            for category, subcats in domain.get("subcategories", {}).items():
                for subcat in subcats:
                    vocab.add_subcategory(category, subcat)

        print_status(
            f"Vocabulary configured with {len(data.get('domains', []))} domains", "success"
        )
        return vocab

    except ImportError:
        print_status("mindcore not installed", "warning")
        return None


def show_summary(memories_count: int, agents_count: int, storage_type: str):
    """Display load summary."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Demo Data Load Summary")
        table.add_column("Item", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Memories loaded", str(memories_count))
        table.add_row("Agents registered", str(agents_count))
        table.add_row("Storage backend", storage_type)

        console.print(table)
    else:
        print("\n=== Demo Data Load Summary ===")
        print(f"Memories loaded: {memories_count}")
        print(f"Agents registered: {agents_count}")
        print(f"Storage backend: {storage_type}")
        print("=" * 30)


def main():
    """Main data loading flow."""
    import argparse

    parser = argparse.ArgumentParser(description="Load demo data into mindcore")
    parser.add_argument(
        "--storage",
        default="sqlite:///mindcore_test.db",
        help="Storage connection string (default: sqlite:///mindcore_test.db)",
    )
    parser.add_argument("--multi-agent", action="store_true", help="Enable multi-agent features")
    parser.add_argument("--clean", action="store_true", help="Clean existing data before loading")

    args = parser.parse_args()

    print_status(f"Loading demo data into {args.storage}", "info")

    try:
        from mindcore.v2 import Mindcore
    except ImportError:
        print_status("mindcore not installed. Run: pip install mindcore", "error")
        sys.exit(1)

    # Load data files
    memories_data = load_json(MEMORIES_FILE)
    agents_data = load_json(AGENTS_FILE)
    vocabularies_data = load_json(VOCABULARIES_FILE)

    # Setup vocabulary
    vocab = setup_vocabulary(vocabularies_data)

    # Initialize mindcore
    mc = Mindcore(storage=args.storage, vocabulary=vocab, enable_multi_agent=args.multi_agent)

    try:
        # Load memories
        print_status("Loading memories...", "info")
        memories_count = load_memories(mc, memories_data)
        print_status(f"Loaded {memories_count} memories", "success")

        # Load agents if multi-agent enabled
        agents_count = 0
        if args.multi_agent:
            print_status("Registering agents...", "info")
            agents_count = load_agents(mc, agents_data)
            print_status(f"Registered {agents_count} agents", "success")

        # Show summary
        storage_type = "PostgreSQL" if "postgresql" in args.storage else "SQLite"
        show_summary(memories_count, agents_count, storage_type)

        # Get stats
        stats = mc.get_stats()
        print_status(f"\nStorage stats: {stats}", "info")

    finally:
        mc.close()

    print_status("\nDemo data loaded successfully!", "success")


if __name__ == "__main__":
    main()
