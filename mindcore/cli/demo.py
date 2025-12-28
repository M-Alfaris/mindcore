"""Interactive demo for Mindcore.

Provides hands-on examples of:
- Storing memories
- Recalling with context
- Multi-agent scenarios
- SVL vocabulary
"""

import click


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
    """Apply ANSI color/style codes to text."""
    return "".join(styles) + text + Colors.RESET


def print_header(title: str):
    """Print a section header."""
    click.echo()
    click.echo(styled(f"  {'─' * 50}", Colors.CYAN))
    click.echo(styled(f"  {title}", Colors.BOLD, Colors.CYAN))
    click.echo(styled(f"  {'─' * 50}", Colors.CYAN))
    click.echo()


def print_code(code: str):
    """Print code block."""
    for line in code.strip().split("\n"):
        click.echo(styled(f"    {line}", Colors.DIM))


def print_output(label: str, value: str):
    """Print labeled output."""
    click.echo(f"    {styled(label + ':', Colors.YELLOW)} {value}")


def pause(message: str = "Press Enter to continue..."):
    """Pause for user."""
    click.echo()
    click.prompt(styled(f"  {message}", Colors.DIM), default="", show_default=False)


def run_basic_demo():
    """Run basic store and recall demo."""
    from mindcore import Mindcore
    from mindcore.svl import SharedVocabularyLayer

    print_header("1. Basic Store and Recall")

    click.echo("  Creating an in-memory Mindcore instance:")
    print_code('memory = Mindcore(storage="sqlite:///:memory:")')

    # Create vocabulary with demo topics
    vocab = SharedVocabularyLayer()
    vocab.add_topics("preferences", "settings", "context", "programming")

    memory = Mindcore(storage="sqlite:///:memory:", vocabulary=vocab)
    click.echo(styled("  ✓ Mindcore initialized", Colors.GREEN))
    click.echo()

    # Store some memories
    click.echo("  Storing user preferences:")
    print_code("""
memory.store(
    content="User prefers dark mode",
    memory_type="preference",
    user_id="demo_user",
    topics=["preferences"],
)
""")

    memories = [
        ("User prefers dark mode and minimal UI", ["preferences", "settings"]),
        ("User is a software developer working on AI projects", ["context"]),
        ("User asked about Python async programming last week", ["programming"]),
        ("User prefers concise, technical explanations", ["preferences"]),
    ]

    memory_ids = []
    for content, topics in memories:
        mid = memory.store(
            content=content,
            memory_type="preference",
            user_id="demo_user",
            topics=topics,
        )
        memory_ids.append(mid)
        click.echo(f"    {styled('→', Colors.CYAN)} Stored: {content[:40]}...")

    click.echo()
    click.echo(styled(f"  ✓ Stored {len(memory_ids)} memories", Colors.GREEN))

    pause()

    # Recall memories
    print_header("2. Recalling Memories")

    click.echo("  Querying for relevant context:")
    print_code("""
result = memory.recall(
    query="What does the user prefer?",
    user_id="demo_user",
    limit=3,
)
""")

    result = memory.recall(
        query="What does the user prefer?",
        user_id="demo_user",
        limit=3,
    )

    click.echo()
    click.echo(f"  Found {len(result.memories)} relevant memories:")
    click.echo()

    for i, mem in enumerate(result.memories, 1):
        click.echo(f"    {styled(str(i) + '.', Colors.CYAN)} {mem.content}")
        click.echo(styled(f"       Topics: {', '.join(mem.topics or [])}", Colors.DIM))

    pause()

    return memory


def run_reinforcement_demo(memory):
    """Demo reinforcement learning."""
    print_header("3. Memory Reinforcement")

    click.echo("  Reinforcement adjusts memory importance based on usage.")
    click.echo()

    # Get a memory to reinforce
    result = memory.recall(query="preferences", user_id="demo_user", limit=1)
    if result.memories:
        mem = result.memories[0]
        old_score = mem.reinforcement_score

        click.echo("  Applying positive reinforcement (memory was useful):")
        print_code(f'memory.reinforce("{mem.memory_id[:20]}...", signal=0.8)')

        new_score = memory.reinforce(mem.memory_id, signal=0.8)

        click.echo()
        print_output("Before", f"{old_score:.3f}")
        if new_score is not None:
            print_output("After", f"{new_score:.3f}")
        else:
            print_output("After", "Updated (score returned on next recall)")
        click.echo()
        click.echo(styled("  ✓ Memory reinforced", Colors.GREEN))

    pause()


def run_search_demo(memory):
    """Demo advanced search."""
    print_header("4. Advanced Search")

    click.echo("  Search with filters:")
    print_code("""
results = memory.search(
    query="programming",
    user_id="demo_user",
    memory_types=["preference"],
    topics=["programming"],
)
""")

    results = memory.search(
        query="programming",
        user_id="demo_user",
        memory_types=["preference"],
    )

    click.echo()
    click.echo(f"  Found {len(results)} matching memories:")
    for mem in results:
        click.echo(f"    {styled('→', Colors.CYAN)} {mem.content[:50]}...")

    pause()


def run_context_demo(memory):
    """Demo context building for LLM."""
    print_header("5. Building Context for LLM")

    click.echo("  Get formatted context for your AI agent:")
    print_code("""
result = memory.recall(
    query="Help with Python code",
    user_id="demo_user",
    limit=3,
)

# Format for LLM system prompt
context = "\\n".join([
    f"- {m.content}" for m in result.memories
])
""")

    result = memory.recall(
        query="Help with Python code",
        user_id="demo_user",
        limit=3,
    )

    click.echo()
    click.echo("  Formatted context for LLM:")
    click.echo(styled("  ─" * 25, Colors.DIM))

    for mem in result.memories:
        click.echo(f"    • {mem.content}")

    click.echo(styled("  ─" * 25, Colors.DIM))
    click.echo()
    click.echo(styled("  ✓ Ready to inject into system prompt", Colors.GREEN))

    pause()


def run_svl_demo():
    """Demo SVL vocabulary."""
    print_header("6. SVL - Shared Vocabulary Layer")

    click.echo("  SVL provides structured metadata for memories:")
    click.echo()

    from mindcore.svl import DEFAULT_SVL

    click.echo("  Available memory types:")
    for mtype in DEFAULT_SVL.schema.memory_types[:5]:
        click.echo(f"    {styled('•', Colors.CYAN)} {mtype}")

    click.echo()
    click.echo("  Available sentiments:")
    for sent in ["positive", "negative", "neutral", "mixed"]:
        click.echo(f"    {styled('•', Colors.CYAN)} {sent}")

    click.echo()
    click.echo("  Pre-built domain vocabularies:")

    try:
        from mindcore.svl import list_domains

        domains = list_domains()
        for domain in domains[:5]:
            click.echo(f"    {styled('•', Colors.CYAN)} {domain}")
    except Exception:
        click.echo("    (Use domains by importing from mindcore.svl)")

    pause()


def print_next_steps():
    """Print next steps."""
    print_header("Next Steps")

    click.echo("  You've seen the basics! Here's what to do next:")
    click.echo()

    steps = [
        ("mindcore init", "Set up your project configuration"),
        ("mindcore doctor", "Verify your setup is correct"),
        ("Read MINDCORE.md", "Deep dive into architecture"),
        ("Integrate", "Add Mindcore to your AI agent"),
    ]

    for cmd, desc in steps:
        click.echo(f"    {styled('→', Colors.CYAN)} {styled(cmd, Colors.YELLOW)}: {desc}")

    click.echo()
    click.echo("  Quick integration example:")
    print_code("""
from mindcore import Mindcore

# In your AI agent
memory = Mindcore(storage="sqlite:///agent_memory.db")

# Before LLM call: get relevant context
context = memory.recall(query=user_message, user_id=user_id)

# After LLM response: store new information
memory.store(
    content=extracted_info,
    memory_type="semantic",
    user_id=user_id,
)
""")

    click.echo()
    click.echo(styled("  Thanks for trying Mindcore!", Colors.GREEN, Colors.BOLD))
    click.echo()


@click.command()
@click.option("--quick", is_flag=True, help="Quick demo without pauses")
@click.option("--section", type=int, help="Run specific section (1-6)")
def demo_command(quick: bool, section: int | None):
    r"""Run an interactive Mindcore demo.

    Walks through core features with live examples.

    \b
    Sections:
        1. Basic store and recall
        2. Recalling memories
        3. Memory reinforcement
        4. Advanced search
        5. Building LLM context
        6. SVL vocabulary

    \b
    Examples:
        mindcore demo           # Full interactive demo
        mindcore demo --quick   # Quick demo, no pauses
        mindcore demo --section 1   # Just section 1
    """
    # Override pause if quick mode
    global pause
    original_pause = pause

    def noop_pause(_msg: str = "") -> None:
        pass

    if quick:
        pause = noop_pause

    click.echo()
    click.echo(styled("  ╔════════════════════════════════════════════════╗", Colors.CYAN))
    click.echo(
        styled("  ║          Mindcore Interactive Demo             ║", Colors.CYAN, Colors.BOLD)
    )
    click.echo(styled("  ╚════════════════════════════════════════════════╝", Colors.CYAN))
    click.echo()
    click.echo(styled("  This demo shows Mindcore's core features.", Colors.DIM))
    click.echo(styled("  Press Enter to proceed through each section.", Colors.DIM))

    if not quick:
        pause("Press Enter to start...")

    try:
        if section is None or section == 1:
            memory = run_basic_demo()
        else:
            from mindcore import Mindcore
            from mindcore.svl import SharedVocabularyLayer

            vocab = SharedVocabularyLayer()
            vocab.add_topics("preferences", "settings", "context", "programming")
            memory = Mindcore(storage="sqlite:///:memory:", vocabulary=vocab)

            # Pre-populate for later sections
            for content, topics in [
                ("User prefers dark mode", ["preferences"]),
                ("User is a developer", ["context"]),
                ("User likes Python", ["programming"]),
            ]:
                memory.store(
                    content=content,
                    memory_type="preference",
                    user_id="demo_user",
                    topics=topics,
                )

        if section is None or section == 3:
            run_reinforcement_demo(memory)

        if section is None or section == 4:
            run_search_demo(memory)

        if section is None or section == 5:
            run_context_demo(memory)

        if section is None or section == 6:
            run_svl_demo()

        if section is None:
            print_next_steps()

    except KeyboardInterrupt:
        click.echo()
        click.echo(styled("  Demo interrupted.", Colors.YELLOW))
        click.echo()

    finally:
        pause = original_pause
