#!/usr/bin/env python3
"""
Example usage of Mindcore framework.

This script demonstrates how to use Mindcore for:
1. Ingesting messages
2. Enriching them with metadata
3. Retrieving assembled context

Before running (choose one):
- Local LLM: Set MINDCORE_LLAMA_MODEL_PATH environment variable
- OpenAI: Set OPENAI_API_KEY environment variable
"""
import os
from mindcore import MindcoreClient
from mindcore.utils import generate_session_id


def main():
    """Run Mindcore examples."""
    print("🧠 Mindcore Examples\n")

    # Check for LLM configuration
    llama_path = os.getenv("MINDCORE_LLAMA_MODEL_PATH")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not llama_path and not openai_key:
        print("⚠️  Warning: No LLM provider configured!")
        print("Set one of:")
        print("  - MINDCORE_LLAMA_MODEL_PATH for local LLM")
        print("  - OPENAI_API_KEY for OpenAI API\n")
        print("To get started with local LLM:")
        print("  mindcore download-model")
        print("  export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf\n")
        return

    # Initialize Mindcore with SQLite (for easy local testing)
    print("1. Initializing Mindcore...")
    try:
        client = MindcoreClient(use_sqlite=True)
        print(f"✓ Mindcore initialized with {client.provider_name} provider\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}\n")
        return

    # Example conversation
    user_id = "user_example_123"
    thread_id = "thread_example_456"
    session_id = generate_session_id()

    print("2. Ingesting messages...\n")

    # Message 1: User asks about AI agents
    message1 = client.ingest_message({
        "user_id": user_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "role": "user",
        "text": "What are the best practices for building AI agents with long-term memory?"
    })

    print(f"Message 1 ingested:")
    print(f"  ID: {message1.message_id}")
    print(f"  Topics: {message1.metadata.topics}")
    print(f"  Intent: {message1.metadata.intent}")
    print(f"  Importance: {message1.metadata.importance}\n")

    # Message 2: Assistant responds
    message2 = client.ingest_message({
        "user_id": user_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "role": "assistant",
        "text": "Here are key best practices: 1) Use vector databases for semantic search, "
                "2) Implement conversation summarization, 3) Add metadata enrichment for better retrieval, "
                "4) Use sliding window context, 5) Implement importance scoring."
    })

    print(f"Message 2 ingested:")
    print(f"  ID: {message2.message_id}")
    print(f"  Topics: {message2.metadata.topics}")
    print(f"  Categories: {message2.metadata.categories}\n")

    # Message 3: User follows up
    message3 = client.ingest_message({
        "user_id": user_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "role": "user",
        "text": "Can you explain more about metadata enrichment?"
    })

    print(f"Message 3 ingested:")
    print(f"  ID: {message3.message_id}")
    print(f"  Intent: {message3.metadata.intent}\n")

    # Get assembled context
    print("3. Retrieving assembled context...\n")

    context = client.get_context(
        user_id=user_id,
        thread_id=thread_id,
        query="metadata enrichment for AI agents"
    )

    print("Assembled Context:")
    print(f"  {context.assembled_context}\n")

    print("Key Points:")
    for i, point in enumerate(context.key_points, 1):
        print(f"  {i}. {point}")
    print()

    print("Relevant Messages:")
    print(f"  {len(context.relevant_message_ids)} messages identified as relevant")
    print(f"  IDs: {context.relevant_message_ids[:3]}...\n")

    print("Context Metadata:")
    print(f"  Topics: {context.metadata.get('topics', [])}")
    print(f"  Importance: {context.metadata.get('importance', 0)}\n")

    # Get cache stats
    print("4. Cache Statistics:")
    stats = client.cache.get_stats()
    print(f"  Total threads cached: {stats['total_threads']}")
    print(f"  Total messages cached: {stats['total_messages']}")
    print(f"  Max size per thread: {stats['max_size_per_thread']}\n")

    print("✓ Examples completed successfully!")
    print("\nNext steps:")
    print("  - Try the FastAPI server: mindcore-server")
    print("  - Check the API docs: http://localhost:8000/docs")
    print("  - Integrate with your AI application")

    # Cleanup
    client.close()


if __name__ == "__main__":
    main()
