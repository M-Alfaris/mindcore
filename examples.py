#!/usr/bin/env python3
"""
Example usage of Mindcore framework.

This script demonstrates how to use Mindcore for:
1. Ingesting messages
2. Enriching them with metadata
3. Retrieving assembled context

Before running:
1. Set up PostgreSQL database
2. Set OPENAI_API_KEY environment variable
3. Update config.yaml with your database credentials
"""
import os
from mindcore import Mindcore
from mindcore.utils import generate_session_id


def main():
    """Run Mindcore examples."""
    print("🧠 Mindcore Examples\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-api-key'\n")
        return

    # Initialize Mindcore
    print("1. Initializing Mindcore...")
    try:
        mindcore = Mindcore()
        print("✓ Mindcore initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}\n")
        print("Make sure PostgreSQL is running and config.yaml is correct.")
        return

    # Example conversation
    user_id = "user_example_123"
    thread_id = "thread_example_456"
    session_id = generate_session_id()

    print("2. Ingesting messages...\n")

    # Message 1: User asks about AI agents
    message1 = mindcore.ingest_message({
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
    message2 = mindcore.ingest_message({
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
    message3 = mindcore.ingest_message({
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

    context = mindcore.get_context(
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
    stats = mindcore.cache.get_stats()
    print(f"  Total threads cached: {stats['total_threads']}")
    print(f"  Total messages cached: {stats['total_messages']}")
    print(f"  Max size per thread: {stats['max_size_per_thread']}\n")

    print("✓ Examples completed successfully!")
    print("\nNext steps:")
    print("  - Try the FastAPI server: mindcore-server")
    print("  - Check the API docs: http://localhost:8000/docs")
    print("  - Integrate with your AI application")

    # Cleanup
    mindcore.close()


if __name__ == "__main__":
    main()
