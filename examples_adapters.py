#!/usr/bin/env python3
"""
Framework integration examples for Mindcore.

Demonstrates how to use Mindcore with:
- LangChain
- LlamaIndex
- Custom AI systems
"""
import os
from mindcore import Mindcore
from mindcore.adapters import LangChainAdapter, LlamaIndexAdapter


def langchain_example():
    """Example: Integrating Mindcore with LangChain."""
    print("\n" + "="*80)
    print("LANGCHAIN INTEGRATION EXAMPLE")
    print("="*80 + "\n")

    try:
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print("❌ LangChain not installed. Install with: pip install langchain")
        return

    # Initialize Mindcore
    mindcore = Mindcore()
    adapter = LangChainAdapter(mindcore)

    # Example conversation
    user_id = "user_langchain_123"
    thread_id = "thread_langchain_456"
    session_id = "session_langchain_789"

    print("1. Ingesting LangChain messages into Mindcore...\n")

    # Simulate a conversation
    messages = [
        HumanMessage(content="I'm building a chatbot with LangChain"),
        AIMessage(content="Great! LangChain is excellent for building chatbots. What features do you need?"),
        HumanMessage(content="I need memory management and context handling"),
        AIMessage(content="For memory, you can use ConversationBufferMemory or ConversationSummaryMemory."),
        HumanMessage(content="How do I persist conversations across sessions?"),
    ]

    # Ingest conversation into Mindcore
    enriched = adapter.ingest_langchain_conversation(
        messages=messages,
        user_id=user_id,
        thread_id=thread_id,
        session_id=session_id
    )

    print(f"✓ Ingested {len(enriched)} messages\n")

    print("2. Retrieving intelligent context...\n")

    # Get context for a new query
    context = adapter.get_enhanced_context(
        user_id=user_id,
        thread_id=thread_id,
        query="memory management and persistence"
    )

    print(f"Assembled Context:\n{context.assembled_context}\n")
    print(f"Key Points:")
    for point in context.key_points:
        print(f"  • {point}")
    print()

    print("3. Injecting context into LangChain prompt...\n")

    # Create enhanced prompt
    base_prompt = "You are a helpful AI assistant specializing in LangChain."
    enhanced_prompt = adapter.inject_context_into_prompt(context, base_prompt)

    print(f"Enhanced Prompt:\n{enhanced_prompt}\n")

    print("4. Using Mindcore callback with LangChain...\n")

    # Create callback for automatic ingestion
    callback = adapter.create_langchain_callback(user_id, thread_id, session_id)

    print("✓ Callback created. Use it with:")
    print("  llm = ChatOpenAI(callbacks=[callback])\n")

    print("5. Using as LangChain Memory...\n")

    # Create LangChain-compatible memory
    memory = adapter.as_langchain_memory(user_id, thread_id, session_id)

    print(f"✓ Memory created with {len(memory.messages)} messages")
    print("  Use it with ConversationChain or ChatAgent\n")

    print("="*80)
    print("✅ LangChain integration complete!")
    print("="*80 + "\n")


def llamaindex_example():
    """Example: Integrating Mindcore with LlamaIndex."""
    print("\n" + "="*80)
    print("LLAMAINDEX INTEGRATION EXAMPLE")
    print("="*80 + "\n")

    # Initialize Mindcore
    mindcore = Mindcore()
    adapter = LlamaIndexAdapter(mindcore)

    # Example conversation
    user_id = "user_llama_123"
    thread_id = "thread_llama_456"
    session_id = "session_llama_789"

    print("1. Ingesting LlamaIndex messages...\n")

    # LlamaIndex uses dicts for chat messages
    messages = [
        {"role": "user", "content": "How do I build a RAG application?"},
        {"role": "assistant", "content": "RAG (Retrieval Augmented Generation) combines retrieval with generation. Use LlamaIndex's VectorStoreIndex."},
        {"role": "user", "content": "What vector database should I use?"},
        {"role": "assistant", "content": "Popular options: Pinecone, Weaviate, Chroma, or FAISS for local development."},
    ]

    enriched = adapter.ingest_llamaindex_conversation(
        messages=messages,
        user_id=user_id,
        thread_id=thread_id,
        session_id=session_id
    )

    print(f"✓ Ingested {len(enriched)} messages\n")

    print("2. Creating chat memory...\n")

    # Create LlamaIndex-compatible memory
    memory = adapter.create_chat_memory(user_id, thread_id, session_id)

    # Get messages
    stored_messages = memory.get_messages()
    print(f"✓ Chat memory created with {len(stored_messages)} messages\n")

    print("3. Retrieving context...\n")

    context = adapter.get_enhanced_context(
        user_id=user_id,
        thread_id=thread_id,
        query="vector databases for RAG"
    )

    print(f"Context: {context.assembled_context}\n")

    print("="*80)
    print("✅ LlamaIndex integration complete!")
    print("="*80 + "\n")


def custom_ai_example():
    """Example: Using Mindcore with a custom AI system."""
    print("\n" + "="*80)
    print("CUSTOM AI SYSTEM INTEGRATION EXAMPLE")
    print("="*80 + "\n")

    from mindcore import Mindcore
    from mindcore.utils import generate_session_id

    # Initialize
    mindcore = Mindcore()

    user_id = "user_custom_123"
    thread_id = "thread_custom_456"
    session_id = generate_session_id()

    print("1. Custom message ingestion with validation...\n")

    # Your custom AI system message format
    custom_messages = [
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "session_id": session_id,
            "role": "user",
            "text": "Build a recommendation engine"
        },
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "session_id": session_id,
            "role": "assistant",
            "text": "I'll help you build a recommendation engine. What type? Collaborative filtering, content-based, or hybrid?"
        },
    ]

    # Ingest with automatic enrichment
    for msg in custom_messages:
        try:
            enriched = mindcore.ingest_message(msg)
            print(f"✓ Enriched: {msg['text'][:50]}...")
            print(f"  Topics: {enriched.metadata.topics}")
            print(f"  Intent: {enriched.metadata.intent}")
            print(f"  Importance: {enriched.metadata.importance}")
            print()
        except ValueError as e:
            print(f"✗ Validation failed: {e}\n")

    print("2. Security validation in action...\n")

    # Try invalid input (security test)
    malicious_message = {
        "user_id": "user'; DROP TABLE messages; --",  # SQL injection attempt
        "thread_id": thread_id,
        "session_id": session_id,
        "role": "user",
        "text": "Test"
    }

    try:
        mindcore.ingest_message(malicious_message)
        print("✗ Security validation failed!\n")
    except ValueError as e:
        print(f"✓ Security validation working: {e}\n")

    print("3. Intelligent context retrieval...\n")

    # Get context
    context = mindcore.get_context(
        user_id=user_id,
        thread_id=thread_id,
        query="recommendation engine approach"
    )

    print("Assembled Context:")
    print(f"  {context.assembled_context}\n")

    print("Key Points:")
    for i, point in enumerate(context.key_points, 1):
        print(f"  {i}. {point}")
    print()

    print("4. Cost tracking...\n")

    from mindcore.utils.cost_analysis import CostAnalyzer

    analyzer = CostAnalyzer(main_model="gpt-4o")

    # Compare costs
    comparison = analyzer.compare_approaches(
        conversation_history=[msg["text"] for msg in custom_messages],
        num_requests=5
    )

    print(f"Traditional approach: ${comparison['traditional']['total_cost']:.4f}")
    print(f"Mindcore approach: ${comparison['mindcore']['total_cost']:.4f}")
    print(f"Savings: ${comparison['savings']['cost_saved']:.4f} ({comparison['savings']['cost_saved_percentage']:.1f}%)\n")

    print("="*80)
    print("✅ Custom integration complete!")
    print("="*80 + "\n")


def main():
    """Run all examples."""
    print("\n" + "🧠 MINDCORE FRAMEWORK INTEGRATION EXAMPLES\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key'\n")
        return

    # Run examples
    custom_ai_example()

    # LangChain and LlamaIndex examples (if installed)
    try:
        langchain_example()
    except Exception as e:
        print(f"⚠️  LangChain example skipped: {e}\n")

    try:
        llamaindex_example()
    except Exception as e:
        print(f"⚠️  LlamaIndex example skipped: {e}\n")

    print("\n✅ All examples completed!\n")
    print("Next steps:")
    print("  • Check SECURITY.md for security best practices")
    print("  • Check COST_EFFICIENCY.md for cost analysis")
    print("  • Integrate with your AI application")
    print()


if __name__ == "__main__":
    main()
