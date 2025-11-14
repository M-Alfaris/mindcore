"""
Mindcore with Ollama - 100% Local Inference Example

This example demonstrates using Mindcore with Ollama for completely local,
cost-free AI inference. No API keys required!

Prerequisites:
1. Install Ollama from https://ollama.ai
2. Pull a model: ollama pull llama2
3. Start Ollama server: ollama serve (runs on http://localhost:11434)

Benefits of Ollama:
- 100% free - no API costs
- 100% local - complete privacy
- No internet required after model download
- Support for many models: llama2, mistral, codellama, etc.
"""

import os
import sys
from mindcore import MindcoreClient
from mindcore.llm_providers import OllamaProvider


def main():
    """Main example function."""

    print("=" * 80)
    print("Mindcore + Ollama: 100% Local AI Example")
    print("=" * 80)
    print()

    # =============================================================================
    # METHOD 1: Using Configuration File (Recommended)
    # =============================================================================

    print("Method 1: Using config.yaml")
    print("-" * 80)
    print()
    print("Edit mindcore/config.yaml:")
    print("""
llm:
  provider: ollama
  model: llama2  # or mistral, codellama, etc.
  base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 1000
""")
    print()
    print("Then simply initialize MindcoreClient:")
    print("  client = MindcoreClient()")
    print()

    # =============================================================================
    # METHOD 2: Programmatic Configuration
    # =============================================================================

    print("Method 2: Programmatic Configuration")
    print("-" * 80)
    print()

    # Initialize Ollama provider
    ollama_provider = OllamaProvider(
        model="llama2",  # or "mistral", "codellama", etc.
        base_url="http://localhost:11434",
        temperature=0.3,
        max_tokens=1000
    )

    print(f"✓ Ollama provider initialized")
    print(f"  Model: {ollama_provider.model}")
    print(f"  Base URL: {ollama_provider.base_url}")
    print()

    # =============================================================================
    # Test Ollama Connection
    # =============================================================================

    print("Testing Ollama connection...")
    print("-" * 80)

    try:
        # Test simple chat completion
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, Mindcore!' in one sentence."}
        ]

        response = ollama_provider.chat_completion(test_messages)
        print(f"✓ Ollama is working!")
        print(f"  Response: {response}")
        print()

    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Is Ollama installed? Visit https://ollama.ai")
        print("  2. Is Ollama running? Run: ollama serve")
        print("  3. Have you pulled a model? Run: ollama pull llama2")
        print()
        return

    # =============================================================================
    # Using Mindcore with Ollama
    # =============================================================================

    print("\nUsing Mindcore with Ollama")
    print("-" * 80)
    print()

    # Note: For full integration, you would need to update MindcoreClient
    # to accept custom LLM providers. Here's how it would work:

    print("After updating config.yaml to use Ollama:")
    print()

    example_code = """
from mindcore import MindcoreClient

# Initialize client (uses Ollama from config)
client = MindcoreClient()

# Ingest messages (metadata enriched using Ollama locally!)
message = client.ingest_message({
    "user_id": "user123",
    "thread_id": "conv456",
    "session_id": "session789",
    "role": "user",
    "text": "What are best practices for building AI agents?"
})

print("Message enriched using local Ollama!")
print(f"Topics: {message.metadata.topics}")
print(f"Intent: {message.metadata.intent}")
print(f"Importance: {message.metadata.importance}")

# Get context (assembled using Ollama locally!)
context = client.get_context(
    user_id="user123",
    thread_id="conv456",
    query="AI agent architecture"
)

print("Context assembled using local Ollama!")
print(f"Context: {context.assembled_context}")
print(f"Key points: {context.key_points}")
"""

    print(example_code)
    print()

    # =============================================================================
    # Available Ollama Models
    # =============================================================================

    print("Available Ollama Models")
    print("-" * 80)
    print()
    print("Popular models you can use:")
    print("  • llama2       - Meta's Llama 2 (7B, 13B, 70B)")
    print("  • mistral      - Mistral AI's model (7B)")
    print("  • codellama    - Code-specialized Llama")
    print("  • neural-chat  - Fine-tuned for conversation")
    print("  • starling-lm  - High-quality chat model")
    print("  • orca-mini    - Small, fast model")
    print()
    print("To pull a model:")
    print("  ollama pull <model-name>")
    print()
    print("To list installed models:")
    print("  ollama list")
    print()

    # =============================================================================
    # Cost Comparison
    # =============================================================================

    print("Cost Comparison: Ollama vs OpenAI")
    print("-" * 80)
    print()
    print("Scenario: 1 million tokens processed")
    print()
    print("  OpenAI GPT-4o-mini:  $0.15 per 1M tokens")
    print("  Ollama (local):      $0.00 (FREE!)")
    print()
    print("  Savings: 100% 🎉")
    print()
    print("Trade-offs:")
    print("  ✓ Ollama: Free, private, local")
    print("  ✗ Ollama: Requires GPU/CPU, slightly slower")
    print("  ✓ OpenAI: Fast, convenient, cloud-based")
    print("  ✗ OpenAI: Costs money, requires internet")
    print()

    # =============================================================================
    # Performance Tips
    # =============================================================================

    print("Performance Tips for Ollama")
    print("-" * 80)
    print()
    print("1. Choose the right model size:")
    print("   - 7B models: Fast, good for most tasks")
    print("   - 13B models: Better quality, moderate speed")
    print("   - 70B models: Best quality, requires powerful GPU")
    print()
    print("2. Adjust temperature and max_tokens:")
    print("   - Lower temperature (0.1-0.3) for focused responses")
    print("   - Reduce max_tokens for faster responses")
    print()
    print("3. Hardware recommendations:")
    print("   - Minimum: 8GB RAM, CPU only")
    print("   - Recommended: 16GB RAM, GPU with 8GB VRAM")
    print("   - Optimal: 32GB RAM, GPU with 16GB+ VRAM")
    print()

    # =============================================================================
    # Next Steps
    # =============================================================================

    print("Next Steps")
    print("-" * 80)
    print()
    print("1. Install Ollama:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print()
    print("2. Pull a model:")
    print("   ollama pull llama2")
    print()
    print("3. Update mindcore/config.yaml:")
    print("   llm:")
    print("     provider: ollama")
    print("     model: llama2")
    print()
    print("4. Run Mindcore normally:")
    print("   client = MindcoreClient()")
    print("   # Everything runs locally now!")
    print()

    print("=" * 80)
    print("Example complete! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()
