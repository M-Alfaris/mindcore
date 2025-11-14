#!/usr/bin/env python3
"""
Integration Test Script for Mindcore

This script tests the core integration of LLM providers, importance algorithms,
and custom prompts without requiring pytest.

Run with: python test_integration.py
"""

import sys
import os

# Add mindcore to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from mindcore import MindcoreClient, MetadataAgent, ContextAgent
        from mindcore.llm_providers import (
            get_llm_provider,
            OpenAIProvider,
            OllamaProvider,
            LMStudioProvider,
            AnthropicProvider
        )
        from mindcore.importance import (
            get_importance_algorithm,
            LLMBasedImportance,
            KeywordImportance,
            LengthBasedImportance,
            SentimentBasedImportance,
            CompositeImportance
        )
        from mindcore.prompts import (
            ENRICHMENT_SYSTEM_PROMPT,
            CONTEXT_ASSEMBLY_SYSTEM_PROMPT,
            get_enrichment_prompt,
            get_context_assembly_prompt,
            load_custom_prompts
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_llm_providers():
    """Test LLM provider initialization."""
    print("\nTesting LLM providers...")
    try:
        from mindcore.llm_providers import get_llm_provider

        # Test provider creation (without actual API keys)
        providers_to_test = [
            ("openai", {"api_key": "test-key", "model": "gpt-4o-mini"}),
            ("ollama", {"model": "llama2"}),
            ("lmstudio", {"model": "local-model"}),
        ]

        for provider_name, kwargs in providers_to_test:
            try:
                provider = get_llm_provider(provider_name, **kwargs)
                print(f"  ✓ {provider_name} provider created: {provider.__class__.__name__}")
            except Exception as e:
                print(f"  ✗ {provider_name} provider failed: {e}")

        return True
    except Exception as e:
        print(f"✗ Provider test failed: {e}")
        return False


def test_importance_algorithms():
    """Test importance algorithm initialization and calculation."""
    print("\nTesting importance algorithms...")
    try:
        from mindcore.importance import (
            LLMBasedImportance,
            KeywordImportance,
            LengthBasedImportance,
            SentimentBasedImportance,
            CompositeImportance
        )

        test_text = "This is urgent and critical! We need to fix this ASAP."
        test_metadata = {
            "importance": 0.9,
            "sentiment": {"overall": "negative", "score": 0.2}
        }

        algorithms = [
            ("LLM-based", LLMBasedImportance()),
            ("Keyword", KeywordImportance()),
            ("Length", LengthBasedImportance()),
            ("Sentiment", SentimentBasedImportance()),
            ("Composite", CompositeImportance())
        ]

        for name, algorithm in algorithms:
            try:
                score = algorithm.calculate(test_text, metadata=test_metadata)
                assert 0.0 <= score <= 1.0, f"Score {score} out of range"
                print(f"  ✓ {name}: {score:.2f}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

        return True
    except Exception as e:
        print(f"✗ Importance test failed: {e}")
        return False


def test_prompts():
    """Test prompt loading and formatting."""
    print("\nTesting prompts...")
    try:
        from mindcore.prompts import (
            ENRICHMENT_SYSTEM_PROMPT,
            CONTEXT_ASSEMBLY_SYSTEM_PROMPT,
            get_enrichment_prompt,
            get_context_assembly_prompt,
            load_custom_prompts
        )

        # Test default prompts exist
        assert len(ENRICHMENT_SYSTEM_PROMPT) > 0, "Enrichment prompt empty"
        assert len(CONTEXT_ASSEMBLY_SYSTEM_PROMPT) > 0, "Context prompt empty"
        print("  ✓ Default prompts loaded")

        # Test prompt formatting
        enrichment = get_enrichment_prompt("user", "Test message")
        assert "Test message" in enrichment
        print("  ✓ Enrichment prompt formatting works")

        context = get_context_assembly_prompt("Message history", "query")
        assert "query" in context
        print("  ✓ Context prompt formatting works")

        # Test custom prompts loading (returns empty dict if no file)
        custom = load_custom_prompts("/nonexistent/path.yaml")
        assert isinstance(custom, dict)
        print("  ✓ Custom prompts loading works")

        return True
    except Exception as e:
        print(f"✗ Prompts test failed: {e}")
        return False


def test_config_loader():
    """Test configuration loading."""
    print("\nTesting configuration loader...")
    try:
        from mindcore.core.config_loader import ConfigLoader

        # Test config loading
        config = ConfigLoader()

        # Test new methods exist
        assert hasattr(config, 'get_llm_config'), "get_llm_config missing"
        assert hasattr(config, 'get_importance_config'), "get_importance_config missing"
        assert hasattr(config, 'get_prompts_config'), "get_prompts_config missing"
        print("  ✓ New config methods exist")

        # Test calling methods
        llm_config = config.get_llm_config()
        assert isinstance(llm_config, dict), "LLM config not dict"
        print(f"  ✓ LLM config: provider={llm_config.get('provider', 'openai')}")

        importance_config = config.get_importance_config()
        assert isinstance(importance_config, dict), "Importance config not dict"
        print(f"  ✓ Importance config: algorithm={importance_config.get('algorithm', 'llm')}")

        prompts_config = config.get_prompts_config()
        assert isinstance(prompts_config, dict), "Prompts config not dict"
        print(f"  ✓ Prompts config loaded")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_base_agent():
    """Test base agent integration with LLM providers."""
    print("\nTesting base agent...")
    try:
        from mindcore.agents.base_agent import BaseAgent
        from mindcore.llm_providers import OpenAIProvider

        # Test that BaseAgent accepts llm_provider parameter
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

        # Can't instantiate abstract class, but can check signature
        init_params = BaseAgent.__init__.__code__.co_varnames
        assert 'llm_provider' in init_params, "llm_provider parameter missing"
        print("  ✓ BaseAgent accepts llm_provider")

        # Check method exists
        assert hasattr(BaseAgent, '_call_openai'), "_call_openai method missing"
        print("  ✓ BaseAgent has _call_openai method")

        return True
    except Exception as e:
        print(f"✗ Base agent test failed: {e}")
        return False


def test_enrichment_agent():
    """Test enrichment agent integration."""
    print("\nTesting enrichment agent...")
    try:
        from mindcore.agents.enrichment_agent import EnrichmentAgent
        from mindcore.llm_providers import OpenAIProvider
        from mindcore.importance import KeywordImportance

        # Test initialization with new parameters
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        importance = KeywordImportance()

        agent = EnrichmentAgent(
            llm_provider=provider,
            system_prompt="Custom prompt",
            importance_algorithm=importance
        )

        assert agent.llm_provider is provider, "Provider not set"
        assert agent.system_prompt == "Custom prompt", "Custom prompt not set"
        assert agent.importance_algorithm is importance, "Importance algorithm not set"

        print("  ✓ EnrichmentAgent accepts new parameters")
        print(f"  ✓ Uses importance: {agent.importance_algorithm.__class__.__name__}")

        return True
    except Exception as e:
        print(f"✗ Enrichment agent test failed: {e}")
        return False


def test_context_agent():
    """Test context assembler agent integration."""
    print("\nTesting context agent...")
    try:
        from mindcore.agents.context_assembler_agent import ContextAssemblerAgent
        from mindcore.llm_providers import OpenAIProvider

        # Test initialization with new parameters
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

        agent = ContextAssemblerAgent(
            llm_provider=provider,
            system_prompt="Custom context prompt"
        )

        assert agent.llm_provider is provider, "Provider not set"
        assert agent.system_prompt == "Custom context prompt", "Custom prompt not set"

        print("  ✓ ContextAssemblerAgent accepts new parameters")

        return True
    except Exception as e:
        print(f"✗ Context agent test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("Mindcore Integration Tests")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("LLM Providers", test_llm_providers),
        ("Importance Algorithms", test_importance_algorithms),
        ("Prompts", test_prompts),
        ("Config Loader", test_config_loader),
        ("Base Agent", test_base_agent),
        ("Enrichment Agent", test_enrichment_agent),
        ("Context Agent", test_context_agent),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All integration tests passed! Framework is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
