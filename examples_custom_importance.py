"""
Custom Importance Algorithms Example

This example demonstrates how to use and create custom importance scoring
algorithms in Mindcore.

Mindcore provides 6 built-in algorithms:
1. LLM-based (default) - AI-generated importance
2. Keyword-based - Scores based on keywords
3. Length-based - Scores based on message length
4. Sentiment-based - Scores based on sentiment intensity
5. Composite - Weighted combination of algorithms
6. Custom - Create your own!
"""

from mindcore.importance import (
    LLMBasedImportance,
    KeywordImportance,
    LengthBasedImportance,
    SentimentBasedImportance,
    CompositeImportance,
    ImportanceAlgorithm,
    get_importance_algorithm
)


def example_1_using_built_in_algorithms():
    """Example 1: Using built-in importance algorithms."""

    print("=" * 80)
    print("Example 1: Using Built-in Importance Algorithms")
    print("=" * 80)
    print()

    test_messages = [
        {
            "text": "Hi",
            "metadata": {"importance": 0.2, "sentiment": {"overall": "neutral", "score": 0.5}}
        },
        {
            "text": "This is urgent and critical! We need to fix this ASAP.",
            "metadata": {"importance": 0.9, "sentiment": {"overall": "negative", "score": 0.2}}
        },
        {
            "text": "Maybe we can discuss this later casually.",
            "metadata": {"importance": 0.3, "sentiment": {"overall": "neutral", "score": 0.5}}
        },
        {
            "text": "This is a detailed technical explanation of the system architecture...",
            "metadata": {"importance": 0.7, "sentiment": {"overall": "neutral", "score": 0.5}}
        }
    ]

    algorithms = [
        ("LLM-based", LLMBasedImportance()),
        ("Keyword", KeywordImportance()),
        ("Length", LengthBasedImportance()),
        ("Sentiment", SentimentBasedImportance()),
        ("Composite", CompositeImportance())
    ]

    for msg in test_messages:
        print(f"Message: {msg['text'][:50]}...")
        print("-" * 80)

        for name, algorithm in algorithms:
            score = algorithm.calculate(msg['text'], metadata=msg['metadata'])
            print(f"  {name:15s}: {score:.2f}")

        print()


def example_2_keyword_customization():
    """Example 2: Customizing keyword-based importance."""

    print("=" * 80)
    print("Example 2: Customizing Keyword-based Importance")
    print("=" * 80)
    print()

    # Customer support keywords
    print("Scenario: Customer Support System")
    print("-" * 80)

    support_algorithm = KeywordImportance(
        high_importance_keywords=[
            "bug", "error", "broken", "not working", "issue",
            "urgent", "critical", "asap", "help", "problem"
        ],
        low_importance_keywords=[
            "question", "wondering", "curious", "maybe", "sometime"
        ]
    )

    test_cases = [
        "Critical bug! The payment system is broken and not working.",
        "I have a small question about the product.",
        "Just wondering if maybe we could add this feature sometime.",
        "Urgent issue: Users cannot login. Need help ASAP!"
    ]

    for text in test_cases:
        score = support_algorithm.calculate(text)
        print(f"Text: {text[:60]}...")
        print(f"  Importance: {score:.2f} {'(HIGH)' if score > 0.7 else '(LOW)' if score < 0.4 else '(MEDIUM)'}")
        print()


def example_3_composite_algorithm():
    """Example 3: Creating composite importance algorithms."""

    print("=" * 80)
    print("Example 3: Composite Importance (Weighted Combination)")
    print("=" * 80)
    print()

    # Create custom composite with specific weights
    print("Custom weights: Keyword 60%, Length 40%")
    print("-" * 80)

    composite = CompositeImportance(algorithms=[
        (KeywordImportance(), 0.6),  # 60% weight
        (LengthBasedImportance(), 0.4)  # 40% weight
    ])

    test_messages = [
        "urgent",  # High keyword, low length
        "This is a very detailed and comprehensive explanation of the technical architecture and implementation details that spans multiple sentences and contains a lot of information.",  # Low keyword, high length
        "Urgent: This is a critical issue that requires immediate attention with detailed explanation.",  # High both
    ]

    for text in test_messages:
        score = composite.calculate(text)
        print(f"Text: {text[:60]}...")
        print(f"  Composite Score: {score:.2f}")
        print()


def example_4_custom_algorithm():
    """Example 4: Creating your own custom importance algorithm."""

    print("=" * 80)
    print("Example 4: Creating Custom Importance Algorithm")
    print("=" * 80)
    print()

    # Custom algorithm for e-commerce platform
    class EcommerceImportance(ImportanceAlgorithm):
        """
        Custom importance algorithm for e-commerce customer messages.

        Prioritizes:
        - Payment issues (highest)
        - Order problems (high)
        - Shipping questions (medium)
        - General inquiries (low)
        """

        def calculate(self, text, metadata=None, **kwargs):
            text_lower = text.lower()

            # Payment issues - highest priority
            if any(word in text_lower for word in ["payment", "charge", "refund", "money", "card"]):
                return 0.95

            # Order problems - high priority
            if any(word in text_lower for word in ["order", "cancel", "wrong item", "missing"]):
                return 0.85

            # Shipping questions - medium priority
            if any(word in text_lower for word in ["shipping", "delivery", "tracking", "when arrive"]):
                return 0.6

            # Account issues - medium priority
            if any(word in text_lower for word in ["account", "login", "password", "access"]):
                return 0.7

            # General inquiries - lower priority
            return 0.4

    print("Custom E-commerce Importance Algorithm")
    print("-" * 80)

    ecommerce_algo = EcommerceImportance()

    test_cases = [
        "I was charged twice for my order!",
        "Where is my package? I need tracking info.",
        "How do I reset my password?",
        "What are your business hours?",
        "My order arrived but it's the wrong item!",
        "When will you restock this product?"
    ]

    for text in test_cases:
        score = ecommerce_algo.calculate(text)
        priority = "CRITICAL" if score > 0.9 else "HIGH" if score > 0.7 else "MEDIUM" if score > 0.5 else "LOW"
        print(f"Text: {text}")
        print(f"  Score: {score:.2f} ({priority})")
        print()


def example_5_domain_specific_composite():
    """Example 5: Domain-specific composite algorithm."""

    print("=" * 80)
    print("Example 5: Domain-Specific Composite Algorithm")
    print("=" * 80)
    print()

    # Technical support system combining multiple signals
    class TechnicalSupportImportance(ImportanceAlgorithm):
        """Importance for technical support tickets."""

        def calculate(self, text, metadata=None, **kwargs):
            score = 0.5  # Base score

            text_lower = text.lower()

            # Severity keywords
            if "critical" in text_lower or "urgent" in text_lower:
                score += 0.3
            if "error" in text_lower or "bug" in text_lower:
                score += 0.2
            if "production" in text_lower or "live" in text_lower:
                score += 0.2

            # Impact indicators
            if "all users" in text_lower or "everyone" in text_lower:
                score += 0.15
            if "cannot" in text_lower or "won't" in text_lower or "not working" in text_lower:
                score += 0.15

            # Use sentiment if available
            if metadata and "sentiment" in metadata:
                sentiment = metadata["sentiment"]
                if isinstance(sentiment, dict) and sentiment.get("overall") == "negative":
                    score += 0.1

            # Clamp to [0, 1]
            return max(0.0, min(1.0, score))

    print("Technical Support Composite Algorithm")
    print("-" * 80)

    tech_algo = TechnicalSupportImportance()

    test_cases = [
        {
            "text": "Critical error in production! All users cannot login.",
            "metadata": {"sentiment": {"overall": "negative", "score": 0.1}}
        },
        {
            "text": "Minor UI bug in the settings page.",
            "metadata": {"sentiment": {"overall": "neutral", "score": 0.5}}
        },
        {
            "text": "Urgent: Payment processing not working for live transactions!",
            "metadata": {"sentiment": {"overall": "negative", "score": 0.2}}
        },
        {
            "text": "Question about API documentation.",
            "metadata": {"sentiment": {"overall": "neutral", "score": 0.5}}
        }
    ]

    for case in test_cases:
        score = tech_algo.calculate(case["text"], metadata=case["metadata"])
        priority = "CRITICAL" if score > 0.8 else "HIGH" if score > 0.6 else "MEDIUM" if score > 0.4 else "LOW"
        print(f"Text: {case['text'][:60]}...")
        print(f"  Score: {score:.2f} ({priority})")
        print()


def example_6_configuration_based():
    """Example 6: Using configuration to select algorithm."""

    print("=" * 80)
    print("Example 6: Configuration-based Algorithm Selection")
    print("=" * 80)
    print()

    print("Using get_importance_algorithm() factory function:")
    print("-" * 80)
    print()

    # Simulate different configurations
    configs = [
        ("llm", {}),
        ("keyword", {}),
        ("length", {}),
        ("sentiment", {}),
        ("composite", {})
    ]

    test_text = "This is urgent! Critical issue that needs immediate attention."
    test_metadata = {
        "importance": 0.9,
        "sentiment": {"overall": "negative", "score": 0.2}
    }

    for algo_name, kwargs in configs:
        algorithm = get_importance_algorithm(algo_name)
        score = algorithm.calculate(test_text, metadata=test_metadata, **kwargs)

        print(f"Algorithm: {algo_name:12s} → Score: {score:.2f}")

    print()
    print("To use in config.yaml:")
    print("""
importance:
  algorithm: keyword  # Choose: llm, keyword, length, sentiment, composite

  # Keyword configuration (if using keyword algorithm)
  keywords:
    high_importance: [urgent, critical, important]
    low_importance: [maybe, casual, fyi]
""")


def main():
    """Run all examples."""

    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  Mindcore Custom Importance Algorithms - Complete Examples".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    example_1_using_built_in_algorithms()
    input("Press Enter to continue...")

    example_2_keyword_customization()
    input("Press Enter to continue...")

    example_3_composite_algorithm()
    input("Press Enter to continue...")

    example_4_custom_algorithm()
    input("Press Enter to continue...")

    example_5_domain_specific_composite()
    input("Press Enter to continue...")

    example_6_configuration_based()

    print()
    print("=" * 80)
    print("All examples complete! 🎉")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  1. Use built-in algorithms for common use cases")
    print("  2. Customize keyword lists for your domain")
    print("  3. Create composite algorithms for complex scoring")
    print("  4. Implement custom algorithms for specific business logic")
    print("  5. Configure via YAML for easy deployment")
    print()


if __name__ == "__main__":
    main()
