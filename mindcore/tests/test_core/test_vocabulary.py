"""Comprehensive tests for VocabularyManager.

CRITICAL: VocabularyManager is the central vocabulary control for all
enrichment and retrieval. Every agent depends on it for controlled vocabulary.
"""

import json
import tempfile
from pathlib import Path

import pytest

from mindcore.core.vocabulary import (
    CommunicationStyle,
    EntityType,
    Intent,
    Sentiment,
    VocabularyEntry,
    VocabularyManager,
    VocabularySource,
    get_vocabulary,
    reset_vocabulary,
)


class TestVocabularyEnums:
    """Tests for vocabulary-related enums."""

    def test_intent_values(self):
        """Test Intent enum has expected values."""
        assert Intent.ASK_QUESTION.value == "ask_question"
        assert Intent.REQUEST_ACTION.value == "request_action"
        assert Intent.PROVIDE_INFO.value == "provide_info"
        assert Intent.COMPLAINT.value == "complaint"
        assert Intent.GREETING.value == "greeting"
        assert Intent.FAREWELL.value == "farewell"

    def test_intent_values_class_method(self):
        """Test Intent.values() returns all values."""
        values = Intent.values()
        assert "ask_question" in values
        assert "greeting" in values
        assert len(values) == len(Intent)

    def test_sentiment_values(self):
        """Test Sentiment enum has expected values."""
        assert Sentiment.POSITIVE.value == "positive"
        assert Sentiment.NEGATIVE.value == "negative"
        assert Sentiment.NEUTRAL.value == "neutral"
        assert Sentiment.MIXED.value == "mixed"

    def test_sentiment_values_class_method(self):
        """Test Sentiment.values() returns all values."""
        values = Sentiment.values()
        assert len(values) == 4
        assert "positive" in values
        assert "neutral" in values

    def test_communication_style_values(self):
        """Test CommunicationStyle enum values."""
        assert CommunicationStyle.FORMAL.value == "formal"
        assert CommunicationStyle.CASUAL.value == "casual"
        assert CommunicationStyle.TECHNICAL.value == "technical"
        assert CommunicationStyle.BALANCED.value == "balanced"

    def test_entity_type_values(self):
        """Test EntityType enum has expected values."""
        assert EntityType.ORDER_ID.value == "order_id"
        assert EntityType.INVOICE_ID.value == "invoice_id"
        assert EntityType.DATE.value == "date"
        assert EntityType.EMAIL.value == "email"
        assert EntityType.PHONE.value == "phone"

    def test_vocabulary_source_values(self):
        """Test VocabularySource enum values."""
        assert VocabularySource.CORE.value == "core"
        assert VocabularySource.CONNECTOR.value == "connector"
        assert VocabularySource.USER.value == "user"
        assert VocabularySource.EXTERNAL.value == "external"


class TestVocabularyEntry:
    """Tests for VocabularyEntry dataclass."""

    def test_vocabulary_entry_creation(self):
        """Test creating a vocabulary entry."""
        entry = VocabularyEntry(
            value="billing",
            source=VocabularySource.CORE,
            description="Billing-related topics",
        )

        assert entry.value == "billing"
        assert entry.source == VocabularySource.CORE
        assert entry.description == "Billing-related topics"

    def test_vocabulary_entry_defaults(self):
        """Test vocabulary entry default values."""
        entry = VocabularyEntry(value="test", source=VocabularySource.USER)

        assert entry.description is None
        assert entry.aliases == []
        assert entry.parent is None
        assert entry.metadata == {}

    def test_vocabulary_entry_with_aliases(self):
        """Test vocabulary entry with aliases."""
        entry = VocabularyEntry(
            value="billing",
            source=VocabularySource.CORE,
            aliases=["payment", "invoice", "charges"],
        )

        assert len(entry.aliases) == 3
        assert "payment" in entry.aliases


class TestVocabularyManagerInitialization:
    """Tests for VocabularyManager initialization."""

    def test_initialization_loads_core_vocabulary(self):
        """Test that core vocabulary is loaded on initialization."""
        vocab = VocabularyManager()

        # Check core topics exist (support is a category, not a topic)
        topics = vocab.get_topics()
        assert "greeting" in topics
        assert "billing" in topics
        assert "orders" in topics
        assert "order" in topics  # Both order and orders are topics

    def test_initialization_loads_core_categories(self):
        """Test that core categories are loaded."""
        vocab = VocabularyManager()

        categories = vocab.get_categories()
        assert "support" in categories
        assert "billing" in categories
        assert "technical" in categories

    def test_initialization_sets_up_mappings(self):
        """Test that common topic mappings are set up."""
        vocab = VocabularyManager()

        # "order" exists as a direct topic, so it returns itself (not "orders")
        # Test mappings that don't have direct topic matches
        assert vocab.resolve_topic("ship") == "shipping"
        assert vocab.resolve_topic("pay") == "payment"
        assert vocab.resolve_topic("docs") == "documentation"


class TestVocabularyManagerTopics:
    """Tests for topic management in VocabularyManager."""

    def test_get_topics_returns_all(self):
        """Test get_topics returns all registered topics."""
        vocab = VocabularyManager()
        topics = vocab.get_topics()

        assert len(topics) > 0
        assert isinstance(topics, list)

    def test_get_topics_filter_by_source(self):
        """Test filtering topics by source."""
        vocab = VocabularyManager()

        core_topics = vocab.get_topics(include_sources=[VocabularySource.CORE])
        assert len(core_topics) > 0

        # No user topics initially
        user_topics = vocab.get_topics(include_sources=[VocabularySource.USER])
        assert len(user_topics) == 0

    def test_is_valid_topic_core(self):
        """Test validating core topics."""
        vocab = VocabularyManager()

        assert vocab.is_valid_topic("billing") is True
        assert vocab.is_valid_topic("greeting") is True
        assert vocab.is_valid_topic("orders") is True

    def test_is_valid_topic_invalid(self):
        """Test invalid topics return False."""
        vocab = VocabularyManager()

        assert vocab.is_valid_topic("completely_invalid_topic") is False
        assert vocab.is_valid_topic("random_garbage_xyz") is False

    def test_is_valid_topic_case_insensitive(self):
        """Test topic validation is case-insensitive."""
        vocab = VocabularyManager()

        assert vocab.is_valid_topic("BILLING") is True
        assert vocab.is_valid_topic("Billing") is True
        assert vocab.is_valid_topic("BiLLiNg") is True

    def test_is_valid_topic_via_mapping(self):
        """Test topics valid through mappings."""
        vocab = VocabularyManager()

        # "order" maps to "orders"
        assert vocab.is_valid_topic("order") is True
        assert vocab.is_valid_topic("pay") is True

    def test_resolve_topic_direct(self):
        """Test resolving a direct topic match."""
        vocab = VocabularyManager()

        assert vocab.resolve_topic("billing") == "billing"
        assert vocab.resolve_topic("orders") == "orders"

    def test_resolve_topic_via_mapping(self):
        """Test resolving topic through mapping."""
        vocab = VocabularyManager()

        # "order" exists as a direct topic, returns itself
        assert vocab.resolve_topic("order") == "order"
        # These are only mappings, no direct topic exists
        assert vocab.resolve_topic("ship") == "shipping"
        assert vocab.resolve_topic("pay") == "payment"
        assert vocab.resolve_topic("docs") == "documentation"

    def test_resolve_topic_case_insensitive(self):
        """Test topic resolution is case-insensitive."""
        vocab = VocabularyManager()

        assert vocab.resolve_topic("BILLING") == "billing"
        assert vocab.resolve_topic("ORDER") == "order"  # Direct match

    def test_resolve_topic_invalid_returns_none(self):
        """Test resolving invalid topic returns None."""
        vocab = VocabularyManager()

        assert vocab.resolve_topic("invalid_topic") is None
        assert vocab.resolve_topic("") is None

    def test_validate_topics_filters_invalid(self):
        """Test validating a list of topics."""
        vocab = VocabularyManager()

        topics = ["billing", "invalid1", "orders", "invalid2", "greeting"]
        valid = vocab.validate_topics(topics)

        assert "billing" in valid
        assert "orders" in valid
        assert "greeting" in valid
        assert "invalid1" not in valid
        assert "invalid2" not in valid

    def test_validate_topics_resolves_mappings(self):
        """Test validate_topics resolves mappings."""
        vocab = VocabularyManager()

        # "order" is a direct topic, "pay" maps to "payment"
        topics = ["order", "pay", "billing"]
        valid = vocab.validate_topics(topics)

        assert "order" in valid  # Direct match
        assert "payment" in valid  # Via mapping
        assert "billing" in valid

    def test_validate_topics_removes_duplicates(self):
        """Test validate_topics removes duplicates."""
        vocab = VocabularyManager()

        topics = ["billing", "billing", "BILLING"]
        valid = vocab.validate_topics(topics)

        assert valid.count("billing") == 1

    def test_register_topics_user(self):
        """Test registering user topics."""
        vocab = VocabularyManager()

        vocab.register_topics(["custom_topic1", "custom_topic2"], source="user")

        assert vocab.is_valid_topic("custom_topic1") is True
        assert vocab.is_valid_topic("custom_topic2") is True

    def test_register_topics_with_descriptions(self):
        """Test registering topics with descriptions."""
        vocab = VocabularyManager()

        vocab.register_topics(
            ["custom_topic"],
            source="user",
            descriptions={"custom_topic": "A custom topic for testing"},
        )

        assert vocab.is_valid_topic("custom_topic") is True

    def test_register_topics_normalizes_case(self):
        """Test registered topics are normalized to lowercase."""
        vocab = VocabularyManager()

        vocab.register_topics(["UPPERCASE_TOPIC"], source="user")

        assert vocab.is_valid_topic("uppercase_topic") is True
        assert vocab.is_valid_topic("UPPERCASE_TOPIC") is True

    def test_add_topic_mapping(self):
        """Test adding a topic mapping."""
        vocab = VocabularyManager()

        # Map external term to internal topic (must be an existing topic, not category)
        # "billing" is both a topic and category
        result = vocab.add_topic_mapping("customer_billing", "billing")

        assert result is True
        assert vocab.resolve_topic("customer_billing") == "billing"

    def test_add_topic_mapping_invalid_target(self):
        """Test adding mapping to invalid topic fails."""
        vocab = VocabularyManager()

        result = vocab.add_topic_mapping("external", "nonexistent_topic")

        assert result is False


class TestVocabularyManagerCategories:
    """Tests for category management in VocabularyManager."""

    def test_get_categories(self):
        """Test getting all categories."""
        vocab = VocabularyManager()
        categories = vocab.get_categories()

        assert len(categories) > 0
        assert "support" in categories
        assert "billing" in categories

    def test_get_categories_filter_by_source(self):
        """Test filtering categories by source."""
        vocab = VocabularyManager()

        core_cats = vocab.get_categories(include_sources=[VocabularySource.CORE])
        assert len(core_cats) > 0

    def test_is_valid_category(self):
        """Test validating categories."""
        vocab = VocabularyManager()

        assert vocab.is_valid_category("support") is True
        assert vocab.is_valid_category("billing") is True
        assert vocab.is_valid_category("invalid_category") is False

    def test_is_valid_category_case_insensitive(self):
        """Test category validation is case-insensitive."""
        vocab = VocabularyManager()

        assert vocab.is_valid_category("SUPPORT") is True
        assert vocab.is_valid_category("Support") is True

    def test_resolve_category(self):
        """Test resolving category."""
        vocab = VocabularyManager()

        assert vocab.resolve_category("support") == "support"
        assert vocab.resolve_category("BILLING") == "billing"
        assert vocab.resolve_category("invalid") is None

    def test_validate_categories(self):
        """Test validating list of categories."""
        vocab = VocabularyManager()

        categories = ["support", "invalid", "billing"]
        valid = vocab.validate_categories(categories)

        assert "support" in valid
        assert "billing" in valid
        assert "invalid" not in valid

    def test_register_categories(self):
        """Test registering custom categories."""
        vocab = VocabularyManager()

        vocab.register_categories(["custom_cat"], source="user")

        assert vocab.is_valid_category("custom_cat") is True


class TestVocabularyManagerConnectors:
    """Tests for connector integration in VocabularyManager."""

    def test_register_connector_topics(self):
        """Test registering topics for a connector."""
        vocab = VocabularyManager()

        vocab.register_connector_topics("orders_connector", ["orders", "shipping", "delivery"])

        # Topics should be valid
        assert vocab.is_valid_topic("orders") is True
        assert vocab.is_valid_topic("shipping") is True

    def test_register_connector_topics_adds_new(self):
        """Test that connector can add new topics."""
        vocab = VocabularyManager()

        vocab.register_connector_topics("custom_connector", ["new_connector_topic"])

        assert vocab.is_valid_topic("new_connector_topic") is True

    def test_get_connector_for_topics(self):
        """Test finding connectors for topics."""
        vocab = VocabularyManager()

        vocab.register_connector_topics("orders", ["orders", "shipping"])
        vocab.register_connector_topics("billing", ["billing", "payment"])

        # Find connector for order topics
        connectors = vocab.get_connector_for_topics(["orders", "shipping"])
        assert "orders" in connectors

        # Find connector for billing topics
        connectors = vocab.get_connector_for_topics(["billing"])
        assert "billing" in connectors

        # Find both
        connectors = vocab.get_connector_for_topics(["orders", "billing"])
        assert "orders" in connectors
        assert "billing" in connectors

    def test_get_connector_for_topics_no_match(self):
        """Test no connector found for unregistered topics."""
        vocab = VocabularyManager()

        connectors = vocab.get_connector_for_topics(["greeting"])

        assert connectors == []


class TestVocabularyManagerIntentsSentiments:
    """Tests for intent and sentiment handling."""

    def test_get_intents(self):
        """Test getting all intents."""
        vocab = VocabularyManager()
        intents = vocab.get_intents()

        assert "ask_question" in intents
        assert "request_action" in intents
        assert "greeting" in intents

    def test_is_valid_intent(self):
        """Test intent validation."""
        vocab = VocabularyManager()

        assert vocab.is_valid_intent("ask_question") is True
        assert vocab.is_valid_intent("greeting") is True
        assert vocab.is_valid_intent("invalid_intent") is False

    def test_resolve_intent_direct(self):
        """Test resolving direct intent match."""
        vocab = VocabularyManager()

        assert vocab.resolve_intent("ask_question") == "ask_question"
        assert vocab.resolve_intent("greeting") == "greeting"

    def test_resolve_intent_alias(self):
        """Test resolving intent through common aliases."""
        vocab = VocabularyManager()

        assert vocab.resolve_intent("question") == "ask_question"
        assert vocab.resolve_intent("ask") == "ask_question"
        assert vocab.resolve_intent("hello") == "greeting"
        assert vocab.resolve_intent("bye") == "farewell"

    def test_resolve_intent_invalid_returns_none(self):
        """Test invalid intent returns None."""
        vocab = VocabularyManager()

        assert vocab.resolve_intent("invalid") is None

    def test_get_sentiments(self):
        """Test getting all sentiments."""
        vocab = VocabularyManager()
        sentiments = vocab.get_sentiments()

        assert "positive" in sentiments
        assert "negative" in sentiments
        assert "neutral" in sentiments
        assert "mixed" in sentiments

    def test_is_valid_sentiment(self):
        """Test sentiment validation."""
        vocab = VocabularyManager()

        assert vocab.is_valid_sentiment("positive") is True
        assert vocab.is_valid_sentiment("NEGATIVE") is True
        assert vocab.is_valid_sentiment("invalid") is False

    def test_get_entity_types(self):
        """Test getting entity types."""
        vocab = VocabularyManager()
        types = vocab.get_entity_types()

        assert "order_id" in types
        assert "date" in types
        assert "email" in types

    def test_get_communication_styles(self):
        """Test getting communication styles."""
        vocab = VocabularyManager()
        styles = vocab.get_communication_styles()

        assert "formal" in styles
        assert "casual" in styles
        assert "balanced" in styles


class TestVocabularyManagerPromptGeneration:
    """Tests for LLM prompt generation."""

    def test_to_prompt_list(self):
        """Test generating prompt-friendly vocabulary list."""
        vocab = VocabularyManager()
        prompt = vocab.to_prompt_list()

        assert "Available Topics:" in prompt
        assert "Available Categories:" in prompt
        assert "Available Intents:" in prompt
        assert "Available Sentiments:" in prompt

    def test_to_prompt_list_includes_topics(self):
        """Test prompt list includes topics."""
        vocab = VocabularyManager()
        prompt = vocab.to_prompt_list()

        assert "billing" in prompt.lower()
        assert "orders" in prompt.lower()

    def test_to_json_schema(self):
        """Test generating JSON schema for tools."""
        vocab = VocabularyManager()
        schema = vocab.to_json_schema()

        assert "topics" in schema
        assert "categories" in schema
        assert "intent" in schema
        assert "sentiment" in schema

        # Check structure
        assert schema["topics"]["type"] == "array"
        assert schema["intent"]["type"] == "string"
        assert "enum" in schema["intent"]


class TestVocabularyManagerExternalIntegration:
    """Tests for external vocabulary loading."""

    def test_load_from_json_file(self):
        """Test loading vocabulary from JSON file."""
        vocab = VocabularyManager()

        # Create temp JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "topics": ["custom_json_topic"],
                    "categories": ["custom_json_category"],
                    "topic_mappings": {"external_term": "billing"},
                },
                f,
            )
            temp_path = f.name

        try:
            vocab.load_from_json(temp_path)

            assert vocab.is_valid_topic("custom_json_topic") is True
            assert vocab.is_valid_category("custom_json_category") is True
            assert vocab.resolve_topic("external_term") == "billing"
        finally:
            Path(temp_path).unlink()

    def test_load_from_json_nonexistent_file(self):
        """Test loading from nonexistent file doesn't crash."""
        vocab = VocabularyManager()

        # Should not raise, just log warning
        vocab.load_from_json("/nonexistent/path/to/file.json")

    def test_register_external_loader(self):
        """Test registering an external loader."""
        vocab = VocabularyManager()

        def loader():
            return {"topics": ["dynamic_topic"], "categories": ["dynamic_category"]}

        vocab.register_external_loader(loader)
        vocab.refresh_from_external()

        assert vocab.is_valid_topic("dynamic_topic") is True
        assert vocab.is_valid_category("dynamic_category") is True

    def test_refresh_from_external_handles_errors(self):
        """Test external loader errors are handled gracefully."""
        vocab = VocabularyManager()

        def failing_loader():
            raise ValueError("Loader failed!")

        vocab.register_external_loader(failing_loader)

        # Should not raise
        vocab.refresh_from_external()


class TestVocabularyManagerStats:
    """Tests for vocabulary statistics."""

    def test_get_stats(self):
        """Test getting vocabulary statistics."""
        vocab = VocabularyManager()
        stats = vocab.get_stats()

        assert "total_topics" in stats
        assert "total_categories" in stats
        assert "topic_mappings" in stats
        assert "category_mappings" in stats
        assert "registered_connectors" in stats
        assert "topics_by_source" in stats

        assert stats["total_topics"] > 0
        assert stats["total_categories"] > 0

    def test_get_stats_by_source(self):
        """Test stats include breakdown by source."""
        vocab = VocabularyManager()
        vocab.register_topics(["user_topic"], source="user")

        stats = vocab.get_stats()

        assert stats["topics_by_source"]["core"] > 0
        assert stats["topics_by_source"]["user"] == 1


class TestVocabularyManagerThreadSafety:
    """Tests for thread safety."""

    def test_has_lock(self):
        """Test vocabulary manager has a lock for thread safety."""
        vocab = VocabularyManager()
        assert vocab._lock is not None

    def test_concurrent_access(self):
        """Test concurrent access to vocabulary."""
        import concurrent.futures

        vocab = VocabularyManager()

        def read_topics():
            for _ in range(100):
                vocab.get_topics()
                vocab.is_valid_topic("billing")
            return True

        def write_topics():
            for i in range(100):
                vocab.register_topics([f"concurrent_topic_{i}"], source="user")
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(read_topics),
                executor.submit(read_topics),
                executor.submit(write_topics),
                executor.submit(read_topics),
            ]

            for future in concurrent.futures.as_completed(futures):
                assert future.result() is True


class TestVocabularyManagerSingleton:
    """Tests for singleton pattern."""

    def test_get_vocabulary_returns_same_instance(self):
        """Test get_vocabulary returns singleton."""
        reset_vocabulary()

        vocab1 = get_vocabulary()
        vocab2 = get_vocabulary()

        assert vocab1 is vocab2

    def test_reset_vocabulary_creates_new_instance(self):
        """Test reset_vocabulary clears singleton."""
        vocab1 = get_vocabulary()
        reset_vocabulary()
        vocab2 = get_vocabulary()

        assert vocab1 is not vocab2

    def test_get_vocabulary_after_customization(self):
        """Test singleton preserves customizations."""
        reset_vocabulary()

        vocab = get_vocabulary()
        vocab.register_topics(["singleton_test_topic"], source="user")

        # Get again and check customization persists
        vocab2 = get_vocabulary()
        assert vocab2.is_valid_topic("singleton_test_topic") is True

        # Cleanup
        reset_vocabulary()
