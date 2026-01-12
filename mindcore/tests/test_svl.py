"""Tests for the Structured Validation Layer (SVL)."""

import pytest

from mindcore.svl import (
    CUSTOMER_SERVICE_DOMAIN,
    DEFAULT_SVL,
    DOMAIN_REGISTRY,
    ECOMMERCE_DOMAIN,
    HEALTHCARE_DOMAIN,
    Confidence,
    DomainLabel,
    # Domains
    DomainVocabulary,
    EmotionalClassification,
    MessageIntent,
    # Ontology
    MessageType,
    PreferenceType,
    SemanticMetadata,
    SharedVocabularyLayer,
    # Layer
    SVLSchema,
    TemporalQualifier,
    Urgency,
    UserRole,
    create_custom_domain,
    get_confidence_levels,
    get_domain,
    get_domain_labels,
    get_emotional_classifications,
    get_message_intents,
    get_message_types,
    get_preference_types,
    get_temporal_qualifiers,
    get_urgency_levels,
    get_user_roles,
    list_domains,
    merge_domains,
)


class TestOntology:
    """Tests for SVL ontology definitions."""

    def test_message_types(self):
        """Test message type enum values."""
        assert MessageType.QUERY.value == "query"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.COMMAND.value == "command"

        types = get_message_types()
        assert "query" in types
        assert "response" in types
        assert len(types) >= 10

    def test_message_intents(self):
        """Test message intent enum values."""
        assert MessageIntent.ASK_QUESTION.value == "ask_question"
        assert MessageIntent.REQUEST_ACTION.value == "request_action"
        assert MessageIntent.GREETING.value == "greeting"

        intents = get_message_intents()
        assert "ask_question" in intents
        assert len(intents) >= 20

    def test_temporal_qualifiers(self):
        """Test temporal qualifier enum values."""
        assert TemporalQualifier.DAILY.value == "daily"
        assert TemporalQualifier.PAST_EVENT.value == "past_event"
        assert TemporalQualifier.PERMANENT.value == "permanent"

        qualifiers = get_temporal_qualifiers()
        assert "daily" in qualifiers
        assert "weekly" in qualifiers
        assert len(qualifiers) >= 10

    def test_emotional_classifications(self):
        """Test emotional classification enum values."""
        assert EmotionalClassification.JOY.value == "joy"
        assert EmotionalClassification.FRUSTRATION.value == "frustration"
        assert EmotionalClassification.NEUTRAL.value == "neutral"

        emotions = get_emotional_classifications()
        assert "joy" in emotions
        assert "neutral" in emotions
        assert len(emotions) >= 15

    def test_user_roles(self):
        """Test user role enum values."""
        assert UserRole.END_USER.value == "end_user"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.CUSTOMER.value == "customer"

        roles = get_user_roles()
        assert "end_user" in roles
        assert "admin" in roles

    def test_preference_types(self):
        """Test preference type enum values."""
        assert PreferenceType.COMMUNICATION_STYLE.value == "communication_style"
        assert PreferenceType.NOTIFICATION.value == "notification"
        assert PreferenceType.THEME.value == "theme"

        prefs = get_preference_types()
        assert "communication_style" in prefs
        assert len(prefs) >= 10

    def test_domain_labels(self):
        """Test domain label enum values."""
        assert DomainLabel.CUSTOMER_SERVICE.value == "customer_service"
        assert DomainLabel.ENGINEERING.value == "engineering"

        labels = get_domain_labels()
        assert "customer_service" in labels

    def test_urgency_levels(self):
        """Test urgency level enum values."""
        assert Urgency.CRITICAL.value == "critical"
        assert Urgency.LOW.value == "low"

        levels = get_urgency_levels()
        assert "critical" in levels
        assert "low" in levels

    def test_confidence_levels(self):
        """Test confidence level enum values."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.INFERRED.value == "inferred"

        levels = get_confidence_levels()
        assert "high" in levels
        assert "inferred" in levels


class TestSemanticMetadata:
    """Tests for SemanticMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating semantic metadata."""
        metadata = SemanticMetadata(
            message_type=MessageType.QUERY,
            message_intent=MessageIntent.ASK_QUESTION,
            temporal_qualifier=TemporalQualifier.CURRENT,
            urgency=Urgency.HIGH,
        )

        assert metadata.message_type == MessageType.QUERY
        assert metadata.urgency == Urgency.HIGH

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = SemanticMetadata(
            message_type="query",
            message_intent="ask_question",
            urgency="high",
            confidence="medium",
        )

        d = metadata.to_dict()
        assert d["message_type"] == "query"
        assert d["message_intent"] == "ask_question"
        assert d["urgency"] == "high"
        assert d["confidence"] == "medium"

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "message_type": "response",
            "temporal_qualifier": "daily",
            "emotional_classification": "joy",
            "emotional_intensity": 0.8,
        }

        metadata = SemanticMetadata.from_dict(data)
        assert metadata.message_type == "response"
        assert metadata.temporal_qualifier == "daily"
        assert metadata.emotional_intensity == 0.8

    def test_custom_fields(self):
        """Test custom tags and metadata."""
        metadata = SemanticMetadata(
            custom_tags=["important", "urgent"],
            custom_metadata={"source": "api", "version": "2.0"},
        )

        d = metadata.to_dict()
        assert "important" in d["custom_tags"]
        assert d["custom_metadata"]["source"] == "api"


class TestDomainVocabulary:
    """Tests for domain vocabulary."""

    def test_customer_service_domain(self):
        """Test customer service domain."""
        domain = CUSTOMER_SERVICE_DOMAIN

        assert domain.name == "customer_service"
        assert "ticket" in domain.topics
        assert "support_request" in domain.categories
        assert "new" in domain.subcategories.get("support_request", [])
        assert "open_ticket" in domain.intents

    def test_ecommerce_domain(self):
        """Test ecommerce domain."""
        domain = ECOMMERCE_DOMAIN

        assert domain.name == "ecommerce"
        assert "cart" in domain.topics
        assert "order" in domain.categories
        assert "pending" in domain.subcategories.get("order", [])
        assert "order" in domain.entity_types

    def test_healthcare_domain(self):
        """Test healthcare domain."""
        domain = HEALTHCARE_DOMAIN

        assert domain.name == "healthcare"
        assert "prescription" in domain.topics
        assert "patient" in domain.entity_types

    def test_get_domain(self):
        """Test getting domain by name."""
        domain = get_domain("customer_service")
        assert domain is not None
        assert domain.name == "customer_service"

        missing = get_domain("nonexistent")
        assert missing is None

    def test_list_domains(self):
        """Test listing available domains."""
        domains = list_domains()

        assert "customer_service" in domains
        assert "ecommerce" in domains
        assert "healthcare" in domains
        assert "finance" in domains
        assert "saas" in domains

    def test_merge_domains(self):
        """Test merging multiple domains."""
        merged = merge_domains("customer_service", "ecommerce")

        assert "customer_service+ecommerce" in merged.name
        assert "ticket" in merged.topics  # From customer_service
        assert "cart" in merged.topics  # From ecommerce
        assert "open_ticket" in merged.intents
        assert "add_to_cart" in merged.intents

    def test_create_custom_domain(self):
        """Test creating custom domain."""
        custom = create_custom_domain(
            name="my_domain",
            topics=["custom_topic"],
            categories=["custom_category"],
        )

        assert custom.name == "my_domain"
        assert "custom_topic" in custom.topics

    def test_create_custom_domain_extending_base(self):
        """Test creating custom domain extending base."""
        custom = create_custom_domain(
            name="extended_ecommerce",
            base_domain="ecommerce",
            topics=["loyalty_program", "gift_cards"],
        )

        assert custom.name == "extended_ecommerce"
        assert "cart" in custom.topics  # From base
        assert "loyalty_program" in custom.topics  # Added


class TestSVLSchema:
    """Tests for SVL schema."""

    def test_create_schema(self):
        """Test creating SVL schema."""
        schema = SVLSchema(
            version="1.0.0",
            topics=["billing", "support"],
            domains=["customer_service"],
        )

        assert schema.version == "1.0.0"
        assert "billing" in schema.topics
        assert "customer_service" in schema.domains

    def test_get_all_topics(self):
        """Test getting all topics including domain-specific."""
        schema = SVLSchema(
            topics=["base_topic"],
            domains=["ecommerce"],
        )

        topics = schema.get_all_topics()
        assert "base_topic" in topics
        assert "cart" in topics  # From ecommerce domain

    def test_disable_features(self):
        """Test disabling SVL features."""
        schema = SVLSchema(
            enable_message_types=False,
            enable_emotional=False,
        )

        assert schema.get_message_types() == []
        assert schema.get_emotional_classifications() == []
        assert len(schema.get_urgency_levels()) > 0  # Not disabled

    def test_to_json_schema(self):
        """Test generating JSON schema via SharedVocabularyLayer."""
        svl = SharedVocabularyLayer(schema=SVLSchema(domains=["customer_service"]))
        json_schema = svl.get_json_schema()

        assert json_schema["type"] == "object"
        assert "message_type" in json_schema["properties"]
        assert "urgency" in json_schema["properties"]

    def test_validate_metadata(self):
        """Test validating metadata via SharedVocabularyLayer."""
        svl = SharedVocabularyLayer(schema=SVLSchema())

        # Valid metadata
        is_valid, errors = svl.validate_metadata(
            {
                "message_type": "query",
                "urgency": "high",
            }
        )
        assert is_valid
        assert len(errors) == 0

        # Invalid metadata
        is_valid, errors = svl.validate_metadata(
            {
                "message_type": "invalid_type",
            }
        )
        assert not is_valid
        assert len(errors) > 0


class TestSharedVocabularyLayer:
    """Tests for main SharedVocabularyLayer class."""

    def test_create_svl(self):
        """Test creating SVL instance."""
        svl = SharedVocabularyLayer()
        assert svl is not None

    def test_create_svl_with_domains(self):
        """Test creating SVL with domains."""
        svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

        active = svl.get_active_domains()
        assert "customer_service" in active
        assert "ecommerce" in active

    def test_add_remove_domain(self):
        """Test adding and removing domains."""
        svl = SharedVocabularyLayer()

        svl.add_domain("healthcare")
        assert "healthcare" in svl.get_active_domains()

        svl.remove_domain("healthcare")
        assert "healthcare" not in svl.get_active_domains()

    def test_add_invalid_domain(self):
        """Test adding invalid domain raises error when strict=True."""
        svl = SharedVocabularyLayer()

        with pytest.raises(ValueError, match="not found"):
            svl.add_domain("nonexistent_domain", strict=True)

    def test_add_topics_categories(self):
        """Test adding topics and categories."""
        svl = SharedVocabularyLayer()

        svl.add_topics("custom1", "custom2")
        svl.add_categories("cat1", "cat2")
        svl.add_subcategory("cat1", "sub1", "sub2")

        assert "custom1" in svl.schema.topics
        assert "cat1" in svl.schema.categories
        assert "sub1" in svl.schema.subcategories["cat1"]

    def test_validate_metadata(self):
        """Test validating metadata through SVL."""
        svl = SharedVocabularyLayer()

        # Valid
        is_valid, _errors = svl.validate_metadata(
            {
                "message_type": "query",
                "message_intent": "ask_question",
            }
        )
        assert is_valid

        # Invalid
        is_valid, _errors = svl.validate_metadata(
            {
                "emotional_intensity": 5.0,  # Out of range
            }
        )
        assert not is_valid

    def test_get_json_schema(self):
        """Test getting JSON schema."""
        svl = SharedVocabularyLayer(domains=["customer_service"])
        schema = svl.get_json_schema()

        assert "properties" in schema
        assert "message_type" in schema["properties"]

    def test_get_full_json_schema(self):
        """Test getting full memory JSON schema."""
        svl = SharedVocabularyLayer(domains=["ecommerce"])
        # Get memory schema without response wrapper
        schema = svl.get_full_memory_schema(include_response=False)

        assert "content" in schema["properties"]
        assert "memory_type" in schema["properties"]
        assert "semantic_metadata" in schema["properties"]
        assert "cart" in schema["properties"]["topics"]["items"]["enum"]

        # With response wrapper
        full_schema = svl.get_full_memory_schema(include_response=True)
        assert "response" in full_schema["properties"]
        assert "memories_to_store" in full_schema["properties"]

    def test_get_prompt_instructions(self):
        """Test getting prompt instructions."""
        svl = SharedVocabularyLayer(domains=["customer_service"])
        instructions = svl.get_prompt_instructions()

        assert "Structured Validation Layer" in instructions
        assert "message_type" in instructions
        assert "customer_service" in instructions

    def test_enrich_memory(self):
        """Test enriching memory with defaults."""
        svl = SharedVocabularyLayer()

        memory = {"content": "test", "memory_type": "semantic"}
        enriched = svl.enrich_memory(memory, defaults={"urgency": "low"})

        assert "semantic_metadata" in enriched
        assert enriched["semantic_metadata"]["urgency"] == "low"

    def test_serialization(self):
        """Test serializing and deserializing SVL."""
        svl = SharedVocabularyLayer(domains=["ecommerce"])
        svl.add_topics("custom")

        data = svl.to_dict()
        restored = SharedVocabularyLayer.from_dict(data)

        assert "ecommerce" in restored.get_active_domains()
        assert "custom" in restored.schema.topics

    def test_default_svl(self):
        """Test default SVL instance."""
        assert DEFAULT_SVL is not None
        assert "customer_service" in DEFAULT_SVL.get_active_domains()


class TestIntegration:
    """Integration tests for SVL with other components."""

    def test_svl_with_semantic_metadata(self):
        """Test SVL validation with SemanticMetadata object."""
        svl = SharedVocabularyLayer()

        metadata = SemanticMetadata(
            message_type=MessageType.QUERY,
            message_intent=MessageIntent.REQUEST_ACTION,
            urgency=Urgency.HIGH,
        )

        is_valid, _errors = svl.validate_metadata(metadata)
        assert is_valid

    def test_domain_entity_types(self):
        """Test getting entity types from domains."""
        svl = SharedVocabularyLayer(domains=["ecommerce", "healthcare"])

        entity_types = svl.schema.get_entity_types()
        assert "order" in entity_types  # From ecommerce
        assert "patient" in entity_types  # From healthcare

    def test_combined_intents(self):
        """Test combined intents from base + domains."""
        svl = SharedVocabularyLayer(domains=["ecommerce"])

        intents = svl.schema.get_message_intents()
        assert "ask_question" in intents  # Base ontology
        assert "add_to_cart" in intents  # Ecommerce domain
