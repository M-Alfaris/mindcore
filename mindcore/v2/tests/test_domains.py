"""Comprehensive tests for SVL domain vocabularies."""

import pytest

from mindcore.v2.svl.domains import (
    DomainVocabulary,
    DOMAIN_REGISTRY,
    CUSTOMER_SERVICE_DOMAIN,
    ECOMMERCE_DOMAIN,
    HEALTHCARE_DOMAIN,
    FINANCE_DOMAIN,
    SAAS_DOMAIN,
    HR_DOMAIN,
    EDUCATION_DOMAIN,
    get_domain,
    list_domains,
    merge_domains,
    create_custom_domain,
)


class TestDomainVocabulary:
    """Test DomainVocabulary dataclass."""

    def test_create_empty_domain(self):
        """Test creating an empty domain."""
        domain = DomainVocabulary(name="test")

        assert domain.name == "test"
        assert domain.topics == []
        assert domain.categories == []
        assert domain.subcategories == {}
        assert domain.entity_types == []
        assert domain.intents == []

    def test_create_full_domain(self):
        """Test creating a domain with all fields."""
        domain = DomainVocabulary(
            name="test_domain",
            description="Test domain description",
            topics=["topic1", "topic2"],
            categories=["cat1", "cat2"],
            subcategories={"cat1": ["sub1", "sub2"]},
            entity_types=["entity1"],
            intents=["intent1"],
            relationship_types=["rel1"],
            custom_fields={"field1": {"type": "string"}},
        )

        assert domain.name == "test_domain"
        assert domain.description == "Test domain description"
        assert len(domain.topics) == 2
        assert len(domain.categories) == 2
        assert "cat1" in domain.subcategories
        assert len(domain.entity_types) == 1
        assert len(domain.intents) == 1

    def test_get_all_subcategories(self):
        """Test getting all subcategories flattened."""
        domain = DomainVocabulary(
            name="test",
            subcategories={
                "cat1": ["sub1", "sub2"],
                "cat2": ["sub3", "sub4"],
                "cat3": ["sub1"],  # Duplicate
            },
        )

        subcats = domain.get_all_subcategories()

        assert len(subcats) == 4  # Deduplicated
        assert "sub1" in subcats
        assert "sub4" in subcats

    def test_merge_domains(self):
        """Test merging two domains."""
        domain1 = DomainVocabulary(
            name="domain1",
            description="First domain",
            topics=["topic1", "topic2"],
            categories=["cat1"],
            entity_types=["entity1"],
        )

        domain2 = DomainVocabulary(
            name="domain2",
            description="Second domain",
            topics=["topic2", "topic3"],  # topic2 is shared
            categories=["cat2"],
            entity_types=["entity2"],
        )

        merged = domain1.merge_with(domain2)

        assert merged.name == "domain1+domain2"
        assert "First domain" in merged.description
        assert "Second domain" in merged.description
        assert len(merged.topics) == 3  # Deduplicated
        assert "topic1" in merged.topics
        assert "topic3" in merged.topics
        assert len(merged.categories) == 2
        assert len(merged.entity_types) == 2

    def test_to_dict(self):
        """Test converting domain to dictionary."""
        domain = DomainVocabulary(
            name="test",
            description="Test description",
            topics=["topic1"],
            categories=["cat1"],
            subcategories={"cat1": ["sub1"]},
            entity_types=["entity1"],
            intents=["intent1"],
            relationship_types=["rel1"],
            custom_fields={"field1": {"type": "string"}},
        )

        data = domain.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test description"
        assert data["topics"] == ["topic1"]
        assert data["categories"] == ["cat1"]
        assert data["subcategories"] == {"cat1": ["sub1"]}
        assert "entity_types" in data
        assert "intents" in data
        assert "custom_fields" in data

    def test_from_dict(self):
        """Test creating domain from dictionary."""
        data = {
            "name": "from_dict_test",
            "description": "Created from dict",
            "topics": ["t1", "t2"],
            "categories": ["c1"],
            "subcategories": {"c1": ["s1"]},
            "entity_types": ["e1"],
            "intents": ["i1"],
            "relationship_types": ["r1"],
            "custom_fields": {"f1": {"type": "number"}},
        }

        domain = DomainVocabulary.from_dict(data)

        assert domain.name == "from_dict_test"
        assert domain.description == "Created from dict"
        assert len(domain.topics) == 2
        assert domain.custom_fields["f1"]["type"] == "number"

    def test_roundtrip_dict(self):
        """Test that to_dict and from_dict are inverses."""
        original = DomainVocabulary(
            name="roundtrip",
            topics=["t1"],
            categories=["c1"],
            subcategories={"c1": ["s1"]},
        )

        restored = DomainVocabulary.from_dict(original.to_dict())

        assert restored.name == original.name
        assert restored.topics == original.topics
        assert restored.categories == original.categories
        assert restored.subcategories == original.subcategories


class TestBuiltinDomains:
    """Test pre-defined domain vocabularies."""

    def test_customer_service_domain(self):
        """Test customer service domain is properly defined."""
        domain = CUSTOMER_SERVICE_DOMAIN

        assert domain.name == "customer_service"
        assert len(domain.topics) > 0
        assert "ticket" in domain.topics
        assert "escalation" in domain.topics
        assert len(domain.categories) > 0
        assert "support_request" in domain.subcategories
        assert len(domain.entity_types) > 0
        assert "customer" in domain.entity_types

    def test_ecommerce_domain(self):
        """Test e-commerce domain is properly defined."""
        domain = ECOMMERCE_DOMAIN

        assert domain.name == "ecommerce"
        assert "cart" in domain.topics
        assert "checkout" in domain.topics
        assert "order" in domain.categories
        assert "order" in domain.subcategories
        assert "pending" in domain.subcategories["order"]
        assert "order_id" in domain.custom_fields

    def test_healthcare_domain(self):
        """Test healthcare domain is properly defined."""
        domain = HEALTHCARE_DOMAIN

        assert domain.name == "healthcare"
        assert "appointment" in domain.topics
        assert "prescription" in domain.topics
        assert "clinical" in domain.categories
        assert "patient" in domain.entity_types
        assert "mrn" in domain.custom_fields

    def test_finance_domain(self):
        """Test finance domain is properly defined."""
        domain = FINANCE_DOMAIN

        assert domain.name == "finance"
        assert "account" in domain.topics
        assert "transaction" in domain.topics
        assert "banking" in domain.categories
        assert "investment" in domain.subcategories
        assert "account_number" in domain.custom_fields

    def test_saas_domain(self):
        """Test SaaS domain is properly defined."""
        domain = SAAS_DOMAIN

        assert domain.name == "saas"
        assert "subscription" in domain.topics
        assert "api" in domain.topics
        assert "integration" in domain.categories
        assert "subscription" in domain.subcategories
        assert "plan_id" in domain.custom_fields

    def test_hr_domain(self):
        """Test HR domain is properly defined."""
        domain = HR_DOMAIN

        assert domain.name == "hr"
        assert "employee" in domain.topics
        assert "payroll" in domain.topics
        assert "recruitment" in domain.categories
        assert "leave" in domain.subcategories
        assert "employee" in domain.entity_types

    def test_education_domain(self):
        """Test education domain is properly defined."""
        domain = EDUCATION_DOMAIN

        assert domain.name == "education"
        assert "course" in domain.topics
        assert "lesson" in domain.topics
        assert "assessment" in domain.categories
        assert "progress" in domain.subcategories
        assert "student" in domain.entity_types


class TestDomainRegistry:
    """Test domain registry functions."""

    def test_domain_registry_populated(self):
        """Test that domain registry contains all domains."""
        assert len(DOMAIN_REGISTRY) >= 7
        assert "customer_service" in DOMAIN_REGISTRY
        assert "ecommerce" in DOMAIN_REGISTRY
        assert "healthcare" in DOMAIN_REGISTRY
        assert "finance" in DOMAIN_REGISTRY
        assert "saas" in DOMAIN_REGISTRY
        assert "hr" in DOMAIN_REGISTRY
        assert "education" in DOMAIN_REGISTRY

    def test_get_domain_existing(self):
        """Test getting an existing domain."""
        domain = get_domain("ecommerce")

        assert domain is not None
        assert domain.name == "ecommerce"
        assert domain is ECOMMERCE_DOMAIN

    def test_get_domain_nonexistent(self):
        """Test getting a non-existent domain."""
        domain = get_domain("nonexistent_domain")
        assert domain is None

    def test_list_domains(self):
        """Test listing all domains."""
        domains = list_domains()

        assert len(domains) >= 7
        assert "customer_service" in domains
        assert "ecommerce" in domains
        assert "healthcare" in domains

    def test_list_domains_returns_list(self):
        """Test that list_domains returns a list."""
        domains = list_domains()
        assert isinstance(domains, list)
        assert all(isinstance(d, str) for d in domains)


class TestMergeDomains:
    """Test domain merging functionality."""

    def test_merge_two_domains(self):
        """Test merging two domains."""
        merged = merge_domains("ecommerce", "customer_service")

        assert merged is not None
        assert "ecommerce" in merged.name
        assert "customer_service" in merged.name

        # Should have topics from both
        assert "cart" in merged.topics  # From ecommerce
        assert "ticket" in merged.topics  # From customer_service

        # Should have categories from both
        assert "order" in merged.categories  # From ecommerce
        assert "support_request" in merged.categories  # From customer_service

    def test_merge_three_domains(self):
        """Test merging three domains."""
        merged = merge_domains("ecommerce", "customer_service", "saas")

        assert merged is not None
        assert "cart" in merged.topics  # From ecommerce
        assert "ticket" in merged.topics  # From customer_service
        assert "subscription" in merged.topics  # From saas

    def test_merge_empty_raises(self):
        """Test that merging no domains raises error."""
        with pytest.raises(ValueError) as exc_info:
            merge_domains()

        assert "at least one domain" in str(exc_info.value).lower()

    def test_merge_nonexistent_raises(self):
        """Test that merging non-existent domain raises error."""
        with pytest.raises(ValueError) as exc_info:
            merge_domains("ecommerce", "nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_merge_single_domain(self):
        """Test merging a single domain returns equivalent."""
        merged = merge_domains("ecommerce")

        assert merged.name == "ecommerce"
        assert merged.topics == ECOMMERCE_DOMAIN.topics


class TestCreateCustomDomain:
    """Test custom domain creation."""

    def test_create_from_scratch(self):
        """Test creating a custom domain from scratch."""
        domain = create_custom_domain(
            name="my_custom",
            description="My custom domain",
            topics=["custom_topic"],
            categories=["custom_category"],
        )

        assert domain.name == "my_custom"
        assert domain.description == "My custom domain"
        assert "custom_topic" in domain.topics
        assert "custom_category" in domain.categories

    def test_create_extending_base(self):
        """Test creating a custom domain extending a base."""
        domain = create_custom_domain(
            name="extended_ecommerce",
            base_domain="ecommerce",
            topics=["loyalty_program"],
            categories=["rewards"],
        )

        assert domain.name == "extended_ecommerce"
        assert "Extended from ecommerce" in domain.description

        # Should have base topics
        assert "cart" in domain.topics
        assert "checkout" in domain.topics

        # Should have new topics
        assert "loyalty_program" in domain.topics

        # Should have base categories
        assert "order" in domain.categories

        # Should have new categories
        assert "rewards" in domain.categories

        # Should keep base subcategories
        assert "order" in domain.subcategories

    def test_create_extending_with_description(self):
        """Test extending with custom description."""
        domain = create_custom_domain(
            name="my_finance",
            base_domain="finance",
            description="My custom finance domain",
        )

        assert domain.description == "My custom finance domain"

    def test_create_extending_with_custom_fields(self):
        """Test extending with additional custom fields."""
        domain = create_custom_domain(
            name="my_ecommerce",
            base_domain="ecommerce",
            custom_fields={"loyalty_points": {"type": "number"}},
        )

        # Should have base custom fields
        assert "order_id" in domain.custom_fields

        # Should have new custom fields
        assert "loyalty_points" in domain.custom_fields

    def test_create_extending_with_subcategories(self):
        """Test extending with additional subcategories."""
        domain = create_custom_domain(
            name="my_ecommerce",
            base_domain="ecommerce",
            subcategories={"loyalty": ["bronze", "silver", "gold"]},
        )

        # Should have base subcategories
        assert "order" in domain.subcategories

        # Should have new subcategories
        assert "loyalty" in domain.subcategories
        assert "gold" in domain.subcategories["loyalty"]

    def test_create_extending_nonexistent_raises(self):
        """Test extending non-existent base raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_custom_domain(
                name="extended",
                base_domain="nonexistent",
            )

        assert "not found" in str(exc_info.value).lower()


class TestDomainUseCases:
    """Test real-world domain usage scenarios."""

    def test_ecommerce_order_flow(self):
        """Test e-commerce domain covers order flow."""
        domain = ECOMMERCE_DOMAIN

        # Should cover order lifecycle
        order_statuses = domain.subcategories.get("order", [])
        assert "pending" in order_statuses
        assert "confirmed" in order_statuses
        assert "shipped" in order_statuses
        assert "delivered" in order_statuses

        # Should have order-related intents
        assert "checkout" in domain.intents
        assert "track_order" in domain.intents
        assert "initiate_return" in domain.intents

    def test_healthcare_patient_flow(self):
        """Test healthcare domain covers patient flow."""
        domain = HEALTHCARE_DOMAIN

        # Should cover clinical flow
        clinical_subs = domain.subcategories.get("clinical", [])
        assert "visit" in clinical_subs
        assert "procedure" in clinical_subs
        assert "test" in clinical_subs

        # Should have patient-related intents
        assert "schedule_appointment" in domain.intents
        assert "check_results" in domain.intents

        # Should have HIPAA-relevant custom fields
        assert "mrn" in domain.custom_fields
        assert "icd_code" in domain.custom_fields

    def test_saas_subscription_flow(self):
        """Test SaaS domain covers subscription flow."""
        domain = SAAS_DOMAIN

        # Should cover subscription tiers
        sub_tiers = domain.subcategories.get("subscription", [])
        assert "trial" in sub_tiers
        assert "basic" in sub_tiers
        assert "pro" in sub_tiers
        assert "enterprise" in sub_tiers

        # Should have subscription-related intents
        assert "upgrade_plan" in domain.intents
        assert "downgrade_plan" in domain.intents

    def test_merged_domain_for_retail_support(self):
        """Test merged domain for retail customer support."""
        # Retail support would need both e-commerce and customer service
        domain = merge_domains("ecommerce", "customer_service")

        # Should handle order inquiries
        assert "order" in domain.categories
        assert "track_order" in domain.intents

        # Should handle support tickets
        assert "ticket" in domain.topics
        assert "support_request" in domain.categories
        assert "escalate" in domain.intents

        # Should have entities from both
        assert "order" in domain.entity_types
        assert "customer" in domain.entity_types
        assert "ticket" in domain.entity_types

    def test_custom_fintech_domain(self):
        """Test creating a fintech domain."""
        domain = create_custom_domain(
            name="fintech",
            base_domain="finance",
            description="Fintech startup domain",
            topics=["crypto", "defi", "nft", "staking"],
            categories=["digital_assets"],
            subcategories={"digital_assets": ["token", "nft", "stablecoin"]},
            entity_types=["wallet", "smart_contract"],
            custom_fields={
                "wallet_address": {"type": "string"},
                "chain_id": {"type": "number"},
            },
        )

        # Should have base finance topics
        assert "account" in domain.topics
        assert "transaction" in domain.topics

        # Should have fintech-specific topics
        assert "crypto" in domain.topics
        assert "defi" in domain.topics
        assert "nft" in domain.topics

        # Should have base categories
        assert "banking" in domain.categories

        # Should have new categories
        assert "digital_assets" in domain.categories

        # Should have new custom fields
        assert "wallet_address" in domain.custom_fields
        assert "chain_id" in domain.custom_fields
