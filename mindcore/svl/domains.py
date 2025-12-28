"""SVL Domain Mappings - Domain-specific vocabulary extensions.

This module provides pre-defined domain vocabularies that extend the base ontology
for specific industries and use cases. Each domain defines:
- Specialized topics
- Domain-specific categories
- Subcategories for fine-grained classification
- Custom metadata fields

Domains can be combined and extended for hybrid use cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainVocabulary:
    """Domain-specific vocabulary extension.

    Example:
        ecommerce = DomainVocabulary(
            name="ecommerce",
            topics=["cart", "checkout", "inventory"],
            categories=["order", "return", "payment"],
            subcategories={"order": ["pending", "shipped", "delivered"]},
        )
    """

    name: str
    description: str = ""

    # Core vocabulary
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    subcategories: dict[str, list[str]] = field(default_factory=dict)

    # Domain-specific entity types
    entity_types: list[str] = field(default_factory=list)

    # Custom intents for this domain
    intents: list[str] = field(default_factory=list)

    # Relationship types between entities
    relationship_types: list[str] = field(default_factory=list)

    # Custom metadata schema
    custom_fields: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_all_subcategories(self) -> list[str]:
        """Get flattened list of all subcategories."""
        result = []
        for subs in self.subcategories.values():
            result.extend(subs)
        return list(set(result))

    def merge_with(self, other: DomainVocabulary) -> DomainVocabulary:
        """Merge with another domain vocabulary."""
        return DomainVocabulary(
            name=f"{self.name}+{other.name}",
            description=f"Merged: {self.description} + {other.description}",
            topics=list(set(self.topics + other.topics)),
            categories=list(set(self.categories + other.categories)),
            subcategories={**self.subcategories, **other.subcategories},
            entity_types=list(set(self.entity_types + other.entity_types)),
            intents=list(set(self.intents + other.intents)),
            relationship_types=list(set(self.relationship_types + other.relationship_types)),
            custom_fields={**self.custom_fields, **other.custom_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "topics": self.topics,
            "categories": self.categories,
            "subcategories": self.subcategories,
            "entity_types": self.entity_types,
            "intents": self.intents,
            "relationship_types": self.relationship_types,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainVocabulary:
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# Pre-defined Domain Vocabularies
# =============================================================================

CUSTOMER_SERVICE_DOMAIN = DomainVocabulary(
    name="customer_service",
    description="Customer support and service domain vocabulary",
    topics=[
        "ticket",
        "escalation",
        "resolution",
        "sla",
        "satisfaction",
        "feedback",
        "complaint",
        "inquiry",
        "refund",
        "replacement",
        "warranty",
        "support_hours",
        "agent_handoff",
        "self_service",
        "knowledge_base",
    ],
    categories=[
        "support_request",
        "billing_issue",
        "technical_issue",
        "account_issue",
        "feedback",
        "complaint",
        "inquiry",
    ],
    subcategories={
        "support_request": ["new", "in_progress", "pending_customer", "resolved", "closed"],
        "billing_issue": ["overcharge", "refund", "payment_failed", "subscription"],
        "technical_issue": ["bug", "outage", "performance", "integration"],
        "account_issue": ["access", "password", "verification", "settings"],
    },
    entity_types=["ticket", "agent", "customer", "product", "case"],
    intents=[
        "open_ticket",
        "check_status",
        "escalate",
        "request_refund",
        "report_bug",
        "request_feature",
        "cancel_subscription",
    ],
    relationship_types=[
        "assigned_to",
        "escalated_from",
        "related_to",
        "duplicate_of",
    ],
)

ECOMMERCE_DOMAIN = DomainVocabulary(
    name="ecommerce",
    description="E-commerce and retail domain vocabulary",
    topics=[
        "cart",
        "checkout",
        "inventory",
        "wishlist",
        "price",
        "discount",
        "coupon",
        "promotion",
        "shipping",
        "delivery",
        "tracking",
        "return",
        "review",
        "rating",
        "recommendation",
        "search",
    ],
    categories=[
        "order",
        "payment",
        "shipping",
        "return",
        "product",
        "promotion",
        "review",
    ],
    subcategories={
        "order": ["pending", "confirmed", "processing", "shipped", "delivered", "cancelled"],
        "payment": ["pending", "completed", "failed", "refunded"],
        "shipping": ["standard", "express", "overnight", "international"],
        "return": ["requested", "approved", "shipped", "received", "refunded"],
    },
    entity_types=["order", "product", "customer", "cart", "address", "payment_method"],
    intents=[
        "add_to_cart",
        "remove_from_cart",
        "checkout",
        "track_order",
        "initiate_return",
        "apply_coupon",
        "check_availability",
    ],
    relationship_types=[
        "purchased",
        "reviewed",
        "wishlisted",
        "recommended_for",
    ],
    custom_fields={
        "order_id": {"type": "string", "description": "Order identifier"},
        "sku": {"type": "string", "description": "Product SKU"},
        "quantity": {"type": "number", "description": "Quantity"},
        "currency": {"type": "string", "description": "Currency code"},
    },
)

HEALTHCARE_DOMAIN = DomainVocabulary(
    name="healthcare",
    description="Healthcare and medical domain vocabulary",
    topics=[
        "appointment",
        "prescription",
        "diagnosis",
        "treatment",
        "symptom",
        "medication",
        "provider",
        "insurance",
        "lab_result",
        "imaging",
        "referral",
        "follow_up",
        "vaccination",
        "wellness",
        "chronic_condition",
    ],
    categories=[
        "clinical",
        "administrative",
        "insurance",
        "pharmacy",
        "wellness",
    ],
    subcategories={
        "clinical": ["visit", "procedure", "test", "surgery", "therapy"],
        "administrative": ["scheduling", "registration", "billing", "records"],
        "insurance": ["coverage", "claim", "authorization", "denial"],
        "pharmacy": ["prescription", "refill", "interaction", "generic"],
    },
    entity_types=[
        "patient",
        "provider",
        "medication",
        "condition",
        "appointment",
        "facility",
        "insurance_plan",
    ],
    intents=[
        "schedule_appointment",
        "request_prescription",
        "check_results",
        "verify_coverage",
        "find_provider",
        "refill_medication",
    ],
    relationship_types=[
        "treats",
        "prescribed_by",
        "diagnosed_with",
        "referred_to",
    ],
    custom_fields={
        "mrn": {"type": "string", "description": "Medical record number"},
        "icd_code": {"type": "string", "description": "ICD diagnosis code"},
        "ndc_code": {"type": "string", "description": "NDC medication code"},
    },
)

FINANCE_DOMAIN = DomainVocabulary(
    name="finance",
    description="Financial services and banking domain vocabulary",
    topics=[
        "account",
        "transaction",
        "transfer",
        "balance",
        "statement",
        "payment",
        "loan",
        "mortgage",
        "investment",
        "portfolio",
        "trading",
        "dividend",
        "credit",
        "debit",
        "interest",
        "fee",
    ],
    categories=[
        "banking",
        "investment",
        "lending",
        "insurance",
        "payment",
    ],
    subcategories={
        "banking": ["checking", "savings", "cd", "money_market"],
        "investment": ["stocks", "bonds", "mutual_funds", "etf", "crypto"],
        "lending": ["personal", "auto", "mortgage", "credit_line"],
        "payment": ["bill_pay", "wire", "ach", "check", "card"],
    },
    entity_types=[
        "account",
        "transaction",
        "customer",
        "security",
        "loan",
        "card",
        "beneficiary",
    ],
    intents=[
        "check_balance",
        "transfer_funds",
        "pay_bill",
        "open_account",
        "apply_loan",
        "dispute_transaction",
        "set_alert",
    ],
    relationship_types=[
        "owns",
        "authorized_on",
        "linked_to",
        "beneficiary_of",
    ],
    custom_fields={
        "account_number": {"type": "string", "description": "Account number"},
        "routing_number": {"type": "string", "description": "Routing number"},
        "ticker": {"type": "string", "description": "Stock ticker symbol"},
    },
)

SAAS_DOMAIN = DomainVocabulary(
    name="saas",
    description="SaaS and software product domain vocabulary",
    topics=[
        "subscription",
        "plan",
        "feature",
        "usage",
        "api",
        "integration",
        "webhook",
        "sdk",
        "dashboard",
        "analytics",
        "report",
        "export",
        "team",
        "workspace",
        "permission",
        "role",
    ],
    categories=[
        "subscription",
        "feature",
        "integration",
        "support",
        "account",
    ],
    subcategories={
        "subscription": ["trial", "basic", "pro", "enterprise", "custom"],
        "feature": ["core", "advanced", "beta", "deprecated"],
        "integration": ["native", "third_party", "custom", "webhook"],
        "support": ["documentation", "community", "standard", "priority"],
    },
    entity_types=[
        "user",
        "team",
        "workspace",
        "subscription",
        "api_key",
        "integration",
    ],
    intents=[
        "upgrade_plan",
        "downgrade_plan",
        "add_users",
        "enable_feature",
        "generate_api_key",
        "configure_integration",
        "export_data",
    ],
    relationship_types=[
        "member_of",
        "admin_of",
        "owns",
        "has_access_to",
    ],
    custom_fields={
        "plan_id": {"type": "string", "description": "Subscription plan ID"},
        "seats": {"type": "number", "description": "Number of seats"},
        "api_version": {"type": "string", "description": "API version"},
    },
)

HR_DOMAIN = DomainVocabulary(
    name="hr",
    description="Human resources and employee management domain vocabulary",
    topics=[
        "employee",
        "applicant",
        "recruitment",
        "onboarding",
        "payroll",
        "benefits",
        "pto",
        "leave",
        "performance",
        "review",
        "goal",
        "training",
        "compliance",
        "policy",
        "handbook",
        "offboarding",
    ],
    categories=[
        "recruitment",
        "employee_data",
        "payroll",
        "benefits",
        "performance",
        "compliance",
        "training",
    ],
    subcategories={
        "recruitment": ["sourcing", "screening", "interview", "offer", "hired", "rejected"],
        "employee_data": ["personal", "employment", "compensation", "emergency"],
        "leave": ["sick", "vacation", "parental", "bereavement", "unpaid"],
        "performance": ["self_assessment", "manager_review", "peer_feedback", "goal_setting"],
    },
    entity_types=[
        "employee",
        "applicant",
        "position",
        "department",
        "manager",
        "team",
        "document",
    ],
    intents=[
        "apply_job",
        "request_pto",
        "submit_timesheet",
        "view_payslip",
        "enroll_benefits",
        "update_info",
        "submit_review",
    ],
    relationship_types=[
        "reports_to",
        "manages",
        "member_of",
        "applied_for",
    ],
)

EDUCATION_DOMAIN = DomainVocabulary(
    name="education",
    description="Education and e-learning domain vocabulary",
    topics=[
        "course",
        "lesson",
        "module",
        "assignment",
        "quiz",
        "exam",
        "grade",
        "certificate",
        "enrollment",
        "progress",
        "instructor",
        "student",
        "curriculum",
        "schedule",
        "material",
        "discussion",
    ],
    categories=[
        "course",
        "assessment",
        "enrollment",
        "progress",
        "administration",
    ],
    subcategories={
        "course": ["self_paced", "instructor_led", "live", "recorded"],
        "assessment": ["quiz", "exam", "assignment", "project", "peer_review"],
        "progress": ["not_started", "in_progress", "completed", "certified"],
        "enrollment": ["open", "waitlist", "enrolled", "dropped", "completed"],
    },
    entity_types=[
        "student",
        "instructor",
        "course",
        "lesson",
        "assessment",
        "certificate",
    ],
    intents=[
        "enroll_course",
        "start_lesson",
        "submit_assignment",
        "take_quiz",
        "view_grades",
        "request_certificate",
        "ask_question",
    ],
    relationship_types=[
        "enrolled_in",
        "teaches",
        "prerequisite_of",
        "assigned_to",
    ],
)

# Registry of all built-in domains
DOMAIN_REGISTRY: dict[str, DomainVocabulary] = {
    "customer_service": CUSTOMER_SERVICE_DOMAIN,
    "ecommerce": ECOMMERCE_DOMAIN,
    "healthcare": HEALTHCARE_DOMAIN,
    "finance": FINANCE_DOMAIN,
    "saas": SAAS_DOMAIN,
    "hr": HR_DOMAIN,
    "education": EDUCATION_DOMAIN,
}


def get_domain(name: str) -> DomainVocabulary | None:
    """Get a domain vocabulary by name."""
    return DOMAIN_REGISTRY.get(name)


def list_domains() -> list[str]:
    """List all available domain names."""
    return list(DOMAIN_REGISTRY.keys())


def merge_domains(*domain_names: str) -> DomainVocabulary:
    """Merge multiple domains into one.

    Args:
        domain_names: Names of domains to merge

    Returns:
        Merged DomainVocabulary

    Raises:
        ValueError: If a domain name is not found
    """
    if not domain_names:
        raise ValueError("At least one domain name required")

    domains = []
    for name in domain_names:
        domain = get_domain(name)
        if domain is None:
            raise ValueError(f"Domain not found: {name}")
        domains.append(domain)

    result = domains[0]
    for domain in domains[1:]:
        result = result.merge_with(domain)

    return result


def create_custom_domain(
    name: str,
    base_domain: str | None = None,
    **kwargs: Any,
) -> DomainVocabulary:
    """Create a custom domain, optionally extending a base domain.

    Args:
        name: Name for the custom domain
        base_domain: Optional base domain to extend
        **kwargs: DomainVocabulary fields to set/override

    Returns:
        New DomainVocabulary
    """
    if base_domain:
        base = get_domain(base_domain)
        if base is None:
            raise ValueError(f"Base domain not found: {base_domain}")

        # Start with base and override
        data = base.to_dict()
        data["name"] = name
        data["description"] = kwargs.get("description", f"Extended from {base_domain}")

        # Merge lists instead of replacing
        for list_field in ["topics", "categories", "entity_types", "intents", "relationship_types"]:
            if list_field in kwargs:
                data[list_field] = list(set(data.get(list_field, []) + kwargs[list_field]))
                del kwargs[list_field]

        # Merge dicts
        for dict_field in ["subcategories", "custom_fields"]:
            if dict_field in kwargs:
                data[dict_field] = {**data.get(dict_field, {}), **kwargs[dict_field]}
                del kwargs[dict_field]

        # Override remaining
        data.update(kwargs)
        return DomainVocabulary.from_dict(data)

    return DomainVocabulary(name=name, **kwargs)
