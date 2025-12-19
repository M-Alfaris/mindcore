"""Shared test fixtures and example data for Mindcore tests.

This module provides:
- Sample memories for different use cases
- Sample vocabulary schemas
- Sample agent configurations
- Reusable pytest fixtures
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest

from mindcore.v2 import Mindcore
from mindcore.v2.flr import Memory
from mindcore.v2.vocabulary import VocabularySchema, FieldSchema
from mindcore.v2.storage import SQLiteStorage
from mindcore.v2.access import AccessController


# =============================================================================
# Sample Memories
# =============================================================================

SAMPLE_MEMORIES = {
    "preference_email": Memory(
        memory_id="mem_pref_email",
        content="User strongly prefers email communication over phone calls",
        memory_type="preference",
        user_id="user_alice",
        agent_id="support_bot",
        topics=["communication", "settings"],
        categories=["support"],
        sentiment="neutral",
        importance=0.8,
        entities=["email"],
        access_level="private",
    ),
    "preference_dark_mode": Memory(
        memory_id="mem_pref_dark",
        content="User has dark mode enabled and prefers dark themes",
        memory_type="preference",
        user_id="user_alice",
        agent_id="support_bot",
        topics=["settings", "ui"],
        categories=["product"],
        sentiment="positive",
        importance=0.6,
        entities=["dark_mode"],
        access_level="private",
    ),
    "billing_issue": Memory(
        memory_id="mem_billing_1",
        content="Customer reported billing discrepancy on invoice #12345, overcharged by $50",
        memory_type="episodic",
        user_id="user_bob",
        agent_id="billing_bot",
        topics=["billing", "issue"],
        categories=["support", "urgent"],
        sentiment="negative",
        importance=0.9,
        entities=["invoice_12345", "$50"],
        access_level="team",
    ),
    "product_feedback": Memory(
        memory_id="mem_feedback_1",
        content="User loves the new dashboard feature, especially the analytics widgets",
        memory_type="semantic",
        user_id="user_carol",
        agent_id="feedback_bot",
        topics=["product", "feedback"],
        categories=["feedback"],
        sentiment="positive",
        importance=0.7,
        entities=["dashboard", "analytics_widgets"],
        access_level="shared",
    ),
    "order_tracking": Memory(
        memory_id="mem_order_1",
        content="Order #98765 shipped via FedEx, tracking: 123456789, expected delivery 2024-12-25",
        memory_type="temporal",
        user_id="user_david",
        agent_id="orders_bot",
        topics=["order", "shipping"],
        categories=["support"],
        sentiment="neutral",
        importance=0.8,
        entities=["order_98765", "fedex", "tracking_123456789"],
        access_level="private",
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    ),
    "api_integration": Memory(
        memory_id="mem_api_1",
        content="Customer integrated with Stripe API using webhook endpoint https://api.example.com/webhooks",
        memory_type="procedural",
        user_id="user_eve",
        agent_id="tech_bot",
        topics=["api", "integration"],
        categories=["technical"],
        sentiment="neutral",
        importance=0.85,
        entities=["stripe", "webhook", "api.example.com"],
        access_level="team",
    ),
    "customer_relationship": Memory(
        memory_id="mem_rel_1",
        content="User is the primary account holder and has 3 team members under their organization",
        memory_type="relationship",
        user_id="user_frank",
        agent_id="account_bot",
        topics=["account", "organization"],
        categories=["account"],
        sentiment="neutral",
        importance=0.7,
        entities=["primary_holder", "team_3"],
        access_level="private",
    ),
    "working_session": Memory(
        memory_id="mem_work_1",
        content="Currently discussing refund process for order #54321",
        memory_type="working",
        user_id="user_grace",
        agent_id="support_bot",
        topics=["billing", "refund"],
        categories=["support"],
        sentiment="neutral",
        importance=0.5,
        entities=["order_54321", "refund"],
        access_level="private",
    ),
}


def get_sample_memory(key: str) -> Memory:
    """Get a copy of a sample memory by key."""
    mem = SAMPLE_MEMORIES[key]
    return Memory(
        memory_id=mem.memory_id,
        content=mem.content,
        memory_type=mem.memory_type,
        user_id=mem.user_id,
        agent_id=mem.agent_id,
        topics=list(mem.topics),
        categories=list(mem.categories),
        sentiment=mem.sentiment,
        importance=mem.importance,
        entities=list(mem.entities),
        access_level=mem.access_level,
        expires_at=mem.expires_at,
    )


def get_all_sample_memories() -> list[Memory]:
    """Get copies of all sample memories."""
    return [get_sample_memory(key) for key in SAMPLE_MEMORIES]


# =============================================================================
# Sample Vocabularies
# =============================================================================

ECOMMERCE_VOCABULARY = VocabularySchema(
    version="1.0.0",
    topics=[
        # Customer service
        "billing", "payment", "refund", "subscription",
        "order", "shipping", "delivery", "tracking",
        # Product
        "product", "feature", "bug", "feedback",
        # Account
        "account", "login", "password", "settings", "profile",
        # Communication
        "communication", "notification", "email",
        # Technical
        "api", "integration", "webhook", "documentation",
        # General
        "greeting", "farewell", "help", "issue", "urgent",
        # UI
        "ui", "theme", "dashboard",
        # Organization
        "organization", "team",
    ],
    categories=[
        "support", "billing", "technical", "account",
        "product", "feedback", "general", "urgent",
    ],
    intents=[
        "ask_question", "request_action", "provide_info",
        "express_opinion", "complaint", "greeting",
        "confirmation", "clarification",
    ],
    description="E-commerce customer support vocabulary",
)


HEALTHCARE_VOCABULARY = VocabularySchema(
    version="1.0.0",
    topics=[
        # Medical
        "appointment", "prescription", "diagnosis", "treatment",
        "medication", "symptoms", "lab_results", "referral",
        # Administrative
        "billing", "insurance", "records", "forms",
        # Communication
        "doctor", "nurse", "specialist", "pharmacy",
    ],
    categories=[
        "medical", "administrative", "urgent", "follow_up",
    ],
    custom_fields=[
        FieldSchema(
            name="patient_id",
            field_type="string",
            required=True,
            description="Patient identifier",
        ),
        FieldSchema(
            name="confidentiality_level",
            field_type="enum",
            enum_values=["standard", "sensitive", "restricted"],
            default="standard",
            description="Medical confidentiality level",
        ),
    ],
    description="Healthcare provider vocabulary",
)


MINIMAL_VOCABULARY = VocabularySchema(
    version="1.0.0",
    topics=["general"],
    categories=["general"],
    description="Minimal vocabulary for basic testing",
)


# =============================================================================
# Sample Agent Configurations
# =============================================================================

SAMPLE_AGENTS = [
    {
        "agent_id": "support_bot",
        "name": "Customer Support Bot",
        "description": "Handles general customer inquiries",
        "teams": ["support_team", "customer_facing"],
        "capabilities": ["customer_support", "ticket_handling"],
    },
    {
        "agent_id": "billing_bot",
        "name": "Billing Assistant",
        "description": "Handles billing and payment issues",
        "teams": ["billing_team", "customer_facing"],
        "capabilities": ["billing", "refunds"],
    },
    {
        "agent_id": "tech_bot",
        "name": "Technical Support Bot",
        "description": "Handles technical and API issues",
        "teams": ["tech_team"],
        "capabilities": ["technical_support", "api_support"],
    },
    {
        "agent_id": "feedback_bot",
        "name": "Feedback Collector",
        "description": "Collects and processes user feedback",
        "teams": ["product_team"],
        "capabilities": ["feedback_collection"],
    },
    {
        "agent_id": "orders_bot",
        "name": "Order Management Bot",
        "description": "Tracks orders and shipping",
        "teams": ["operations_team", "customer_facing"],
        "capabilities": ["order_tracking", "shipping"],
    },
    {
        "agent_id": "account_bot",
        "name": "Account Manager Bot",
        "description": "Manages user accounts",
        "teams": ["account_team"],
        "capabilities": ["account_management"],
    },
]


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sqlite_storage(temp_db_path: str) -> Generator[SQLiteStorage, None, None]:
    """Create a SQLite storage instance."""
    storage = SQLiteStorage(temp_db_path)
    yield storage
    storage.close()


@pytest.fixture
def ecommerce_vocab() -> VocabularySchema:
    """Get e-commerce vocabulary."""
    return ECOMMERCE_VOCABULARY


@pytest.fixture
def healthcare_vocab() -> VocabularySchema:
    """Get healthcare vocabulary."""
    return HEALTHCARE_VOCABULARY


@pytest.fixture
def minimal_vocab() -> VocabularySchema:
    """Get minimal vocabulary."""
    return MINIMAL_VOCABULARY


@pytest.fixture
def access_controller_with_agents() -> AccessController:
    """Create access controller with sample agents."""
    ac = AccessController()
    for agent_config in SAMPLE_AGENTS:
        ac.register_agent(**agent_config)
    return ac


@pytest.fixture
def mindcore_ecommerce(temp_db_path: str) -> Generator[Mindcore, None, None]:
    """Create Mindcore with e-commerce vocabulary."""
    mc = Mindcore(
        storage=f"sqlite:///{temp_db_path}",
        vocabulary=ECOMMERCE_VOCABULARY,
        enable_multi_agent=True,
    )
    # Register sample agents
    for agent_config in SAMPLE_AGENTS:
        try:
            mc.register_agent(**agent_config)
        except ValueError:
            pass  # Agent already exists
    yield mc
    mc.close()


@pytest.fixture
def mindcore_with_data(temp_db_path: str) -> Generator[Mindcore, None, None]:
    """Create Mindcore with sample data preloaded."""
    mc = Mindcore(
        storage=f"sqlite:///{temp_db_path}",
        vocabulary=ECOMMERCE_VOCABULARY,
        enable_multi_agent=True,
    )

    # Register agents
    for agent_config in SAMPLE_AGENTS:
        try:
            mc.register_agent(**agent_config)
        except ValueError:
            pass

    # Store sample memories
    for key, memory in SAMPLE_MEMORIES.items():
        mc.store(
            content=memory.content,
            memory_type=memory.memory_type,
            user_id=memory.user_id,
            agent_id=memory.agent_id,
            topics=memory.topics,
            categories=memory.categories,
            importance=memory.importance,
            entities=memory.entities,
            access_level=memory.access_level,
        )

    yield mc
    mc.close()


# =============================================================================
# Test Data Helpers
# =============================================================================

def create_test_memories_for_user(
    user_id: str,
    count: int = 5,
    agent_id: str = "test_agent",
) -> list[Memory]:
    """Create a set of test memories for a user."""
    memories = []
    memory_types = ["episodic", "semantic", "preference", "procedural"]
    topics_list = [["billing"], ["support"], ["product"], ["account"]]

    for i in range(count):
        memory = Memory(
            memory_id=f"mem_{user_id}_{i}",
            content=f"Test memory {i} for user {user_id}",
            memory_type=memory_types[i % len(memory_types)],
            user_id=user_id,
            agent_id=agent_id,
            topics=topics_list[i % len(topics_list)],
            importance=0.5 + (i * 0.1),
        )
        memories.append(memory)

    return memories


def create_llm_response_with_memories(
    response_text: str,
    memories: list[dict],
) -> dict:
    """Create a simulated LLM structured response."""
    return {
        "response": response_text,
        "memories_to_store": memories,
    }


# Sample LLM responses
SAMPLE_LLM_RESPONSES = {
    "customer_preference": create_llm_response_with_memories(
        response_text="I've noted your preference for email communication.",
        memories=[
            {
                "content": "User prefers email communication",
                "memory_type": "preference",
                "topics": ["communication"],
                "importance": 0.7,
            }
        ],
    ),
    "billing_interaction": create_llm_response_with_memories(
        response_text="I'll help you resolve this billing issue.",
        memories=[
            {
                "content": "User reported billing discrepancy",
                "memory_type": "episodic",
                "topics": ["billing"],
                "sentiment": "negative",
                "importance": 0.9,
            }
        ],
    ),
    "no_memories": create_llm_response_with_memories(
        response_text="Hello! How can I help you today?",
        memories=[],
    ),
    "multiple_memories": create_llm_response_with_memories(
        response_text="I've updated your account settings.",
        memories=[
            {
                "content": "User updated notification preferences",
                "memory_type": "preference",
                "topics": ["settings"],
                "importance": 0.6,
            },
            {
                "content": "User enabled two-factor authentication",
                "memory_type": "procedural",
                "topics": ["account", "settings"],
                "importance": 0.8,
            },
        ],
    ),
}
