# Mindcore Quick Start Guide

Get started with Mindcore in minutes. Choose your path based on your use case.

## Table of Contents

- [Simple Mode (Single Agent)](#simple-mode-single-agent)
- [Multi-Agent Mode](#multi-agent-mode)
- [Adding Vector Search](#adding-vector-search)
- [Full Setup with All Features](#full-setup-with-all-features)
- [Migration Path](#migration-path)

---

## Simple Mode (Single Agent)

The fastest way to get started. Zero configuration, works out of the box.

### Installation

```bash
pip install mindcore
```

### Basic Usage

```python
from mindcore import KnowledgeStore

# Create store - just works!
store = KnowledgeStore()

# Store a message
item = store.add_message(
    user_id="user123",
    text="How do I reset my password?",
    role="user"
)

# Get context for a query
context = store.get_context(
    user_id="user123",
    query="password reset"
)

print(context["recent_messages"])
```

### With Conversation Threading

```python
from mindcore import KnowledgeStore

store = KnowledgeStore()

# Store messages in a thread
thread_id = "support-ticket-123"

store.add_message(
    user_id="user123",
    text="I can't log in to my account",
    role="user",
    thread_id=thread_id
)

store.add_message(
    user_id="user123",
    text="I'll help you reset your password. What email is your account under?",
    role="assistant",
    thread_id=thread_id
)

# Get context for the thread
context = store.get_context(
    user_id="user123",
    thread_id=thread_id,
    query="account access"
)
```

### Storing Documents

```python
# Add knowledge base articles
store.add_document(
    content="To reset your password, go to Settings > Security > Reset Password...",
    title="Password Reset Guide",
    source="https://docs.example.com/password-reset"
)

# Documents are searchable via semantic search (if enabled)
```

---

## Multi-Agent Mode

For systems with multiple AI agents that need privacy and knowledge sharing.

### Enable Multi-Agent Mode

```python
from mindcore import KnowledgeStore

# Enable multi-agent mode
store = KnowledgeStore(
    multi_agent=True,
    organization_id="my-company"
)

# Register agents
support_agent = store.register_agent(
    name="Support Agent",
    groups=["support-team", "customer-facing"]
)

billing_agent = store.register_agent(
    name="Billing Agent",
    groups=["billing-team", "customer-facing"]
)

print(f"Support Agent ID: {support_agent.agent_id}")
print(f"API Key: {support_agent.api_key}")  # Store securely!
```

### Store Messages with Visibility

```python
# Private message (only Support Agent can see)
store.add_message(
    user_id="user123",
    text="Internal note: Customer seems frustrated",
    role="assistant",
    agent_id=support_agent.agent_id,
    visibility="private"
)

# Shared message (all customer-facing agents can see)
store.add_message(
    user_id="user123",
    text="Customer is asking about refund policy",
    role="assistant",
    agent_id=support_agent.agent_id,
    visibility="shared",
    sharing_groups=["customer-facing"]
)

# Public message (all agents in organization can see)
store.add_message(
    user_id="user123",
    text="General product inquiry",
    role="user",
    visibility="public"
)
```

### Share Knowledge Between Agents

```python
# Share a specific item with another agent
store.share_with_agent(
    item_id="msg-abc123",
    target_agent_id=billing_agent.agent_id,
    sharing_agent_id=support_agent.agent_id,
    can_reshare=False  # Target can't share further
)

# Share with a group
store.share_with_group(
    item_id="doc-xyz789",
    group_name="support-team",
    sharing_agent_id=support_agent.agent_id
)
```

### Access Control in Queries

```python
# Get context respecting access control
context = store.get_context(
    user_id="user123",
    query="refund status",
    agent_id=billing_agent.agent_id,  # Only sees what billing_agent can access
    include_shared=True,
    include_public=True
)
```

---

## Adding Vector Search

Enable semantic search for finding relevant content by meaning.

### With OpenAI Embeddings

```python
from mindcore import KnowledgeStore

store = KnowledgeStore(
    enable_vector_search=True,
    embedding_provider="openai",  # Requires OPENAI_API_KEY
    vector_store_type="memory"    # In-memory for development
)

# Add documents
store.add_document(
    content="Our refund policy allows returns within 30 days...",
    title="Refund Policy"
)

# Semantic search
results = store.search(
    query="Can I return something I bought last week?",
    k=5
)
```

### With Local Embeddings (No API Key)

```python
from mindcore import KnowledgeStore

store = KnowledgeStore(
    enable_vector_search=True,
    embedding_provider="sentence_transformers",  # Runs locally!
    embedding_config={"model_name": "all-MiniLM-L6-v2"},
    vector_store_type="memory"
)
```

### With Chroma (Persistent)

```python
store = KnowledgeStore(
    enable_vector_search=True,
    embedding_provider="openai",
    vector_store_type="chroma",
    vector_store_config={
        "collection_name": "my_knowledge",
        "persist_directory": "./data/chroma"
    }
)
```

---

## Full Setup with All Features

Production-ready configuration with all features enabled.

```python
from mindcore import KnowledgeStore

store = KnowledgeStore(
    # Database
    database_path="mindcore.db",     # SQLite for simplicity
    # use_postgresql=True,           # Or PostgreSQL for production
    # postgresql_url="postgresql://user:pass@localhost/mindcore",

    # Multi-agent
    multi_agent=True,
    organization_id="my-company",

    # Vector search
    enable_vector_search=True,
    vector_store_type="chroma",
    vector_store_config={
        "collection_name": "knowledge",
        "persist_directory": "./data/vectors"
    },
    embedding_provider="openai",

    # Caching
    enable_cache=True,
    cache_type="disk",  # Survives restarts

    # Enrichment
    enable_enrichment=True,  # Auto-extract topics, sentiment, etc.
    llm_provider="auto"      # Uses llama.cpp if available, else OpenAI
)

# Register agents
support = store.register_agent(name="Support", groups=["support"])
sales = store.register_agent(name="Sales", groups=["sales"])

# Store enriched messages
store.add_message(
    user_id="user123",
    text="I need help with billing for order #12345",
    role="user",
    agent_id=support.agent_id,
    visibility="shared",
    sharing_groups=["support", "billing"]
)

# Semantic search with access control
results = store.search(
    query="order billing issues",
    agent_id=support.agent_id,
    k=10
)

# Health check
status = store.health_check()
print(status)
# {'database': True, 'cache': True, 'vector_store': True, 'enrichment': True, 'access_control': True}
```

---

## Migration Path

### From Simple to Multi-Agent

```python
# Step 1: Start simple
store = KnowledgeStore()

# ... use for a while ...

# Step 2: When you need multi-agent, create new store
store = KnowledgeStore(
    multi_agent=True,
    organization_id="my-company",
    database_path="mindcore.db"  # Same database!
)

# Existing data is preserved, new data gets agent/visibility fields
```

### From Memory to Persistent Vector Store

```python
# Step 1: Start with in-memory
store = KnowledgeStore(
    enable_vector_search=True,
    vector_store_type="memory"
)

# Step 2: Move to Chroma for persistence
store = KnowledgeStore(
    enable_vector_search=True,
    vector_store_type="chroma",
    vector_store_config={
        "persist_directory": "./data/chroma"
    }
)

# Re-index existing documents if needed
```

---

## Context Manager Support

```python
# Automatic cleanup with context manager
with KnowledgeStore() as store:
    store.add_message(
        user_id="user123",
        text="Hello!",
        role="user"
    )
# Resources automatically cleaned up
```

---

## Properties and Status

```python
store = KnowledgeStore(multi_agent=True, enable_vector_search=True)

# Check mode
print(store.mode)           # StoreMode.MULTI_AGENT
print(store.is_multi_agent) # True
print(store.has_vector_search) # True

# Default agent
print(store.default_agent.name)  # "Default Agent"

# Health check
status = store.health_check()
```

---

## Advanced: Casbin RBAC/ABAC

For enterprise scenarios requiring fine-grained access control, Mindcore integrates with [Casbin](https://casbin.org/) - a powerful authorization library.

```bash
pip install mindcore[rbac]
```

```python
from mindcore.core import get_casbin_access_control

# Get Casbin-powered access control
CasbinACM = get_casbin_access_control()

# Create with built-in RBAC model
acm = CasbinACM.with_rbac()

# Register agents and assign roles
agent, api_key = acm.register_agent(
    agent_id="support-agent",
    name="Support Agent",
    owner_id="acme-corp"
)

# Assign roles
acm.add_role_for_agent("support-agent", "support")
acm.add_role_for_agent("admin-agent", "admin")

# Add policies
acm.add_policy("support", "customer-data", "read")
acm.add_policy("admin", "customer-data", "write")

# Check access
acm.enforce("support-agent", "customer-data", "read")  # True
acm.enforce("support-agent", "customer-data", "write")  # False

# Multi-tenant RBAC
acm = CasbinACM.with_rbac_domains()
acm.add_role_for_agent("agent-1", "admin", domain="tenant-a")
acm.enforce("agent-1", "tenant-a", "resource", "write")  # True
acm.enforce("agent-1", "tenant-b", "resource", "write")  # False
```

---

## Optional Dependencies

Install only what you need:

```bash
# Core (messages, cache, database)
pip install mindcore

# With local LLM support
pip install mindcore[llama]

# With vector stores
pip install mindcore[vectors]      # Chroma + SentenceTransformers
pip install mindcore[chroma]       # Just Chroma
pip install mindcore[pinecone]     # Pinecone
pip install mindcore[pgvector]     # PostgreSQL pgvector

# With Casbin RBAC
pip install mindcore[rbac]

# With async support
pip install mindcore[async]

# Everything
pip install mindcore[all]
```

---

## What's Next?

- **[Vector Stores Guide](../config/README.md#vector-stores-configuration)** - Deep dive into vector databases
- **[External Connectors](../config/README.md#configuring-external-connectors)** - Connect to orders, billing, etc.
- **[API Server](./dashboard.md)** - Run Mindcore as a service
- **[Casbin Docs](https://casbin.org/docs/overview)** - Advanced RBAC/ABAC patterns
