# Mindcore Documentation

> Ground truth documentation for Mindcore - a memory layer for AI agents.
> Last updated: 2025-12-13

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| `mindcore/v2/` | Primary | Modern memory layer with FLR/CLST protocols |
| `mindcore/v2/cross_agent/` | Primary | Multi-agent memory sharing and routing |
| `mindcore/context_lake/` | Plugin | Unified context aggregation |
| `mindcore/observability/` | Optional | Metrics, alerts, quality scoring |
| `mindcore/utils/` | Utilities | Logging |

---

## 1. Overview

Mindcore is a **memory layer** for AI agents. It provides:

- **Structured memory storage** with vocabulary-controlled metadata
- **Fast retrieval** via FLR (Fast Learning Recall) protocol
- **Long-term storage** via CLST (Cognitive Long-term Storage Transfer) protocol
- **Multi-agent support** with access control
- **MCP and REST APIs** for LLM integration

### Design Principles

1. **Structured Output Only** - LLMs produce memories via JSON schema, no fallbacks
2. **Fail Hard** - Validation errors crash, no silent failures
3. **Vocabulary Controlled** - All metadata follows versioned vocabulary
4. **Full-Text Search** - No vector DB required (optional enhancement)
5. **Multi-Backend** - PostgreSQL (production) / SQLite (development)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Agent / LLM                           │
│                   (Structured Output JSON)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Mindcore                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ MCP Server  │  │  REST API   │  │  Direct Python Import   │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
│         └────────────────┴─────────────────────┬┘              │
│                                                │               │
│  ┌─────────────────────────────────────────────▼─────────────┐ │
│  │                     Mindcore Class                        │ │
│  │  store() | recall() | search() | extract_from_response()  │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         ▼                  ▼                  ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │     FLR     │    │    CLST     │    │  Extractor  │       │
│  │ (Hot Path)  │    │ (Cold Path) │    │ (Parse LLM) │       │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘       │
│         │                  │                                  │
│         └────────┬─────────┘                                  │
│                  ▼                                            │
│  ┌───────────────────────────────────────────────────────────┐│
│  │                    Storage Backend                        ││
│  │         PostgreSQL (prod) | SQLite (dev)                  ││
│  └───────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation

```python
# Basic usage
from mindcore import Mindcore

memory = Mindcore(storage="sqlite:///dev.db")

# Production with PostgreSQL
memory = Mindcore(storage="postgresql://user:pass@localhost/mindcore")
```

---

## 4. Core Components

### 4.1 Mindcore Class

**Location:** `mindcore/v2/mindcore.py`

Main entry point for all operations.

```python
from mindcore import Mindcore

# Initialize
memory = Mindcore(
    storage="postgresql://user:pass@localhost/mindcore",  # or "sqlite:///dev.db"
    vocabulary=None,  # Uses DEFAULT_VOCABULARY
    enable_multi_agent=False,
)

# Store memory
memory_id = memory.store(
    content="User prefers dark mode",
    memory_type="preference",  # episodic|semantic|procedural|preference|entity|relationship|temporal|working
    user_id="user123",
    topics=["settings"],  # Must be in vocabulary
    importance=0.8,
)

# Recall memories
result = memory.recall(
    query="user preferences",
    user_id="user123",
    attention_hints=["settings"],  # Prioritize these topics
    limit=10,
)

# Search with filters
memories = memory.search(
    user_id="user123",
    topics=["settings"],
    memory_types=["preference"],
)

# Extract from LLM response
llm_response = {
    "response": "I'll remember that.",
    "memories_to_store": [
        {"content": "User likes email", "memory_type": "preference", "topics": ["settings"]}
    ]
}
memories = memory.extract_from_response(llm_response, user_id="user123")

# Get JSON schema for LLM
schema = memory.get_json_schema()
```

### 4.2 FLR Protocol

**Location:** `mindcore/v2/flr/recall.py`

Fast Learning Recall - hot path for inference-time memory access.

```python
from mindcore import FLR, Memory, RecallResult

# Query with scoring
result: RecallResult = flr.query(
    query="order status",
    user_id="user123",
    attention_hints=["order"],
    limit=10,
)

# Reinforce useful memories
flr.reinforce(memory_id, signal=+1.0)  # Positive: helpful
flr.reinforce(memory_id, signal=-0.5)  # Negative: not helpful

# Promote working memory to long-term
flr.promote(memory_id)
```

**RecallResult fields:**
- `memories`: List of Memory objects
- `scores`: Relevance scores (0.0-1.0)
- `sources`: Where memories came from ("cache", "storage")
- `attention_focus`: Top topics to focus on
- `suggested_memory_types`: Relevant memory types

### 4.3 CLST Protocol

**Location:** `mindcore/v2/clst/storage.py`

Cognitive Long-term Storage Transfer - cold path for persistent storage.

```python
from mindcore import CLST, CompressionStrategy

# Store memory
memory_id = clst.store(memory, validate=True)

# Compress old memories
result = clst.compress(
    user_id="user123",
    older_than_days=30,
    strategy=CompressionStrategy.SUMMARIZE,
)

# Sync between agents
result = clst.sync(
    source_agent="agent_a",
    target_agent="agent_b",
    user_id="user123",
)

# Transfer memories (export/import)
manifest = clst.transfer(memories, destination="backup")
```

### 4.4 Vocabulary Schema

**Location:** `mindcore/v2/vocabulary/schema.py`

Versioned vocabulary with JSON Schema export for LLM structured output.

```python
from mindcore import VocabularySchema, DEFAULT_VOCABULARY

# Create custom vocabulary
vocab = VocabularySchema(
    version="1.0.0",
    topics=["billing", "order", "product"],
    categories=["urgent", "normal"],
    memory_types=["episodic", "semantic", "preference"],
)

# Get JSON schema for LLM
schema = vocab.to_json_schema()

# Validate memory
is_valid, errors = vocab.validate(memory_dict)

# Migrate between versions
migrated = vocab.migrate_memory(old_memory, from_version="0.9.0")

# Export for TypeScript/Pydantic
ts_code = vocab.to_typescript()
pydantic_code = vocab.to_pydantic()
```

**Default vocabulary topics:**
```
greeting, farewell, thanks, help, feedback,
issue, bug, error, problem, complaint,
billing, payment, refund, subscription,
feature, product, service, pricing,
api, integration, setup, documentation,
account, login, password, settings, profile,
order, shipping, delivery, tracking
```

**Memory types:**
```
episodic     - Events, conversations, interactions
semantic     - Facts, knowledge, learned information
procedural   - Workflows, how-to, processes
preference   - User preferences, settings
entity       - People, places, things
relationship - Connections between entities
temporal     - Time-bound info (auto-expires)
working      - Current session context (cleared)
```

### 4.5 Memory Extraction

**Location:** `mindcore/v2/extraction/extractor.py`

Parses memories from LLM structured output. **Fails hard on errors.**

```python
from mindcore import MemoryExtractor

extractor = MemoryExtractor(vocabulary=vocab)

# Extract from LLM response
result = extractor.extract(
    output={
        "response": "...",
        "memories_to_store": [
            {"content": "...", "memory_type": "...", "topics": [...]}
        ]
    },
    user_id="user123",
)

# Raises on errors:
# - TypeError: Wrong output format
# - KeyError: Missing required fields
# - ValueError: Validation failure
```

### 4.6 Access Control

**Location:** `mindcore/v2/access/permissions.py`

Multi-agent access control with permission levels.

```python
from mindcore import AccessController, Permission

controller = AccessController()

# Register agent
controller.register_agent(
    agent_id="support_bot",
    name="Support Agent",
    teams=["support", "customer_service"],
    permissions={Permission.READ, Permission.WRITE},
)

# Check access
decision = controller.can_access(
    agent_id="support_bot",
    memory_access_level="team",  # private|team|shared|global
    memory_agent_id="other_agent",
    permission=Permission.READ,
)

if decision.allowed:
    # Access granted
    pass
```

### 4.7 Cross-Agent Layer

**Location:** `mindcore/v2/cross_agent/`

Multi-agent memory sharing, synchronization, and intelligent query routing.

```python
from mindcore import CrossAgentLayer, SQLiteStorage, RoutingStrategy

# Initialize
storage = SQLiteStorage("mindcore.db")
layer = CrossAgentLayer(storage)

# Register agents
layer.register_agent(
    agent_id="support_bot",
    name="Support Agent",
    capabilities=["customer_support", "billing"],
    specializations=["refunds", "complaints"],
    teams=["customer_service"],
)

layer.register_agent(
    agent_id="sales_bot",
    name="Sales Agent",
    capabilities=["sales", "product_info"],
    teams=["customer_service"],
)

# Create team
layer.create_team(
    team_id="customer_service",
    name="Customer Service Team",
    shared_topics=["billing", "orders", "products"],
)

# Store memory with agent ownership
from mindcore import Memory
memory = Memory(
    memory_id="",
    content="Customer requested refund for order #123",
    memory_type="entity",
    user_id="user123",
)
layer.store_memory(memory, agent_id="support_bot", access_level="team")

# Query across agents with routing
result = layer.query(
    query="refund requests",
    user_id="user123",
    requesting_agent="sales_bot",
    strategy=RoutingStrategy.CAPABILITY_MATCH,
    attention_hints=["billing", "refunds"],
)

# Sync memories between agents
from mindcore.v2.cross_agent import SyncDirection
sync_result = layer.sync(
    source_agent="support_bot",
    target_agent="sales_bot",
    user_id="user123",
    direction=SyncDirection.ONE_WAY,
)
```

**Components:**

| Component | Purpose |
|-----------|---------|
| `AgentRegistry` | Agent/team registration and management |
| `CrossAgentMemory` | Memory sharing and synchronization |
| `AttentionRouter` | Query routing with scoring algorithms |
| `CrossAgentLayer` | Unified interface combining all above |

**Access Levels:**
```
private  - Only owning agent can access
team     - Agents in same team can access
shared   - Any agent for same user can access
global   - All agents can access (knowledge base)
```

**Routing Strategies:**
```
BROADCAST            - Query all agents
CAPABILITY_MATCH     - Route to agents with matching capabilities
SPECIALIZATION_MATCH - Route to specialized agents
TEAM_FIRST           - Prioritize team members
BEST_MATCH           - Use scoring to find best agent (default)
ROUND_ROBIN          - Distribute queries evenly
```

---

## 5. Storage Backends

### PostgreSQL (Production)

**Location:** `mindcore/v2/storage/postgres.py`

```python
from mindcore import Mindcore, PostgresStorage

# Via connection string
memory = Mindcore(storage="postgresql://user:pass@localhost:5432/mindcore")

# Via storage instance
storage = PostgresStorage(
    connection_string="postgresql://...",
    pool_size=10,
)
memory = Mindcore(storage=storage)
```

**Features:**
- Connection pooling (psycopg v3)
- Full-text search via `tsvector`
- JSONB for topics, categories, entities
- GIN indexes for array containment

### SQLite (Development)

**Location:** `mindcore/v2/storage/sqlite.py`

```python
from mindcore import Mindcore

# Development mode
memory = Mindcore(storage="sqlite:///dev.db")

# In-memory for testing
memory = Mindcore(storage="sqlite:///:memory:")
```

**Features:**
- Thread-safe with WAL mode
- FTS5 full-text search
- JSON arrays stored as TEXT

---

## 6. API Servers

### MCP Server

**Location:** `mindcore/v2/server/mcp.py`

Model Context Protocol server for native LLM tool integration.

```python
from mindcore import Mindcore

memory = Mindcore(storage="...")
mcp = memory.get_mcp_server()

# Get available tools
tools = mcp.get_tools()
# Returns: store_memory, search_memories, recall, reinforce

# Handle JSON-RPC request
response = mcp.handle_json_rpc({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "recall", "arguments": {...}},
    "id": 1
})
```

### REST API

**Location:** `mindcore/v2/server/rest.py`

FastAPI REST endpoints.

```python
from mindcore import Mindcore

memory = Mindcore(storage="...")
memory.serve_rest(host="0.0.0.0", port=8000)
```

**Endpoints:**
```
POST   /memories              - Store memory
GET    /memories/{id}         - Get memory by ID
DELETE /memories/{id}         - Delete memory
POST   /memories/search       - Search memories
POST   /recall                - FLR recall
POST   /extract               - Extract from LLM response
GET    /schema                - Get JSON schema
GET    /stats                 - Get statistics
GET    /health                - Health check
```

---

## 7. Plugins

### Context Lake

**Location:** `mindcore/context_lake/`

Unified context aggregation from multiple sources.

```python
from mindcore.context_lake import ContextLake

lake = ContextLake(mindcore=memory)

# Register sources
lake.register_knowledge_base("docs", path="/path/to/docs")
lake.register_api_listener("orders", url="https://api.example.com/orders")

# Get aggregated context
context = lake.get_context(
    query="order status",
    user_id="user123",
    sources=["memory", "docs", "orders"],
)
```

### Observability

**Location:** `mindcore/observability/`

Optional metrics, alerts, and quality scoring.

```python
from mindcore.observability import Observer, QualityScorer

observer = Observer()
observer.record_recall(query, result, latency_ms)

scorer = QualityScorer()
score = scorer.score_memory(memory)
```

---

## 8. Testing

### Run All Tests

```bash
# v2 tests
pytest mindcore/v2/tests/ -v

# All tests
pytest mindcore/tests/ -v
```

### Test Files

| Test | Purpose |
|------|---------|
| `v2/tests/test_mindcore_v2.py` | Core v2 functionality |
| `v2/tests/test_cross_agent.py` | Cross-agent memory layer |
| `tests/test_context_lake.py` | Context lake plugin |
| `tests/test_observability.py` | Observability tests |

---

## 9. Configuration

### Environment Variables

```bash
# PostgreSQL
MINDCORE_DATABASE_URL=postgresql://user:pass@localhost:5432/mindcore

# SQLite (dev)
MINDCORE_DATABASE_URL=sqlite:///mindcore.db

# Logging
MINDCORE_LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from mindcore import Mindcore, VocabularySchema

memory = Mindcore(
    storage="postgresql://...",
    vocabulary=VocabularySchema(
        version="1.0.0",
        topics=["custom", "topics"],
        categories=["custom", "categories"],
    ),
    enable_multi_agent=True,
)
```

---

## 10. File Structure

```
mindcore/
├── __init__.py                  # Main exports (v2 + utils)
├── py.typed                     # PEP 561 marker
│
├── v2/                          # Core memory layer
│   ├── mindcore.py              # Main Mindcore class
│   ├── flr/                     # Fast Learning Recall
│   │   ├── recall.py            # FLR protocol
│   │   └── __init__.py
│   ├── clst/                    # Cognitive Long-term Storage
│   │   ├── storage.py           # CLST protocol
│   │   └── __init__.py
│   ├── vocabulary/              # Versioned vocabulary
│   │   ├── schema.py            # VocabularySchema
│   │   └── __init__.py
│   ├── extraction/              # Memory extraction
│   │   ├── extractor.py         # MemoryExtractor
│   │   └── __init__.py
│   ├── access/                  # Multi-agent access control
│   │   ├── permissions.py       # AccessController
│   │   └── __init__.py
│   ├── storage/                 # Storage backends
│   │   ├── base.py              # BaseStorage interface
│   │   ├── postgres.py          # PostgreSQL backend
│   │   ├── sqlite.py            # SQLite backend
│   │   └── __init__.py
│   ├── server/                  # API servers
│   │   ├── mcp.py               # MCP server
│   │   ├── rest.py              # REST API
│   │   └── __init__.py
│   ├── cross_agent/             # Multi-agent memory layer
│   │   ├── registry.py          # Agent/Team registration
│   │   ├── sharing.py           # Memory sharing and sync
│   │   ├── routing.py           # Attention routing
│   │   ├── layer.py             # CrossAgentLayer (unified)
│   │   └── __init__.py
│   ├── tests/                   # v2 tests
│   │   ├── test_mindcore_v2.py  # Core v2 tests
│   │   └── test_cross_agent.py  # Cross-agent tests
│   └── __init__.py
│
├── context_lake/                # Context aggregation plugin
│   ├── lake.py                  # ContextLake
│   ├── knowledge_base.py        # Static knowledge
│   ├── api_listener.py          # External API ingestion
│   └── __init__.py
│
├── observability/               # Optional observability
│   ├── observer.py              # Metrics collection
│   ├── metrics.py               # Metric definitions
│   ├── alerts.py                # Alert rules
│   ├── quality.py               # Quality scoring
│   └── __init__.py
│
├── utils/                       # Utilities
│   ├── logger.py                # Logging
│   └── __init__.py
│
└── tests/                       # Integration tests
    ├── conftest.py              # Pytest fixtures
    ├── test_context_lake.py     # Context lake tests
    └── test_observability.py    # Observability tests
```

---

## 11. Common Patterns

### Pattern: LLM with Memory

```python
from mindcore import Mindcore
import json

memory = Mindcore(storage="postgresql://...")

# Get schema for LLM
schema = memory.get_json_schema()

# Include in system prompt
system_prompt = f"""
You are a helpful assistant with memory.

When you learn something important about the user, include it in your response
using this JSON structure:

{json.dumps(schema, indent=2)}
"""

# After LLM response
llm_response = call_llm(messages, response_format=schema)
memories = memory.extract_from_response(llm_response, user_id)

# Before next turn - recall relevant memories
context = memory.recall(user_query, user_id)
```

### Pattern: Multi-Agent Memory Sharing

```python
from mindcore import CrossAgentLayer, SQLiteStorage, Memory, RoutingStrategy

# Initialize cross-agent layer
storage = SQLiteStorage("mindcore.db")
layer = CrossAgentLayer(storage)

# Register agents with capabilities
layer.register_agent(
    agent_id="support",
    name="Support Bot",
    capabilities=["customer_support", "complaints"],
    teams=["customer_service"],
)
layer.register_agent(
    agent_id="sales",
    name="Sales Bot",
    capabilities=["sales", "upselling"],
    teams=["customer_service"],
)

# Create shared team
layer.create_team(
    team_id="customer_service",
    name="Customer Service",
    shared_topics=["customers", "orders"],
)

# Support agent stores memory with team access
memory = Memory(
    memory_id="",
    content="Customer interested in premium plan",
    memory_type="entity",
    user_id="user123",
)
layer.store_memory(memory, agent_id="support", access_level="team")

# Sales agent can query with intelligent routing
result = layer.query(
    query="customer interests",
    user_id="user123",
    requesting_agent="sales",
    strategy=RoutingStrategy.TEAM_FIRST,
)

# Get ranked agent suggestions for a topic
suggestions = layer.suggest_agents(attention_hints=["upselling"], limit=3)
```

---

## 12. Troubleshooting

### FTS Search Returns Empty

**Problem:** `recall()` returns no results even though memories exist.

**Cause:** FTS5/tsvector word matching requires actual word overlap.

**Solution:** Use queries with words that appear in stored content.
```python
# Stored: "User prefers dark mode"
memory.recall("dark mode", user_id)  # Works
memory.recall("preferences", user_id)  # May not work (no word overlap)
```

### Validation Errors

**Problem:** `ValueError: Memory validation failed`

**Cause:** Topics/categories not in vocabulary.

**Solution:** Use topics from `DEFAULT_VOCABULARY` or create custom vocabulary.
```python
from mindcore import DEFAULT_VOCABULARY
print(DEFAULT_VOCABULARY.topics)  # See valid topics
```

### PostgreSQL Connection Issues

**Problem:** `psycopg.OperationalError`

**Solution:** Ensure psycopg v3 is installed and connection string is correct.
```bash
pip install "psycopg[binary]>=3.0"
```

---

## 13. Roadmap

### Planned
- [ ] pgvector support for semantic search
- [ ] Embedding generation hook
- [ ] Memory importance decay over time
- [ ] Automatic compression scheduling

### Considering
- [ ] Redis cache layer
- [ ] Distributed FLR across instances
- [ ] GraphQL API
- [ ] Memory visualization dashboard

---

*This document is the source of truth for Mindcore. Update it when making architectural changes.*
