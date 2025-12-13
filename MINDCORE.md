# Mindcore Documentation

> Ground truth documentation for Mindcore - a memory layer for AI agents.
> Last updated: 2025-12-13

## Quick Reference

| Component | Status | Location | Purpose |
|-----------|--------|----------|---------|
| `mindcore/v2/` | **ACTIVE** | Primary | Modern memory layer with FLR/CLST protocols |
| `mindcore/context_lake/` | **ACTIVE** | Plugin | Unified context aggregation |
| `mindcore/agents/` | MIXED | Legacy+Active | Agent implementations |
| `mindcore/core/` | LEGACY | Deprecated | Old core infrastructure |
| `mindcore/vectorstores/` | LEGACY | Deprecated | Old vector store adapters |
| `mindcore/api/` | LEGACY | Deprecated | Old REST API |

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
│                      Mindcore v2                                │
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

## 3. Version Status

### v2 (ACTIVE) - `mindcore/v2/`

The current, actively developed version. Use this for all new projects.

```python
from mindcore.v2 import Mindcore

memory = Mindcore(storage="postgresql://localhost/mindcore")
memory.store("User prefers dark mode", "preference", "user123", topics=["settings"])
result = memory.recall("user preferences", "user123")
```

### v1/Legacy (DEPRECATED) - `mindcore/core/`, `mindcore/agents/`, etc.

Original implementation with complex agent pipelines. **Do not use for new projects.**

---

## 4. Feature Status Matrix

### Active Features (v2)

| Feature | Module | Description |
|---------|--------|-------------|
| FLR Protocol | `v2/flr/` | Fast retrieval, caching, scoring |
| CLST Protocol | `v2/clst/` | Long-term storage, compression, sync |
| Vocabulary Schema | `v2/vocabulary/` | Versioned metadata with JSON Schema export |
| Memory Extraction | `v2/extraction/` | Parse structured LLM output |
| Access Control | `v2/access/` | Multi-agent permissions |
| PostgreSQL Storage | `v2/storage/postgres.py` | Production backend with FTS |
| SQLite Storage | `v2/storage/sqlite.py` | Development backend with FTS5 |
| MCP Server | `v2/server/mcp.py` | Model Context Protocol integration |
| REST API | `v2/server/rest.py` | FastAPI REST endpoints |

### Active Features (Plugins)

| Feature | Module | Description |
|---------|--------|-------------|
| Context Lake | `context_lake/lake.py` | Unified context aggregation |
| API Listener | `context_lake/api_listener.py` | External API context ingestion |
| Knowledge Base | `context_lake/knowledge_base.py` | Static knowledge storage |

### Active Agents

| Agent | Module | Description |
|-------|--------|-------------|
| DeterministicContextAgent | `agents/deterministic_context_agent.py` | 2-call context retrieval (no randomness) |

### Legacy/Deprecated Features

| Feature | Module | Status | Replacement |
|---------|--------|--------|-------------|
| SmartContextAgent | `agents/smart_context_agent.py` | DEPRECATED | DeterministicContextAgent |
| EnrichmentAgent | `agents/enrichment_agent.py` | DEPRECATED | Structured output extraction |
| SummarizationAgent | `agents/summarization_agent.py` | DEPRECATED | CLST compression |
| RetrievalQueryAgent | `agents/retrieval_query_agent.py` | DEPRECATED | FLR.query() |
| ContextAssemblerAgent | `agents/context_assembler_agent.py` | DEPRECATED | FLR.update_context() |
| TrivialDetector | `agents/trivial_detector.py` | DEPRECATED | Not needed |
| Old Vocabulary | `core/vocabulary.py` | DEPRECATED | v2/vocabulary/schema.py |
| Old Access Control | `core/access_control.py` | DEPRECATED | v2/access/permissions.py |
| CacheManager | `core/cache_manager.py` | DEPRECATED | FLR hot cache |
| Celery Workers | `workers/` | DEPRECATED | Not needed for v2 |
| Vector Stores | `vectorstores/` | DEPRECATED | FTS (vector optional) |
| Old REST API | `api/` | DEPRECATED | v2/server/rest.py |
| Observability | `observability/` | AVAILABLE | Use if needed |

### Experimental/Future

| Feature | Status | Notes |
|---------|--------|-------|
| pgvector Support | PLANNED | Optional semantic search |
| Embedding Generation | PLANNED | External function hook |
| Cross-Agent Sync | PARTIAL | CLST.sync() implemented |

---

## 5. v2 Core Components

### 5.1 Mindcore Class

**Location:** `mindcore/v2/mindcore.py`

Main entry point for all operations.

```python
from mindcore.v2 import Mindcore

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

### 5.2 FLR Protocol

**Location:** `mindcore/v2/flr/recall.py`

Fast Learning Recall - hot path for inference-time memory access.

```python
from mindcore.v2.flr import FLR, Memory, RecallResult

# Query with scoring
result: RecallResult = flr.query(
    query="order status",
    user_id="user123",
    attention_hints=["orders"],
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

### 5.3 CLST Protocol

**Location:** `mindcore/v2/clst/storage.py`

Cognitive Long-term Storage Transfer - cold path for persistent storage.

```python
from mindcore.v2.clst import CLST, CompressionStrategy

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

### 5.4 Vocabulary Schema

**Location:** `mindcore/v2/vocabulary/schema.py`

Versioned vocabulary with JSON Schema export for LLM structured output.

```python
from mindcore.v2.vocabulary import VocabularySchema, DEFAULT_VOCABULARY

# Create custom vocabulary
vocab = VocabularySchema(
    version="1.0.0",
    topics=["billing", "orders", "support"],
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

### 5.5 Memory Extraction

**Location:** `mindcore/v2/extraction/extractor.py`

Parses memories from LLM structured output. **Fails hard on errors.**

```python
from mindcore.v2.extraction import MemoryExtractor

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

### 5.6 Access Control

**Location:** `mindcore/v2/access/permissions.py`

Multi-agent access control with permission levels.

```python
from mindcore.v2.access import AccessController, Permission

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

---

## 6. Storage Backends

### PostgreSQL (Production)

**Location:** `mindcore/v2/storage/postgres.py`

```python
from mindcore.v2 import Mindcore, PostgresStorage

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

**Schema:**
```sql
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    topics JSONB,
    categories JSONB,
    search_vector TSVECTOR,  -- Auto-updated
    ...
);

CREATE INDEX idx_memories_search ON memories USING GIN(search_vector);
CREATE INDEX idx_memories_topics ON memories USING GIN(topics);
```

### SQLite (Development)

**Location:** `mindcore/v2/storage/sqlite.py`

```python
from mindcore.v2 import Mindcore

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

## 7. API Reference

### MCP Server

**Location:** `mindcore/v2/server/mcp.py`

Model Context Protocol server for native LLM tool integration.

```python
from mindcore.v2 import Mindcore

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
from mindcore.v2 import Mindcore

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

## 8. Context Lake (Plugin)

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

---

## 9. Testing

### Run v2 Tests

```bash
# All v2 tests
pytest mindcore/v2/tests/ -v

# Specific test
pytest mindcore/v2/tests/test_mindcore_v2.py::TestMindcoreBasics -v
```

### Test Coverage

```bash
pytest mindcore/v2/tests/ --cov=mindcore/v2 --cov-report=html
```

---

## 10. Configuration

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
from mindcore.v2 import Mindcore, VocabularySchema

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

## 11. Migration Guide

### From v1 to v2

1. **Replace imports:**
   ```python
   # Old
   from mindcore.core import MindcoreClient
   from mindcore.agents import SmartContextAgent

   # New
   from mindcore.v2 import Mindcore
   ```

2. **Replace agents with direct calls:**
   ```python
   # Old
   agent = SmartContextAgent(client)
   context = agent.get_context(query, user_id)

   # New
   result = memory.recall(query, user_id)
   ```

3. **Replace auto-extraction with structured output:**
   ```python
   # Old
   memories = agent.extract_memories(messages)

   # New - LLM outputs structured JSON
   memories = memory.extract_from_response(llm_response, user_id)
   ```

4. **Update storage:**
   ```python
   # Old
   client = MindcoreClient(db_path="mindcore.db")

   # New
   memory = Mindcore(storage="sqlite:///mindcore.db")
   # or
   memory = Mindcore(storage="postgresql://...")
   ```

---

## 12. File Structure

```
mindcore/
├── v2/                          # ACTIVE - Modern memory layer
│   ├── mindcore.py              # Main class
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
│   ├── tests/                   # v2 tests
│   │   └── test_mindcore_v2.py
│   └── __init__.py
│
├── context_lake/                # ACTIVE - Context aggregation plugin
│   ├── lake.py                  # ContextLake
│   ├── knowledge_base.py        # Static knowledge
│   ├── api_listener.py          # External API ingestion
│   └── __init__.py
│
├── agents/                      # MIXED - Some active, most deprecated
│   ├── deterministic_context_agent.py  # ACTIVE - 2-call context
│   ├── smart_context_agent.py          # DEPRECATED
│   ├── enrichment_agent.py             # DEPRECATED
│   └── ...
│
├── observability/               # AVAILABLE - Use if needed
│   ├── observer.py              # Metrics collection
│   ├── metrics.py               # Metric definitions
│   ├── alerts.py                # Alert rules
│   └── quality.py               # Quality scoring
│
├── core/                        # DEPRECATED - Old infrastructure
├── api/                         # DEPRECATED - Old REST API
├── vectorstores/                # DEPRECATED - Old vector adapters
├── workers/                     # DEPRECATED - Celery workers
├── llm/                         # DEPRECATED - LLM providers
├── integrations/                # DEPRECATED - LangChain/LlamaIndex
├── connectors/                  # DEPRECATED - External connectors
└── utils/                       # MIXED - Some utilities still useful
```

---

## 13. Common Patterns

### Pattern: LLM with Memory

```python
from mindcore.v2 import Mindcore

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
from mindcore.v2 import Mindcore

memory = Mindcore(storage="postgresql://...", enable_multi_agent=True)

# Register agents
memory.register_agent("support", "Support Bot", teams=["customer_service"])
memory.register_agent("sales", "Sales Bot", teams=["customer_service"])

# Support agent stores memory with team access
memory.store(
    content="Customer interested in premium plan",
    memory_type="entity",
    user_id="user123",
    agent_id="support",
    access_level="team",  # Visible to sales too
)

# Sales agent can read it
result = memory.recall("customer interests", user_id="user123", agent_id="sales")
```

---

## 14. Troubleshooting

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
from mindcore.v2.vocabulary import DEFAULT_VOCABULARY
print(DEFAULT_VOCABULARY.topics)  # See valid topics
```

### PostgreSQL Connection Issues

**Problem:** `psycopg.OperationalError`

**Solution:** Ensure psycopg v3 is installed and connection string is correct.
```bash
pip install "psycopg[binary]>=3.0"
```

---

## 15. Roadmap

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
