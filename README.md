<div align="center">

# Mindcore - SAGE Platform

### Structured Augmented Generation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 14+](https://img.shields.io/badge/postgresql-14+-336791.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/M-Alfaris/mindcore)

**A PostgreSQL-first memory platform for AI agents. Not RAG - SAGE.**

RAG searches documents. SAGE understands structure.

[Quick Start](#quick-start) | [Architecture](#sage-architecture) | [SVL Kernel](#svl-as-kernelcompiler) | [PostgreSQL-First](#postgresql-first-design)

---

</div>

## RAG vs SAGE

| RAG | SAGE |
|-----|------|
| Retrieves documents | Manages structured memory |
| Probabilistic scoring | Deterministic `sage_score()` |
| Vector similarity only | FTS + fuzzy + metadata weights |
| No schema enforcement | SVL kernel validates all data |
| Python-heavy | PostgreSQL-first |

**SAGE** is an alternative to RAG, not built on top of it.

---

## Quick Start

```bash
pip install mindcore
```

```python
from mindcore.v2.svl import SVLPipeline

# Initialize with PostgreSQL
pipeline = SVLPipeline(storage="postgresql://localhost/mindcore")

# Auto-configure external data sources
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    topics=["orders", "products", "users"],  # Maps to same-named tables
)

# Store a memory (SVL kernel validates & canonicalizes)
result = pipeline.store(
    llm_output={
        "content": "User prefers dark mode",
        "memory_type": "preference",
        "topics": ["settings", "ui"],
        "importance": 0.8,
    },
    user_id="user_123",
    session_id="session_abc",
)

# Query with SAGE scoring (scoring happens in PostgreSQL)
result = pipeline.query(
    query="user preferences",
    user_id="user_123",
    limit=10,
)

for memory in result.memories:
    print(f"[{memory['importance']:.2f}] {memory['content']}")
```

---

## SAGE Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR AI AGENTS                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SVL KERNEL (Compiler/Gate)                          │
│                                                                              │
│   All data flows through SVL for validation & canonicalization              │
│                                                                              │
│   STANDARD METADATA (Enforced)        USER METADATA (Assignable)            │
│   ───────────────────────────         ──────────────────────────            │
│   • message_type                      • topics[] (JSONB)                    │
│   • message_intent                    • categories[] (JSONB)                │
│   • memory_type                       • tags[] (JSONB)                      │
│   • importance (0-1)                  • entities[] (JSONB)                  │
│   • confidence (0-1)                  • metadata (JSONB)                    │
│   • sentiment                                                                │
│   • message_id, session_id                                                  │
│   • user_id, conversation_id                                                │
│   • created_at, expires_at                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │    SimpleFLR      │               │       CLST        │
        │   (Hot Path)      │               │   (Cold Path)     │
        │                   │               │                   │
        │ • O(1) LRU cache  │               │ • Complex scoring │
        │ • Deterministic   │               │ • Signal process  │
        │ • Metadata hints  │               │ • Compression     │
        └───────────────────┘               └───────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              POSTGRESQL                                      │
│                                                                              │
│   FUNCTIONS:                        FEATURES:                               │
│   • sage_score()                    • Full-text search (tsvector)           │
│   • search_memories_scored()        • Fuzzy matching (pg_trgm)              │
│   • find_relevant_sessions()        • JSONB for flexible metadata           │
│   • update_reinforcement()          • GIN indexes for fast queries          │
│                                                                              │
│   TRIGGERS:                         TABLES:                                 │
│   • Auto session aggregates         • sessions (parent)                     │
│   • Auto message_index              • memories (child)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SVL as Kernel/Compiler

SVL (Shared Vocabulary Layer) acts as the **kernel** - all data must pass through it:

```text
LLM Output ──► SVL Gate ──► Validated Memory ──► PostgreSQL
                  │
                  ├── Canonicalizes metadata
                  ├── Enforces schema
                  ├── Validates values
                  └── Rejects invalid data
```

### Standard vs User Metadata

| Standard (System Enforced) | User (Assignable) |
|---------------------------|-------------------|
| `message_type` (question, answer, instruction...) | `topics[]` |
| `message_intent` (inform, request, confirm...) | `categories[]` |
| `memory_type` (episodic, semantic, preference...) | `tags[]` |
| `importance` (0-1) | `entities[]` |
| `confidence` (0-1) | `metadata{}` |
| `sentiment` (positive, negative, neutral) | |
| `session_id`, `user_id`, `conversation_id` | |

```python
# SVL validates and canonicalizes before storage
result = pipeline.store(
    llm_output={
        # Standard (SVL enforces these)
        "content": "User wants to cancel order",
        "memory_type": "episodic",           # Must be valid type
        "message_type": "request",           # Must be valid type
        "message_intent": "request_action",  # Must be valid intent
        "importance": 0.8,                   # 0-1 range enforced
        "sentiment": "negative",             # Must be valid value

        # User-assignable (flexible)
        "topics": ["orders", "cancellation"],
        "categories": ["support"],
        "entities": [{"type": "order", "value": "ORD-12345"}],
    },
    user_id="user_123",
    session_id="session_abc",
)
```

---

## PostgreSQL-First Design

Core logic lives in PostgreSQL, not Python:

### SAGE Scoring Function

```sql
-- Deterministic scoring in PostgreSQL
CREATE FUNCTION sage_score(
    p_search_rank REAL,      -- ts_rank result
    p_recency_hours REAL,    -- Hours since created
    p_reinforcement REAL,    -- -1 to 1
    p_importance REAL,       -- 0 to 1
    p_confidence REAL,       -- 0 to 1
    p_access_count INTEGER,  -- Popularity
    p_topic_match_count INTEGER,
    p_total_topics INTEGER
) RETURNS REAL AS $$
    -- Weighted combination (deterministic)
    RETURN (
        0.30 * p_search_rank +
        0.20 * (1.0 / (1.0 + p_recency_hours / 24.0)) +
        0.15 * ((p_reinforcement + 1.0) / 2.0) +
        0.15 * p_importance +
        0.10 * p_confidence +
        0.05 * LEAST(1.0, LN(1 + p_access_count) / LN(100)) +
        0.05 * (p_topic_match_count::REAL / p_total_topics::REAL)
    );
$$ LANGUAGE plpgsql IMMUTABLE;
```

### Search with Scoring

```python
# Scoring happens in PostgreSQL, not Python
results = storage.search_scored(
    user_id="user_123",
    query="order shipping",
    topics=["orders", "shipping"],
    min_importance=0.5,
    limit=20,
)

for memory, score in results:
    print(f"[{score:.3f}] {memory.content}")
```

### Session Aggregates

Sessions are pre-aggregated for fast hierarchical retrieval:

```sql
-- Sessions table with weighted metadata
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,

    -- Weighted aggregates (auto-updated via trigger)
    topic_weights JSONB,       -- {"orders": 0.85, "shipping": 0.6}
    category_weights JSONB,
    importance_avg REAL,
    dominant_topic TEXT,
    ...
);

-- Trigger auto-updates on memory insert
CREATE TRIGGER trg_memory_session_update
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_session_on_memory_insert();
```

```python
# Find relevant sessions first, then drill down
sessions = storage.find_relevant_sessions(
    user_id="user_123",
    topics=["orders"],
    min_importance=0.5,
    limit=10,
)
```

---

## The Three Protocols

### SimpleFLR (Fast Learning Recall) - Hot Path

Deterministic O(1) cache lookup:

```python
from mindcore.v2.flr import SimpleFLR

flr = SimpleFLR(storage, cache_size=1000)

# Fast cache lookup with metadata hints
result = flr.query(
    user_id="user_123",
    topics=["settings"],
    metadata_hints={"is_clst_needed": False},  # Skip cold path
    limit=10,
)

# Collect signals (processed by CLST later)
flr.collect_signal(
    memory_id="mem_123",
    signal_type="positive",
    signal_value=0.5,
    source="user_feedback",
)
```

### CLST (Cognitive Long-term Storage) - Cold Path

Complex scoring and signal processing:

```python
from mindcore.v2.clst import CLST

clst = CLST(storage, vocabulary=svl)

# Complex scoring (moved from FLR)
scored = clst.score_memories_complex(
    memories=memories,
    query="order status",
    attention_hints=["orders"],
)

# Process reinforcement signals
result = clst.process_signals(pending_signals)

# Compression
compression = clst.compress(
    user_id="user_123",
    older_than_days=30,
    strategy="merge",
)
```

### SVL (Shared Vocabulary Layer) - Kernel

Schema enforcement and data source mapping:

```python
from mindcore.v2.svl import SharedVocabularyLayer, SVLPipeline

svl = SharedVocabularyLayer(domains=["ecommerce"])
pipeline = SVLPipeline(storage=storage, vocabulary=svl)

# Auto-configure database sources
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    preset="ecommerce",  # orders, products, customers, cart, etc.
)

# Or explicit topics
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    topics=["orders", "products"],
    overrides={
        "orders": {"table": "customer_orders"},
    },
)

# Or auto-discover from schema
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    auto_discover=True,
)
```

---

## Database Schema

### Memories Table

```sql
CREATE TABLE memories (
    -- Identifiers
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT REFERENCES sessions(session_id),
    conversation_id TEXT,

    -- Content
    content TEXT NOT NULL,

    -- Standard metadata (SVL enforced)
    message_type TEXT NOT NULL DEFAULT 'general',
    message_intent TEXT DEFAULT 'inform',
    memory_type TEXT NOT NULL DEFAULT 'episodic',
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.8,
    sentiment TEXT DEFAULT 'neutral',

    -- User metadata (flexible JSONB)
    topics JSONB DEFAULT '[]'::jsonb,
    categories JSONB DEFAULT '[]'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,
    entities JSONB DEFAULT '[]'::jsonb,

    -- Reinforcement
    reinforcement_score REAL DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,

    -- Temporal
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    -- Full-text search (auto-generated)
    search_vector tsvector GENERATED ALWAYS AS (...) STORED
);

-- Indexes
CREATE INDEX idx_memories_search ON memories USING GIN(search_vector);
CREATE INDEX idx_memories_topics ON memories USING GIN(topics jsonb_path_ops);
CREATE INDEX idx_memories_content_trgm ON memories USING GIN(content gin_trgm_ops);
```

### Sessions Table

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,

    -- Aggregated metadata (auto-updated via trigger)
    topic_weights JSONB DEFAULT '{}'::jsonb,
    category_weights JSONB DEFAULT '{}'::jsonb,
    importance_avg REAL DEFAULT 0.0,
    dominant_topic TEXT,

    -- Temporal
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Enterprise Features

```python
from mindcore.enterprise import (
    MindcoreMetrics,    # OpenTelemetry metrics
    MindcoreTracer,     # Distributed tracing
    RateLimiter,        # Per-user rate limiting
    AuditLogger,        # Security audit logs
    FieldEncryptor,     # Encryption at rest
)
```

See [Enterprise Documentation](docs/enterprise.md) for details.

---

## Project Structure

```text
mindcore/
├── v2/
│   ├── flr/
│   │   ├── recall.py           # Legacy FLR
│   │   └── simple_recall.py    # SimpleFLR (deterministic)
│   │
│   ├── clst/
│   │   ├── storage.py          # CLST with score_memories_complex()
│   │   ├── signals.py          # Signal history persistence
│   │   └── session_segmentation.py  # Topic shift detection
│   │
│   ├── svl/
│   │   ├── layer.py            # SharedVocabularyLayer
│   │   ├── gate.py             # SVL Gate (kernel)
│   │   ├── pipeline.py         # SVLPipeline (orchestrator)
│   │   ├── defaults.py         # Auto-configuration defaults
│   │   └── sources.py          # External data sources
│   │
│   └── storage/
│       ├── schema.sql          # PostgreSQL schema
│       ├── functions.sql       # sage_score(), triggers
│       └── postgres.py         # PostgresStorage
│
└── enterprise/                 # Enterprise features
```

---

## Testing

```bash
pytest mindcore/tests/ -v
pytest mindcore/tests/ --cov=mindcore
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**RAG searches. SAGE understands structure.**

```bash
pip install mindcore
```

[Quick Start](#quick-start) | [Architecture](#sage-architecture) | [SVL Kernel](#svl-as-kernelcompiler)

---

Made with care by the Mindcore team

</div>
