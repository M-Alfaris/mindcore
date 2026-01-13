# Mindcore Quick Start Guide

Get started with Mindcore in minutes.

## Table of Contents

- [Installation](#installation)
- [Option 1: SQLite (Zero Setup)](#option-1-sqlite-zero-setup)
- [Option 2: PostgreSQL (Production)](#option-2-postgresql-production)
- [Option 3: Full Pipeline with SVL](#option-3-full-pipeline-with-svl)
- [Auto-Configure External Sources](#auto-configure-external-sources)
- [Next Steps](#next-steps)

---

## Installation

```bash
pip install mindcore
```

Verify installation:

```bash
mindcore version
```

---

## Option 1: SQLite (Zero Setup)

Perfect for development and testing. No database setup required.

```python
from mindcore.storage import SQLiteStorage
from mindcore.flr import Memory

# Create storage (in-memory for testing)
storage = SQLiteStorage(":memory:")

# Or persistent file
# storage = SQLiteStorage("mindcore.db")

# Store a memory
memory = Memory(
    content="User prefers dark mode and brief responses",
    memory_type="preference",
    user_id="user_123",
    topics=["settings", "ui"],
    importance=0.8,
)

memory_id = storage.store(memory)
print(f"Stored: {memory_id}")

# Search memories
results = storage.search(
    user_id="user_123",
    query="user preferences",
    limit=5,
)

for mem in results:
    print(f"[{mem.importance:.1f}] {mem.content}")
```

---

## Option 2: PostgreSQL (Production)

PostgreSQL with Mindcore scoring functions for production workloads.

### Setup Database

```bash
# Using Docker (recommended)
cd examples/
docker-compose up -d

# Or use existing PostgreSQL and run:
mindcore init --postgres postgresql://user:pass@localhost/mindcore
```

### Connect and Query

```python
from mindcore.storage import PostgresStorage

# Connect
storage = PostgresStorage("postgresql://mindcore:mindcore@localhost/mindcore")

# Initialize schema (first time only)
storage.initialize_full_schema()

# Store a memory
from mindcore.flr import Memory

memory = Memory(
    content="Customer asked about order #12345",
    memory_type="episodic",
    user_id="user_456",
    session_id="session_abc",
    topics=["orders", "shipping"],
    importance=0.7,
)

memory_id = storage.store(memory)

# Mindcore scored search (scoring in PostgreSQL!)
results = storage.search_scored(
    user_id="user_456",
    query="order shipping",
    topics=["orders"],
    limit=10,
)

for memory, score in results:
    print(f"[{score:.3f}] {memory.content}")

# Find relevant sessions
sessions = storage.find_relevant_sessions(
    user_id="user_456",
    topics=["orders"],
    limit=5,
)
```

### Fuzzy Search (Typo Tolerance)

```python
# Handles typos via pg_trgm
results = storage.search_fuzzy(
    user_id="user_456",
    query="shiping",  # typo in "shipping"
    similarity_threshold=0.3,
    limit=10,
)

for memory, similarity, score in results:
    print(f"[sim={similarity:.2f}] {memory.content}")
```

---

## Option 3: Full Pipeline with SVL

Use the SVL kernel for metadata validation and the full Mindcore pipeline.

```python
from mindcore.storage import SQLiteStorage
from mindcore.svl import SharedVocabularyLayer, SVLPipeline

# 1. Create storage
storage = SQLiteStorage(":memory:")

# 2. Create SVL vocabulary (the kernel)
svl = SharedVocabularyLayer(domains=["customer_service"])

# 3. Create pipeline
pipeline = SVLPipeline(
    storage=storage,
    vocabulary=svl,
    use_simple_flr=True,  # Deterministic hot path
)

# 4. Store via pipeline (SVL validates metadata)
result = pipeline.store(
    llm_output={
        "content": "User wants to cancel subscription",
        "memory_type": "episodic",
        "topics": ["billing", "cancellation"],
        "importance": 0.8,
        "sentiment": "negative",
    },
    user_id="user_789",
    session_id="session_xyz",
)

print(f"Stored: {result.memory_id}")

# 5. Query via pipeline
query_result = pipeline.query(
    query="subscription cancellation",
    user_id="user_789",
    limit=5,
)

print(f"Found: {len(query_result.memories)} memories")
print(f"CLST needed: {query_result.clst_decision.needs_clst}")

# 6. Get stats
stats = pipeline.get_stats()
print(f"Hot path ratio: {stats['hot_path_ratio']:.1%}")
```

---

## Auto-Configure External Sources

Connect topics to database tables with zero configuration.

```python
from mindcore.svl import SVLPipeline

pipeline = SVLPipeline(storage=storage, vocabulary=svl)

# Simple: topic "orders" → table "orders"
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    topics=["orders", "products", "users"],
)

# Or use presets
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    preset="ecommerce",  # orders, products, customers, cart, etc.
)

# Or auto-discover tables
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    auto_discover=True,
)

# With customization
pipeline.auto_configure_database(
    "postgresql://localhost/mydb",
    topics=["orders"],
    overrides={
        "orders": {
            "table": "customer_orders",
            "query_template": "SELECT * FROM customer_orders WHERE user_id = :user_id",
        },
    },
)
```

---

## CLI Commands

```bash
# Initialize database
mindcore init                                           # SQLite (default)
mindcore init --postgres postgresql://localhost/mydb    # PostgreSQL

# Check database
mindcore check                                          # SQLite
mindcore check --postgres postgresql://localhost/mydb   # PostgreSQL

# Show version
mindcore version
```

---

## Next Steps

- **[README](../README.md)** - Full architecture documentation
- **[examples/quickstart.py](../examples/quickstart.py)** - Runnable examples
- **[examples/docker-compose.yml](../examples/docker-compose.yml)** - PostgreSQL setup

---

## Common Patterns

### Store LLM Response

```python
result = pipeline.store(
    llm_output={
        "content": response_text,
        "memory_type": "episodic",
        "topics": extracted_topics,
        "importance": calculated_importance,
    },
    user_id=user_id,
    session_id=session_id,
)
```

### Build Context for LLM

```python
query_result = pipeline.query(
    query=user_message,
    user_id=user_id,
    session_id=session_id,
    limit=10,
)

context = "\n".join([
    f"- {m.content}" for m in query_result.memories
])

llm_prompt = f"""
Context:
{context}

User: {user_message}
"""
```

### Reinforce Memory (Feedback Loop)

```python
# Positive feedback
storage.update_reinforcement(memory_id, signal=0.3)

# Negative feedback
storage.update_reinforcement(memory_id, signal=-0.2)
```
