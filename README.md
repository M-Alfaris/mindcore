<div align="center">

# Mindcore - The Memory Protocol for AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/M-Alfaris/mindcore)
[![Tests](https://img.shields.io/badge/tests-585%20passing-brightgreen.svg)](https://github.com/M-Alfaris/mindcore)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**A modular memory layer framework built on three foundational protocols: FLR, CLST, and SVL.**

Like MCP standardized tool connections, Mindcore standardizes AI agent memory.

[Quick Start](#quick-start) | [Protocols](#the-three-protocols) | [Enterprise](#enterprise-features) | [Architecture](#architecture)

---

</div>

## The Problem

Every team building AI agents faces the same challenges:

| Challenge | What Teams Do Today | The Pain |
|-----------|---------------------|----------|
| **Memory & Persistence** | Build custom storage, caching, retrieval | 2-4 weeks reinventing the wheel |
| **Multi-Agent Consistency** | Each agent has its own memory silo | Agents contradict each other |
| **Vocabulary Alignment** | Ad-hoc metadata, no schema | "Is it `topic` or `topics`?" |
| **Production Features** | DIY rate limiting, audit, encryption | Security vulnerabilities |

**Result**: Months of infrastructure work before you can focus on your actual product.

## The Solution

Mindcore provides **three foundational protocols** that standardize AI agent memory:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR AI AGENTS                                  │
│         (Support Bot, Sales Assistant, Internal Tools, etc.)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MINDCORE                                        │
│                        The Memory Protocol Stack                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FLR (Fast Learning Recall)                        │   │
│  │              Hot-path memory access for inference time               │   │
│  │     query() | reinforce() | context() | promote() | cache           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CLST (Cognitive Long-term Storage)                │   │
│  │              Cold-path persistence, compression, sync                │   │
│  │     store() | compress() | sync() | transfer() | migrate()          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SVL (Shared Vocabulary Layer)                     │   │
│  │              Unified semantic system with migrations                 │   │
│  │     validate() | map_source() | enrich() | migrate_memory()         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐                 ┌─────────────────┐
          │    PostgreSQL   │                 │     SQLite      │
          │   (Production)  │                 │  (Development)  │
          └─────────────────┘                 └─────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install mindcore
```

### Basic Usage

```python
from mindcore import Mindcore

# Initialize with SQLite (development)
memory = Mindcore(storage="sqlite:///dev.db")

# Or PostgreSQL (production)
# memory = Mindcore(storage="postgresql://localhost/mindcore")

# Store a memory
memory_id = memory.store(
    content="User prefers dark mode and brief responses",
    memory_type="preference",
    user_id="user_123",
    topics=["settings", "ui"],
    importance=0.8,
)

# Recall relevant memories
result = memory.recall(
    query="user preferences",
    user_id="user_123",
    limit=5,
)

for mem in result.memories:
    print(f"[{mem.relevance:.2f}] {mem.content}")

# Reinforce positive memories (learning signal)
memory.reinforce(memory_id, signal=0.5)

# Compress old memories (CLST)
compression = memory.compress(
    user_id="user_123",
    older_than_days=7,
    strategy="merge",  # or "summarize", "deduplicate"
)
print(f"Compressed {compression.original_count} → {compression.compressed_count}")
```

### Multi-Agent Example

```python
from mindcore import Mindcore, AccessLevel

memory = Mindcore(storage="sqlite:///shared.db")

# Support agent stores a memory (shared with team)
memory.store(
    content="Customer reported billing issue, refund processed",
    memory_type="episodic",
    user_id="customer_456",
    agent_id="support_agent",
    access_level=AccessLevel.TEAM,  # Visible to all agents
    topics=["billing", "refund"],
)

# Sales agent can see it
result = memory.recall(
    query="customer history",
    user_id="customer_456",
    agent_id="sales_agent",
    include_cross_agent=True,
)

# Sales agent now knows about the refund!
```

---

## The Three Protocols

### FLR (Fast Learning Recall)

The **hot-path** protocol for inference-time memory access.

```python
from mindcore import FLR, Memory, SQLiteStorage

storage = SQLiteStorage("memories.db")
flr = FLR(storage)

# Fast query with relevance scoring
result = flr.query(
    query="user preferences for notifications",
    user_id="user_123",
    attention_hints=["settings", "notifications"],
    memory_types=["preference", "semantic"],
    limit=10,
    min_score=0.3,
)

# Reinforcement learning signal
flr.reinforce(memory_id, signal=0.8)  # Positive feedback
flr.reinforce(memory_id, signal=-0.5)  # Negative feedback

# Context window management (per-session)
flr.update_context(
    session_id="session_abc",
    messages=[{"role": "user", "content": "Hello"}],
    attention_hints=["greeting"],
)

# Promote working memory to long-term
flr.promote(memory_id)
```

**Key Features:**

- LRU cache with TTL (1000 items, 300s default)
- 6-factor relevance scoring (similarity, topics, recency, reinforcement, importance, popularity)
- Bounded reinforcement scores (-1.0 to +1.0) with diminishing returns
- Per-session context windows

### CLST (Cognitive Long-term Storage Transfer)

The **cold-path** protocol for durable storage and memory management.

```python
from mindcore import CLST, CompressionStrategy, SyncDirection

clst = CLST(storage, vocabulary=svl)

# Store with vocabulary validation
memory_id = clst.store(memory, validate=True)

# Compression strategies
result = clst.compress(
    user_id="user_123",
    older_than_days=30,
    strategy=CompressionStrategy.MERGE,  # Combine similar memories
    min_memories=10,
)

# Cross-agent sync
sync_result = clst.sync(
    source_agent="support",
    target_agent="sales",
    user_id="user_123",
    direction=SyncDirection.BIDIRECTIONAL,
    conflict_resolution="source_wins",
)

# Transfer memories between instances
manifest = clst.transfer(
    memories=memories_to_transfer,
    destination="backup_instance",
)

# Vocabulary migration with rollback support
migration_result = clst.migrate(
    from_version="1.0.0",
    user_id="user_123",
    create_checkpoints=True,  # Enable rollback
)

if migration_result.can_rollback:
    clst.rollback_migration(migration_result)
```

**Compression Strategies:**

| Strategy | Description |
|----------|-------------|
| `DEDUPLICATE` | Remove duplicate content (MD5 hash) |
| `MERGE` | Combine memories with same topics |
| `SUMMARIZE` | LLM-based summarization (requires LLM) |
| `EXTRACT` | Extract key facts (requires LLM) |

### SVL (Shared Vocabulary Layer)

The **semantic foundation** that ensures consistent metadata across all agents.

```python
from mindcore import SharedVocabularyLayer, Migration, TableSource

# Create vocabulary with domain
svl = SharedVocabularyLayer(domains=["customer_service", "ecommerce"])

# Add custom vocabulary
svl.add_topics("product_feedback", "feature_request")
svl.add_categories("urgent", "normal", "low")
svl.add_custom_field(
    name="priority_score",
    field_type="number",
    required=False,
    description="0-100 priority score",
)

# Validate memories before storage
is_valid, errors = svl.validate_memory(memory_dict)
if not is_valid:
    print(f"Validation errors: {errors}")

# Map data sources to vocabulary terms
svl.map_source(
    term="orders",
    source=TableSource(
        connection_string="postgresql://localhost/orders",
        table="orders",
        query_template="SELECT * FROM orders WHERE user_id = :user_id",
    ),
)

# Fetch data for topics (auto-triggers on query)
results = svl.fetch_for_topics(
    topics=["orders", "billing"],
    context={"user_id": "user_123"},
)

# Define migrations between vocabulary versions
migration = Migration(
    from_version="1.0.0",
    to_version="2.0.0",
    renames={"category": "categories"},
    merges={"topic": {"sources": ["tag", "label"]}},
    defaults={"priority": "normal"},
)
svl.add_migration(migration)

# Migrate a memory with checkpoint (rollback support)
migrated, checkpoint = svl.migrate_memory(
    memory_dict,
    from_version="1.0.0",
    create_checkpoint=True,
)

# Rollback if needed
original = svl.rollback_memory(migrated, checkpoint)

# Generate schema for LLMs
json_schema = svl.get_json_schema()
typescript_types = svl.to_typescript()
pydantic_models = svl.to_pydantic()
```

**Built-in Domains:**

- `customer_service` - tickets, escalation, satisfaction
- `ecommerce` - cart, checkout, shipping, returns
- `healthcare` - appointments, diagnosis, medication
- `finance` - transactions, accounts, investments
- `saas` - subscriptions, features, onboarding
- `hr` - hiring, training, performance
- `education` - courses, assignments, grades

---

## Enterprise Features

Mindcore includes production-ready enterprise features:

```python
from mindcore.enterprise import (
    MindcoreMetrics,
    MindcoreTracer,
    RateLimiter,
    AuditLogger,
    FieldEncryptor,
)
```

### Observability (OpenTelemetry)

```python
from mindcore.enterprise import MindcoreMetrics, MindcoreTracer, ObservabilityConfig

config = ObservabilityConfig(
    service_name="my-ai-agent",
    otlp_endpoint="http://localhost:4317",
)

metrics = MindcoreMetrics(config)
tracer = MindcoreTracer(config)

# Auto-instrumented operations
with tracer.start_span("recall_memories") as span:
    span.set_attribute("user_id", "user_123")
    result = memory.recall(query="preferences", user_id="user_123")

    metrics.record_recall(
        user_id="user_123",
        result_count=len(result.memories),
        latency_ms=result.latency_ms,
    )
```

### Rate Limiting

```python
from mindcore.enterprise import RateLimiter, RateLimitConfig

config = RateLimitConfig(
    default_limit="100/minute",
    tier_limits={
        "free": "10/minute",
        "pro": "100/minute",
        "enterprise": "1000/minute",
    },
    operation_limits={
        "store": "50/minute",
        "recall": "200/minute",
    },
)

limiter = RateLimiter(config)

# Check before operation
if limiter.is_allowed("user_123", operation="store", user_tier="pro"):
    memory.store(...)
else:
    remaining = limiter.get_remaining("user_123", "store", "pro")
    retry_after = limiter.get_reset_time("user_123", "store", "pro")
    raise RateLimitExceededError(f"Retry after {retry_after}s")

# Or use context manager
with limiter.limit("user_123", operation="recall"):
    result = memory.recall(...)
```

### Audit Logging

```python
from mindcore.enterprise import AuditLogger, AuditConfig

config = AuditConfig(
    enabled=True,
    file_path="/var/log/mindcore/audit.log",
    include_content=False,  # Don't log sensitive content
    redact_fields=["password", "token", "secret"],
)

audit = AuditLogger(config)

# Automatic redaction of sensitive fields
audit.log_store(
    user_id="user_123",
    memory_id="mem_abc",
    memory_type="preference",
    metadata={"password": "secret123"},  # Auto-redacted
)

audit.log_access(
    user_id="user_123",
    resource="memories",
    action="recall",
    granted=True,
)

audit.log_security_event(
    event_type="rate_limit_exceeded",
    user_id="user_123",
    details={"limit": "100/minute"},
)
```

### Encryption at Rest

```python
from mindcore.enterprise import FieldEncryptor, EncryptionConfig, KeyRotator

config = EncryptionConfig(
    key=os.environ["ENCRYPTION_KEY"],  # Fernet key
    # Or derive from password:
    # password="strong-password",
    # salt="unique-salt",
    # kdf_iterations=1_200_000,  # Django 2025 recommendation
)

encryptor = FieldEncryptor(config)

# Encrypt sensitive fields before storage
encrypted_content = encryptor.encrypt("sensitive user data")
memory.store(content=encrypted_content, ...)

# Decrypt on retrieval
decrypted = encryptor.decrypt(encrypted_content)

# Key rotation
rotator = KeyRotator(old_key, new_key)
rotated_data = rotator.rotate(encrypted_content)
```

### GDPR/CCPA Compliance

```python
from mindcore.enterprise import ComplianceManager, RetentionPolicy, AnonymizationStrategy

compliance = ComplianceManager(storage)

# GDPR Article 15: Right of Access (data export)
export = await compliance.export_user_data("user_123")

# GDPR Article 17: Right to Erasure
result = await compliance.delete_user_data("user_123")

# Anonymize for analytics
compliance.anonymize_user_data("user_123", strategy=AnonymizationStrategy.PSEUDONYMIZE)

# Retention policies
compliance.set_retention_policy(RetentionPolicy(
    memory_type_policies={"episodic": 730, "working": 1},
    default_max_age_days=365,
))
compliance.enforce_retention()  # Run periodically
```

### Smart Cache & Preferences

```python
from mindcore.v2.flr import SmartCache, PreferenceManager

# Smart write-through cache with pattern invalidation
cache = SmartCache(storage, max_size=10000, ttl_seconds=3600)
cache.invalidate_pattern("user:123:*")
print(f"Hit rate: {cache.get_stats().hit_rate:.1%}")

# Temporal preference handling with versioning
prefs = PreferenceManager(storage, flr)
prefs.set_preference(user_id="user_123", key="theme", value="dark mode")
prefs.update_preference(user_id="user_123", key="theme", value="light mode")
history = prefs.get_preference_history("user_123", "theme")
```

### Time-Based Partitioning (PostgreSQL)

```python
from mindcore.v2.storage.partitioning import PartitionManager

partitions = PartitionManager(postgres_storage)
partitions.setup_partitioning(interval="monthly")
partitions.create_future_partitions(months_ahead=3)
partitions.archive_partitions(older_than_months=12)
```

---

## Architecture

### Protocol Stack

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│                  (Your AI Agents, LLM Apps)                       │
├─────────────────────────────────────────────────────────────────┤
│                      Mindcore Orchestrator                        │
│                   store() | recall() | compress()                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │       FLR        │  │       CLST       │  │      SVL       │ │
│  │   (Hot Path)     │  │   (Cold Path)    │  │  (Semantics)   │ │
│  │                  │  │                  │  │                │ │
│  │ • Smart Cache    │  │ • Compression    │  │ • Validation   │ │
│  │ • Preferences    │  │ • Sync           │  │ • Migrations   │ │
│  │ • Reinforcement  │  │ • Transfer       │  │ • Data Sources │ │
│  │ • Context        │  │ • Partitioning   │  │ • Domains      │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                       Storage Abstraction                         │
│                 SQLiteStorage | PostgresStorage                   │
├─────────────────────────────────────────────────────────────────┤
│                        Enterprise Layer                           │
│    Observability | Rate Limiting | Audit | Encryption             │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Object

```python
from mindcore import Memory

memory = Memory(
    memory_id="mem_abc123",
    content="User prefers dark mode",
    memory_type="preference",  # episodic|semantic|procedural|preference|entity|...
    user_id="user_123",
    agent_id="support_agent",

    # Semantic metadata
    topics=["settings", "ui"],
    categories=["preference"],
    sentiment="neutral",
    importance=0.8,
    entities=["dark mode"],

    # Access control
    access_level="team",  # private|team|shared|global

    # Learning
    reinforcement_score=0.0,  # -1.0 to +1.0
    access_count=0,

    # Versioning
    vocabulary_version="2.0.0",
)
```

### Data Flow

```text
Store Flow:
  Content → SVL.validate() → CLST.store() → Storage
                                  ↓
                            FLR.cache_update()
                                  ↓
                       SessionAggregate.update()

Recall Flow (with ContextGateway):
  Query → MetadataExtractor.get_context_decision_prompt()
              ↓
          LLM decides: HistoricalContextNeeded?
              │
              ├─ "False" → FLR only (current session)
              │            ↓
              │     ContextGateway._build_flr_only_context()
              │            ↓
              │     Return current session memories
              │
              └─ "True" → Full hierarchical query
                          ↓
                   ContextGateway.build_context()
                          ↓
                   Query sessions by weighted metadata
                          ↓
                   Query memories from matched sessions
                          ↓
                   Fetch SVL data sources
                          ↓
                   Return ContextResult
              ↓
          MetadataExtractor.get_extraction_prompt()
              ↓
          LLM assigns EnforcedMetadata (SVL-compliant)
              ↓
          ContextGateway.record_response()
```

---

## Project Structure

```text
mindcore/
├── __init__.py                 # Public API exports
├── mindcore.py                 # Main Mindcore orchestrator
│
├── v2/                         # Version 2 with ContextGateway
│   ├── flr/                    # Fast Learning Recall protocol
│   │   ├── __init__.py
│   │   ├── recall.py           # FLR, Memory
│   │   ├── cache.py            # Smart write-through cache
│   │   └── preferences.py      # Temporal preference handling
│   │
│   ├── clst/                   # Cognitive Long-term Storage Transfer
│   │   ├── __init__.py
│   │   ├── storage.py          # CLST, CompressionResult, SyncResult
│   │   └── aggregates.py       # SessionAggregate, WeightCalculator
│   │
│   ├── svl/                    # Shared Vocabulary Layer
│   │   ├── __init__.py
│   │   ├── layer.py            # SharedVocabularyLayer, Migration
│   │   ├── ontology.py         # MessageType, Intent, Sentiment enums
│   │   ├── domains.py          # Pre-built domain vocabularies
│   │   ├── sources.py          # TableSource, APISource, MCPSource
│   │   ├── enforced_metadata.py # EnforcedMetadata, ContextDecision, MetadataExtractor
│   │   └── llm_providers.py    # OpenAIConfig, ClaudeConfig, GeminiConfig
│   │
│   ├── context/                # Unified context assembly
│   │   ├── __init__.py
│   │   └── gateway.py          # ContextGateway, QueryMetadata, ResponseMetadata
│   │
│   └── storage/                # Storage backends
│       ├── base.py             # BaseStorage with session aggregate methods
│       ├── sqlite.py           # SQLiteStorage
│       ├── postgres.py         # PostgresStorage with session_aggregates table
│       └── partitioning.py     # Time-based partitioning for PostgreSQL
│
├── cross_agent/                # Multi-agent support
│   ├── layer.py                # CrossAgentLayer
│   ├── sharing.py              # Memory sharing logic
│   ├── registry.py             # AgentRegistry
│   └── routing.py              # AttentionRouter
│
├── enterprise/                 # Enterprise features
│   ├── observability.py        # OpenTelemetry metrics/tracing
│   ├── rate_limiting.py        # Rate limiter
│   ├── audit.py                # Audit logging
│   ├── encryption.py           # Field encryption
│   └── compliance.py           # GDPR/CCPA compliance tools
│
├── server/                     # API servers
│   ├── mcp.py                  # MCP server
│   └── rest.py                 # FastAPI REST server
│
├── exceptions.py               # Standardized exceptions
├── tests/                      # Test suite
│
└── utils/                      # Logging utilities
```

---

## API Reference

### Mindcore (Orchestrator)

```python
class Mindcore:
    def store(content, memory_type, user_id, ...) -> str
    def recall(query, user_id, ...) -> RecallResult
    def search(query, user_id, ...) -> list[Memory]
    def reinforce(memory_id, signal) -> float
    def compress(user_id, older_than_days, strategy) -> CompressionResult
    def delete(memory_id) -> None
    def get_json_schema() -> dict
    def extract_from_response(response, user_id) -> list[Memory]
```

### FLR

```python
class FLR:
    def query(query, user_id, ...) -> RecallResult
    def reinforce(memory_id, signal) -> float
    def promote(memory_id) -> bool
    def update_context(session_id, ...) -> ContextWindow
    def get_context(session_id) -> ContextWindow | None
    def flush_reinforcements() -> int
```

### CLST

```python
class CLST:
    def store(memory, validate=True) -> str
    def retrieve(memory_id) -> Memory | None
    def search(...) -> list[Memory]
    def compress(user_id, ...) -> CompressionResult
    def sync(source_agent, target_agent, ...) -> SyncResult
    def transfer(memories, destination) -> TransferManifest
    def migrate(from_version, ...) -> MigrationResult
    def rollback_migration(result) -> MigrationResult
```

### SVL

```python
class SharedVocabularyLayer:
    def add_domain(domain_name) -> None
    def add_topics(*topics) -> None
    def add_custom_field(name, field_type, ...) -> None
    def validate_memory(memory) -> tuple[bool, list[str]]
    def map_source(term, source) -> None
    def fetch_for_topics(topics, context) -> dict
    def add_migration(migration) -> None
    def migrate_memory(memory, from_version, create_checkpoint) -> dict | tuple
    def rollback_memory(memory, checkpoint) -> dict
    def get_json_schema() -> dict
    def to_typescript() -> str
    def to_pydantic() -> str
```

### ContextGateway

```python
class ContextGateway:
    def build_context(query, user_id, session_id, attention_hints, ...) -> ContextResult
    def build_context_with_decision(query, context_decision, user_id, ...) -> ContextResult
    def record_response(query_metadata, response_text, memories_to_store, ...) -> ResponseMetadata
```

---

## ContextGateway & Hierarchical Retrieval

The **ContextGateway** is the unified entry point for building LLM context, orchestrating FLR (hot path), CLST (cold path), and SVL (data sources).

### Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   ContextGateway                         │
│                                                          │
│  1. LLM decides: HistoricalContextNeeded?               │
│     ├─ "False" → FLR only (current session)             │
│     └─ "True"  → Full CLST hierarchical query           │
│                                                          │
│  2. Hierarchical Retrieval (if CLST needed)             │
│     ├─ Query sessions by weighted metadata              │
│     └─ Query memories from matched sessions             │
│                                                          │
│  3. Fetch SVL data sources for matched topics           │
│                                                          │
│  4. Create SVL-compliant QueryMetadata                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
ContextResult (unified context for LLM)
```

### Session Aggregates with Weighted Metadata

Instead of flat memory search, Mindcore uses **hierarchical retrieval**:

```python
from mindcore.v2.clst import SessionAggregate

# Session-level metadata with weights
aggregate = SessionAggregate(
    session_id="session_123",
    user_id="user_456",

    # Weighted distributions (term → weight 0-1)
    topic_weights={"orders": 0.85, "shipping": 0.6, "returns": 0.3},
    category_weights={"support": 0.9, "billing": 0.4},
    intent_weights={"ask_question": 0.7, "request_action": 0.5},

    # Importance statistics
    importance_avg=0.72,
    importance_max=0.95,

    # Dominant values
    dominant_topic="orders",
    dominant_category="support",
)
```

**Weight Calculation:**

```python
topic_weight = (frequency * 0.4) + (avg_importance * 0.4) + (recency * 0.2)
```

### HistoricalContextNeeded Decision

The LLM decides if historical context (CLST) is needed:

```python
from mindcore.v2.svl import ContextDecision, HistoricalContextNeeded, MetadataExtractor
from mindcore.v2.context import ContextGateway

extractor = MetadataExtractor(svl=shared_vocabulary_layer)
gateway = ContextGateway(storage=postgres_storage, svl=svl)

# Get LLM to decide
decision_prompt = extractor.get_context_decision_prompt(
    user_message="What did you tell me about my order last week?",
)
# ... send to LLM ...

# Parse decision
decision = extractor.parse_context_decision(llm_response)
# ContextDecision(historical_context_needed="True", suggested_topics=["orders"], ...)

# Build context based on decision
context = gateway.build_context_with_decision(
    query="What did you tell me about my order last week?",
    context_decision=decision,
    user_id="user_123",
    session_id="session_abc",
)

# If HistoricalContextNeeded = "False", only current session is queried (faster)
# If HistoricalContextNeeded = "True", full hierarchical CLST query
```

---

## Enforced SVL Metadata Extraction

The **MetadataExtractor** forces LLMs to assign metadata from SVL vocabulary:

### Enforced Metadata Schema

```python
from mindcore.v2.svl import EnforcedMetadata

metadata = EnforcedMetadata(
    message_id="msg_abc123",
    user_id="user_123",
    session_id="session_456",
    thread_id="thread_789",  # For multi-thread conversations

    # SVL-enforced (LLM must choose from vocabulary)
    topics=["orders", "shipping"],
    categories=["support"],
    entities=["Order #12345"],
    message_type="query",           # query, command, statement, ...
    message_intent="ask_question",  # ask_question, request_action, ...

    # Scores
    importance=0.7,
    confidence=0.9,
    urgency="medium",
    sentiment="neutral",
    emotional_classification="neutral",

    # Memory classification
    memory_type="episodic",
    access_level="private",
)
```

### Multi-Provider LLM Support

Mindcore supports the latest LLM APIs with reasoning/thinking modes:

```python
from mindcore.v2.svl import (
    MetadataExtractor,
    OpenAIConfig,
    ClaudeConfig,
    GeminiConfig,
    ReasoningEffort,
    ThinkingMode,
)

extractor = MetadataExtractor(svl=shared_vocabulary_layer)

# OpenAI GPT-5 with Responses API
request = extractor.get_openai_request(
    user_message="What's my order status?",
    model="gpt-5",
    reasoning_effort="high",      # low, medium, high, xhigh
    use_responses_api=True,       # 3-5% better intelligence
)

# Claude with Extended Thinking
request = extractor.get_claude_request(
    user_message="What's my order status?",
    model="claude-sonnet-4-5-20250514",
    thinking_budget=16000,        # Max reasoning tokens
    use_extended_thinking=True,
)

# Gemini with Thinking Mode
request = extractor.get_gemini_request(
    user_message="What's my order status?",
    model="gemini-2.5-flash",
    thinking_mode="dynamic",      # disabled, dynamic, fixed
)
```

### Provider Configurations

| Provider | API | Key Features | Temperature |
|----------|-----|--------------|-------------|
| **OpenAI GPT-5** | Responses API | `reasoning_effort`, preserved reasoning across turns, structured outputs | 0.0 |
| **Claude** | Messages API | `thinking.budget_tokens`, interleaved thinking, `output_format` | N/A (thinking) |
| **Gemini** | GenerativeAI | `thinkingBudget`, dynamic thinking, `response_schema` | 0.0 |

### Best Practices

- **Temperature 0** for deterministic metadata extraction
- **Reasoning/thinking enabled** for accurate classification
- **JSON Schema validation** ensures SVL vocabulary compliance
- **Seed parameter** (42) for reproducibility (OpenAI, Gemini)

---

## Configuration

### Environment Variables

```bash
# Database
export DATABASE_URL="postgresql://user:pass@localhost/mindcore"

# Enterprise features (optional)
export ENCRYPTION_KEY="your-fernet-key"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

### Programmatic Configuration

```python
from mindcore import Mindcore, SharedVocabularyLayer

# Custom vocabulary
svl = SharedVocabularyLayer(
    domains=["customer_service"],
    version="1.0.0",
)
svl.add_topics("custom_topic")

# Initialize with custom vocabulary
memory = Mindcore(
    storage="postgresql://localhost/mindcore",
    vocabulary=svl,
    enable_multi_agent=True,
)
```

---

## Testing

```bash
# Run all tests
pytest mindcore/tests/ -v

# Run with coverage
pytest mindcore/tests/ --cov=mindcore --cov-report=html

# Run specific test file
pytest mindcore/tests/test_enterprise.py -v
```

Current test status: **585 tests passing, 61% coverage**

---

## Contributing

```bash
# Clone
git clone https://github.com/M-Alfaris/mindcore.git
cd mindcore

# Install with dev dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
ruff format mindcore/
ruff check --fix mindcore/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **OpenTelemetry** - Observability framework
- **limits** - Rate limiting library
- **structlog** - Structured logging
- **cryptography** - Encryption (Fernet)
- **FastAPI** - REST API framework
- **PostgreSQL/SQLite** - Storage backends

---

<div align="center">

**Like MCP standardized connections, Mindcore standardizes memory.**

```bash
pip install mindcore
```

[Quick Start](#quick-start) | [Protocols](#the-three-protocols) | [Enterprise](#enterprise-features)

---

Made with care by the Mindcore team

</div>
