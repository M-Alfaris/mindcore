# Claude.md - Mindcore Development Guide

## What is Mindcore?

Mindcore is a **Structured Augmented Generation Engine (SAGE)** - a deterministic alternative to RAG for AI agents.

Instead of vector embeddings and similarity search, Mindcore generates enriched metadata dimensions for every piece of content and uses PostgreSQL queries on those dimensions for retrieval. Every inbound message gets metadata extracted, every outbound response gets metadata assigned and stored with versioning.

**SAGE replaces RAG.** Same query always returns same results. No embeddings, no vector database, no cosine similarity. Just structured metadata and SQL.

```
RAG:   Content → Chunk → Embed → Vector DB → Similarity Search → Fuzzy Results
SAGE:  Content → Extract Metadata → PostgreSQL → Dimension Matching → Ranked Results
```

## Core Architecture

```
User Message
    ↓
SVL extracts metadata dimensions:
  message_type, intent, topics[], category, sentiment,
  urgency, confidence, chapter, section, page,
  is_preference, preferences[], memory_type...
    ↓
PostgreSQL matches stored content by metadata dimensions
  + Domain data triggered by topics (tables, APIs)
  + User preferences
  + Past interactions
  + Documents with matching metadata
    ↓
Ranked, structured results → Agent LLM
    ↓
Agent Response → also gets metadata → stored with versioning
```

## The Three Protocols

### FLR (Fast Learning Recall) - Hot Path
- `mindcore/flr/simple_recall.py` → `DeterministicRecall` class
- O(1) LRU cache lookup, deterministic filtering
- Decides if CLST is needed via `CLSTDecision`
- Collects reinforcement signals, passes to CLST
- No probabilistic scoring - just cache + filter + decision

### CLST (Cognitive Long-term Storage Transfer) - Cold Path
- `mindcore/clst/storage.py` → `CLST` class
- PostgreSQL-centric persistent storage
- Complex weighted scoring (BM25, topic weights, recency, reinforcement)
- Session management and segmentation
- Compression strategies (summarize, merge, deduplicate)
- Signal processing with full audit history

### SVL (Structured Validation Layer) - Metadata Kernel
- `mindcore/svl/layer.py` → `StructuredValidationLayer` class
- `mindcore/svl/gate.py` → `SVLGate` class (mandatory validation choke point)
- Forces LLM to output structured JSON with SVL-compliant vocabulary
- Validates every piece of data inbound and outbound
- Manages domain vocabularies, data sources, migrations
- No bypass paths - all data goes through SVL Gate

## Key Design Principles

1. **Metadata over Embeddings** - Content is indexed by structured dimensions, not vectors
2. **PostgreSQL is the brain** - All heavy lifting in PostgreSQL functions, triggers, and queries. Python is just the thin client
3. **Deterministic retrieval** - Same metadata query always returns same results
4. **Bidirectional tracking** - Both user messages and agent responses get metadata and versioning
5. **SVL Gate has no bypass** - Every data flow is validated. `GateDecision`: ACCEPT, REJECT, RETRY, CANONICALIZE, FALLBACK
6. **FLR decides, CLST processes** - Hot path makes fast decisions, cold path does complex scoring and signal processing
7. **Reinforcement improves over time** - Memories gain/lose relevance through feedback signals with temporal decay

## Data Flow

### Storing a memory
```
LLM structured output → SVL Gate validate → GatedCLST.store() → PostgreSQL
                         (canonicalize,       (generate ID,
                          validate types,      auto-fill fields,
                          check policy)        persist)
```

### Recalling memories
```
User query → SVL extract metadata from query
           → DeterministicRecall.query() (cache lookup)
           → CLSTDecision: needs CLST?
               YES → CLST.search() (PostgreSQL multi-factor scoring)
               NO  → return cache results
           → Fetch domain sources triggered by topics
           → SVL Gate validate outbound
           → Return ranked, structured results
```

## Module Map

| Module | Purpose |
|--------|---------|
| `mindcore/mindcore.py` | Main orchestrator class |
| `mindcore/svl/` | Structured Validation Layer - metadata extraction, validation, domains |
| `mindcore/svl/gate.py` | SVL Gate - mandatory validation kernel |
| `mindcore/svl/pipeline.py` | Complete orchestrated data flow |
| `mindcore/svl/enforced_metadata.py` | LLM metadata extraction with structured outputs |
| `mindcore/svl/domain_sources.py` | PostgreSQL-centric domain source management |
| `mindcore/svl/sources.py` | TableSource, APISource, MCPSource, FunctionSource |
| `mindcore/svl/domains.py` | Pre-built domain vocabularies |
| `mindcore/flr/simple_recall.py` | DeterministicRecall - hot path cache |
| `mindcore/flr/reinforcement.py` | Robust reinforcement with temporal decay |
| `mindcore/clst/storage.py` | CLST - cold path persistent storage |
| `mindcore/clst/session_segmentation.py` | Session management and topic detection |
| `mindcore/context/gateway.py` | ContextGateway - unified context assembly |
| `mindcore/federation/` | Multi-agent federation with access control |
| `mindcore/server/mcp.py` | MCP server for Claude/GPT/Gemini integration |
| `mindcore/server/rest.py` | FastAPI REST server |
| `mindcore/cli/` | CLI commands (init, doctor, demo, mcp, serve) |
| `mindcore/storage/` | Storage backends (SQLite dev, PostgreSQL prod) |
| `mindcore/storage/schema/` | PostgreSQL schema extensions, triggers, functions |
| `mindcore/enterprise/` | Audit, encryption, compliance, observability, rate limiting |

## Naming Conventions

- **StructuredValidationLayer** (not SharedVocabularyLayer - old name, alias exists for backwards compat)
- **DeterministicRecall** (not SimpleFLR - old name, alias exists for backwards compat)
- No "v2", "v3", "simple" prefixes - we're in active development

## How Agents Connect

```
Python Agent  → pip install mindcore → from mindcore import Mindcore (direct SDK)
MCP Agent     → mindcore mcp        → stdio/HTTP (Claude Desktop, Cursor)
Any Language  → mindcore serve       → REST API
All           → PostgreSQL           → the actual engine
```

No extra Go/Rust layers. PostgreSQL does the heavy lifting. Python is the thin client.

## PostgreSQL Schema

Key schema files in `mindcore/storage/schema/`:
- `domain_sources.sql` - Domain source configs, audit log, user preferences, topic-to-source mapping
- `ranking_functions.sql` - Multi-component scoring (content, topic, recency, reinforcement, importance)
- `session_triggers.sql` - Auto-update session aggregates on memory insert
- `materialized_views.sql` - Pre-computed stats for dashboards
- `search_columns.sql` - Generated columns and indexes for fast search
- `extensions.sql` - pg_trgm for fuzzy search
- `partitioning.sql` - Time-based partitioning for 10M+ memories

## Reinforcement System

Signals improve memory relevance over time:
- **Signal types**: RELEVANCE (0.35), USEFULNESS (0.30), CORRECTNESS (0.20), TIMELINESS (0.10), COMPLETENESS (0.05)
- **Signal sources**: USER_EXPLICIT (1.0), USER_IMPLICIT (0.7), LLM_EVALUATION (0.5), CROSS_AGENT (0.6), AUTOMATED (0.3)
- **Temporal decay**: Exponential with configurable half-life
- **Score bounds**: Always [-1.0, 1.0]
- **Audit trail**: Every signal recorded with old/new scores

## Test Structure

Tests in `mindcore/tests/` and `testing/tests/`. Run with:
```bash
pytest mindcore/tests/ --override-ini="addopts=" -v
```

Key test files:
- `test_svl.py`, `test_svl_gate.py` - SVL validation
- `test_flr_recall.py`, `test_flr_reinforcement.py` - Hot path
- `test_enforced_metadata.py` - LLM metadata extraction
- `test_domain_sources.py` - PostgreSQL source management
- `test_federation.py` - Multi-agent federation
- `test_compliance.py` - GDPR/CCPA

## Configuration

YAML config at `mindcore.yaml` or via CLI `mindcore init`. Key sections:
- `storage` - SQLite or PostgreSQL connection
- `svl.domains` - Active domain vocabularies
- `svl.policies` - Strict mode, required fields
- `svl.sources` - Topic-to-table/API mappings
- `llm` - Provider config (OpenAI, Anthropic, Google)
- `enterprise` - Audit, encryption, compliance, rate limiting

## What NOT to Do

- Don't add vector embeddings as the primary retrieval mechanism - metadata dimensions are the core approach
- Don't bypass SVL Gate - all data must be validated
- Don't put complex scoring in FLR - keep hot path fast, CLST handles complex scoring
- Don't add "v2", "v3", "simple" prefixes to names
- Don't create extra language layers (Go, Rust) between Python and PostgreSQL
- Don't treat Mindcore as just a memory layer - it's a complete RAG replacement (SAGE)
