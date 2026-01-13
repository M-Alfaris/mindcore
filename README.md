<div align="center">

# Mindcore

## Memory Layer for AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 14+](https://img.shields.io/badge/postgresql-14+-336791.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/M-Alfaris/mindcore)

**Build once. Deploy endlessly. Minutes, not months.**

[The Problem](#the-problem) | [Quick Start](#quick-start) | [Multi-Agent](#multi-agent-support) | [Integrations](#framework-integrations)

---

</div>

## The Problem

Building a reliable memory system for AI agents is **hard**:

| Challenge | What Teams Face |
|-----------|-----------------|
| **Time** | 2-4 months building custom memory infrastructure |
| **Cost** | Vector databases ($50-500+/mo) + embedding API calls |
| **Reliability** | Probabilistic retrieval = unpredictable results |
| **Traceability** | Black-box scoring = impossible to debug |
| **Security** | DIY access control, no audit trail |
| **Scale** | Every new agent starts from zero |

**The result?** Teams spend months on infrastructure instead of building their product. Every new agent requires rebuilding context from scratch. Cross-agent memory sharing is a nightmare.

---

## The Solution

Mindcore is an open-source memory layer that makes deploying a solid, traceable memory system **feasible in minutes**.

```bash
pip install mindcore
```

### What You Get

- **Predictable** - Deterministic SQL scoring. Every retrieval is traceable and explainable.
- **Fast** - <160ms P95 latency. PostgreSQL full-text search, no vector DB required.
- **Cheap** - No embeddings, no vector database fees. Just PostgreSQL.
- **Secure** - Built-in RBAC, agent isolation, full audit trail.
- **Scalable** - Build once, deploy to unlimited agents. Horizontal scaling.

### Build Once, Deploy Endlessly

New agents inherit organizational knowledge from day one:

```python
from mindcore import Mindcore

# Your memory layer - shared across all agents
memory = Mindcore(storage="postgresql://localhost/mindcore")

# Agent 1 stores knowledge
memory.store(
    content="Customer prefers email communication",
    memory_type="preference",
    user_id="customer_123",
    topics=["communication", "preferences"],
    agent_id="support_agent",
)

# Agent 2 immediately has access (with proper RBAC)
result = memory.recall(
    query="how does customer prefer to be contacted",
    user_id="customer_123",
    agent_id="sales_agent",  # Different agent, shared context
)
```

---

## Quick Start

```python
from mindcore import Mindcore

# Initialize with PostgreSQL (production) or SQLite (development)
memory = Mindcore(storage="postgresql://localhost/mindcore")

# Store a memory
memory.store(
    content="User prefers dark mode",
    memory_type="preference",
    user_id="user_123",
    topics=["settings", "ui"],
    importance=0.8,
)

# Recall relevant memories - deterministic, traceable scoring
result = memory.recall(
    query="user preferences",
    user_id="user_123",
)

for mem in result.memories:
    print(f"[{mem['importance']:.2f}] {mem['content']}")
```

---

## Multi-Agent Support

Mindcore handles single agents and multi-agent systems with solid isolation and RBAC.

### Agent Isolation & RBAC

Define what each agent can access:

```python
from mindcore import Mindcore

memory = Mindcore(
    storage="postgresql://localhost/mindcore",
    enable_multi_agent=True,
)

# Register agents with permissions
memory.register_agent(
    agent_id="support_agent",
    name="Customer Support",
    teams=["support"],
    access_levels=["public", "support_internal"],
)

memory.register_agent(
    agent_id="sales_agent",
    name="Sales Assistant",
    teams=["sales"],
    access_levels=["public", "sales_internal"],
)

# Store with access control
memory.store(
    content="Customer complained about pricing",
    memory_type="episodic",
    user_id="customer_123",
    access_level="support_internal",  # Only support agents can see this
    agent_id="support_agent",
)

# Sales agent cannot access support_internal memories
result = memory.recall(
    query="customer feedback",
    user_id="customer_123",
    agent_id="sales_agent",  # Won't see support_internal content
)
```

### Shared Context, Zero Rebuild

New agents start with full organizational context:

```text
Agent 1 (Month 1)     Agent 2 (Month 3)     Agent 3 (Month 6)
      │                     │                     │
      └──────────┬──────────┴──────────┬──────────┘
                 │                     │
                 ▼                     ▼
         ┌─────────────────────────────────────┐
         │           MINDCORE                   │
         │                                      │
         │  Shared Memory + Vocabulary + RBAC   │
         │                                      │
         │  New agents inherit everything.      │
         │  No cold start. No rebuild.          │
         └─────────────────────────────────────┘
```

---

## Why Not RAG?

| | RAG | Mindcore |
|--|-----|----------|
| **Cost** | Embeddings + Vector DB | Just PostgreSQL |
| **Latency** | 200-500ms | <160ms |
| **Scoring** | Probabilistic (cosine similarity) | Deterministic (SQL function) |
| **Debug** | Black box | Full SQL trace |
| **Predict** | "Probably relevant" | Exact score breakdown |
| **Schema** | Unstructured chunks | Enforced vocabulary |

### Traceable Scoring

Every memory retrieval is explainable:

```sql
-- Mindcore's sage_score() is deterministic
sage_score(
    search_rank,      -- 30% Full-text relevance
    recency_hours,    -- 20% Time decay (24h half-life)
    reinforcement,    -- 15% User feedback
    importance,       -- 15% Memory importance
    confidence,       -- 10% LLM confidence
    access_count,     -- 5%  Popularity
    topic_match       -- 5%  Topic overlap
) → 0.847

-- You know exactly why this memory was ranked #1
```

---

## Framework Integrations

Mindcore integrates with popular AI frameworks.

### LangChain

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from mindcore.integrations import MindcoreMemory

memory = MindcoreMemory(
    storage="postgresql://localhost/mindcore",
    user_id="user_123",
)

chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
)

response = chain.predict(input="What are my preferences?")
```

### CrewAI

```python
from crewai import Crew, Agent, Task
from mindcore.integrations import MindcoreCrewMemory

memory = MindcoreCrewMemory(
    storage="postgresql://localhost/mindcore",
    crew_id="research_crew",
    enable_cross_crew=True,  # Share memory across crews
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=memory,
)
```

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex
from mindcore.integrations import MindcoreIndexMemory

memory = MindcoreIndexMemory(
    storage="postgresql://localhost/mindcore",
    user_id="user_123",
)

chat_engine = index.as_chat_engine(
    memory=memory,
    chat_mode="context",
)
```

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR AI AGENTS                                  │
│                  (Single Agent or Multi-Agent Fleet)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MINDCORE                                        │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │     FLR     │    │    CLST     │    │     SVL     │    │    RBAC    │  │
│   │  (Hot Path) │    │ (Cold Path) │    │  (Kernel)   │    │ (Security) │  │
│   │             │    │             │    │             │    │            │  │
│   │ • O(1) Cache│    │ • Sessions  │    │ • Validate  │    │ • Agents   │  │
│   │ • Signals   │    │ • Scoring   │    │ • Schema    │    │ • Teams    │  │
│   │ • Learning  │    │ • Compress  │    │ • Sources   │    │ • Audit    │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              POSTGRESQL                                      │
│                                                                              │
│   • sage_score() - Deterministic, traceable scoring                         │
│   • Full-text search (tsvector) - No embeddings needed                      │
│   • Session aggregates - Hierarchical retrieval                             │
│   • JSONB metadata - Flexible, indexed, fast                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Enterprise & Managed

Mindcore is **open-source** (MIT licensed).

For teams that need more, we offer **Mindcore Enterprise**:

| Feature | Open Source | Enterprise |
|---------|-------------|------------|
| Core protocols (FLR, CLST, SVL) | ✓ | ✓ |
| PostgreSQL + SQLite backends | ✓ | ✓ |
| Multi-agent RBAC | ✓ | ✓ |
| Framework integrations | ✓ | ✓ |
| Managed cloud hosting | - | ✓ |
| SSO / SAML | - | ✓ |
| Priority support | - | ✓ |
| SLAs | - | ✓ |
| Custom integrations | - | ✓ |

[Contact us](mailto:enterprise@mindcore.dev) for Enterprise.

---

## LLM Providers

Works with any LLM:

```python
from mindcore.svl import OpenAIConfig, ClaudeConfig, GeminiConfig

# OpenAI
config = OpenAIConfig(model="gpt-4o")

# Anthropic
config = ClaudeConfig(model="claude-sonnet-4-20250514")

# Google
config = GeminiConfig(model="gemini-2.5-pro")
```

---

## Project Structure

```text
mindcore/
├── mindcore.py          # Main orchestrator
├── flr/                 # Fast Learning Recall (hot path)
├── clst/                # Cognitive Long-term Storage (cold path)
├── svl/                 # Shared Vocabulary Layer (validation)
├── storage/             # PostgreSQL & SQLite backends
├── integrations/        # LangChain, CrewAI, LlamaIndex
├── access/              # RBAC and permissions
├── enterprise/          # Metrics, tracing, audit, encryption
└── server/              # MCP and REST API
```

---

## Get Started

```bash
# Install
pip install mindcore

# Quick test
python -c "from mindcore import Mindcore; print('Ready!')"

# Run with SQLite (dev)
python -c "
from mindcore import Mindcore
m = Mindcore(storage='sqlite:///test.db')
m.store('Hello world', 'episodic', 'user1')
print(m.recall('hello', 'user1').memories)
"
```

---

## License

MIT License - see [LICENSE](LICENSE).

---

<div align="center">

**Build once. Deploy endlessly.**

**Minutes, not months.**

```bash
pip install mindcore
```

[Documentation](docs/) | [Examples](examples/) | [GitHub](https://github.com/M-Alfaris/mindcore)

---

Made with care by the Mindcore team

</div>
