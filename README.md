<div align="center">

# Mindcore

### The Context Protocol for AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/M-Alfaris/mindcore)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**A semi-deterministic framework for persistent memory, intelligent context retrieval, and external data integration for AI agents.**

Stop rebuilding memory infrastructure. Start building features.

[Quick Start](#-quick-start) • [Why Mindcore](#-why-mindcore) • [Architecture](#-architecture) • [Features](#-features) • [Roadmap](#-roadmap)

---

</div>

## The Problem

Every team building AI agents faces the same challenges:

| Challenge | What Teams Do Today | Time Spent |
|-----------|---------------------|------------|
| **Memory & Persistence** | Build custom storage, caching, retrieval | 2-4 weeks |
| **Context Engineering** | Trial and error with vector DBs, embeddings | 3-6 weeks |
| **User Preferences** | Ad-hoc storage, no consistency | 1-2 weeks |
| **External Data** | Custom integrations per system | 2-4 weeks per system |
| **Multi-Agent Consistency** | Each agent has its own memory | Ongoing pain |

**Result**: 2-3 months before you can focus on what matters — your actual product.

## The Solution

Mindcore is an **open-source Context Protocol** that provides:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR AI AGENTS                                  │
│         (Support Bot, Sales Assistant, Internal Tools, etc.)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MINDCORE                                        │
│                        The Context Protocol                                  │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ MEMORY LAYER    │  │ INTELLIGENCE    │  │ EXTERNAL CONNECTORS         │  │
│  │                 │  │                 │  │                             │  │
│  │ • Messages      │  │ • Enrichment    │  │ • Orders (read-only)        │  │
│  │ • Summaries     │  │ • Smart Context │  │ • Billing (read-only)       │  │
│  │ • Preferences   │  │ • Tool Calling  │  │ • CRM (read-only)           │  │
│  │ • Cache         │  │ • Compression   │  │ • Your Systems...           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
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

## Why Mindcore?

### 1. Semi-Deterministic Retrieval (No Vector DB Required)

Most teams assume they need a vector database for context retrieval. **They don't.**

| Approach | How It Works | The Reality |
|----------|--------------|-------------|
| **Pure Vector DB** | Embed everything, semantic search | Expensive, unpredictable, overkill for conversations |
| **Pure Rules** | "Last 10 messages + keyword match" | Misses nuance, brittle |
| **Mindcore** | LLM enriches metadata → Deterministic retrieval → LLM summarizes | Predictable, debuggable, cost-effective |

```python
# User: "What did we discuss about billing last week?"

# Mindcore approach:
# 1. Message metadata already extracted: topics=["billing"], date=2024-03-15
# 2. Deterministic query: SELECT * WHERE topics @> 'billing' AND date > '2024-03-08'
# 3. LLM summarizes the filtered results

# Result: Faster, cheaper, and more predictable than vector similarity search
```

### 2. Shared Memory Across All Your Agents

Without Mindcore:
```
Support Bot: "I see you're a new customer!"
Sales Bot (same day): "Welcome! Interested in our product?"
User: "I literally just bought it an hour ago..."
```

With Mindcore:
```
All agents share the same memory → Consistent user experience
```

### 3. External Data Integration (Read-Only)

Connect your AI agents to real business data:

```python
# User: "What's the status of my order from last week?"

# Mindcore automatically:
# 1. Detects topic: "orders"
# 2. Extracts entities: date_range="last week"
# 3. Queries your Orders DB (read-only)
# 4. Includes order data in context

# Your agent responds with actual order information
```

### 4. Production-Ready in Hours, Not Months

```python
from mindcore import MindcoreClient

# Initialize
client = MindcoreClient(use_sqlite=True)

# Ingest messages (auto-enriched with metadata)
client.ingest_message({
    "user_id": "user_123",
    "thread_id": "thread_456",
    "session_id": "session_789",
    "role": "user",
    "text": "I need help with my billing issue"
})

# Get intelligent context for any query
context = client.get_context_smart(
    user_id="user_123",
    thread_id="thread_456",
    query="What billing issues has this user mentioned?"
)

# Use in your LLM prompt
print(context.assembled_context)
```

---

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With local LLM support (zero API costs)
pip install -e ".[llama]"

# With async support
pip install -e ".[async]"

# Everything
pip install -e ".[all]"
```

### Option A: Local LLM (Recommended - Zero API Costs)

```bash
# Download a model (~2GB)
mindcore download-model

# Set the model path
export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### Option B: OpenAI API

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

### Basic Usage

```python
from mindcore import MindcoreClient

client = MindcoreClient(use_sqlite=True)

# Ingest a message
message = client.ingest_message({
    "user_id": "user_123",
    "thread_id": "thread_456",
    "session_id": "session_789",
    "role": "user",
    "text": "What are the best practices for building AI agents?"
})

# Auto-enriched metadata
print(message.metadata.topics)      # ['AI', 'agents', 'best practices']
print(message.metadata.intent)      # 'ask_question'
print(message.metadata.importance)  # 0.8

# Get intelligent context (single LLM call with tool calling)
context = client.get_context_smart(
    user_id="user_123",
    thread_id="thread_456",
    query="AI agent architecture"
)

print(context.assembled_context)  # Relevant, summarized context
print(context.key_points)         # Key insights from history
```

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MINDCORE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         MEMORY LAYER                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │  Hot Messages   │  │ Thread Summaries│  │  User Preferences   │   │  │
│  │  │  (recent)       │  │ (compressed)    │  │  (amendable)        │   │  │
│  │  │                 │  │                 │  │                     │   │  │
│  │  │ Full message    │  │ Old threads →   │  │ • Language          │   │  │
│  │  │ history with    │  │ LLM-generated   │  │ • Timezone          │   │  │
│  │  │ rich metadata   │  │ summaries       │  │ • Interests         │   │  │
│  │  └─────────────────┘  └─────────────────┘  │ • Custom context    │   │  │
│  │                                            └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      INTELLIGENCE LAYER                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │ EnrichmentAgent │  │SmartContextAgent│  │ SummarizationAgent  │   │  │
│  │  │                 │  │                 │  │                     │   │  │
│  │  │ • Topics        │  │ • Tool calling  │  │ • Compress threads  │   │  │
│  │  │ • Intent        │  │ • Smart routing │  │ • Extract key facts │   │  │
│  │  │ • Sentiment     │  │ • Context merge │  │ • Preserve entities │   │  │
│  │  │ • Entities      │  │                 │  │                     │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    EXTERNAL CONNECTORS (Read-Only)                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Orders    │  │   Billing   │  │     CRM     │  │  Your APIs  │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ topic:orders│  │topic:billing│  │ topic:crm   │  │ topic:...   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  │                                                                       │  │
│  │  Connectors are READ-ONLY — they fetch data, never modify it          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
                    User Message
                         │
                         ▼
              ┌─────────────────────┐
              │    ingest_message   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   EnrichmentAgent   │ ◄── Extracts: topics, intent,
              │                     │     sentiment, entities, importance
              └──────────┬──────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐
    │  Database   │             │    Cache    │
    │ (persistent)│             │ (fast read) │
    └─────────────┘             └─────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  get_context_smart  │ ◄── Query arrives
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  SmartContextAgent  │ ◄── Single LLM call with tools:
              │                     │     • get_recent_messages
              │   Tool Calling      │     • search_history
              │                     │     • get_historical_summaries
              │                     │     • get_user_preferences
              │                     │     • lookup_external_data
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  AssembledContext   │
              │                     │
              │ • Summary           │
              │ • Key points        │
              │ • External data     │
              │ • User preferences  │
              └─────────────────────┘
```

---

## Features

### Current Features

| Feature | Description |
|---------|-------------|
| **Message Ingestion** | Store messages with automatic metadata enrichment |
| **Smart Context Retrieval** | Single LLM call with tool calling for intelligent context assembly |
| **Multi-tenant** | User/thread/session isolation out of the box |
| **Flexible Storage** | SQLite (dev) or PostgreSQL (production) |
| **LLM Agnostic** | OpenAI, local llama.cpp, or any OpenAI-compatible API |
| **Async Support** | Full async client for high-performance applications |
| **Background Enrichment** | Persistent queue for reliable metadata processing |
| **REST API** | FastAPI server with interactive docs |
| **Web Dashboard** | Vue.js dashboard for monitoring and configuration |

### Planned Features (Roadmap)

| Feature | Status | Description |
|---------|--------|-------------|
| **Thread Summarization** | Planned | Compress old threads into summaries, reduce storage 90% |
| **User Preferences** | Planned | Amendable settings (language, timezone, interests) |
| **External Connectors** | Planned | Read-only access to Orders, Billing, CRM systems |
| **Forgetting Policies** | Planned | Auto-delete old data based on retention rules |
| **Cross-thread Context** | Planned | "User mentioned X in another conversation" |

---

## Use Cases

### Customer Support Bot

```python
# Support bot with full context awareness
context = client.get_context_smart(
    user_id="customer_123",
    thread_id="support_ticket_456",
    query="Customer is asking about their refund"
)

# Context includes:
# - Previous support interactions
# - Order history (via Orders connector)
# - Billing status (via Billing connector)
# - User preferences (language, communication style)
```

### Multi-Agent Organization

```python
# All agents share the same Mindcore instance
support_bot = YourSupportBot(mindcore=client)
sales_bot = YourSalesBot(mindcore=client)
internal_assistant = YourInternalBot(mindcore=client)

# When a user talks to support, sales bot knows about it
# No more "I see you're a new customer" after they just purchased
```

### Enterprise AI Assistant

```python
# Connect to internal systems
from mindcore.connectors import OrdersConnector, BillingConnector

client.register_connector(OrdersConnector(
    db_url="postgresql://readonly:pass@orders-db/orders"
))

client.register_connector(BillingConnector(
    db_url="postgresql://readonly:pass@billing-db/billing"
))

# Now context automatically includes relevant business data
context = client.get_context_smart(
    user_id="employee_123",
    thread_id="internal_chat",
    query="What's the status of order #12345?"
)
# Context includes actual order data from your Orders DB
```

---

## Async Support

For high-performance applications using FastAPI or other async frameworks:

```python
import asyncio
from mindcore import get_async_client

async def main():
    AsyncMindcoreClient = get_async_client()

    async with AsyncMindcoreClient(use_sqlite=True) as client:
        # Ingest (async)
        message = await client.ingest_message({
            "user_id": "user_123",
            "thread_id": "thread_456",
            "session_id": "session_789",
            "role": "user",
            "text": "Hello!"
        })

        # Get context (async)
        context = await client.get_context_smart(
            user_id="user_123",
            thread_id="thread_456",
            query="greeting"
        )

asyncio.run(main())
```

### FastAPI Integration

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from mindcore import get_async_client

mindcore_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mindcore_client
    AsyncMindcoreClient = get_async_client()
    mindcore_client = AsyncMindcoreClient(use_sqlite=True)
    await mindcore_client.connect()
    yield
    await mindcore_client.close()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(user_id: str, thread_id: str, message: str):
    await mindcore_client.ingest_message({
        "user_id": user_id,
        "thread_id": thread_id,
        "session_id": "web",
        "role": "user",
        "text": message
    })

    context = await mindcore_client.get_context_smart(
        user_id=user_id,
        thread_id=thread_id,
        query=message
    )

    return {"context": context.assembled_context}
```

---

## Cost Comparison

### Token Usage: Traditional vs Mindcore

| Approach | Tokens per Request | Cost per 1000 Requests | Annual (100k requests) |
|----------|-------------------|------------------------|------------------------|
| **Traditional** (full history) | 50,000+ | $130 | $156,000 |
| **Mindcore** (smart context) | ~1,500 | $4 | $4,800 |
| **Mindcore + Local LLM** | ~1,500 | $0 | $0 |

### Why It's Efficient

1. **Metadata extracted once** — Never recomputed
2. **Deterministic retrieval** — No expensive embedding operations
3. **Smart summarization** — Only relevant messages processed
4. **Optional local LLM** — Zero API costs for metadata operations

---

## Configuration

### Environment Variables

```bash
# LLM Provider
export MINDCORE_LLAMA_MODEL_PATH="~/.mindcore/models/model.gguf"
export OPENAI_API_KEY="sk-your-api-key"

# Self-hosted LLM (optional)
export MINDCORE_OPENAI_BASE_URL="http://localhost:8000/v1"
export MINDCORE_OPENAI_MODEL="your-model-name"

# Database (PostgreSQL mode)
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="mindcore"
export DB_USER="postgres"
export DB_PASSWORD="your-password"
```

### Config File (config.yaml)

```yaml
llm:
  provider: auto  # auto, llama_cpp, or openai

  llama_cpp:
    model_path: ${MINDCORE_LLAMA_MODEL_PATH}
    n_ctx: 4096

  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini

database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  database: ${DB_NAME:mindcore}

# Coming soon
summarization:
  enabled: true
  max_age_days: 7
  delete_after_summary: false

preferences:
  enabled: true
  include_in_context: true

connectors:
  cache_ttl: 300
```

---

## CLI Commands

```bash
# Download a model
mindcore download-model                    # Default model
mindcore download-model -m qwen2.5-3b     # Specific model

# List available models
mindcore list-models -v

# Check status
mindcore status

# Show configuration
mindcore config --show
```

---

## REST API

```bash
# Start the server
python -m mindcore.api.server

# Or with custom host/port
python -m mindcore.api.server --host 0.0.0.0 --port 8080
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest a message |
| `/context` | POST | Get assembled context |
| `/context/smart` | POST | Get context with tool calling |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

---

## Roadmap

### Phase 1: Thread Summarization (Next)
- Compress old threads into LLM-generated summaries
- 90% storage reduction for threads > 7 days old
- Background worker for automatic summarization

### Phase 2: User Preferences
- Amendable settings (language, timezone, interests)
- Read-only system data separation (orders, billing never modified)
- Preferences automatically included in context

### Phase 3: External Connectors
- Read-only connectors for Orders, Billing, CRM
- Topic-based routing (mention "orders" → query Orders DB)
- Custom connector SDK for any data source

### Phase 4: Cloud Platform
- Managed Mindcore service
- Dashboard with analytics
- SOC 2 compliance for enterprise

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed technical plans.

---

## Project Structure

```
mindcore/
├── __init__.py              # Main client & public API
├── async_client.py          # AsyncMindcoreClient
│
├── agents/                  # AI agents
│   ├── enrichment_agent.py  # Metadata extraction
│   ├── smart_context_agent.py # Tool-calling context assembly
│   ├── context_assembler_agent.py
│   └── summarization_agent.py  # (planned)
│
├── connectors/              # External data connectors (planned)
│   ├── base.py
│   ├── registry.py
│   └── orders.py
│
├── core/                    # Core functionality
│   ├── schemas.py           # Data models
│   ├── sqlite_manager.py    # SQLite operations
│   ├── async_db.py          # Async database managers
│   ├── cache_manager.py     # In-memory caching
│   └── preferences_manager.py  # (planned)
│
├── llm/                     # LLM providers
│   ├── llama_cpp_provider.py
│   ├── openai_provider.py
│   └── provider_factory.py
│
├── api/                     # REST API
│   └── server.py
│
└── workers/                 # Background workers (planned)
    └── summarization_worker.py

dashboard/                   # Vue.js web dashboard
```

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/mindcore.git
cd mindcore

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mindcore/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **llama.cpp** — CPU-optimized local LLM inference
- **OpenAI** — GPT-4o-mini powers cloud agents
- **FastAPI** — High-performance API framework
- **PostgreSQL/SQLite** — Robust storage options
- **cachetools, limits, structlog** — Battle-tested libraries

---

<div align="center">

### The Context Protocol for AI Agents

**Stop rebuilding memory infrastructure. Start building features.**

```bash
pip install -e ".[llama]" && mindcore download-model && mindcore status
```

[Quick Start](#quick-start) • [Architecture](#-architecture) • [Roadmap](#-roadmap)

---

Made with care by the Mindcore team

</div>
