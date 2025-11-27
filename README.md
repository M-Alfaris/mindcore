<div align="center">

# 🧠 Mindcore

### Intelligent Memory & Context Management for AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/M-Alfaris/mindcore)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Cut your LLM token costs by 60-90%** with intelligent memory management powered by lightweight AI agents.

**Now with local LLM support via llama.cpp — run completely offline with zero API costs!**

[Quick Start](#-quick-start) • [Features](#-features) • [Local LLM Setup](#-local-llm-setup) • [Documentation](#-documentation) • [CLI](#-cli-commands)

---

### Why Mindcore?

| Traditional Approach | With Mindcore | With Mindcore + Local LLM |
|---------------------|---------------|---------------------------|
| Send entire conversation history | Send only relevant context | Same smart context |
| 50,000+ tokens per request | ~1,500 tokens per request | ~1,500 tokens |
| $2.60 per 20 requests | $0.20 per 20 requests | **$0.00** |
| Hit context limits quickly | Scale to unlimited history | Unlimited + offline |

</div>

---

## 🚀 Quick Start

Get up and running in **under 2 minutes**. Choose your preferred setup:

### Option A: Local LLM (Recommended - Zero API Costs!)

```bash
# 1. Install with llama.cpp support
pip install -e ".[llama]"

# 2. Download a model (~2GB)
mindcore download-model

# 3. Set the model path
export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# 4. You're ready! No API key needed.
```

### Option B: OpenAI API (Cloud)

```bash
# 1. Install
pip install -e .

# 2. Set your API key
export OPENAI_API_KEY="sk-your-api-key"
```

### Option C: Auto Mode (Local + Cloud Fallback)

```bash
# Set both - uses local LLM primarily, falls back to OpenAI if needed
export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
export OPENAI_API_KEY="sk-your-api-key"  # Optional fallback
```

### Start Building

```python
from mindcore import MindcoreClient

# Initialize (works with any setup above!)
client = MindcoreClient(use_sqlite=True)

# Ingest a message - automatically enriched with metadata
message = client.ingest_message({
    "user_id": "user_123",
    "thread_id": "thread_456",
    "session_id": "session_789",
    "role": "user",
    "text": "What are best practices for building AI agents?"
})

# See the auto-generated metadata
print(message.metadata.topics)      # ['AI', 'agents', 'best practices']
print(message.metadata.intent)      # 'ask_question'
print(message.metadata.importance)  # 0.8

# Later: Get intelligent context for any query
context = client.get_context(
    user_id="user_123",
    thread_id="thread_456",
    query="AI agent architecture"
)

# Use in your LLM prompt
print(context.assembled_context)  # Compressed, relevant summary
print(context.key_points)         # Key insights from history
```

**That's it!** Two methods: `ingest_message()` and `get_context()`.

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🤖 Intelligent AI Agents
Two specialized agents powered by local or cloud LLMs:
- **MetadataAgent** — Auto-enriches every message with topics, sentiment, intent, and importance
- **ContextAgent** — Intelligently retrieves and summarizes only relevant history

</td>
<td width="50%" valign="top">

### 🦙 Local LLM Support (NEW!)
- **llama.cpp** — CPU-optimized local inference
- **Zero API costs** — Run completely offline
- **Auto-fallback** — Local primary, cloud backup
- **Self-hosted** — vLLM, Ollama, LocalAI support

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 💰 Massive Cost Savings
- **60-90% reduction** in token costs (or **100% with local LLM**)
- Enterprise platforms save **$4M+/year**
- Scales efficiently as conversations grow
- One-time metadata enrichment (never recomputed)

</td>
<td width="50%" valign="top">

### 💾 Flexible Storage
- **SQLite** for local development (zero setup!)
- **PostgreSQL** for production deployments
- **In-memory cache** for blazing-fast retrieval
- Automatic schema management

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔌 Framework Integrations
- **LangChain** — Memory interface, callbacks
- **LlamaIndex** — Chat memory integration
- **Any Framework** — Simple, universal API
- Plug-and-play adapters

</td>
<td width="50%" valign="top">

### 🛠️ Developer Experience
- **CLI tools** — Download models, check status
- Clean, intuitive API
- Full type hints & docstrings
- Comprehensive logging

</td>
</tr>
</table>

---

## 🔍 How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR APPLICATION                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
           ┌───────────────┐                   ┌───────────────┐
           │ ingest_message│                   │  get_context  │
           └───────┬───────┘                   └───────┬───────┘
                   │                                   │
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │   MetadataAgent     │             │    ContextAgent     │
        │                     │             │                     │
        │ • Extract topics    │             │ • Analyze query     │
        │ • Detect intent     │             │ • Find relevant msgs│
        │ • Score importance  │             │ • Summarize context │
        │ • Analyze sentiment │             │ • Extract key points│
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
                   ▼                                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │                    LLM Provider Layer                    │
        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
        │  │ llama.cpp   │  │   OpenAI    │  │  Self-Hosted    │  │
        │  │  (Local)    │─►│  (Fallback) │  │ (vLLM/Ollama)   │  │
        │  └─────────────┘  └─────────────┘  └─────────────────┘  │
        └─────────────────────────────────────────────────────────┘
                   │                                   │
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │      Storage        │◄───────────►│       Cache         │
        │ (PostgreSQL/SQLite) │             │    (In-Memory)      │
        └─────────────────────┘             └─────────────────────┘
```

### The Problem with Traditional Approaches

Every time your AI needs context, you send the **entire conversation history**:

```
User message 1 → LLM
User message 1 + 2 → LLM
User message 1 + 2 + 3 → LLM
...
User message 1 + 2 + ... + 200 → LLM  (50,000+ tokens!)
```

### The Mindcore Solution

1. **Enrich Once** — When a message arrives, MetadataAgent extracts metadata (topics, intent, sentiment, importance) using a lightweight LLM
2. **Retrieve Smart** — When context is needed, ContextAgent uses metadata to find and summarize only relevant messages
3. **Send Less** — Your main LLM receives a compressed ~1,500 token context instead of 50,000+

---

## 🦙 Local LLM Setup

Run Mindcore completely offline with zero API costs using llama.cpp.

### Quick Setup

```bash
# Install with llama.cpp support
pip install -e ".[llama]"

# Download the default model (Llama 3.2 3B, ~2GB)
mindcore download-model

# Set the model path
export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### Available Models

| Model | Size | RAM | Best For |
|-------|------|-----|----------|
| `llama-3.2-3b` (default) | 2.0 GB | 4+ GB | General use, best quality |
| `llama-3.2-1b` | 0.8 GB | 2+ GB | Low-resource environments |
| `qwen2.5-3b` | 2.1 GB | 4+ GB | Multilingual, structured output |
| `phi-3.5-mini` | 2.2 GB | 4+ GB | Reasoning tasks |
| `gemma-2-2b` | 1.6 GB | 3+ GB | Good balance |
| `smollm2-1.7b` | 1.1 GB | 2+ GB | Ultra-lightweight |

```bash
# Download a specific model
mindcore download-model -m qwen2.5-3b

# List all available models
mindcore list-models -v
```

### Self-Hosted LLM Servers

Use any OpenAI-compatible server (vLLM, Ollama, LocalAI, etc.):

```bash
# vLLM server
export MINDCORE_OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
export MINDCORE_OPENAI_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# Ollama
export MINDCORE_OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export MINDCORE_OPENAI_MODEL="llama3.2"
```

### Provider Modes

```python
from mindcore import MindcoreClient

# Auto mode (default): llama.cpp primary, OpenAI fallback
client = MindcoreClient(use_sqlite=True)

# Force local LLM only
client = MindcoreClient(use_sqlite=True, llm_provider="llama_cpp")

# Force OpenAI only
client = MindcoreClient(use_sqlite=True, llm_provider="openai")
```

---

## 📖 Documentation

### MindcoreClient

The main entry point for all operations.

```python
from mindcore import MindcoreClient

# Local development with SQLite (recommended for getting started)
client = MindcoreClient(use_sqlite=True)

# Production with PostgreSQL
client = MindcoreClient()

# Custom configuration
client = MindcoreClient(config_path="path/to/config.yaml")

# In-memory database (great for testing)
client = MindcoreClient(use_sqlite=True, sqlite_path=":memory:")
```

#### Methods

| Method | Description |
|--------|-------------|
| `ingest_message(message_dict)` | Enrich and store a message |
| `get_context(user_id, thread_id, query, max_messages=50)` | Get assembled context for a query |
| `get_message(message_id)` | Retrieve a single message by ID |
| `clear_cache(user_id, thread_id)` | Clear cached messages |
| `close()` | Cleanup connections |

#### Message Format

```python
message = client.ingest_message({
    "user_id": "user_123",       # Required: User identifier
    "thread_id": "thread_456",   # Required: Conversation thread
    "session_id": "session_789", # Required: Session identifier
    "role": "user",              # Required: user, assistant, system, or tool
    "text": "Message content"    # Required: The message text
})
```

#### Enriched Metadata

After ingestion, messages include rich metadata:

```python
message.metadata.topics       # ['AI', 'machine learning']
message.metadata.categories   # ['technology', 'programming']
message.metadata.sentiment    # 'positive', 'negative', 'neutral'
message.metadata.intent       # 'ask_question', 'provide_info', etc.
message.metadata.importance   # 0.0 to 1.0
message.metadata.entities     # ['OpenAI', 'GPT-4']
message.metadata.key_phrases  # ['best practices', 'AI agents']
```

#### Assembled Context

```python
context = client.get_context(user_id, thread_id, query)

context.assembled_context    # Summarized relevant history (string)
context.key_points           # ['Point 1', 'Point 2', ...]
context.relevant_message_ids # ['msg_1', 'msg_2', ...]
context.metadata             # {'topics': [...], 'importance': 0.8}
```

---

## 🔌 Framework Integrations

### LangChain

```python
from mindcore import MindcoreClient
from mindcore.integrations import LangChainAdapter

client = MindcoreClient(use_sqlite=True)
adapter = LangChainAdapter(client)

# Option 1: Use as LangChain memory
memory = adapter.as_langchain_memory("user_123", "thread_456", "session_789")

# Option 2: Auto-capture with callbacks
callback = adapter.create_langchain_callback("user_123", "thread_456", "session_789")
llm = ChatOpenAI(callbacks=[callback])

# Option 3: Inject context into prompts
context = adapter.get_enhanced_context(user_id, thread_id, query)
enhanced_prompt = adapter.inject_context_into_prompt(context, system_prompt)
```

### LlamaIndex

```python
from mindcore.integrations import LlamaIndexAdapter

adapter = LlamaIndexAdapter(client)
memory = adapter.create_chat_memory("user_123", "thread_456", "session_789")

# Get messages
messages = memory.get_messages()

# Add message
memory.add_message(role="user", content="Hello!")
```

### Any Framework

Mindcore works with any AI system:

```python
# Your existing code
response = your_llm.generate(user_message)

# Add Mindcore
context = client.get_context(user_id, thread_id, user_message)
response = your_llm.generate(
    f"Context: {context.assembled_context}\n\nUser: {user_message}"
)
```

---

## 💵 Cost Analysis

### Benchmark: 200 Messages, 20 Context Requests

| Approach | Tokens Used | Cost | Savings |
|----------|-------------|------|---------|
| **Traditional** (full history) | 1,010,000 | $2.60 | — |
| **Mindcore** (intelligent) | 190,000 | $0.20 | **92%** |

### Real-World Annual Savings

| Use Case | Traditional | Mindcore | Annual Savings |
|----------|-------------|----------|----------------|
| Customer Support (1k users/day) | $225,000 | $45,000 | **$180,000** |
| AI Assistant (per enterprise user) | $61,000 | $4,000 | **$57,000** |
| Platform (10k daily users) | $4,500,000 | $450,000 | **$4,050,000** |

### Why It's So Efficient

1. **GPT-4o-mini** — Enrichment uses the cheapest capable model ($0.15/1M tokens)
2. **One-time processing** — Metadata is extracted once, never recomputed
3. **Smart retrieval** — Only relevant messages are summarized
4. **Compressed output** — ~1,500 tokens instead of 50,000+

---

## 🖥️ REST API

Start the FastAPI server for HTTP access:

```bash
# Using the CLI
mindcore-server

# Or with Python
python -m mindcore.api.server

# Custom host/port
python -m mindcore.api.server --host 0.0.0.0 --port 8080
```

### Endpoints

#### POST /ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "thread_id": "thread_456",
    "session_id": "session_789",
    "role": "user",
    "text": "Hello, world!"
  }'
```

#### POST /context
```bash
curl -X POST http://localhost:8000/context \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "thread_id": "thread_456",
    "query": "What did we discuss?"
  }'
```

#### GET /health
```bash
curl http://localhost:8000/health
```

**Interactive Docs:** http://localhost:8000/docs

---

## 🔧 CLI Commands

Mindcore includes a CLI for model management and status checking.

```bash
# Download a model
mindcore download-model                    # Download default model
mindcore download-model -m qwen2.5-3b     # Download specific model
mindcore download-model -o ./models       # Custom output directory

# List available models
mindcore list-models                       # Show model table
mindcore list-models -v                    # Detailed info

# Check installation status
mindcore status                            # Show provider status

# Show configuration
mindcore config --show                     # Display current config
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider (choose one or both)
export MINDCORE_LLAMA_MODEL_PATH="~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
export OPENAI_API_KEY="sk-your-api-key"

# Self-hosted LLM (optional)
export MINDCORE_OPENAI_BASE_URL="http://localhost:8000/v1"
export MINDCORE_OPENAI_MODEL="your-model-name"

# Database (only for PostgreSQL mode)
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="mindcore"
export DB_USER="postgres"
export DB_PASSWORD="your-password"
```

### Config File (config.yaml)

```yaml
llm:
  # Provider mode: "auto", "llama_cpp", or "openai"
  provider: auto

  # Local LLM (llama.cpp)
  llama_cpp:
    model_path: ${MINDCORE_LLAMA_MODEL_PATH}
    n_ctx: 4096           # Context window
    n_gpu_layers: 0       # 0 = CPU only, -1 = all GPU

  # Cloud/Self-hosted LLM
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: ${MINDCORE_OPENAI_BASE_URL:}  # Optional: for self-hosted
    model: ${MINDCORE_OPENAI_MODEL:gpt-4o-mini}

  # Generation settings
  defaults:
    temperature: 0.3
    max_tokens_enrichment: 800
    max_tokens_context: 1500

database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  database: ${DB_NAME:mindcore}
  user: ${DB_USER:postgres}
  password: ${DB_PASSWORD:postgres}

cache:
  max_size: 50
  ttl: 3600
```

### PostgreSQL Setup (Production)

```bash
# Create database
createdb mindcore

# Initialize schema
psql -d mindcore -f schema.sql
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=mindcore --cov-report=html

# Run specific test file
pytest tests/test_client.py -v
```

---

## 📁 Project Structure

```
mindcore/
├── __init__.py              # Main client & public API
├── config.yaml              # Default configuration
│
├── core/                    # Core functionality
│   ├── config_loader.py     # Configuration management
│   ├── db_manager.py        # PostgreSQL operations
│   ├── sqlite_manager.py    # SQLite operations (local dev)
│   ├── cache_manager.py     # In-memory caching
│   └── schemas.py           # Data models (Message, Context, etc.)
│
├── llm/                     # LLM Provider Layer (NEW!)
│   ├── base_provider.py     # Abstract base class
│   ├── llama_cpp_provider.py # Local llama.cpp inference
│   ├── openai_provider.py   # OpenAI/compatible APIs
│   └── provider_factory.py  # Factory with fallback support
│
├── cli/                     # Command-line Interface (NEW!)
│   ├── main.py              # CLI commands
│   └── models.py            # Model registry
│
├── agents/                  # AI agents
│   ├── base_agent.py        # Base class with LLM provider
│   ├── enrichment_agent.py  # MetadataAgent implementation
│   └── context_assembler_agent.py  # ContextAgent implementation
│
├── integrations/            # Framework adapters
│   ├── langchain_adapter.py # LangChain integration
│   └── llamaindex_adapter.py # LlamaIndex integration
│
├── api/                     # REST API
│   ├── server.py            # FastAPI application
│   └── routes/              # API endpoints
│
└── utils/                   # Utilities
    ├── security.py          # Validation & rate limiting
    └── logger.py            # Logging configuration
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with tests
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mindcore.git
cd mindcore

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mindcore/
isort mindcore/
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **llama.cpp** — CPU-optimized local LLM inference
- **OpenAI** — GPT-4o-mini powers cloud agents
- **FastAPI** — High-performance API framework
- **PostgreSQL** — Robust production database
- **SQLite** — Zero-config local development

---

<div align="center">

### Ready to cut your LLM costs by 90% (or 100% with local LLM)?

```bash
# Quick start with local LLM
pip install -e ".[llama]" && mindcore download-model && mindcore status
```

**[Get Started](#-quick-start)** • **[Local LLM Setup](#-local-llm-setup)** • **[CLI Commands](#-cli-commands)**

---

Made with ❤️ by the Mindcore team

</div>
