# 🧠 Mindcore

**Intelligent Memory and Context Management for AI Agents**

Mindcore is a lightweight, open-source Python framework that provides intelligent memory and context management for AI agents. It uses two specialized AI agents powered by GPT-4o-mini to enrich conversations with metadata and retrieve relevant historical context.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/yourusername/mindcore)

---

## 🚀 Features

- **🤖 Two Lightweight AI Agents**
  - **Metadata Enrichment Agent**: Automatically enriches messages with topics, categories, sentiment, intent, and more
  - **Context Assembly Agent**: Intelligently retrieves and summarizes relevant historical context

- **💾 Dual-Layer Storage**
  - PostgreSQL for persistent message storage
  - In-memory cache for fast access to recent messages

- **🌐 FastAPI REST API**
  - `/ingest` - Ingest messages for enrichment and storage
  - `/context` - Retrieve assembled context for queries

- **🔌 Easy Integration**
  - Works with LangChain, LlamaIndex, custom AI systems, or standalone
  - Framework adapters for seamless integration
  - JSON-based input/output for universal compatibility
  - Pip-installable package

- **🔒 Production-Ready Security**
  - SQL injection protection with parameterized queries
  - Input validation and sanitization
  - Rate limiting support
  - Security headers and best practices

- **💰 Cost Efficient**
  - **Saves 60-90% on token costs** vs traditional approaches
  - Uses cheap GPT-4o-mini for enrichment/assembly
  - Only sends compressed context to main LLM
  - See [COST_EFFICIENCY.md](COST_EFFICIENCY.md) for detailed analysis

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- OpenAI API key

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/mindcore.git
cd mindcore

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install from PyPI (coming soon)

```bash
pip install mindcore
```

---

## 🔧 Configuration

### 1. Set up PostgreSQL

Create a PostgreSQL database:

```sql
CREATE DATABASE mindcore;
```

### 2. Configure environment variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="mindcore"
export DB_USER="postgres"
export DB_PASSWORD="your-password"
```

### 3. Edit config.yaml (optional)

You can customize the configuration by editing `mindcore/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  database: mindcore
  user: postgres
  password: postgres

openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  temperature: 0.3
  max_tokens: 1000

cache:
  max_size: 50
  ttl: 3600

api:
  host: 0.0.0.0
  port: 8000
  debug: false
```

---

## 🎯 Quick Start

### Python API

```python
from mindcore import Mindcore

# Initialize Mindcore
mindcore = Mindcore()

# Ingest a message (automatically enriched with metadata)
message = mindcore.ingest_message({
    "user_id": "user_123",
    "thread_id": "conversation_456",
    "session_id": "session_789",
    "role": "user",
    "text": "What are the best practices for building AI agents?"
})

print(f"Message ID: {message.message_id}")
print(f"Topics: {message.metadata.topics}")
print(f"Intent: {message.metadata.intent}")
print(f"Importance: {message.metadata.importance}")

# Get assembled context for a query
context = mindcore.get_context(
    user_id="user_123",
    thread_id="conversation_456",
    query="AI agent best practices"
)

print(f"Context: {context.assembled_context}")
print(f"Key Points: {context.key_points}")
```

### FastAPI Server

Start the API server:

```bash
# Using the CLI
mindcore-server

# Or using Python
python -m mindcore.api.server
```

### API Endpoints

**Ingest a message:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "thread_id": "thread_456",
    "session_id": "session_789",
    "role": "user",
    "text": "How do I build a chatbot?"
  }'
```

**Get context:**

```bash
curl -X POST http://localhost:8000/context \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "thread_id": "thread_456",
    "query": "chatbot development",
    "max_messages": 50
  }'
```

---

## 🏗️ Architecture

```
mindcore/
├── __init__.py           # Main Mindcore class
├── config.yaml           # Configuration file
├── core/                 # Core functionality
│   ├── config_loader.py  # Configuration management
│   ├── db_manager.py     # PostgreSQL database operations
│   ├── cache_manager.py  # In-memory cache
│   └── schemas.py        # Data models and schemas
├── agents/               # AI agents
│   ├── base_agent.py     # Base agent class
│   ├── enrichment_agent.py      # Metadata enrichment
│   └── context_assembler_agent.py  # Context assembly
├── api/                  # FastAPI server
│   ├── server.py         # FastAPI application
│   └── routes/           # API routes
│       ├── ingest.py
│       └── context.py
└── utils/                # Utilities
    ├── logger.py         # Logging
    ├── tokenizer.py      # Text processing
    └── helper.py         # Helper functions
```

---

## 📊 Database Schema

Mindcore automatically creates the following PostgreSQL schema:

```sql
CREATE TABLE messages (
    message_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    raw_text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_thread ON messages(user_id, thread_id);
CREATE INDEX idx_thread_created ON messages(thread_id, created_at DESC);
CREATE INDEX idx_metadata_topics ON messages USING GIN ((metadata->'topics'));
CREATE INDEX idx_created_at ON messages(created_at DESC);
```

---

## 🔍 How It Works

### 1. Metadata Enrichment Agent

When you ingest a message, the Enrichment Agent:

1. Analyzes the message content
2. Uses GPT-4o-mini to extract structured metadata:
   - **Topics**: Main subjects discussed
   - **Categories**: Message type (question, statement, code, etc.)
   - **Importance**: Relevance score (0.0-1.0)
   - **Sentiment**: Positive/negative/neutral with score
   - **Intent**: Primary intent (ask_question, provide_info, etc.)
   - **Tags**: Relevant keywords
   - **Entities**: Named entities (people, places, technologies)
   - **Key Phrases**: Important phrases
3. Stores enriched message in PostgreSQL and cache

### 2. Context Assembly Agent

When you request context, the Context Assembler:

1. Retrieves recent messages from cache and database
2. Uses GPT-4o-mini to:
   - Analyze message history
   - Identify relevant messages for the query
   - Summarize key information
   - Extract key points
3. Returns structured JSON ready for LLM prompt injection

---

## 🎨 Use Cases

- **Chatbots**: Maintain conversation context across sessions
- **Customer Support**: Retrieve relevant past interactions
- **AI Assistants**: Build agents with long-term memory
- **Documentation**: Auto-tag and categorize content
- **Analytics**: Extract insights from conversation data

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mindcore --cov-report=html

# Run specific test file
pytest mindcore/tests/test_enrichment.py
```

---

## 🛠️ Development

### Set up development environment

```bash
# Clone the repository
git clone https://github.com/yourusername/mindcore.git
cd mindcore

# Install with dev dependencies
pip install -e ".[dev]"

# Run code formatting
black mindcore/
isort mindcore/

# Run linting
flake8 mindcore/

# Run type checking
mypy mindcore/
```

---

## 📝 Example: LangChain Integration

```python
from mindcore import Mindcore
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize
mindcore = Mindcore()
llm = ChatOpenAI()

user_id = "user_123"
thread_id = "thread_456"
session_id = "session_789"

# User message
user_message = "What were we discussing about AI agents?"

# Ingest user message
mindcore.ingest_message({
    "user_id": user_id,
    "thread_id": thread_id,
    "session_id": session_id,
    "role": "user",
    "text": user_message
})

# Get historical context
context = mindcore.get_context(
    user_id=user_id,
    thread_id=thread_id,
    query=user_message
)

# Build prompt with context
messages = [
    SystemMessage(content=f"""You are a helpful assistant.

Historical Context:
{context.assembled_context}

Key Points:
{chr(10).join(f"- {point}" for point in context.key_points)}
"""),
    HumanMessage(content=user_message)
]

# Get LLM response
response = llm(messages)

# Ingest assistant response
mindcore.ingest_message({
    "user_id": user_id,
    "thread_id": thread_id,
    "session_id": session_id,
    "role": "assistant",
    "text": response.content
})

print(response.content)
```

---

## 🔌 Framework Adapters

Mindcore provides plug-and-play adapters for popular AI frameworks:

### LangChain Adapter

```python
from mindcore import Mindcore
from mindcore.adapters import LangChainAdapter
from langchain.schema import HumanMessage, AIMessage

mindcore = Mindcore()
adapter = LangChainAdapter(mindcore)

# Automatically ingest LangChain messages
messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there!")
]

adapter.ingest_langchain_conversation(
    messages=messages,
    user_id="user123",
    thread_id="thread456",
    session_id="session789"
)

# Use as LangChain memory
memory = adapter.as_langchain_memory("user123", "thread456", "session789")

# Or create automatic callback
callback = adapter.create_langchain_callback("user123", "thread456", "session789")
llm = ChatOpenAI(callbacks=[callback])  # Auto-ingest all messages
```

### LlamaIndex Adapter

```python
from mindcore.adapters import LlamaIndexAdapter

adapter = LlamaIndexAdapter(mindcore)

# Create chat memory
memory = adapter.create_chat_memory("user123", "thread456", "session789")

# Get messages
messages = memory.get_messages()

# Add message
memory.add_message(role="user", content="Hello!")
```

### Custom AI Systems

```python
# Direct integration - works with any system
message = mindcore.ingest_message({
    "user_id": "user123",
    "thread_id": "thread456",
    "session_id": "session789",
    "role": "user",
    "text": "Your message here"
})

# Get context
context = mindcore.get_context(
    user_id="user123",
    thread_id="thread456",
    query="your query"
)

# Use context.assembled_context in your prompts
```

See `examples_adapters.py` for complete examples.

---

## 🔒 Security

Mindcore implements production-grade security:

### Built-in Protections

- ✅ **SQL Injection Protection** - Parameterized queries throughout
- ✅ **Input Validation** - Strict validation of all user inputs
- ✅ **Rate Limiting** - Configurable rate limiting support
- ✅ **Sanitization** - Text sanitization and length limits
- ✅ **Security Headers** - Recommended security headers for API

### Using Security Features

```python
from mindcore.utils import SecurityValidator, get_rate_limiter

# Validate messages automatically (done by Mindcore)
is_valid, error = SecurityValidator.validate_message_dict(message_dict)

# Rate limiting
rate_limiter = get_rate_limiter()
if not rate_limiter.is_allowed(user_id):
    raise Exception("Rate limit exceeded")

# Get remaining requests
remaining = rate_limiter.get_remaining(user_id)
```

### Security Best Practices

```python
# Use environment variables for secrets
export OPENAI_API_KEY="your-key"
export DB_PASSWORD="your-password"

# Enable SSL for database
# Use HTTPS for API endpoints
# Implement authentication for production
```

See [SECURITY.md](SECURITY.md) for comprehensive security documentation.

---

## 💰 Cost Efficiency

Mindcore saves **60-90% on token costs** compared to traditional memory management:

### Why Mindcore is Cheaper

| Approach | Cost for 200 msgs, 20 requests |
|----------|-------------------------------|
| **Traditional** (Full history every time) | $2.60 |
| **Mindcore** (Intelligent compression) | $0.20 |
| **Savings** | **92%** |

### How It Works

1. **Enrichment uses GPT-4o-mini** ($0.15/1M tokens vs $2.50/1M for GPT-4o)
2. **Context assembly uses GPT-4o-mini** (cheap summarization)
3. **Main LLM gets compressed context** (1.5k tokens vs 50k+ full history)
4. **One-time enrichment** (metadata never recomputed)

### Run Your Own Benchmark

```python
from mindcore.utils.cost_analysis import run_cost_benchmark

report = run_cost_benchmark()
print(report)
```

See [COST_EFFICIENCY.md](COST_EFFICIENCY.md) for detailed cost analysis and ROI calculations.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [OpenAI](https://openai.com/) GPT-4o-mini
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- Database: [PostgreSQL](https://www.postgresql.org/)

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Happy Building! 🚀**
