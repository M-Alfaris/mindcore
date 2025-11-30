# Mindcore Configuration Guide

This directory contains example configuration files for Mindcore. Copy these files, remove the `.example` suffix, and customize for your environment.

## Quick Start

```bash
# Copy example files
cp mindcore.example.yaml mindcore.yaml
cp connectors.example.yaml connectors.yaml
cp agents.example.yaml agents.yaml
cp .env.example .env

# Edit .env with your API keys
nano .env
```

## Configuration Files

| File | Purpose |
|------|---------|
| `mindcore.yaml` | Core application settings (database, cache, LLM, API server) |
| `connectors.yaml` | External data connector configurations |
| `vectorstores.yaml` | Vector database settings (Chroma, Pinecone, pgvector) |
| `agents.yaml` | AI agent behavior and tool settings |
| `.env` | Environment variables and secrets |

## Environment Variables

**Important**: Never commit `.env` to version control!

### Required Variables

```bash
# At minimum, you need an LLM provider
OPENAI_API_KEY=sk-your-openai-api-key
```

### Optional Variables

```bash
# Database (defaults to SQLite)
MINDCORE_DB_PATH=./data/mindcore.db

# Or PostgreSQL for production
MINDCORE_DB_URL=postgresql://user:pass@localhost:5432/mindcore

# External connectors (use READ-ONLY credentials!)
ORDERS_DB_URL=postgresql://readonly:pass@orders-db:5432/orders
STRIPE_API_KEY=rk_live_your-restricted-key
```

## Configuring External Connectors

Connectors provide **read-only** access to your business systems. They enable the AI to answer questions about orders, billing, support tickets, etc.

### Orders Connector

```yaml
# connectors.yaml
orders:
  enabled: true
  backend: database
  database:
    url: ${ORDERS_DB_URL}
    table: orders
    user_id_column: customer_id
  max_results: 10
  cache_ttl: 300
```

### Billing Connector (Stripe)

```yaml
# connectors.yaml
billing:
  enabled: true
  backend: stripe
  stripe:
    api_key: ${STRIPE_API_KEY}
    customer_id_field: stripe_customer_id
```

### Custom Connector

```yaml
# connectors.yaml
my_connector:
  enabled: true
  topics:
    - my_topic
    - related_topic
  backend: api
  api:
    url: https://api.example.com/data
    key: ${MY_API_KEY}
```

## Using Configuration in Code

### Loading Configuration

```python
import yaml
import os
from pathlib import Path

def load_config(name: str) -> dict:
    """Load a YAML config file."""
    config_dir = Path(__file__).parent / "config"
    config_path = config_dir / f"{name}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    return expand_env_vars(config)

def expand_env_vars(obj):
    """Recursively expand ${VAR} patterns."""
    if isinstance(obj, str):
        import re
        pattern = r'\$\{(\w+)\}'
        def replace(match):
            return os.environ.get(match.group(1), match.group(0))
        return re.sub(pattern, replace, obj)
    elif isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(item) for item in obj]
    return obj
```

### Setting Up Connectors

```python
from mindcore.connectors import (
    ConnectorRegistry,
    OrdersConnector,
    BillingConnector
)

# Load connector config
config = load_config("connectors")

# Create registry
registry = ConnectorRegistry()

# Register Orders connector
if config.get("orders", {}).get("enabled"):
    orders_config = config["orders"]
    registry.register(OrdersConnector(
        db_url=orders_config["database"]["url"],
        user_id_column=orders_config["database"]["user_id_column"],
        max_results=orders_config.get("max_results", 10)
    ))

# Register Billing connector
if config.get("billing", {}).get("enabled"):
    billing_config = config["billing"]
    registry.register(BillingConnector(
        stripe_api_key=billing_config["stripe"]["api_key"]
    ))

# Use in your agent
results = await registry.lookup(
    user_id="user123",
    topics=["orders", "billing"],
    context={"order_id": "ORD-12345"}
)
```

### Setting Up SmartContextAgent

```python
from mindcore.agents import SmartContextAgent, ContextTools
from mindcore.connectors import ConnectorRegistry

# Create tools with connector support
def create_tools(registry: ConnectorRegistry) -> ContextTools:
    return ContextTools(
        get_user_context=lambda uid: get_user_from_db(uid),
        get_conversation_history=lambda uid, limit: get_history(uid, limit),
        search_knowledge_base=lambda q, k: search_kb(q, k),
        lookup_external_data=lambda uid, topics, ctx: registry.lookup(uid, topics, ctx),
        extract_entities=lambda text, topics: registry.extract_entities_for_topics(text, topics)
    )

# Initialize agent
agent = SmartContextAgent(
    llm_client=your_llm_client,
    tools=create_tools(registry)
)
```

## Security Best Practices

### 1. Use Read-Only Credentials

External connectors should **never** have write access:

```bash
# PostgreSQL: Create read-only user
CREATE USER readonly WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE orders TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
```

### 2. Use Restricted API Keys

For Stripe, create a restricted key with only read permissions:
- Go to Stripe Dashboard → Developers → API Keys
- Create Restricted Key
- Enable only: `Charges: Read`, `Invoices: Read`, `Subscriptions: Read`

### 3. Never Commit Secrets

```bash
# .gitignore
.env
*.yaml
!*.example.yaml
config/mindcore.yaml
config/connectors.yaml
```

### 4. Use Environment Variables

```yaml
# Good - reference environment variable
api_key: ${STRIPE_API_KEY}

# Bad - hardcoded secret
api_key: sk_live_abc123
```

## Configuration Precedence

1. Environment variables (highest priority)
2. YAML config files
3. Default values in code (lowest priority)

## Troubleshooting

### "Connector not found"

Ensure the connector is enabled and registered:

```yaml
orders:
  enabled: true  # Must be true
```

### "Database connection failed"

Check your connection URL format:

```bash
# SQLite
MINDCORE_DB_PATH=./data/mindcore.db

# PostgreSQL
MINDCORE_DB_URL=postgresql://user:pass@host:5432/dbname
```

### "API key invalid"

Verify environment variable is set:

```bash
echo $OPENAI_API_KEY  # Should print your key
```

### "Cache miss on every request"

Check cache settings:

```yaml
cache:
  type: memory
  default_ttl: 300  # 5 minutes
```

## Modular Context Layer

Mindcore uses a modular architecture inspired by LangChain. You can choose which components to enable based on your needs.

### Configuration Tiers

| Tier | Components | Use Case |
|------|------------|----------|
| **Basic** | Messages + Cache | Simple chatbots, prototypes |
| **Standard** | + External Connectors | Customer support bots with order/billing lookup |
| **Advanced** | + Vector Store | RAG applications, semantic search |
| **Full** | All features | Production applications with all capabilities |

### Basic Setup (Messages Only)

```python
from mindcore import ContextLayer

# Simplest setup - just message history
layer = ContextLayer.basic(sqlite_path="mindcore.db")
```

### With Vector Store (Semantic Search)

```python
from mindcore import ContextLayer

# Add semantic search capabilities
layer = ContextLayer.with_vector_store(
    vector_store_type="chroma",
    embedding_provider="openai",  # or "sentence_transformers" for local
    vector_store_config={
        "collection_name": "my_docs",
        "persist_directory": "./data/chroma"
    }
)
```

### Full Setup (All Features)

```python
from mindcore import ContextLayer

layer = ContextLayer.full(
    vector_store_type="pinecone",
    vector_store_config={
        "api_key": os.environ["PINECONE_API_KEY"],
        "index_name": "mindcore"
    }
)

# Register external connectors
from mindcore.connectors import OrdersConnector, BillingConnector
layer.register_connector(OrdersConnector(db_url=...))
layer.register_connector(BillingConnector(stripe_api_key=...))
```

## Vector Stores Configuration

Vector stores enable semantic search - finding content by meaning rather than keywords.

### Choosing a Vector Store

| Store | Best For | Dependencies |
|-------|----------|--------------|
| **In-Memory** | Development, testing | None |
| **Chroma** | Local dev, small-medium scale | `pip install chromadb` |
| **Pinecone** | Production, cloud-native | `pip install pinecone-client` |
| **pgvector** | PostgreSQL users, hybrid queries | `pip install 'psycopg[binary]' pgvector` |

### Chroma (Recommended for Development)

```yaml
# vectorstores.yaml
vector_store:
  enabled: true
  type: chroma

chroma:
  collection_name: mindcore_vectors
  persist_directory: ./data/chroma_db
```

```python
from mindcore.vectorstores import ChromaVectorStore, OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = ChromaVectorStore(
    collection_name="my_docs",
    embedding=embeddings,
    persist_directory="./data/chroma"
)
```

### Pinecone (Recommended for Production)

```yaml
# vectorstores.yaml
vector_store:
  enabled: true
  type: pinecone

pinecone:
  api_key: ${PINECONE_API_KEY}
  index_name: mindcore
```

```python
from mindcore.vectorstores import PineconeVectorStore, OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = PineconeVectorStore(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name="mindcore",
    embedding=embeddings
)
```

### pgvector (For PostgreSQL Users)

```yaml
# vectorstores.yaml
vector_store:
  enabled: true
  type: pgvector

pgvector:
  connection_string: ${PGVECTOR_CONNECTION_STRING}
  collection_name: mindcore_vectors
```

```python
from mindcore.vectorstores import PGVectorStore, OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = PGVectorStore(
    connection_string="postgresql://user:pass@localhost/db",
    embedding=embeddings
)
```

## Embedding Providers

Embeddings convert text into vectors for semantic search.

### OpenAI (Best Quality)

```yaml
embeddings:
  provider: openai
  openai:
    api_key: ${OPENAI_API_KEY}
    model: text-embedding-3-small  # 1536 dimensions
```

### Sentence Transformers (Local, No API)

```yaml
embeddings:
  provider: sentence_transformers
  sentence_transformers:
    model_name: all-MiniLM-L6-v2  # 384 dimensions, fast
```

```python
from mindcore.vectorstores import SentenceTransformerEmbeddings

# No API key needed!
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

### Ollama (Local with More Models)

```yaml
embeddings:
  provider: ollama
  ollama:
    model: nomic-embed-text
    base_url: http://localhost:11434
```

## Need Help?

- Check the [Mindcore Documentation](../docs/)
- Review connector source code in `mindcore/connectors/`
- Review vector store source code in `mindcore/vectorstores/`
- Open an issue on GitHub
