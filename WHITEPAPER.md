# MindCore: A Structured Augmented Generation Engine for AI Agents

## A Technical White Paper

**Version 1.0 | December 2025**

**Authors**: MindCore Project Contributors

---

## Abstract

MindCore is an open-source Structured Augmented Generation Engine (SAGE) - a deterministic alternative to Retrieval-Augmented Generation (RAG) for AI agent systems. Where RAG depends on vector embeddings and similarity search for fuzzy, unexplainable retrieval, MindCore generates enriched metadata dimensions for every piece of content and uses PostgreSQL queries on those dimensions for deterministic, explainable retrieval.

Every message - inbound and outbound - gets structured metadata extracted by the agent's LLM: message type, category, topic, intent, sentiment, section, chapter, and more. Retrieval matches the metadata of the user's prompt against the metadata of stored content. Results are ranked by metadata match quality, not vector cosine similarity. Outbound agent responses are also stored with metadata and versioning for future reference.

MindCore introduces three foundational protocols: **FLR** (Fast Learning Recall) for hot-path deterministic cache access, **CLST** (Cognitive Long-term Storage Transfer) for PostgreSQL-centric persistent storage with session aggregation, and **SVL** (Structured Validation Layer) for LLM-enforced metadata extraction and validation. Together, these protocols enable deterministic, explainable, and cost-effective retrieval that replaces RAG's vector-based approach.

The framework achieves <160ms context assembly without embeddings through structured metadata queries, provides deterministic and traceable operations via controlled vocabulary and structured outputs, and includes enterprise-grade features for audit logging, encryption, compliance, and multi-agent federation. MindCore works with any LLM provider (OpenAI, Anthropic, Google, local models) and is MIT licensed with no vendor lock-in.

---

<div align="center">

*"Structured Metadata Replaces Vector Embeddings"*

**The Open-Source SAGE That Replaces RAG for AI Agents**

</div>

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | December 2025 | Initial release with FLR, CLST, SVL protocols |
| 0.9.0 | November 2025 | Beta release, enterprise features |
| 0.5.0 | October 2025 | Alpha release, core protocols |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem: Memory Fragmentation in AI Systems](#2-the-problem-memory-fragmentation-in-ai-systems)
3. [The MindCore Solution](#3-the-mindcore-solution)
4. [The Three Foundational Protocols](#4-the-three-foundational-protocols)
5. [Build It Once, Deploy It Endlessly](#5-build-it-once-deploy-it-endlessly)
6. [Universal LLM Compatibility](#6-universal-llm-compatibility)
7. [Structured Output & Metadata Enrichment](#7-structured-output--metadata-enrichment)
8. [Hierarchical Retrieval Architecture](#8-hierarchical-retrieval-architecture)
9. [Multi-Agent Federation](#9-multi-agent-federation)
10. [Enterprise-Grade Features](#10-enterprise-grade-features)
11. [Security Considerations](#11-security-considerations)
12. [Performance, Reliability & Determinism](#12-performance-reliability--determinism)
13. [Failure Handling Strategies](#13-failure-handling-strategies)
14. [Real-World Examples & Use Cases](#14-real-world-examples--use-cases)
15. [Conclusion](#15-conclusion)

**Appendices**

- [Appendix A: API Quick Reference](#appendix-a-api-quick-reference)
- [Appendix B: Configuration Reference](#appendix-b-configuration-reference)
- [Appendix C: Glossary](#appendix-c-glossary)
- [Appendix D: References & Further Reading](#appendix-d-references--further-reading)

---

## 1. Executive Summary

**MindCore** is an open-source memory protocol stack that provides a universal, standardized approach to memory management for AI agents. Just as the Model Context Protocol (MCP) standardized tool connections for LLMs, MindCore standardizes how AI agents store, retrieve, learn from, and share memories.

### Key Value Propositions

| Capability | Benefit |
|------------|---------|
| **Universal Protocol** | Works with any LLM provider (OpenAI, Anthropic, Google, local models) |
| **Build Once, Deploy Endlessly** | Shared vocabulary (SVL) and storage (CLST) means new agents gain context from day one |
| **Deterministic & Traceable** | Every memory operation is auditable with SVL-compliant metadata |
| **Fast & Accurate** | Hierarchical retrieval without embeddings achieves <160ms context assembly |
| **Failure Resilient** | Built-in strategies for graceful degradation and recovery |
| **Open Source** | MIT licensed, community-driven, no vendor lock-in |

### The Core Innovation

MindCore introduces three foundational protocols that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI AGENTS                                │
│    (Customer Support, Sales, Internal Tools, Autonomous...)     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MINDCORE                                 │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │      FLR        │  │      CLST       │  │       SVL       │  │
│  │   (Hot Path)    │  │   (Cold Path)   │  │  (Vocabulary)   │  │
│  │                 │  │                 │  │                 │  │
│  │ • Fast Recall   │  │ • Persistence   │  │ • Shared Schema │  │
│  │ • Reinforcement │  │ • Aggregates    │  │ • LLM Enforced  │  │
│  │ • Learning      │  │ • Compression   │  │ • Data Sources  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Problem: Memory Fragmentation in AI Systems

### The Current State of AI Agent Memory

Every organization building AI agents faces the same fundamental challenges:

| Challenge | Current Approach | Pain Points |
|-----------|------------------|-------------|
| **Memory & Persistence** | Custom storage, caching, retrieval systems | 2-4 weeks reinventing infrastructure |
| **Multi-Agent Consistency** | Each agent maintains isolated memory silos | Agents contradict each other, no shared learning |
| **Vocabulary Alignment** | Ad-hoc metadata schemas per agent | "Is it `topic` or `topics`? `category` or `type`?" |
| **Production Features** | DIY rate limiting, audit trails, encryption | Security vulnerabilities, compliance gaps |
| **Onboarding New Agents** | Rebuild context from scratch for each agent | Weeks of "training" before agents become useful |

### The Cost of Fragmentation

```
Traditional Approach: Each Agent Builds Its Own Memory

Agent A                    Agent B                    Agent C
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ Custom Store │          │ Custom Store │          │ Custom Store │
│ Custom Schema│          │ Custom Schema│          │ Custom Schema│
│ Custom Cache │          │ Custom Cache │          │ Custom Cache │
│ No Sharing   │          │ No Sharing   │          │ No Sharing   │
└──────────────┘          └──────────────┘          └──────────────┘
       ↓                         ↓                         ↓
   Isolated                  Isolated                  Isolated
   Learning                  Learning                  Learning
```

**Result**: Months of infrastructure work before focusing on actual product value. Each new agent starts from zero context.

### Why Existing Solutions Fall Short

1. **Vector Databases Alone Are Not Enough**: Embeddings are expensive, slow, and don't provide structured queryability
2. **RAG Without Structure**: Retrieval-Augmented Generation without vocabulary control leads to inconsistent, untraceable results
3. **No Cross-Agent Learning**: Most solutions treat each agent as an island
4. **No Production Features**: Open-source solutions lack enterprise requirements (audit, encryption, compliance)

---

## 3. The MindCore Solution

### A Protocol-First Approach

MindCore doesn't just provide storage—it provides **protocols** that standardize how AI agents interact with memory. This is analogous to how HTTP standardized web communication or how MCP standardized tool connections.

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Structured Output Only** | LLMs produce memories via JSON schema with SVL validation—no fallback parsing |
| **Fail Hard** | Validation errors raise exceptions, not silent failures—predictable behavior |
| **Vocabulary Controlled** | All metadata follows versioned SVL vocabulary—deterministic queries |
| **Hierarchical First** | Query sessions by weighted metadata, then drill into memories—10-100x faster |
| **Feedback Loops** | Reinforcement signals improve future retrievals—continuous learning |
| **Multi-Backend** | PostgreSQL for production, SQLite for development—flexible deployment |

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI Agent / LLM                                  │
│                         (Structured Output JSON)                             │
│                                                                              │
│   LLM assigns SVL-compliant metadata: topics, categories, importance, etc.  │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Context Gateway                                     │
│              (Unified context assembly from FLR + CLST + SVL)                │
│                                                                              │
│  • HistoricalContextNeeded decision (LLM decides if CLST query needed)      │
│  • Hierarchical retrieval: Sessions → Memories                              │
│  • SVL data source auto-fetching (tables, APIs, MCP)                        │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│       FLR       │  │      CLST       │  │       SVL       │
│   (Hot Path)    │  │   (Cold Path)   │  │  (Vocabulary)   │
│                 │  │                 │  │                 │
│ • Session cache │  │ • Hierarchical  │  │ • Ontology      │
│ • Reinforcement │  │ • Aggregates    │  │ • LLM Providers │
│ • Usage detect  │  │ • Weighted meta │  │ • Data Sources  │
│ • Query optim   │  │ • Decay/compress│  │ • Feedback      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Storage Backend                                    │
│                   PostgreSQL (prod) | SQLite (dev)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Three Foundational Protocols

### 4.1 FLR: Fast Learning Recall (Hot Path)

FLR is the inference-time memory access layer with built-in reinforcement learning.

#### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Fast Query** | LRU cache with TTL for sub-10ms retrieval |
| **6-Factor Scoring** | Semantic similarity, topic match, recency, reinforcement, importance, popularity |
| **Robust Reinforcement** | Temporal decay, multi-signal types, exploration balancing |
| **Usage Detection** | Tracks which memories were actually used in responses |
| **Query Optimization** | Dynamically adjusts queries based on effectiveness patterns |

#### Reinforcement Signal System

MindCore's reinforcement system goes far beyond simple +1/-1 feedback:

```python
# Signal Types (what aspect of quality)
SignalType.RELEVANCE      # Was the memory relevant to the query?
SignalType.USEFULNESS     # Did it help solve the task?
SignalType.CORRECTNESS    # Was the information accurate?
SignalType.TIMELINESS     # Was it appropriately current?
SignalType.COMPLETENESS   # Did it provide sufficient detail?

# Signal Sources (reliability weights)
SignalSource.USER_EXPLICIT     # 1.0 - Direct user feedback
SignalSource.USER_IMPLICIT     # 0.7 - Inferred from behavior
SignalSource.LLM_EVALUATION    # 0.5 - LLM self-assessment
SignalSource.CROSS_AGENT       # 0.6 - Feedback from other agents
SignalSource.AUTOMATED_METRIC  # 0.3 - System metrics
```

**Temporal Decay Formula:**

```
effective_score = base_score × e^(-λt) + exploration_bonus

Where:
- λ = ln(2) / half_life_hours (configurable, default 168 hours)
- exploration_bonus = UCB1 formula for less-accessed memories
```

### 4.2 CLST: Cognitive Long-term Storage Transfer (Cold Path)

CLST handles persistent storage with hierarchical retrieval and session aggregation.

#### Session Aggregates: The Key Innovation

Instead of embedding every memory, CLST aggregates metadata at the session level:

```python
SessionAggregate:
  session_id: "session_123"
  user_id: "user_456"

  # Weighted distributions (term → weight 0-1)
  topic_weights: {"orders": 0.85, "shipping": 0.6, "returns": 0.3}
  category_weights: {"support": 0.9, "billing": 0.4}
  intent_weights: {"ask_question": 0.7, "request_action": 0.5}

  # Statistics for filtering
  importance_avg: 0.72
  importance_max: 0.95
  memory_count: 47

  # Dominant values for quick matching
  dominant_topic: "orders"
  dominant_category: "support"
```

**Weight Calculation Formula:**

```
topic_weight = (frequency × 0.4) + (avg_importance × 0.4) + (recency × 0.2)
```

#### Hierarchical Query Flow

```
1. Query Sessions    →  Match by weighted topic/category
2. Rank Sessions     →  Calculate relevance scores
3. Query Memories    →  Only from top-N relevant sessions
4. Return Context    →  With session summaries and memories

Result: 10-100x search space reduction, no embeddings required
```

#### Compression Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `DEDUPLICATE` | Remove duplicate content (MD5 hash) | General cleanup |
| `MERGE` | Combine memories with same topics | Consolidation |
| `SUMMARIZE` | LLM-based summarization | Long-term archival |
| `EXTRACT` | Extract key facts only | Knowledge base |

### 4.3 SVL: Structured Validation Layer (Semantic Foundation)

SVL is the semantic spine that ensures consistent metadata across all agents and memories.

#### Core Ontology

```python
# Memory Types
MemoryType.EPISODIC      # Events, conversations, interactions
MemoryType.SEMANTIC      # Facts, knowledge, learned information
MemoryType.PROCEDURAL    # Workflows, how-to, processes
MemoryType.PREFERENCE    # User preferences, settings
MemoryType.ENTITY        # People, places, things
MemoryType.RELATIONSHIP  # Connections between entities
MemoryType.TEMPORAL      # Time-bound info (auto-expires)
MemoryType.WORKING       # Current session context (cleared)

# Access Levels
AccessLevel.PRIVATE      # Only this agent
AccessLevel.TEAM         # Agents in same team/group
AccessLevel.SHARED       # All agents for this user
AccessLevel.GLOBAL       # Cross-user (knowledge base)

# Message Types (for classification)
MessageType.QUERY        # User asking a question
MessageType.COMMAND      # User giving an instruction
MessageType.STATEMENT    # User providing information
MessageType.FEEDBACK     # User giving feedback/opinion
MessageType.RESPONSE     # Agent answering a query
# ... and 15+ more types
```

#### Built-in Domain Vocabularies

MindCore includes pre-built vocabularies for common domains:

- `customer_service` - tickets, escalation, satisfaction
- `ecommerce` - cart, checkout, shipping, returns
- `healthcare` - appointments, diagnosis, medication
- `finance` - transactions, accounts, investments
- `saas` - subscriptions, features, onboarding
- `hr` - hiring, training, performance
- `education` - courses, assignments, grades

#### Data Source Mapping: Connect Vocabulary to External Resources

One of SVL's most powerful features is **automatic data fetching from external resources** based on detected topics. When a memory query involves a topic like "orders" or "billing", SVL can automatically fetch relevant data from databases, APIs, or MCP servers—eliminating the need to manually orchestrate data retrieval.

##### The Four Source Types

| Source Type | Description | Use Case |
|-------------|-------------|----------|
| **TableSource** | SQL database queries | Orders, user preferences, transaction history |
| **APISource** | REST API endpoints | External services, internal microservices, third-party APIs |
| **MCPSource** | MCP server tool calls | Brave Search, filesystem access, custom MCP tools |
| **FunctionSource** | Custom Python functions | Complex logic, data transformations, aggregations |

##### Trigger Conditions

Data sources can be triggered under different conditions:

```python
TriggerCondition.ALWAYS      # Always fetch when topic is used
TriggerCondition.ON_QUERY    # Only on memory queries (default)
TriggerCondition.ON_STORE    # Only when storing memories
TriggerCondition.ON_DEMAND   # Only when explicitly requested
TriggerCondition.CONDITIONAL # Based on context conditions
```

##### Example: Orders from Database

When a user asks "Where is my order?", the SVL detects the "orders" topic and automatically fetches relevant order data:

```python
from mindcore.svl.sources import TableSource, TriggerCondition

# Map "orders" topic to database table
svl.map_source("orders", TableSource(
    name="orders_db",
    connection_string="postgresql://localhost/ecommerce",
    table="orders",
    query_template="""
        SELECT order_id, status, total, created_at, tracking_number
        FROM orders
        WHERE user_id = :user_id
        AND created_at >= NOW() - INTERVAL '30 days'
        ORDER BY created_at DESC
        LIMIT 10
    """,
    param_mapping={"user_id": "user_id"},  # Context key -> SQL param
    cache_ttl_seconds=60,
    trigger=TriggerCondition.ON_QUERY,
))

# When agent queries with topic "orders", data is auto-fetched
context = gateway.build_context(
    query="Where is my order?",
    user_id="user_123",
    topics=["orders"],  # Triggers TableSource fetch
)
# context now includes recent orders from database!
```

##### Example: Billing from Internal API

```python
from mindcore.svl.sources import APISource

# Map "billing" category to billing microservice
svl.map_source("billing", APISource(
    name="billing_service",
    url="${BILLING_SERVICE_URL}/api/v1/customers/{user_id}/summary",
    method="GET",
    headers={
        "Authorization": "Bearer ${INTERNAL_API_KEY}",
        "X-Request-ID": "{request_id}",
    },
    url_params={"user_id": "user_id"},
    header_params={"request_id": "request_id"},
    response_path="data",  # Extract from response.data
    cache_ttl_seconds=180,
    trigger=TriggerCondition.ON_QUERY,
), term_type="category")
```

##### Example: Refunds with Date Range

```python
# Map "refunds" to database with date range filtering
svl.map_source("refunds", TableSource(
    name="refunds_db",
    connection_string="postgresql://localhost/ecommerce",
    query_template="""
        SELECT r.refund_id, r.order_id, r.amount, r.status, r.reason, r.created_at
        FROM refunds r
        JOIN orders o ON r.order_id = o.order_id
        WHERE o.user_id = :user_id
        AND r.created_at BETWEEN :start_date AND :end_date
        ORDER BY r.created_at DESC
    """,
    param_mapping={
        "user_id": "user_id",
        "start_date": "start_date",
        "end_date": "end_date",
    },
    cache_ttl_seconds=120,
))
```

##### Example: MCP Server Integration

```python
from mindcore.svl.sources import MCPSource

# Map "web_search" to Brave Search MCP server
svl.map_source("web_search", MCPSource(
    name="brave_search",
    server_name="brave-search",
    tool_name="brave_web_search",
    argument_mapping={"query": "search_query"},
    static_arguments={"count": 5},
    cache_ttl_seconds=300,
    trigger=TriggerCondition.ON_DEMAND,
))

# Map "file_context" to filesystem MCP server
svl.map_source("file_context", MCPSource(
    name="filesystem",
    server_name="filesystem",
    tool_name="read_file",
    argument_mapping={"path": "file_path"},
    trigger=TriggerCondition.ON_DEMAND,
))
```

##### Decorator-Based Source Registration

For sources requiring custom logic, use the `@source` decorator:

```python
from mindcore.svl.registry import source
from mindcore.svl import TriggerCondition

@source(
    term="orders",
    term_type="topic",
    description="Fetch user's recent orders from database",
    trigger=TriggerCondition.ON_QUERY,
    cache_ttl=60,
    priority=10,
    tags=["ecommerce", "database"],
)
async def get_user_orders(context: dict) -> list[dict]:
    """Fetch recent orders for a user."""
    user_id = context.get("user_id")
    if not user_id:
        return []

    async with db.connection() as conn:
        return await conn.fetch(
            """
            SELECT order_id, status, total, created_at, tracking_number
            FROM orders
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 10
            """,
            user_id
        )

@source(
    term="order_details",
    term_type="topic",
    description="Fetch detailed order with items and shipping",
    trigger=TriggerCondition.ON_DEMAND,
    cache_ttl=30,
)
async def get_order_details(context: dict) -> dict | None:
    """Aggregate order data from multiple sources."""
    order_id = context.get("order_id")
    if not order_id:
        return None

    # Aggregate from multiple sources
    order = await db.fetch_order(order_id)
    items = await db.fetch_order_items(order_id)
    shipping = await shipping_api.get_tracking(order["tracking_id"])

    return {
        "order": order,
        "items": items,
        "shipping": shipping,
    }
```

##### YAML Configuration for Simple Sources

For sources that don't require custom logic, use YAML configuration:

```yaml
# svl_sources.yaml
sources:
  # Database: User preferences
  - term: "user_preferences"
    term_type: "topic"
    type: "table"
    name: "user_prefs_db"
    connection_string: "${DATABASE_URL}"
    query_template: |
      SELECT preference_key, preference_value, updated_at
      FROM user_preferences
      WHERE user_id = :user_id
    param_mapping:
      user_id: "user_id"
    cache_ttl: 300
    trigger: "on_query"

  # API: Billing summary
  - term: "billing"
    term_type: "category"
    type: "api"
    name: "billing_service"
    url: "${BILLING_SERVICE_URL}/api/v1/customers/{user_id}/summary"
    method: "GET"
    headers:
      Authorization: "Bearer ${INTERNAL_API_KEY}"
    url_params:
      user_id: "user_id"
    cache_ttl: 180
    trigger: "on_query"

  # MCP: Web search
  - term: "web_search"
    term_type: "topic"
    type: "mcp"
    name: "brave_search"
    server_name: "brave-search"
    tool_name: "brave_web_search"
    argument_mapping:
      query: "search_query"
    static_arguments:
      count: 5
    trigger: "on_demand"
    cache_ttl: 300
```

##### How It Works: The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          User Query: "Where is my order?"                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. SVL Topic Detection                                                      │
│     Detected topics: ["orders", "shipping"]                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Source Registry Lookup                                                   │
│     "orders" → TableSource (orders_db)                                       │
│     "shipping" → APISource (shipping_api)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
┌───────────────────────────────┐    ┌───────────────────────────────────────┐
│  3a. Database Query           │    │  3b. API Request                       │
│  SELECT * FROM orders         │    │  GET /tracking/{tracking_id}          │
│  WHERE user_id = 'user_123'   │    │  Authorization: Bearer ...            │
└───────────────────────────────┘    └───────────────────────────────────────┘
                    │                                      │
                    └──────────────────┬──────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. FetchResult Aggregation                                                  │
│  {                                                                           │
│    "orders": [{"order_id": "ORD-123", "status": "shipped", ...}],           │
│    "shipping": {"carrier": "UPS", "eta": "2024-01-20", ...}                 │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Context Assembly                                                         │
│  Memory context + External data → Complete agent context                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Automatic Fetching** | No manual orchestration—data fetched when topics are detected |
| **Unified Caching** | Built-in LRU cache with configurable TTL per source |
| **Fail-Safe** | Failed fetches don't block memory retrieval |
| **Traceable** | Every fetch logged with latency, source, and metadata |
| **Flexible Triggers** | Control when data is fetched (query, store, on-demand) |
| **Security** | SQL injection prevention, URL validation, path traversal protection |

---

## 5. Build It Once, Deploy It Endlessly

### The Universal Memory Foundation

This is MindCore's core value proposition: **CLST and SVL are universal foundations that benefit every agent from day one.**

```
Traditional Approach:
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Agent A │   │ Agent B │   │ Agent C │
│ Memory  │   │ Memory  │   │ Memory  │
│ (empty) │   │ (empty) │   │ (empty) │
└─────────┘   └─────────┘   └─────────┘
     ↓             ↓             ↓
  Weeks of     Weeks of     Weeks of
  Learning     Learning     Learning


MindCore Approach:
                    ┌─────────────────────────────┐
                    │    Shared SVL + CLST        │
                    │  (Organization Knowledge)   │
                    │                             │
                    │  • Customer preferences     │
                    │  • Product knowledge        │
                    │  • Historical interactions  │
                    │  • Cross-agent learnings    │
                    └─────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         ┌─────────┐      ┌─────────┐      ┌─────────┐
         │ Agent A │      │ Agent B │      │ Agent C │
         │   FLR   │      │   FLR   │      │   FLR   │
         │(private)│      │(private)│      │(private)│
         └─────────┘      └─────────┘      └─────────┘
              ↓                ↓                ↓
           Instant          Instant          Instant
           Context          Context          Context
```

### Benefits of the Universal Foundation

| Benefit | Description |
|---------|-------------|
| **Zero Cold Start** | New agents inherit existing organizational knowledge immediately |
| **Cross-Agent Learning** | Reinforcement signals from one agent improve retrieval for all |
| **Consistent Vocabulary** | Every agent uses the same topics, categories, and schemas |
| **Reduced Redundancy** | No duplicate memory storage across agents |
| **Traceable History** | Complete audit trail across all agent interactions |

### Adding a New Agent: Before vs After

**Before MindCore:**

```
Week 1: Build custom storage layer
Week 2: Define metadata schema
Week 3: Implement retrieval logic
Week 4: Build caching system
Week 5: Add production features
Week 6+: Start accumulating context
```

**With MindCore:**

```python
# Day 1: New agent with full organizational context
from mindcore.federation import quick_setup

federation = quick_setup(
    org_id="acme-corp",
    departments={"support": ["tier-1", "tier-2"]},
)

new_agent = federation.create_agent(
    agent_id="new-support-bot",
    agent_type="support-bot",
    department="support",
    team="tier-1",
)

# Agent immediately has access to:
# - All customer preferences (from SVL)
# - Historical interactions (from CLST)
# - Cross-agent learnings (from shared reinforcement)
# - Domain vocabulary (from SVL domains)
```

---

## 6. Universal LLM Compatibility

### Works With Any LLM Provider

MindCore is designed to work with any LLM through provider-agnostic protocols. The SVL layer handles provider-specific differences in how structured outputs are requested.

#### Supported Providers

| Provider | API | Key Features | Integration Method |
|----------|-----|--------------|-------------------|
| **OpenAI GPT-5** | Responses API | `reasoning_effort`, preserved reasoning | `text.format` with JSON Schema |
| **Claude** | Messages API | Extended thinking, `budget_tokens` | `output_format` with JSON Schema |
| **Gemini 2.5** | GenerativeAI | `thinkingBudget` (dynamic) | `response_schema` |
| **Gemini 3** | GenerativeAI | `thinkingLevel` (level-based) | `response_schema` |
| **Local Models** | Generic | Standard chat completion | `response_format` |

### Provider-Specific Configurations

MindCore provides optimized configurations for each provider:

```python
from mindcore.svl.llm_providers import (
    OpenAIConfig,
    ClaudeConfig,
    GeminiConfig,
    ReasoningEffort,
    ThinkingLevel,
)

# OpenAI GPT-5 with Responses API
openai_config = OpenAIConfig(
    model="gpt-5",
    reasoning_effort=ReasoningEffort.HIGH,
    use_responses_api=True,  # 3-5% better intelligence
    temperature=0.0,  # Deterministic
)

# Claude with Extended Thinking
claude_config = ClaudeConfig(
    model="claude-sonnet-4-5-20250514",
    thinking_budget=16000,
    use_extended_thinking=True,
    use_interleaved_thinking=True,
)

# Gemini 3 with Thinking Level
gemini_config = GeminiConfig(
    model="gemini-3-flash",
    thinking_level=ThinkingLevel.HIGH,
    temperature=0.0,
)
```

### How Metadata Is Injected Per Provider

MindCore uses different API mechanisms to inject metadata requirements:

| Provider | Injection Method | Example |
|----------|------------------|---------|
| **OpenAI** | `instructions` parameter or `developer` role | High-priority guidance separate from user input |
| **Claude** | System prompt suffix + `output_format` | Structured output with thinking |
| **Gemini** | `systemInstruction` + `response_schema` | Schema-validated JSON output |
| **Generic** | System message + `response_format` | Standard JSON mode |

---

## 7. Structured Output & Metadata Enrichment

### LLM-Enforced Metadata Extraction

MindCore forces LLMs to assign metadata from the SVL vocabulary through structured outputs. This ensures deterministic, queryable, and traceable memories.

> **📋 Canonical Schema Reference**: All metadata field definitions, valid values, and examples are defined in [`mindcore/svl/metadata_schema.yaml`](./mindcore/svl/metadata_schema.yaml). This file is the single source of truth for all LLM prompts, JSON schemas, and validation logic.

#### The Enforced Metadata Schema

```python
@dataclass
class EnforcedMetadata:
    # Identifiers
    message_id: str
    user_id: str
    session_id: str
    thread_id: str | None  # Multi-thread support

    # SVL-enforced classifications (LLM must choose from vocabulary)
    topics: list[str]              # ["orders", "shipping"]
    categories: list[str]          # ["support"]
    entities: list[str]            # ["Order #12345"]
    message_type: str              # "query", "command", "statement"
    message_intent: str            # "ask_question", "request_action"

    # Scores
    importance: float              # 0.0 - 1.0
    confidence: float              # 0.0 - 1.0
    urgency: str                   # "critical", "high", "medium", "low", "informational"
    sentiment: str                 # "positive", "negative", "neutral", "mixed"
    emotional_classification: str  # "neutral", "frustration", "satisfaction", "joy", ...

    # Memory classification
    memory_type: str               # episodic, semantic, preference, ...
    access_level: str              # private, team, shared, global
```

### Example: Complete LLM Request and Response

#### Input to LLM (with MindCore prompt)

```json
{
  "model": "gpt-5",
  "reasoning": {"effort": "high"},
  "text": {
    "format": {
      "type": "json_schema",
      "name": "svl_metadata",
      "schema": {
        "type": "object",
        "properties": {
          "topics": {
            "type": "array",
            "items": {"type": "string", "enum": ["orders", "shipping", "billing", "refund", "account", "settings"]},
            "minItems": 1,
            "maxItems": 5
          },
          "categories": {
            "type": "array",
            "items": {"type": "string", "enum": ["support", "inquiry", "complaint", "feedback", "order", "payment", "account"]},
            "minItems": 1,
            "maxItems": 3
          },
          "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Extracted entity names from the message"
          },
          "message_type": {"type": "string", "enum": ["query", "command", "statement", "feedback", "response", "clarification"]},
          "message_intent": {"type": "string", "enum": ["ask_question", "request_action", "provide_info", "give_feedback", "complaint", "greeting"]},
          "importance": {"type": "number", "minimum": 0, "maximum": 1},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "urgency": {"type": "string", "enum": ["critical", "high", "medium", "low", "informational"]},
          "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
          "emotional_classification": {"type": "string", "enum": ["neutral", "joy", "frustration", "satisfaction", "confusion", "anger"]},
          "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "preference", "entity", "working"]},
          "access_level": {"type": "string", "enum": ["private", "team", "shared", "global"]},
          "memories_to_store": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "content": {"type": "string", "description": "The actual information to remember"},
                "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "preference", "entity", "working"]},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "topics": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
                "access_level": {"type": "string", "enum": ["private", "team", "shared", "global"]}
              },
              "required": ["content", "memory_type", "importance"]
            }
          }
        },
        "required": ["topics", "categories", "message_type", "message_intent", "importance", "confidence", "urgency", "sentiment", "memory_type", "access_level"]
      },
      "strict": true
    }
  },
  "input": "User message: 'I need a refund for order #12345, it arrived damaged'"
}
```

#### LLM Response (Metadata-Enriched)

```json
{
  "message_id": "msg_a1b2c3d4e5f6",
  "user_id": "user_123",
  "session_id": "session_abc",

  "topics": ["refund", "orders", "shipping"],
  "categories": ["support", "order"],
  "entities": ["Order #12345"],

  "message_type": "command",
  "message_intent": "request_action",

  "importance": 0.85,
  "confidence": 0.9,
  "urgency": "high",
  "sentiment": "negative",
  "emotional_classification": "frustration",
  "temporal_qualifier": null,
  "domain_label": "customer_service",

  "memory_type": "episodic",
  "access_level": "team",

  "memories_to_store": [
    {
      "content": "Customer reported damaged order #12345 and requested refund",
      "memory_type": "episodic",
      "importance": 0.85,
      "topics": ["refund", "orders"],
      "categories": ["support"],
      "entities": ["Order #12345"],
      "access_level": "team"
    },
    {
      "content": "Order #12345 arrived damaged - product quality issue",
      "memory_type": "entity",
      "importance": 0.7,
      "topics": ["orders", "shipping"],
      "categories": ["order"],
      "entities": ["Order #12345"],
      "access_level": "team"
    }
  ]
}
```

### Context Injection: What Gets Sent to the LLM

When context is built, MindCore injects structured context into the LLM prompt:

```markdown
## Current Session Context
- Topics discussed: orders (85%), shipping (60%), refund (30%)
- Messages in session: 12
- Session importance: 0.72
- Dominant sentiment: neutral

## Relevant Memories

### Session: a1b2c3d4...
- [preference] Customer prefers email notifications ⭐
- [episodic] Previous order #12340 delivered successfully
- [entity] Customer account created January 2024

### Session: e5f6g7h8...
- [episodic] Customer reported issue with order #12300 last month
- [episodic] Issue resolved with 10% discount coupon

## Related Data

### orders
- order_id: #12345, status: delivered, date: 2025-12-20
- items: Widget Pro (qty: 2), total: $199.00
- shipping: Express, tracking: 1Z999AA10123456784
```

### Feedback-Enhanced Schema Annotations

MindCore can inject effectiveness feedback directly into JSON Schema descriptions:

```json
{
  "properties": {
    "topics": {
      "type": "array",
      "items": {"type": "string", "enum": ["refund", "orders", "shipping", "billing"]},
      "description": "Topics from SVL. Prefer: 'refund' (85%), 'orders' (72%). Avoid: 'general'."
    },
    "categories": {
      "type": "array",
      "items": {"type": "string", "enum": ["support", "complaint", "inquiry"]},
      "description": "Categories from SVL. Prefer: 'complaint' (90%), 'support' (78%)."
    }
  }
}
```

---

## 8. Hierarchical Retrieval Architecture

### The Problem with Traditional Approaches

```sql
-- Traditional: Fetch everything, huge context window waste
SELECT * FROM memories WHERE user_id = 'U-123';
-- Returns 10,000 rows, 500KB of data, 95% irrelevant
```

### MindCore's Hierarchical Solution

```sql
-- MindCore: Multi-stage filtering with controlled vocabulary
-- Stage 1: Query sessions by weighted metadata
-- Stage 2: Get memories only from relevant sessions
-- Returns 10-20 most relevant memories, 5KB of data, 95% relevant
```

### The Query Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                        User Query                                   │
│           "What about my order from last week?"                     │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              Step 1: HistoricalContextNeeded Decision              │
│                                                                    │
│  LLM analyzes query and decides:                                   │
│  - "True" → Query needs historical context (CLST)                  │
│  - "False" → Answer from current session only (FLR)                │
│                                                                    │
│  Result: {                                                         │
│    "historical_context_needed": "True",                            │
│    "suggested_topics": ["orders", "shipping"],                     │
│    "reasoning": "User references 'last week' - past interaction"   │
│  }                                                                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              Step 2: Session-Level Query (~15ms)                   │
│                                                                    │
│  Query session aggregates by weighted metadata:                    │
│                                                                    │
│  Sessions Found:                                                   │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Session A: topic_weights={"orders": 0.9, "shipping": 0.7}│      │
│  │            relevance_score: 0.85                         │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │ Session B: topic_weights={"billing": 0.8, "refund": 0.5} │      │
│  │            relevance_score: 0.45                         │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │ Session C: topic_weights={"orders": 0.6, "account": 0.4} │      │
│  │            relevance_score: 0.62                         │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
│  Selected: Sessions A and C (above threshold)                      │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              Step 3: Memory Retrieval (~25ms)                      │
│                                                                    │
│  Query memories only from selected sessions:                       │
│                                                                    │
│  Memories Retrieved: 15 (from 2 sessions)                          │
│  vs. Total Possible: 847 (all user memories)                       │
│                                                                    │
│  Search Space Reduction: 98.2%                                     │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              Step 4: SVL Source Fetch (~50ms)                      │
│                                                                    │
│  Auto-fetch data for matched topics:                               │
│  - "orders" → Query orders database                                │
│  - "shipping" → Fetch tracking info                                │
│                                                                    │
│  Source Data:                                                      │
│  {                                                                 │
│    "orders": [{"order_id": "#12345", "status": "delivered"}],     │
│    "shipping": [{"tracking": "1Z999AA1...", "delivered": true}]   │
│  }                                                                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              Step 5: Context Assembly (~5ms)                       │
│                                                                    │
│  Assemble unified context for LLM:                                 │
│  - Session summaries                                               │
│  - Relevant memories (ordered by relevance)                        │
│  - Source data (orders, shipping info)                             │
│  - Query metadata (for traceability)                               │
│                                                                    │
│  Total Latency: ~95ms                                              │
└────────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Approach | Query Time | Data Retrieved | Relevance |
|----------|------------|----------------|-----------|
| **Full Scan** | 500-2000ms | 100% of memories | ~5% relevant |
| **Vector Search** | 100-300ms | Top-K by embedding | ~60% relevant |
| **MindCore Hierarchical** | 50-100ms | Session-filtered | ~95% relevant |

---

## 8.5 Controlled Context Generation

### Priority-Based Context Assembly

MindCore provides fine-grained control over what context is assembled and how it's prioritized. This enables agents to focus on the most relevant information without manual orchestration.

#### Source Priority Levels

Data sources can be assigned priority levels to control fetch order and importance:

```python
from mindcore.svl.registry import source
from mindcore.svl import TriggerCondition

# High priority sources are fetched first and given more weight
@source(
    term="orders",
    term_type="topic",
    priority=10,  # Higher = more important
    cache_ttl=60,
    trigger=TriggerCondition.ON_QUERY,
    tags=["critical", "ecommerce"],
)
async def get_orders(context: dict) -> list[dict]:
    return await db.fetch_orders(context["user_id"])

@source(
    term="recommendations",
    term_type="topic",
    priority=3,  # Lower priority - fetched after orders
    cache_ttl=300,
    trigger=TriggerCondition.ON_DEMAND,
)
async def get_recommendations(context: dict) -> list[dict]:
    return await recommender.get_suggestions(context["user_id"])
```

#### Topic and Category Weights

Sessions aggregate weighted metadata that controls retrieval priority:

```python
# Weight calculation formula:
# weight = (frequency × 0.4) + (avg_importance × 0.4) + (recency × 0.2)

@dataclass
class SessionAggregate:
    session_id: str
    user_id: str

    # Weighted distributions (term → weight 0-1)
    topic_weights: dict[str, float]      # {"orders": 0.9, "shipping": 0.7}
    category_weights: dict[str, float]   # {"support": 0.8, "billing": 0.3}
    entity_weights: dict[str, float]     # {"Order #12345": 0.95}
    intent_weights: dict[str, float]     # {"check_status": 0.85}
    sentiment_weights: dict[str, float]  # {"neutral": 0.6, "frustrated": 0.4}

    # Importance statistics for filtering
    importance_min: float
    importance_max: float
    importance_avg: float

    # Dominant values (highest weighted)
    dominant_topic: str
    dominant_category: str
    dominant_sentiment: str
```

#### Importance Thresholds

Control what gets retrieved based on importance scoring:

```python
# Query with importance threshold - only high-value memories
context = gateway.build_context(
    query="What's my order status?",
    user_id="user_123",
    min_importance=0.5,           # Filter out low-importance memories
    min_topic_weight=0.3,         # Require significant topic match
    session_limit=5,              # Limit sessions searched
    memory_limit=20,              # Limit memories returned
)

# Fine-grained control for different scenarios
high_stakes_context = gateway.build_context(
    query="Cancel my subscription",
    user_id="user_123",
    min_importance=0.7,           # Only critical memories
    attention_hints=["billing", "subscription", "cancellation"],
    category_hints=["account_management"],
)
```

#### Attention Hints for Focused Retrieval

Direct the context gateway to prioritize specific topics:

```python
# Attention hints focus retrieval on specific topics
context = gateway.build_context(
    query="My package hasn't arrived",
    user_id="user_123",

    # These topics get priority in session matching
    attention_hints=["shipping", "delivery", "orders"],

    # Category-level filtering
    category_hints=["logistics", "support"],

    # Memory type filtering
    memory_types=["episodic", "semantic"],  # Skip preferences
)

# Context result shows what matched
print(f"Matched topics: {context.matched_topics}")
print(f"Sessions searched: {context.sessions_searched}")
print(f"Relevance scores: ...")
```

#### Preference Management with Priority Versioning

MindCore handles mutable preferences with temporal versioning and conflict resolution:

```python
from mindcore.flr import PreferenceManager, ConflictResolutionStrategy

prefs = PreferenceManager(storage, flr)

# Set preference with importance
pref = prefs.set_preference(
    user_id="user_123",
    key="communication_style",
    value="User prefers formal, professional communication",
    importance=0.7,
    categories=["preferences", "communication"],
)

# Update preference - old version is automatically deprecated
new_pref = prefs.update_preference(
    user_id="user_123",
    key="communication_style",
    value="User now prefers casual, friendly communication",
)
# Old preference gets negative reinforcement signal
# New preference gets positive reinforcement
# version=2, supersedes previous preference

# Multi-agent conflict resolution with priority
result = prefs.set_preference_with_conflict_check(
    user_id="user_123",
    key="product_preference",
    value="User prefers budget options",
    agent_id="sales_agent",
    conflict_strategy=ConflictResolutionStrategy.AGENT_PRIORITY,
    agent_priorities={
        "sales_agent": 10,      # High trust
        "support_agent": 8,     # Medium trust
        "marketing_agent": 5,   # Lower trust
    },
)
```

#### Conflict Resolution Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **NEWER_WINS** | Most recent timestamp wins | General updates |
| **HIGHER_CONFIDENCE** | Higher confidence score wins | Quality-based |
| **AGENT_PRIORITY** | Agent with higher priority wins | Trust hierarchy |
| **LLM_MERGE** | Use LLM to intelligently merge | Nuanced conflicts |
| **HUMAN_REVIEW** | Flag for manual review | Critical preferences |
| **KEEP_BOTH** | Store both as separate preferences | When both are valid |

#### Complete Controlled Context Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Query: "Check my order status"                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Attention Hints Applied                                                  │
│     attention_hints: ["orders", "shipping"]                                  │
│     category_hints: ["support"]                                              │
│     min_importance: 0.3                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Session Relevance Scoring                                                │
│                                                                              │
│  Session A: topic_weights={"orders": 0.9, "shipping": 0.7}                  │
│             relevance_score = 0.85 ✓ (above threshold)                      │
│                                                                              │
│  Session B: topic_weights={"billing": 0.8, "refund": 0.5}                   │
│             relevance_score = 0.25 ✗ (below threshold)                      │
│                                                                              │
│  Session C: topic_weights={"orders": 0.6, "account": 0.4}                   │
│             relevance_score = 0.52 ✓ (above threshold)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. Source Priority Execution                                                │
│                                                                              │
│  Priority 10: orders_db → Fetch user orders from database                   │
│  Priority 8:  shipping_api → Fetch tracking info from carrier               │
│  Priority 3:  recommendations → Skip (ON_DEMAND trigger)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Memory Importance Filtering                                              │
│                                                                              │
│  Memories from Sessions A, C filtered by:                                   │
│  - importance >= 0.3                                                         │
│  - topic match with attention_hints                                          │
│  - category match with category_hints                                        │
│                                                                              │
│  Result: 12 highly relevant memories (from 150+ in sessions)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Assembled Context                                                        │
│                                                                              │
│  {                                                                           │
│    "memories": [...12 relevant memories...],                                │
│    "source_data": {                                                          │
│      "orders": [{"order_id": "ORD-123", "status": "shipped"}],              │
│      "shipping": [{"carrier": "UPS", "eta": "Tomorrow"}]                    │
│    },                                                                        │
│    "matched_topics": ["orders", "shipping"],                                │
│    "query_metadata": {...traceability data...}                              │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Multi-Agent Federation

### Organization-Wide Memory Sharing

MindCore provides a complete federation architecture for organizations with multiple AI agents:

```
                    ┌─────────────────────────────────────┐
                    │           Organization SVL           │
                    │   (Shared Vocabulary + Feedback)     │
                    └─────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Support  │               │   Sales   │               │ Internal  │
    │ Namespace │               │ Namespace │               │ Namespace │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Tier-1   │               │  Inbound  │               │   HR Bot  │
    │  Tier-2   │               │  Outbound │               │  IT Bot   │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
   ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
   ▼      ▼      ▼             ▼      ▼      ▼             ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐       ┌─────┐┌─────┐┌─────┐       ┌─────┐┌─────┐┌─────┐
│FLR 1││FLR 2││FLR 3│       │FLR 4││FLR 5││FLR 6│       │FLR 7││FLR 8││FLR 9│
└─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────────┐
                    │         Federated CLST                │
                    │  (Shared Storage + Access Control)    │
                    └───────────────────────────────────────┘
```

### Access Control Hierarchy

```python
class AccessLevel(IntEnum):
    PRIVATE = 0       # Only the specific agent
    AGENT_TYPE = 10   # Shared with same agent type
    TEAM = 20         # Shared within team
    DEPARTMENT = 30   # Shared within department
    AD_HOC_GROUP = 40 # Custom group membership
    ORGANIZATION = 50 # Visible to all agents in org
    PUBLIC = 100      # Public (rare)
```

### Cross-Agent Signal Aggregation

When one agent reinforces a memory, the signal propagates to other agents with appropriate weighting:

```python
# Trust Policies for Cross-Agent Signals
TrustPolicy.EQUAL              # All signals weighted equally
TrustPolicy.NAMESPACE_WEIGHTED # Same-namespace signals boosted
TrustPolicy.REPUTATION_BASED   # Agent reputation weighting
TrustPolicy.RECENCY_WEIGHTED   # Recent signals weighted higher
TrustPolicy.HIERARCHICAL       # Organizational hierarchy weighting
```

### Example: Quick Federation Setup

```python
from mindcore.federation import quick_setup, AccessLevel

# Setup organization in minutes
federation = quick_setup(
    org_id="acme-corp",
    departments={
        "customer-success": ["support-tier-1", "support-tier-2", "escalation"],
        "sales": ["inbound", "outbound", "enterprise"],
        "internal": ["hr", "it-helpdesk"],
    },
)

# Create agents
support_agent = federation.create_agent(
    agent_id="support-001",
    agent_type="support-bot",
    department="customer-success",
    team="support-tier-1",
)

# Store memory with access control
support_agent.store(
    content="Customer prefers morning callbacks",
    user_id="customer-123",
    access_level=AccessLevel.DEPARTMENT,  # All customer-success can see
)

# Sales agent queries cross-department (if allowed)
sales_agent = federation.create_agent(
    agent_id="sales-001",
    agent_type="sales-bot",
    department="sales",
    team="inbound",
)

context = sales_agent.query(
    query="customer preferences",
    user_id="customer-123",
)
# Gets customer preferences from support agent's memories!
```

---

## 10. Enterprise-Grade Features

### Audit Logging

```python
from mindcore.enterprise import AuditLogger, AuditEventType

audit = AuditLogger(output="file", path="/var/log/mindcore")

# Automatic event types
AuditEventType.STORE           # Memory stored
AuditEventType.RETRIEVE        # Memory accessed
AuditEventType.DELETE          # Memory deleted
AuditEventType.UPDATE          # Memory updated
AuditEventType.SEARCH          # Search performed
AuditEventType.ACCESS_GRANTED  # Access allowed
AuditEventType.ACCESS_DENIED   # Access denied
AuditEventType.RATE_LIMIT_EXCEEDED
AuditEventType.SECURITY_EVENT
```

### Encryption at Rest

```python
from mindcore.enterprise import FieldEncryptor, EncryptionConfig, KeyRotator

config = EncryptionConfig(
    key=os.environ["ENCRYPTION_KEY"],
    # Or derive from password:
    password="strong-password",
    salt="unique-salt",
    kdf_iterations=1_200_000,  # Django 2025 recommendation
)

encryptor = FieldEncryptor(config)
encrypted = encryptor.encrypt("sensitive user data")

# Key rotation support
rotator = KeyRotator(old_key, new_key)
rotated = rotator.rotate(encrypted)
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

if limiter.is_allowed("user_123", operation="store", user_tier="pro"):
    memory.store(...)
else:
    retry_after = limiter.get_reset_time("user_123", "store", "pro")
    raise RateLimitExceeded(f"Retry after {retry_after}s")
```

### GDPR/CCPA Compliance

```python
from mindcore.enterprise import ComplianceManager, RetentionPolicy, AnonymizationStrategy

compliance = ComplianceManager(storage)

# GDPR Article 15: Right of Access
user_data = compliance.export_user_data("user_123")

# GDPR Article 17: Right to Erasure
result = compliance.delete_user_data("user_123")

# Anonymization options
result = compliance.anonymize_user_data(
    "user_123",
    strategy=AnonymizationStrategy.PSEUDONYMIZE
)

# Retention policies
policy = RetentionPolicy(
    memory_type_policies={
        "episodic": 730,      # 2 years
        "preference": None,   # Forever
        "working": 1,         # 1 day
    },
    default_max_age_days=365,
)
compliance.enforce_retention("user_123")
```

### OpenTelemetry Observability

```python
from mindcore.enterprise import MindcoreMetrics, MindcoreTracer, ObservabilityConfig

config = ObservabilityConfig(
    service_name="my-ai-agent",
    otlp_endpoint="http://localhost:4317",
)

metrics = MindcoreMetrics(config)
tracer = MindcoreTracer(config)

with tracer.start_span("recall_memories") as span:
    span.set_attribute("user_id", "user_123")
    result = memory.recall(query="preferences", user_id="user_123")

    metrics.record_recall(
        user_id="user_123",
        result_count=len(result.memories),
        latency_ms=result.latency_ms,
    )
```

---

## 11. Security Considerations

### Defense in Depth

MindCore implements multiple layers of security to protect memory data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Security Layers                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Transport Security                                                 │
│  • TLS 1.3 for all network communication                                    │
│  • Certificate pinning for LLM provider connections                         │
│  • mTLS support for internal services                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Authentication & Authorization                                     │
│  • API key validation with rotation support                                 │
│  • JWT-based session authentication                                         │
│  • Role-based access control (RBAC)                                         │
│  • Hierarchical access levels (private → team → org → global)               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Data Protection                                                    │
│  • AES-256-GCM encryption at rest                                           │
│  • Field-level encryption for sensitive data                                │
│  • Key rotation with zero-downtime migration                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Input Validation                                                   │
│  • SQL injection prevention (parameterized queries)                         │
│  • Path traversal protection for file sources                               │
│  • URL validation for API sources                                           │
│  • JSON Schema validation for all LLM outputs                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Threat Model

| Threat | Mitigation |
|--------|------------|
| **Prompt Injection** | Structured outputs with strict JSON Schema; no free-form parsing |
| **Memory Poisoning** | Confidence scores, reinforcement decay, cross-agent validation |
| **Data Exfiltration** | Access level enforcement, audit logging, rate limiting |
| **Credential Theft** | Environment variable isolation, secret management integration |
| **SQL Injection** | Parameterized queries only; no string interpolation |
| **Unauthorized Access** | Hierarchical RBAC, namespace isolation, federation policies |

### Secure Configuration Best Practices

```python
from mindcore.security import SecurityConfig, SecretManager

# Never hardcode secrets
config = SecurityConfig(
    # Use environment variables or secret managers
    encryption_key=SecretManager.get("MINDCORE_ENCRYPTION_KEY"),

    # Enable all security features
    enable_audit_logging=True,
    enable_rate_limiting=True,
    enable_field_encryption=True,

    # Strict access control
    default_access_level="private",
    require_explicit_sharing=True,

    # Input validation
    max_content_length=50_000,  # Characters
    max_topics_per_memory=5,
    max_entities_per_memory=10,

    # Session security
    session_timeout_minutes=60,
    require_session_binding=True,
)
```

### Memory Access Control

```python
# Fine-grained access control for multi-agent systems
from mindcore.security import AccessPolicy, MemoryFilter

policy = AccessPolicy(
    # Who can read
    read_access={
        "private": ["owner_agent"],
        "team": ["owner_agent", "team_members"],
        "department": ["owner_agent", "team_members", "dept_members"],
        "organization": ["all_org_agents"],
    },

    # Who can write/modify
    write_access={
        "private": ["owner_agent"],
        "team": ["owner_agent"],  # Only owner can modify team-visible
        "shared": ["owner_agent", "with_explicit_grant"],
    },

    # Automatic filtering based on caller
    apply_filter=MemoryFilter.BY_ACCESS_LEVEL,
)

# Memories are automatically filtered based on requesting agent
memories = agent.recall(query="preferences", user_id="user_123")
# Only returns memories the agent has access to
```

### Audit Trail Requirements

All security-relevant events are logged:

```python
# Security events automatically captured
SecurityEvent.AUTHENTICATION_SUCCESS
SecurityEvent.AUTHENTICATION_FAILURE
SecurityEvent.AUTHORIZATION_DENIED
SecurityEvent.MEMORY_ACCESS_GRANTED
SecurityEvent.MEMORY_ACCESS_DENIED
SecurityEvent.ENCRYPTION_KEY_ROTATED
SecurityEvent.RATE_LIMIT_EXCEEDED
SecurityEvent.SUSPICIOUS_PATTERN_DETECTED
SecurityEvent.DATA_EXPORT_REQUESTED
SecurityEvent.DATA_DELETION_REQUESTED

# Audit log format
{
    "timestamp": "2025-12-25T10:30:00Z",
    "event_type": "MEMORY_ACCESS_GRANTED",
    "agent_id": "support-001",
    "user_id": "user_123",
    "memory_id": "mem_abc123",
    "access_level": "team",
    "source_ip": "10.0.1.50",
    "request_id": "req_xyz789"
}
```

---

## 12. Performance, Reliability & Determinism

### Performance Benchmarks

| Operation | Target Latency | Actual (p95) | Notes |
|-----------|---------------|--------------|-------|
| **FLR Cache Hit** | <10ms | 5ms | LRU cache lookup |
| **Session Query** | <20ms | 15ms | PostgreSQL GIN indexes |
| **Memory Retrieval** | <30ms | 25ms | From session subset |
| **SVL Source Fetch** | <100ms | 50-80ms | Depends on source |
| **Full Context Build** | <200ms | 95-160ms | End-to-end |
| **Memory Store** | <50ms | 30ms | With aggregate update |

### Determinism Guarantees

MindCore provides deterministic behavior through:

1. **Controlled Vocabulary**: All metadata from SVL enums, no free-form tags
2. **Structured Outputs**: JSON Schema validation with `strict: true`
3. **Temperature 0**: Recommended for metadata extraction
4. **Seed Parameter**: Reproducible LLM outputs (OpenAI, Gemini)
5. **Versioned Schemas**: Migrations with rollback support

### Traceability Features

Every memory operation is traceable:

```python
# Query metadata captures full context
QueryMetadata:
  query_id: "qry_a1b2c3d4e5f6"
  query_text: "What about my order?"
  session_id: "session_abc"
  user_id: "user_123"
  topics: ["orders", "shipping"]
  categories: ["support"]
  attention_hints: ["orders"]
  sessions_searched: 3
  memories_retrieved: 15
  sources_fetched: 2
  latency_ms: 95.3
  created_at: "2025-12-25T10:30:00Z"

# Response metadata links back to query
ResponseMetadata:
  response_id: "rsp_f6e5d4c3b2a1"
  query_id: "qry_a1b2c3d4e5f6"  # Links to query
  memories_stored: 2
  memory_ids: ["mem_123", "mem_456"]
```

---

## 13. Failure Handling Strategies

### Graceful Degradation Hierarchy

MindCore implements multiple fallback strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Normal Operation                            │
│                                                                  │
│  1. FLR Cache → 2. Session Query → 3. Memory Query → 4. SVL     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ On Failure
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Fallback Level 1                            │
│                                                                  │
│  Skip failed component, continue with available data             │
│  Example: SVL source timeout → Return memories without source   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ On Failure
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Fallback Level 2                            │
│                                                                  │
│  Use cached/stale data with freshness indicator                 │
│  Example: Database timeout → Return cached session data         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ On Failure
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Fallback Level 3                            │
│                                                                  │
│  Return empty context with error metadata                       │
│  Agent can respond without context, acknowledging limitation    │
└─────────────────────────────────────────────────────────────────┘
```

### LLM Provider Failure Strategies

```python
# Strategy 1: Provider Fallback Chain
providers = ["openai", "anthropic", "gemini"]

for provider in providers:
    try:
        result = extract_metadata(message, provider=provider)
        break
    except ProviderError:
        continue

# Strategy 2: Simplified Extraction on Failure
try:
    # Full SVL-enforced extraction
    metadata = extractor.parse_metadata(llm_response)
except ExtractionError:
    # Fallback to basic extraction
    metadata = FallbackExtractor.extract_basic(
        message=message,
        default_topics=["general"],
        default_category="uncategorized"
    )

# Strategy 3: Queue for Retry
if extraction_failed:
    retry_queue.enqueue(
        message=message,
        attempts=1,
        backoff=ExponentialBackoff(base=2, max=300)
    )
```

### Storage Failure Handling

```python
# Connection pooling with automatic retry
storage = PostgresStorage(
    connection_string="postgresql://...",
    pool_size=10,
    max_overflow=5,
    pool_recycle=3600,
    retry_policy=RetryPolicy(
        max_attempts=3,
        backoff_factor=2,
        max_backoff=30
    )
)

# Write-ahead logging for durability
storage.enable_wal_mode()

# Automatic failover for replicas
storage.configure_replicas([
    "postgresql://replica1/...",
    "postgresql://replica2/...",
])
```

---

## 14. Real-World Examples & Use Cases

### Use Case 1: Customer Support Bot

```python
from mindcore import Mindcore
from mindcore.federation import quick_setup

# Setup
federation = quick_setup(
    org_id="support-org",
    departments={"support": ["tier-1", "tier-2", "escalation"]}
)

support_bot = federation.create_agent(
    agent_id="support-001",
    agent_type="support-bot",
    department="support",
    team="tier-1",
)

# Customer interaction
async def handle_message(user_id: str, session_id: str, message: str):
    # 1. Build context (95ms)
    context = support_bot.gateway.build_context(
        query=message,
        user_id=user_id,
        session_id=session_id,
    )

    # 2. Generate response with context
    response = await llm.generate(
        messages=[
            {"role": "system", "content": "You are a helpful support agent."},
            {"role": "context", "content": context.to_llm_context()},
            {"role": "user", "content": message}
        ],
        response_format=support_bot.get_json_schema()
    )

    # 3. Extract and store memories
    metadata, memories = extractor.parse_metadata(response)
    for mem in memories:
        support_bot.store(mem, session_id=session_id)

    # 4. Apply reinforcement from previous interactions
    for mem_id in context.query_metadata.memory_ids_used:
        support_bot.reinforce(mem_id, signal=0.3)  # Implicit positive

    return response["content"]
```

**Result**: Support bot with instant access to:

- Customer preferences across all channels
- Previous interaction history
- Cross-agent learnings from tier-2 and escalation
- Product knowledge from shared SVL

### Use Case 2: Sales Intelligence Agent

```python
# Sales agent with cross-department access
sales_agent = federation.create_agent(
    agent_id="sales-001",
    agent_type="sales-bot",
    department="sales",
    team="enterprise",
    cross_department_access=["support"]  # Can see support memories
)

# Before sales call
def prepare_customer_brief(customer_id: str):
    context = sales_agent.gateway.build_context(
        query="customer history and preferences",
        user_id=customer_id,
        attention_hints=["preferences", "issues", "orders"]
    )

    return f"""
    ## Customer Brief for {customer_id}

    ### Key Preferences
    {format_preferences(context.memories)}

    ### Recent Support Issues
    {format_issues(context.memories)}

    ### Purchase History
    {context.source_data.get("orders", [])}

    ### Recommended Talking Points
    {generate_talking_points(context)}
    """
```

### Use Case 3: Internal Knowledge Assistant

```python
# HR/IT knowledge base with global access memories
kb_agent = federation.create_agent(
    agent_id="kb-001",
    agent_type="knowledge-base",
    department="internal",
    team="shared-services",
)

# Store organizational knowledge
kb_agent.store(
    content="Company holiday schedule: Dec 24-26 and Jan 1 are non-working days",
    memory_type="semantic",
    topics=["hr", "holidays"],
    access_level=AccessLevel.ORGANIZATION,  # All agents can access
    importance=0.9,
)

# Any agent can now query this knowledge
support_agent.recall(
    query="office hours during holidays",
    user_id="internal",
)
# Returns the holiday schedule memory
```

### Example Memory Objects in System

```json
{
  "memory_id": "mem_pref_001",
  "content": "Customer prefers email notifications over SMS for order updates",
  "memory_type": "preference",
  "user_id": "customer_123",
  "agent_id": "support-001",
  "session_id": "session_abc",

  "topics": ["notifications", "settings"],
  "categories": ["preferences", "account"],
  "entities": ["email", "SMS"],

  "message_type": "statement",
  "message_intent": "provide_info",
  "importance": 0.8,
  "confidence": 0.95,
  "urgency": "low",
  "sentiment": "neutral",
  "emotional_classification": "neutral",
  "temporal_qualifier": "permanent",
  "domain_label": "customer_service",
  "access_level": "private",

  "reinforcement_score": 0.72,
  "access_count": 15,

  "vocabulary_version": "1.0.0",
  "created_at": "2025-12-20T14:30:00Z",
  "last_accessed": "2025-12-25T09:15:00Z"
}
```

---

## 15. Conclusion

### MindCore: The Universal Memory Standard

MindCore represents a fundamental shift in how AI agents manage memory. Instead of each agent building isolated, incompatible memory systems, MindCore provides:

| Traditional Approach | MindCore Approach |
|---------------------|-------------------|
| Custom storage per agent | Shared CLST across organization |
| Ad-hoc metadata schemas | Unified SVL vocabulary |
| Weeks to build memory layer | Minutes to deploy with full context |
| No cross-agent learning | Federated reinforcement signals |
| Embedding-heavy retrieval | Hierarchical weighted metadata |
| DIY production features | Built-in enterprise capabilities |

### The "Build It Once, Deploy It Endlessly" Promise

When you adopt MindCore:

1. **First Agent**: Define your SVL vocabulary, set up CLST storage, configure federation
2. **Second Agent**: Inherit all vocabulary, access shared memories, benefit from existing reinforcement
3. **Third Agent**: Same as second—instant context, zero cold start
4. **Nth Agent**: Continues to benefit from accumulated organizational knowledge

Every new agent makes the entire system smarter. Every reinforcement signal improves retrieval for all agents. Every memory enriches the collective understanding.

### Why MindCore?

- **Fast**: <160ms context assembly without embeddings
- **Accurate**: Hierarchical retrieval with 95% relevance
- **Reliable**: Built-in failure strategies and graceful degradation
- **Deterministic**: Controlled vocabulary, structured outputs, traceable operations
- **Traceable**: Complete audit trail from query to response
- **Open Source**: MIT licensed, no vendor lock-in, community-driven

### Getting Started

```bash
pip install mindcore
```

```python
from mindcore import Mindcore

memory = Mindcore(storage="sqlite:///dev.db")

memory.store(
    content="User prefers dark mode",
    memory_type="preference",
    user_id="user_123",
    topics=["settings", "ui"],
    importance=0.8,
)

result = memory.recall(
    query="user preferences",
    user_id="user_123",
)
```

---

<div align="center">

## Join the Memory Protocol Revolution

**GitHub**: [github.com/M-Alfaris/mindcore](https://github.com/M-Alfaris/mindcore)

**License**: MIT

**Version**: 2.0.0

---

*Like MCP standardized tool connections, MindCore standardizes memory.*

*Build it once. Deploy it endlessly.*

</div>

---

## Appendix A: API Quick Reference

### Core Operations

```python
# Store
memory_id = mindcore.store(content, memory_type, user_id, topics, importance)

# Recall
result = mindcore.recall(query, user_id, attention_hints, limit)

# Reinforce
new_score = mindcore.reinforce(memory_id, signal)

# Compress
result = mindcore.compress(user_id, older_than_days, strategy)

# Search
memories = mindcore.search(query, user_id, topics, categories)
```

### Federation

```python
# Setup
federation = quick_setup(org_id, departments)

# Create agent
agent = federation.create_agent(agent_id, agent_type, department, team)

# Store with access control
agent.store(content, user_id, access_level)

# Cross-agent query
context = agent.query(query, user_id)
```

### Context Gateway

```python
# Build context
context = gateway.build_context(query, user_id, session_id, attention_hints)

# With LLM decision
context = gateway.build_context_with_decision(query, context_decision, user_id)

# Record response
response_meta = gateway.record_response(query_meta, response_text, memories)
```

---

## Appendix B: Configuration Reference

### Environment Variables

```bash
# Storage
MINDCORE_DATABASE_URL=postgresql://user:pass@localhost:5432/mindcore

# Enterprise
MINDCORE_ENCRYPTION_KEY=your-secret-key
MINDCORE_AUDIT_PATH=/var/log/mindcore

# Observability
OTEL_SERVICE_NAME=mindcore
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Logging
MINDCORE_LOG_LEVEL=INFO
```

### YAML Configuration

```yaml
mindcore:
  app_name: "my-ai-agent"
  environment: production

database:
  type: postgresql
  pool:
    min_size: 5
    max_size: 20
    timeout: 30

cache:
  type: redis
  default_ttl: 300
  max_size: 10000

llm:
  provider: openai
  model: gpt-5
  temperature: 0.0
  reasoning_effort: high

federation:
  enabled: true
  trust_policy: namespace_weighted
  signal_propagation: true
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **CLST** | Cognitive Long-term Storage Transfer. The persistent storage protocol for hierarchical memory organization with session aggregation and compression. |
| **FLR** | Fast Learning Recall. The inference-time memory access protocol with reinforcement learning, caching, and 6-factor scoring. |
| **SVL** | Structured Validation Layer. The semantic foundation that enforces consistent metadata through controlled vocabulary and LLM structured outputs. |
| **Session Aggregate** | A weighted summary of all memories in a session, containing topic/category weights, importance statistics, and dominant values for efficient hierarchical retrieval. |
| **Reinforcement Signal** | Feedback applied to memories indicating relevance, usefulness, or correctness. Signals decay over time and influence future retrieval rankings. |
| **Access Level** | Permission scope for memory visibility: private (agent only), team, department, organization, or global. |
| **Memory Type** | Classification of memory content: episodic (events), semantic (facts), procedural (processes), preference (settings), entity (objects), or working (session-scoped). |
| **Context Gateway** | The unified interface that assembles context from FLR cache, CLST storage, and SVL data sources for LLM consumption. |
| **Attention Hints** | Topics or categories specified in a query to focus retrieval on specific areas of memory. |
| **Data Source** | External resource (database, API, MCP server, function) mapped to SVL vocabulary terms for automatic data fetching. |
| **Trigger Condition** | When a data source is fetched: ALWAYS, ON_QUERY, ON_STORE, ON_DEMAND, or CONDITIONAL. |
| **Federation** | Multi-agent architecture where agents share vocabulary (SVL), storage (CLST), and reinforcement signals across an organization. |
| **Namespace** | Organizational grouping for agents (e.g., department, team) that controls access and signal propagation. |
| **Hierarchical Retrieval** | Two-stage query: first match sessions by weighted metadata, then retrieve memories only from relevant sessions. |
| **Structured Output** | LLM response format enforced via JSON Schema to ensure consistent, validated metadata. |
| **Temporal Decay** | Exponential reduction in reinforcement scores over time, prioritizing recent relevance. |
| **Compression Strategy** | Method for reducing storage: DEDUPLICATE, MERGE, SUMMARIZE, or EXTRACT. |

---

## Appendix D: References & Further Reading

### Academic & Industry Research

1. **Memory-Augmented Neural Networks**
   - Graves, A., et al. "Neural Turing Machines." arXiv:1410.5401 (2014)
   - Weston, J., et al. "Memory Networks." arXiv:1410.3916 (2014)

2. **Retrieval-Augmented Generation (RAG)**
   - Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS (2020)

3. **Multi-Agent Systems**
   - Wooldridge, M. "An Introduction to MultiAgent Systems." Wiley (2009)

4. **Reinforcement Learning for Information Retrieval**
   - Nogueira, R., et al. "Document Ranking with a Pretrained Sequence-to-Sequence Model." EMNLP (2020)

### Related Projects & Standards

- **Model Context Protocol (MCP)**: [modelcontextprotocol.io](https://modelcontextprotocol.io) - Tool connection standard for LLMs
- **OpenTelemetry**: [opentelemetry.io](https://opentelemetry.io) - Observability framework
- **JSON Schema**: [json-schema.org](https://json-schema.org) - Schema validation standard

### LLM Provider Documentation

- **OpenAI Responses API**: Structured outputs with reasoning
- **Anthropic Claude**: Extended thinking and interleaved reasoning
- **Google Gemini**: ThinkingLevel and response schemas

### MindCore Resources

- **GitHub Repository**: [github.com/M-Alfaris/mindcore](https://github.com/M-Alfaris/mindcore)
- **Metadata Schema**: `mindcore/svl/metadata_schema.yaml`
- **API Documentation**: See Appendix A
- **Configuration Guide**: See Appendix B

---

*© 2025 MindCore Project. MIT License.*
