# MindCore Documentation

> Ground truth documentation for MindCore - a memory protocol stack for AI agents.
> Last updated: 2025-12-25

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| `mindcore/v2/flr/` | Core | Fast Local Recall - hot path with reinforcement learning |
| `mindcore/v2/clst/` | Core | Cognitive Long-term Storage Transfer - cold path with aggregates |
| `mindcore/v2/svl/` | Core | Shared Vocabulary Layer - LLM-enforced metadata extraction |
| `mindcore/v2/context/` | Core | Context Gateway - unified context assembly |
| `mindcore/v2/federation/` | Core | Multi-agent memory federation with access control |
| `mindcore/v2/cross_agent/` | Core | Cross-agent memory sharing and routing |
| `mindcore/v2/enterprise/` | Enterprise | Audit, encryption, observability, rate limiting |
| `mindcore/v2/patterns/` | Patterns | Customer-facing agent patterns |

---

## 1. Overview

MindCore is a **memory protocol stack** for AI agents that provides:

- **Three-Layer Architecture**: FLR (hot) → CLST (cold) → SVL (vocabulary)
- **Hierarchical Retrieval**: Sessions → Memories (reduces search space without embeddings)
- **LLM-Enforced Metadata**: Main LLM tags memories with SVL-compliant vocabulary
- **Robust Reinforcement**: Temporal decay, multi-signal types, exploration balancing
- **Multi-Agent Federation**: Isolated FLRs with shared CLST/SVL and cross-agent signals
- **Enterprise Features**: Audit trails, encryption, observability, rate limiting

### Design Principles

1. **Structured Output Only** - LLMs produce memories via JSON schema, no fallbacks
2. **Fail Hard** - Validation errors crash, no silent failures
3. **Vocabulary Controlled** - All metadata follows versioned SVL vocabulary
4. **Hierarchical First** - Query sessions by weighted metadata, then drill into memories
5. **Feedback Loops** - Reinforcement signals improve future retrievals
6. **Multi-Backend** - PostgreSQL (production) / SQLite (development)

---

## 2. Architecture

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

### Multi-Agent Federation Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Organization SVL           │
                    │   (Shared Vocabulary + Feedback)     │
                    └─────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Dept A   │               │  Dept B   │               │  Dept C   │
    │ Namespace │               │ Namespace │               │ Namespace │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
    ┌─────▼─────┐               ┌─────▼─────┐               ┌─────▼─────┐
    │  Team 1   │               │  Team 2   │               │  Team 3   │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
   ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
   │      │      │             │      │      │             │      │      │
┌──▼──┐┌──▼──┐┌──▼──┐       ┌──▼──┐┌──▼──┐┌──▼──┐       ┌──▼──┐┌──▼──┐┌──▼──┐
│FLR 1││FLR 2││FLR 3│       │FLR 4││FLR 5││FLR 6│       │FLR 7││FLR 8││FLR 9│
│Agent││Agent││Agent│       │Agent││Agent││Agent│       │Agent││Agent││Agent│
└─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘       └─────┘└─────┘└─────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────────┐
                    │         Federated CLST                │
                    │  (Shared Storage + Access Control)    │
                    └───────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 FLR (Fast Local Recall)

**Location:** `mindcore/v2/flr/`

Hot path for inference-time memory access with robust reinforcement learning.

#### Files

| File | Purpose |
|------|---------|
| `recall.py` | Core FLR protocol - query, reinforce, promote |
| `reinforcement.py` | Robust reinforcement with temporal decay, multi-signal types |
| `metadata_feedback.py` | Track metadata effectiveness for LLM feedback |
| `usage_detector.py` | Detect which memories were actually used in responses |
| `query_optimizer.py` | Dynamic query optimization based on usage patterns |
| `cache.py` | Smart write-through cache with pattern-based invalidation |
| `preferences.py` | Temporal preference handling with versioning and conflict resolution |

#### Robust Reinforcement System

```python
from mindcore.v2.flr.reinforcement import (
    RobustReinforcement,
    ReinforcementSignal,
    SignalType,
    SignalSource,
)

# Create reinforcement tracker
reinforcement = RobustReinforcement(decay_half_life_hours=168)  # 1 week

# Apply a signal with full context
signal = ReinforcementSignal(
    signal_type=SignalType.RELEVANCE,  # relevance|usefulness|correctness|timeliness|completeness
    value=0.8,  # -1.0 to 1.0
    source=SignalSource.USER_EXPLICIT,  # user_explicit|user_implicit|llm_evaluation|automated_metric|cross_agent
    context_similarity=0.9,  # How similar was retrieval context?
)
new_score = reinforcement.apply_signal(signal)

# Get effective score with exploration bonus (UCB-like)
effective = reinforcement.get_effective_score(exploration_factor=0.1)

# Check trends
if reinforcement.is_trending_up():
    print("Memory gaining relevance")

# Get breakdown by signal type
breakdown = reinforcement.get_signal_breakdown()
```

#### Signal Types and Sources

**Signal Types** (what aspect of quality):

- `RELEVANCE` - Was the memory relevant to the query?
- `USEFULNESS` - Did it help solve the task?
- `CORRECTNESS` - Was the information accurate?
- `TIMELINESS` - Was it appropriately current?
- `COMPLETENESS` - Did it provide sufficient detail?

**Signal Sources** (reliability weights):

- `USER_EXPLICIT` (1.0) - Direct user feedback (thumbs up/down)
- `USER_IMPLICIT` (0.7) - Inferred from behavior
- `LLM_EVALUATION` (0.5) - LLM self-assessment
- `CROSS_AGENT` (0.6) - Feedback from other agents
- `AUTOMATED_METRIC` (0.3) - System metrics

#### Metadata Feedback Tracking

```python
from mindcore.v2.flr.metadata_feedback import MetadataFeedbackTracker

tracker = MetadataFeedbackTracker()

# Track topic/category effectiveness
tracker.record_topic_retrieval("billing", was_used=True, signal=0.8)
tracker.record_topic_retrieval("general", was_used=False, signal=-0.2)

# Get feedback for LLM metadata extractor
feedback = tracker.get_feedback_for_extractor()
# Returns: {
#   "high_quality_topics": [("billing", 0.85), ("refund", 0.72)],
#   "low_quality_topics": [("general", 0.15), ("misc", 0.10)],
#   "high_quality_categories": [...],
#   "guidance": "Prioritize 'billing', avoid 'general'..."
# }
```

#### Usage Detection

```python
from mindcore.v2.flr.usage_detector import UsageDetector

detector = UsageDetector()

# After LLM generates response, detect which memories were used
result = detector.detect_usage(
    retrieved_memories=memories,
    llm_response=response_text,
    detection_methods=["content_match", "entity_overlap", "semantic_similarity"],
)

# Result includes:
# - used_memories: List of memories referenced in response
# - unused_memories: List of retrieved but unused memories
# - usage_rate: Proportion of memories actually used
# - suggested_signals: Reinforcement signals for each memory
```

#### Query Optimizer

```python
from mindcore.v2.flr.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer()

# Feed usage data over time
optimizer.record_usage(usage_result)

# Get optimized query parameters
optimization = optimizer.optimize_query(
    original_topics=["billing", "refund", "general"],
    original_limit=10,
)
# Returns QueryOptimization:
# - optimized_topics: ["refund", "billing"]  # "general" removed (low effectiveness)
# - boosted_topics: ["refund"]  # High usage rate
# - optimized_limit: 7  # Reduced based on usage patterns
# - reasoning: "Removed 'general' (usage rate: 15%)"

# Get recommendations
recs = optimizer.get_recommendations()
# {
#   "overall_usage_rate": 0.45,
#   "top_performing_topics": [{"topic": "refund", "score": 0.85}],
#   "recommendations": ["Consider removing 'general' from SVL"]
# }
```

#### Smart Cache

Write-through cache with pattern-based invalidation and cache warming:

```python
from mindcore.v2.flr.cache import SmartCache, CacheConfig

# Create cache with configuration
cache = SmartCache(
    storage=storage,
    max_size=10000,
    ttl_seconds=3600,
    warm_high_importance=True,  # Pre-warm important memories
)

# Pattern-based invalidation
cache.invalidate_pattern("user:123:*")  # Invalidate all user memories
cache.invalidate_pattern("*:preference:*")  # Invalidate all preferences

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hits: {stats.hits}, Misses: {stats.misses}")

# Cache warming for session
cache.warm_for_user("user_123", importance_threshold=0.7)
```

#### Preference Manager

Handles mutable preferences with versioning and temporal validity:

```python
from mindcore.v2.flr.preferences import (
    PreferenceManager,
    ConflictResolutionStrategy,
)

prefs = PreferenceManager(storage, flr)

# Set initial preference
pref = prefs.set_preference(
    user_id="user_123",
    key="theme",
    value="User prefers light mode",
    categories=["ui", "display"],
)

# Update preference (creates new version, supersedes old)
new_pref = prefs.update_preference(
    user_id="user_123",
    key="theme",
    value="User now prefers dark mode with purple accents",
)

# Get current preference
current = prefs.get_preference("user_123", "theme")
print(current.content)  # "User now prefers dark mode..."

# View preference history
history = prefs.get_preference_history("user_123", "theme")
for version in history:
    print(f"v{version.turn_index}: {version.content}")

# Handle conflicts from multiple agents
prefs.set_conflict_resolution(ConflictResolutionStrategy.NEWER_WINS)
# Options: NEWER_WINS, HIGHER_CONFIDENCE, LLM_MERGE, HUMAN_REVIEW, AGENT_PRIORITY, KEEP_BOTH
```

---

### 3.2 CLST (Cognitive Long-term Storage Transfer)

**Location:** `mindcore/v2/clst/`

Cold path for persistent storage with hierarchical retrieval and session aggregates.

#### Files

| File | Purpose |
|------|---------|
| `storage.py` | Core CLST protocol - store, compress, sync, transfer |
| `aggregates.py` | Session aggregates with weighted metadata |

#### Session Aggregates

The key insight: Topics and categories have different importance/density within sessions. By tracking weights, we can query relevant sessions without embeddings, then drill into memories.

```python
from mindcore.v2.clst.aggregates import SessionAggregate, WeightCalculator

# Session aggregates are auto-updated when memories are stored
aggregate = SessionAggregate(
    session_id="session_123",
    user_id="user_456",
)

# Aggregates track weighted distributions
aggregate.update_from_memory(memory)

# Query sessions by weighted metadata (no embeddings needed!)
relevance = aggregate.calculate_relevance_score(
    topic_hints=["billing", "refund"],
    category_hints=["complaint"],
    min_importance=0.5,
    recency_weight=0.2,
)

# Get top topics/categories
top_topics = aggregate.get_top_topics(5)  # [("billing", 0.85), ("refund", 0.72), ...]
```

#### Weight Calculation

```
topic_weight = (frequency * 0.4) + (avg_importance * 0.4) + (recency * 0.2)
```

Where:

- `frequency`: How often the topic appears in the session
- `avg_importance`: Average importance of memories with this topic
- `recency`: Exponential decay based on last mention time

#### Hierarchical Query Flow

1. **Query Sessions** by weighted topic/category matching
2. **Get Top Sessions** ranked by relevance score
3. **Query Memories** only from relevant sessions
4. **Return Context** with session summaries and memories

This reduces search space dramatically and eliminates need for per-memory embeddings.

---

### 3.3 SVL (Shared Vocabulary Layer)

**Location:** `mindcore/v2/svl/`

Semantic spine of MindCore - standardized vocabulary for consistent metadata with LLM enforcement.

#### Files

| File | Purpose |
|------|---------|
| `ontology.py` | Core semantic definitions and schema |
| `domains.py` | Domain-specific vocabularies (ecommerce, healthcare, etc.) |
| `sources.py` | Data source mapping (Table, API, MCP) |
| `layer.py` | SharedVocabularyLayer main class |
| `enforced_metadata.py` | LLM-enforced metadata extraction with HistoricalContextNeeded |
| `llm_providers.py` | Provider configs for OpenAI, Claude, Gemini |
| `registry.py` | SVL registry for multi-domain management |

#### Enforced Metadata Extraction

The LLM is **forced** to assign metadata from SVL vocabulary through structured outputs:

```python
from mindcore.v2.svl.enforced_metadata import (
    MetadataExtractor,
    EnforcedMetadata,
    ContextDecision,
    HistoricalContextNeeded,
)

extractor = MetadataExtractor(svl=shared_vocabulary_layer)

# Step 1: LLM decides if historical context is needed
decision_prompt = extractor.get_context_decision_prompt(
    user_message="What about my previous order?",
    session_context="Currently discussing shipping",
)
# LLM returns: {"historical_context_needed": "True", "suggested_topics": ["orders", "shipping"]}

decision = extractor.parse_context_decision(llm_response)
if decision.needs_clst():
    # Query CLST for historical context
    pass

# Step 2: Extract SVL-compliant metadata
extraction_prompt = extractor.get_extraction_prompt(
    user_message="I want a refund for order #123",
    session_id="session_abc",
    user_id="user_456",
)

# LLM must assign from vocabulary:
# - topics: ["refund", "orders"]
# - categories: ["complaint"]
# - message_type: "command"
# - message_intent: "request_action"
# - importance: 0.8
# - urgency: "high"
# - sentiment: "frustration"

metadata, memories = extractor.parse_metadata(llm_response)
```

#### LLM Provider Configurations

Support for latest LLM API features:

```python
from mindcore.v2.svl.llm_providers import (
    OpenAIConfig,
    ClaudeConfig,
    GeminiConfig,
    ReasoningEffort,
    ThinkingLevel,
    ThinkingMode,
)

# OpenAI GPT-5 with Responses API
openai_config = OpenAIConfig(
    model="gpt-5",
    reasoning_effort=ReasoningEffort.HIGH,  # low|medium|high|xhigh
    use_responses_api=True,  # Preserves reasoning across turns
)

# Claude with Extended Thinking
claude_config = ClaudeConfig(
    model="claude-sonnet-4-5-20250514",
    thinking_budget=16000,  # Tokens for internal reasoning
    use_extended_thinking=True,
    use_interleaved_thinking=True,  # Think between tool calls
)

# Gemini 2.5 (thinkingBudget-based)
gemini25_config = GeminiConfig(
    model="gemini-2.5-flash",
    thinking_mode=ThinkingMode.DYNAMIC,  # disabled|dynamic|fixed
)

# Gemini 3 (thinkingLevel-based) - DIFFERENT API!
gemini3_config = GeminiConfig(
    model="gemini-3-flash",
    thinking_level=ThinkingLevel.HIGH,  # minimal|low|medium|high
)
# Note: Cannot mix thinkingBudget and thinkingLevel - API will error
```

#### Feedback Injection (API-Level, No Prompt Modification)

Inject feedback through API mechanisms without modifying user prompts:

```python
from mindcore.v2.svl.llm_providers import (
    ContextInjector,
    FeedbackInjection,
    create_injector_from_flr,
)

# Get feedback from FLR
feedback = flr.get_metadata_feedback_for_extractor()
injector = create_injector_from_flr(feedback)

# For OpenAI - use instructions parameter or developer role
openai_injection = injector.get_openai_injection()
# Returns: {"instructions": "Prioritize 'refund', avoid 'general'...", "messages": [...]}

# For Claude - use system prompt suffix
claude_injection = injector.get_claude_injection()
# Returns: {"system_suffix": "\n[Quality Guidance]\n...", "meta_messages": [...]}

# For Gemini - use systemInstruction
gemini_injection = injector.get_gemini_injection()
# Returns: {"system_suffix": "..."}

# Or annotate the JSON Schema directly
annotated_schema = injector.annotate_schema(original_schema)
# Schema descriptions now include: "Prefer: 'refund' (85%). Avoid: 'general'."
```

---

### 3.4 Context Gateway

**Location:** `mindcore/v2/context/`

Unified entry point for building LLM context from FLR, CLST, and SVL.

```python
from mindcore.v2.context.gateway import ContextGateway, ContextResult

gateway = ContextGateway(
    storage=postgres_storage,
    svl=shared_vocabulary_layer,
    flr_cache_size=1000,
)

# Standard context building
context = gateway.build_context(
    query="What about my order #12345?",
    user_id="user_123",
    session_id="session_abc",
    attention_hints=["orders", "shipping"],
)

# LLM-driven context building (respects HistoricalContextNeeded)
context = gateway.build_context_with_decision(
    query="What about my order?",
    context_decision=ContextDecision(
        historical_context_needed=HistoricalContextNeeded.TRUE,
        suggested_topics=["orders", "shipping"],
    ),
    user_id="user_123",
    session_id="session_abc",
)

# Context result includes:
# - memories: Relevant memories (hierarchically retrieved)
# - current_session: SessionAggregate for current session
# - related_sessions: Other relevant sessions
# - source_data: Auto-fetched SVL data sources
# - query_metadata: SVL-compliant tracking metadata
# - latency_ms: Query timing

# Format for LLM consumption
llm_context = context.to_llm_context(max_memories=20)
```

---

### 3.5 Federation (Multi-Agent Memory)

**Location:** `mindcore/v2/federation/`

Enterprise-grade memory federation for organizations with multiple AI agents.

#### Files

| File | Purpose |
|------|---------|
| `access_control.py` | AccessLevel, AccessScope, AccessPolicy, MemoryACL |
| `namespace.py` | MemoryNamespace, NamespaceHierarchy |
| `federated_clst.py` | FederatedCLST with access control |
| `federated_svl.py` | FederatedSVL with scoped feedback |
| `signal_aggregator.py` | CrossAgentSignalAggregator, TrustPolicy |
| `agent_bridge.py` | AgentMemoryBridge connecting FLR to federation |
| `config.py` | FederationConfig, quick_setup() |

#### Access Levels

```python
from mindcore.v2.federation import AccessLevel

class AccessLevel(IntEnum):
    PRIVATE = 0       # Only the specific agent
    AGENT_TYPE = 10   # Shared with same agent type
    TEAM = 20         # Shared within team
    DEPARTMENT = 30   # Shared within department
    AD_HOC_GROUP = 40 # Custom group membership
    ORGANIZATION = 50 # Visible to all agents in org
    PUBLIC = 100      # Public (rare)
```

#### Quick Setup

```python
from mindcore.v2.federation import quick_setup

# Create federation with departments and teams
federation = quick_setup(
    org_id="acme-corp",
    departments={
        "customer-success": ["support-tier-1", "support-tier-2"],
        "sales": ["inbound", "outbound"],
    },
)

# Create an agent with its own FLR connected to shared CLST/SVL
agent = federation.create_agent(
    agent_id="support-bot-001",
    agent_type="support-bot",
    department="customer-success",
    team="support-tier-1",
)

# Store memory with access control
agent.store(
    content="Customer prefers email contact",
    user_id="customer-123",
    access_level=AccessLevel.TEAM,  # Visible to support-tier-1
)

# Reinforcement signals propagate to other agents
agent.reinforce("memory-id", signal=0.8)
```

#### Cross-Agent Signal Aggregation

```python
from mindcore.v2.federation import (
    CrossAgentSignalAggregator,
    TrustPolicy,
)

aggregator = CrossAgentSignalAggregator(
    trust_policy=TrustPolicy.NAMESPACE_WEIGHTED,  # equal|namespace_weighted|reputation_based|recency_weighted|hierarchical
)

# Signals from multiple agents are aggregated with trust weighting
aggregator.add_signal(
    memory_id="mem_123",
    agent_id="agent_a",
    signal=0.8,
    namespace="support-tier-1",
)
aggregator.add_signal(
    memory_id="mem_123",
    agent_id="agent_b",
    signal=0.6,
    namespace="support-tier-2",
)

# Get aggregated signal
final_signal = aggregator.get_aggregated_signal("mem_123")
```

---

### 3.6 Enterprise Features

**Location:** `mindcore/v2/enterprise/`

Production-ready features for enterprise deployments.

#### Files

| File | Purpose |
|------|---------|
| `audit.py` | Structured audit logging for compliance |
| `encryption.py` | At-rest encryption for sensitive content |
| `observability.py` | OpenTelemetry-based metrics and tracing |
| `rate_limiting.py` | Configurable rate limits with multiple backends |
| `compliance.py` | GDPR/CCPA compliance tools (data export, erasure, anonymization) |

```python
from mindcore.v2.enterprise import (
    # Audit
    AuditLogger,
    AuditEventType,

    # Encryption
    FieldEncryptor,
    EncryptionConfig,
    KeyRotator,

    # Observability
    MindcoreMetrics,
    MindcoreTracer,
    ObservabilityConfig,

    # Rate Limiting
    RateLimiter,
    RateLimitConfig,
)

# Quick setup with all enterprise features
from mindcore import Mindcore

mc = Mindcore(storage="postgresql://...")
mc.enable_observability(ObservabilityConfig(service_name="my-service"))
mc.enable_rate_limiting(RateLimiter(limit="1000/hour"))
mc.enable_audit_logging(AuditLogger(output="file", path="/var/log/mindcore"))
mc.enable_encryption(EncryptionConfig(key_from_env="MINDCORE_ENCRYPTION_KEY"))
```

#### GDPR/CCPA Compliance

```python
from mindcore.v2.enterprise.compliance import (
    ComplianceManager,
    RetentionPolicy,
    AnonymizationStrategy,
)

compliance = ComplianceManager(storage)

# GDPR Article 15: Right of Access (data export)
export = await compliance.export_user_data("user_123")
with open("user_data.json", "w") as f:
    f.write(export.to_json())

# GDPR Article 17: Right to Erasure
result = await compliance.delete_user_data("user_123")
print(f"Deleted {result.memories_deleted} memories")

# Anonymize user data for analytics
result = compliance.anonymize_user_data(
    "user_123",
    strategy=AnonymizationStrategy.PSEUDONYMIZE,
)
# Strategies: PSEUDONYMIZE, HASH, REDACT, AGGREGATE

# Configure retention policies
retention = RetentionPolicy(
    memory_type_policies={
        "episodic": 730,      # 2 years
        "preference": None,   # Forever
        "working": 1,         # 1 day
    },
    default_max_age_days=365,
)
compliance.set_retention_policy(retention)

# Enforce retention (run periodically via cron)
result = compliance.enforce_retention()
print(f"Deleted {result.deleted_count} expired memories")
```

---

### 3.7 Cross-Agent Layer (Legacy)

**Location:** `mindcore/v2/cross_agent/`

Multi-agent memory sharing, synchronization, and intelligent query routing.

> **Note:** For new deployments, prefer the Federation module (`mindcore/v2/federation/`) which provides more granular access control and namespace hierarchy.

```python
from mindcore.v2.cross_agent import (
    CrossAgentLayer,
    RoutingStrategy,
    SyncDirection,
)

layer = CrossAgentLayer(storage)

# Register agents with capabilities
layer.register_agent(
    agent_id="support_bot",
    name="Support Agent",
    capabilities=["customer_support", "billing"],
    teams=["customer_service"],
)

# Query with routing strategy
result = layer.query(
    query="refund requests",
    user_id="user123",
    requesting_agent="sales_bot",
    strategy=RoutingStrategy.CAPABILITY_MATCH,  # broadcast|capability_match|team_first|best_match|round_robin
)
```

---

### 3.8 Patterns

**Location:** `mindcore/v2/patterns/`

Ready-to-use patterns for common use cases.

#### Customer-Facing Agents

```python
from mindcore.v2.patterns.customer_facing import (
    UserMemoryHelper,
    consent_to_access_level,
    mask_pii,
    contains_pii,
)

# Users are just namespaces with access control
helper = UserMemoryHelper(
    federation=federation,
    user_id="visitor_123",
    consent_level="minimal",  # none|minimal|standard|full
)

# Store memory respecting consent
helper.store(
    content="User prefers dark mode",
    memory_type="preference",
)

# Map consent to access level
access = consent_to_access_level("standard")  # Returns AccessLevel.TEAM

# PII helpers
if contains_pii(text):
    safe_text = mask_pii(text)  # "Email: ***@***.com"
```

---

## 4. Storage Backends

### PostgreSQL (Production)

**Location:** `mindcore/v2/storage/postgres.py`

```python
from mindcore import Mindcore, PostgresStorage

storage = PostgresStorage(
    connection_string="postgresql://user:pass@localhost:5432/mindcore",
    pool_size=10,
)
memory = Mindcore(storage=storage)
```

**Features:**

- Connection pooling (psycopg v3)
- Full-text search via `tsvector`
- JSONB for topics, categories, entities
- GIN indexes for array containment
- Session aggregate tables

### SQLite (Development)

**Location:** `mindcore/v2/storage/sqlite.py`

```python
from mindcore import Mindcore

memory = Mindcore(storage="sqlite:///dev.db")  # File
memory = Mindcore(storage="sqlite:///:memory:")  # In-memory
```

**Features:**

- Thread-safe with WAL mode
- FTS5 full-text search
- JSON arrays stored as TEXT

### Time-Based Partitioning (PostgreSQL)

**Location:** `mindcore/v2/storage/partitioning.py`

For large-scale deployments, partition the memories table by time:

```python
from mindcore.v2.storage.partitioning import PartitionManager, PartitionInterval

partitions = PartitionManager(postgres_storage)

# Setup partitioning (one-time)
partitions.setup_partitioning(interval=PartitionInterval.MONTHLY)

# Create partitions for next 3 months
partitions.create_future_partitions(months_ahead=3)

# Get partitioning status
status = partitions.get_status()
print(f"Partitions: {status.total_partitions}")
print(f"Total rows: {status.total_rows}")
print(f"Total size: {status.total_size_pretty}")

# Archive old partitions (move to cold storage)
partitions.archive_partitions(older_than_months=12)

# Drop old partitions (delete data)
partitions.drop_partitions(older_than_months=24)
```

**Benefits:**

- Faster queries when filtering by time
- Parallel query execution across partitions
- Easier data archival and cleanup
- Smaller indexes per partition

---

## 5. API Servers

### MCP Server

**Location:** `mindcore/v2/server/mcp.py`

```python
mcp = memory.get_mcp_server()
tools = mcp.get_tools()
# Returns: store_memory, search_memories, recall, reinforce
```

### REST API

**Location:** `mindcore/v2/server/rest.py`

```python
memory.serve_rest(host="0.0.0.0", port=8000)
```

**Endpoints:**

```
POST   /memories              - Store memory
GET    /memories/{id}         - Get memory
DELETE /memories/{id}         - Delete memory
POST   /memories/search       - Search memories
POST   /recall                - FLR recall
GET    /schema                - Get JSON schema
GET    /stats                 - Get statistics
GET    /health                - Health check
```

---

## 6. File Structure

```
mindcore/
├── __init__.py                     # Main exports
├── py.typed                        # PEP 561 marker
│
├── v2/                             # Core memory layer
│   ├── mindcore.py                 # Main Mindcore class
│   ├── exceptions.py               # Custom exceptions
│   │
│   ├── flr/                        # Fast Local Recall (hot path)
│   │   ├── recall.py               # FLR protocol
│   │   ├── reinforcement.py        # Robust reinforcement with decay
│   │   ├── metadata_feedback.py    # Metadata effectiveness tracking
│   │   ├── usage_detector.py       # Detect memory usage in responses
│   │   ├── query_optimizer.py      # Dynamic query optimization
│   │   ├── cache.py                # Smart write-through cache
│   │   ├── preferences.py          # Temporal preference handling
│   │   └── __init__.py
│   │
│   ├── clst/                       # Cognitive Long-term Storage
│   │   ├── storage.py              # CLST protocol
│   │   ├── aggregates.py           # Session aggregates with weights
│   │   └── __init__.py
│   │
│   ├── svl/                        # Shared Vocabulary Layer
│   │   ├── ontology.py             # Core semantic definitions
│   │   ├── domains.py              # Domain vocabularies
│   │   ├── sources.py              # Data source mapping
│   │   ├── layer.py                # SharedVocabularyLayer
│   │   ├── enforced_metadata.py    # LLM-enforced metadata extraction
│   │   ├── llm_providers.py        # OpenAI/Claude/Gemini configs
│   │   ├── registry.py             # SVL registry
│   │   ├── user_sources/           # User-defined source mappings
│   │   │   ├── topics/
│   │   │   └── categories/
│   │   └── __init__.py
│   │
│   ├── context/                    # Context Gateway
│   │   ├── gateway.py              # Unified context assembly
│   │   └── __init__.py
│   │
│   ├── federation/                 # Multi-Agent Federation
│   │   ├── access_control.py       # AccessLevel, AccessScope, MemoryACL
│   │   ├── namespace.py            # MemoryNamespace, NamespaceHierarchy
│   │   ├── federated_clst.py       # FederatedCLST with access control
│   │   ├── federated_svl.py        # FederatedSVL with scoped feedback
│   │   ├── signal_aggregator.py    # Cross-agent signal aggregation
│   │   ├── agent_bridge.py         # AgentMemoryBridge
│   │   ├── config.py               # FederationConfig, quick_setup()
│   │   └── __init__.py
│   │
│   ├── cross_agent/                # Cross-agent layer (legacy)
│   │   ├── registry.py             # Agent/Team registration
│   │   ├── sharing.py              # Memory sharing and sync
│   │   ├── routing.py              # Attention routing
│   │   ├── layer.py                # CrossAgentLayer
│   │   └── __init__.py
│   │
│   ├── access/                     # Access control
│   │   ├── permissions.py          # AccessController
│   │   └── __init__.py
│   │
│   ├── storage/                    # Storage backends
│   │   ├── base.py                 # BaseStorage interface
│   │   ├── postgres.py             # PostgreSQL backend
│   │   ├── sqlite.py               # SQLite backend
│   │   ├── partitioning.py         # Time-based partitioning for PostgreSQL
│   │   └── __init__.py
│   │
│   ├── server/                     # API servers
│   │   ├── mcp.py                  # MCP server
│   │   ├── rest.py                 # REST API
│   │   └── __init__.py
│   │
│   ├── enterprise/                 # Enterprise features
│   │   ├── audit.py                # Audit logging
│   │   ├── encryption.py           # At-rest encryption
│   │   ├── observability.py        # OpenTelemetry metrics/tracing
│   │   ├── rate_limiting.py        # Rate limiting
│   │   ├── compliance.py           # GDPR/CCPA compliance tools
│   │   └── __init__.py
│   │
│   ├── patterns/                   # Usage patterns
│   │   ├── customer_facing.py      # Customer-facing agent patterns
│   │   └── __init__.py
│   │
│   ├── vocabulary/                 # Vocabulary schema
│   │   ├── schema.py               # VocabularySchema
│   │   └── __init__.py
│   │
│   └── tests/                      # Tests (585 passing)
│       ├── test_mindcore_v2.py
│       ├── test_cross_agent.py
│       ├── test_svl.py
│       ├── test_svl_registry.py
│       ├── test_enterprise.py
│       ├── test_federation.py
│       ├── test_flr_reinforcement.py
│       ├── test_reinforcement_enhanced.py
│       ├── test_extraction_fallback.py
│       ├── test_security_fixes.py
│       ├── test_smart_cache.py
│       ├── test_preferences.py
│       ├── test_compliance.py
│       ├── test_importance_decay.py
│       └── __init__.py
│
├── observability/                  # Optional observability
│   ├── observer.py
│   ├── metrics.py
│   ├── alerts.py
│   ├── quality.py
│   └── __init__.py
│
└── utils/                          # Utilities
    ├── logger.py
    └── __init__.py
```

---

## 7. Common Patterns

### Pattern: LLM with Memory and Feedback Loop

```python
from mindcore import Mindcore
from mindcore.v2.context.gateway import ContextGateway
from mindcore.v2.svl.enforced_metadata import MetadataExtractor
from mindcore.v2.flr.usage_detector import UsageDetector
from mindcore.v2.flr.query_optimizer import QueryOptimizer

# Initialize
memory = Mindcore(storage="postgresql://...")
gateway = ContextGateway(storage=memory.storage, svl=svl)
extractor = MetadataExtractor(svl=svl)
detector = UsageDetector()
optimizer = QueryOptimizer()

# === BEFORE LLM CALL ===

# 1. Get optimized query params based on past effectiveness
optimization = optimizer.optimize_query(
    original_topics=["billing", "refund"],
    original_limit=10,
)

# 2. Build context with optimization
context = gateway.build_context(
    query=user_query,
    user_id=user_id,
    session_id=session_id,
    attention_hints=optimization.optimized_topics,
    memory_limit=optimization.optimized_limit,
)

# 3. Call LLM with context
response = call_llm(
    messages=[...],
    context=context.to_llm_context(),
)

# === AFTER LLM CALL ===

# 4. Detect which memories were actually used
usage = detector.detect_usage(
    retrieved_memories=context.memories,
    llm_response=response,
)

# 5. Apply reinforcement signals
for mem_usage in usage.used_memories:
    memory.reinforce(mem_usage.memory_id, signal=mem_usage.suggested_signal)

# 6. Record usage for future optimization
optimizer.record_usage(usage)

# 7. Store any new memories from response
metadata, new_memories = extractor.parse_metadata(response)
for mem in new_memories:
    memory.store(**mem)
```

### Pattern: Multi-Agent with Federation

```python
from mindcore.v2.federation import quick_setup, AccessLevel

# Setup organization
federation = quick_setup(
    org_id="acme",
    departments={
        "support": ["tier-1", "tier-2", "escalation"],
        "sales": ["inbound", "outbound"],
    },
)

# Create agents
support_agent = federation.create_agent(
    agent_id="support-001",
    agent_type="support-bot",
    department="support",
    team="tier-1",
)

sales_agent = federation.create_agent(
    agent_id="sales-001",
    agent_type="sales-bot",
    department="sales",
    team="inbound",
)

# Support agent stores customer preference
support_agent.store(
    content="Customer interested in premium plan",
    user_id="customer-123",
    access_level=AccessLevel.DEPARTMENT,  # Visible to all support teams
)

# Sales agent can query across teams
context = sales_agent.query(
    query="customer interests",
    user_id="customer-123",
)

# Reinforcement signals propagate
support_agent.reinforce("memory-id", signal=0.8)
# Signal is weighted and aggregated with other agent signals
```

---

## 8. Testing

```bash
# All tests (585 passing, 61% coverage)
pytest mindcore/v2/tests/ testing/tests/ -v

# Run with coverage
pytest mindcore/v2/tests/ testing/tests/ --cov=mindcore --cov-report=html

# Specific test files
pytest mindcore/v2/tests/test_flr_reinforcement.py -v
pytest mindcore/v2/tests/test_federation.py -v
pytest mindcore/v2/tests/test_enterprise.py -v
pytest mindcore/v2/tests/test_smart_cache.py -v
pytest mindcore/v2/tests/test_preferences.py -v
pytest mindcore/v2/tests/test_compliance.py -v
```

---

## 9. Configuration

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

---

## 10. Key Concepts

### Hierarchical Retrieval (No Embeddings Required)

Instead of embedding every memory and doing vector search:

1. **Store memories with SVL metadata** (topics, categories, importance)
2. **Aggregate at session level** (weighted topic distributions)
3. **Query sessions first** by weighted metadata matching
4. **Drill into memories** from relevant sessions only

This reduces search space by 10-100x and eliminates embedding costs.

### HistoricalContextNeeded

The LLM decides whether to query CLST:

- `TRUE`: Query needs historical context (past interactions, preferences, patterns)
- `FALSE`: Query can be answered from current session only

This prevents unnecessary CLST queries for simple questions.

### Reinforcement Signal Flow

```
User Feedback → Signal → Memory Reinforcement → Score Update
                              ↓
                    Metadata Feedback Tracker
                              ↓
                      Query Optimizer
                              ↓
                   Future Query Optimization
                              ↓
                    LLM Context Injection
```

Reinforcement signals flow:

1. Into memory scores (better ranking)
2. Into metadata effectiveness tracking (which topics work)
3. Into query optimization (adjust limits, filter topics)
4. Back to LLM (annotated schemas, system instructions)

---

*This document is the source of truth for MindCore. Update it when making architectural changes.*
