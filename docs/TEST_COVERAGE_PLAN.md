# Test Coverage Improvement Plan

## Current State

- **Overall Coverage**: 17%
- **Test Files**: 3 (test_context.py, test_db.py, test_enrichment.py)
- **Test Functions**: 14 total

## Coverage Targets

| Component | Current | Target |
|-----------|---------|--------|
| Core (**init**.py, context_layer.py) | 18-25% | 85%+ |
| Agents | 10-82% | 80%+ |
| Database Managers | 9-34% | 90%+ |
| LLM Providers | 12-23% | 80%+ |
| VectorStores | 0% | 75%+ |
| Connectors | 0% | 70%+ |
| CLI | 0% | 70%+ |
| Utils | 14-35% | 60%+ |
| **Overall** | **17%** | **75%+** |

---

## Phase 1: Critical Infrastructure (Priority: CRITICAL)

### 1.1 SmartContextAgent Tests

**File**: `test_agents/test_smart_context_agent.py`
**Current**: 10% | **Target**: 80%+

Tests needed:

- [ ] Tool calling mechanism with LLM
- [ ] Early conversation detection (MIN_MESSAGES thresholds)
- [ ] Context assembly with vocabulary integration
- [ ] Handling multiple tool types (messages, history, metadata, summaries)
- [ ] Error handling and graceful degradation
- [ ] Token optimization logic
- [ ] Provider fallback (OpenAI ↔ local)

### 1.2 AsyncClient Tests

**File**: `test_async_client.py`
**Current**: 0% | **Target**: 70%+

Tests needed:

- [ ] Message ingestion (async)
- [ ] Context retrieval (async)
- [ ] Lifecycle management (connect/disconnect)
- [ ] Error handling
- [ ] Agent coordination in async context
- [ ] Integration with async database managers

### 1.3 AccessControlManager Tests

**File**: `test_core/test_access_control.py`
**Current**: 0% | **Target**: 75%+

Tests needed:

- [ ] Agent registration and authentication
- [ ] Visibility tiers (private, shared, public)
- [ ] Access control list (ACL) enforcement
- [ ] Permission checking across agent groups
- [ ] Knowledge access with visibility enforcement
- [ ] Multi-tenant isolation

---

## Phase 2: Database Layer (Priority: HIGH)

### 2.1 SQLiteManager Tests

**File**: `test_core/test_sqlite_manager.py`
**Current**: 9% | **Target**: 80%+

Tests needed:

- [ ] Schema creation and initialization
- [ ] Message CRUD operations
- [ ] Query building with filters
- [ ] Thread safety
- [ ] Datetime normalization
- [ ] Batch operations
- [ ] Error recovery

### 2.2 AsyncDatabaseManager Tests

**File**: `test_core/test_async_db.py`
**Current**: 0% | **Target**: 70%+

Tests needed:

- [ ] Async connection management
- [ ] Message CRUD operations (async)
- [ ] Connection pooling
- [ ] Concurrent access patterns
- [ ] Error recovery

### 2.3 CacheManager Tests

**File**: `test_core/test_cache_manager.py`
**Current**: 12% | **Target**: 75%+

Tests needed:

- [ ] TTL expiration behavior
- [ ] LRU eviction
- [ ] Thread safety under concurrent access
- [ ] Hit/miss rate tracking
- [ ] Message ordering preservation

---

## Phase 3: LLM & VectorStore Layer (Priority: HIGH)

### 3.1 OpenAIProvider Tests

**File**: `test_llm/test_openai_provider.py`
**Current**: 12% | **Target**: 80%+

Tests needed:

- [ ] Provider initialization
- [ ] Message generation (mocked API)
- [ ] Retry with exponential backoff
- [ ] Rate limit handling
- [ ] JSON mode responses
- [ ] Custom base_url support (vLLM, Ollama)
- [ ] Tool calling

### 3.2 LlamaCppProvider Tests

**File**: `test_llm/test_llama_provider.py`
**Current**: 23% | **Target**: 70%+

Tests needed:

- [ ] Model loading
- [ ] Inference with various parameters
- [ ] GPU/CPU selection
- [ ] Memory management

### 3.3 VectorStore Tests

**Files**: `test_vectorstores/test_*.py`
**Current**: 0% | **Target**: 75%+

For each implementation (base, chroma, pgvector, pinecone):

- [ ] Connection/initialization
- [ ] Document indexing (add_texts, add_documents)
- [ ] Similarity search
- [ ] Delete operations
- [ ] Metadata filtering
- [ ] Error handling

### 3.4 Embeddings Tests

**File**: `test_vectorstores/test_embeddings.py`
**Current**: 0% | **Target**: 75%+

Tests needed:

- [ ] OpenAIEmbeddings initialization and generation
- [ ] SentenceTransformerEmbeddings (local)
- [ ] OllamaEmbeddings
- [ ] CustomEmbeddings wrapper
- [ ] Batch processing

---

## Phase 4: Connectors & CLI (Priority: MEDIUM)

### 4.1 Connector Framework Tests

**Files**: `test_connectors/test_*.py`
**Current**: 0% | **Target**: 70%+

Tests needed:

- [ ] BaseConnector interface
- [ ] ConnectorRegistry registration/lookup
- [ ] BillingConnector (mocked external API)
- [ ] OrdersConnector (mocked external API)
- [ ] Topic mapping to vocabulary
- [ ] Caching behavior

### 4.2 CLI Tests

**Files**: `test_cli/test_*.py`
**Current**: 0% | **Target**: 70%+

Tests needed:

- [ ] Command parsing (Click framework)
- [ ] `mindcore init` command
- [ ] `mindcore download` command
- [ ] `mindcore status` command
- [ ] Error messages and help text
- [ ] Configuration file handling

---

## Phase 5: Integrations & Utils (Priority: LOW)

### 5.1 Framework Adapters

**Files**: `test_integrations/test_*.py`
**Current**: 0% | **Target**: 50%+

Tests needed:

- [ ] LangChain adapter (message conversion)
- [ ] LlamaIndex adapter (context injection)

### 5.2 Utility Modules

**Current**: 14-35% | **Target**: 60%+

- [ ] Cost analysis calculations
- [ ] Tokenizer utilities
- [ ] Timezone handling
- [ ] Security validators

---

## Test Categories

### Unit Tests

- Individual class/function behavior
- Input validation
- Error conditions
- Edge cases
- Mock all external dependencies

### Integration Tests

- SmartContextAgent + VocabularyManager + Database
- AsyncClient + agents + async database
- Connector Registry + vocabulary integration
- VectorStore implementations with embeddings

### Database Tests

- Schema creation and migrations
- Thread safety
- Connection pooling
- ACID compliance

### Async/Concurrency Tests

- Thread safety of managers
- Async context manager lifecycle
- Concurrent message processing
- Race condition detection

### Security Tests

- Access control enforcement
- SQL injection prevention (parameterized queries)
- API key handling

---

## Test File Structure

```text
mindcore/tests/
├── conftest.py                        # Shared fixtures
├── test_context.py                    # (existing)
├── test_db.py                         # (existing)
├── test_enrichment.py                 # (existing)
├── test_async_client.py               # NEW
├── test_context_layer.py              # NEW
├── test_knowledge_store.py            # NEW
│
├── test_agents/
│   ├── __init__.py
│   ├── test_smart_context_agent.py    # NEW - CRITICAL
│   ├── test_trivial_detector.py       # NEW
│   └── test_summarization_agent.py    # NEW
│
├── test_core/
│   ├── __init__.py
│   ├── test_access_control.py         # NEW - CRITICAL
│   ├── test_cache_manager.py          # NEW
│   ├── test_sqlite_manager.py         # NEW
│   ├── test_async_db.py               # NEW
│   ├── test_vocabulary.py             # NEW
│   └── test_preferences_manager.py    # NEW
│
├── test_llm/
│   ├── __init__.py
│   ├── test_openai_provider.py        # NEW
│   └── test_llama_provider.py         # NEW
│
├── test_vectorstores/
│   ├── __init__.py
│   ├── test_base_vectorstore.py       # NEW
│   ├── test_chroma.py                 # NEW
│   ├── test_pgvector.py               # NEW
│   ├── test_pinecone.py               # NEW
│   └── test_embeddings.py             # NEW
│
├── test_connectors/
│   ├── __init__.py
│   ├── test_base_connector.py         # NEW
│   ├── test_registry.py               # NEW
│   ├── test_billing.py                # NEW
│   └── test_orders.py                 # NEW
│
├── test_cli/
│   ├── __init__.py
│   ├── test_main.py                   # NEW
│   └── test_models.py                 # NEW
│
└── test_integrations/
    ├── __init__.py
    ├── test_langchain_adapter.py      # NEW
    └── test_llamaindex_adapter.py     # NEW
```

---

## Shared Test Fixtures (conftest.py)

```python
# Key fixtures to implement:

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for LLM tests."""
    pass

@pytest.fixture
def mock_llm_provider():
    """Generic mock LLM provider."""
    pass

@pytest.fixture
def sqlite_db(tmp_path):
    """Temporary SQLite database."""
    pass

@pytest.fixture
def sample_messages():
    """Standard set of test messages."""
    pass

@pytest.fixture
def vocabulary_manager():
    """Configured VocabularyManager."""
    pass

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    pass
```

---

## Execution Order

1. **Week 1-2**: Phase 1 (SmartContextAgent, AsyncClient, AccessControl)
2. **Week 3-4**: Phase 2 (Database managers, Cache)
3. **Week 5-6**: Phase 3 (LLM providers, VectorStores)
4. **Week 7-8**: Phase 4-5 (Connectors, CLI, Integrations)

---

## Success Metrics

- [ ] Overall coverage reaches 75%+
- [ ] All CRITICAL modules have 80%+ coverage
- [ ] All tests pass in CI/CD pipeline
- [ ] No regressions from new tests
- [ ] Test execution time < 60 seconds
