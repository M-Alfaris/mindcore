# Mindcore Critical Test Coverage Plan

## Overview

This plan prioritizes test coverage based on component criticality, usage frequency, and impact on system reliability.

## Priority 1: CRITICAL (Target: 85%+ coverage)

### 1.1 `mindcore/core/schemas.py` (Current: 63.6%)

**Why Critical:** Every module in mindcore depends on these data models.

- [ ] Test `Message` creation, serialization, and deserialization
- [ ] Test `MessageMetadata.is_enriched` property logic
- [ ] Test `UserPreferences` amendable field updates
- [ ] Test `UserPreferences.add_to_list` and `remove_from_list`
- [ ] Test `ThreadSummary.to_context_string`
- [ ] Test `MetadataSchema` validation methods

### 1.2 `mindcore/core/vocabulary.py` (Current: 53.3%)

**Why Critical:** Central vocabulary control for all enrichment and retrieval.

- [ ] Test topic registration and validation
- [ ] Test category registration and validation
- [ ] Test topic/category mappings and resolution
- [ ] Test connector topic registration
- [ ] Test `get_connector_for_topics` routing
- [ ] Test intent and sentiment resolution
- [ ] Test `to_prompt_list` and `to_json_schema`
- [ ] Test `load_from_json` external vocabulary loading
- [ ] Test singleton pattern with `get_vocabulary`/`reset_vocabulary`

### 1.3 `mindcore/context_layer.py` (Current: 25.3%)

**Why Critical:** The core abstraction for context assembly.

- [ ] Test `ContextLayerConfig` factory methods (basic, standard, advanced, full)
- [ ] Test `ContextLayer` initialization with different configs
- [ ] Test `get_recent_messages` cache-first fallback
- [ ] Test `search_messages` with and without vector store
- [ ] Test factory methods (basic, with_connectors, with_vector_store, full)
- [ ] Test `health_check` method
- [ ] Test context manager protocol

### 1.4 `mindcore/agents/trivial_detector.py` (Current: 34.1%)

**Why Critical:** Cost optimization - detects trivial messages to skip LLM calls.

- [ ] Test pattern matching for all trivial categories
- [ ] Test confidence calculation
- [ ] Test `auto_enrich` method
- [ ] Test category-to-intent/sentiment mapping
- [ ] Test importance calculation
- [ ] Test singleton pattern

### 1.5 `mindcore/__init__.py` MindcoreClient (Current: 17.9%)

**Why Critical:** Main public API - the entry point for all users.

- [ ] Test initialization with SQLite
- [ ] Test `ingest()` flow (immediate return, background enrichment)
- [ ] Test `get_context()` flow
- [ ] Test vocabulary integration
- [ ] Test multi-agent mode initialization
- [ ] Test worker thread management
- [ ] Test graceful shutdown

## Priority 2: HIGH (Target: 75%+ coverage)

### 2.1 `mindcore/agents/summarization_agent.py` (Current: 16.3%)

- [ ] Test summarization prompt generation
- [ ] Test key facts extraction
- [ ] Test thread summary creation

### 2.2 `mindcore/core/db_manager.py` (Current: 33.9%)

- [ ] Test connection pooling behavior
- [ ] Test message CRUD operations
- [ ] Test search_by_relevance
- [ ] Test error handling and retry logic

### 2.3 `mindcore/core/multi_agent.py` (Current: 23.9%)

- [ ] Test agent registration
- [ ] Test memory sharing modes (ISOLATED, SHARED, FEDERATED)
- [ ] Test access filtering
- [ ] Test visibility controls

## Priority 3: MEDIUM (Target: 60%+ coverage)

### 3.1 `mindcore/core/cache_invalidation.py` (Current: 22.8%)

- [ ] Test cache registration
- [ ] Test invalidation by topic/age
- [ ] Test notification callbacks

### 3.2 `mindcore/core/retention_policy.py` (Current: 17.5%)

- [ ] Test memory tier classification
- [ ] Test importance decay
- [ ] Test policy enforcement

### 3.3 `mindcore/core/adaptive_preferences.py` (Current: 13.7%)

- [ ] Test preference signal detection
- [ ] Test learning from message patterns
- [ ] Test preference updates

## Implementation Order

1. **Phase 1:** schemas.py + vocabulary.py (foundation for everything else)
2. **Phase 2:** trivial_detector.py + context_layer.py (core functionality)
3. **Phase 3:** MindcoreClient ingest/get_context paths
4. **Phase 4:** summarization_agent.py + db_manager.py
5. **Phase 5:** multi_agent.py + cache_invalidation.py + retention_policy.py

## Success Criteria

- All Priority 1 components reach 85%+ coverage
- All Priority 2 components reach 75%+ coverage
- All Priority 3 components reach 60%+ coverage
- Overall project coverage reaches 65%+
- All critical paths have explicit test cases
- Edge cases and error handling are covered
