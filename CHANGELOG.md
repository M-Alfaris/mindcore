# Changelog

All notable changes to Mindcore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No changes yet._

## [0.1.0] - 2025-12-02

### Features

- **Core Framework**
  - `Mindcore` client for intelligent memory and context management
  - SQLite and PostgreSQL database support
  - Disk-backed persistent caching with TTL support
  - Background enrichment worker with persistent queue

- **AI Agents**
  - `EnrichmentAgent` - Extracts metadata (topics, categories, intent, sentiment) from messages
  - `SmartContextAgent` - Tool-calling agent for intelligent context assembly
  - `ContextAssemblerAgent` - Assembles relevant context for AI responses
  - `SummarizationAgent` - Generates thread summaries
  - `TrivialDetector` - Detects trivial messages to skip enrichment

- **LLM Providers**
  - OpenAI provider with tool calling support
  - Llama.cpp provider for local model inference
  - Automatic provider selection and fallback

- **Context Management**
  - Smart context retrieval with early conversation handling
  - Vocabulary management for controlled metadata values
  - Cache invalidation with topic-based indexing
  - Adaptive user preferences learning

- **API & CLI**
  - FastAPI-based REST API
  - CLI for model management and server control
  - Dashboard routes for monitoring

- **Integrations**
  - LangChain adapter
  - LlamaIndex adapter
  - Vector store support (Chroma, Pinecone, pgvector)
  - External connectors (orders, billing)

- **Developer Experience**
  - Configurable structured logging with category filtering
  - Prometheus metrics export (optional)
  - Pre-commit hooks configuration
  - Ruff linting and formatting
  - Google-style docstrings
  - Type hints with py.typed marker

### Security

- Bandit security scanning integration
- Input validation and sanitization
- Safe JSON parsing with fallbacks

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2025-12-02 | Initial release |

[Unreleased]: https://github.com/M-Alfaris/mindcore/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/M-Alfaris/mindcore/releases/tag/v0.1.0
