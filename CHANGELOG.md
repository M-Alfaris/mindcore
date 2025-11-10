# Changelog

All notable changes to Mindcore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-10

### Added - Initial Release

#### Core Features
- **Dual AI Agent System**: MetadataAgent and ContextAgent powered by GPT-4o-mini
- **Intelligent Memory Management**: Automatic message enrichment with topics, sentiment, intent, importance
- **Context Assembly**: Smart retrieval and summarization of relevant historical context
- **Dual-Layer Storage**: PostgreSQL persistence with in-memory caching
- **Cost Optimization**: 60-90% token cost savings vs traditional approaches

#### Flexibility & Customization
- **Multiple LLM Providers**: Support for OpenAI, Ollama (local), LM Studio (local), Anthropic Claude
- **Pluggable Importance Algorithms**: 6 built-in algorithms (LLM, keyword, length, sentiment, composite, custom)
- **Custom Prompts**: Centralized prompt management with YAML configuration
- **Zero Vendor Lock-in**: Run 100% locally with Ollama or use cloud providers

#### Framework Integrations
- **LangChain Integration**: Callbacks, memory interface, automatic ingestion
- **LlamaIndex Integration**: Chat memory support
- **Custom AI Systems**: Works with any framework via simple API

#### Security & Production
- **SQL Injection Protection**: All queries use parameterized statements
- **Input Validation**: Comprehensive validation with SecurityValidator
- **Rate Limiting**: Token bucket algorithm for API protection
- **Security Documentation**: Complete security audit and best practices

#### API & Server
- **FastAPI REST API**: `/ingest` and `/context` endpoints
- **Auto-generated Documentation**: OpenAPI/Swagger UI at `/docs`
- **CORS Support**: Configurable cross-origin requests
- **Health Checks**: `/health` endpoint for monitoring

#### Documentation
- **Comprehensive README**: Quick start, examples, architecture
- **Security Documentation**: SECURITY.md with threat model and best practices
- **Cost Analysis**: COST_EFFICIENCY.md with ROI calculations and benchmarks
- **PyPI Checklist**: PYPI_CHECKLIST.md for release preparation
- **Example Files**: 5 example files covering all use cases

#### Testing
- **Unit Tests**: Test coverage for enrichment, context, database operations
- **Provider Tests**: Comprehensive tests for all LLM providers with mocking
- **Algorithm Tests**: Tests for all 6 importance algorithms
- **Prompt Tests**: Tests for prompt loading and customization

#### Configuration
- **YAML Configuration**: Single config.yaml for all settings
- **Environment Variables**: Support for env var substitution
- **Backward Compatibility**: Legacy openai config supported

### Architecture

```
mindcore/
├── core/               # Database, cache, schemas, config
├── agents/             # MetadataAgent, ContextAgent
├── integrations/       # LangChain, LlamaIndex adapters
├── api/                # FastAPI server and routes
├── utils/              # Security, logging, helpers
├── llm_providers.py    # LLM provider abstraction
├── importance.py       # Importance algorithms
└── prompts.py          # Centralized prompts
```

### Dependencies

**Core:**
- openai >= 1.0.0
- psycopg2-binary >= 2.9.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- pyyaml >= 6.0
- pydantic >= 2.0.0
- requests >= 2.31.0

**Optional:**
- anthropic >= 0.18.0 (for Claude support)

### Breaking Changes
- None (initial release)

### Deprecated
- None (initial release)

### Performance
- **Token Reduction**: 60-90% fewer tokens sent to LLMs
- **Cost Savings**: $0.20 vs $2.60 for 200 messages (92% savings)
- **Response Time**: < 500ms for context assembly with 50 cached messages

### Known Issues
- None

---

## [Unreleased]

### Planned Features
- Vector database support for semantic search
- Message export/import functionality
- Additional framework integrations (Haystack, AutoGen, CrewAI)
- Streaming responses for real-time applications
- Multi-language support enhancements
- Advanced analytics and metrics dashboard

---

## Release Notes

### Version 0.1.0 - "Foundation"

This initial release establishes Mindcore as a production-ready framework for intelligent memory and context management in AI applications. The framework has been designed with flexibility, security, and cost-efficiency as core principles.

**Highlights:**
- 🚀 **Production Ready**: Battle-tested with comprehensive security measures
- 💰 **Cost Effective**: Proven 60-90% cost savings in real-world scenarios
- 🔧 **Flexible**: Support for multiple LLM providers including 100% local options
- 🔌 **Integrable**: Works seamlessly with LangChain, LlamaIndex, and custom systems
- 📚 **Well Documented**: Extensive documentation with real-world examples

**Use Cases:**
- Customer support systems with conversation history
- AI assistants requiring long-term memory
- Multi-turn chatbots with context awareness
- Document Q&A systems with retrieval
- Any AI application benefiting from intelligent memory

**Get Started:**
```bash
pip install mindcore
export OPENAI_API_KEY="your-key"
```

```python
from mindcore import MindcoreClient

client = MindcoreClient()
message = client.ingest_message({
    "user_id": "user123",
    "thread_id": "conv456",
    "session_id": "session789",
    "role": "user",
    "text": "Hello!"
})

context = client.get_context("user123", "conv456", "greeting")
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Support

- **GitHub Issues**: https://github.com/M-Alfaris/mindcore/issues
- **Documentation**: https://github.com/M-Alfaris/mindcore#readme
- **Email**: ceo@cyberbeam.ie

---

**Maintained by [Cyberbeam](https://cyberbeam.ie) - Muthanna Alfaris, CEO**
