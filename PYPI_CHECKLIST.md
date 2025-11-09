# PyPI Release Checklist

## Current Status: NOT READY

Mindcore is well-structured and functional, but needs these improvements before PyPI release.

---

## ✅ Completed

- [x] Core functionality (MindcoreClient, agents, storage)
- [x] Security hardening (SQL injection protection, validation, rate limiting)
- [x] Framework integrations (LangChain, LlamaIndex)
- [x] Comprehensive documentation (README, SECURITY, COST_EFFICIENCY)
- [x] Cost analysis and benchmarking
- [x] Clean API with intuitive naming
- [x] Type hints throughout
- [x] Centralized prompts (prompts.py)
- [x] Pluggable importance algorithms (importance.py)
- [x] LLM provider abstraction (llm_providers.py)

---

## ⚠️ Required Before PyPI

### 1. Configuration System Enhancement

**Status:** Partial - needs updates for new features

**Tasks:**
- [ ] Add `llm_provider` config option (openai, ollama, lmstudio, etc.)
- [ ] Add `importance_algorithm` config option
- [ ] Add `custom_prompts_path` config option
- [ ] Update config.yaml with new options
- [ ] Document all configuration options

**Example config.yaml needed:**
```yaml
llm:
  provider: openai  # or ollama, lmstudio, anthropic
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: null  # for custom endpoints

importance:
  algorithm: llm  # or keyword, length, sentiment, composite

prompts:
  custom_path: null  # optional path to custom prompts file
```

### 2. Update Agents to Use New Systems

**Status:** Not started

**Tasks:**
- [ ] Update EnrichmentAgent to use prompts.py
- [ ] Update EnrichmentAgent to use llm_providers.py
- [ ] Update EnrichmentAgent to use importance.py algorithms
- [ ] Update ContextAssemblerAgent to use prompts.py
- [ ] Update ContextAssemblerAgent to use llm_providers.py
- [ ] Update BaseAgent to accept LLMProvider instead of just OpenAI

### 3. Testing

**Status:** Basic tests exist, need expansion

**Tasks:**
- [ ] Add tests for prompts.py
- [ ] Add tests for importance.py algorithms
- [ ] Add tests for llm_providers.py (with mocking)
- [ ] Add integration tests for Ollama provider
- [ ] Add integration tests for custom prompts
- [ ] Add tests for all importance algorithms
- [ ] Increase test coverage to >80%

### 4. Documentation

**Status:** Good, needs updates for new features

**Tasks:**
- [ ] Document LLM provider options in README
- [ ] Document importance algorithms in README
- [ ] Document custom prompts in README
- [ ] Add examples for Ollama usage
- [ ] Add examples for custom importance algorithms
- [ ] Add examples for custom prompts
- [ ] Create CONFIGURATION.md guide

### 5. Dependencies

**Status:** Needs clarification

**Tasks:**
- [ ] Make `anthropic` optional dependency
- [ ] Make `requests` optional (only for Ollama)
- [ ] Document optional dependencies clearly
- [ ] Test installation with minimal dependencies
- [ ] Update setup.py/pyproject.toml with optional groups

**Example:**
```toml
[project.optional-dependencies]
local = ["requests>=2.31.0"]  # For Ollama
anthropic = ["anthropic>=0.7.0"]  # For Claude
all = ["requests>=2.31.0", "anthropic>=0.7.0"]
```

### 6. Error Handling

**Status:** Needs improvement

**Tasks:**
- [ ] Better error messages for missing dependencies
- [ ] Graceful degradation when LLM provider unavailable
- [ ] Connection retry logic for Ollama/LM Studio
- [ ] Clear error for invalid importance algorithm
- [ ] Clear error for invalid LLM provider

### 7. Examples

**Status:** Good, needs expansion

**Tasks:**
- [ ] Add example: Using Ollama (examples_ollama.py)
- [ ] Add example: Custom importance algorithm
- [ ] Add example: Custom prompts
- [ ] Add example: LM Studio integration
- [ ] Update existing examples with new features

### 8. Package Metadata

**Status:** Needs updates

**Tasks:**
- [ ] Update pyproject.toml with accurate description
- [ ] Add keywords for discoverability
- [ ] Add project URLs (docs, issues, source)
- [ ] Update author information
- [ ] Add classifiers for Python versions
- [ ] Verify license file

### 9. Performance

**Status:** Not tested

**Tasks:**
- [ ] Benchmark with Ollama vs OpenAI
- [ ] Benchmark importance algorithms
- [ ] Add performance tips to documentation
- [ ] Test with large conversation histories (1000+ messages)
- [ ] Profile and optimize hot paths

### 10. Security Review

**Status:** Good, needs final check

**Tasks:**
- [ ] Review all user inputs for validation
- [ ] Ensure no API keys in logs
- [ ] Review file permissions
- [ ] Check for SQL injection in new code
- [ ] Security audit of LLM provider implementations

---

## 🎯 Nice to Have (Not Blockers)

### Features

- [ ] Vector database support for semantic search
- [ ] Message export/import functionality
- [ ] Conversation analytics dashboard
- [ ] Web UI for configuration
- [ ] Docker compose for easy deployment

### Integrations

- [ ] Haystack integration
- [ ] AutoGen integration
- [ ] CrewAI integration

### Documentation

- [ ] Video tutorials
- [ ] Interactive examples
- [ ] API reference with Sphinx
- [ ] Migration guide from other solutions

---

## 📋 Release Process

### Pre-Release

1. Complete all "Required Before PyPI" tasks
2. Bump version to 0.1.0 (or 1.0.0 if ready)
3. Update CHANGELOG.md
4. Run full test suite
5. Test installation from source
6. Review all documentation

### Release

1. Build distribution:
   ```bash
   python -m build
   ```

2. Test on TestPyPI:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. Test installation from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ mindcore
   ```

4. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

5. Create GitHub release
6. Update documentation with installation from PyPI
7. Announce release

---

## 🔧 Quick Wins (Do These First)

1. **Update agents to use prompts.py** - 1 hour
2. **Add Ollama example** - 30 minutes
3. **Update config.yaml** - 15 minutes
4. **Document LLM providers in README** - 30 minutes
5. **Add importance algorithm docs** - 30 minutes

Total: ~3 hours for major improvements!

---

## 📊 Estimated Timeline

- **Quick wins:** 1 day
- **Required tasks:** 1-2 weeks
- **Nice to have:** Ongoing

**Recommendation:** Focus on "Required Before PyPI" + "Quick Wins" first, then release v0.1.0 to PyPI.

---

## 📞 Questions to Answer Before Release

1. **Versioning:** Start with 0.1.0 or go straight to 1.0.0?
   - **Recommendation:** 0.1.0 (signal it's early but functional)

2. **Name:** Keep "mindcore" or rename?
   - **Recommendation:** Keep it, it's good!

3. **License:** MIT is good?
   - **Recommendation:** Yes, MIT is perfect for open source

4. **Support:** How to handle issues/support?
   - **Recommendation:** GitHub issues only for v0.1.0

5. **Breaking changes:** How to handle in future?
   - **Recommendation:** Semver (0.x.x for pre-1.0, then strict semver)

---

**Last Updated:** 2024-11-08
