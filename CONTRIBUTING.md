# Contributing to Mindcore

Thank you for your interest in contributing to Mindcore! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

### Our Standards

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Collaborative**: Work together towards common goals
- **Be Open**: Welcome newcomers and diverse perspectives

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Spam or irrelevant content
- Violation of privacy or confidentiality

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- Git
- GitHub account
- OpenAI API key (or local LLM setup)

### Quick Start

1. **Fork the repository**
   ```bash
   # Visit https://github.com/M-Alfaris/mindcore
   # Click "Fork" in the top right
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mindcore.git
   cd mindcore
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/M-Alfaris/mindcore.git
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### 1. Install Dependencies

```bash
# Install package in development mode
pip install -e ".[dev]"

# Or with all optional dependencies
pip install -e ".[dev,all]"
```

### 2. Set Up Database

```bash
# Create database
createdb mindcore_dev

# Run schema
psql -d mindcore_dev -f schema.sql
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# - OPENAI_API_KEY (required)
# - DB_* variables (if not using defaults)
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Check code quality
black --check mindcore/
flake8 mindcore/
mypy mindcore/
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 🐛 Bug Reports
- Check existing issues first
- Provide clear reproduction steps
- Include error messages and logs
- Specify your environment (OS, Python version, etc.)

#### ✨ Feature Requests
- Describe the problem you're trying to solve
- Explain your proposed solution
- Consider backward compatibility
- Discuss with maintainers before implementing large features

#### 📝 Documentation
- Fix typos or unclear explanations
- Add examples or use cases
- Improve API documentation
- Translate documentation (future)

#### 🧪 Tests
- Add missing test coverage
- Improve existing tests
- Add integration tests
- Performance benchmarks

#### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Refactoring

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 100 characters (not 79)
# Use black for formatting
# Use type hints where possible
```

### Code Formatting

```bash
# Format code with black
black mindcore/

# Sort imports with isort
isort mindcore/

# Check with flake8
flake8 mindcore/

# Type checking with mypy
mypy mindcore/
```

### Naming Conventions

```python
# Classes: PascalCase
class MetadataAgent:
    pass

# Functions/methods: snake_case
def enrich_message():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_TOKENS = 1000

# Private: _leading_underscore
def _internal_method():
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Longer description if needed. Explain what the function does,
    any important behavior, and edge cases.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param2 is negative.

    Example:
        >>> example_function("test", 42)
        True
    """
    pass
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Dict, Optional, Any

def process_messages(
    messages: List[Dict[str, Any]],
    max_results: Optional[int] = None
) -> List[Message]:
    """Process messages and return results."""
    pass
```

---

## Testing Guidelines

### Writing Tests

```python
# Use pytest for all tests
# One test file per module
# Clear, descriptive test names

def test_enrichment_extracts_topics():
    """Test that enrichment correctly extracts topics."""
    # Arrange
    message_dict = {
        "user_id": "test",
        "thread_id": "test",
        "session_id": "test",
        "role": "user",
        "text": "I want to learn about AI agents"
    }

    # Act
    result = agent.process(message_dict)

    # Assert
    assert "AI" in result.metadata.topics
    assert "agents" in result.metadata.topics
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest mindcore/tests/test_enrichment.py

# Run with coverage
pytest --cov=mindcore --cov-report=html

# Run specific test
pytest mindcore/tests/test_enrichment.py::test_enrichment_extracts_topics -v
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests
- Critical paths must have comprehensive tests

### Mocking External Services

```python
from unittest.mock import Mock, patch

@patch('mindcore.llm_providers.OpenAI')
def test_with_mocked_openai(mock_openai_class):
    """Test with mocked OpenAI to avoid real API calls."""
    # Setup mock
    mock_response = Mock()
    mock_response.choices[0].message.content = "test response"
    mock_openai_class.return_value.chat.completions.create.return_value = mock_response

    # Test code
    # ...
```

---

## Documentation

### Documentation Types

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document all public APIs
3. **README**: Getting started and examples
4. **Guides**: Detailed how-to documentation
5. **API Reference**: Comprehensive API docs

### Documentation Standards

- Clear and concise language
- Include code examples
- Explain "why" not just "what"
- Keep documentation up to date with code changes

### Building Documentation

```bash
# Build API documentation (future)
# sphinx-build -b html docs/ docs/_build/

# Preview documentation locally
# python -m http.server 8000 --directory docs/_build/
```

---

## Submitting Changes

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

### Commit Messages

Use conventional commits format:

```
type(scope): brief description

Longer description if needed. Explain the motivation for the change,
what problem it solves, and any important implementation details.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or fixes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/config changes

**Examples:**
```
feat(agents): add support for Claude 3 Opus

Add Anthropic Claude 3 Opus to the list of supported models.
Includes configuration updates and tests.

Closes #45

---

fix(cache): prevent race condition in cache updates

Adds locking mechanism to prevent concurrent cache updates
from causing data corruption.

Fixes #78
```

### Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Go to https://github.com/M-Alfaris/mindcore
   - Click "New Pull Request"
   - Choose your fork and branch
   - Fill out the PR template

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] All tests pass
   - [ ] New tests added
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated

   ## Related Issues
   Fixes #(issue number)
   ```

---

## Review Process

### Review Timeline

- Initial review: Within 3 business days
- Follow-up reviews: Within 2 business days
- Approval: 1-2 maintainer approvals required

### Review Criteria

Reviewers will check:
- Code quality and style
- Test coverage
- Documentation completeness
- Backward compatibility
- Security implications
- Performance impact

### Responding to Feedback

- Be open to suggestions
- Ask questions if feedback is unclear
- Make requested changes promptly
- Update the PR description if scope changes

### After Approval

- Maintainer will merge your PR
- Delete your feature branch
- Pull the latest main branch

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions (future)
- **Email**: ceo@cyberbeam.ie for private inquiries

### Getting Help

If you need help:
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Issues with "question" label
4. Email the maintainers

### Recognition

Contributors will be:
- Listed in CHANGELOG.md for their contributions
- Mentioned in release notes
- Added to CONTRIBUTORS.md (future)

---

## Development Workflow

### Feature Development

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes
# ... edit files ...

# 4. Test changes
pytest
black mindcore/
flake8 mindcore/

# 5. Commit changes
git add .
git commit -m "feat(scope): add amazing feature"

# 6. Push to your fork
git push origin feature/amazing-feature

# 7. Create PR on GitHub
```

### Bug Fix Workflow

```bash
# 1. Create bug fix branch
git checkout -b fix/bug-description

# 2. Add regression test
# Add test that fails with the bug

# 3. Fix the bug
# Implement the fix

# 4. Verify test passes
pytest mindcore/tests/test_specific.py

# 5. Commit and push
git commit -m "fix(scope): fix bug description"
git push origin fix/bug-description
```

---

## Additional Resources

### Useful Links

- **Main Repository**: https://github.com/M-Alfaris/mindcore
- **Issue Tracker**: https://github.com/M-Alfaris/mindcore/issues
- **PyPI Package**: https://pypi.org/project/mindcore/ (coming soon)
- **Cyberbeam**: https://cyberbeam.ie
- **Cyberbeam GitHub**: https://github.com/cyberbeamhq

### Related Projects

- OpenAI Python SDK
- LangChain
- LlamaIndex
- Ollama

---

## Questions?

If you have questions about contributing, please:
1. Check this document first
2. Search existing issues
3. Open a new issue with "question" label
4. Email: ceo@cyberbeam.ie

---

## Thank You!

Your contributions make Mindcore better for everyone. We appreciate your time and effort!

**Maintained by:**
- **Muthanna Alfaris** - CEO, Cyberbeam
- Email: ceo@cyberbeam.ie
- GitHub: [@cyberbeamhq](https://github.com/cyberbeamhq)
- Company: [Cyberbeam](https://cyberbeam.ie)

---

*Last updated: November 10, 2024*
