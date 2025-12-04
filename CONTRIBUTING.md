# Contributing to Mindcore

Thank you for your interest in contributing to Mindcore! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community and project

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/mindcore.git
   cd mindcore
   ```

3. **Add the upstream remote**:

   ```bash
   git remote add upstream https://github.com/M-Alfaris/mindcore.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) PostgreSQL for database tests

### Installation

1. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

4. **Verify installation**:

   ```bash
   # Run linting
   ruff check mindcore/

   # Run tests
   pytest
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-redis-cache` - New features
- `fix/context-assembly-error` - Bug fixes
- `docs/update-readme` - Documentation changes
- `refactor/simplify-enrichment` - Code refactoring

### Workflow

1. **Sync with upstream**:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the code standards below

4. **Commit your changes** using conventional commits:

   ```bash
   git commit -m "feat: add Redis cache support"
   git commit -m "fix: resolve context assembly timeout"
   git commit -m "docs: update installation instructions"
   ```

## Code Standards

### Linting and Formatting

We use **Ruff** for linting and formatting (replaces black, flake8, isort):

```bash
# Check for issues
ruff check mindcore/

# Fix auto-fixable issues
ruff check mindcore/ --fix

# Format code
ruff format mindcore/
```

### Type Hints

Add type hints to all public functions:

```python
def process_message(
    self,
    message: str,
    user_id: str,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Process a message and return enriched result."""
    ...
```

### Docstrings

Use **Google-style docstrings**:

```python
def calculate_importance(self, text: str, context: list[str]) -> float:
    """Calculate the importance score for a message.

    Analyzes the message content and context to determine
    how important this message is for future retrieval.

    Args:
        text: The message text to analyze.
        context: List of recent messages for context.

    Returns:
        Importance score between 0.0 and 1.0.

    Raises:
        ValueError: If text is empty.

    Example:
        >>> agent = EnrichmentAgent(provider)
        >>> score = agent.calculate_importance("Order #123", [])
        >>> assert 0.0 <= score <= 1.0
    """
    ...
```

### Security

- Never commit secrets, API keys, or credentials
- Use environment variables for configuration
- Run security scanning before submitting:

  ```bash
  bandit -r mindcore/
  ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mindcore --cov-report=html

# Run specific test file
pytest mindcore/tests/test_context.py

# Run tests matching a pattern
pytest -k "test_enrichment"
```

### Writing Tests

- Place tests in `mindcore/tests/`
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Mock external services (LLM APIs, databases)

Example test:

```python
import pytest
from unittest.mock import Mock, patch

from mindcore.agents.enrichment_agent import EnrichmentAgent


class TestEnrichmentAgent:
    """Tests for EnrichmentAgent."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.generate.return_value = '{"topics": ["orders"]}'
        return provider

    def test_process_extracts_topics(self, mock_provider):
        """Test that process correctly extracts topics."""
        agent = EnrichmentAgent(mock_provider)

        result = agent.process({
            "user_id": "user123",
            "thread_id": "thread456",
            "session_id": "session789",
            "text": "Where is my order?",
            "role": "user",
        })

        assert "orders" in result.metadata.topics
```

## Submitting Changes

### Pull Request Process

1. **Ensure all checks pass**:

   ```bash
   pre-commit run --all-files
   pytest
   ```

2. **Push your branch**:

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what and why
   - Link to related issues (if any)
   - Screenshots for UI changes

### PR Title Format

Use conventional commit format for PR titles:

- `feat: Add Redis cache support`
- `fix: Resolve context assembly timeout`
- `docs: Update API documentation`
- `refactor: Simplify enrichment pipeline`
- `test: Add integration tests for SmartContextAgent`
- `chore: Update dependencies`

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Release Process

Releases are managed by maintainers following semantic versioning:

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.1.1): Bug fixes, backwards compatible

## Questions?

- Open an [issue](https://github.com/M-Alfaris/mindcore/issues) for bugs or feature requests
- Start a [discussion](https://github.com/M-Alfaris/mindcore/discussions) for questions

Thank you for contributing to Mindcore!
