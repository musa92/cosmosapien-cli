# Contributing to Cosmosapien CLI

Thank you for your interest in contributing to Cosmosapien CLI! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda
- (Optional) Docker for containerized development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cosmosapien-cli.git
   cd cosmosapien-cli
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/cosmosapien/cli.git
   ```

## Development Setup

### 1. Install Dependencies

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Configuration

```bash
# Copy example configuration
cp .cosmosrc.example ~/.cosmosrc

# Set up API keys (optional for development)
cosmo login openai  # or other providers
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following standards:

- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Imports**: Organized with isort
- **Type Hints**: Required for all functions
- **Docstrings**: Google style docstrings

### Code Formatting

```bash
# Format code
black cosmosapien/
isort cosmosapien/

# Check formatting
black --check cosmosapien/
isort --check-only cosmosapien/
```

### Linting

```bash
# Run linters
flake8 cosmosapien/
mypy cosmosapien/
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cosmosapien --cov-report=html

# Run specific test file
pytest tests/test_router.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Use fixtures for common setup

Example:
```python
def test_smart_router_selects_local_model_for_simple_task():
    """Test that smart router selects local model for simple tasks."""
    # Arrange
    router = SmartRouter(config_manager, local_manager)
    prompt = "Hello, how are you?"
    
    # Act
    decision = router.smart_route(prompt)
    
    # Assert
    assert decision.selected_provider == "llama"
    assert decision.complexity == TaskComplexity.SIMPLE
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clear, focused commits
- Follow conventional commit messages
- Update documentation if needed
- Add tests for new functionality

### 3. Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(router): add smart routing for complex tasks
fix(auth): resolve API key validation issue
docs(readme): update installation instructions
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

### 5. PR Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what and why (not how)
- **Checklist**: Complete all checklist items
- **Tests**: Ensure all tests pass
- **Documentation**: Update docs if needed

### 6. Review Process

- At least one maintainer must approve
- All CI checks must pass
- Code review feedback must be addressed
- Squash commits before merging

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, package versions
- **Steps to Reproduce**: Clear, numbered steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error traceback
- **Additional Context**: Screenshots, logs, etc.

### Feature Requests

- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity
- Check if similar features exist

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up to date
- Use proper markdown formatting

### Documentation Structure

```
docs/
├── README.md              # Main documentation
├── INSTALLATION.md        # Installation guide
├── USAGE.md              # Usage examples
├── API.md                # API reference
├── CONTRIBUTING.md       # This file
├── CHANGELOG.md          # Release notes
└── examples/             # Code examples
```

### Updating Documentation

- Update docs with code changes
- Add examples for new features
- Keep README current
- Update API documentation

## Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] GitHub release created

### Creating a Release

```bash
# Update version
bump2version patch  # or minor/major

# Create release
git tag v1.0.0
git push origin v1.0.0
```

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

### Resources

- [Python Documentation](https://docs.python.org/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)

## Recognition

Contributors will be recognized in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- GitHub repository contributors
- Release notes
- Project documentation

Thank you for contributing to Cosmosapien CLI! 🚀 