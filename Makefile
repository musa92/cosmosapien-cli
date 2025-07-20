.PHONY: help install install-dev clean test test-cov lint format check security docs build release docker-build docker-run docker-test

# Default target
help: ## Show this help message
	@echo "Cosmosapien CLI - Development Commands"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

# Development
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Testing
test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=cosmosapien --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	pytest --no-cov

test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests only
	pytest tests/integration/

# Code Quality
lint: ## Run all linters
	flake8 cosmosapien/
	mypy cosmosapien/
	bandit -r cosmosapien/ -f json -o bandit-report.json || true

format: ## Format code with black and isort
	black cosmosapien/
	isort cosmosapien/

format-check: ## Check code formatting
	black --check cosmosapien/
	isort --check-only cosmosapien/

check: format-check lint ## Run all code quality checks

# Security
security: ## Run security checks
	bandit -r cosmosapien/ -f json -o bandit-report.json
	safety check
	pip-audit

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Building and Distribution
build: ## Build the package
	python -m build

build-check: ## Check the built package
	twine check dist/*

# Release Management
release-patch: ## Release a patch version
	bump2version patch
	git push --tags
	git push

release-minor: ## Release a minor version
	bump2version minor
	git push --tags
	git push

release-major: ## Release a major version
	bump2version major
	git push --tags
	git push

# Docker
docker-build: ## Build Docker image
	docker build -t cosmosapien/cli:latest .

docker-run: ## Run Docker container
	docker run -it --rm cosmosapien/cli:latest

docker-test: ## Run tests in Docker
	docker-compose --profile test up --build --abort-on-container-exit

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Development Workflow
dev-setup: install-dev pre-commit-install ## Set up development environment
	@echo "Development environment setup complete!"

dev-check: check test ## Run all development checks

dev-clean: clean ## Clean development artifacts

# CI/CD
ci-test: ## Run CI test suite
	pytest --cov=cosmosapien --cov-report=xml --cov-report=term-missing

ci-lint: ## Run CI linting
	flake8 cosmosapien/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 cosmosapien/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	black --check cosmosapien/
	isort --check-only cosmosapien/
	mypy cosmosapien/

# Utilities
version: ## Show current version
	@python -c "import cosmosapien; print(cosmosapien.__version__)"

deps-tree: ## Show dependency tree
	pipdeptree

deps-check: ## Check for outdated dependencies
	pip list --outdated

deps-update: ## Update dependencies
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

# Local Development
local-models: ## Start local Ollama service
	docker-compose --profile local-models up -d

local-models-stop: ## Stop local Ollama service
	docker-compose --profile local-models down

# Performance
profile: ## Run performance profiling
	python -m cProfile -o profile.stats -m pytest tests/ -v
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Database (if applicable)
db-migrate: ## Run database migrations
	@echo "No database migrations needed for this project"

db-reset: ## Reset database
	@echo "No database to reset for this project"

# Monitoring
monitor: ## Start monitoring tools
	@echo "Starting monitoring..."
	@echo "Use 'cosmo usage' to check usage statistics"
	@echo "Use 'cosmo job-stats' to check job distribution"
	@echo "Use 'cosmo token-stats' to check token usage"

# Quick Commands
quick-test: ## Quick test run
	pytest -xvs tests/unit/

quick-lint: ## Quick lint check
	flake8 cosmosapien/ --max-line-length=88 --count

quick-format: ## Quick format check
	black --check cosmosapien/ --diff

# Helpers
show-config: ## Show current configuration
	@echo "Python version:"
	@python --version
	@echo "\nInstalled packages:"
	@pip list | grep cosmosapien
	@echo "\nConfiguration file:"
	@ls -la ~/.cosmosrc 2>/dev/null || echo "No configuration file found"

show-env: ## Show environment information
	@echo "Environment Information:"
	@echo "========================"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "OS: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Git Branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')" 