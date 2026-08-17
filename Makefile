.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help install lint format typecheck test test-unit coverage contracts smoke up down logs clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install all dependencies (ml + dev)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[ml,dev]"

lint: ## Ruff lint check
	$(VENV)/bin/ruff check app tests

format: ## Ruff format + autofix
	$(VENV)/bin/ruff format app tests
	$(VENV)/bin/ruff check --fix app tests

typecheck: ## MyPy type check
	$(VENV)/bin/mypy app

test: ## Run full test suite
	$(VENV)/bin/pytest

test-unit: ## Run unit tests only
	$(VENV)/bin/pytest tests/unit -v

coverage: ## Run tests with coverage report
	$(VENV)/bin/pytest --cov=app --cov-report=term-missing

contracts: ## Verify architectural import contracts
	$(VENV)/bin/lint-imports

smoke: ## Smoke-test a running stack
	bash scripts/smoke_test.sh

up: ## Build and start the full stack
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

logs: ## Tail API + worker logs
	docker compose logs -f api worker

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
migrate: ## Apply all pending DB migrations
	$(VENV)/bin/alembic upgrade head

revision: ## Create a new migration: make revision MSG="describe change"
	$(VENV)/bin/alembic revision --autogenerate -m "$(MSG)"

test-integration: ## Integration tests (requires: docker compose up -d postgres)
	$(VENV)/bin/pytest tests/integration -v

experiments-chunking: ## Run Phase 5 chunking experiments (writes docs/chunking-experiments.md)
	$(VENV)/bin/python scripts/run_chunking_experiments.py
collection-info: ## Show Qdrant collection status for the default embedding model
	$(VENV)/bin/python scripts/manage_collection.py info
benchmark-retrieval: ## Run Phase 8 retrieval benchmark (writes docs/retrieval-benchmarks.md)
	$(VENV)/bin/python scripts/run_retrieval_benchmark.py
benchmark-reranking: ## Run Phase 9 reranking benchmark (writes docs/reranking-experiments.md)
	$(VENV)/bin/python scripts/run_reranking_benchmark.py