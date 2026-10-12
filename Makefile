.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration cov check ingest seed dev eval

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install deps + git hooks
	uv sync --all-extras
	uv run pre-commit install

lint: ## Lint (no fix)
	uv run ruff check src tests

format: ## Auto-format
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## mypy strict
	uv run mypy

test: ## Unit tests (no network, no LLM)
	uv run pytest tests/unit -m "not llm and not integration" -s

test-integration: ## Integration tests
	uv run pytest tests/integration -m "integration"

cov: ## Tests with coverage gate
	uv run pytest tests -m "not llm" --cov --cov-report=term-missing

check: lint typecheck cov ## Everything CI runs

# Cibles complétées dans les phases suivantes
ingest: ## (Phase 2) Build vector index
	@echo "TODO(exercice)"
seed: ## (Phase 3) Seed fake DB
	@echo "TODO(exercice)"
dev: ## (Phase 8) Run local LangGraph server
	uv run langgraph dev
eval: ## (Phase 6) Run LangSmith evals
	@echo "TODO(exercice)"
