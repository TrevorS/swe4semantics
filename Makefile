.PHONY: help install lint format typecheck test test-cov clean run-test run-prod prepare-data prepare-test prepare-prod compare eval

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code with ruff"
	@echo "  make typecheck    - Run ty type checker"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make check        - Run all checks (format, lint, typecheck, test)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make prepare-data - Download evaluation datasets"
	@echo "  make prepare-test - Download all data for test mode"
	@echo "  make prepare-prod - Download all data for production mode"
	@echo "  make run-test     - Run pipeline in test mode"
	@echo "  make run-prod     - Run pipeline in production mode"
	@echo "  make eval         - Evaluate test embeddings"
	@echo "  make compare      - Compare static vs transformer model"

install:
	uv sync

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run ty check src/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src/qwen3_static_embeddings --cov-report=term-missing

check: format lint typecheck test
	@echo "All checks passed!"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

prepare-data:
	uv run python scripts/prepare_data.py --eval-only

prepare-test:
	uv run python scripts/prepare_data.py --mode test

prepare-prod:
	uv run python scripts/prepare_data.py --mode production

run-test:
	uv run python scripts/run_pipeline.py --mode test --output-dir outputs/test

run-prod:
	uv run python scripts/run_pipeline.py --mode production --output-dir outputs/prod

eval:
	uv run python scripts/evaluate.py --embeddings outputs/test/exports/qwen3_static.w2v.txt --benchmarks all

compare:
	uv run python scripts/compare_models.py --static outputs/test/exports/qwen3_static.w2v.txt --word-only
