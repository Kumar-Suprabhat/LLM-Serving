# Local checks. GPU benches are scripts/remote/* and scripts/reproduce_all.sh, not Make targets.

PYTHON ?= python
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help lint test cov

help:  ## Show this help
	@echo "  lint   ruff format check, ruff, mypy"
	@echo "  test   unit tests, no GPU"
	@echo "  cov    unit tests with coverage"

lint:  ## Lint and type-check
	ruff format --check src tests
	ruff check src tests
	mypy src

test:  ## Unit tests (no hardware)
	pytest -m "not gpu"

cov:  ## Coverage report
	pytest -m "not gpu" --cov --cov-report=term-missing
