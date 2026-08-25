# LedgerLoop
#
# Only targets that work today are listed. `demo`, `eval` and `data` arrive with
# the steps that implement them, rather than sitting here failing.

PY := .venv/Scripts/python.exe

.PHONY: help install test cov lint typecheck check

help:
	@echo "install    create .venv and install with dev extras"
	@echo "test       run the test suite"
	@echo "cov        run tests with coverage"
	@echo "lint       ruff"
	@echo "typecheck  mypy --strict"
	@echo "check      lint + typecheck + cov"

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=ledgerloop --cov-report=term-missing

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy

check: lint typecheck cov
