# LedgerLoop
#
# Only targets that work today are listed. `demo`, `eval` and `data` arrive with
# the steps that implement them, rather than sitting here failing.

PY := .venv/Scripts/python.exe

.PHONY: help install test cov lint typecheck check data fixtures

help:
	@echo "install    create .venv and install with dev extras"
	@echo "test       run the test suite"
	@echo "cov        run tests with coverage"
	@echo "lint       ruff"
	@echo "typecheck  mypy --strict"
	@echo "check      lint + typecheck + cov"
	@echo "data       generate every split at standard difficulty (gitignored)"
	@echo "fixtures   regenerate the committed 60-order fixture set"

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

# The evaluated splits never use --ensure-class-coverage: forcing a class into
# them would distort the prevalence they are measured against.
data:
	$(PY) -m ledgerloop.cli generate --split dev         --seed 42
	$(PY) -m ledgerloop.cli generate --split train       --seed 42
	$(PY) -m ledgerloop.cli generate --split calibration --seed 42
	$(PY) -m ledgerloop.cli generate --split test        --seed 42

# The fixture set is the exception: its job is to exercise every code path, not
# to be statistically representative, so class coverage is forced.
fixtures:
	$(PY) -m ledgerloop.cli generate --split dev --difficulty standard --seed 42 \
		--ensure-class-coverage --out data/fixtures/dev-standard-42
