# LedgerLoop
#
# Only targets that work today are listed. `demo` arrives with the step that
# implements it, rather than sitting here failing.

PY := .venv/Scripts/python.exe

.PHONY: help install test cov lint typecheck check data sweep-data fixtures ingest \
        calibrate ablation sweep baseline-llm eval

help:
	@echo "install      create .venv and install with dev extras"
	@echo "test         run the test suite"
	@echo "cov          run tests with coverage"
	@echo "lint         ruff"
	@echo "typecheck    mypy --strict"
	@echo "check        lint + typecheck + cov"
	@echo "data         generate every split at standard difficulty (gitignored)"
	@echo "sweep-data   generate the 5-seed x 3-difficulty test corpora"
	@echo "fixtures     regenerate the committed 60-order fixture set"
	@echo "ingest       parse and normalise the committed fixture set"
	@echo "calibrate    fit the blender, isotonic and tau_high (train + calibration)"
	@echo "ablation     six ladders x 5 seeds (PLAN.md 9.3)"
	@echo "sweep        headline config x 5 seeds x 3 difficulties (PLAN.md 9.4)"
	@echo "baseline-llm B2, on dev only (PLAN.md 9.2). The only target that can call out."
	@echo "eval         regenerate everything and write EVALUATION.md"

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
#
# TRAIN_SEEDS / CAL_SEEDS: the blender is fitted across several seeds of each
# split. One 400-order corpus leaves the residual tiers a few dozen decision
# points, which is too few to fit anything against and far too few for a
# precision target of 0.99 to mean anything. The seeds are disjoint between the
# two halves and neither half is ever `test` -- CalibrationProvenance refuses to
# construct a bundle that breaks either rule.
#
# EVAL_SEEDS is the `test` side and reuses 42-46. That is not an overlap: the
# generator seeds its streams on `<seed>:<split>:<purpose>`, so `train-42` and
# `test-42` are independent corpora. The bundle's provenance records the split
# beside every seed precisely so this is checkable rather than assumed.
TRAIN_SEEDS := 42 43 44 45 46
CAL_SEEDS   := 47 48 49 50
EVAL_SEEDS  := 42 43 44 45 46
DIFFICULTIES := easy standard hard
GEN         := data/generated
TRAIN_DIRS  := $(foreach s,$(TRAIN_SEEDS),$(GEN)/train-standard-$(s))
CAL_DIRS    := $(foreach s,$(CAL_SEEDS),$(GEN)/calibration-standard-$(s))
TEST_DIRS   := $(foreach s,$(EVAL_SEEDS),$(GEN)/test-standard-$(s))
SWEEP_DIRS  := $(foreach d,$(DIFFICULTIES),$(foreach s,$(EVAL_SEEDS),$(GEN)/test-$(d)-$(s)))
HEADLINE    := $(GEN)/test-standard-42
DEV         := $(GEN)/dev-standard-42
BUNDLE      := reports/calibration.json
ABLATION    := reports/ablation.json
SWEEP       := reports/sweep.json
B2          := reports/llm_baseline.json

data:
	$(PY) -m ledgerloop.cli generate --split dev  --seed 42
	$(foreach s,$(TRAIN_SEEDS),$(PY) -m ledgerloop.cli generate --split train --seed $(s);)
	$(foreach s,$(CAL_SEEDS),$(PY) -m ledgerloop.cli generate --split calibration --seed $(s);)

# The difficulty dial changes how much goes wrong without changing what goes
# wrong, so the three difficulties are comparable to each other. Fifteen corpora
# rather than one: PLAN.md 9.4 asks for mean +/- std, and a single run's number
# is noise.
sweep-data: data
	$(foreach d,$(DIFFICULTIES),$(foreach s,$(EVAL_SEEDS),\
		$(PY) -m ledgerloop.cli generate --split test --difficulty $(d) --seed $(s);))

# The fixture set is the exception: its job is to exercise every code path, not
# to be statistically representative, so class coverage is forced.
fixtures:
	$(PY) -m ledgerloop.cli generate --split dev --difficulty standard --seed 42 \
		--ensure-class-coverage --out data/fixtures/dev-standard-42

# Strict: a fixture that starts quarantining rows should fail, not shrink quietly.
ingest:
	$(PY) -m ledgerloop.cli ingest --data data/fixtures/dev-standard-42 --strict

# The fit is a separate command from the report on purpose: the split discipline
# has to be visible in the invocation. Nothing here names `test`.
calibrate: sweep-data
	$(PY) -m ledgerloop.cli calibrate \
		--train $(TRAIN_DIRS) \
		--calibration $(CAL_DIRS) \
		--out $(BUNDLE)

# PLAN.md 9.3. Six ladders over five seeds at standard difficulty. Deliberately
# not multiplied across the difficulty dial: ninety runs would be more numbers,
# not more evidence, and difficulty does not change what a tier is for.
ablation: calibrate
	$(PY) -m ledgerloop.cli ablation \
		--data $(TEST_DIRS) \
		--calibration $(BUNDLE) \
		--out $(ABLATION)

# PLAN.md 9.4. The headline configuration, five seeds, three difficulties. One
# bundle across all of them -- a deployed system has one threshold.
sweep: calibrate
	$(PY) -m ledgerloop.cli sweep \
		--data $(SWEEP_DIRS) \
		--calibration $(BUNDLE) \
		--out $(SWEEP)

# PLAN.md 9.2. B2 on `dev` and nowhere else, because it sends the corpus rather
# than its residual. The only target in this file that can reach a network, and
# it writes an artefact so `eval` never re-runs it.
# --cold empties B2's cache first: a warm cache reports zero calls and zero
# tokens, which is true of any rerun and would delete the comparison this row
# exists to make.
#
# --offline-provider is here because there is no provider key in this
# environment. It answers B2's prompts with the documented stand-in reasoner in
# eval/offline_provider.py; the artefact records that it did and EVALUATION.md
# prints a banner saying which figures are measured machinery (cost, cache,
# calls, failures) and which are a property of that rule (precision, recall).
# On a machine with a key, drop the flag.
baseline-llm: calibrate
	$(PY) -m ledgerloop.cli baseline-llm \
		--data $(DEV) \
		--calibration $(BUNDLE) \
		--cold --offline-provider \
		--out $(B2)

# PLAN.md 9.4: every reported number comes from `test`, and every number is
# regenerated by one command. Generation, the fit, the ablation, the sweep and
# B2 all run first, so the report can never be produced against a stale dataset,
# a stale model, or a stale table.
eval: ablation sweep baseline-llm
	$(PY) -m ledgerloop.cli eval \
		--data $(HEADLINE) \
		--calibration $(BUNDLE) \
		--ablation $(ABLATION) \
		--sweep $(SWEEP) \
		--llm-baseline $(B2) \
		--out EVALUATION.md
