# LedgerLoop
#
# Only targets that work today are listed.
#
# `make` is a convenience here, not a prerequisite. Every target below is a
# one-line wrapper around a `ledgerloop` subcommand, and DEMO.md gives the same
# commands verbatim -- because GNU Make is absent from a default Windows install
# (including the machine this was built on), and a project whose claim is that it
# runs on nothing should not require Make to be seen running.

# The venv's interpreter, on either layout: Scripts/ on Windows, bin/ elsewhere.
VENV := .venv
PY   := $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts/python.exe,$(VENV)/bin/python)

.PHONY: help install install-demo test cov lint typecheck check data sweep-data \
        fixtures ingest calibrate ablation sweep baseline-llm comparison llm-report \
        eval run demo ui

help:
	@echo "install      create .venv and install with dev extras"
	@echo "test         run the test suite"
	@echo "cov          run tests with coverage"
	@echo "lint         ruff"
	@echo "typecheck    mypy --strict"
	@echo "check        lint + typecheck + cov"
	@echo "data         generate the corpora eval needs (gitignored)"
	@echo "sweep-data   add the easy/hard test corpora the difficulty sweep needs"
	@echo "fixtures     regenerate the committed 60-order fixture set"
	@echo "ingest       parse and normalise the committed fixture set"
	@echo "calibrate    fit the blender, isotonic and tau_high (train + calibration)"
	@echo "ablation     six ladders x 5 seeds (PLAN.md 9.3)"
	@echo "sweep        headline config x 5 seeds x 3 difficulties (PLAN.md 9.4)"
	@echo "baseline-llm B2, on dev only (PLAN.md 9.2)."
	@echo "comparison   Phase 2.3 before/after: 5 seeds x 3 difficulties, both arms"
	@echo "llm-report   one measured run of the production LLM path, with a control"
	@echo "eval         regenerate everything and write EVALUATION.md"
	@echo ""
	@echo "demo         generate, calibrate, reconcile, then open the UI (~10s)"
	@echo "run          reconcile one dataset through the LangGraph pipeline"
	@echo "ui           open the Streamlit interface over the runs already stored"

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

# The deterministic core needs neither extra. `install` above is enough for
# every metric in EVALUATION.md; this adds LangGraph and Streamlit for the demo.
install-demo:
	uv pip install -e ".[demo]"

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
COMPARISON  := reports/comparison.json
SCALE       := reports/scale.json
LLM_REPORT  := reports/llm_report.json

# Everything the fit and the ablation need: the demo corpus, the two fitting
# halves, and the five standard-difficulty `test` seeds.
data:
	$(PY) -m ledgerloop.cli generate --split dev  --seed 42
	$(foreach s,$(TRAIN_SEEDS),$(PY) -m ledgerloop.cli generate --split train --seed $(s);)
	$(foreach s,$(CAL_SEEDS),$(PY) -m ledgerloop.cli generate --split calibration --seed $(s);)
	$(foreach s,$(EVAL_SEEDS),$(PY) -m ledgerloop.cli generate --split test --seed $(s);)

# The other two difficulties, which only the sweep needs. The dial changes how
# much goes wrong without changing what goes wrong, so the three columns are
# comparable to each other. Ten more corpora rather than none: PLAN.md 9.4 asks
# for mean +/- std across difficulties, and a single run's number is noise.
sweep-data: data
	$(foreach d,easy hard,$(foreach s,$(EVAL_SEEDS),\
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
calibrate: data
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
sweep: calibrate sweep-data
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

# Phase 2.4. The before/after study for the one substantive change Phase 2.3
# made to the reconciliation system: the duplicate-posting pass. Both arms over
# the SAME fifteen corpora with the SAME bundle, differing in exactly one
# configuration field, and each arm's tuning_hash is written into the artefact so
# "nothing else changed" is a check rather than a claim.
comparison: calibrate sweep-data
	$(PY) -m ledgerloop.cli comparison \
		--data $(SWEEP_DIRS) \
		--calibration $(BUNDLE) \
		--out $(COMPARISON)

# Phase 2.6. The size curve, up to the `scale` split's 5,000 orders.
#
# NOT part of `eval`, for two reasons. It generates corpora far larger than any
# published number uses, and its throughput columns are the only figures this
# project writes that a second run will not reproduce -- so folding it into the
# document whose byte-identity is a test would break that test by design.
#
# Precision is the column to read first, and this target runs FIVE SEEDS at each
# size because one could not be trusted to show it: the single-seed version
# printed `precision held at every size` while seed 45 carried 17 false
# positives at 5,000 orders. Ten were a real defect and are fixed; seven remain.
#
# So this target currently EXITS NON-ZERO, and that is deliberate. A gate that
# went green over a known defect would be worth less than no gate at all. It
# will pass again when the remaining case is closed -- see README's Limitations.
scale: calibrate
	$(PY) -m ledgerloop.cli scale \
		--calibration $(BUNDLE) \
		--out $(SCALE)

# Phase 2.2. One measured run of the PRODUCTION LLM path -- prompts, cache,
# budget, provider ladder, grounding gate, verify_arithmetic, cost ledger --
# together with a --no-llm control over the same corpus, so "the model proposes
# and deterministic code decides" is measured rather than asserted.
#
# --offline-provider is here because there is no provider key in this
# environment. The machinery columns are measured either way; the artefact
# records `live: false` and EVALUATION.md prints a banner. On a machine with a
# key -- GROQ_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY -- drop the flag and
# the same command produces a live measurement.
llm-report: calibrate
	$(PY) -m ledgerloop.cli llm-report \
		--data $(HEADLINE) \
		--calibration $(BUNDLE) \
		--offline-provider \
		--cache-dir reports/llm_cache_report \
		--out $(LLM_REPORT)

# Step 11. One reconciliation through the LangGraph state machine, written to
# reports/runs/<run_id>/ as run.json, audit.jsonl, exceptions.json and
# decisions.json. The same numbers `eval` reports -- both paths call the same
# node functions and the same scorer.
run: calibrate
	$(PY) -m ledgerloop.cli run \
		--data $(HEADLINE) \
		--calibration $(BUNDLE) \
		--show-nodes

# Step 12. The four screens, over whatever runs are already in reports/runs/.
# Reads; never computes. Ctrl-C to stop.
ui:
	$(PY) -m streamlit run src/ledgerloop/ui/app.py

# The one command a reviewer needs, and a one-line wrapper on purpose: the demo
# is `ledgerloop demo`, so it works identically with or without Make. It
# generates only what it needs (the demo corpus plus the two fitting halves),
# fits the bundle on train/calibration -- never on test -- reconciles through
# the LangGraph pipeline, and opens the four screens. About ten seconds from a
# clean checkout.
#
# No prerequisite target: the command skips corpora that already exist, so it is
# safe to re-run and does not drag in the eval-only datasets.
demo:
	$(PY) -m ledgerloop.cli demo

# PLAN.md 9.4: every reported number comes from `test`, and every number is
# regenerated by one command. Generation, the fit, the ablation, the sweep and
# B2 all run first, so the report can never be produced against a stale dataset,
# a stale model, or a stale table.
eval: ablation sweep baseline-llm comparison llm-report
	$(PY) -m ledgerloop.cli eval \
		--data $(HEADLINE) \
		--calibration $(BUNDLE) \
		--ablation $(ABLATION) \
		--sweep $(SWEEP) \
		--llm-baseline $(B2) \
		--comparison $(COMPARISON) \
		--llm-report $(LLM_REPORT) \
		--out EVALUATION.md
