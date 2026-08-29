# DEMO.md — running LedgerLoop from a clean checkout

Every command here was run on the machine this was built on, in the order
written. Nothing needs a network, an API key, a database, or Docker.

**Total time from `git clone` to a populated UI: about a minute**, most of it
installing dependencies.

---

## Prerequisites

| | |
|---|---|
| **Python 3.11+** | the only hard requirement |
| **[uv](https://docs.astral.sh/uv/)** | recommended; `pip` works equally well |
| ~~Make~~ | **not required.** Every `make` target is a one-line wrapper around a `ledgerloop` subcommand, and both forms are given below |
| ~~Docker~~ | **not used.** See [Why there is no Compose file](#why-there-is-no-compose-file) |
| ~~API key~~ | **not required.** The whole system runs deterministically; see [Running with an LLM](#running-with-an-llm-optional) |

---

## 1. Setup

```bash
git clone https://github.com/SatvikO7/ledgerloop.git
cd ledgerloop

uv venv --python 3.11
uv pip install -e ".[demo]"
```

`[demo]` adds LangGraph and Streamlit. If you only want the numbers and not the
interface, `uv pip install -e ".[dev]"` is enough — `ledgerloop eval` imports
neither extra, and a test blocks the LangGraph import to prove it.

With `pip` instead of `uv`:

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[demo]"
```

Everything below assumes the venv's interpreter. On Windows that is
`.venv/Scripts/python.exe`; elsewhere `.venv/bin/python`. Activate the venv and
`python` alone works.

---

## 2. The demo, in one command

```bash
python -m ledgerloop.cli demo
```

or, equivalently, `make demo`.

It generates the corpora, fits the calibration bundle, reconciles a 60-order
batch through the LangGraph pipeline, and opens the four screens. Roughly **8
seconds** before the browser opens; it skips any corpus already on disk, so it
is safe to re-run.

<details>
<summary>What it prints</summary>

```
LedgerLoop demo
  1. generate    three heterogeneous sources plus link-level ground truth
  2. calibrate   fit the blender and tau_high on train + calibration
  3. reconcile   run the LangGraph pipeline over the demo corpus
  4. inspect     open the four screens

[1/4] generating corpora
      10 generated, 0 already present
      demo corpus data/generated/dev-standard-42: 145 records, 59 evaluation
      links, 12 unmatchable by construction

[2/4] fitting the calibration bundle
      wrote reports/calibration.json: tau_high = 1.000000, fitted on
      calibration seeds 47, 48, 49, 50
      achieved precision 1.0000 [0.9733, 1.0000] on 140 calibration links (0 wrong)
      the `test` split is never fitted on; the bundle's provenance says so

[3/4] reconciling
      11 node visit(s), 2 residual pass(es), 123 audit event(s) in 1244 ms
      precision 1.0000 [0.8318, 1.0000] - recall 0.3220 - match rate 0.3088
      19 correct - 0 false positives costing ₹0.00 - 40 missed
      13 exception(s) covering ₹23,63,279.53, exception recall 1.0000 over 5
      - 3 unmatchable (the honest floor)
      llm: disabled (no key in $LEDGERLOOP_LLM_API_KEY); every number above is
      deterministic
      wrote reports/runs/t0t4-dev-42

[4/4] the four screens
      opening Streamlit. Ctrl-C to stop.
```
</details>

`--no-ui` stops after the reconciliation and prints the UI command instead of
launching it. `--split test` demonstrates the 300-order corpus instead.

### What to look at, in order

1. **Results** — precision with its Wilson interval, the money view, the tier
   waterfall, and per-class recall *including the classes that score badly*.
2. **Exceptions** — sorted by rupee impact descending. Open the largest one: it
   carries a class, a severity, a price, an evidence chain pointing back at
   source records, and a suggested action.
3. **Audit replay** — pick any record and see which tier proposed it, what the
   blender scored it, what the policy returned, and why.
4. **Run** — reconcile any other corpus on disk, including the 300-order `test`
   split.

---

## 3. The same thing, one stage at a time

`demo` chains these; none of them is reimplemented inside it.

### Generate

```bash
python -m ledgerloop.cli generate --split dev --difficulty standard --seed 42
```

Ground truth is generated **first** and the data derived from it. Generation is
a pure function of `(seed, split, difficulty)`, so the same command twice
produces byte-identical files — a property the test suite asserts.

| Split | Orders | Purpose |
|---|---|---|
| `dev` | 60 | the demo batch; meets the challenge's 50+ bar |
| `train` | 400 | fits the score blender |
| `calibration` | 200 | fits isotonic calibration and selects `tau_high` — never evaluated on |
| `test` | 300 | every published number comes from here |
| `scale` | 5,000 | throughput benchmark |

### Calibrate

```bash
python -m ledgerloop.cli calibrate \
    --train data/generated/train-standard-4{2,3,4,5,6} \
    --calibration data/generated/calibration-standard-4{7,8,9} \
                  data/generated/calibration-standard-50 \
    --out reports/calibration.json
```

Fitting is a **separate command** from reporting on purpose: the split
discipline has to be visible in the invocation. Nothing here names `test`, and
`CalibrationProvenance` refuses to build a bundle whose halves overlap or whose
either half is the test split.

### Reconcile

```bash
python -m ledgerloop.cli run \
    --data data/generated/dev-standard-42 \
    --calibration reports/calibration.json \
    --show-nodes
```

Writes `reports/runs/<run_id>/` — `run.json`, `audit.jsonl` (append-only),
`exceptions.json`, `decisions.json`. `--show-nodes` prints the graph's actual
path, including every repeat of the residual loop.

### Evaluate

```bash
python -m ledgerloop.cli eval \
    --data data/generated/test-standard-42 \
    --calibration reports/calibration.json \
    --out EVALUATION.md
```

Or the full published report — four baselines, the six-row ablation, the
multi-seed sweep and the difficulty curve — with `make eval`. Without Make:

```bash
python -m ledgerloop.cli ablation --data data/generated/test-standard-4{2,3,4,5,6} \
    --calibration reports/calibration.json --out reports/ablation.json
python -m ledgerloop.cli sweep --data data/generated/test-{easy,standard,hard}-4{2,3,4,5,6} \
    --calibration reports/calibration.json --out reports/sweep.json
python -m ledgerloop.cli baseline-llm --data data/generated/dev-standard-42 \
    --calibration reports/calibration.json --cold --offline-provider \
    --out reports/llm_baseline.json
python -m ledgerloop.cli eval --data data/generated/test-standard-42 \
    --calibration reports/calibration.json --ablation reports/ablation.json \
    --sweep reports/sweep.json --llm-baseline reports/llm_baseline.json \
    --out EVALUATION.md
```

The sweep needs the easy and hard corpora; `make sweep-data` generates them, or
loop `generate --split test --difficulty {easy,hard} --seed {42..46}`.

### The UI on its own

```bash
python -m streamlit run src/ledgerloop/ui/app.py
```

It reads whatever is already in `reports/runs/`. Point it elsewhere with
`LEDGERLOOP_RUNS_DIR`, `LEDGERLOOP_DATA_DIR` and `LEDGERLOOP_CALIBRATION`.

---

## 4. Verifying the build

```bash
python -m pytest          # 2187 passed
python -m ruff check .    # All checks passed!
python -m mypy            # Success: no issues found in 94 source files
```

> **If `mypy` fails to start** with `ImportError: DLL load failed while importing
> ..._mypyc`, an Application Control policy is blocking the compiled binary. Run
> it through the API, which reads the same `pyproject.toml`:
>
> ```bash
> python -c "from mypy import api; print(api.run(['--config-file','pyproject.toml'])[0])"
> ```

---

## Running with an LLM (optional)

**The demo above uses no model, and every number it prints is deterministic.**

To enable the LLM tier, set a key for any OpenAI-compatible provider:

```bash
export LEDGERLOOP_LLM_API_KEY=...        # Groq, Gemini, OpenRouter, Ollama
python -m ledgerloop.cli demo
```

`--no-llm` forces the deterministic path even when a key is present, which is
how a run proves it does not need one.

What changes, and what cannot:

- The model may **read a narration** the regex layer could not, **propose** a
  link for the residual, and **rewrite the prose** on an exception that already
  has a class, a severity and a rupee figure.
- It may **never** decide a match, do arithmetic, set a probability, produce a
  metric, or classify an exception. Every proposal is re-derived from the source
  documents by `verify_arithmetic` — whose signature has no parameter for where
  a proposal came from, and there is a test asserting that.
- A proposal whose money does not close is **demoted, not dropped**: it becomes
  a candidate routed to a human, because "the model suggested this and the
  arithmetic disagrees" is information a controller wants.

Responses are content-hash cached, so a second identical run makes **zero** live
calls.

---

## Why there is no Compose file

PLAN.md's original architecture had four containers: Postgres, Neo4j, a FastAPI
gateway and a React UI. All four were cut before implementation, and the
reasoning is in `ARCHITECTURE.md` §5 — Neo4j because the fallback was required
to produce *identical* decisions (which concedes the database buys no decision
quality), ChromaDB because embeddings are weak on vowel-dropped abbreviations,
and React because the four screens are tables and a form.

What is left is **one Python process with no services**, so Compose would have
nothing to orchestrate: it would wrap `pip install` in a container and call that
infrastructure. Docker is not installed on the machine this was built on, so any
Dockerfile shipped here would also be **unverified** — and an unverified
deployment path is worse than an honest absence.

`uv pip install -e ".[demo]"` is the whole setup, and it is tested.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `LangGraph is not installed` | you installed `.[dev]` rather than `.[demo]`. `uv pip install -e ".[demo]"` |
| `no such dataset directory` | run `python -m ledgerloop.cli generate --split dev --seed 42` first, or just `demo`, which generates what it needs |
| `calibration bundle was fitted on generator X but this dataset is Y` | the bundle predates a generator change. Delete `reports/calibration.json` and re-run `demo --refit` |
| `UnicodeEncodeError` on `₹` | a non-UTF-8 Windows console. The CLI reconfigures its own streams; if you are piping output, set `PYTHONIOENCODING=utf-8` |
| Streamlit opens on a busy port | `python -m streamlit run src/ledgerloop/ui/app.py --server.port 8899` |
| A run appears twice in the UI | run ids are derived from the ladder and corpus, so re-running overwrites its own record. Different corpora get different ids |
