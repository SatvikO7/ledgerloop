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

It generates the corpora, fits the calibration bundle, reconciles the
**300-order `test` corpus** through the LangGraph pipeline, and opens the four
screens. It skips any corpus already on disk, so it is safe to re-run.

**It opens on `test` deliberately.** That is the corpus every number in
`README.md` and `EVALUATION.md` is measured on, so the figures on screen are the
figures the documents contain — a demo that opened on a different split would
show a reviewer numbers they could not find anywhere else. `--split dev` runs the
60-order corpus instead: it still clears the challenge's 50+ record bar and is
faster, but its exception recall rests on five records and means very little.

<details>
<summary>What it prints</summary>

```
LedgerLoop demo
  1. generate    three heterogeneous sources plus link-level ground truth
  2. calibrate   fit the blender and tau_high on train + calibration
  3. reconcile   run the LangGraph pipeline over the demo corpus
  4. inspect     open the dashboard

[1/4] generating corpora
      10 generated, 0 already present (generation is a pure function of the
      seed, so an existing corpus is byte-identical)
      demo corpus data/generated/test-standard-42: 742 records, 294 evaluation
      links, 69 unmatchable by construction

[2/4] fitting the calibration bundle
      wrote reports/calibration.json: tau_high = 1.000000, fitted on
      calibration seeds 47, 48, 49, 50
      achieved precision 1.0000 [0.9768, 1.0000] on 162 calibration links (0 wrong)
      the `test` split is never fitted on; the bundle's provenance says so

[3/4] reconciling
      12 node visit(s), 2 residual pass(es), 704 audit event(s) in 3068 ms
      precision 1.0000 [0.9866, 1.0000] - recall 0.9626 - match rate 0.9159
      283 correct - 0 false positives costing ₹0.00 - 11 missed
      67 exception(s) covering ₹50,20,195.68, exception recall 1.0000 over 30
      - 35 unmatchable (the honest floor)
      llm: disabled (no provider key in $LEDGERLOOP_LLM_API_KEY or any of
      $GEMINI_API_KEY, $GROQ_API_KEY, $OLLAMA_API_KEY, $OPENROUTER_API_KEY);
      every number above is deterministic
      wrote reports/runs/t0t4-test-42

[4/4] the dashboard
      opening Streamlit. Ctrl-C to stop.
```
</details>

Those figures are the single-seed ones for `test` seed 42. The claims in
`README.md` are the five-seed means beside them — recall 0.8844 ± 0.0788, match
rate 0.8533 ± 0.0810, exception recall 0.9818 ± 0.0250 — because a single run's
number is noise.

Two of those numbers are worth reading with their intervals. Seed 42's match
rate of 0.9159 has a 95% Wilson lower bound of 0.8819, which **clears** the
≥ 0.85 target on the interval and not merely on the point estimate. Its
exception recall of 30/30 rests on thirty records with an interval of
[0.8865, 1.0000], and `EVALUATION.md` reports that as *undecided against the
≥ 0.95 target* rather than as a pass — the same rule, cutting the other way.

`--no-ui` stops after the reconciliation and prints the UI command instead of
launching it. `--split dev` runs the 60-order corpus.

### What to look at, in order

1. **Overview** — the four headline proportions as cards. Each carries its
   sample, its 95% Wilson interval drawn to scale, and a verdict. Precision
   reads *undecided* rather than green: 283 of 283 gives [98.66%, 100.00%], and
   the lower bound does not clear the ≥ 99% target. That is the same ruling
   `EVALUATION.md` prints, from the same call.
2. **Pipeline** — the ladder as a flow. **T4 Graph shows 0 and says why**: it
   ran, it found nothing, and the panel explains that every earlier rung matches
   at settlement granularity so the partial assignments graph inference exists
   to finish never arise. T5 is drawn differently again — dashed, *did not run* —
   because a rung that never executed has no result rather than a zero.
3. **Exceptions** — sorted by rupee impact descending. Open the largest one: it
   carries a class, a severity, a price, an evidence chain pointing back at
   source records, and a suggested action.
4. **Evidence** — pick any record and follow it: source records → normalisation
   → candidate → tier → arithmetic verification → decision.
5. **Evaluation** — per-class recall *including the classes that score badly*
   (`A09_SPLIT_PAYOUT` at 0.74 is the row to look at: the remaining gap is one
   settlement whose payments do not partition uniquely across its two tranches,
   which the system refuses rather than guesses).
6. **Run** — reconcile any other corpus on disk, including the 300-order `test`
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
| `dev` | 60 | the fast batch (`demo --split dev`); meets the challenge's 50+ bar |
| `train` | 400 | fits the score blender |
| `calibration` | 200 | fits isotonic calibration and selects `tau_high` — never evaluated on |
| `test` | 300 | the demo default, and where every published number comes from |
| `scale` | 5,000 | benchmarked by `ledgerloop scale`: precision **1.0000**, 0 false positives, 12,233 records in 7.4 s |

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
python -m ledgerloop.cli comparison --data data/generated/test-{easy,standard,hard}-4{2,3,4,5,6} \
    --calibration reports/calibration.json --switch split-completion \
    --out reports/comparison.json
python -m ledgerloop.cli llm-report --data data/generated/test-standard-42 \
    --calibration reports/calibration.json --offline-provider \
    --cache-dir reports/llm_cache_report --out reports/llm_report.json
python -m ledgerloop.cli eval --data data/generated/test-standard-42 \
    --calibration reports/calibration.json --ablation reports/ablation.json \
    --sweep reports/sweep.json --llm-baseline reports/llm_baseline.json \
    --comparison reports/comparison.json --llm-report reports/llm_report.json \
    --out EVALUATION.md
```

The size curve is a separate command (`make scale`), not part of `eval`. It
generates corpora far larger than any published number uses, and its throughput
columns are the only figures this project writes that a rerun will not
reproduce -- folding them into the document whose byte-identity is a test would
break that test by design. It exits non-zero if any size produced a false
positive, which is what keeps T3's two scale-only guards honest.

```bash
python -m ledgerloop.cli scale --calibration reports/calibration.json \
    --out reports/scale.json
```

`comparison` is the before/after study for a change to the reconciliation
system: it runs all fifteen corpora **twice**, once with the change and once
without, and writes both arms with their tuning hashes. `--switch` picks which
change — `split-completion` (the default, and the most recent) or `duplicates`
— from a fixed list rather than an arbitrary configuration, so the artefact
means the same thing however it was invoked. `llm-report` runs the production LLM path once with a `--no-llm` control
over the same corpus, so "the model proposes, deterministic code decides" is
measured rather than asserted. Drop `--offline-provider` on a machine with a
provider key and the same command produces a live measurement.

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
python -m pytest          # 2342 passed
python -m ruff check .    # All checks passed!
python -m mypy            # Success: no issues found in 99 source files
```

These are the same three commands `.github/workflows/ci.yml` runs on every push
and pull request, against the same `pyproject.toml` — the workflow passes no
flags of its own, so there is no second configuration to drift.

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

To enable the LLM tier, set a key for any rung of the failover ladder. Copy
[`.env.example`](.env.example) to `.env` (gitignored) and fill in what you have:

```bash
export GROQ_API_KEY=...                  # or GEMINI_API_KEY / OPENROUTER_API_KEY
export LEDGERLOOP_LLM_API_KEY=...        # or one key shared by every rung
python -m ledgerloop.cli demo
```

The ladder is **Groq → Gemini → OpenRouter → Ollama**, in that order. A rung with
no credential is skipped; `ollama` needs none, so it joins the ladder only when
you name it (`LEDGERLOOP_LLM_PROVIDERS=ollama`) or point `OLLAMA_BASE_URL`
somewhere — otherwise a machine with no keys would wait for a localhost
connection to time out before reaching the deterministic path.

A rate limit waits once (honouring `Retry-After`, capped) and then moves down a
rung; an outage moves down immediately; an exhausted ladder raises, and every
call site treats that exactly like `--no-llm`. How far down the run had to go is
recorded as `fallback_depth` and printed in the report, so a rate-limited run is
visible rather than silent.

`--llm-providers groq,ollama` overrides the order for one invocation.
`--no-llm` forces the deterministic path even when a key is present, which is
how a run proves it does not need one.

**To measure the LLM path rather than just use it:**

```bash
python -m ledgerloop.cli llm-report \
    --data data/generated/test-standard-42 \
    --calibration reports/calibration.json \
    --out reports/llm_report.json
```

It runs the corpus with the model and again with `--no-llm`, and writes both
scores side by side together with the calls, cache hits, tokens, latency,
failures, budget refusals, actual and equivalent-paid cost, how many references
the grounding gate refused and how many proposals `verify_arithmetic` demoted.
Add `--offline-provider` to run it with no key at all: every machinery column is
still measured on the real code path, the artefact records `live: false`, and
`EVALUATION.md` prints a banner saying **no claim is being made about any
language model's answer quality**.

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
calls. The cache key is stable across a failover, so a transient rate limit
cannot turn into a permanent extra cost.

**The production LLM path has been measured live, against real Gemini**
(`gemini-3.6-flash`, fallback depth 0) — `EVALUATION.md` reports it under *The
production LLM path, measured*, and the artefact records `live: true`. Nine calls,
13,097 tokens, ₹0 actual and ₹8.81 equivalent-paid cost. **Six of fifteen calls
failed** (three read timeouts, three HTTP 503s) and the ladder absorbed all six.
Of what came back, 11 of 35 narration repairs were accepted, **9 outputs were
refused by the grounding gate** for citing records that were not in the evidence
pack, and the single link proposal was **not** accepted.

**The control is the point.** The same corpus with `--no-llm` produces precision,
recall, match rate and exception recall identical to six decimal places. The model
changed no published number — which is what *"the LLM proposes, deterministic code
decides"* looks like when it is measured instead of asserted.

Every deterministic figure in this repository still reproduces with `--no-llm` and
no key; the live figures are **one measured run** and will not reproduce, because
call counts, failures and latency depend on the network.

---

## Why there is no Compose file

PLAN.md's original architecture had four containers: Postgres, Neo4j, a FastAPI
gateway and a React UI. All four were cut before implementation, and the
reasoning is in `ARCHITECTURE.md` §5 — Neo4j because the fallback was required
to produce *identical* decisions (which concedes the database buys no decision
quality), ChromaDB because embeddings are weak on vowel-dropped abbreviations,
and React because the dashboard is one stylesheet over a finished run.

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
| The demo hangs before `[3/4]` | a keyless machine should never reach a provider. If you set `LEDGERLOOP_LLM_PROVIDERS=ollama` without Ollama running, the ladder waits for localhost. Unset it, or pass `--no-llm` |
| `llm-report did not run` | no provider key and no `--offline-provider`. That is a refusal, not a failure: the command's job is to measure the LLM path, and a row of zeros for a path that never executed would be a false measurement |
