# PROGRESS.md — session handoff

**Last updated:** 2026-08-25 · **Branch:** `main` · **HEAD:** Step 2

Read this first, then [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the build
order and [ARCHITECTURE.md](ARCHITECTURE.md) for the decisions that are already settled.
[PLAN.md](PLAN.md) is the original project design and is deliberately left unedited —
every divergence from it is recorded in `ARCHITECTURE.md` §6.

---

## 1. Current status

**Step 2 of 14 complete.** The project has a tested type layer, a seeded synthetic data
generator with link-level ground truth, and **a working scoreboard with a real number on
it**. B0 (exact join on UTR) is the only system scored so far. **No ingest and no
matching tier exists yet** — that starts at Step 3.

Everything runs offline with one runtime dependency (`pydantic`). No LLM calls, no
Docker, no external services have been introduced. Step 2 added no dependency.

```
433 tests passing · 99% coverage · ruff clean · mypy --strict clean (33 files)
```

**The floor to beat**, B0 on `test`, seed 42, generator `0.2.0`:

```
precision 0.5909  [0.5371, 0.6426]   recall 0.6633   match rate 0.6580
195 correct · 135 false positives costing ₹34,98,306.00 · 99 missed
```

---

## 2. What has been completed

### Step 0 — Data contracts (commit `8eb6f91`)

- `money.py` — integer-minor-unit arithmetic. `float`, `bool` and `Decimal` rejected at
  every entry point. `allocate_minor` splits totals with exact conservation.
- `models/` — nine contract modules: canonical records (4 entity types), typed
  `RecordRef` addressing, link-level `GroundTruth`, `MatchCandidate` + `FeatureVector` +
  `Evidence`, immutable `MatchDecision`, `ReconException`, `RunMetrics`, `AuditEvent`.
- `config.py` — frozen `RunConfig` tree with a stable `config_hash`.
- `state.py` — `ReconState`.
- `graph/interface.py`, `vector/interface.py` — Protocols only, **no implementations**.
- `ARCHITECTURE.md` — including the evaluation-unit definition (§3 below).

### Step 1 — Generator + ground truth (commit `a7042fa`)

- Two-phase generation: build a world that reconciles perfectly, then break it in eleven
  labelled ways. Truth is assembled from each scenario's own record, never inferred.
- All 11 anomaly classes (A11 FX is cut), 5 splits, difficulty dial, seeded
  byte-identical regeneration.
- Three heterogeneous sources + 2 truth files + manifest.
- `ledgerloop generate` CLI.
- Committed fixture: `data/fixtures/dev-standard-42/` — 60 orders, 5 settlements,
  23 bank rows, 59 evaluation links, 13 unmatchable, 13 anomalies applied.

### Step 2 — Eval harness + B0

- `eval/truth_io.py` — reads `GroundTruth` and the manifest back off disk, the exact
  inverse of the emitters. The report is scored against the *files*, so it can never
  disagree with the committed CSVs.
- `eval/metrics.py` — link-level P/R/F1 with a **Wilson** interval on precision, match
  rate, per-class recall, money view, and `evaluate()` producing a `RunMetrics`.
- `eval/baselines.py` — B0, exact join on UTR, with its own deliberately naive readers.
- `eval/report.py` — renders `EVALUATION.md`; gitignored and fully regenerated.
- `ledgerloop eval --data <dir>` and a `make eval` target that regenerates `test` first.

---

## 3. What has NOT been completed

**Steps 3–14 — nothing in this list exists yet:**

| Missing | Step |
|---|---|
| Ingest / normalize — the three parsers, narration regex, DD/MM resolution | 3 |
| Any matching tier (T0–T5) | 4–6, 9 |
| Score blender, isotonic calibration, threshold selection, decision policy | 7 |
| Exception classifier, root causes, bounded auto-resolution | 8 |
| LLM client, response cache, `--no-llm` | 9 |
| Baselines B1/B2/B3, ablation table, multi-seed sweep | 10 |
| LangGraph assembly, audit replay | 11 |
| Streamlit UI | 12 |
| `make demo`, Docker Compose, DEMO.md, video | 13 |
| All stretch items (provider ladder, React, Neo4j, Chroma, hosting, k3s) | 14 |

**Also not done:** no `data/seeds.json` of blessed seeds; no CI workflow
(`.github/workflows/ci.yml`); the `scale` split generates but is not asserted on in tests.

---

## 4. Current implementation step

**Completed: Step 2.**
**Next: Step 3 — "Ingest + normalize."**

The scoreboard now exists and has a number on it, which was the whole point of ordering
Step 2 before any matcher: no change from here is a guess about whether it helped. The
moment T0 exists it gets scored against the same truth B0 was scored on.

---

## 5. Files created / modified

### Source (`src/ledgerloop/`)

```
__init__.py  money.py  config.py  state.py  cli.py
models/      __init__ base enums refs records truth candidates
             decisions recon_exception metrics audit
generator/   __init__ vocab world baseline scenarios ground_truth
             emitters generate
eval/        __init__ truth_io metrics baselines report
graph/       __init__ interface        (Protocol only — no implementation)
vector/      __init__ interface        (Protocol only — no implementation)
```

### Tests (`tests/`)

```
unit/      test_money  test_models  test_truth  test_config  test_metrics
           test_state_and_interfaces  test_generator  test_emitters
           test_generator_edges
           test_eval_metrics  test_eval_truth_io  test_eval_baselines
           test_eval_report
property/  test_money_invariants  test_generator_invariants
           test_metrics_invariants
```

### Docs / config

```
README.md  ARCHITECTURE.md  IMPLEMENTATION_PLAN.md  PROGRESS.md  PLAN.md (unedited)
pyproject.toml  Makefile  .gitignore  .gitattributes
data/fixtures/dev-standard-42/  (6 committed files)
```

### Local-only, never committed

```
.local/steps/step-00-foundation.md
.local/steps/step-01-data-generation.md
.local/steps/step-02-eval-harness.md
```

Detailed per-step write-ups. `.local/` is gitignored. **Write one for every step.**

---

## 6. Important architectural decisions

These are settled. Do not re-litigate them; `ARCHITECTURE.md` has the full reasoning.

| # | Decision |
|---|---|
| 1 | **The atomic unit of evaluation is the `PAYMENT_CREDITED_AS` link** — a `(payment_id, bank_txn_id)` pair, not a record. `GroundTruth.evaluation_pairs` is the truth set; `MatchCandidate.pair` and `MatchDecision.pair` produce comparable keys. Structural edges (`ORDER_PAID_BY`, `PAYMENT_SETTLED_IN`) are excluded — the sources assert them, so counting them inflates every score. |
| 2 | **Ground truth is link-level.** PLAN.md's flat row could not represent split payouts (A09) or duplicate credits (A05). |
| 3 | **No float in the money path.** Enforced by `assert_minor`, the `MinorUnits` annotated type, and a reflective test that walks every model. `delta_ratio()` is the one sanctioned money→float crossing and its result must never re-enter a money field. |
| 4 | **A `train` split (400) exists** so the blender is fitted on different data than the isotonic calibrator. Fit on `train` → calibrate on `calibration` → report from `test`. |
| 5 | **T0/T1 bypass the blender at p=1.0**; calibration is measured on residual tiers only (`CalibrationMetrics.residual_only`). Including ~70% of volume at p≈1.0 gives a near-zero ECE that measures the corpus, not the calibrator. |
| 6 | **`tier` is one-hot in the blender, never ordinal** — it is near-perfectly predictive and would collapse the model into a lookup table. |
| 7 | **Two taxonomies, 11 anomaly vs 13 exception classes.** Their mapping is a measured rectangular confusion matrix, never a hardcoded identity. |
| 8 | **`NEEDS_REVIEW` is not a positive prediction** (`MatchDecision.is_positive_prediction`). Counting referrals as matches is the precision-inflating trap. |
| 9 | **`AUTO_MATCHED` requires `arithmetic_verified`** — enforced by a model validator, not convention. |
| 10 | **Neo4j and ChromaDB are cut**, retained as Protocols. NetworkX is the real T4; T3 is lexical-only and `semantic_score` stays `0.0`. If embeddings return, they return as an *ablation row*. |
| 11 | **A11 FX/multicurrency is cut.** `Currency.USD.supported is False` so the cut is testable rather than merely absent. |
| 12 | **Streamlit is the UI plan; React is stretch only.** |
| 13 | **The settlement identity `net == gross − fee − tax + adjustments` is deliberately NOT validated** — A03 breaks it on purpose. Exposed as `net_delta_minor`, reported as evidence. |
| 14 | **Anomalies compose along independent aspects** (amount / structure / date / narration). Conflicting mutations on the same aspect are declined and reported, not applied. |
| 15 | **Draws and effects are counted separately.** `scenario_draws` is what the prevalence dial produced; `effects` is what actually happened. Never conflate them. |
| 16 | **Money conservation is checkable, not rhetorical:** every scenario that moves money declares `bank_delta_minor`, so `sum(settlement-linked credits) − sum(declared nets) − sum(declared deltas) == 0` exactly. |
| 17 | **The split name is mixed into the RNG seed.** `Random(f"{seed}:{split}:baseline")`. |
| 18 | **`ReconState` lives at `ledgerloop.state`, not `models.state`** — it holds a `RunConfig`, and putting it in `models` creates a real import cycle. |
| 19 | **Per-class recall is attributed through the link's endpoint *records*, not `GroundTruthLink.anomaly_class`** — the generator never populates that field, so grouping on it hides ten of the eleven classes. A link is counted under every anomaly touching either endpoint, so **the rows overlap and do not sum to the link total.** |
| 20 | **The precision interval is Wilson, not the normal approximation** — at 250/250 the normal approximation returns `[1.0, 1.0]`, breaking exactly where this project's headline claim lives. The two boundary cases are special-cased to exact 0.0 / 1.0 because floating point lands them a few ulps short. |
| 21 | **A zero denominator is reported as `n/a`, never as `0.00%`** — and the interval widens to `[0.0, 1.0]` rather than narrowing. A system that predicted nothing has not achieved perfect precision. |
| 22 | **Match rate measures reach, precision measures correctness** — `match_rate` counts records the system asserted *anything* about, and its denominator is `reconcilable_refs` restricted to payment and bank records. Folding correctness in would double-count precision; including orders and settlements would charge the matcher for edges the sources assert. |
| 23 | **B0 is a stdlib join, not `pandas.merge`** — the semantics PLAN.md §9.2 asks for are the exact join, not the library, and a runtime dependency for a twenty-line lookup contradicts the "runs on nothing" claim. |

---

## 7. Tests

```bash
.venv/Scripts/python.exe -m pytest                                   # 433 passed in ~29s
.venv/Scripts/python.exe -m pytest --cov=ledgerloop --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check .                             # All checks passed!
.venv/Scripts/python.exe -m mypy                                     # 33 files, clean
```

| Metric | Result |
|---|---|
| Tests | **433 passed**, 0 failed |
| Coverage | **99%** (1805 stmts, 6 miss; 290 branch, 6 partial); `eval/` at **100%** |
| `ruff check .` | clean |
| `mypy` (strict) | clean, 33 source files |

**Step 1 acceptance criteria — all met:**

- Byte-identical regeneration verified with `cmp` on all six files.
- Conservation residual **0** across 72 split × difficulty × seed combinations.
- All 11 anomaly classes present in the fixture and in `test`.
- Realised prevalence within **±2%** of configured, measured on 2,000 draws.

**Step 2 acceptance criteria — all met:**

- Metric correctness pinned on hand-built truth sets small enough to verify by eye.
- CI edge cases covered: zero predictions, zero truth, both, and a flawless run.
- Ground truth round-trips through the emitted files exactly, including `note is None`.
- B0 produces a plausible non-trivial number on `test` and fails where predicted:
  `A07_MISSING_REFERENCE` recall 0.00, `A09_SPLIT_PAYOUT` 0.60.
- `EVALUATION.md` is deterministic apart from the two labelled measured-timing rows.

---

## 8. Git / GitHub status

| | |
|---|---|
| Local repo | initialised, branch `main`, working tree clean |
| Commits | `8eb6f91` Step 0 · `a7042fa` Step 1 · `46f4117`/`fb8f568` handoff · Step 2 |
| Tracked files | 67 |
| Remote | `origin` → `https://github.com/SatvikO7/ledgerloop.git` (**private**) |
| Pushed | all three commits; `main` tracks `origin/main` |

Excluded from the repo and verified: `.local/`, `.claude/`, `.venv/`, `data/generated/`,
`.hypothesis/`, `.env`, `EVALUATION.md`, `reports/`, caches. No secret-shaped files are
tracked.

---

## 9. Blockers and pending setup

**No blockers.** The former one — the missing GitHub remote — is resolved. The `gh` CLI is
now installed (`winget install GitHub.cli`) and authenticates as `SatvikO7` using the
OAuth token already in Windows Credential Manager (`repo`, `workflow`, `gist` scopes),
retrievable with `git credential fill`. It lacks `read:org`, so `gh auth login --with-token`
is rejected; export it as `GH_TOKEN` for `gh` commands instead, or run
`gh auth login --web` once to get a full-scope token.

**Not blockers, but pending:** no CI workflow yet (planned alongside Step 2's eval gate);
no Groq / Google AI Studio keys obtained yet — not needed until Step 9.

---

## 10. Exact next action

1. **Begin Step 3 — ingest + normalize.** Three parsers producing the canonical records
   that already exist in `models/records.py`:
   - `src/ledgerloop/ingest/ledger.py` — `ledger_orders.csv` → `CanonicalOrder`. The
     easy one; do it first to settle the reader's shape.
   - `src/ledgerloop/ingest/psp.py` — `psp_settlements.json` → `CanonicalSettlement` +
     `CanonicalPayment`. Must populate `order_ref_normalized` from `order_ref_raw`,
     recovering the three deliberate corruptions (null, space-separated, and the
     `chr(0x2011)` non-breaking hyphen) and leaving `None` when it cannot.
   - `src/ledgerloop/ingest/bank.py` — `bank_statement.csv` → `CanonicalBankTxn`, with
     the **regex-first** narration extraction filling `extracted_utr` /
     `extracted_merchant`, and `DD/MM/YYYY` resolution. Both stay `None` under A07.
   - Every record keeps its `RawRecord` provenance — the audit trail has to be able to
     show a controller the original line, not just the system's reading of it.
   - Reject `Currency.USD` with a clear message (`Currency.USD.supported is False`),
     so the A11 cut is testable rather than merely absent.
2. Write `.local/steps/step-03-ingest.md`, commit, push.

**Do not** build a matching tier in Step 3. T0 is Step 4, and it gets scored the moment
it exists — `ledgerloop eval` is already waiting for it.

**Reuse, do not rebuild:** the naive readers in `eval/baselines.py` are B0's own and
must stay naive. A baseline sharing the system's normalisation measures the system
twice.

---

## 11. Context that would otherwise be lost

### Bugs already found and fixed — do not reintroduce

- **Split leakage.** `Random(seed)` alone made `train` and `test` share their first 300
  orders. The split name **must** stay in the seed string.
- **A06 claimed the wrong settlement's `amount` aspect** (source instead of target),
  letting a claw-back overwrite an A02 drift and lose 2 paise. Caught by the conservation
  property test.
- **Primary-label collision.** Two settlement-level anomalies overwrote each other's
  `anomaly_class`, making a class vanish from the truth set. Fixed with `ASPECT_PRIMARY`;
  A09 now labels the *second bank credit*, not the settlement.
- **Trailing one-payment batches** were 1:1 joins that inflated tier yields. Folded away.
- `MinorUnits` briefly serialised to a JSON string, which broke round-tripping. It
  serialises as `int`.
- **`GroundTruthLink.anomaly_class` is never populated by the generator** — every
  emitted link carries the default `A01_CLEAN`. The first per-class recall table
  rendered exactly one row (`A01_CLEAN | 100%`) with all ten other classes silently
  missing, which is the precise failure PLAN.md §9.1 asks that table to prevent, and it
  looked entirely plausible. Fixed in the evaluator, not the generator: see decision 19.
  **Do not "fix" this in `ground_truth.py`** — a single link label cannot represent a
  link broken in two ways, and relabelling would rewrite the committed fixture's bytes
  for no measurement gain.
- The Wilson upper bound at 250/250 evaluates to `0.9999999999999998`. The two boundary
  cases are pinned to exact 0.0 / 1.0 rather than left to round.

### Environment quirks (Windows)

- Use `.venv/Scripts/python.exe`, not `python`.
- The console is **cp1252** — printing `₹` raises `UnicodeEncodeError`. `cli.main()` calls
  `_force_utf8_output()` first. Any new entry point needs the same, and ad-hoc
  `python -c` snippets need `sys.stdout.reconfigure(encoding='utf-8')`.
- In the Bash tool, `$TMPDIR` resolves to the Git install directory. Use an explicit
  scratch path.
- `mypy` needs no arguments — `packages = ["ledgerloop"]` is set in `pyproject.toml`.

### Conventions to keep

- **Commits omit the AI co-author trailer**, by the user's instruction to keep the public
  repo free of tooling references. `.claude/settings.local.json` is Claude Code's local
  permission allowlist — it was **not renamed** (the directory name is hardcoded and
  renaming breaks permissions) and is gitignored instead.
- `PLAN.md` stays unedited. Divergences go in `ARCHITECTURE.md` §6.
- Step docs go in `.local/steps/`, never committed.
- Dependencies are added by the step that first needs them. Step 2 may add `pandas` or
  `polars`; nothing else yet.

### Generator gotchas for the next step

- The test helper `_dataset()` in `test_generator.py` defaults to
  `ensure_class_coverage=True`. Pass `ensure_class_coverage=False` when testing anything
  prevalence-related.
- **`ensure_class_coverage` must stay off for `train` / `calibration` / `test`** — it
  distorts prevalence. It exists only for the fixture.
- The `calibration` split (200 orders) may legitimately contain **no A12** (1% prevalence
  → ~13% chance of zero). That is expected, not a bug; the test asserts only classes with
  expected count ≥ 3.
- The dev fixture has only ~5 settlements. It is for exercising code paths, **not** a
  statistical sample.
- ~20% of `order_ref` values are corrupted in the baseline (null, space-separated,
  or carrying `chr(0x2011)` a non-breaking hyphen). This is *not* an anomaly class — it is
  why T0 cannot reach 100% on clean money.
- Regenerate the fixture with `make fixtures` if the generator changes; it is committed
  and a diff will show up.

### Eval-harness gotchas

- `EVALUATION.md` is **not** byte-identical between runs: wall clock and throughput are
  measured. They are confined to a `#### Measured timings` block, and the determinism
  test filters exactly those two lines. Anything else that varies is a bug.
- **A05 `DUPLICATE_CREDIT` never appears in the per-class recall table**, correctly: the
  duplicate bank row has no truth link by construction, so it cannot be recalled. Its
  damage shows up in precision, as false positives.
- `run_b0` asserts each payment's **gross** amount. It has no fee model and no
  allocation, so its reconciled-rupee figure runs above the truth even where its links
  are right. That is part of what the baseline demonstrates — leave it.
- Tests reach the committed fixture via `Path(__file__).resolve().parents[2]`, not a
  relative path, so the suite passes from any working directory.
- The `scored` fixture in `test_eval_baselines.py` is **module-scoped**. A class-scoped
  fixture defined as an instance method is deprecated in pytest 8 and warns.
- `make eval` regenerates the `test` split before scoring, so a report can never be
  produced against stale data.
