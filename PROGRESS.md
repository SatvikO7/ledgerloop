# PROGRESS.md — session handoff

**Last updated:** 2026-08-25 · **Branch:** `main` · **HEAD:** `a7042fa`

Read this first, then [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the build
order and [ARCHITECTURE.md](ARCHITECTURE.md) for the decisions that are already settled.
[PLAN.md](PLAN.md) is the original project design and is deliberately left unedited —
every divergence from it is recorded in `ARCHITECTURE.md` §6.

---

## 1. Current status

**Step 1 of 14 complete.** The project has a tested type layer and a working, seeded
synthetic data generator with link-level ground truth. **No matching, ingest, or
evaluation logic exists yet** — that starts at Step 2.

Everything runs offline with one runtime dependency (`pydantic`). No LLM calls, no
Docker, no external services have been introduced.

```
324 tests passing · 99% coverage · ruff clean · mypy --strict clean (28 files)
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

---

## 3. What has NOT been completed

**Steps 2–14 — nothing in this list exists yet:**

| Missing | Step |
|---|---|
| Eval harness (`metrics.py`, `report.py`), B0 baseline | 2 |
| Ingest / normalize — the three parsers, narration regex, DD/MM resolution | 3 |
| Any matching tier (T0–T5) | 4–6, 9 |
| Score blender, isotonic calibration, threshold selection, decision policy | 7 |
| Exception classifier, root causes, bounded auto-resolution | 8 |
| LLM client, response cache, `--no-llm` | 9 |
| Baselines B1/B2, ablation table, multi-seed sweep, `EVALUATION.md` | 10 |
| LangGraph assembly, audit replay | 11 |
| Streamlit UI | 12 |
| `make demo`, Docker Compose, DEMO.md, video | 13 |
| All stretch items (provider ladder, React, Neo4j, Chroma, hosting, k3s) | 14 |

**Also not done:** no `data/seeds.json` of blessed seeds; no CI workflow
(`.github/workflows/ci.yml`); the `scale` split generates but is not asserted on in tests.

---

## 4. Current implementation step

**Completed: Step 1.**
**Next: Step 2 — "Eval harness, before any matcher exists."**

Step 2 is deliberately ordered before any matching logic: building the scoreboard first
means no later change is ever a guess about whether it helped. B0 (exact-join) is ~20
lines and immediately produces a real number.

---

## 5. Files created / modified

### Source (`src/ledgerloop/`)

```
__init__.py  money.py  config.py  state.py  cli.py
models/      __init__ base enums refs records truth candidates
             decisions recon_exception metrics audit
generator/   __init__ vocab world baseline scenarios ground_truth
             emitters generate
graph/       __init__ interface        (Protocol only — no implementation)
vector/      __init__ interface        (Protocol only — no implementation)
```

### Tests (`tests/`)

```
unit/      test_money  test_models  test_truth  test_config  test_metrics
           test_state_and_interfaces  test_generator  test_emitters
           test_generator_edges
property/  test_money_invariants  test_generator_invariants
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

---

## 7. Tests

```bash
.venv/Scripts/python.exe -m pytest                                   # 324 passed in ~4.4s
.venv/Scripts/python.exe -m pytest --cov=ledgerloop --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check .                             # All checks passed!
.venv/Scripts/python.exe -m mypy                                     # 28 files, clean
```

| Metric | Result |
|---|---|
| Tests | **324 passed**, 0 failed |
| Coverage | **99%** (1434 stmts, 6 miss; 232 branch, 6 partial) |
| `ruff check .` | clean |
| `mypy` (strict) | clean, 28 source files |

**Step 1 acceptance criteria — all met:**

- Byte-identical regeneration verified with `cmp` on all six files.
- Conservation residual **0** across 72 split × difficulty × seed combinations.
- All 11 anomaly classes present in the fixture and in `test`.
- Realised prevalence within **±2%** of configured, measured on 2,000 draws.

---

## 8. Git / GitHub status

| | |
|---|---|
| Local repo | initialised, branch `main`, working tree clean |
| Commits | `8eb6f91` Step 0 · `a7042fa` Step 1 · `46f4117` handoff |
| Tracked files | 57 |
| Remote | `origin` → `https://github.com/SatvikO7/ledgerloop.git` (**private**) |
| Pushed | all three commits; `main` tracks `origin/main` |

Excluded from the repo and verified: `.local/`, `.claude/`, `.venv/`, `data/generated/`,
`.hypothesis/`, `.env`, caches. No secret-shaped files are tracked.

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

1. **Begin Step 2 — the eval harness**, in this order:
   - `src/ledgerloop/eval/metrics.py` — link-level precision / recall / F1 against
     `GroundTruth.evaluation_pairs`, with a **confidence interval** on precision
     (`LinkMetrics.precision_ci_low/high` already exist). Match rate denominator is
     `GroundTruth.reconcilable_refs` — `UNMATCHABLE` records are excluded and reported
     separately as the honest ceiling.
   - `src/ledgerloop/eval/baselines.py` — **B0 only**: a `pandas`/stdlib exact-join on
     UTR. This is the "why not just SQL" answer and the harness's first real input.
   - `src/ledgerloop/eval/report.py` — writes `EVALUATION.md` (gitignored; regenerated,
     never hand-typed).
   - Wire `ledgerloop eval` into `cli.py`; add a `make eval` target.
   - Tests: metric correctness on hand-built tiny truth sets, the CI-bound edge cases
     (zero predictions, zero truth), and B0 producing a plausible non-trivial number on
     the fixture.
2. Write `.local/steps/step-02-eval-harness.md`, commit, push.

**Do not** start ingest or any matching tier before the harness produces a number.

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
