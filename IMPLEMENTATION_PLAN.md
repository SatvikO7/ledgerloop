# IMPLEMENTATION_PLAN.md — LedgerLoop

The approved build order. [PLAN.md](PLAN.md) is the project design; this file is the
sequence it gets built in, and [ARCHITECTURE.md](ARCHITECTURE.md) records the
decisions each step settles.

**Ordering principle:** get to a measurable number as early as possible, then never
lose it. Every step from 4 onward ends with the evaluation still passing.

**Two structural properties of this order:**

1. **Steps 0–8 involve zero LLM calls and zero external services.** The entire
   "never cut" list (three-way structure, aggregation solver, calibration, typed
   exceptions, baseline table) is reachable with Python and a CSV reader alone.
   That is the risk management that matters most in a solo, time-boxed build.
2. **LangGraph comes at step 11, not step 3.** Wrapping working, tested functions in
   a state machine is a day. Building *inside* a framework while the data model is
   still settling taxes every step before it.

---

## MVP scope decisions

Settled before implementation; reasoning in `ARCHITECTURE.md` §5.

| Decision | Effect |
|---|---|
| **Neo4j cut** | NetworkX is the real T4 implementation, behind `graph/interface.py` |
| **ChromaDB cut** | T3 is lexical-only; `vector/interface.py` holds the contract |
| **A11 FX/multicurrency cut** | INR only; `Currency.USD.supported is False` |
| **Streamlit UI** | React is stretch only |

---

## Steps

| # | Step | Status |
|---|---|---|
| 0 | **Contracts** — money module, Pydantic models, link-level truth schema, config, deferred interfaces, and the written definition of the evaluation unit | ✅ Complete |
| 1 | **Generator + ground truth + property tests** — 11 anomaly classes, 5 splits, seeded byte-identical regeneration, money conservation | ✅ Complete |
| 2 | **Eval harness — before any matcher exists** — `metrics.py`, `report.py`, and B0 (exact-join) as the first "system" | ⬜ |
| 3 | **Ingest + normalize** — three parsers, canonical schema, regex narration, DD/MM resolution | ⬜ |
| 4 | **T0 + T1** — exact key and tolerance. *First defensible number.* | ⬜ |
| 5 | **T2 aggregation solver** — bucketing, settlement anchoring, meet-in-the-middle DP, greedy fallback, 200 ms cap, uniqueness check → `AMBIGUOUS_AGGREGATION` | ⬜ |
| 6 | **T3 lexical + T4 graph inference** — normaliser + RapidFuzz; four constraint rules over NetworkX; the re-run loop | ⬜ |
| 7 | **Blender + calibration** — top-k candidate labelling, logistic on residual tiers, isotonic, precision-targeted thresholds, decision policy | ⬜ |
| 8 | **Exception taxonomy + deterministic classifier** — class, severity, ₹ impact, evidence chain, template root cause, bounded auto-resolution, `UNMATCHABLE` floor. *System is complete and demoable with zero LLM calls.* | ⬜ |
| 9 | **LLM layer** — client, cache, Pydantic validation, `verify_arithmetic` gate, `--no-llm`, then the three call sites, then the cost ledger | ⬜ |
| 10 | **B2 (LLM-only) + full ablation + multi-seed sweep** | ⬜ |
| 11 | **LangGraph assembly** — wrap existing node functions; checkpointing for replay | ⬜ |
| 12 | **Streamlit UI + audit replay** — four tabs | ⬜ |
| 13 | **Packaging** — `make demo`, Compose, README metrics table, DEMO.md, video | ⬜ |
| 14 | **Stretch, in order** — provider failover ladder · LLM-written root causes · FastAPI + React + Cytoscape · Postgres · Neo4j adapter · A11 FX · Chroma as an ablation row · 5,000-record scale run · hosting · PDF source · k3s | ⬜ |

---

## Per-step definition of done

Every step from here ends with all of:

- Relevant tests written and passing
- `ruff check .` clean
- `mypy` clean (strict on `models`, `config`, `money`)
- Private step documentation written to `.local/steps/step-NN-*.md` (never committed)
- Implementation committed and pushed

> The remote is `SatvikO7/ledgerloop` (private), tracked as `origin/main`. Steps 0 and 1
> are pushed; every step from here pushes as part of its definition of done.

---

## Progress notes

Execution has matched the planned order with no reordering or skipped work. Two
within-step additions are worth recording, both already reflected in
`ARCHITECTURE.md`:

- **Step 0** added a `train` split (400 orders) that PLAN.md did not have. The score
  blender is a fitted model, so it needs its own fitting data — otherwise the isotonic
  calibrator sees in-sample scores.
- **Step 1** added `GeneratorConfig.ensure_class_coverage`, an opt-in pass that
  guarantees one effect per anomaly class. It **distorts prevalence** and is therefore
  used *only* for the committed fixture set, never for `train` / `calibration` / `test`.
