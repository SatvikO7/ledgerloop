# ARCHITECTURE

> **Scope of this document.** As of Step 0 this covers the **data model and the
> definitions the evaluation depends on**. The tier ladder, graph model, calibration
> approach and decision policy get their sections as they are built. What is written
> here is settled and enforced by tests; nothing below is aspirational.

---

## 1. Layering

```
ledgerloop.models     pure Pydantic contracts, no internal deps beyond each other
        ↑
ledgerloop.config     RunConfig and friends — thresholds, tolerances, bounds
        ↑
ledgerloop.state      ReconState — holds a RunConfig, so it is not a "model"
```

`ReconState` lives at `ledgerloop.state`, not `ledgerloop.models.state`. Putting it
inside `models` creates a real import cycle: importing `models.base` runs
`models/__init__`, which imports state, which imports config, which imports
`models.base`.

`ledgerloop.money` sits beneath everything and imports nothing from the project.

---

## 2. The unit of evaluation — `PAYMENT_CREDITED_AS`

**This is the single most important definition in the project.** Every headline
number depends on it, and PLAN.md §9 never stated it: "match rate = auto-matched /
total reconcilable" leaves open *auto-matched what* — orders, payments, bank
transactions, or links. In a three-way N:M problem those give materially different
numbers.

### The atomic unit is a link, not a record

```
(Order) ──ORDER_PAID_BY──▶ (Payment) ──PAYMENT_SETTLED_IN──▶ (Settlement) ──SETTLEMENT_CREDITED_AS──▶ (BankTxn)
             │                                                                                             ▲
             └──────────────────────────── PAYMENT_CREDITED_AS ────────────────────────────────────────────┘
                                          (derived closure — THE EVALUATION UNIT)
```

Precision, recall and F1 are computed over the set of `(payment_id, bank_txn_id)`
pairs. `GroundTruth.evaluation_pairs` is the truth set; `MatchCandidate.pair` and
`MatchDecision.pair` produce comparable keys.

**Why links and not records.** A settlement of 14 payments arriving as one bank
credit is 14 links, not 1 match and not 14 matches-of-a-record. Counting records
forces an arbitrary answer to "is a partially-resolved settlement 0.5 matched?";
counting links does not. It is also the only unit under which a split payout (A09)
and a duplicate credit (A05) are expressible at all.

**Why only the closure edge.** `ORDER_PAID_BY` and `PAYMENT_SETTLED_IN` are largely
*asserted by the sources* — the PSP file says which payments are in which settlement.
Counting them would inflate every score with edges the system never had to work for.
`PAYMENT_CREDITED_AS` is the edge that has to be *earned*, because nothing in the
three files states it.

### Definitions, stated once

| Term | Definition |
|---|---|
| **True positive** | An `AUTO_MATCHED` decision whose pair is in `evaluation_pairs` |
| **False positive** | An `AUTO_MATCHED` decision whose pair is not |
| **False negative** | A pair in `evaluation_pairs` with no `AUTO_MATCHED` decision |
| **Auto-match precision** | TP / (TP + FP) |
| **Match rate** | auto-matched links / links among `reconcilable_refs` |
| **False-positive cost** | Σ `impact_minor` over false positives — a rupee figure, not a ratio |

`NEEDS_REVIEW` is **not** a positive prediction. It is a referral to a human, and
counting it as a match would inflate precision exactly where the plan warns against
it. Enforced by `MatchDecision.is_positive_prediction`.

### The denominator excludes `UNMATCHABLE`

`ExpectedStatus.UNMATCHABLE` marks items irreconcilable without data that does not
exist in the three sources. They leave the match-rate denominator
(`GroundTruth.reconcilable_refs`) and are reported as a separate line. Charging the
system for them would understate it as dishonestly as excusing real failures would
overstate it.

An `EXCEPTION` is **not** unmatchable — it is a resolvable item the system failed to
resolve, and it stays in the denominator.

### Secondary, business-facing metric

Order-level **closure** — "did this order's money reach the bank?" — is reported
alongside the link metrics because it is what a controller actually asks. It is a
derived rollup, never the primary unit.

---

## 3. The money invariant

> No `float` ever touches the money path.

Enforced in three places, not by convention:

1. **`ledgerloop.money.assert_minor`** — the single gate. Rejects `float`, `bool`
   (an `int` subclass, so `True + 499900 == 499901` would pass silently) and
   `Decimal` (must be converted deliberately).
2. **`MinorUnits`** — an annotated type that routes every `*_minor` model field
   through that gate, so a float cannot enter a model even via `model_validate` on
   parsed JSON.
3. **Tests** — property tests over the arithmetic, plus a reflective test that walks
   every model and fails if a `_minor` field is ever added unguarded.

### The one sanctioned crossing

`money.delta_ratio()` returns a `float`. It exists to feed
`FeatureVector.amount_delta_ratio`, and its result may never be written back into a
money field.

| Space | Type | Examples |
|---|---|---|
| **Money** | `int` minor units | `amount_minor`, `impact_minor`, `net_delta_minor` |
| **Feature** | `float` | `amount_delta_ratio`, `lexical_score`, `calibrated_p` |

### Parsing

Text becomes money exactly once, at the ingest boundary, through
`parse_minor_units()` (already-minor integer text) or `parse_major_to_minor()`
(major-unit decimal text, via `Decimal`, exact). Sub-minor-unit precision is an
**error**, not a silent round — `"36803.234"` means the file or the scale assumption
is wrong, and both deserve a loud failure.

### Allocation

`allocate_minor()` splits a total by weights using largest-remainder allocation, so
parts sum to *exactly* the total. Required by A09 `SPLIT_PAYOUT` and by fee
apportionment. Ties break by position, keeping it a pure function — seeded
regeneration must be byte-identical.

---

## 4. Ground truth is link-level

Generated first; data derived from it; never inferred after the fact.

PLAN.md §5.3 specified a flat wide row (`order_id | payment_id | settlement_id |
bank_txn_id | ...`). That schema cannot represent two of its own anomaly classes:

- **A09 `SPLIT_PAYOUT`** — one settlement arrives as *two* bank credits. A single
  `bank_txn_id` column has nowhere to put the second.
- **A05 `DUPLICATE_CREDIT`** — the same UTR credited twice. The wide row cannot
  distinguish the genuine credit from the duplicate.

So truth is:

- **`GroundTruthLink`** — a typed edge that *should* be discovered, with the money
  attributed to it. A bank credit that should match nothing (A05's duplicate, A10's
  orphan, the rent/salary noise rows) simply has **no link**, and any system that
  produces one has made a false positive the evaluator will count.
- **`GroundTruthRecord`** — one verdict per record: `expected_status`,
  `anomaly_class`, `impact_minor`, and a generator `note` describing what it did.

---

## 5. Scope decisions and their reasoning

### 5.1 Neo4j — cut; NetworkX is the real implementation

PLAN.md Phase 4 required the NetworkX fallback to produce **identical decisions** to
Neo4j on the fixture set. That acceptance criterion is an admission: if the outputs
must match exactly, the database contributes zero decision quality while costing a
container, a driver, a query dialect, health checks and testcontainers.

The four T4 rules — sibling completion, path closure, exclusivity pruning, ring
detection — are constraint propagation over an adjacency structure, a few dozen lines
in memory. `graph/interface.py` defines the `GraphRepo` Protocol so a `Neo4jGraphRepo`
slots in later without touching a call site. "Swappable backend, zero-infra default"
is a better story than "requires a graph database".

### 5.2 ChromaDB — cut; T3 is lexical-only

PLAN.md §6.3 proposed embedding merchant-name variants so kNN could resolve
`RZRPAY SFTWR P L` → `Razorpay Software Private Limited`. Two problems:

1. **Sentence embeddings are weak at exactly this task.** MiniLM is trained on
   natural-language semantics and has no reason to place a vowel-dropped consonant
   skeleton near its expansion. What solves it is deterministic: uppercase, strip
   legal suffixes (`PVT`, `LTD`, `P L`), expand a small abbreviation table, then
   fuzzy-match the skeleton.
2. **The vocabulary is tiny and self-generated.** Merchant names come from our own
   generator — tens of distinct names. A vector database for that is infrastructure
   without a job.

`FeatureVector.semantic_score` stays `0.0`. `vector/interface.py` holds the contract.
If embeddings are added later they become an **ablation row**: "we tried embeddings
and lexical won" is a stronger result than a silent dependency that never earned its
place.

### 5.3 A11 FX / multicurrency — cut

Multicurrency forces an FX rounding policy into the money path for 2% of records, and
`amount_gross_paise` with `currency: USD` is incoherent as specified. `Currency.USD`
exists in the enum with `supported == False`, so the cut is **explicit and testable**
rather than merely absent — ingest will reject it with a clear message instead of
mis-scaling it as paise. Its 2% prevalence is reassigned to `CLEAN` (0.67, against the
plan's 0.65).

### 5.4 Streamlit — the UI plan; React is stretch

The UI is scheduled after every metric exists. Two days is not a React + Vite +
Tailwind + shadcn + Recharts + Cytoscape app.

---

## 6. Corrections to PLAN.md, and why

| # | Plan said | Now | Reason |
|---|---|---|---|
| 1 | Splits: dev / calibration / test | **+ `train` (400)** | The blender is a fitted model. Fitting the logistic and the isotonic on the same split lets the calibrator see in-sample scores and report a calibration quality the system does not have. Fit on `train`, calibrate on `calibration`, report from `test`. |
| 2 | Flat ground-truth row | **Link-level truth** | The flat row cannot represent A05 or A09. |
| 3 | `tier_id` as a logistic feature | **One-hot; T0/T1 bypass the blender** | Tier is near-perfectly predictive (T0 exact-key is always right). As an ordinal it dominates every coefficient and collapses the model into a tier lookup. |
| 4 | ECE over all matches | **Calibration measured on residual tiers only** | T0/T1 are ~70% of volume at p≈1.0. Including them yields one populated bin and a near-zero ECE that measures the shape of the corpus, not the calibrator. `CalibrationMetrics.residual_only` defaults to `True`, and `populated_bins` is reported so a degenerate diagram is visible. |
| 5 | Exception taxonomy "mirrors" the 12 anomaly classes | **13 exception classes vs 11 anomaly classes** | They answer different questions. `CLEAN` is never an exception; `AMBIGUOUS_AGGREGATION` and `UNKNOWN_RESIDUAL` are system states with no generator counterpart; `UNMATCHABLE` is producible by several anomalies. The mapping is a *measured* artefact — the confusion matrix — never a hardcoded identity, and the matrix is rectangular. |
| 6 | `Exception` model | **`ReconException`** | Shadowed the builtin inside a module that also raises errors. It is a data record describing money, not a raisable error, and does not inherit `BaseException`. |
| 7 | Bare precision point estimate | **Precision + confidence interval** | ~250 auto-matched links means one error moves precision 1.000 → 0.996. A point estimate cannot distinguish 0.99 from 0.97 at that N. |
| 8 | `mypy --strict` on `core/` | **`models` + `config` + `money`** | There was no `core/` directory in the plan's own structure. |
| 9 | B0 is `pandas.merge` on UTR | **Stdlib dict join on UTR** | The semantics PLAN.md §9.2 asks for are the exact join, not the library. Adding `pandas` as a runtime dependency for a twenty-line group-and-lookup would contradict the project's own claim to run on nothing, and B0 is faster and easier to audit without it. The join is identical; only the implementation differs. |
| 10 | Per-anomaly-class recall read off the truth labels | **Attributed through the link's endpoint records** | `GroundTruthLink.anomaly_class` exists on the model but the generator only ever labels *records* — every emitted link carries the default `A01_CLEAN`. Grouping on the link field produces a table with one 100%-clean row and all ten other classes silently missing, which is the exact failure PLAN.md §9.1 asks the table to prevent. A link is therefore attributed to every anomaly touching either endpoint. |
| 11 | Per-class recall rows partition the links | **Rows may overlap** | Decision 14 is that anomalies compose along independent aspects, so a link can be genuinely broken in two ways. Forcing it into one row would misreport whichever class lost the tiebreak. The rows do not sum to the total and the report says so. |
| 12 | Match rate = auto-matched / total reconcilable | **Denominator restricted to payment and bank records** | The evaluation unit runs between a payment and a bank transaction, so those are the only records producing one can resolve. Orders and settlements are attached by structural edges the sources assert: counting them would charge the matcher for records it was never asked to match. |

---

## 7. Invariants enforced at the type boundary

These cannot be bypassed by forgetting a convention:

| Invariant | Where |
|---|---|
| No float or bool in any money field | `MinorUnits` annotation |
| `AUTO_MATCHED` requires `arithmetic_verified` | `MatchDecision` validator |
| `AMBIGUOUS_AGGREGATION` carries ≥2 hypotheses | `ReconException` validator |
| Hypotheses ordered by descending probability | `ReconException` validator |
| `UNMATCHABLE` is never agent-resolvable | `ReconException` validator |
| A candidate's tier agrees with its feature tier | `MatchCandidate` validator |
| `subset_members` only on T2, and `subset_size` agrees | `MatchCandidate` validator |
| Anomaly prevalence sums to exactly 1.0 | `GeneratorConfig` validator |
| T5 cannot be enabled with `llm.enabled=False` | `RunConfig` validator |
| Decisions and audit events are immutable | `FrozenLedgerModel` |
| Unexpected keys are rejected (LLM output validation) | `extra="forbid"` |

The settlement identity `net = gross − fee − tax + adjustments` is deliberately **not**
enforced — A03 `FEE_TAX_MISMATCH` breaks it on purpose, and a validator would delete
the anomaly the system exists to detect. It is exposed as
`CanonicalSettlement.net_delta_minor` and reported as *evidence*.

---

## 8. Audit and replay

Append-only. A revision writes a new `MatchDecision` carrying `supersedes`; history is
never edited. `ReconState.open_decisions` is the current view over that history.

`AuditEvent.sequence` is the replay cursor, not the timestamp — several events
routinely share a millisecond, and replay must be exactly reproducible. LLM
provenance (`prompt_hash`, `provider`, token counts, latency) lives on the event, so a
replay can prove a run consumed zero live API calls.
