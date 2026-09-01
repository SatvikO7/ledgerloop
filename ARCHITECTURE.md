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

Decisions 1-51 are Steps 0-13. **52-59 are Phase 2.1-2.4** and **60-63 are Phase
2.5** -- each of them a change the final audit asked for, or a finding that came out
of measuring one. Two of the four below (62 and the placement half of 63) are defects
the probing turned up rather than features that were planned.

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
| 13 | `DD/MM/YYYY` is the bank's format | **The component order is inferred from the column** | Per row the ambiguity is unresolvable; per column it usually is not, because a statement covering more than a fortnight contains a day past the twelfth. Assuming the Indian convention would be right on this corpus and unverifiable on any other. Three outcomes are distinguished: proven, contradictory (raises — picking a winner silently misdates half the rows), and undecidable (the convention applies and `proven` is `False`, so the report can say the dates rest on a convention). Getting this wrong is invisible: it shifts transactions by up to eleven months, destroying T1's date window and T2's bucketing while leaving T0 untouched. |
| 14 | `ingest/normalize/canonical.py` and `ingest/normalize/money.py` as separate packages | **One `ingest/` package; money stays at `ledgerloop.money`** | The plan's `normalize/money.py` would be a second money module alongside the one Step 0 built, which is exactly how two rounding conventions end up in one codebase. `ingest/` reads and normalises; the money gate it routes every amount through is the same `parse_minor_units` the rest of the system uses. |
| 15 | `parse_narration` is "LLM + regex" | **Regex only at Step 3; the LLM is a Step 9 fallback over the same result** | PLAN.md §7.3 already says regex-first. Building the deterministic half alone, and measuring it, is what makes the LLM's contribution attributable later: `NarrationParse.resolved_by_regex` is the routing signal and the cost ledger reports the ratio. On `test` the regex layer resolves 45 of 80 credits with no model involved. |
| 16 | (unstated) Ingest resolves references to orders | **Ingest asserts no cross-source link** | The data is right there, and joining it here would move T0's work into an unmeasured layer — so Step 4's "first defensible number" would be measuring two things at once. Ingest normalises; step 4 relates. |
| 17 | T0 is exact-key, T1 is tolerance — two implementations | **One resolver, parameterised by a rule; T0 is the zero-width-band case** | `tolerance_minor(net, floor=0, bps=0)` is `0` and `within_tolerance(a, b, 0)` is equality, so T0 is not a special case of the resolver but the resolver at its limit. Two modules would be a hundred duplicated lines that could drift apart, and the one place drift would show is T1 quietly accepting something T0 declined for a reason. |
| 18 | T0/T1 bypass the blender at `p = 1.0` | **`p = 1/n` over `n` indistinguishable contenders; 1.0 is the `n = 1` case** | Deterministic certainty is certainty about *which* link, and it holds only where the key resolves uniquely. The common path is unchanged. The gain is that a contested pair at `p = 0.5` falls below the configured `tau_low` and routes to `EXCEPTION` **through the policy** — which is exactly ground truth's verdict for an A05 duplicate — instead of through a special case written into the tier. |
| 19 | An exact key identifies the payout | **Mutual uniqueness: a match must be unique from both sides** | Counting keyed contenders asks only whether the *settlement* is uniquely explained. Anomalies compose: A05+A07 can strip the reference from the row that was really the payout, leaving the duplicate as the only keyed row. A one-sided check then matches the entire batch to the duplicate at `p = 1.0`. Found by a property test at seed 4, not by inspection. |
| 20 | A tolerance band absorbs amount noise | **A band absorbs *drift*; a shortfall equal to another unclaimed credit is a missing tranche** | 0.5% of a ₹4.37 lakh net is ₹2,186 — wider than a small A09 tranche — so the surviving tranche fits the band and is credited with payments it never carried. Exact equality against the open credits catches it with no search. The guard only ever *declines*: finding the partition stays T2's. |
| 21 | A record leaves the pool once decided | **Undecided is not decided, and a suspected split is undecided** | Three dispositions, not two. *No* qualifying credit means no conclusion — the settlement falls through, which is how A02 reaches T1. *Several* means contested and consumed. A key on several credits means the payout was split, and that is left **in** the pool: the question is how the payments partition, and consuming it would take it away from T2. |
| 22 | T2 finds the subset that composes a credit | **The solver counts subsets; it never chooses one** | PLAN.md §6.2's refusal rule is the solver's purpose, not a check applied to its output. It stops at `want` solutions and the caller reads the count. `SubsetSearch.exhaustive` travels with the answer because "I found one" and "I found the only one" license different decisions — so a greedy result, which can never prove exhaustiveness, is treated exactly like an outright ambiguity. |
| 23 | Subset-sum over payment amounts against the credit | **Search in gross space, verify in credit space** | Payments carry gross and credits carry net. The bridge is `allocate_minor(N, [g, G-g])[0]` — the same split the generator built the truth links from. The search window is that inversion widened by a paise at each end; the exact re-derivation through the `accept` hook is what decides. The window only prunes, so no rounding assumption in the inversion can admit a subset that does not reconcile. |
| 24 | Bucket candidate payments by a ±3-day window and merchant | **The settlement anchor is the bucket; no date filter on top** | Every payment carries `settlement_id` from the PSP's own nesting, so the anchor is strictly tighter than any date bucket and the difference is correctness, not speed — a global search would happily explain a credit with an unrelated batch's payments. Adding the date window back could only remove true tranches: A04 `TIMING_SHIFT` composed with A09 puts a legitimate second tranche five days out. The date gap is evidence, not a gate. |
| 25 | Greedy fallback with local swaps | **Greedy accumulation, no swap repair** | Provably useless under descending accumulation. If the run ends short with chosen set `C`, every skipped `x` satisfied `R + x > high` at the time. For `x > c`: `x` was seen before `c`, so `R <= T - c`, so `T - c + x > high` — overshoots. For `x < c`: `T - c + x < T < low` — undershoots. A brute force found no counterexample, then the proof explained why. Deleted rather than left as untested dead code. |
| 26 | (unstated) T2 solves each credit independently | **Transactional per settlement, and only for partitions covering the whole net** | A batch is solved whole or not at all: credits largest-first, each search over what the previous left, and the final tranche *verified against the remainder* rather than searched. Restricting to partitions that account for the net is what makes the arithmetic checkable end to end — it conserves exactly and assigns every payment once. A lone tranche whose sibling lost its reference to A07 belongs to the tiers that can match an unreferenced row. |
| 27 | T3 fuzzy-matches narration against a merchant legal name | **T3 derives its own merchant master from the statement's references** | No source file maps `RZRPAY SFTWR P L` to `MRCH_0001` -- there is no merchant master among the three inputs and the strings share no characters, so the comparison does not exist until it is built. Every credit carrying a UTR names a settlement, whose payments name orders, which name a merchant: each keyed credit is a labelled example. Built from *references* rather than from our own matches, so it cannot compound an earlier tier's error and does not depend on how well they did. |
| 28 | Pick a fuzzy-match threshold | **The two score distributions overlap, so the gate is a trade, not a tuning** | Measured over the whole vocabulary: the weakest *same*-merchant variant pair scores 0.667, the strongest *different*-merchant pairing 0.75. No cut admits every true pair and rejects every false one. The gate is placed at 0.90 for precision -- clearing every cross-merchant pairing by 0.15 -- and the four merchants whose variant pairs fall below it are named. Tuning down to catch them would admit a false auto-match. |
| 29 | NetworkX is the T4 implementation | **In-memory adjacency behind the Step 0 Protocol** | The same argument that cut Neo4j: PLAN.md §4 required identical decisions from the fallback, which concedes the backend buys no decision quality. The four rules are adjacency lookups over a few hundred nodes, and `graph/memory_repo.py` implements the Protocol in about as many lines as the import would cost. Either backend stays a drop-in. |
| 30 | T4's rules produce matches | **Deduction and induction get different confidence, and one rule produces none at all** | Path closure is deductive (`p = 1.0`); sibling completion is inductive and takes its confidence from the fraction of siblings supporting it, so an 80% majority routes to review rather than auto-matching. Exclusivity produces only refusals. Ring detection produces no candidates by design -- PLAN.md §6.4 calls it a bonus signal, not a match decision. |
| 31 | (unstated) Every tier contributes on every corpus | **T4's inference rules fire zero times here, and that is reported** | Every earlier tier matches at *settlement* granularity, so the partial assignments path closure and sibling completion exist to finish never arise. Both are tested against constructed partial states and reported as zero on the corpus. Loosening a rule until it fired would trade precision for the appearance of contribution. |
| 32 | scikit-learn for logistic + isotonic | **Both written out: ridge IRLS and PAVA, about 120 lines** | The argument that cut NetworkX (29) and pandas (9), with two additions specific to a fitted model. **Reproducibility:** the report claims a rerun reproduces byte for byte, which a fixed-tolerance Newton iteration does and an L-BFGS whose convergence path can shift between library versions does not. **Inspectability:** the pitch calls the blender inspectable, and a coefficient table printed by an optimiser in the same file is inspectable in a way `LogisticRegression().fit()` is not. The trade is named: scikit-learn is better tested, so these are tested against closed-form and separable cases whose answers are known independently. |
| 33 | (unstated) The blender is fitted on the candidates the tiers emit | **Fitted on the population it will *score*: contenders from decision points a tier resolved** | Every candidate the residual tiers assert is correct — 575 of 575 on five `train` corpora. A model fitted on that learns a base rate. The negatives are the pairings the tiers *considered and refused*, which is why Step 7 harvests top-k contenders at all. But a contender from a **refused** decision point is not in the deployment population: nothing from a refused point ever reaches the blender, and those points differ in a way the feature vector cannot see (an unbeatable rival). Fitting on them injects label noise conditional on the features — measured at a 30-link recall loss on `test` for no precision gain. So refused points are collected, labelled, reported as a diagnostic, and never fitted on. |
| 34 | (unstated) Calibration replaces every tier's probability | **A tier's *refusal* is never re-scored** | `arithmetic_verified=False` is how T2 marks two subsets that both fit and T3 marks a runner-up it cannot beat. That conclusion rests on evidence no feature carries, so a model that cannot see the rival must not overturn it. The tier's `1/n` stands, and the routing it produces — a two-way ambiguity at 0.5 falling below `tau_low` to an exception — is preserved exactly. |
| 35 | One `train` and one `calibration` split | **Several seeds of each, disjoint, recorded in the bundle** | One 400-order corpus leaves the residual tiers a few dozen decision points; a precision target of 0.99 cannot mean anything against that. Five `train` seeds and four `calibration` seeds give 575 and 140 links. `CalibrationProvenance` refuses to construct a bundle whose halves share a corpus, or whose either half is `test`, so the discipline is a type error rather than a convention. |
| 36 | `τ_high` is selected on achieved precision | **Selected on the point estimate, reported with its Wilson interval** | PLAN.md §6.5 states the rule in terms of achieved precision, and a Wilson *lower bound* at 0.99 would demand roughly three hundred consecutive correct predictions before any threshold qualified — more than the calibration half contains in total. So the selection follows the plan and the interval travels with the answer, exactly as every other precision figure in this project does. |
| 37 | Calibration is reported over the matches | **Two populations, side by side: asserted and contender** | The asserted population is what the system claimed, and on this corpus it is 68 links that are all correct — one populated bin and an ECE of 0.0000 that describes the tiers' refusal discipline rather than the calibrator. The contender population contains the wrong pairings, so it is the only place a reliability diagram has anything to show. Reporting one without the other would be either an empty claim or a claim about links the system never made. |
| 38 | The exception taxonomy mirrors the anomaly taxonomy | **A cascade over the sources, most-specific-first, with `UNKNOWN_RESIDUAL` as a real answer** | Decision 5 settled that the two vocabularies differ; Step 8 settled *how* the system reaches one. Every rule tests arithmetic or structure present in the three documents, tried most-specific-first, and the order is the argument: a settlement whose declared identity does not close is a `FEE_TAX_MISMATCH` before it is anything else, because the source states the discrepancy itself. An item nothing explains becomes `UNKNOWN_RESIDUAL` — a system state, not an anomaly. Smoothing it into the nearest plausible class would make the confusion matrix look good while telling a controller the wrong thing to do. |
| 39 | Exceptions are raised for what the ladder failed to match | **A matched record is not necessarily a clean one** | The first classifier scored **0.4667** exception recall and every miss was the same shape: a record the ladder matched *correctly* while something about it was still wrong. A charged-back payment is excluded from its batch's credit — the right match — but its money never arrived and it appears in no link. A settlement can be credited in full while its own identity does not close. Both are now items; the impact is the **discrepancy**, not the payout, because the money that did arrive is not at stake. Recall 0.4667 → **0.9333**. |
| 40 | Exception recall counts everything unresolved | **`UNMATCHABLE` is reported on its own line, and debits leave the unit** | Crediting the honest floor inside the headline would let a system inflate recall by describing items nobody could resolve — tested directly: 50 unmatchable rows and no real ones scores 0.0000. And a bank **debit** is money leaving the account, not a payout being reconciled, so 34 outgoing rows would be noise in a controller's queue; they leave both denominators and the report prints the count. A scope decision, not a silent omission. |
| 41 | (unstated) The agent may resolve what it is confident about | **Three classes, hard bounds, nothing posted, refusals printed** | PLAN.md §8.3 names `ROUNDING_DRIFT`, `TIMING_SHIFT` and `DUPLICATE_CREDIT`; everything else is proposal-only and `UNMATCHABLE` is refused by the model itself. A proposal past its bound is emitted as *refused* with the bound named rather than dropped — a leash nobody can see is not a leash. The resolver never touches a prediction: one that could add a link would be a sixth matching tier in disguise, and every precision figure in the project would become a claim about it. |
| 42 | `openai` / `httpx` for the LLM transport | **`urllib.request`, thirty lines** | The fourth application of decisions 9, 29 and 32. An OpenAI-compatible chat completion is one POST of one JSON body, and Groq, Gemini's compatibility layer, OpenRouter and Ollama all speak it — which is why PLAN.md §10 chose that shape. The trade is named: no connection pooling and a plainer error surface, neither of which matters at fewer than thirty calls per run. |
| 43 | T5 is a tier in the ladder | **T5 is *injected* into the ladder as a Protocol the ladder declares** | `matching` must not import `llm`. The moment it does, `--no-llm` stops being the same code path with one branch taken and becomes a second implementation — and a second implementation is one nobody measures. The pipeline declares `ResidualAdjudicator`; the CLI, which owns the model, supplies it or does not. |
| 44 | `llm_confidence` is calibrated with everything else | **And until it is, an LLM proposal carries an *unmeasured* probability** | Decision from PLAN.md §7.4, with the missing case filled in. Raw self-reported confidence is systematically overconfident, so a proposal asserting 0.99 would auto-match itself — precisely "the LLM decides a match by itself". Where a fitted bundle covers T5 the blender prices it like any other tier. Where it does not, `calibrated_p` is the middle of the review band: below any `tau_high`, above any `tau_low`, so an arithmetic-verified proposal reaches a human and never a ledger. On this corpus the calibrator never saw T5, so it abstains and T5 auto-matches nothing. |
| 45 | Validation is what makes LLM output safe | **A schema proves an answer's *shape*; three gates prove its *provenance*** | A model returning `{"utr": "UTR2026030412345"}` for a narration containing no such reference has produced a schema-valid, plausible, completely invented join key — and a join key creates a match out of nothing. So every value is checked against the text it was read from, every citation against the pack it was sent, and every id in prose against the records the exception involves. A partial hallucination discards the whole extraction: partial trust in an answer that invented a record is not a defensible position. |
| 46 | (unstated) The ablation reads tier counters off one run | **Every row is a separate run of the ladder** | `TierContribution` already records what each tier auto-matched, and subtracting those columns answers a different question. The pool is shared: switching T1 off does not remove T1's matches, it leaves T1's settlements *undecided* and T2 then sees them. A tier's marginal contribution is what the ladder does **without** it, which can only be measured by running the ladder without it. Six runs per seed instead of one — a few seconds — against a table that would otherwise describe an arithmetic identity rather than a system. |
| 47 | `config_hash` proves a configuration was held fixed | **It proves a *run's* identity; `tuning_hash` proves the configuration** | Split, difficulty and seed are in `config_hash`, correctly — a run over seed 43 is not a run over seed 42, and an audit trail saying otherwise would be wrong. But that makes it differ on every row of a multi-seed table by construction, so comparing it there witnesses nothing. `tuning_hash` excludes corpus identity *and* the ladder, which the ablation's row label already states. One value across a table is then a real check, and `run_ablation` refuses to build a table whose rows disagree. |
| 48 | (unstated) The report may import what produces its tables | **A document needs the *shape* of a result, never the machinery** | `report` renders the Step 10 tables; the runners import the run harness, which imports `llm`; and `matching.pipeline` imports `eval.metrics` for one contract type, which initialises the whole `eval` package. Those three close into an import cycle — and worse, they hand `matching` a **transitive dependency on `llm`** through a document renderer, which is exactly what decision 43 forbids. `eval/artifacts.py` holds the models and nothing that runs. A subprocess test asserts that importing `matching.pipeline` loads no `ledgerloop.llm` module. |
| 49 | An LLM baseline needs an LLM, so B2 cannot run here | **It runs against a labelled stand-in, and the report says which figures that covers** | No provider key exists in this environment. `--offline-provider` (opt-in, and **never** a fallback — without it and without a key B2 reports `ran=False`) answers B2's prompts by reading the prompt text and nothing else. Its **cost, cache, call and failure figures are measured machinery**: the same client, cache, budget, schema and ledger a live provider goes through. Its **precision and recall are a property of a documented rule and are not a claim about any model**, and `EVALUATION.md` prints a banner drawing that line. What the row demonstrates is reasoner-independent and is the point of B2: output asserted with no `verify_arithmetic` behind it is asserted wrong as readily as right — 18 false positives costing ₹4.33 lakh — and tokens scale with the corpus rather than the residual. |
| 50 | (unstated) A metric artefact may carry its wall clock | **No artefact carries a timing, and the tier table lost the one it had** | Byte-identity between two runs is the check that says a rerun *reproduced* a result rather than resembling it, and a stopwatch reading makes it unavailable. Timings are still reported — in the single labelled block of `EVALUATION.md` that exists to hold the numbers a diff should ignore. Finding this also turned up a contradiction the report had been carrying since Step 4: it claimed timings were confined to that block while the tier table published a per-tier wall clock. |
| 51 | A baseline should get every technique available | **B1 is given what a script reaches for, and not what T3 invents** | B1 gets fuzzy reference recovery and nearest-amount matching, argmax, with no uniqueness check and no margin — which is the shape of the code a reconciliation script actually contains, and it reaches recall 0.8810 with 137 false positives costing ₹35.38 lakh. It is **not** given a merchant master: there is none among the three sources, and deriving one from the statement's own references is T3 (decision 27). Handing a baseline the technique being evaluated would measure LedgerLoop twice; reading the generator's vocabulary would be worse, since that is not an input a real system gets. |
| 52 | A payout the bank posted twice is a contest no tier can settle | **It is a statement-hygiene problem, settled before the ladder runs** | Phase 2 measured where recall was going and found it was not spread across payments: 17 of 35 bank credits on `test` seed 42 were fully resolved, 18 were not resolved at all, and **none was partial**. 118 of the 164 missing links — 72% — belonged to ten settlements whose payout appears twice in the statement. The ladder's refusal was correct: two credits that qualify equally under a tier's own rule cannot be told apart *by that rule*. But they are not two candidate payouts; they are one payout posted twice, and *which of them is the payout* needs no similarity score — the **first** one is. `matching/duplicates.py` runs before T0, groups credits by exact amount and normalised narration, and calls the later members of a group re-postings. Recall 0.5808 → 0.8049 at standard difficulty, 0.4446 → 0.7414 on hard, **precision unmoved at 1.0000 and zero false positives across all thirty runs of both arms**. The re-posting is held out of the *matchable* pool and out of nothing else — it is still raised as `E_DUPLICATE_CREDIT` with its full amount, because somebody has to reverse it. |
| 53 | A recall improvement can be argued for | **It has to be run both ways** | The pass is `RunConfig.duplicates.enabled`, and `ledgerloop comparison` runs all fifteen corpora twice — same bundle, same ladder, one field apart — writing each arm's `tuning_hash` so *nothing else changed* is a check rather than a sentence. Every guard in the pass can only make it *decline* more (exact amount equality, identical narration, a unique earliest, a bounded window), so the knob cannot be turned to buy recall. Switching it off reproduces every pre-Phase-2 number to the digit — 130/0/164, recall 0.4422, match rate 0.4261, every ladder prefix and per-class figure to six places — asserted exactly by `test_metrics_regression.py` rather than approximately. |
| 54 | Exception recall was 0.8753 because two A06 records were hard | **They were being covered by accident, and removing the accident exposed it** | An order refunded *after* its payout is clawed back from a **later** batch, and nothing in the queue reached it: the later batch reconciles to the paise, the earlier one was paid in full, and the order appears in no unresolved link. It used to be reported only when some unrelated anomaly happened to leave its batch contested and that settlement's evidence chain named every order in it. The duplicate-posting pass matched those batches, the accident stopped, and nine coverages across five seeds went silent — which is how the gap became visible. `clawback_items` now attributes a negative adjustment matching **no** nested payment to the refunded order the ledger itself marks `REFUNDED`, when exactly one such order of that amount was paid out in an earlier batch. Exception recall 0.8753 ± 0.0689 → **0.9818 ± 0.0250**, in *both* arms — it is independent of the pass. |
| 55 | Wilson intervals belong on precision | **On every headline proportion, or the argument is not being applied to itself** | The project argues at length that a point estimate without an interval is not a measurement, then applied that rule to precision alone. The metric with the smallest denominator in the whole report — exception recall, n = 30 — was the one printed to four significant figures with nothing beside it. `Proportion` now carries successes, trials and the interval in one object so a renderer that shows the estimate cannot drop the rest, and the report's verdict column reads the interval **one-sided**: *met* when the lower bound clears the target, *missed* when the upper bound is below it, *undecided* when it straddles. That cuts both ways — 30 of 30 exceptions does not demonstrate ≥ 0.95 from thirty records, and it is reported as undecided rather than as a pass. |
| 56 | The calibration disclosure is spread across four places | **The conclusion is stated in one paragraph, derived from the fitted bundle at render time** | Every component was disclosed — *single class* in one section, *one populated bin* in another, `tau_high_is_fitted` in the config table, the decision breakdown further down — and a reader had to assemble a conclusion the document never drew: **on this corpus the calibration layer cannot change a single decision, and removing it would change no published number**. A criticism a judge assembles themselves lands harder than one the author already made. `_calibration_verdict` reads the coefficients, the isotonic blocks, the fitted threshold and the run's own probability distribution and writes the paragraph from them, so it cannot survive the fit ceasing to be degenerate — which is exactly when it should stop being printed. |
| 57 | The provider ladder is a stretch item | **Built, and the failure it exists for is a 429 mid-demo** | `FailoverProvider` walks Groq → Gemini → OpenRouter → Ollama behind the same one-method protocol, so the cache, budget, validation, gates and ledger sit unchanged. A rate limit is retried once in place (honouring `Retry-After`, capped) and then drops a rung; an outage drops immediately, because waiting half a second does not fix a misconfigured endpoint; a schema violation is **not** a rung change, because it is a property of the answer rather than of the backend. `FailoverProvider.name` is the *ladder's* identity and not the rung that answered — it is the cache key, and a name that changed when a 429 pushed one call down a rung would turn a transient limit into permanent extra cost. `CostLedger.provider_used` reports the rung. A keyless machine builds no ladder at all, and Ollama — the one rung needing no credential — joins only when it is asked for, so no run waits on a localhost timeout before reaching the deterministic path. |
| 58 | Never having run a model is the end of the LLM story | **The path is measurable without one, and the measurement includes a control** | `ledgerloop llm-report` runs the corpus with the model and again with `--no-llm`, and writes both scores beside the calls, cache hits, tokens, latency, failures, budget refusals, actual and equivalent-paid cost, references the grounding gate refused and proposals `verify_arithmetic` demoted. The control is the point: *the LLM proposes, deterministic code decides* is a claim about **authority**, and the way to check authority is to take the model away and see whether the answer moves. With `--offline-provider` — a documented rule that reads the prompt and nothing else — every machinery column is still measured on the real code path; the artefact records `live: false` and the report banners it. Measured here: 16 calls, 20,087 tokens, 35 narration repairs accepted, 4 proposals returned and **0 accepted with 2 demoted on arithmetic**, 74 explanations rewritten, and four headline figures identical with and without. |
| 59 | `EVALUATION.md` and `ARCHITECTURE.md` are working files, so they are gitignored | **A deliverable a fresh clone cannot see has not been delivered** | The reasoning for ignoring them was sound about the *working* copy — a committed report is one that can be quietly corrected — and it did not follow that the repository should ship with no metrics document and no design record at all. A judge who clones and does not run `make eval` was seeing neither. Both are now tracked as snapshots; `make eval` still regenerates the report in place, two runs differ only in one labelled timings block, and two tests enforce that — so the committed copy is checkable rather than authoritative. `.local/`, `reports/`, `data/generated/` and `.env` remain ignored. |
| 60 | A split payout whose tranches lost their reference is somebody else's problem | **It was nobody's, and that is where every remaining missing link was** | `run_tier2` finds tranches through the settlement's key, so when A09 composes with A07 and every tranche's narration loses its UTR, `credit_bucket` returns nothing and T2 never sees the batch. T3 cannot help: it tests a candidate against the **whole net**, and a tranche never is. `run_tier2`'s own docstring had named the gap and assigned it to "the tiers that can match an unreferenced row" — no tier could. Phase 2.5 gives T2's arithmetic a candidate set T3's merchant master supplies: unclaimed, unreferenced credits naming the batch's merchant, then a subset-sum over the **credit** amounts. The partition is `_solve`, T2's own, called unchanged — this is candidate generation, not a second reconciliation mechanism. 42 settlements found across 29 corpora, 38 resolved, 398 links gained, **0 false positives**. |
| 61 | The tranche search should use the same tolerance the tier does | **Exact, and the tolerance stays where rounding actually accumulates** | The target window for the credit subset-sum is `[net, net]` with no band at all. A split payout conserves money by construction — the tranches *are* the payout — so an epsilon here would not absorb drift, it would admit sets that are merely close, and "close" over a pool of similar amounts is exactly where a false positive would come from. T2's `aggregation_epsilon_minor` still governs the **partition** below, where per-payment rounding genuinely accumulates across a subset. Two further refusals carry the rest: at least two credits (one equal to the net is a one-to-one match and belongs to T0/T1/T3), and `SubsetSearch.is_unique`, which is `exhaustive and len(solutions) == 1` — a greedy fallback can find a set but never prove it alone. Measured: the pass is **insensitive** to the merchant gate and the date window (7 days and 90 give identical results; removing both entirely still yields zero wrong sets), so the report credits the arithmetic rather than the filters. |
| 62 | `payment_bucket` excludes a charged-back payment, so the arithmetic already accounts for it | **It excluded the payment from the *subset* and left it in the *denominator*** | A pre-existing T2 defect, found by probing why a correctly-identified tranche set still refused. `payment_bucket` drops a payment a negative adjustment identifies as charged back — its money never reached the bank, so it belongs to no tranche — but `_solve` allocated the net across the gross of **every** nested payment. Each tranche came out short by its share of money that was never paid: on `test` seed 42's SETL-0004 the larger tranche was predicted at ₹1,83,195.85 against an actual ₹2,02,486.64, out by ₹19,290.79 and far outside any epsilon. Allocating over the bucket makes the prediction exact, delta 0, on every tranche of every case. **Not switchable**: a config flag for a bug is an invitation to run the buggy arm. Its effect is measured against the previous commit instead — seed 42 unaffected, seeds 43–46 gain, full-ladder recall 0.8049 ± 0.0792 → 0.8396 ± 0.0720 and the `T0-T2` ablation row 0.6625 ± 0.0565 → 0.6972 ± 0.0700, zero false positives throughout. Published numbers changing, reported per seed rather than left to drift. |
| 63 | A new pass should be a new tier | **It is T2, because `subset_members` is a type-level invariant of T2** | `MatchCandidate` refuses `subset_members` on any tier but `T2_AGGREGATION`, and this pass produces them — so the tier tag is settled by the type system rather than by preference, and every figure in its candidates is produced by T2's own code. It is gated on tiers 2 **and** 3 both being enabled, because it needs T3's merchant master for the pool and T2's solver for the partition. That gate is also what leaves the published `T0-T2` ablation row untouched *by the pass*: with T3 off there is no pool, so the contribution lands on `T0-T3`, which is the honest place for a result needing both. `TestPhase25Defaults.test_the_t0_t2_ablation_row_is_untouched` checks that rather than trusting it. It runs **after** the residual loop converges rather than inside it, and that ordering is measured, not assumed: from inside the loop seed 42 resolved 1 of 3, because credits T2 and T3 had yet to claim were still in the pool and the uniqueness test refused batches it could resolve once the pool settled. |
| 64 | T3's margin makes its matches unique | **It made them unique in one direction, and the other one only opens up at scale** | Gate 3 asks whether *one settlement* has two credits the scorer cannot separate. The mirror-image question — do two **settlements** have a claim on this credit? — was never asked, and a settlement-ordered loop silently gave the credit to whichever it reached first. It cannot arise on the committed corpora and is close to certain on a large one: a merchant's payouts sit lakhs apart on a base of crores, so two of its settlements land inside each other's tolerance band once it has enough of them, the date window admits both, and the name is identical because it *is* the same merchant. Zero contested credits across all 29 committed corpora; one at 2,500 orders and nine at 5,000. The credit side now carries the same test against the same `min_margin` — no new constant — and every claimant refuses a contested credit at the uniform prior the module already documents. Two same-merchant settlements always score identically, so such a contest can never be resolved on a name, which is why refusing is the only honest answer available to a *lexical* tier. |
| 65 | A settlement still open at T3 is a settlement T3 may match | **Not if the bank has already written that settlement's UTR on a credit** | The rule that a *credit* carrying a reference is already explained runs the other way too, and only one direction was implemented. A09 composed with A07 on **one** tranche makes it concrete: the payout goes out in two tranches, one keeps the settlement's UTR and one loses it, so T2 correctly will not close it — the keyed half does not sum to the net — and the batch is still open when T3 arrives. T3 compares a **whole** net against single credits, and on a large statement some other settlement's payout lands inside that band: at 5,000 orders two do, 0.27% and 0.22% away from a net several crore wide, inside the date window, spelt with the same merchant name. Every gate passes and the tier auto-matches at `p = 1.0` with the arithmetic verifying, because the amounts genuinely agree. **22 wrong links**, and none of them reachable by tightening a threshold — the evidence T3 reads really does point that way. What rules them out is evidence it was not consulting. Finding the missing tranche of a partly referenced payout is T2's arithmetic, where the sum has to close; it is not something a name and an amount band should be allowed to guess at. |
| 66 | The `scale` split is a throughput question | **Throughput was the smaller half of the answer** | The item as written was *generate 5,000 orders, record throughput*. Running it found the two defects above, which is the argument for the run: every published precision figure comes from 60 to 400 orders, and uniqueness arguments that hold there can be size-dependent without anything saying so. `ledgerloop scale` therefore reports precision, recall, match rate and false positives at each size **before** the timings, and walks a curve rather than a point — one figure at 5,000 cannot distinguish *fast* from *fast so far*. Quality and timing live in separate fields of `ScalePoint` and the artefact records the machine, so a throughput number can never be read as a property of the system; that is decision 50's rule applied to a new artefact. Measured: precision **1.0000** and zero false positives at every size, 12,233 records in 7.4 s, and growth of roughly `O(n^1.5)` — recorded rather than optimised, because nothing in the evaluation is bounded by it. |
| 67 | T4 contributes zero because the corpus does not happen to exercise it | **Because the pipeline's own design cannot produce the state it needs, which is a stronger claim and is now measured** | Decision 31 said the right thing and had measured only the outcome. Phase 2.7 instrumented T4's premise set on every corpus on disk: **1228 settlements arrive fully linked and 242 arrive not linked at all — zero arrive partial**, and at 5,000 orders it is 762 against 142, again zero. Path closure needs `S -> C` with payments outstanding; sibling completion needs some payments linked and others not. Every tier that asserts the settlement edge asserts the payment edges in the same breath, so neither premise can form. The tier is **unexercised, not broken**: the same code completes a partial state correctly when one is constructed, and reproduces T0's own links and rupee shares when asked to deduce them back. Two tests pin the invariant rather than the counts, so they hold on any corpus and fail the moment a tier leaves a batch half-assigned — which would be good news, because that is the situation T4 was built for. **Nothing in T4 was changed.** Loosening a rule until it fired would have traded precision for the appearance of contribution, and the instruction to do so is the one this project exists to refuse. |
| 68 | The dashboard may compute a verdict, since it is only presentation | **It renders a ruling and never reaches one** | *met / missed / undecided* is a real judgement about a sample, and it was stated once in `eval/report.py` as a private function. A dashboard needing the same judgement had two options: restate the rule, or share it. Restating it is how two surfaces come to disagree about whether a target was hit -- the exact failure decision 59 avoids for metrics, applied to a conclusion instead of a number. `Proportion.verdict` now holds the rule beside the type that already refuses to exist without its interval, `METRIC_TARGETS` holds the floors as data, and both the report and the dashboard call them. The prose differs; the ruling cannot. On `test` seed 42 that makes the dashboard print *undecided* over a precision of 100.00%, because [98.66%, 100.00%] does not clear >= 99% -- a green tick there would have been the flattering error and the same error. |
| 69 | A run record only needs the numbers the report prints | **It needs the intervals too, or the dashboard has to recompute them** | The store wrote `precision_ci_low` and `precision_ci_high` as loose floats and nothing for the other three proportions. A dashboard showing four intervals therefore had exactly two options, and one of them was to derive Wilson bounds in the view -- which is the one thing the UI is not allowed to do, and would have put a second implementation of the project's own statistics in its least-tested module. `run.json` now carries each headline `Proportion` **whole**: successes, trials, value and both bounds. That keeps the type's guarantee across the file boundary -- a reader of the record can re-derive every interval rather than taking the bounds on trust -- and a record written before this renders as *not measured* rather than as 0.00%, which is the same rule the report applies to a component that did not run. |
| 70 | The offline stand-in measures the LLM path well enough | **It measures the machinery and cannot produce the failures the gates exist for** | The first live run, Gemini `gemini-3.6-flash` on 2026-08-30, produced three things no stand-in could: **six failures in fifteen calls** (three read timeouts, three HTTP 503s), **149.8 s of provider time against the stand-in's 15 ms**, and **nine outputs refused by the grounding gate against the stand-in's zero**. That last one had been predicted in writing -- `EVALUATION.md` argued that *a zero from a reasoner incapable of the failure is not evidence that the failure does not happen* -- and a real model duly cited records absent from its evidence pack. Every failure was absorbed per batch and every ungrounded output refused, so **the model moved no published metric**: precision, recall, match rate and exception recall are identical with and without it, to six decimal places, against its own `--no-llm` control in the same invocation. That is *the LLM proposes, deterministic code decides* measured rather than asserted. The live figures are **one measured run and are not reproducible**; every deterministic number still reproduces with `--no-llm` and no key. |
| 71 | A provider model id can be left to track the vendor's latest | **Pinned to an explicit version, and the pin is what caught the retirement** | Gemini publishes `gemini-flash-latest`; using it would keep the ladder working forever and would silently repoint at a different model between two runs of `make eval`. That is the one failure this project cannot detect, because the numbers would simply differ and nothing would say why. A pin fails loudly instead, and it did: the first live call returned `HTTP 404 -- This model models/gemini-2.0-flash is no longer available`, with the endpoint naming its own successor. One line changed. The credential was confirmed valid first (`GET /v1beta/models` returned 200) so the 404 was diagnosed rather than guessed at, and both candidate successors were smoke-tested before one was chosen. A test now asserts no rung is pinned to a moving alias. |
| 72 | Recall falls at scale because bigger corpora are harder | **More than half of it was one lucky seed, and the rest is corpus shape rather than difficulty** | The curve compared seed 42's 300-order run against seed 42's 5,000-order run and reported a fall of 0.9628 -> 0.8532. Across five seeds, recall at 300 orders spans **0.7407 to 0.9796**: 0.9628 was near the top of its own spread. Corrected, the fall is 0.8971 -> 0.8464, and the standard deviation collapses **0.0980 -> 0.0188** as n grows -- 300 orders is ~27 settlements, so one dead settlement moves recall by 3.7 points. The anomaly mix is flat across sizes (A01_CLEAN holds 0.9285 -> 0.9210 of records), so the corpus is **not** getting proportionally harder. What does change is density: `baseline.py` caps merchants at twelve, so settlements per merchant rise **2.2 -> 37.7**, and same-merchant amount collisions rise with it. That is a property of the generator; the matcher was not changed to compensate for it. **Recall is a settlement resolution rate**: at every size, 100% of missing links belong to settlements that are *entirely* unresolved, and there is not one partially-resolved settlement anywhere -- the same all-or-nothing structure decision 67 found on the T4 side. |
| 73 | A settlement the ladder refused is finished with | **It still speaks for its credit, and forgetting that cost 10 wrong links** | Phase 2.6 built T3's contention test over `open_settlements()`, tying two different rights to one condition: the right to **claim** a credit (evidence that it is spoken for) and the right to be **assigned** one (an act). A settlement refused by an earlier tier is consumed, so it could not claim -- and contention then saw a single claimant where there were two. At 5,000 orders on seed 45, `SETL-0231` was refused by T0 as contested; `BNK-00231` equalled its net **to the paise**; T3 gave that credit to a settlement 0.03% away at `p = 1.0`. The claim map now covers **open or refused** settlements while assignment stays restricted to open ones. A **resolved** settlement claims nothing: its money is accounted for, and letting it claim would manufacture refusals out of nothing -- the blunt version that did so cost 35 true positives for the same precision gain, and the precise one costs **zero**. Claims can only be added, never removed, so the tier can only decline more. |
| 74 | One seed is enough for a benchmark nobody publishes a mean from | **It printed a false claim for months, and the claim was about precision** | `ledgerloop scale` reported *precision held at every size* because it only ever ran seed 42. Seed 45 carries **17 false positives at 5,000 orders**, `p = 1.0000`, Rs 4,51,272.72. PLAN.md 9.4 requires five seeds of every other published figure in this project and the scale curve had been exempt from its own rule. It now runs `DEFAULT_SCALE_SEEDS = (42, 43, 44, 45, 46)` -- the same five -- reports mean +/- sd, names the seed of any false positive, and exits non-zero while one stands. It currently exits non-zero: seven of the seventeen survive, and a gate that went green over a known defect would be worth less than no gate. |
| 75 | T3 should use the same tolerance band the keyed tiers use | **Without a reference the amount *is* the identity claim, so it must be exact -- decision 61, applied to the tier that had not been applying it** | T0 and T1 match on a reference and *then* check the money: their band absorbs fee rounding around an identity the reference already proved. T3 has no reference. The amount is not a check on the match, it **is** the match, alongside a merchant name every batch of that merchant shares -- and an approximate identity claim over a pool of same-merchant amounts is exactly where a false positive comes from. That is decision 61's argument for `find_tranche_set` verbatim, and T3 matches on money alone for the same reason. Measured over 49 corpora before anything was written: **delta == 0 gives 543 correct and 0 wrong; delta != 0 gives 0 correct and 1 wrong.** Every legitimate whole-net match in the corpus family is exact to the paise, and the band was admitting exactly one thing -- the last open false positive, `SETL-0015` taking a *tranche* of `SETL-0018`'s split payout 0.059% away. **A band was removed, not a threshold added**: there is no new constant, and none that could be tuned to trade precision for recall here. Exactness governs the **assignment**; the band still governs the **pool**, so a near-miss rival can still contest a credit -- Phase 2.9's lesson that a claim is evidence. Cost: **zero**. Precision 1.0000 on every seed at every size, FP cost Rs 0, and recall at 5,000 orders moved 0.8464 -> 0.8467, because the credit that had been taken wrongly was freed. Neither barred fix was reintroduced: T3 still does not know what a tranche is, it simply declines to assert an identity it cannot establish. **Honest limit at the time:** the invariant was empirical on one generator. Decision 76 tested it against a second. |
| 76 | The exactness rule can be validated by re-running the generator that produced it | **It cannot -- that generator forbids the case by construction, so a second construction model was built, and it changed what is known** | Model A defines the bank credit as the settlement's declared net. A02 `ROUNDING_DRIFT` is the only class that moves a whole-net credit off it (A05 and A09 change the credit *count*; A03, A06 and A08 re-derive the credit *from* the net and are exact), and A02 cannot land on the same settlement as A07 `MISSING_REFERENCE` -- both take `ASPECT_PRIMARY`, because a `GroundTruthRecord` carries one anomaly class per record. So "unreferenced" and "off the net" are mutually exclusive there for a **truth-representation** reason, not a financial one. Measured over the same 49 corpora: referenced credits are 1720 exact / 818 inexact (**32% of them drift**), unreferenced credits 654 exact / **0** inexact. `generator/adversarial.py` is model B: the bank decides what lands, a credit is `declared_net - deduction` for a deduction the PSP file never sees, and the deduction is **independent of the reference**. Fifteen case shapes per merchant, truth still from `covered_payment_ids` and the effect list, no new `AnomalyClass`. **Result, five seeds, matcher unchanged.** Legitimate reference-free non-exact matches **do exist** and the rule refuses every one: recall 0.2764 +/- 0.0085 against 0.4809 +/- 0.0176 for the pre-2.10 band. **The band is still the worse arm on the same corpus** -- precision 0.8469 +/- 0.0131 -> 0.7729 +/- 0.0173, false positives 226 -> 639, FP cost roughly doubled -- and 36 of the 78 it adds are the Phase 2.10 defect rebuilt: an unreferenced settlement taking a *tranche* of a sibling's split payout 6 bps away, refused by exactness in every corpus and taken by the band in every corpus. **Exactness is necessary, not sufficient**: the 43 false positives per corpus that remain are all one shape -- an orphan credit carrying the merchant's name and the net *to the paise*, which no amount rule can refuse without refusing every true T3 match. **A candidate alternative exists and is not shipped.** A deduction the merchant's own *referenced* credits already attest to is observable without ground truth and would recover the flat-charge half with no new false positive here -- but it rests on one witness per merchant, model B contains no case built to attack it (the rule was found *after* the corpus), and it is a new mechanism, which by standing rule goes behind `ledgerloop comparison` with both arms published rather than into the precision-critical path on one corpus's word. **No production code changed.** Decision 75 stands, with its cost now measured rather than assumed. Real-data validation remains outstanding. |

---

## 6a. The calibration path (Step 7)

```
tier proposes candidate
        │
        ├── T0 / T1 ──────────────────────────────► p = 1/n, set by the tier
        │                                            (Tier.is_deterministic_certain)
        ├── refusal (arithmetic_verified = False) ─► p = 1/n, set by the tier
        │
        └── T2..T5, verified ──► logistic (train) ──► isotonic (calibration) ──► p
                                                                  │
                                                          τ_high (calibration)
                                                                  │
                                            p ≥ τ_high → AUTO_MATCHED
                                            τ_low < p < τ_high → NEEDS_REVIEW
                                            p ≤ τ_low → EXCEPTION
```

**Three splits, three jobs, enforced in the type.** `train` fits the logistic
coefficients. `calibration` fits the isotonic map and selects `τ_high`. `test`
is measured and never fitted on. `CalibrationProvenance` will not construct if
the halves share a corpus or if either names `test`.

**What the blender may not do.** It re-scores assertions and never refusals
(decision 34); it never proposes a link a tier did not, so it cannot raise
recall; and it abstains on a tier it was never fitted for rather than scoring it
as the reference level.

**The fitted threshold lives on the `RunConfig`.** `configure_for()` returns a
copy carrying it, so a fitted `τ_high` is inside `config_hash` rather than
beside it — a run under a fitted threshold must not hash identically to a run
under the placeholder default. `tau_high_is_fitted` records which it was.

**The measured result on this corpus.** The accepted population is single-class,
so the logistic has no contrast to learn from and stops when its Hessian
degenerates; the isotonic collapses to one block at 1.0; `τ_high` is fitted at
1.000 attaining precision 1.0000 (95% CI [0.9733, 1.0000]) on 140 calibration
links. Precision and recall on `test` are unchanged at 1.0000 and 0.4422. That
is the honest reading: the deterministic tiers' refusal discipline is what makes
the auto-matches correct, and the calibrator confirms it rather than improving
it. The machinery is what T5 will need in Step 9, where an LLM proposes links
that *are* sometimes wrong.

---

## 6b. The exception queue and the LLM boundary (Steps 8-9)

```
                          three source documents
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                      ▼
  T0..T4 ladder  ──►  decisions  ──►  Step 8 classifier  ──►  queue
        │                                   ▲                   │
        │                       (sources + decisions only;      │
        ▼                        never ground truth, never      ▼
   T5 (injected)                 a model)              bounded resolution
        ▲                                               (proposes, never posts)
        │
   LLM ─┴─ schema → budget → grounding → verify_arithmetic → decision policy
```

**Nothing an LLM says is authoritative.** It may read unstructured text, propose a
hypothesis, and write prose. It may never decide a match, do arithmetic, determine a
metric, classify an exception, set a severity, price an impact, or bypass a gate.
`verify_arithmetic` takes no argument saying where a proposal came from.

**The whole system runs with `--no-llm`**, and the report it produces is
byte-identical to a run with no key — timings aside, which the report already
declares as the only non-deterministic figures.

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
| Exception impact is a non-negative magnitude | `ReconException` validator |
| A proposed link names both endpoints | `ResidualHypothesis` validator |
| LLM output cannot carry an amount, a class or a severity | the `llm.contracts` schemas |
| `subset_members` only on T2, and `subset_size` agrees | `MatchCandidate` validator |
| Anomaly prevalence sums to exactly 1.0 | `GeneratorConfig` validator |
| T5 cannot be enabled with `llm.enabled=False` | `RunConfig` validator |
| Decisions and audit events are immutable | `FrozenLedgerModel` |
| Unexpected keys are rejected (LLM output validation) | `extra="forbid"` |
| The logistic and the isotonic are fitted on different corpora | `CalibrationProvenance` validator |
| The `test` split is never fitted on | `CalibrationProvenance` validator |
| An isotonic map is monotone and its values are probabilities | `IsotonicCalibrator` validator |
| A blender's coefficients agree with its feature names | `LogisticBlender` validator |

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
