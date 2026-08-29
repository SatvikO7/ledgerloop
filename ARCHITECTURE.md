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

Decisions 1-51 are Steps 0-13. **52-59 are Phase 2**, and each of them is a
change the final audit asked for or a finding that came out of measuring one.

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
