# PLAN.md — LedgerLoop

**A three-way, confidence-calibrated reconciliation agent with an honest exception list.**

Target: Razorpay AI Buildathon — **Track 04, AI Finance Controller**
Challenge: *"Build an agent that closes one finance-ops loop across a 50+ record batch of synthetic data, reporting its match rate and the exceptions it could not resolve."*

Build mode: solo developer, ~3 weeks part-time, **₹0 budget — every component free-tier or self-hosted** (see §10.1).
Priority order: **working > measurable > impressive > complete.**

---

## Table of Contents

1. [Problem & Chosen Finance-Ops Loop](#1-problem--chosen-finance-ops-loop)
2. [What Makes This Different](#2-what-makes-this-different)
3. [System Architecture](#3-system-architecture)
4. [Agent Workflow](#4-agent-workflow)
5. [Synthetic Data Design & Ground Truth](#5-synthetic-data-design--ground-truth)
6. [Matching & Reconciliation Logic](#6-matching--reconciliation-logic)
7. [Agent, Tools & LLM Responsibilities](#7-agent-tools--llm-responsibilities)
8. [Exception Handling](#8-exception-handling)
9. [Accuracy & Evaluation Methodology](#9-accuracy--evaluation-methodology)
10. [Tech Stack & Technical Decisions](#10-tech-stack--technical-decisions)
11. [Project Structure](#11-project-structure)
12. [Implementation Phases & Milestones](#12-implementation-phases--milestones)
13. [Testing Strategy](#13-testing-strategy)
14. [UI & Demo](#14-ui--demo)
15. [Deployment](#15-deployment)
16. [Final Deliverables](#16-final-deliverables)
17. [Definition of Done](#17-definition-of-done)
18. [Risks & Cut Lines](#18-risks--cut-lines)

---

## 1. Problem & Chosen Finance-Ops Loop

### 1.1 The loop

**Three-way payment settlement reconciliation.**

Money moves through three systems that never agree perfectly:

```
Internal Ledger          PSP Settlement            Bank Account
(what we sold)     →     (what the PSP paid)  →    (what actually landed)
  orders                   payouts/batches           credit entries
```

A finance controller must answer, every single day:

> For every rupee we booked as revenue, did it actually arrive in the bank — and if not, **why not**, and **how much** is at stake?

### 1.2 Why this loop and not something easier

Most hackathon submissions will do **two-way, one-to-one** reconciliation: one CSV of invoices against one CSV of bank rows, joined on a reference number. That is a `pandas.merge` with a fuzzy fallback, dressed up as an agent.

This project deliberately takes the hard version:

| Dimension | Easy version (what most will build) | This project |
|---|---|---|
| Sources | 2 | **3**, heterogeneous formats |
| Cardinality | 1:1 | **N:1 and N:M** (many payments → one payout → possibly split bank credits) |
| Reference keys | Clean IDs present | UTRs missing, mangled, or buried in free-text narration |
| Amounts | Equal | Gross ≠ net (fees, tax, refunds, chargebacks netted off) |
| Output | matched / unmatched | **Calibrated confidence + typed root cause + ₹ impact + suggested action** |

The N:1 aggregation problem is the crux. A settlement batch of 14 payments arrives as **one** bank credit for a net amount after fees, tax, one refund and one chargeback. Nothing joins. You have to *solve* for the subset.

### 1.3 Scope boundary (what this is NOT)

Explicitly out of scope, and stated as such in the README so judges know it was a choice, not an oversight:

- No live Razorpay API calls (synthetic data only — the challenge asks for synthetic)
- No writes to any real financial system; the agent **proposes** journal adjustments, never posts them
- No multi-entity / multi-GAAP consolidation
- No tax filing logic beyond flagging a tax-line mismatch

---

## 2. What Makes This Different

These are the eight levers that separate this from a generic reconciliation bot. Every one of them is cheap to build and expensive to fake.

**D1 — Three-way, not two-way.** Ledger ↔ PSP ↔ Bank, with the PSP layer in the middle where fees and netting happen. Nobody accidentally builds this.

**D2 — N:1 aggregation solver.** A constrained subset-sum that finds *which* payments compose a bank credit. This is real algorithmic work, not prompt engineering.

**D3 — Calibrated confidence, not binary matched/unmatched.** Every match carries a probability that has been fit on a calibration split and **validated with a reliability diagram and ECE**. When the system says 90% confident, it is right ~90% of the time. Almost no hackathon project measures calibration.

**D4 — Typed, self-explaining exceptions.** Each unresolved item gets a class from a 12-way taxonomy, a plain-English root-cause hypothesis, the evidence chain that led there, a ₹ impact, and a suggested next action. "Unmatched" is not an answer; "₹4,312 short because a chargeback was netted off payout SETL-0091, evidence: [3 links]" is.

**D5 — Deterministic-first, LLM-last cost discipline.** Tiers 0–4 are deterministic/statistical. The LLM only sees the residual (~10–15% of records). Ablation table proves the LLM adds measurable lift *and* that an LLM-only baseline is both worse and ~40× more expensive. This directly answers "did you actually need AI here?" — the question that kills weak submissions.

**D6 — Adversarial data generator.** The synthetic data is not clean-plus-noise. It is generated with 12 named, parameterised anomaly classes at controlled prevalence, seeded and reproducible, with a difficulty dial. Judges can regenerate a harder dataset and rerun.

**D7 — Full audit trail with replay.** Every decision — tier fired, score, evidence, LLM prompt hash, tokens, latency — is persisted. The UI can step through any single reconciliation decision like a debugger. The track brief asks for an audit trail; this is one you can actually inspect.

**D8 — Honest metrics, including the ones that hurt.** Reported: auto-match precision, false-positive cost in ₹, exception-class confusion matrix, calibration error, per-anomaly-class recall (including the classes where recall is bad), throughput and cost. The bar says *"one cherry-picked match proves nothing"* — this is the direct answer.

---

## 3. System Architecture

### 3.1 Component diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                          React UI (Vite)                          │
│   Run Dashboard │ Exception Queue │ Match Explorer │ Audit Replay  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ REST / SSE
┌───────────────────────────▼──────────────────────────────────────┐
│                        FastAPI Gateway                            │
│   POST /runs  ·  GET /runs/{id}  ·  GET /exceptions  ·  /audit    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                   LangGraph Orchestrator (agent)                  │
│                                                                   │
│  ingest → normalize → build_graph → tier_ladder → adjudicate      │
│         → calibrate → classify_exceptions → report                │
└──┬───────────┬───────────┬────────────┬──────────────┬───────────┘
   │           │           │            │              │
┌──▼──────┐ ┌──▼───────┐ ┌─▼────────┐ ┌─▼──────────┐ ┌─▼──────────┐
│Postgres │ │  Neo4j   │ │ ChromaDB │ │ LLM Client │ │ Audit Log  │
│ledger,  │ │ entity   │ │ narration│ │ (Claude /  │ │ JSONL +    │
│runs,    │ │ graph &  │ │ & name   │ │  OpenAI)   │ │ Postgres   │
│results  │ │ lineage  │ │ embeds   │ │ structured │ │            │
└─────────┘ └──────────┘ └──────────┘ └────────────┘ └────────────┘
```

### 3.2 Design principles

1. **Deterministic core, probabilistic edge.** The matching engine is a pure function of its inputs; the LLM is an injectable dependency behind an interface. This makes the system testable without API calls.
2. **Every match is an object, not a row.** A `MatchCandidate` carries source refs, tier, raw score, calibrated probability, evidence list, and decision. Nothing is lost between stages.
3. **Append-only audit.** Decisions are never mutated; a revision writes a new record. Replay is just reading the log in order.
4. **Swappable stores.** Neo4j and Chroma sit behind thin repository interfaces with in-memory implementations (NetworkX, numpy) so tests run with zero infra and the demo degrades gracefully if a container dies.

### 3.3 Why a graph database earns its place

This is not graph-for-the-résumé. The reconciliation chain is genuinely a path problem:

```
(:Order)-[:PAID_BY]->(:Payment)-[:SETTLED_IN]->(:Settlement)-[:CREDITED_AS]->(:BankTxn)
```

When one edge is missing, you infer it by **traversal**: if 12 of 14 payments in a batch link cleanly to a bank credit, the remaining 2 are constrained to that same credit. Cypher expresses this in a few lines; SQL self-joins do not. Ring detection (same customer, many refunds, across merchants) is also a native graph query.

**Fallback:** an in-memory NetworkX implementation behind the same interface, so a Neo4j outage during the demo does not kill the run.

---

## 4. Agent Workflow

### 4.1 LangGraph state machine

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
              ┌──────────────────────┐
              │  ingest_sources      │  parse 3 formats → raw records
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  normalize_records   │  canonical schema, currency→paise
              └──────────┬───────────┘   (LLM tool: parse_narration)
                         ▼
              ┌──────────────────────┐
              │  build_entity_graph  │  Neo4j nodes + known edges
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  tier_ladder         │◄──┐  T0→T1→T2→T3→T4
              │  (deterministic)     │   │  loop until no new matches
              └──────────┬───────────┘───┘
                         ▼
                   ┌───────────┐
              ┌────┤ residual? ├────┐
              │ no └───────────┘ yes│
              │                     ▼
              │          ┌──────────────────────┐
              │          │  llm_adjudicate (T5) │  batched, structured out
              │          └──────────┬───────────┘
              │                     │
              ▼                     ▼
        ┌────────────────────────────────┐
        │  calibrate_confidence          │  raw scores → probabilities
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  apply_decision_policy         │  auto / review / exception
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  classify_exceptions           │  taxonomy + root cause + ₹
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  generate_report               │  metrics, exception list, trail
        └──────────────┬─────────────────┘
                       ▼
                    ┌──────┐
                    │ END  │
                    └──────┘
```

### 4.2 Shared state object

```python
ReconState:
    run_id: str
    config: RunConfig              # thresholds, tolerances, seed, LLM on/off
    raw: dict[SourceName, list[RawRecord]]
    normalized: list[CanonicalRecord]
    graph_handle: GraphRef
    candidates: list[MatchCandidate]
    decisions: list[MatchDecision]
    exceptions: list[Exception]
    metrics: RunMetrics
    audit: list[AuditEvent]        # append-only
    cost: CostLedger               # tokens, ₹, latency per node
```

Every node takes state, returns state. No hidden globals. This makes the whole pipeline a testable pure-ish function.

### 4.3 Why LangGraph over a plain function chain

Honest answer for the interview, and worth writing in the README: for the happy path, a function chain would do. LangGraph earns its place because of the **loop** (`tier_ladder` re-runs when a late resolution unlocks new matches — resolving one bank credit constrains the remaining ones), the **conditional branch** on residual, and because the state machine gives free, structured checkpointing for the audit replay feature. Don't oversell it beyond that.

---

## 5. Synthetic Data Design & Ground Truth

### 5.1 The three sources

Deliberately heterogeneous — different formats, different field names, different granularity.

**Source A — Internal Ledger (`ledger_orders.csv`)**
Clean, structured, our own system of record.

| field | example | notes |
|---|---|---|
| `order_id` | `ORD-2026-004821` | primary key |
| `merchant_id` | `MRCH_0007` | |
| `customer_ref` | `CUST_11902` | |
| `amount_gross_paise` | `499900` | integers only, no floats |
| `currency` | `INR` | ~5% `USD` |
| `booked_at` | `2026-03-04T11:22:09+05:30` | |
| `status` | `CAPTURED` | also `REFUNDED`, `PARTIAL_REFUND` |

**Source B — PSP Settlement Report (`psp_settlements.json`)**
Nested JSON, one object per payout batch containing many payments. Fees and tax live here.

```json
{
  "settlement_id": "SETL-0091",
  "utr": "UTR2026030412345",
  "settled_on": "2026-03-06",
  "gross_paise": 4210900,
  "fee_paise": 84218,
  "tax_paise": 15159,
  "adjustments_paise": -431200,
  "net_paise": 3680323,
  "payments": [
    {"payment_id": "PAY-88301", "order_ref": "ORD-2026-004821",
     "amount_paise": 499900, "captured_at": "2026-03-04T11:22:11+05:30"}
  ]
}
```

Note `order_ref` is **sometimes null, sometimes malformed** (`ord 2026 004821`, `ORD‑2026‑004821` with a non-ASCII hyphen). That is intentional.

**Source C — Bank Statement (`bank_statement.csv`)**
The messy one. Free-text narration, no structured reference field.

| field | example |
|---|---|
| `txn_id` | `BNK-77120` |
| `value_date` | `06/03/2026` (DD/MM/YYYY — deliberately ambiguous format) |
| `narration` | `NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT` |
| `credit_paise` | `3680323` |
| `debit_paise` | `0` |
| `balance_paise` | `19844210` |

Narration variants include: UTR present / UTR truncated / UTR absent with only a merchant-name variant (`RAZORPAY SOFTWARE`, `RZRPAY SOFTWARE PVT LTD`, `RAZORPAY SFTWR P L`), plus unrelated noise rows (rent, salary, vendor payments) that must **not** match anything.

> **Stretch (only if ahead of schedule):** emit Source C as a PDF as well, and parse it. Adds a genuine unstructured-extraction story. Cut it without hesitation if behind.

### 5.2 The 12 anomaly classes

Injected at controlled prevalence. Each carries an `anomaly_class` label in ground truth.

| # | Class | What it looks like | Default rate |
|---|---|---|---|
| A01 | `CLEAN` | everything reconciles | 65% |
| A02 | `ROUNDING_DRIFT` | ±1–3 paise from FX or fee rounding | 5% |
| A03 | `FEE_TAX_MISMATCH` | net ≠ gross − fee − tax | 4% |
| A04 | `TIMING_SHIFT` | bank credit lands T+2 instead of T+1, may cross month-end | 5% |
| A05 | `DUPLICATE_CREDIT` | same UTR credited twice | 2% |
| A06 | `POST_SETTLEMENT_REFUND` | refund issued after payout, netted into next batch | 4% |
| A07 | `MISSING_REFERENCE` | UTR absent from narration entirely | 4% |
| A08 | `CHARGEBACK_NETTED` | chargeback debited inside `adjustments_paise` | 3% |
| A09 | `SPLIT_PAYOUT` | one settlement arrives as 2 bank credits | 3% |
| A10 | `ORPHAN_BANK_CREDIT` | bank credit with no settlement (e.g. direct transfer) | 2% |
| A11 | `FX_MULTICURRENCY` | USD order settled in INR at an unstated rate | 2% |
| A12 | `LATE_ARRIVAL` | settlement record appears in the *next* batch file | 1% |

**Difficulty dial:** `--difficulty {easy,standard,hard}` scales anomaly prevalence (hard ≈ 50% anomalous) and narration noise. Report metrics on all three. Showing that performance degrades gracefully — and saying so — reads as far more credible than a single perfect number.

### 5.3 Ground truth

Generated *first*, data derived *from it*. Never inferred after the fact.

`ground_truth_links.csv`:

| `order_id` | `payment_id` | `settlement_id` | `bank_txn_id` | `expected_status` | `anomaly_class` | `impact_paise` |
|---|---|---|---|---|---|---|
| `ORD-...4821` | `PAY-88301` | `SETL-0091` | `BNK-77120` | `MATCHED` | `A01` | `0` |
| `ORD-...4822` | `PAY-88302` | `SETL-0091` | `` | `EXCEPTION` | `A08` | `431200` |

`expected_status` ∈ `MATCHED` | `EXCEPTION` | `UNMATCHABLE` (genuinely irreconcilable — the honest ceiling; a perfect system still cannot resolve these, and the report must say so).

### 5.4 Dataset sizes

| Set | Records | Purpose |
|---|---|---|
| `dev` | 60 orders | fast iteration, meets the 50+ bar |
| `calibration` | 200 orders | **fit calibration only — never evaluated on** |
| `test` | 300 orders | held-out headline metrics |
| `scale` | 5,000 orders | throughput benchmark only |

Fixed seeds per set, committed as a `data/seeds.json`. Anyone can regenerate byte-identical data.

---

## 6. Matching & Reconciliation Logic

### 6.1 The tier ladder

Each tier proposes `MatchCandidate`s with a raw score. Cheapest and most certain first. A record leaves the pool once decided.

| Tier | Name | Method | Typical yield |
|---|---|---|---|
| **T0** | Exact key | UTR / settlement_id / order_ref exact join, exact amount | ~60% |
| **T1** | Tolerance | amount within ±max(₹1, 0.5%), date within ±3 days | ~10% |
| **T2** | Aggregation | subset-sum: which payments compose this credit | ~12% |
| **T3** | Fuzzy + semantic | RapidFuzz on narration, embedding kNN on merchant names | ~6% |
| **T4** | Graph inference | constraint propagation over the entity graph | ~4% |
| **T5** | LLM adjudication | structured reasoning on the residual only | ~5% |
| — | Unresolved | → exception queue | ~3% |

### 6.2 T2, the aggregation solver (the algorithmic core)

**Problem:** given a bank credit of `net_paise = N`, find the subset of open payments `S` plus adjustments such that

```
sum(S.amount) - fees(S) - tax(S) + adjustments ≈ N   (within tolerance ε)
```

**Approach — pruned subset-sum, not brute force:**

1. **Bucket** candidate payments by settlement window (±3 days) and merchant. Typical bucket: 10–40 payments, not thousands.
2. **Anchor** on the declared `settlement_id` when present — this collapses the search to verifying one subset rather than searching all.
3. **Search** with meet-in-the-middle DP over paise-integers for buckets ≤ 30; greedy + local swap for larger buckets, capped at 200 ms per credit.
4. **Score** by `1 - |residual| / tolerance`, penalised by subset size (prefer parsimonious explanations).
5. **Uniqueness check:** if two different subsets both fit within ε, **do not match** — emit an `AMBIGUOUS_AGGREGATION` exception with both hypotheses. Silently picking one is exactly the dishonesty the track brief warns against.

All arithmetic in **integer paise**. No floats anywhere in the money path — this is a stated invariant, property-tested.

### 6.3 T3, fuzzy + semantic

- **Lexical:** RapidFuzz `token_set_ratio` on normalized narration vs merchant legal name; UTR regex with edit-distance-1 tolerance for OCR-ish corruption.
- **Semantic:** merchant name variants embedded into ChromaDB; kNN retrieval handles `RZRPAY SFTWR P L` → `Razorpay Software Private Limited` where lexical similarity is poor.
- Both scores feed the blend; neither alone decides.

### 6.4 T4, graph inference

Cypher-driven constraint propagation. Representative rules:

- **Sibling completion:** if ≥80% of a settlement's payments are matched to bank credit `B`, remaining payments are constrained to `B`.
- **Path closure:** `Order → Payment → Settlement` known and `Settlement → BankTxn` known ⇒ infer `Order → BankTxn`.
- **Exclusivity:** a `BankTxn` already fully consumed by a settlement cannot absorb more payments — prunes the T2 search space.
- **Ring detection:** `customer_ref` appearing in >N refund/chargeback events across >M merchants ⇒ flag `SUSPECTED_ABUSE_RING` (a bonus signal in the exception report, not a match decision).

### 6.5 Score blending & the decision policy

Each candidate carries a feature vector: `[tier_id, amount_delta_ratio, date_delta_days, lexical_score, semantic_score, graph_support, subset_size, llm_confidence]`.

A **logistic regression** (deliberately simple, inspectable, and fast) maps features → raw score. **Isotonic regression** fit on the calibration split maps raw score → calibrated probability `p`.

```
p ≥ τ_high  (default 0.95) → AUTO_MATCHED
τ_low < p < τ_high         → NEEDS_REVIEW   (surfaced in UI queue)
p ≤ τ_low   (default 0.60) → EXCEPTION
```

**Thresholds are not hand-picked.** `τ_high` is selected on the calibration set as the lowest threshold achieving **auto-match precision ≥ 0.99**. This is the single most important design decision in the project and should be said out loud in the pitch: *in finance ops, a wrong auto-match is far more expensive than a human reviewing an extra item.* Optimising for match rate instead of precision is the trap most submissions will fall into.

---

## 7. Agent, Tools & LLM Responsibilities

### 7.1 The hard rule

> **The LLM never decides a match by itself, and never does arithmetic.**

Money math is deterministic Python. The LLM proposes hypotheses and writes explanations; code verifies and decides. Every LLM-proposed match is re-checked against the arithmetic invariants before it can be accepted. This is both the correct engineering choice and a strong interview answer.

### 7.2 Tool inventory

| Tool | Type | Called by | Purpose |
|---|---|---|---|
| `parse_narration` | LLM + regex | normalize | extract UTR, merchant, txn type from free text; regex first, LLM only on regex miss |
| `subset_sum_solver` | pure Python | T2 | find composing payment subsets |
| `fuzzy_match` | RapidFuzz | T3 | lexical similarity |
| `semantic_search` | ChromaDB | T3 | merchant-name variant resolution |
| `graph_query` | Neo4j | T4 | constraint propagation, ring detection |
| `verify_arithmetic` | pure Python | all | integer-paise invariant check; hard gate |
| `adjudicate_residual` | LLM | T5 | structured hypothesis on unmatched items |
| `explain_exception` | LLM | classify | plain-English root cause from evidence |
| `suggest_action` | LLM | classify | recommended next step for a human |

### 7.3 LLM call budget

Three call sites only. Everything else is deterministic.

1. **`parse_narration`** — batched, 20 narrations per call, only for regex misses. ~15 calls per 300 records.
2. **`adjudicate_residual`** — batched, 10 residual items per call, given a compact evidence pack (candidate links, amount deltas, graph neighbourhood). Returns strict Pydantic-validated JSON: `{item_id, hypothesis, proposed_link|null, confidence, reasoning, evidence_refs[]}`. ~5 calls per 300 records.
3. **`explain_exception`** — one batched call per exception cluster, grouped by class. ~8 calls per 300 records.

**Target: < 30 LLM calls per 300 records.** Tracked and reported as `llm_calls_per_100_records`, `tokens_per_run`, and `actual_cost_inr` (**₹0 on the free tier**) alongside `equivalent_paid_cost_inr` — what the same run would cost on a frontier paid API. That second figure is the interesting one for judges: it quantifies what the deterministic-first design saves.

Temperature 0, seeded, responses content-hashed and cached to disk so reruns and CI are free and deterministic.

**Free-tier resilience (built in Phase 5, not retrofitted):** every call goes through one OpenAI-compatible client with exponential backoff on 429, then automatic failover down the provider ladder (Groq → Gemini → OpenRouter → Ollama), then graceful degradation to `--no-llm` behaviour for that node. A rate limit slows a run; it never fails one. Provider and fallback depth are recorded in the audit trail.

### 7.4 Guardrails

- All LLM output through Pydantic strict schemas; validation failure → one retry with the error appended → then fall through to exception. Never a crash, never a silent default.
- LLM-proposed links pass `verify_arithmetic` before acceptance. Fail ⇒ demoted to `NEEDS_REVIEW`, with the failure logged as evidence.
- `llm_confidence` is a *feature*, never the final probability — it goes through the same calibration as every other signal, because raw LLM self-reported confidence is famously overconfident. Showing the before/after calibration curve for LLM confidence is a great slide.
- `--no-llm` flag runs the whole pipeline deterministically. Powers the ablation and guarantees the demo survives an API outage.

---

## 8. Exception Handling

### 8.1 Exception object

```python
Exception:
    exception_id: str
    class: ExceptionClass          # 12-way taxonomy, mirrors anomaly classes
    severity: CRITICAL | HIGH | MEDIUM | LOW   # driven by ₹ impact + age
    impact_paise: int              # money at stake — the sort key
    involved_records: list[RecordRef]
    root_cause: str                # LLM-written, evidence-grounded
    evidence: list[EvidenceItem]   # every item links back to a source record
    suggested_action: str          # e.g. "Request chargeback detail for SETL-0091"
    confidence: float              # calibrated confidence in the *classification*
    resolvable_by_agent: bool
    ambiguity: list[Hypothesis]|None  # populated when >1 explanation fits
```

### 8.2 Principles

1. **Never force a match.** Below threshold ⇒ exception. Match rate is not the objective function.
2. **Every exception is typed and explained.** A bare "unmatched" count is not a deliverable.
3. **Ranked by money, not by count.** The queue sorts by `impact_paise` descending. A controller cares about the ₹4L exception, not the 200 one-paise drifts.
4. **Ambiguity is preserved, not collapsed.** Two viable hypotheses ⇒ both are shown with their probabilities.
5. **The `UNMATCHABLE` floor is reported honestly.** The report states: *"X items are unmatchable by construction; no system could resolve them without external data."* This distinguishes a real ceiling from a model failure and is exactly the kind of intellectual honesty the track bar rewards.

### 8.3 Auto-resolution (bounded)

A narrow set of classes the agent may auto-resolve, each with a hard bound and a full audit entry:

| Class | Auto-resolution | Bound |
|---|---|---|
| `ROUNDING_DRIFT` | post rounding adjustment | ≤ ₹5 per record, ≤ ₹500 per run |
| `TIMING_SHIFT` | re-window and re-match | ≤ 5 days |
| `DUPLICATE_CREDIT` | flag second credit, link to first | never deletes anything |

Everything else is proposed only. Bounds are config, printed in the report, and enforced in code.

---

## 9. Accuracy & Evaluation Methodology

### 9.1 Metric set

**Primary (the headline three):**

| Metric | Definition | Target |
|---|---|---|
| **Auto-match precision** | correct / all auto-matched | **≥ 0.99** |
| **Match rate** | auto-matched / total reconcilable | ≥ 0.85 (standard difficulty) |
| **Exception recall** | true exceptions correctly flagged / all true exceptions | ≥ 0.95 |

**Secondary:**

- Pair-level precision / recall / F1
- Per-anomaly-class recall (12-row table — **including the classes that do badly**)
- Exception-class confusion matrix (12×12)
- False-positive cost: `sum(impact_paise)` of incorrect auto-matches — a rupee figure, not a ratio
- ₹ reconciled vs ₹ outstanding
- Calibration: **ECE** + Brier score + reliability diagram
- Throughput: records/sec, wall-clock for 5,000-record scale run
- Cost: LLM calls / 100 records, tokens per run, **actual ₹0** vs equivalent paid-API ₹ per 1,000 records

### 9.2 Baselines (this is what makes the numbers mean something)

Run all four on the identical held-out test set:

| # | Baseline | Purpose |
|---|---|---|
| B0 | Exact-join only (`pandas.merge` on UTR) | the "why not just SQL" answer |
| B1 | Exact + fuzzy | the typical hackathon submission |
| B2 | **LLM-only** — dump all records, ask it to reconcile | the "why not just an LLM" answer |
| B3 | **LedgerLoop, full** | ours |

Expected story, to be verified not assumed: B2 scores *worse* than B1 on precision while consuming ~40× the tokens and being non-deterministic.

> **Run B2 on the small `dev` set (60 records), not `test`.** An LLM-only baseline burns tokens fast and would eat a meaningful slice of a daily free-tier quota on 300 records. Sixty records is more than enough to demonstrate the point, and the reduced scope is stated in `EVALUATION.md`.

This single table is the strongest slide in the deck.

### 9.3 Ablation

| Config | Match rate | Auto precision | LLM calls | Cost |
|---|---|---|---|---|
| T0 only | | | 0 | |
| T0–T1 | | | 0 | |
| T0–T2 (+aggregation) | | | 0 | |
| T0–T3 (+fuzzy/semantic) | | | ~15 | |
| T0–T4 (+graph) | | | ~15 | |
| T0–T5 (full) | | | ~28 | |

The marginal contribution of each tier, priced. This is what "measured accuracy" actually means.

### 9.4 Protocol hygiene

- **Calibration set is never evaluated on.** Thresholds and isotonic fit come from `calibration`; all reported numbers come from `test`.
- Seeded, deterministic reruns; LLM at temperature 0 with a response cache.
- **5 seeds × 3 difficulties**, report **mean ± std**. A single run's number is noise.
- Every metric regenerated by one command: `make eval` → writes `EVALUATION.md` and `reports/*.png`. Nothing in the report is hand-typed.

---

## 10. Tech Stack & Technical Decisions

| Layer | Choice | Why (and what was rejected) |
|---|---|---|
| Language | Python 3.11 | ecosystem; JD asks for it |
| Deps | `uv` | 10× faster than Poetry, lockfile, one binary |
| Agent | **LangGraph** | needs the loop + conditional branch + checkpointing. Rejected CrewAI: role-play abstraction adds nothing to a deterministic pipeline |
| Schemas | Pydantic v2 | strict LLM output validation, one source of truth |
| LLM | **Groq free tier** (primary) → Gemini free tier (fallback) → Ollama (offline), behind one OpenAI-compatible client | ₹0. Groq publishes its limits (~30 RPM, ~1,000 RPD), is OpenAI-API-compatible, and needs no card. Swappable, cached, `--no-llm` mode |
| Embeddings | **`all-MiniLM-L6-v2` locally via sentence-transformers** (Chroma default) | ₹0, offline, CPU-only. Explicitly **not** a hosted embeddings API — those bill per token |
| Vector DB | **ChromaDB** (embedded, local persist) | zero-ops, free, sufficient at this scale. Rejected Qdrant/pgvector: extra infra for no gain here |
| Graph DB | **Neo4j 5 Community Edition** (self-hosted, Docker) | free and unrestricted for this use. **Not** Enterprise, **not** AuraDB cloud tiers. NetworkX fallback behind the same interface |
| RDBMS | **Postgres 16** | runs, decisions, audit. Rejected DuckDB: wanted concurrent API reads during a run |
| Data | Polars | 5–10× pandas on the scale run; pandas kept only for eval convenience |
| Fuzzy | RapidFuzz | C++ speed |
| ML | scikit-learn | logistic + isotonic regression only. **Deliberately no deep learning** — it would be worse and unjustifiable here, and saying so is a stronger signal than bolting on a neural net |
| API | FastAPI + SSE | streams run progress to the UI live |
| UI | React + Vite + Tailwind + shadcn/ui + Recharts | **Streamlit fallback if behind schedule** |
| Graph viz | Cytoscape.js | renders the lineage chain |
| Container | Docker + Compose | one-command demo |
| Orchestration | k3s manifests | *stretch only* — do not start before Phase 8 |
| Tests | pytest, Hypothesis, pytest-cov | property tests on money invariants |
| Quality | ruff, mypy (strict on `core/`), pre-commit | |
| CI | GitHub Actions | free with unlimited minutes on public repos |
| Hosting | **HF Spaces (Docker)** or **Oracle Cloud Always Free** | see §15. **Not AWS t3.medium** — that is not free tier |
| Observability | structlog JSONL + OpenTelemetry spans | the audit trail |

### 10.1 Zero-cost guarantee

**Hard constraint: this project must cost ₹0 to build, run, and demo.** Every component above is either open-source and self-hosted, or has a standing free tier requiring no credit card.

**The only component that could ever cost money is LLM inference**, and it is contained three ways:

1. **Free-tier provider.** Groq's free tier allows on the order of 1,000 requests/day. The pipeline uses **< 30 calls per 300-record run** — roughly 0.3% of a single day's quota. Even 50 full evaluation runs in a day stays comfortably inside it.
2. **Response cache.** Every LLM response is content-hash cached to disk and committed as a test fixture. Reruns, CI, and the demo consume **zero** API calls.
3. **`--no-llm` mode.** The entire pipeline runs deterministically with no network at all. This is not just a cost measure — it powers the ablation study and guarantees the demo survives a rate limit or outage.

**Provider ladder** (all behind one OpenAI-compatible interface, switched by env var):

| Order | Provider | Why | Card needed |
|---|---|---|---|
| 1 | **Groq** | published limits, OpenAI-compatible, fastest | No |
| 2 | **Google Gemini** (AI Studio) | current Flash models free; limits visible only in AI Studio | No |
| 3 | **OpenRouter** | breadth, free model variants, useful failover | No |
| 4 | **Ollama** (local, Llama 3.1 8B / Qwen) | fully offline, no limits, no network. Needs ~8 GB free RAM | N/A |

> ⚠️ Free-tier limits change frequently — verify current quotas at signup rather than trusting any number written here. Design for 429s: exponential backoff, then automatic failover down the ladder, then `--no-llm`. This is implemented in Phase 5, not bolted on later.

**Privacy note:** free tiers may use prompts for model training outside the EU/UK. Irrelevant here — all data is synthetic and self-generated — but the README should say so explicitly, since a judge may wonder.

### Decisions worth defending in the pitch

- **Integer paise everywhere.** No floats in the money path, enforced by a property test. Every finance engineer who sees this nods.
- **LLM is a dependency, not the architecture.** Swappable, cacheable, ablatable, disableable.
- **Precision over match rate.** Threshold selection targets 99% auto-match precision, not maximum coverage.
- **No deep learning.** Justified, not omitted.
- **₹0 to run, by architecture not by luck.** The deterministic-first design is what makes a free tier sufficient — a naive LLM-per-record system would blow through any free quota in one run. Cost discipline and accuracy came from the same decision. Say this in the pitch; it reframes a constraint as a design win.

---

## 11. Project Structure

```
ledgerloop/
├── README.md                     # hero: architecture diagram + headline metrics table
├── PLAN.md                       # this file
├── ARCHITECTURE.md               # deep dive: tiers, graph model, decision policy
├── EVALUATION.md                 # AUTO-GENERATED by `make eval`
├── DEMO.md                       # 5-min walkthrough script for the video
├── Makefile                      # demo | eval | test | up | down | data
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
│
├── data/
│   ├── seeds.json                # committed seeds → byte-identical regeneration
│   ├── generated/                # gitignored
│   └── fixtures/                 # small committed sets for tests
│
├── src/ledgerloop/
│   ├── config.py                 # RunConfig: thresholds, tolerances, bounds
│   ├── models/                   # Pydantic: CanonicalRecord, MatchCandidate,
│   │                             #   MatchDecision, Exception, RunMetrics, AuditEvent
│   ├── generator/
│   │   ├── scenarios.py          # the 12 anomaly classes
│   │   ├── emitters.py           # csv / json / (pdf) writers
│   │   └── ground_truth.py       # truth first, data derived from it
│   ├── ingest/
│   │   ├── ledger.py  psp.py  bank.py
│   │   └── narration.py          # regex-first, LLM-fallback parser
│   ├── normalize/
│   │   ├── canonical.py
│   │   └── money.py              # integer paise, currency, invariants
│   ├── graph/
│   │   ├── interface.py          # GraphRepo protocol
│   │   ├── neo4j_repo.py
│   │   ├── networkx_repo.py      # zero-infra fallback
│   │   └── queries.py            # Cypher: sibling completion, closure, rings
│   ├── matching/
│   │   ├── tier0_exact.py
│   │   ├── tier1_tolerance.py
│   │   ├── tier2_aggregation.py  # ← the algorithmic core
│   │   ├── tier3_fuzzy.py
│   │   ├── tier4_graph.py
│   │   ├── blender.py            # feature vector → raw score
│   │   └── calibration.py        # isotonic fit, threshold selection
│   ├── agent/
│   │   ├── graph.py              # LangGraph assembly
│   │   ├── nodes/                # one file per node
│   │   ├── tools/                # the 9 tools
│   │   └── prompts/              # versioned .md prompts, hashed into audit
│   ├── exceptions/
│   │   ├── taxonomy.py
│   │   ├── classifier.py
│   │   └── resolver.py           # bounded auto-resolution
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── baselines.py          # B0, B1, B2
│   │   ├── ablation.py
│   │   ├── calibration_plots.py
│   │   └── report.py             # writes EVALUATION.md
│   ├── audit/
│   │   ├── logger.py
│   │   └── replay.py
│   ├── api/
│   │   ├── main.py  routes/  sse.py
│   └── cli.py                    # ledgerloop generate | run | eval | replay
│
├── ui/
│   ├── src/
│   │   ├── pages/                # Dashboard, Exceptions, Explorer, Audit
│   │   ├── components/
│   │   └── lib/api.ts
│   └── package.json
│
├── tests/
│   ├── unit/  property/  integration/  golden/
│   └── fixtures/llm_cache/       # cached LLM responses → free, deterministic CI
│
├── deploy/
│   ├── k8s/                      # stretch, local k3s
│   ├── hf-space/                 # Dockerfile + README.md for HF Spaces
│   └── oracle-setup.sh           # Always Free ARM provisioning
│
└── reports/                      # generated charts + metrics json
```

---

## 12. Implementation Phases & Milestones

**~20 working days, solo, part-time.** Each phase ends with a demoable artifact. If a phase slips, cut scope inside it — never skip its acceptance criteria.

---

### Phase 0 — Scaffold (Day 1)

Repo, `uv` env, ruff/mypy/pre-commit, Docker Compose (Postgres + Neo4j Community), Pydantic models, `RunConfig`, CLI skeleton, CI running an empty test suite.

**Also on Day 1:** sign up for **Groq** and **Google AI Studio** keys (both free, no card), record the *current* published rate limits in `README.md`, and confirm `all-MiniLM-L6-v2` downloads and embeds locally. Doing this first means you discover any quota surprise on Day 1, not Day 12.

**Acceptance:** `make up` starts all containers healthy; `pytest` green; `mypy --strict src/ledgerloop/models` clean; a single smoke call succeeds against Groq **and** the run completes with `--no-llm`.

---

### Phase 1 — Synthetic data + ground truth (Days 2–3) ⭐ *do not rush this*

All 12 anomaly classes; three emitters; ground-truth-first generation; difficulty dial; committed seeds; the four dataset sizes.

Everything downstream is measured against this. A bug here silently invalidates every metric in the project.

**Acceptance:**
- `ledgerloop generate --profile test --seed 42` twice ⇒ byte-identical output
- All 12 classes present at configured prevalence (±2%)
- `data/fixtures/` committed with a 60-record set
- Property test: total money conserved across all three sources modulo declared anomalies

---

### Phase 2 — Ingest + normalize (Days 4–5)

Three parsers, canonical schema, integer-paise money module, regex narration parser (LLM fallback stubbed), records persisted to Postgres.

**Acceptance:** 300-record set ingests with 0 parse failures; ambiguous `DD/MM` dates resolved correctly (test asserts it); no float appears in any money field (property test).

---

### Phase 3 — Deterministic tiers T0–T2 (Days 6–8) ⭐ *the algorithmic core*

Exact, tolerance, and the aggregation solver. Ambiguity detection. Feature vector emission. First real metrics.

**Acceptance:**
- T0+T1+T2 ≥ **70% match rate** on `test`, standard difficulty
- Aggregation solver ≤ 200 ms per bank credit at bucket size 40
- Ambiguous subsets emit `AMBIGUOUS_AGGREGATION`, never a silent pick
- **First `EVALUATION.md` generated** — you now have a measurable system

---

### Phase 4 — T3 fuzzy/semantic + T4 graph (Days 9–11)

Chroma embedding of merchant variants, RapidFuzz narration matching, Neo4j model + Cypher rules, NetworkX fallback, iterative re-run loop.

**Acceptance:** match rate ≥ **82%** with zero LLM calls; all four Cypher rules unit-tested; `--graph-backend networkx` produces identical decisions on the fixture set.

---

### Phase 5 — LangGraph agent + T5 adjudication (Days 12–13)

Full state machine, the 9 tools, batched LLM adjudication with Pydantic validation, response caching, `--no-llm` flag, cost ledger, and the **provider failover ladder with 429 backoff**.

**Acceptance:** end-to-end run completes on `test`; **< 30 LLM calls per 300 records**; `--no-llm` runs green; every LLM-proposed match passes `verify_arithmetic` or is demoted; malformed LLM output never crashes a run (fault-injection test); **a simulated 429 storm fails over Groq → Gemini → `--no-llm` without failing the run**; a second identical run consumes **0 API calls** (cache hit rate 100%).

---

### Phase 6 — Calibration + evaluation + exceptions (Days 14–15) ⭐ *the credibility phase*

Isotonic calibration, precision-targeted threshold selection, exception taxonomy + LLM explanations, bounded auto-resolution, three baselines, ablation table, reliability diagrams, auto-generated `EVALUATION.md`.

**Acceptance:**
- Auto-match precision ≥ **0.99** on held-out `test`
- Match rate ≥ **0.85**, exception recall ≥ **0.95**
- **ECE ≤ 0.05**; reliability diagram rendered
- Baselines B0/B1/B2 and the full ablation table populated
- 5 seeds × 3 difficulties, mean ± std reported
- Every exception has class + root cause + evidence + ₹ impact + action

---

### Phase 7 — UI + audit replay (Days 16–17)

FastAPI + SSE, four React pages, Cytoscape lineage view, decision-by-decision replay.

**Acceptance:** run triggerable from the browser with live progress; exception queue sortable by ₹ impact; clicking any match opens its full evidence chain; replay steps through the audit log.

---

### Phase 8 — Package, deploy, tell the story (Days 18–20)

Compose polish, `make demo` one-command, README with headline metrics table, `ARCHITECTURE.md`, `DEMO.md` script, 5-minute video, free hosting deploy (HF Spaces or Oracle Always Free), k3s manifests *if and only if* time remains.

**Acceptance:** fresh clone → `make demo` → working UI with a completed run in **under 5 minutes** on a clean machine (verify on a fresh VM, not your laptop); video under 5:00; repo public and clean.

---

### Day 21 — Buffer. Do not plan work here.

---

## 13. Testing Strategy

| Layer | What | Tool |
|---|---|---|
| **Unit** | each tier, money module, taxonomy, parsers | pytest |
| **Property** | money conserved; no floats in money path; matching is order-independent; a match is symmetric | Hypothesis |
| **Golden** | fixed 60-record fixture → committed expected decisions; any diff fails CI | pytest + JSON snapshot |
| **Integration** | full Compose stack, end-to-end run, API contract | pytest + testcontainers |
| **Fault injection** | malformed LLM JSON, Neo4j down, truncated source file, duplicate input file | pytest |
| **Eval gate** | CI **fails** if auto-match precision < 0.99 or match rate < 0.80 on the fixture set | GH Actions |
| **Performance** | 5,000-record run under a wall-clock budget | pytest-benchmark |

**The eval gate is the point.** Metrics as a CI gate — not a one-off number in a slide — is what turns "we measured it" into "we hold ourselves to it." Mention it in the pitch.

LLM responses are cached as committed fixtures, so CI is deterministic, free, and requires no API key.

Coverage target: **≥ 85% on `src/ledgerloop/matching/` and `/normalize/`** (the money-critical paths). Do not chase coverage on UI glue.

---

## 14. UI & Demo

### 14.1 Four screens

**1. Run Dashboard** — the money view.
Big numbers: ₹ reconciled, ₹ outstanding, match rate, auto-match precision, records/sec, LLM cost. Tier-contribution waterfall (how many matches each tier produced). Live SSE progress during a run.

**2. Exception Queue** — the controller's actual workday.
Sorted by ₹ impact descending. Each row expands to: root cause in plain English, evidence chain with links to source records, suggested action, confidence, and — where relevant — competing hypotheses with their probabilities.

**3. Match Explorer** — the lineage graph.
Cytoscape rendering of `Order → Payment → Settlement → BankTxn`. Edge colour = confidence. Click an edge to see which tier produced it and why. This is the screenshot that goes in the README.

**4. Audit Replay** — the debugger.
Step through a run decision by decision: tier fired, features, raw score, calibrated probability, evidence, and (where applicable) the exact prompt hash and token count.

### 14.2 The 5-minute pitch video

| Time | Beat |
|---|---|
| 0:00–0:30 | The problem, in money: three systems, one truth, ₹X unaccounted |
| 0:30–1:15 | The data: three real formats, 12 anomaly classes, ground truth |
| 1:15–2:30 | Live run: `make demo`, SSE progress, tier waterfall filling in |
| 2:30–3:30 | **The exception queue** — pick the biggest ₹ exception, show root cause + evidence chain. *This is the emotional peak of the demo; give it the most time.* |
| 3:30–4:30 | **The numbers:** precision 0.99, match rate, calibration curve, baseline table (LLM-only is worse and 40× costlier), honest per-class recall including the weak classes |
| 4:30–5:00 | Architecture in one diagram; what you'd build next |

Record the *actual* run. No mockups, no sped-up footage passed off as real time.

---

## 15. Deployment

**All three tiers are ₹0. No cloud bill, no credit card.**

**Tier 1 — Docker Compose, local (primary, must work).**
`docker compose up` → Postgres, Neo4j Community, API, UI. `make demo` additionally generates data and executes a run, so a judge sees a populated dashboard immediately. This is the path that must be bulletproof, and it is the only one that is non-negotiable.

**Tier 2 — Free public hosting (pick one).**

| Option | Why | Watch out for |
|---|---|---|
| **Hugging Face Spaces** (Docker SDK) | free, no card, your Dockerfile ports over almost unchanged, gives a clean public URL | 16 GB disk / limited RAM — run the demo against the pre-generated 300-record set, not the 5,000 scale set |
| **Oracle Cloud Always Free** | genuinely permanent, 4 ARM cores + 24 GB RAM — runs all four containers comfortably | ARM64: build multi-arch images, or `--platform linux/arm64`. Signup can be slow |
| **Render free tier** | easiest setup | spins down when idle; a judge clicking the link waits ~50 s for a cold start. Say so in the README if you use it |

Recommendation: **HF Spaces** for the demo URL, since a hackathon judge clicking a link that works instantly matters more than raw horsepower. If the full stack is too heavy for a Space, deploy a **read-only results viewer** seeded with a completed run — the live run stays local in the video. That is a legitimate trade, not a cheat, as long as the README says which it is.

**Tier 3 — k3s (stretch, Phase 8 only).**
Local single-node k3s on your own machine — free. Manifests for each service, a Job for the reconciliation run, HPA on the API. This exists to demonstrate Kubernetes competence for the internship JD; it adds nothing to the hackathon score. **Do not touch it before Phase 8 is otherwise complete.**

**If hosting proves fiddly, drop it entirely.** A great demo video plus a `make demo` that works from a fresh clone is worth more than a flaky live URL.

Config via environment; `.env.example` committed; **no API keys in the repo, ever** — CI runs against the cached LLM fixtures, so it needs no key at all. Images built and pushed to GHCR by CI (free for public repos).

---

## 16. Final Deliverables

1. **Public GitHub repo** — clean history, MIT licence, no secrets
2. **`README.md`** — architecture diagram, headline metrics table, `make demo` quickstart, live URL (if deployed), and a **"reproduce this for ₹0"** section listing the free providers and setup steps
3. **`ARCHITECTURE.md`** — tier ladder, graph model, calibration approach, decision policy
4. **`EVALUATION.md`** — auto-generated: all metrics, baselines, ablation, confusion matrix, calibration plots, per-class recall
5. **5-minute pitch video** — unlisted YouTube, linked in README
6. **`make demo`** — one command, under 5 minutes, from a fresh clone
7. **Seeded datasets + generator** — anyone can reproduce every number, at zero cost
8. **Live deployed URL** *(optional stretch — free hosting only)*
9. **`DEMO.md`** — the walkthrough script

---

## 17. Definition of Done

The project ships only when **all** of these hold on the held-out `test` set:

**Functional**
- [ ] Runs end-to-end on ≥ 300 records (challenge asks 50+; exceed it visibly)
- [ ] Three heterogeneous sources ingested
- [ ] All six tiers implemented and contributing measurably
- [ ] Every exception carries class + root cause + evidence + ₹ impact + suggested action
- [ ] `--no-llm` mode runs green
- [ ] Audit trail complete and replayable
- [ ] Provider failover ladder works (429 → next provider → `--no-llm`)
- [ ] No API keys anywhere in the repo; CI passes with no key configured

**Measured**
- [ ] Auto-match precision **≥ 0.99**
- [ ] Match rate **≥ 0.85** (standard difficulty)
- [ ] Exception recall **≥ 0.95**
- [ ] ECE **≤ 0.05** with reliability diagram
- [ ] < 30 LLM calls per 300 records; actual spend **₹0**, with equivalent paid-API cost reported for contrast
- [ ] Second identical run consumes 0 API calls (cache verified)
- [ ] Full pipeline runs green with `--no-llm` and with Ollama
- [ ] 5,000-record scale run completes within stated budget
- [ ] 5 seeds × 3 difficulties, mean ± std
- [ ] Baselines B0/B1/B2 + full ablation table published

**Engineering**
- [ ] ≥ 85% coverage on matching + normalize
- [ ] Property tests on money invariants passing
- [ ] Golden regression test in CI
- [ ] CI eval gate active and enforcing
- [ ] `mypy --strict` clean on `core/`
- [ ] Fresh clone → `make demo` → working UI in < 5 min, verified on a clean machine

**Story**
- [ ] README leads with the metrics table
- [ ] Video under 5:00, real footage
- [ ] Live URL reachable *(optional — cut without guilt if hosting fights back)*
- [ ] README states the whole project costs ₹0 to reproduce, and how
- [ ] Limitations section written — including what the system does badly

---

## 18. Risks & Cut Lines

| Risk | Mitigation |
|---|---|
| Data generator bugs invalidate all metrics | Property tests in Phase 1; ground truth generated first, never inferred |
| Subset-sum blows up combinatorially | Bucket by window+merchant; anchor on settlement_id; hard 200 ms cap; greedy fallback |
| Calibration overfits | Separate calibration split, never evaluated on; report mean ± std across 5 seeds |
| LLM non-determinism breaks reproducibility | temp 0 + content-hash response cache committed as fixtures |
| **Free-tier rate limit (429) mid-run or mid-demo** | backoff → provider failover ladder → `--no-llm`; response cache means the demo run needs 0 live calls |
| **Free tier revoked or quota cut without notice** | four providers behind one interface; Ollama local as the floor; `--no-llm` always works |
| **Free hosting too small for the stack** | fall back to a read-only results viewer seeded with a completed run, or drop hosting entirely — local `make demo` is what's graded |
| Neo4j dies mid-demo | NetworkX fallback behind the same interface, exercised in CI |
| UI eats the schedule | Streamlit fallback; UI is Phase 7, after all metrics exist |
| Over-scoping | Cut list below, in order |

### Cut in this order if behind schedule

1. k3s manifests (Tier 3 deploy)
2. Public hosting entirely — local `make demo` + video is what's graded
3. PDF bank statement source
4. Ring detection in T4
5. React UI → **Streamlit**
6. Scale run 5,000 → 1,000 records
7. Difficulty dial → standard difficulty only

### Never cut, under any circumstances

- The three-way structure (D1)
- The aggregation solver (D2)
- Calibration + honest metrics (D3, D8)
- Typed, explained exceptions (D4)
- The baseline comparison table (D5)

Those five *are* the project. Everything else is packaging.

---

## Appendix — One-Line Pitch

> **LedgerLoop** reconciles the internal ledger, PSP settlements, and bank statements three ways across heterogeneous formats — solving the many-payments-to-one-payout aggregation problem deterministically, calling an LLM only on the ~5% residual, and reporting every match with a calibrated confidence and every exception with a typed root cause, an evidence chain, and a rupee figure.
>
> **0.99 auto-match precision. 0.85 match rate. Under 30 LLM calls per 300 records — which is why the whole thing runs on a free tier for ₹0. And an exception list we're not embarrassed by.**
