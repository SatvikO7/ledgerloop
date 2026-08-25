# LedgerLoop

**A three-way, confidence-calibrated reconciliation agent with an honest exception list.**

Reconciles the internal ledger, PSP settlements, and bank statements across heterogeneous
formats — solving the many-payments-to-one-payout aggregation problem deterministically,
calling an LLM only on the residual, and reporting every match with a calibrated confidence
and every exception with a typed root cause, an evidence chain, and a rupee figure.


## Generating data

```bash
ledgerloop generate --split test --difficulty standard --seed 42
```

Ground truth is generated **first**, and the data is derived from it — never inferred
afterwards. Generation is a pure function of `(seed, split, difficulty, order_count)`, so the
same command twice produces byte-identical files and anyone can reproduce every number.

| Split | Orders | Purpose |
|---|---|---|
| `dev` | 60 | fast iteration; meets the challenge's 50+ bar |
| `train` | 400 | fits the score blender |
| `calibration` | 200 | fits isotonic calibration and selects thresholds — never evaluated on |
| `test` | 300 | every published number comes from here |
| `scale` | 5,000 | throughput benchmark (~1.5 s to generate) |

Eleven anomaly classes are injected at controlled prevalence, with a `--difficulty
{easy,standard,hard}` dial that changes *how much* goes wrong without changing *what* goes
wrong. Money conservation is enforced as a property test: the bank credits reconcile to the
declared settlement nets exactly, modulo the deltas each anomaly explicitly declares.

A committed 60-order fixture set lives in `data/fixtures/dev-standard-42/`.

The data model and the definitions the evaluation depends on are documented in the modules
themselves — `models/truth.py` for the link-level ground truth, `eval/metrics.py` for the
metric definitions, and `money.py` for the integer-minor-unit invariant. The `§` references
in those docstrings point at an internal design document that is not published.

## MVP scope

Four scope decisions taken before implementation:

| Decision | Rationale |
|---|---|
| **Neo4j cut** — NetworkX is the real implementation, behind `graph/interface.py` | The plan required both to produce identical decisions, which makes the database pure overhead |
| **ChromaDB cut** — T3 is lexical-only, behind `vector/interface.py` | Sentence embeddings are weak on vowel-dropped abbreviations; a normaliser plus fuzzy matching is better and simpler |
| **Streamlit UI** — React is stretch only | The UI is scheduled after every metric exists; two days is not a React app |
| **A11 FX/multicurrency cut** | Multicurrency forces an FX rounding policy into the money path for 2% of records |

## Development

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest
uv run mypy
uv run ruff check .
```

## Licence

MIT.
