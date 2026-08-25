# LedgerLoop

**A three-way, confidence-calibrated reconciliation agent with an honest exception list.**

Reconciles the internal ledger, PSP settlements, and bank statements across heterogeneous
formats — solving the many-payments-to-one-payout aggregation problem deterministically,
calling an LLM only on the residual, and reporting every match with a calibrated confidence
and every exception with a typed root cause, an evidence chain, and a rupee figure.

> **Status: Step 0 of 14 — data contracts.** The money module, the Pydantic contracts and the
> deferred-infrastructure interfaces are in place and tested. No matching logic exists yet.
> Headline metrics will appear here once `make eval` can generate them; nothing in this file
> will be hand-typed.

See [PLAN.md](PLAN.md) for the full project plan and [ARCHITECTURE.md](ARCHITECTURE.md) for the
data model and the definitions the evaluation depends on.

## MVP scope

Four scope decisions taken before implementation, each recorded with its reasoning in
`ARCHITECTURE.md` §5:

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
