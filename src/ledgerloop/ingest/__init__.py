"""Ingest and normalise the three heterogeneous sources.

The boundary layer. Three files that agree on nothing -- clean CSV, nested
JSON, and CSV with free-text narration and ambiguous dates -- become one
corpus of :mod:`ledgerloop.models.records`, with every record still pointing
back at the row it came from.

Read :mod:`ledgerloop.ingest.dataset` first; the rest is the machinery it uses.
No tier, no join and no LLM lives here: ingest produces records, and step 4 is
the first thing entitled to relate them to each other.
"""

from ledgerloop.ingest.bank import BankRecords, parse_bank_csv, parse_bank_rows
from ledgerloop.ingest.dataset import IngestResult, ingest_dataset
from ledgerloop.ingest.dates import (
    DateOrder,
    DateOrderEvidence,
    infer_date_order,
    parse_slash_date,
    parse_timestamp,
)
from ledgerloop.ingest.ledger import parse_ledger_csv, parse_ledger_rows
from ledgerloop.ingest.narration import NarrationParse, parse_narration
from ledgerloop.ingest.normalize import (
    is_order_ref_shaped,
    merchant_skeleton,
    normalize_merchant_name,
    normalize_narration,
    normalize_order_ref,
    normalize_utr,
)
from ledgerloop.ingest.problems import (
    IngestError,
    IngestProblem,
    IngestProblemCode,
    ProblemLog,
)
from ledgerloop.ingest.psp import PspRecords, parse_psp_json, parse_psp_payload
from ledgerloop.ingest.schemas import (
    BANK_SCHEMA,
    LEDGER_SCHEMA,
    PSP_PAYMENT_SCHEMA,
    PSP_SETTLEMENT_SCHEMA,
    SourceSchema,
)

__all__ = [
    "BANK_SCHEMA",
    "LEDGER_SCHEMA",
    "PSP_PAYMENT_SCHEMA",
    "PSP_SETTLEMENT_SCHEMA",
    "BankRecords",
    "DateOrder",
    "DateOrderEvidence",
    "IngestError",
    "IngestProblem",
    "IngestProblemCode",
    "IngestResult",
    "NarrationParse",
    "ProblemLog",
    "PspRecords",
    "SourceSchema",
    "infer_date_order",
    "ingest_dataset",
    "is_order_ref_shaped",
    "merchant_skeleton",
    "normalize_merchant_name",
    "normalize_narration",
    "normalize_order_ref",
    "normalize_utr",
    "parse_bank_csv",
    "parse_bank_rows",
    "parse_ledger_csv",
    "parse_ledger_rows",
    "parse_narration",
    "parse_psp_json",
    "parse_psp_payload",
    "parse_slash_date",
    "parse_timestamp",
]
