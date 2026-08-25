"""Closed vocabularies for the reconciliation domain.

Every enum here is a *contract*: the generator emits these values, the matcher
consumes them, and the evaluator counts them. Adding a member is a schema
change that ripples into the confusion matrix, so members carry stable string
values suitable for CSV and JSON round-tripping.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

__all__ = [
    "AnomalyClass",
    "Currency",
    "DecisionOutcome",
    "Difficulty",
    "EvidenceKind",
    "ExceptionClass",
    "ExpectedStatus",
    "LinkType",
    "OrderStatus",
    "ProseSource",
    "RecordType",
    "Severity",
    "SourceName",
    "SplitName",
    "Tier",
]


class SourceName(StrEnum):
    """The three heterogeneous inputs (PLAN.md §5.1)."""

    LEDGER = "ledger"
    PSP = "psp"
    BANK = "bank"


class RecordType(StrEnum):
    """Canonical entity types.

    Four types from three sources: the PSP file yields both settlements and the
    payments nested inside them, and those are separate entities because the
    N:1 aggregation problem is precisely about payments composing a settlement.
    """

    ORDER = "order"
    PAYMENT = "payment"
    SETTLEMENT = "settlement"
    BANK_TXN = "bank_txn"


class Currency(StrEnum):
    """Currencies the money path understands.

    INR only in the MVP. ``USD`` is declared so the A11 FX cut is explicit and
    testable rather than merely absent: ingest rejects it with a clear message
    instead of mis-scaling it as paise.
    """

    INR = "INR"
    USD = "USD"

    @property
    def supported(self) -> bool:
        """Whether the MVP money path can handle this currency."""
        return self is Currency.INR


class OrderStatus(StrEnum):
    CAPTURED = "CAPTURED"
    REFUNDED = "REFUNDED"
    PARTIAL_REFUND = "PARTIAL_REFUND"


class SplitName(StrEnum):
    """Dataset splits.

    ``TRAIN`` is the correction called out in the plan review: PLAN.md §5.4
    defined only dev/calibration/test, but the blender is a fitted model and
    needs its own fitting data. Fitting the logistic and the isotonic on the
    same split would let the calibrator see in-sample scores and report a
    calibration quality the system does not have.

    The three-way discipline: fit the blender on ``TRAIN``, fit isotonic and
    select thresholds on ``CALIBRATION``, report every published number from
    ``TEST``.
    """

    DEV = "dev"
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"
    SCALE = "scale"


class Difficulty(StrEnum):
    """Anomaly-prevalence dial (PLAN.md §5.2)."""

    EASY = "easy"
    STANDARD = "standard"
    HARD = "hard"


class AnomalyClass(StrEnum):
    """Ground-truth labels applied by the generator.

    Eleven classes: PLAN.md §5.2 listed twelve, and A11 ``FX_MULTICURRENCY``
    is cut from the MVP. Its 2% prevalence is reassigned to ``CLEAN``.

    These describe *what the generator did to the data*. They are deliberately
    NOT the same vocabulary as :class:`ExceptionClass`, which describes what the
    system concluded -- see that class's docstring.
    """

    CLEAN = "A01_CLEAN"
    ROUNDING_DRIFT = "A02_ROUNDING_DRIFT"
    FEE_TAX_MISMATCH = "A03_FEE_TAX_MISMATCH"
    TIMING_SHIFT = "A04_TIMING_SHIFT"
    DUPLICATE_CREDIT = "A05_DUPLICATE_CREDIT"
    POST_SETTLEMENT_REFUND = "A06_POST_SETTLEMENT_REFUND"
    MISSING_REFERENCE = "A07_MISSING_REFERENCE"
    CHARGEBACK_NETTED = "A08_CHARGEBACK_NETTED"
    SPLIT_PAYOUT = "A09_SPLIT_PAYOUT"
    ORPHAN_BANK_CREDIT = "A10_ORPHAN_BANK_CREDIT"
    LATE_ARRIVAL = "A12_LATE_ARRIVAL"


class ExceptionClass(StrEnum):
    """What the system concluded about an item it could not auto-match.

    Thirteen classes against eleven anomaly classes, because the two taxonomies
    answer different questions and the mapping between them is many-to-many.
    PLAN.md §8.1 called the exception taxonomy a mirror of the anomaly
    taxonomy; it is not, and pretending otherwise would make the confusion
    matrix uninterpretable:

    * ``CLEAN`` is an anomaly label but never an exception -- a clean record
      that lands in the queue is a *system* failure, reported as
      ``UNKNOWN_RESIDUAL``.
    * ``AMBIGUOUS_AGGREGATION`` (§6.2) and ``UNKNOWN_RESIDUAL`` are system
      states with no generator counterpart at all.
    * ``UNMATCHABLE`` is the honest ceiling of §8.2.5 -- irreconcilable by
      construction, and several anomaly classes can produce it.

    The anomaly -> exception mapping is therefore a measured artefact (the
    confusion matrix), never a hardcoded identity.
    """

    ROUNDING_DRIFT = "E_ROUNDING_DRIFT"
    FEE_TAX_MISMATCH = "E_FEE_TAX_MISMATCH"
    TIMING_SHIFT = "E_TIMING_SHIFT"
    DUPLICATE_CREDIT = "E_DUPLICATE_CREDIT"
    POST_SETTLEMENT_REFUND = "E_POST_SETTLEMENT_REFUND"
    MISSING_REFERENCE = "E_MISSING_REFERENCE"
    CHARGEBACK_NETTED = "E_CHARGEBACK_NETTED"
    SPLIT_PAYOUT_INCOMPLETE = "E_SPLIT_PAYOUT_INCOMPLETE"
    ORPHAN_BANK_CREDIT = "E_ORPHAN_BANK_CREDIT"
    LATE_ARRIVAL = "E_LATE_ARRIVAL"
    AMBIGUOUS_AGGREGATION = "E_AMBIGUOUS_AGGREGATION"
    UNMATCHABLE = "E_UNMATCHABLE"
    UNKNOWN_RESIDUAL = "E_UNKNOWN_RESIDUAL"


class Severity(StrEnum):
    """Exception urgency, driven by rupee impact and age (PLAN.md §8.1)."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Tier(IntEnum):
    """The tier ladder (PLAN.md §6.1).

    ``int``-valued so tiers order naturally (cheapest and most certain first),
    but the blender one-hots this rather than consuming the ordinal. Tier is
    near-perfectly predictive of correctness -- T0 exact-key matches are always
    right -- so feeding the raw integer to a logistic regression would let it
    dominate every other coefficient and collapse the model into a tier lookup.
    """

    T0_EXACT = 0
    T1_TOLERANCE = 1
    T2_AGGREGATION = 2
    T3_FUZZY = 3
    T4_GRAPH = 4
    T5_LLM = 5

    @property
    def is_deterministic_certain(self) -> bool:
        """Whether this tier's matches bypass the blender at ``p = 1.0``.

        T0 and T1 are exact-key and tolerance joins whose correctness follows
        from the match itself. They are excluded from blender fitting and from
        the calibration report: including ~70% of volume at a probability of
        essentially 1.0 would produce a degenerate reliability diagram with one
        populated bin and a near-zero ECE that measures nothing. Calibration is
        reported over the residual tiers, where uncertainty actually lives.
        """
        return self in (Tier.T0_EXACT, Tier.T1_TOLERANCE)


class LinkType(StrEnum):
    """Edges in the reconciliation chain.

    ``PAYMENT_CREDITED_AS`` is the derived closure edge and **the atomic unit
    of evaluation** -- see ``ARCHITECTURE.md`` §2. The other three are the
    structural edges the sources assert or the matcher infers.
    """

    ORDER_PAID_BY = "ORDER_PAID_BY"
    PAYMENT_SETTLED_IN = "PAYMENT_SETTLED_IN"
    SETTLEMENT_CREDITED_AS = "SETTLEMENT_CREDITED_AS"
    PAYMENT_CREDITED_AS = "PAYMENT_CREDITED_AS"


class ExpectedStatus(StrEnum):
    """Ground-truth verdict for a single record.

    ``UNMATCHABLE`` is the honest floor: no system could resolve these without
    data that does not exist in the three sources. They are excluded from the
    match-rate denominator and reported as a separate line, so a real ceiling
    is never confused with a model failure.
    """

    MATCHED = "MATCHED"
    EXCEPTION = "EXCEPTION"
    UNMATCHABLE = "UNMATCHABLE"


class DecisionOutcome(StrEnum):
    """Where the decision policy routed a candidate (PLAN.md §6.5)."""

    AUTO_MATCHED = "AUTO_MATCHED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    EXCEPTION = "EXCEPTION"
    REJECTED = "REJECTED"


class EvidenceKind(StrEnum):
    """Why a candidate was proposed. Every exception's evidence chain is a list
    of these, and each carries refs back to source records so a controller can
    verify the claim rather than trust it."""

    EXACT_KEY = "EXACT_KEY"
    AMOUNT_MATCH = "AMOUNT_MATCH"
    DATE_PROXIMITY = "DATE_PROXIMITY"
    SUBSET_SUM = "SUBSET_SUM"
    LEXICAL_SIMILARITY = "LEXICAL_SIMILARITY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    GRAPH_RULE = "GRAPH_RULE"
    ARITHMETIC_CHECK = "ARITHMETIC_CHECK"
    LLM_HYPOTHESIS = "LLM_HYPOTHESIS"
    NEGATIVE_EVIDENCE = "NEGATIVE_EVIDENCE"


class ProseSource(StrEnum):
    """Who wrote a human-readable string.

    Root causes and suggested actions have a deterministic template path and an
    LLM path. Recording which produced a given string keeps ``--no-llm`` runs
    honest in the report and lets the ablation attribute prose quality.
    """

    TEMPLATE = "TEMPLATE"
    LLM = "LLM"
