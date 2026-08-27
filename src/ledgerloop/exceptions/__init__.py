"""The exception queue -- the actual deliverable of a reconciliation run.

Four parts, and the split is the argument:

* :mod:`ledgerloop.exceptions.taxonomy` -- what an unresolved item *is*, as a
  cascade of rules over the three source documents. Never over ground truth.
* :mod:`ledgerloop.exceptions.templates` -- the deterministic prose. Built and
  measured before any LLM exists, so Step 9's contribution is attributable.
* :mod:`ledgerloop.exceptions.classifier` -- assembles the queue: class,
  severity, rupee impact, evidence chain, hypotheses, ordering.
* :mod:`ledgerloop.exceptions.resolver` -- the three bounded rules of PLAN.md
  8.3, which propose and never post.

Nothing in this package calls a model, and nothing in it reads an anomaly
label. The confusion matrix between the two vocabularies is assembled by the
evaluator afterwards, from two independently produced answers.
"""

from __future__ import annotations

from ledgerloop.exceptions.classifier import (
    ExceptionOutcome,
    classify_exceptions,
    exception_id,
    queue_order,
)
from ledgerloop.exceptions.resolver import (
    AutoResolution,
    ResolutionOutcome,
    mark_resolvable,
    resolve_bounded,
)
from ledgerloop.exceptions.taxonomy import (
    AGENT_RESOLVABLE_CLASSES,
    CreditItem,
    SettlementItem,
    classify_credit,
    classify_settlement,
    residual_items,
    severity_for,
)
from ledgerloop.exceptions.templates import PROSE_VERSION, Prose, prose_for

__all__ = [
    "AGENT_RESOLVABLE_CLASSES",
    "PROSE_VERSION",
    "AutoResolution",
    "CreditItem",
    "ExceptionOutcome",
    "Prose",
    "ResolutionOutcome",
    "SettlementItem",
    "classify_credit",
    "classify_exceptions",
    "classify_settlement",
    "exception_id",
    "mark_resolvable",
    "prose_for",
    "queue_order",
    "residual_items",
    "resolve_bounded",
    "severity_for",
]
