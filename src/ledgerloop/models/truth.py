"""Ground truth -- generated first, data derived from it. Never inferred.

WHY THIS IS LINK-LEVEL
----------------------
PLAN.md §5.3 specified a flat wide row::

    order_id | payment_id | settlement_id | bank_txn_id | expected_status | ...

That schema cannot express two of its own anomaly classes:

* **A09 SPLIT_PAYOUT** -- one settlement arrives as *two* bank credits. A single
  ``bank_txn_id`` column has nowhere to put the second.
* **A05 DUPLICATE_CREDIT** -- the same UTR is credited twice. The wide row
  cannot distinguish the genuine credit from the duplicate.

Truth is therefore stored as a set of typed **links** plus a set of per-record
**verdicts**. A link that should exist is in :attr:`GroundTruth.links`; a bank
credit that should match nothing (A05's duplicate, A10's orphan, the noise
rows) simply has no link, and any system that produces one has made a false
positive that the evaluator will count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel, LedgerModel, MinorUnits
from ledgerloop.models.enums import AnomalyClass, Difficulty, ExpectedStatus, LinkType, SplitName
from ledgerloop.models.refs import RecordRef

__all__ = ["GroundTruth", "GroundTruthLink", "GroundTruthRecord", "TruthPair"]

#: An unordered evaluation key: the two endpoints of a link, by flat key.
TruthPair = tuple[str, str]


class GroundTruthLink(FrozenLedgerModel):
    """One edge that *should* be discovered."""

    link_type: LinkType
    source_ref: RecordRef
    target_ref: RecordRef
    amount_minor: MinorUnits = Field(
        description="Money attributed to this specific edge. For a split payout the "
        "settlement's net is allocated across its bank credits, so the parts sum "
        "exactly to the whole (see money.allocate_minor)."
    )
    anomaly_class: AnomalyClass = AnomalyClass.CLEAN

    @property
    def pair(self) -> TruthPair:
        """Endpoint keys, for set comparison against predicted links."""
        return (self.source_ref.key, self.target_ref.key)


class GroundTruthRecord(FrozenLedgerModel):
    """The verdict for one record, and what the generator did to it."""

    record_ref: RecordRef
    expected_status: ExpectedStatus
    anomaly_class: AnomalyClass
    impact_minor: MinorUnits = Field(
        default=0,
        description="Money at stake if this record is never resolved. Zero for "
        "cleanly matched records. This is the exception queue's sort key and the "
        "unit of the false-positive cost metric.",
    )
    note: str | None = Field(
        default=None,
        description="Generator's own account of what it did, e.g. 'chargeback of "
        "₹4,312 netted into SETL-0091'. Never shown to the matcher; used to "
        "explain failures during development and to seed the report's honesty "
        "section.",
    )


class GroundTruth(LedgerModel):
    """The complete truth for one generated dataset."""

    split: SplitName
    difficulty: Difficulty
    seed: int
    generator_version: str = Field(
        description="Bumped whenever generation logic changes. A metric is only "
        "comparable to another metric produced by the same generator version."
    )
    links: tuple[GroundTruthLink, ...] = ()
    records: tuple[GroundTruthRecord, ...] = ()

    scenario_draws: dict[AnomalyClass, int] = Field(
        default_factory=dict,
        description="How many times each anomaly was *drawn* during generation, one "
        "draw per order. This is what prevalence is checked against, and it is not "
        "the same as counting labelled records: a single A10 draw injects an orphan "
        "bank credit while leaving its order clean, and a single A09 draw relabels "
        "one settlement while touching two bank rows. Counting records would make "
        "the configured prevalence unverifiable.",
    )

    @model_validator(mode="after")
    def _unique_records(self) -> GroundTruth:
        seen: set[str] = set()
        for record in self.records:
            key = record.record_ref.key
            if key in seen:
                raise ValueError(f"duplicate ground-truth verdict for {key}")
            seen.add(key)
        return self

    # ------------------------------------------------------------------
    # Evaluation surface
    #
    # ARCHITECTURE.md §2 fixes PAYMENT_CREDITED_AS as the atomic unit of
    # evaluation. These accessors are what the evaluator consumes, so that
    # definition lives in exactly one place instead of being re-derived (and
    # re-argued) in every metric function.
    # ------------------------------------------------------------------

    @cached_property
    def evaluation_pairs(self) -> frozenset[TruthPair]:
        """The atomic truth set: every ``(payment, bank_txn)`` edge that should exist.

        Precision and recall are computed against this set. Structural edges
        (``ORDER_PAID_BY``, ``PAYMENT_SETTLED_IN``) are excluded -- they are
        largely given by the sources and counting them would inflate every
        score with edges the system never had to work for.
        """
        return frozenset(
            link.pair for link in self.links if link.link_type is LinkType.PAYMENT_CREDITED_AS
        )

    @cached_property
    def links_by_type(self) -> Mapping[LinkType, tuple[GroundTruthLink, ...]]:
        grouped: dict[LinkType, list[GroundTruthLink]] = {}
        for link in self.links:
            grouped.setdefault(link.link_type, []).append(link)
        return {key: tuple(value) for key, value in grouped.items()}

    @cached_property
    def verdict_by_ref(self) -> Mapping[str, GroundTruthRecord]:
        return {record.record_ref.key: record for record in self.records}

    @cached_property
    def reconcilable_refs(self) -> frozenset[str]:
        """Records that a perfect system *could* resolve.

        The match-rate denominator. Excludes ``UNMATCHABLE`` -- PLAN.md §8.2.5's
        honest floor. Dividing by the full record count would charge the system
        for items that are irreconcilable by construction, understating it in a
        way that is no more honest than overstating it.
        """
        return frozenset(
            record.record_ref.key
            for record in self.records
            if record.expected_status is not ExpectedStatus.UNMATCHABLE
        )

    @cached_property
    def unmatchable_refs(self) -> frozenset[str]:
        """The reported ceiling: irreconcilable without data outside the three sources."""
        return frozenset(
            record.record_ref.key
            for record in self.records
            if record.expected_status is ExpectedStatus.UNMATCHABLE
        )

    def anomaly_counts(self) -> Mapping[AnomalyClass, int]:
        """Labelled records per class.

        Reports what each anomaly actually *touched*. For the configured
        prevalence check use :meth:`realised_prevalence` instead -- see
        :attr:`scenario_draws`.
        """
        counts: dict[AnomalyClass, int] = dict.fromkeys(AnomalyClass, 0)
        for record in self.records:
            counts[record.anomaly_class] += 1
        return counts

    def realised_prevalence(self) -> Mapping[AnomalyClass, float]:
        """Observed draw frequency per class, comparable to the configured weights."""
        total = sum(self.scenario_draws.values())
        if total == 0:
            return dict.fromkeys(AnomalyClass, 0.0)
        return {
            anomaly: self.scenario_draws.get(anomaly, 0) / total for anomaly in AnomalyClass
        }

    def impact_total_minor(self, refs: Iterable[str] | None = None) -> int:
        """Total money at stake across the given records (all records if ``None``)."""
        selected = self.records if refs is None else [
            self.verdict_by_ref[ref] for ref in refs if ref in self.verdict_by_ref
        ]
        return sum(record.impact_minor for record in selected)
