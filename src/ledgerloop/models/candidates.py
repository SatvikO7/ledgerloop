"""Match candidates, features and evidence.

"Every match is an object, not a row" (PLAN.md §3.2). A candidate carries where
it came from, why it was proposed, what the arithmetic said, and -- once the
blender and calibrator have run -- how likely it is to be correct. Nothing is
lost between stages, because the audit replay has to reconstruct the decision.

THE FEATURE/MONEY BOUNDARY
--------------------------
:class:`FeatureVector` mixes ``int`` money fields and ``float`` score fields on
purpose, and labels which is which. ``amount_delta_minor`` is money and stays
exact; ``amount_delta_ratio`` is a feature derived from it via
:func:`~ledgerloop.money.delta_ratio` and is free to be a float because it
never becomes a rupee figure again.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel, LedgerModel, MinorUnits
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.refs import RecordRef

__all__ = ["Evidence", "FeatureVector", "MatchCandidate"]


class Evidence(FrozenLedgerModel):
    """One verifiable reason, pointing back at source records.

    An exception's evidence chain is a list of these. The rule is that a
    controller can check every item against the sources rather than trusting
    the system's summary -- which is what separates "₹4,312 short because a
    chargeback was netted off SETL-0091, evidence: [3 links]" from "unmatched".
    """

    kind: EvidenceKind
    detail: str = Field(description="One human-readable sentence.")
    refs: tuple[RecordRef, ...] = ()
    amount_minor: MinorUnits | None = Field(
        default=None, description="Money referenced by this item, when it references any."
    )
    score: float | None = Field(
        default=None, description="Feature-space score for similarity evidence."
    )


class FeatureVector(FrozenLedgerModel):
    """Inputs to the blender (PLAN.md §6.5), with tier handled separately.

    ``tier`` is present for provenance but is **one-hot encoded** by the
    blender, never consumed as an ordinal. It is also near-perfectly predictive
    of correctness for T0/T1, which is why those tiers bypass the blender
    entirely (see :attr:`~ledgerloop.models.enums.Tier.is_deterministic_certain`)
    and the model is fit only on the residual tiers.
    """

    tier: Tier

    # --- money space (exact) ---
    amount_delta_minor: MinorUnits = Field(
        default=0, description="Signed difference between the two sides, in minor units."
    )
    tolerance_band_minor: MinorUnits = Field(
        default=0, description="The band this candidate was judged against."
    )

    # --- feature space (float, never written back to money) ---
    amount_delta_ratio: float = Field(
        default=0.0, ge=0.0, description="abs(delta) / base; inf when base is zero."
    )
    date_delta_days: int = Field(
        default=0, description="Signed day gap between the two sides' dates."
    )
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reserved. Always 0.0 in the MVP -- ChromaDB is cut, and the "
        "semantic path is scheduled as an ablation row rather than a dependency.",
    )
    graph_support: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Strength of T4 constraint support."
    )
    subset_size: int = Field(
        default=0,
        ge=0,
        description="Payments in a T2 subset. Larger subsets are penalised: a "
        "parsimonious explanation is more likely to be the real one.",
    )
    llm_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="T5 self-reported confidence. A FEATURE, never a probability. "
        "Raw LLM confidence is systematically overconfident, so it is calibrated "
        "alongside every other signal rather than trusted directly.",
    )


class MatchCandidate(LedgerModel):
    """A proposed link, before the decision policy has ruled on it.

    Mutable because the pipeline fills it in progressively: a tier proposes it
    with features, the blender adds ``raw_score``, the calibrator adds
    ``calibrated_p``. The immutable record of what was decided is
    :class:`~ledgerloop.models.decisions.MatchDecision`.
    """

    candidate_id: str
    link_type: LinkType
    source_ref: RecordRef
    target_ref: RecordRef
    tier: Tier
    features: FeatureVector
    evidence: tuple[Evidence, ...] = ()

    subset_members: tuple[RecordRef, ...] = Field(
        default=(),
        description="For T2: the payments whose amounts compose the target credit.",
    )

    raw_score: float | None = Field(
        default=None, description="Blender output. None until the blender has run."
    )
    calibrated_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Isotonic-calibrated probability that this link is correct.",
    )
    arithmetic_verified: bool = Field(
        default=False,
        description="Whether verify_arithmetic() passed. A hard gate for LLM-proposed "
        "links: failure demotes to NEEDS_REVIEW rather than rejecting silently.",
    )

    is_truth_positive: bool | None = Field(
        default=None,
        description="Ground-truth label, populated ONLY when building a training or "
        "calibration set. Must be None during evaluation on the test split -- see "
        "the model validator below.",
    )

    @model_validator(mode="after")
    def _tier_consistency(self) -> MatchCandidate:
        if self.features.tier is not self.tier:
            raise ValueError(
                f"candidate tier {self.tier!r} disagrees with feature tier {self.features.tier!r}"
            )
        if self.subset_members and self.tier is not Tier.T2_AGGREGATION:
            raise ValueError("subset_members are only meaningful for T2 aggregation candidates")
        if self.tier is Tier.T2_AGGREGATION and self.features.subset_size != len(
            self.subset_members
        ):
            raise ValueError(
                f"subset_size {self.features.subset_size} disagrees with "
                f"{len(self.subset_members)} subset members"
            )
        return self

    @property
    def pair(self) -> tuple[str, str]:
        """Endpoint keys, comparable against :attr:`GroundTruth.evaluation_pairs`."""
        return (self.source_ref.key, self.target_ref.key)

    @property
    def is_evaluable(self) -> bool:
        """Whether this candidate is scored by the headline metrics.

        Only ``PAYMENT_CREDITED_AS`` links count -- ARCHITECTURE.md §2.
        """
        return self.link_type is LinkType.PAYMENT_CREDITED_AS
