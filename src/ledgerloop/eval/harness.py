"""One scored system run over one dataset directory.

Step 10 needs the full pipeline run many times -- six ablation rows, five seeds,
three difficulties -- and every one of those runs has to be the *same* run the
headline reports, or the ablation is measuring a second implementation. So the
sequence the CLI used to inline (ingest, optional narration repair, the ladder,
the exception classifier, bounded resolution, optional prose, scoring) moves
here and the CLI calls it once for the headline like everybody else.

WHERE GROUND TRUTH IS ALLOWED TO ENTER
--------------------------------------
Exactly twice, both **after** every decision the run makes:

1. :func:`~ledgerloop.eval.metrics.evaluate`, to score the finished predictions.
2. :func:`~ledgerloop.eval.reliability.measure_calibration`, to attach labels to
   probabilities the run already produced.

The second is why the ordering in :func:`run_system` is what it is.
``measure_calibration`` writes ``is_truth_positive`` onto the candidate objects
in place, and those same objects are handed to the exception classifier. Running
the measurement first would put ground-truth labels inside the classifier's
input -- not a leak today, because the classifier does not read the field, but a
leak that would arrive silently the day someone added a rule that did. The
classifier therefore runs first, unconditionally, and there is a test that fails
if it ever starts consulting the label.

WHAT VARIES BETWEEN ROWS, AND WHAT MUST NOT
-------------------------------------------
``enabled_tiers`` and the client vary. The tolerances, the lexical gates, the
graph parameters, the severity bands, the resolution bounds and the fitted
threshold do **not**: they come from the same ``RunConfig`` defaults and the
same bundle for every row, so a difference between two ablation rows is the
tier that was switched off and nothing else. ``RunSummary.config_hash`` records
the configuration each row actually ran, so that claim is checkable rather than
asserted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ledgerloop.config import (
    DuplicateDetection,
    LLMConfig,
    RunConfig,
    SplitCompletion,
)
from ledgerloop.eval.metrics import ExceptionCoverage, evaluate, exception_coverage
from ledgerloop.eval.reliability import (
    CalibrationEvaluation,
    measure_calibration,
    score_contenders,
)
from ledgerloop.eval.summary import RunSummary, summarise
from ledgerloop.eval.truth_io import DatasetManifest, load_ground_truth, load_manifest
from ledgerloop.exceptions import classify_exceptions, mark_resolvable, resolve_bounded
from ledgerloop.exceptions.resolver import ResolutionOutcome
from ledgerloop.ingest import ingest_dataset
from ledgerloop.ingest.dataset import IngestResult, ingest_available
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.integration import (
    LLMRunSummary,
    adjudicator_for,
    explain_queue,
    repair_narrations,
)
from ledgerloop.llm.tasks import AdjudicationOutcome, ExplanationOutcome, NarrationOutcome
from ledgerloop.matching import run_matching
from ledgerloop.matching.calibration import CalibrationBundle, configure_for
from ledgerloop.matching.harvest import harvest
from ledgerloop.matching.pipeline import MatchRun
from ledgerloop.models.enums import DecisionOutcome, LinkType
from ledgerloop.models.metrics import CostLedger, RunMetrics
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.resolution import AutoResolution
from ledgerloop.models.truth import GroundTruth

__all__ = [
    "DEFAULT_TIERS",
    "DETERMINISTIC_TIERS",
    "RunSetup",
    "StaleCalibrationError",
    "SystemRun",
    "assemble_system_run",
    "load_bundle_for",
    "prepare_run",
    "run_system",
]

#: The full ladder, T5 included. What the headline configuration runs when a
#: model is reachable.
DEFAULT_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: The full ladder without T5. What ``--no-llm`` and a machine with no key run,
#: and what every ablation row below the last one runs.
DETERMINISTIC_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4)


class StaleCalibrationError(ValueError):
    """The bundle was fitted against a different generator than the data."""


def load_bundle_for(path: Path, manifest: DatasetManifest) -> CalibrationBundle:
    """Load a fitted bundle and refuse it if it does not describe this corpus.

    A probability fitted against generator ``0.2.0`` is not a probability about
    ``0.3.0`` data. The bundle records the version precisely so a later run can
    refuse a stale one rather than quietly applying it.
    """
    bundle = CalibrationBundle.load(path)
    if bundle.provenance.generator_version != manifest.generator_version:
        raise StaleCalibrationError(
            f"calibration bundle was fitted on generator "
            f"{bundle.provenance.generator_version} but this dataset is "
            f"{manifest.generator_version}; a probability fitted on one is not "
            f"a probability about the other"
        )
    return bundle


@dataclass(frozen=True)
class SystemRun:
    """Everything one scored run of the production pipeline produced."""

    directory: Path
    manifest: DatasetManifest
    truth: GroundTruth
    config: RunConfig
    ingest: IngestResult
    matched: MatchRun
    metrics: RunMetrics
    exceptions: tuple[ReconException, ...]
    coverage: ExceptionCoverage
    resolutions: tuple[AutoResolution, ...]
    rounding_spent_minor: int
    calibration: CalibrationEvaluation | None
    cost: CostLedger
    llm: LLMRunSummary
    llm_available: bool

    @property
    def label(self) -> str:
        return self.matched.name

    @property
    def candidates_proposed(self) -> int:
        """Candidate yield on the evaluation unit, before the policy ruled.

        Restricted to ``PAYMENT_CREDITED_AS`` so it shares a denominator with
        precision and recall. Counting the structural ``ORDER_PAID_BY`` and
        intermediate ``SETTLEMENT_CREDITED_AS`` edges -- 283 and 35 of them on
        `test` -- would make the yield column incomparable to every column
        beside it.
        """
        return self.matched.evaluable_candidates

    @property
    def auto_matched(self) -> int:
        """Conviction on the same unit: links the policy actually committed.

        ``MatchRun.auto_matched`` counts every auto-matched decision, structural
        edges included. Reporting that beside a yield restricted to the
        evaluation unit would put a numerator and a denominator from two
        different populations in adjacent columns.
        """
        return len(self.matched.predictions)

    @property
    def needs_review(self) -> int:
        """Evaluation-unit links the policy referred to a human."""
        return sum(
            1
            for decision in self.matched.decisions
            if decision.link_type is LinkType.PAYMENT_CREDITED_AS
            and decision.outcome is DecisionOutcome.NEEDS_REVIEW
        )

    def summary(self) -> RunSummary:
        """The row this run contributes to an ablation or multi-seed table."""
        return summarise(
            self.label,
            self.metrics,
            split=self.manifest.split,
            difficulty=self.manifest.difficulty,
            seed=self.manifest.seed,
            config_hash=self.config.config_hash,
            tuning_hash=self.config.tuning_hash,
            candidates_proposed=self.candidates_proposed,
            auto_matched=self.auto_matched,
            needs_review=self.needs_review,
            exceptions_raised=len(self.exceptions),
            exceptions_expected=len(self.coverage.expected),
            cost=self.cost,
            llm_available=self.llm_available,
        )


@dataclass(frozen=True)
class RunSetup:
    """What a run needs before any node executes: the corpus and the config.

    Split out of :func:`run_system` at Step 11 so the LangGraph assembly and the
    direct call resolve their inputs the same way. Which tiers run and which
    ``run_id`` is stamped are decided **once**, here, rather than twice with a
    chance of disagreeing.
    """

    directory: Path
    manifest: DatasetManifest
    truth: GroundTruth
    config: RunConfig
    tiers: tuple[int, ...]
    llm_ready: bool

    @property
    def llm_active(self) -> bool:
        """Whether the model is both reachable and inside the enabled ladder."""
        return self.llm_ready and 5 in self.tiers


def prepare_run(
    directory: Path,
    *,
    bundle: CalibrationBundle | None = None,
    client: LLMClient | None = None,
    enabled_tiers: tuple[int, ...] | None = None,
    duplicates: DuplicateDetection | None = None,
    split_completion: SplitCompletion | None = None,
    run_id: str | None = None,
) -> RunSetup:
    """Read the corpus and settle the configuration. No matching happens here.

    ``enabled_tiers`` drives the ablation. ``None`` means the full ladder, with
    T5 included only when ``client`` is enabled -- a config listing T5 on a
    machine with no key would otherwise report a tier that never ran.

    ``split_completion`` is the Phase 2.5 switch and works exactly like
    ``duplicates``: it travels in ``RunConfig.tuning_hash``, so a comparison
    table can witness that its two arms differ in that field and nothing else.

    ``duplicates`` drives the Phase 2.3 before/after comparison the same way:
    passing ``DuplicateDetection(enabled=False)`` reproduces every pre-Phase-2
    number exactly, and the value travels in ``RunConfig.tuning_hash`` so a
    comparison table can witness that nothing else moved between its two arms.
    """
    manifest = load_manifest(directory)
    truth = load_ground_truth(directory)
    tag = f"{manifest.split.value}-{manifest.seed}"

    llm_ready = client is not None and client.enabled
    requested = DEFAULT_TIERS if enabled_tiers is None else tuple(enabled_tiers)
    tiers = requested if llm_ready else tuple(t for t in requested if t != 5)

    llm_config = client.config if client is not None else LLMConfig(enabled=False)
    config = RunConfig(
        run_id=run_id or f"{_ladder_tag(tiers)}-{tag}",
        split=manifest.split,
        difficulty=manifest.difficulty,
        seed=manifest.seed,
        enabled_tiers=tiers,
        llm=llm_config,
        duplicates=duplicates if duplicates is not None else DuplicateDetection(),
        split_completion=(
            split_completion if split_completion is not None else SplitCompletion()
        ),
    )
    if bundle is not None:
        config = configure_for(config, bundle)
    return RunSetup(
        directory=directory,
        manifest=manifest,
        truth=truth,
        config=config,
        tiers=tiers,
        llm_ready=llm_ready,
    )


def assemble_system_run(
    setup: RunSetup,
    *,
    ingested: IngestResult,
    matched: MatchRun,
    exceptions: Sequence[ReconException],
    resolutions: ResolutionOutcome,
    calibration: CalibrationEvaluation | None,
    cost: CostLedger,
    llm: LLMRunSummary,
) -> SystemRun:
    """Score a finished run and package it. **The only place a run is scored.**

    Both the direct path and the LangGraph assembly end here, so a metric can
    never differ between them by construction rather than by a test that
    happens to pass. The test asserting the two are identical is still there;
    this is what makes it cheap to keep true.
    """
    metrics = evaluate(
        matched.predictions,
        setup.truth,
        run_id=setup.config.run_id,
        wall_clock_ms=matched.wall_clock_ms,
        tier_contributions=matched.tier_contributions,
        exceptions=exceptions,
        out_of_scope_refs=matched.out_of_scope_refs,
    )
    if calibration is not None:
        metrics.calibration = calibration.asserted.metrics()
    metrics.cost = cost

    return SystemRun(
        directory=setup.directory,
        manifest=setup.manifest,
        truth=setup.truth,
        config=setup.config,
        ingest=ingested,
        matched=matched,
        metrics=metrics,
        exceptions=tuple(exceptions),
        coverage=exception_coverage(
            exceptions, setup.truth, out_of_scope=matched.out_of_scope_refs
        ),
        resolutions=resolutions.resolutions,
        rounding_spent_minor=resolutions.rounding_spent_minor,
        calibration=calibration,
        cost=cost,
        llm=llm,
        llm_available=setup.llm_active,
    )



@dataclass(frozen=True)
class ReconcileResult:
    """One reconciliation over files nobody has an answer key for.

    ``run_system`` scores; this does not, and cannot. Precision, recall and the
    match rate are all computed against ``GroundTruth.evaluation_pairs``, and a
    real bank statement does not come with one -- producing it means a person
    reconciling the corpus by hand, which is the work this tool automates.

    So the figures here are all statements about what the system **did**, never
    about whether it was right: how many links it committed, what they were
    worth, and what it refused and sent to a person instead. Those are the
    honest numbers for uploaded data, and they are the ones the upload screen
    shows.
    """

    ingest: IngestResult
    matched: MatchRun
    exceptions: tuple[ReconException, ...]
    llm_calls: int
    llm_used: bool
    """Whether a model actually answered -- **calls made**, not a key existing.

    Read from the run's own cost ledger. A dashboard that reported "LLM used"
    because a credential was present would be claiming something it had not
    observed, and it is the one claim that would undo the deterministic result.
    """

    @property
    def committed_links(self) -> int:
        """Payment-to-bank links the policy committed without a human."""
        return len(self.matched.predictions)

    @property
    def committed_minor(self) -> int:
        """What those links were claimed to be worth, in paise.

        Summed from the predictions themselves. Not compared with anything,
        because there is nothing to compare it with.
        """
        return sum(link.amount_minor for link in self.matched.predictions)

    @property
    def queue_size(self) -> int:
        return len(self.exceptions)

    @property
    def queue_minor(self) -> int:
        return sum(item.impact_minor for item in self.exceptions)


def reconcile_only(
    directory: Path,
    *,
    config: RunConfig | None = None,
    bundle: CalibrationBundle | None = None,
    client: LLMClient | None = None,
) -> ReconcileResult:
    """Ingest, match and classify a directory that has **no ground truth**.

    The path a person's own files take. It is the same sequence ``run_system``
    runs, calling the same functions in the same order, and it stops where the
    scorer would begin::

        ingest_available -> [repair narrations] -> run_matching
                         -> classify_exceptions -> resolve_bounded

    Two differences from ``run_system``, both forced by the absence of an answer
    key rather than chosen:

    * :func:`~ledgerloop.ingest.dataset.ingest_available` instead of
      ``ingest_dataset``, because a person may not have all three sources.
    * no ``prepare_run``: that loads the manifest and the ground truth, and
      neither exists here. The config is supplied or defaulted instead.

    **Nothing about matching changes.** The tiers, the thresholds, the guards and
    the refusals are the objects ``run_system`` uses, reached through the same
    call. A corpus that does have truth still goes through ``run_system``; this
    is not a second pipeline, it is the same one with the scorer left off.
    """
    run_config = config or RunConfig(run_id="uploaded")
    ingested = ingest_available(directory)

    llm_ready = client is not None and client.enabled
    if llm_ready and 5 in run_config.enabled_tiers:
        assert client is not None
        ingested, _ = repair_narrations(
            client, ingested, batch_size=run_config.llm.narration_batch_size
        )

    adjudicator = None
    if llm_ready and 5 in run_config.enabled_tiers:
        assert client is not None
        adjudicator = adjudicator_for(client, run_config)

    matched = run_matching(
        ingested, run_config, bundle=bundle, adjudicator=adjudicator
    )
    assert matched.context is not None  # run_matching always sets it
    queue = classify_exceptions(
        matched.context,
        matched.decisions,
        matched.candidates,
        run_config,
        merchant_profiles=matched.merchant_spellings,
    )
    resolutions = resolve_bounded(queue.exceptions, run_config.auto_resolution)
    exceptions = tuple(mark_resolvable(queue.exceptions, resolutions))

    # Observed, not permitted. `client.enabled` says a ladder was built; the
    # cost ledger says a model answered.
    calls = client.ledger().llm_calls if client is not None else 0
    return ReconcileResult(
        ingest=ingested,
        matched=matched,
        exceptions=exceptions,
        llm_calls=calls,
        llm_used=calls > 0,
    )


def run_system(
    directory: Path,
    *,
    bundle: CalibrationBundle | None = None,
    client: LLMClient | None = None,
    enabled_tiers: tuple[int, ...] | None = None,
    duplicates: DuplicateDetection | None = None,
    split_completion: SplitCompletion | None = None,
    run_id: str | None = None,
    measure_calibration_quality: bool = True,
) -> SystemRun:
    """Ingest, match, classify, resolve and score one dataset directory.

    ``enabled_tiers`` drives the ablation. ``None`` means the full ladder, with
    T5 included only when ``client`` is enabled -- a config listing T5 on a
    machine with no key would otherwise report a tier that never ran.

    ``measure_calibration_quality`` is off for ablation and sweep rows. It
    harvests top-k contenders over the whole corpus purely to produce the
    contender reliability diagram, which costs roughly as much as the run
    itself and answers a question about the *calibrator* rather than about the
    row. The headline run measures it; the rows that would only re-measure it
    do not.
    """
    setup = prepare_run(
        directory,
        bundle=bundle,
        client=client,
        enabled_tiers=enabled_tiers,
        duplicates=duplicates,
        split_completion=split_completion,
        run_id=run_id,
    )
    config, tiers, truth = setup.config, setup.tiers, setup.truth
    llm_ready = setup.llm_ready
    llm_config = config.llm

    ingested = ingest_dataset(directory, strict=False)

    # Call site 1, before matching: a narration the regex layer could not read.
    # Gated on T5 as well as on the client, because an ablation row that ran the
    # deterministic ladder while an LLM quietly repaired its inputs would credit
    # the deterministic tiers with the model's contribution.
    narration = NarrationOutcome()
    if llm_ready and 5 in tiers:
        assert client is not None  # llm_ready implies a client
        ingested, narration = repair_narrations(
            client, ingested, batch_size=llm_config.narration_batch_size
        )

    adjudicator = None
    if llm_ready and 5 in tiers:
        assert client is not None  # llm_ready implies a client
        adjudicator = adjudicator_for(client, config)
    matched = run_matching(ingested, config, bundle=bundle, adjudicator=adjudicator)

    adjudication = matched.adjudication
    if not isinstance(adjudication, AdjudicationOutcome):
        adjudication = AdjudicationOutcome()

    # The exception queue, built from the sources and the run's own decisions.
    # Ground truth is not an input to it, and -- see the module docstring -- it
    # is built BEFORE any labelling touches the candidate objects.
    assert matched.context is not None  # run_matching always sets it
    queue = classify_exceptions(
        matched.context,
        matched.decisions,
        matched.candidates,
        config,
        merchant_profiles=matched.merchant_spellings,
    )
    resolutions = resolve_bounded(queue.exceptions, config.auto_resolution)
    exceptions = mark_resolvable(queue.exceptions, resolutions)

    # Call site 3. The class, the severity and the money are already decided and
    # are not sent back for revision; only the prose can change.
    explanation = ExplanationOutcome(exceptions=tuple(exceptions))
    if llm_ready and 5 in tiers:
        assert client is not None
        exceptions, explanation = explain_queue(client, exceptions)

    calibration_view: CalibrationEvaluation | None = None
    if bundle is not None and measure_calibration_quality:
        contenders = score_contenders(bundle, harvest(ingested, truth, config).rows)
        calibration_view = measure_calibration(
            matched.candidates,
            truth,
            contender_probabilities=contenders.probabilities,
            contender_labels=contenders.labels,
        )

    cost = client.ledger() if client is not None else CostLedger()
    return assemble_system_run(
        setup,
        ingested=ingested,
        matched=matched,
        exceptions=exceptions,
        resolutions=resolutions,
        calibration=calibration_view,
        cost=cost,
        llm=LLMRunSummary(
            narration=narration, adjudication=adjudication, explanation=explanation
        ),
    )


def _ladder_tag(tiers: tuple[int, ...]) -> str:
    """``t0t4`` -- the run-id fragment naming the ladder that ran."""
    if not tiers:  # pragma: no cover - RunConfig refuses an empty ladder
        return "none"
    return f"t{tiers[0]}t{tiers[-1]}"
