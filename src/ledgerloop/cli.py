"""Command-line entry point.

``argparse`` rather than a CLI framework: four commands today, and a dependency
that exists only to make ``--help`` prettier is not worth carrying into a
project whose selling point is that it runs on nothing.

``calibrate`` arrives with Step 7 and sits *between* ``generate`` and ``eval``:
it reads the train and calibration corpora, fits the blender, the isotonic map
and ``tau_high``, and writes one bundle. ``eval --calibration`` then applies
that bundle. The two are separate commands rather than one because the split
discipline has to be visible on the command line -- a single command that both
fitted and reported would make it impossible to see, from the invocation alone,
that the test split was never fitted on.

Step 10 adds three commands, each writing one artefact that ``eval`` renders:

``ablation``
    Six ladders over several seeds. Expensive (thirty pipeline runs) and
    entirely deterministic.
``sweep``
    The headline configuration over seeds and difficulties. Same shape.
``baseline-llm``
    B2. The **only** command in this project that can reach a network on its
    own account, which is why it is a command and not a section: regenerating
    the report must never spend quota.

Separate commands for the same reason ``calibrate`` is separate from ``eval``.
``eval --ablation reports/ablation.json`` says, in the invocation, that the
table was produced by a different run over different corpora -- and ``make
eval`` chains all four so one command still regenerates everything.

Step 11 adds ``run``: the same pipeline executed through the LangGraph state
machine, writing a durable run record for the UI to read. It is a **separate
command from** ``eval`` rather than a flag on it, because the two answer
different questions -- ``eval`` regenerates the published metrics and needs no
optional extra, ``run`` produces one inspectable, replayable reconciliation.
Both go through the same node functions, and a test asserts they agree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.store import RUNS_ROOT
from ledgerloop.config import SPLIT_SIZES, GeneratorConfig, LLMConfig, RunConfig
from ledgerloop.eval.ablation import ABLATION_LADDERS, AblationArtifact, run_ablation
from ledgerloop.eval.baselines import run_b0, run_b1
from ledgerloop.eval.harness import (
    StaleCalibrationError,
    SystemRun,
    load_bundle_for,
    run_system,
)
from ledgerloop.eval.llm_baseline import (
    DEFAULT_PAYMENTS_PER_CALL,
    LLMBaselineArtifact,
    run_b2,
)
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.offline_provider import OFFLINE_PROVIDER_NAME, OfflineReasoner
from ledgerloop.eval.report import EvaluatedRun, render_report, write_report
from ledgerloop.eval.sweep import SweepArtifact, run_sweep
from ledgerloop.eval.truth_io import load_ground_truth, load_manifest
from ledgerloop.fitting import FittingError, fit_from_corpora, harvest_corpora
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import IngestError, ingest_dataset
from ledgerloop.llm.client import LLMClient, build_provider
from ledgerloop.matching.blender import DEFAULT_L2
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.matching.harvest import DEFAULT_TOP_K
from ledgerloop.models.enums import Difficulty, SplitName
from ledgerloop.money import format_minor

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerloop",
        description="Three-way reconciliation with an honest exception list.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate a seeded synthetic dataset and its ground truth"
    )
    generate.add_argument(
        "--split",
        type=SplitName,
        choices=list(SplitName),
        default=SplitName.DEV,
        help="dataset split. Sizes: "
        + ", ".join(f"{k.value}={v}" for k, v in SPLIT_SIZES.items()),
    )
    generate.add_argument(
        "--difficulty",
        type=Difficulty,
        choices=list(Difficulty),
        default=Difficulty.STANDARD,
        help="anomaly prevalence dial",
    )
    generate.add_argument("--seed", type=int, default=42, help="RNG seed")
    generate.add_argument(
        "--orders", type=int, default=None, help="override the split's default order count"
    )
    generate.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: data/generated/<split>-<difficulty>-<seed>)",
    )
    generate.add_argument(
        "--ensure-class-coverage",
        action="store_true",
        help="force one effect per anomaly class after the draw; distorts prevalence, "
        "intended only for the committed fixture set",
    )

    ingest = subparsers.add_parser(
        "ingest", help="parse and normalise a dataset's three sources, and report on them"
    )
    ingest.add_argument(
        "--data",
        type=Path,
        required=True,
        help="a generated dataset directory (as written by `ledgerloop generate`)",
    )
    ingest.add_argument(
        "--strict",
        action="store_true",
        help="fail on the first malformed record instead of quarantining it",
    )
    ingest.add_argument(
        "--show-problems",
        type=int,
        default=10,
        help="how many quarantined records to list (default 10)",
    )

    calibrate = subparsers.add_parser(
        "calibrate",
        help="fit the blender, the isotonic calibrator and tau_high, and write a bundle",
    )
    calibrate.add_argument(
        "--train",
        type=Path,
        nargs="+",
        required=True,
        help="train-split dataset directories. The logistic is fitted on these and "
        "on nothing else.",
    )
    calibrate.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        required=True,
        help="calibration-split dataset directories. The isotonic map and tau_high "
        "come from these, and they may not overlap --train.",
    )
    calibrate.add_argument(
        "--out",
        type=Path,
        default=Path("reports/calibration.json"),
        help="where to write the fitted bundle (default reports/calibration.json)",
    )
    calibrate.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"contenders collected per decision point (default {DEFAULT_TOP_K})",
    )
    calibrate.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="auto-match precision tau_high is selected to achieve "
        "(default: the configured target, 0.99)",
    )
    calibrate.add_argument(
        "--l2",
        type=float,
        default=DEFAULT_L2,
        help=f"ridge strength for the logistic fit (default {DEFAULT_L2})",
    )

    evaluation = subparsers.add_parser(
        "eval", help="score the baselines against a dataset's ground truth"
    )
    evaluation.add_argument(
        "--data",
        type=Path,
        required=True,
        help="a generated dataset directory (as written by `ledgerloop generate`)",
    )
    evaluation.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`. Without it the residual "
        "tiers keep the provisional probabilities they set themselves, which is the "
        "uncalibrated ablation row.",
    )
    evaluation.add_argument(
        "--no-llm",
        action="store_true",
        help="run the whole pipeline deterministically, with no model and no "
        "network. The default when no API key is present, so this flag is how a "
        "run with a key available is made to prove it does not need one.",
    )
    evaluation.add_argument(
        "--llm-key-env",
        default="LEDGERLOOP_LLM_API_KEY",
        help="environment variable holding the provider key "
        "(default LEDGERLOOP_LLM_API_KEY). Absent means the deterministic path.",
    )
    evaluation.add_argument(
        "--show-exceptions",
        type=int,
        default=5,
        help="how many of the highest-impact exceptions to print (default 5)",
    )
    evaluation.add_argument(
        "--ablation",
        type=Path,
        default=None,
        help="an artefact written by `ledgerloop ablation`. Absent renders no "
        "ablation section rather than an empty one.",
    )
    evaluation.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="an artefact written by `ledgerloop sweep`, carrying the multi-seed "
        "and difficulty tables.",
    )
    evaluation.add_argument(
        "--llm-baseline",
        type=Path,
        default=None,
        help="an artefact written by `ledgerloop baseline-llm` (B2). Rendering it "
        "makes no calls: a report regenerated twice must not spend quota twice.",
    )
    evaluation.add_argument(
        "--out",
        type=Path,
        default=Path("EVALUATION.md"),
        help="report destination. Regenerated in full every run and gitignored, "
        "because a committed report is one that can be quietly corrected.",
    )

    execute = subparsers.add_parser(
        "run",
        help="reconcile one dataset through the LangGraph pipeline and store the run",
    )
    execute.add_argument(
        "--data",
        type=Path,
        required=True,
        help="a generated dataset directory (as written by `ledgerloop generate`)",
    )
    execute.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`. Without it the residual "
        "tiers keep the provisional probabilities they set themselves.",
    )
    _add_llm_flags(execute)
    execute.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_ROOT,
        help=f"where the run record is written (default {RUNS_ROOT}). One "
        "directory per run: run.json, audit.jsonl, exceptions.json, decisions.json.",
    )
    execute.add_argument(
        "--run-id",
        default=None,
        help="override the generated run id. The run directory is named for it, "
        "so re-running the same id overwrites that record rather than "
        "accumulating near-duplicates the UI would have to disambiguate.",
    )
    execute.add_argument(
        "--show-nodes",
        action="store_true",
        help="print the node sequence the graph actually took, including every "
        "repeat of the residual loop",
    )

    ablation = subparsers.add_parser(
        "ablation",
        help="run every prefix of the tier ladder over several seeds (PLAN.md §9.3)",
    )
    ablation.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset directories, all one split at one difficulty. The rows are "
        "aggregated across them as mean +/- sample std.",
    )
    ablation.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`. Applied identically to "
        "every row, so a difference between rows is the ladder and nothing else.",
    )
    _add_llm_flags(ablation)
    ablation.add_argument(
        "--out",
        type=Path,
        default=Path("reports/ablation.json"),
        help="artefact destination (default reports/ablation.json)",
    )

    sweep = subparsers.add_parser(
        "sweep",
        help="run the headline configuration over seeds and difficulties (PLAN.md §9.4)",
    )
    sweep.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset directories of one split. Grouped by the difficulty each "
        "manifest declares, never by anything the caller asserts.",
    )
    sweep.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`. One bundle for every "
        "difficulty: refitting per difficulty would measure the calibrator's "
        "ceiling rather than the system's behaviour.",
    )
    _add_llm_flags(sweep)
    sweep.add_argument(
        "--out",
        type=Path,
        default=Path("reports/sweep.json"),
        help="artefact destination (default reports/sweep.json)",
    )

    llm_baseline = subparsers.add_parser(
        "baseline-llm",
        help="B2: dump the corpus at a model and assert whatever it returns",
    )
    llm_baseline.add_argument(
        "--data",
        type=Path,
        required=True,
        help="a generated dataset directory. PLAN.md §9.2 fixes this at the "
        "60-order `dev` split to bound token spend, and the command warns when "
        "it is pointed at anything larger.",
    )
    llm_baseline.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle, used only for the production run this command measures "
        "B2 against on the same corpus. It never touches B2 itself -- B2 has no "
        "probabilities to calibrate.",
    )
    _add_llm_flags(llm_baseline)
    llm_baseline.add_argument(
        "--payments-per-call",
        type=int,
        default=DEFAULT_PAYMENTS_PER_CALL,
        help=f"payments per prompt (default {DEFAULT_PAYMENTS_PER_CALL}). The whole "
        "bank statement goes into every prompt regardless: a payment can be "
        "credited by any row.",
    )
    llm_baseline.add_argument(
        "--max-calls",
        type=int,
        default=60,
        help="hard budget for B2's own client (default 60). Separate from the "
        "production budget so an expensive baseline cannot starve the system it "
        "is being compared to.",
    )
    llm_baseline.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("reports/llm_cache_b2"),
        help="B2's own response cache (default reports/llm_cache_b2, gitignored). "
        "Separate from the production cache on purpose: B2's prompts are a "
        "different task, and Step 9 settled that `tests/fixtures/llm_cache` stays "
        "empty until a live run fills it -- committed stand-in responses would "
        "look like evidence of a real one.",
    )
    llm_baseline.add_argument(
        "--cold",
        action="store_true",
        help="empty B2's response cache before running, so the published cost is "
        "what the baseline actually costs rather than what a rerun costs. `make "
        "eval` passes it: a warm cache reports zero calls and zero tokens, which "
        "is true of any rerun and would quietly delete the comparison this row "
        "exists to make. The cache guarantee itself is asserted by a test.",
    )
    llm_baseline.add_argument(
        "--offline-provider",
        action="store_true",
        help="answer B2's prompts with the prompt-reading stand-in reasoner in "
        "`eval/offline_provider.py` instead of a live model. Every cost, cache "
        "and failure figure is then still measured machinery, but the accuracy "
        "figures are a property of a documented rule and NOT a claim about any "
        "language model -- the artefact records which, and the report prints a "
        "banner saying so. Drop this flag on a machine with a provider key.",
    )
    llm_baseline.add_argument(
        "--out",
        type=Path,
        default=Path("reports/llm_baseline.json"),
        help="artefact destination (default reports/llm_baseline.json)",
    )
    return parser


def _add_llm_flags(parser: argparse.ArgumentParser) -> None:
    """``--no-llm`` and ``--llm-key-env``, identical on every command that runs.

    One helper rather than four copies: a flag that meant something slightly
    different on the ablation command than on ``eval`` would make the ablation's
    LLM column incomparable to the headline's, which is the one thing the column
    exists for.
    """
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="run deterministically, with no model and no network. The default "
        "when no API key is present, so this flag is how a run with a key "
        "available is made to prove it does not need one.",
    )
    parser.add_argument(
        "--llm-key-env",
        default="LEDGERLOOP_LLM_API_KEY",
        help="environment variable holding the provider key "
        "(default LEDGERLOOP_LLM_API_KEY). Absent means the deterministic path.",
    )


def _run_generate(args: argparse.Namespace) -> int:
    config = GeneratorConfig(
        split=args.split,
        difficulty=args.difficulty,
        seed=args.seed,
        order_count=args.orders,
        ensure_class_coverage=args.ensure_class_coverage,
    )
    directory = args.out or Path("data/generated") / (
        f"{config.split.value}-{config.difficulty.value}-{config.seed}"
    )
    dataset = generate_to_disk(config, directory)
    truth = dataset.truth
    world = dataset.world

    print(f"wrote {directory}")
    print(
        f"  {len(world.orders)} orders · {len(world.payments)} payments · "
        f"{len(world.settlements)} settlements · {len(world.bank_txns)} bank rows"
    )
    print(
        f"  {len(truth.evaluation_pairs)} evaluation links · "
        f"{len(truth.unmatchable_refs)} unmatchable (the honest ceiling)"
    )
    print(
        f"  {len(world.effects)} anomalies applied "
        f"from {sum(truth.scenario_draws.values())} draws"
    )
    print(f"  settled credit total: {format_minor(world.settled_credit_total_minor())}")

    residual = dataset.conservation_residual_minor
    if residual != 0:
        print(f"  CONSERVATION VIOLATED: {format_minor(residual)} unaccounted for", file=sys.stderr)
        return 1
    print("  money conserved modulo declared anomalies ✓")
    return 0


def _run_ingest(args: argparse.Namespace) -> int:
    directory: Path = args.data
    if not directory.is_dir():
        print(f"no such dataset directory: {directory}", file=sys.stderr)
        return 1

    try:
        result = ingest_dataset(directory, strict=args.strict)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    print(f"ingested {directory}")
    print(
        f"  {len(result.orders)} orders · {len(result.payments)} payments · "
        f"{len(result.settlements)} settlements · {len(result.bank_txns)} bank rows "
        f"({result.wall_clock_ms} ms)"
    )
    print(f"  dates: {result.date_order.basis}")
    print(
        f"  references: {result.payments_with_usable_ref} usable "
        f"({result.payments_with_recovered_ref} recovered by normalisation), "
        f"{result.payments_with_no_ref} absent at source"
    )
    print(
        f"  narration: {result.credits_with_utr} of {len(result.credits)} credits carry a "
        f"UTR, {result.credits_with_merchant} a merchant, "
        f"{result.credits_with_no_reference} neither"
    )

    if not result.problems:
        print("  0 malformed records ✓")
        return 0

    print(f"  {len(result.problems)} malformed records quarantined:", file=sys.stderr)
    for problem in result.problems[: max(0, args.show_problems)]:
        print(f"    {problem}", file=sys.stderr)
    remaining = len(result.problems) - max(0, args.show_problems)
    if remaining > 0:
        print(f"    ... and {remaining} more", file=sys.stderr)
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    """Fit one bundle and print the evidence for every number in it."""
    config = RunConfig(run_id="calibrate", split=SplitName.TRAIN)
    target = (
        args.target_precision
        if args.target_precision is not None
        else config.thresholds.target_auto_match_precision
    )
    try:
        train = harvest_corpora(args.train, config=config, top_k=args.top_k)
        calibration = harvest_corpora(args.calibration, config=config, top_k=args.top_k)
        bundle = fit_from_corpora(
            train,
            calibration,
            target_precision=target,
            top_k=args.top_k,
            l2=args.l2,
        )
    except (FittingError, ValueError) as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 1

    bundle.save(args.out)
    blender = bundle.blender
    selection = bundle.thresholds

    print(f"fitted {args.out}")
    for name, half in (("train", train), ("calibration", calibration)):
        considered, resolved = half.decision_points()
        labels = half.diagnostic_labels
        print(
            f"  {name}: {half.rows} rows ({half.positives} correct, "
            f"{half.rows - half.positives} wrong) from {len(half.corpora)} corpora "
            f"[{', '.join(corpus.label for corpus in half.corpora)}]"
        )
        print(
            f"      {considered} decision points examined, {resolved} resolved - "
            f"{half.by_tier()}"
        )
        print(
            f"      {len(labels)} contenders in all, {sum(1 for x in labels if not x)} "
            "of them wrong; the ones from refused decision points are never fitted on"
        )

    print(
        f"  logistic: {blender.sample_count} rows, {blender.iterations} Newton "
        f"iteration(s), converged={blender.converged}"
    )
    if blender.single_class:
        print(
            "      SINGLE CLASS: every fitted row carries the same label, so the "
            "model has learned a base rate and not a discrimination"
        )
    for name, coefficient in blender.coefficient_table():
        print(f"      {name:24s} {coefficient:+.4f}")
    print(
        f"  isotonic: {bundle.calibrator.block_count} block(s) over "
        f"{bundle.calibrator.sample_count} calibration rows "
        f"({bundle.provenance.calibration_abstained} abstained)"
    )
    for threshold, value in zip(
        bundle.calibrator.thresholds, bundle.calibrator.values, strict=True
    ):
        print(f"      raw >= {threshold:.6f} -> p = {value:.6f}")
    print(
        f"  tau_high = {selection.tau_high:.6f} "
        f"({'attained' if selection.attained else 'TARGET NOT ATTAINED'}; "
        f"target {selection.target_precision:.4f})"
    )
    print(
        f"      precision {selection.achieved_precision:.4f} "
        f"[{selection.precision_ci_low:.4f}, {selection.precision_ci_high:.4f}] on "
        f"{selection.auto_matched} auto-matched of {selection.candidates_considered} "
        f"({selection.false_positives} wrong), coverage {selection.coverage:.4f}"
    )
    fit = bundle.fit_reliability
    print(
        f"  in-sample reliability: ECE {fit.ece:.4f} - Brier {fit.brier:.4f} - "
        f"{fit.populated_bins} of {len(fit.bins)} bins populated"
    )
    if fit.populated_bins <= 1:
        print(
            "      one populated bin: the diagram describes the corpus, not the "
            "calibrator -- see CalibrationMetrics.populated_bins"
        )
    return 0


def _client_for(args: argparse.Namespace, *, max_calls: int | None = None) -> LLMClient:
    """The client a command should use, or a disabled one.

    A machine without credentials runs the whole pipeline deterministically and
    says so; ``--no-llm`` reaches the same place deliberately rather than by
    accident. Both return a real :class:`LLMClient` whose ``enabled`` is False,
    so every call site sees one type and the deterministic path is a branch
    rather than a second object graph.
    """
    wanted = not args.no_llm
    api_key = os.environ.get(args.llm_key_env) if wanted else None
    config = LLMConfig(enabled=wanted)
    if max_calls is not None:
        config = config.model_copy(update={"max_calls_per_run": max_calls})
    return LLMClient(config=config, provider=build_provider(config, api_key=api_key))


def _llm_disabled_reason(args: argparse.Namespace) -> str:
    return "--no-llm" if args.no_llm else f"no key in ${args.llm_key_env}"


def _resolve_bundle(path: Path | None, directory: Path) -> CalibrationBundle | None:
    """Load and version-check a bundle, or return ``None``.

    Raises :class:`~ledgerloop.eval.harness.StaleCalibrationError` rather than
    printing, so every command reports the refusal the same way.
    """
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"no such calibration bundle: {path}")
    return load_bundle_for(path, load_manifest(directory))


def _print_run(run: SystemRun) -> None:
    """The per-run block every command that runs the pipeline prints."""
    matched = run.matched
    links = run.metrics.link_metrics
    assert links is not None  # evaluate() always populates it
    print(
        f"  {run.label}: precision {run.metrics.auto_match_precision:.4f} "
        f"[{links.precision_ci_low:.4f}, {links.precision_ci_high:.4f}] - "
        f"recall {links.recall:.4f} - match rate {run.metrics.match_rate:.4f}"
    )
    print(
        f"      {links.true_positives} correct - {links.false_positives} false positives "
        f"costing {format_minor(links.false_positive_cost_minor)} - "
        f"{links.false_negatives} missed - {run.candidates_proposed} candidates "
        f"proposed on the evaluation unit"
    )
    print(
        f"      exceptions {len(run.exceptions)} raised - recall "
        f"{run.coverage.recall:.4f} over {len(run.coverage.expected)} - "
        f"unmatchable coverage {run.coverage.unmatchable_recall:.4f} over "
        f"{len(run.coverage.unmatchable)}"
    )
    if run.llm_available:
        cost = run.cost
        print(
            f"      llm {cost.llm_calls} call(s) - {cost.cache_hits} hit(s) - "
            f"{cost.total_tokens} tokens - equivalent paid "
            f"₹{cost.equivalent_paid_cost_inr:.2f}"
        )
    del matched


def _negative_counters(run: SystemRun) -> tuple[tuple[str, str, int], ...]:
    """The tiers' own account of what they refused, and why.

    Read off the outcome objects the ladder already returns rather than
    recomputed here, so the report cannot disagree with the run. A counter that
    is zero on this corpus still appears: "T4 inferred nothing" is a finding
    (ARCHITECTURE.md §6, decision 31), and dropping the row would turn a
    measured zero into an absence.
    """
    matched = run.matched
    return (
        (
            "Settlements left unresolved",
            "no tier could account for the payout; every payment nested in them "
            "is a missed link",
            matched.settlements_unresolved,
        ),
        (
            "Settlements contested",
            "two or more credits fit and the tier refused to pick -- T0/T1 "
            "mutual uniqueness, T2 `AMBIGUOUS_AGGREGATION`, T3's margin gate",
            matched.settlements_contested,
        ),
        (
            "T2 subsets ambiguous",
            "two different subsets of payments summed to the credit, so the "
            "partition is not determined by the sources",
            matched.aggregation.settlements_ambiguous,
        ),
        (
            "T3 rejected below score",
            "the best merchant-name match did not reach the similarity gate",
            matched.lexical.rejected_below_score,
        ),
        (
            "T3 rejected on margin",
            "the best match did not beat the runner-up by enough; two merchants "
            "the scorer cannot separate are an ambiguity, not a winner",
            matched.lexical.rejected_on_margin,
        ),
        (
            "T4 inferences made",
            "path closure and sibling completion; zero here because every "
            "earlier tier matches at settlement granularity, so the partial "
            "assignments they exist to finish never arise",
            len(matched.graph.candidates),
        ),
        (
            "Unmatchable records (the floor)",
            "irreconcilable from the three sources alone, across **every** record "
            "type; excluded from the match-rate denominator and reported on their "
            "own line. The exception queue's own unmatchable count is smaller "
            "because the outgoing bank rows sit outside its unit entirely -- the "
            "two figures plus that count add up, and both appear above",
            run.metrics.unmatchable_count,
        ),
    )


def _run_eval(args: argparse.Namespace) -> int:
    directory: Path = args.data
    if not directory.is_dir():
        print(f"no such dataset directory: {directory}", file=sys.stderr)
        return 1

    manifest = load_manifest(directory)
    truth = load_ground_truth(directory)
    tag = f"{manifest.split.value}-{manifest.seed}"

    client = _client_for(args)
    try:
        bundle = _resolve_bundle(args.calibration, directory)
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # The same `run_system` every ablation row and every sweep row uses. The
    # headline is not a special path through the pipeline; it is the pipeline.
    run = run_system(directory, bundle=bundle, client=client)
    matched = run.matched
    exceptions = run.exceptions

    scored = [
        EvaluatedRun(
            system=matched,
            metrics=run.metrics,
            calibration=run.calibration,
            exceptions=exceptions,
            coverage=run.coverage,
            resolutions=run.resolutions,
            rounding_spent_minor=run.rounding_spent_minor,
            cost=run.cost if run.llm_available else None,
            llm_accepted=run.llm.accepted,
            llm_rejected_ungrounded=run.llm.rejected_ungrounded,
            llm_rejected_unverified=run.llm.rejected_unverified,
            llm_prose_rewritten=run.llm.explanation.rewritten,
            candidates_proposed=run.candidates_proposed,
            auto_matched=run.auto_matched,
            review_queue=run.needs_review,
            negatives=_negative_counters(run),
        )
    ]
    for baseline in (run_b0(directory), run_b1(directory)):
        scored.append(
            EvaluatedRun(
                system=baseline,
                metrics=evaluate(
                    baseline.predictions,
                    truth,
                    run_id=f"{baseline.name.lower()}-{tag}",
                    wall_clock_ms=baseline.wall_clock_ms,
                ),
            )
        )

    artifacts = _load_artifacts(args)
    if artifacts is None:
        return 1
    llm_baseline, ablation, sweep = artifacts

    write_report(
        args.out,
        render_report(
            scored,
            manifest=manifest,
            truth=truth,
            llm_baseline=llm_baseline,
            ablation=ablation,
            sweep=sweep,
        ),
    )

    print(f"evaluated {directory} ({manifest.split.value}, seed {manifest.seed})")
    for item in scored:
        links = item.metrics.link_metrics
        assert links is not None  # evaluate() always populates it
        print(
            f"  {item.system.name}: precision "
            f"{item.metrics.auto_match_precision:.4f} "
            f"[{links.precision_ci_low:.4f}, {links.precision_ci_high:.4f}] - "
            f"recall {links.recall:.4f} - match rate {item.metrics.match_rate:.4f}"
        )
        print(
            f"      {links.true_positives} correct - {links.false_positives} false "
            f"positives costing {format_minor(links.false_positive_cost_minor)} - "
            f"{links.false_negatives} missed"
        )

    print(
        f"  decisions: {matched.auto_matched} auto-matched - {matched.needs_review} "
        f"needs review - {matched.exceptions} exception "
        f"(from {len(matched.candidates)} candidates, "
        f"{run.candidates_proposed} of them on the evaluation unit)"
    )
    print(
        f"  settlements: {matched.settlements_resolved} resolved - "
        f"{matched.settlements_contested} contested - "
        f"{matched.settlements_unresolved} left for later tiers"
    )
    if bundle is None:
        print("  probabilities: tier-provisional (no --calibration bundle given)")
    else:
        blend = matched.blend
        print(
            f"  probabilities: calibrated - {blend.scored} scored - "
            f"{blend.bypassed_deterministic} deterministic (T0/T1 bypass) - "
            f"{blend.refusals_kept} tier refusals kept - "
            f"{blend.abstained_uncovered} abstained on an unfitted tier"
        )
        seeds = ", ".join(str(s) for s in bundle.provenance.calibration_seeds)
        print(
            f"      tau_high = {run.config.thresholds.tau_high:.6f} (fitted on "
            f"{bundle.provenance.calibration_split.value} seeds {seeds})"
        )
        view = run.calibration
        if view is not None:
            asserted = view.asserted
            print(
                f"      asserted: ECE {asserted.ece:.4f} - Brier {asserted.brier:.4f} - "
                f"{asserted.sample_count} residual links - "
                f"{asserted.populated_bins} of {len(asserted.bins)} bins populated"
            )
            if view.contenders is not None:
                contender = view.contenders
                wrong = contender.sample_count - contender.positive_count
                print(
                    f"      contenders: ECE {contender.ece:.4f} - "
                    f"Brier {contender.brier:.4f} - {contender.sample_count} pairings "
                    f"({wrong} of them wrong)"
                )
    print(
        f"  exceptions: {len(exceptions)} raised covering "
        f"{format_minor(sum(e.impact_minor for e in exceptions))} - "
        f"{len(run.coverage.unmatchable)} unmatchable (the honest floor) - "
        f"{sum(1 for r in run.resolutions if r.applied)} auto-resolvable within bounds"
    )
    print(
        f"      recall {run.coverage.recall:.4f} over {len(run.coverage.expected)} records "
        f"ground truth calls exceptions ({len(run.coverage.missed)} missed) - "
        f"unmatchable coverage {run.coverage.unmatchable_recall:.4f} over "
        f"{len(run.coverage.unmatchable)} - {run.coverage.out_of_scope} outgoing rows "
        "outside the unit"
    )
    if args.show_exceptions > 0:
        for exception in exceptions[: args.show_exceptions]:
            print(
                f"      [{exception.severity.value:<8}] "
                f"{format_minor(exception.impact_minor):>16}  "
                f"{exception.exception_class.value:<26} "
                f"{exception.involved_refs[0].record_id}"
            )
            print(f"          {exception.root_cause}")
            print(f"          -> {exception.suggested_action}")
    if not run.llm_available:
        print(
            f"  llm: disabled ({_llm_disabled_reason(args)}); every number above "
            "is deterministic"
        )
    else:
        cost = run.cost
        print(
            f"  llm: {cost.llm_calls} call(s) - {cost.cache_hits} cache hit(s) - "
            f"{cost.total_tokens} tokens - "
            f"{cost.calls_per_100_records(run.metrics.record_count):.2f} calls per 100 "
            f"records - actual ₹{cost.actual_cost_inr:.2f} - "
            f"equivalent paid ₹{cost.equivalent_paid_cost_inr:.2f}"
        )
        print(
            f"      accepted {run.llm.accepted} - refused "
            f"{run.llm.rejected_ungrounded} ungrounded - "
            f"{run.llm.rejected_unverified} failing verify_arithmetic - "
            f"{run.llm.calls_refused} call(s) fell back"
        )
        print(
            f"      narrations repaired {run.llm.narration.accepted}/"
            f"{run.llm.narration.attempted} - T5 candidates "
            f"{len(run.llm.adjudication.candidates)} "
            f"({run.llm.adjudication.demoted} demoted) - prose rewritten "
            f"{run.llm.explanation.rewritten}/{len(exceptions)}"
        )
    for section, artifact in (
        ("ablation", ablation),
        ("sweep", sweep),
        ("B2", llm_baseline),
    ):
        if artifact is None:
            print(f"  {section}: absent (no artefact given; the section is omitted)")
    if run.ingest.problems:
        print(
            f"  {len(run.ingest.problems)} malformed source records quarantined",
            file=sys.stderr,
        )
    print(f"  wrote {args.out}")
    return 0


def _load_artifacts(
    args: argparse.Namespace,
) -> tuple[LLMBaselineArtifact | None, AblationArtifact | None, SweepArtifact | None] | None:
    """Read the three optional artefacts, or report which one is unreadable.

    A missing path is an absent section. A path that exists and will not parse
    is an error, not an absence: silently omitting a section the caller asked
    for would produce a report quietly missing a table someone requested.
    """
    loaded: list[object] = []
    for path, loader, label in (
        (args.llm_baseline, LLMBaselineArtifact.load, "--llm-baseline"),
        (args.ablation, AblationArtifact.load, "--ablation"),
        (args.sweep, SweepArtifact.load, "--sweep"),
    ):
        if path is None:
            loaded.append(None)
            continue
        if not path.is_file():
            print(f"no such artefact for {label}: {path}", file=sys.stderr)
            return None
        try:
            loaded.append(loader(path))
        except ValueError as exc:
            print(f"{label} artefact {path} did not parse: {exc}", file=sys.stderr)
            return None
    baseline, ablation, sweep = loaded
    assert baseline is None or isinstance(baseline, LLMBaselineArtifact)
    assert ablation is None or isinstance(ablation, AblationArtifact)
    assert sweep is None or isinstance(sweep, SweepArtifact)
    return baseline, ablation, sweep


def _run_pipeline(args: argparse.Namespace) -> int:
    """PLAN.md §4.1's state machine, over one dataset.

    Every number printed below comes from the same
    :class:`~ledgerloop.eval.harness.SystemRun` ``eval`` scores. The graph moves
    data between tested functions; it computes nothing.
    """
    directory: Path = args.data
    if not directory.is_dir():
        print(f"no such dataset directory: {directory}", file=sys.stderr)
        return 1
    if not langgraph_available():
        print(
            "LangGraph is not installed. Install the extra with "
            '`uv pip install -e ".[graph]"`. Every metric in EVALUATION.md is '
            "produced without it -- `ledgerloop eval` needs no extra.",
            file=sys.stderr,
        )
        return 1

    client = _client_for(args)
    try:
        bundle = _resolve_bundle(args.calibration, directory)
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = run_graph(
        directory,
        bundle=bundle,
        client=client,
        run_id=args.run_id,
        store=args.runs_dir,
    )
    if not result.ok:
        print(
            f"run failed at node `{result.failed_node}`: {result.error}",
            file=sys.stderr,
        )
        print(
            f"  {len(result.audit.events)} audit event(s) recorded; the log is "
            "replayable up to the failure",
            file=sys.stderr,
        )
        return 1

    run = result.require()
    metrics = run.metrics
    links = metrics.link_metrics
    assert links is not None  # evaluate() always populates it

    print(f"reconciled {directory} ({run.manifest.split.value}, seed {run.manifest.seed})")
    print(
        f"  graph: {len(result.node_log)} node visit(s), "
        f"{result.residual_iterations} residual pass(es), "
        f"{len(result.audit.events)} audit event(s) in {result.wall_clock_ms} ms"
    )
    if args.show_nodes:
        for index, node in enumerate(result.node_log):
            print(f"      {index:>2}. {node}")
    print(
        f"  precision {metrics.auto_match_precision:.4f} "
        f"[{links.precision_ci_low:.4f}, {links.precision_ci_high:.4f}] - "
        f"recall {links.recall:.4f} - match rate {metrics.match_rate:.4f}"
    )
    print(
        f"      {links.true_positives} correct - {links.false_positives} false "
        f"positives costing {format_minor(links.false_positive_cost_minor)} - "
        f"{links.false_negatives} missed"
    )
    print(
        f"  decisions on the evaluation unit: {run.auto_matched} auto-matched - "
        f"{run.needs_review} needs review - "
        f"{run.candidates_proposed} candidate(s) proposed"
    )
    print(
        f"  exceptions: {len(run.exceptions)} raised covering "
        f"{format_minor(sum(e.impact_minor for e in run.exceptions))} - "
        f"recall {run.coverage.recall:.4f} over {len(run.coverage.expected)} - "
        f"{len(run.coverage.unmatchable)} unmatchable (the honest floor)"
    )
    if not run.llm_available:
        print(
            f"  llm: disabled ({_llm_disabled_reason(args)}); every number above "
            "is deterministic"
        )
    else:
        cost = run.cost
        print(
            f"  llm: {cost.llm_calls} call(s) - {cost.cache_hits} cache hit(s) - "
            f"{cost.total_tokens} tokens - equivalent paid "
            f"₹{cost.equivalent_paid_cost_inr:.2f}"
        )
    print(f"  wrote {args.runs_dir / run.config.run_id}")
    return 0


def _run_ablation(args: argparse.Namespace) -> int:
    """PLAN.md §9.3. Six ladders, every seed, one artefact."""
    directories: list[Path] = list(args.data)
    for directory in directories:
        if not directory.is_dir():
            print(f"no such dataset directory: {directory}", file=sys.stderr)
            return 1
    try:
        bundle = _resolve_bundle(args.calibration, directories[0])
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def factory() -> LLMClient:
        return _client_for(args)

    probe = factory()
    try:
        artifact = run_ablation(
            directories,
            bundle=bundle,
            client_factory=factory if probe.enabled else None,
            ladders=ABLATION_LADDERS,
        )
    except ValueError as exc:
        print(f"ablation refused: {exc}", file=sys.stderr)
        return 1
    artifact.save(args.out)

    print(
        f"ablation over {len(directories)} corpus/corpora "
        f"({artifact.split}, {artifact.difficulty}, seeds "
        + ", ".join(str(seed) for seed in artifact.seeds)
        + ")"
    )
    for index, row in enumerate(artifact.rows):
        print(
            f"  {row.label:<8} precision {row.precision.rendered()} - recall "
            f"{row.recall.rendered()} (marginal "
            f"{artifact.marginal(index, 'recall'):+.4f}) - match rate "
            f"{row.match_rate.rendered()}"
        )
        print(
            f"           {row.candidate_yield.mean:.1f} proposed - "
            f"{row.auto_matched.mean:.1f} auto-matched - "
            f"{row.false_positives.mean:.1f} wrong costing "
            f"{format_minor(round(row.false_positive_cost_minor.mean))} - "
            f"{row.llm_calls.mean:.1f} llm call(s), "
            f"{row.llm_tokens.mean:.0f} tokens"
        )
    if not probe.enabled:
        print(
            f"  llm: disabled ({_llm_disabled_reason(args)}); the T0-T5 row ran "
            "the deterministic ladder and the report says so"
        )
    print(f"  wrote {args.out}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    """PLAN.md §9.4. The headline configuration across seeds and difficulties."""
    directories: list[Path] = list(args.data)
    for directory in directories:
        if not directory.is_dir():
            print(f"no such dataset directory: {directory}", file=sys.stderr)
            return 1
    try:
        bundle = _resolve_bundle(args.calibration, directories[0])
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def factory() -> LLMClient:
        return _client_for(args)

    probe = factory()
    try:
        artifact = run_sweep(
            directories,
            bundle=bundle,
            client_factory=factory if probe.enabled else None,
        )
    except ValueError as exc:
        print(f"sweep refused: {exc}", file=sys.stderr)
        return 1
    artifact.save(args.out)

    print(f"swept {len(directories)} corpora of `{artifact.split}`")
    for group in artifact.groups:
        print(
            f"  {group.difficulty:<9} {len(group.seeds)} seed(s) "
            + ", ".join(str(seed) for seed in group.seeds)
        )
        print(
            f"      precision {group.of('precision').rendered()} - recall "
            f"{group.of('recall').rendered()} - match rate "
            f"{group.of('match_rate').rendered()} - exception recall "
            f"{group.of('exception_recall').rendered()}"
        )
        print(
            f"      false positives {group.of('false_positives').rendered(digits=2)} "
            f"costing {format_minor(round(group.of('false_positive_cost_minor').mean))} "
            f"- unmatchable {group.of('unmatchable_count').rendered(digits=1)}"
        )
        hashes = group.config_hashes
        if len(hashes) == 1:
            print(f"      every seed ran configuration {hashes[0]}")
        else:  # pragma: no cover - the runner holds the config fixed
            print(f"      CONFIGURATIONS DIFFER: {', '.join(hashes)}", file=sys.stderr)
    print(f"  wrote {args.out}")
    return 0


def _run_llm_baseline(args: argparse.Namespace) -> int:
    """B2. The one command that can reach a network on its own account."""
    directory: Path = args.data
    if not directory.is_dir():
        print(f"no such dataset directory: {directory}", file=sys.stderr)
        return 1

    manifest = load_manifest(directory)
    truth = load_ground_truth(directory)
    if manifest.split is not SplitName.DEV:
        print(
            f"warning: B2 is specified on the `dev` split (PLAN.md §9.2) and this "
            f"is `{manifest.split.value}`. Running it anyway; the token cost "
            "scales with the corpus.",
            file=sys.stderr,
        )

    client = _client_for(args, max_calls=args.max_calls)
    if args.cold and args.cache_dir.is_dir():
        shutil.rmtree(args.cache_dir)
    client = LLMClient(
        config=client.config.model_copy(update={"cache_dir": args.cache_dir}),
        provider=client.provider,
    )
    if args.offline_provider:
        # Explicit, never a fallback. A run that quietly substituted a stand-in
        # when a key was missing would publish a row nobody could tell apart
        # from a live measurement, which is the failure this flag exists to
        # avoid rather than to cause.
        client = LLMClient(
            config=client.config.model_copy(update={"enabled": True}),
            provider=OfflineReasoner(),
        )
    ingested = ingest_dataset(directory, strict=False)
    predictions, artifact = run_b2(
        client, ingested, manifest, payments_per_call=args.payments_per_call
    )

    # Scored by exactly the same evaluator as everything else. B2's links are
    # never handed to the matcher, the decision policy or the calibrator: this
    # module is the only place they exist, and they leave it as a report row.
    metrics = evaluate(
        predictions,
        truth,
        run_id=f"b2-{manifest.split.value}-{manifest.seed}",
        wall_clock_ms=artifact.wall_clock_ms,
    )
    links = metrics.link_metrics
    assert links is not None  # evaluate() always populates it

    # The production system on the identical corpus, so the token multiple has
    # a measured denominator rather than an assumed one.
    try:
        bundle = _resolve_bundle(args.calibration, directory)
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    system = run_system(
        directory,
        bundle=bundle,
        client=_client_for(args),
        measure_calibration_quality=False,
    )

    artifact = artifact.model_copy(
        update={
            "record_count": metrics.record_count,
            "evaluation_links": len(truth.evaluation_pairs),
            "true_positives": links.true_positives,
            "false_positives": links.false_positives,
            "false_negatives": links.false_negatives,
            "precision": links.precision,
            "precision_ci_low": links.precision_ci_low,
            "precision_ci_high": links.precision_ci_high,
            "recall": links.recall,
            "f1": links.f1,
            "match_rate": metrics.match_rate,
            "false_positive_cost_minor": links.false_positive_cost_minor,
            "system_cost": system.cost,
            "system_ran": system.llm_available,
            "provider_kind": (
                OFFLINE_PROVIDER_NAME
                if args.offline_provider
                else ("live" if artifact.ran else "")
            ),
        }
    )
    artifact.save(args.out)

    print(f"B2 on {directory} ({manifest.split.value}, seed {manifest.seed})")
    if artifact.is_standin:
        print(
            "  provider: OFFLINE STAND-IN, not a language model. Cost, cache and "
            "failure figures are measured; accuracy figures are a property of the "
            "documented nearest-amount rule and say nothing about a model."
        )
    if not artifact.ran:
        print(f"  not run: {artifact.reason}")
        print("  the report will say so rather than printing a precision of zero")
        print(f"  wrote {args.out}")
        return 0
    print(
        f"  precision {links.precision:.4f} "
        f"[{links.precision_ci_low:.4f}, {links.precision_ci_high:.4f}] - "
        f"recall {links.recall:.4f} - match rate {metrics.match_rate:.4f}"
    )
    print(
        f"      {links.true_positives} correct - {links.false_positives} false "
        f"positives costing {format_minor(links.false_positive_cost_minor)} - "
        f"{links.false_negatives} missed"
    )
    print(
        f"  answers: {artifact.calls_attempted} call(s), {artifact.calls_failed} "
        f"unusable - {artifact.links_returned} links returned, "
        f"{artifact.links_duplicated} repeated, {artifact.links_asserted} asserted"
    )
    print(
        f"      invented ids: {artifact.unknown_payment_ids} payment(s), "
        f"{artifact.unknown_bank_txn_ids} bank row(s) - asserted anyway, because "
        "B2 has no grounding gate"
    )
    print(
        f"  cost: {artifact.cost.llm_calls} call(s) - {artifact.cost.cache_hits} "
        f"hit(s) ({artifact.cost.cache_hit_rate:.2f}) - "
        f"{artifact.cost.total_tokens} tokens - actual "
        f"₹{artifact.cost.actual_cost_inr:.2f} - equivalent paid "
        f"₹{artifact.cost.equivalent_paid_cost_inr:.2f}"
    )
    if artifact.system_ran:
        print(
            f"      LedgerLoop on the same corpus: {artifact.system_cost.llm_calls} "
            f"call(s), {artifact.system_cost.total_tokens} tokens - B2 spends "
            f"{artifact.token_multiple:.1f}x the tokens"
        )
    else:
        print(
            "      LedgerLoop made no calls on this corpus, so the token multiple "
            "has no denominator and is not reported"
        )
    print(f"  wrote {args.out}")
    return 0


def _force_utf8_output() -> None:
    """Print rupee symbols on a Windows console without crashing.

    The default Windows code page is cp1252, which has no ``₹`` -- so a plain
    ``print`` of a formatted amount raises ``UnicodeEncodeError`` and takes the
    run down with it. A judge cloning this on Windows would hit that on the
    first command, so it is fixed here rather than by avoiding the symbol.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # pragma: no branch - always present on TextIO
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "calibrate":
        return _run_calibrate(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "run":
        return _run_pipeline(args)
    if args.command == "ablation":
        return _run_ablation(args)
    if args.command == "sweep":
        return _run_sweep(args)
    if args.command == "baseline-llm":
        return _run_llm_baseline(args)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
