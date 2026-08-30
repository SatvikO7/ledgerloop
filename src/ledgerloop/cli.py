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

Step 13 adds ``demo``: generate, calibrate, reconcile, open the UI. It exists
because ``make`` is **not** a reasonable prerequisite -- it is absent from the
Windows machine this project is developed on, and a project whose claim is that
it runs on nothing should not require GNU Make to be seen running. ``make demo``
is now a one-line wrapper over this command, so there is one implementation of
the demo and the Makefile is a convenience rather than the only door.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.store import RUNS_ROOT
from ledgerloop.config import SPLIT_SIZES, GeneratorConfig, LLMConfig, RunConfig
from ledgerloop.eval.ablation import ABLATION_LADDERS, AblationArtifact, run_ablation
from ledgerloop.eval.artifacts import ComparisonArtifact, LLMReportArtifact
from ledgerloop.eval.baselines import run_b0, run_b1
from ledgerloop.eval.comparison import COMPARABLE, run_comparison
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
from ledgerloop.eval.llm_report import run_llm_report
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.offline_provider import OFFLINE_PROVIDER_NAME, OfflineReasoner
from ledgerloop.eval.report import EvaluatedRun, render_report, write_report
from ledgerloop.eval.scale import DEFAULT_SCALE_SIZES, run_scale
from ledgerloop.eval.sweep import SweepArtifact, run_sweep
from ledgerloop.eval.truth_io import load_ground_truth, load_manifest
from ledgerloop.fitting import FittingError, fit_from_corpora, harvest_corpora
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import IngestError, ingest_dataset
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.offline_analyst import OFFLINE_ANALYST_NAME, OfflineAnalyst
from ledgerloop.llm.providers import (
    PROVIDER_KEY_ENVS,
    build_ladder,
    configured_rungs,
)
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
    _add_llm_flags(evaluation)
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
        "--comparison",
        type=Path,
        default=None,
        help="an artefact written by `ledgerloop comparison`, carrying the "
        "before/after study for a substantive change. Absent renders no section.",
    )
    evaluation.add_argument(
        "--llm-report",
        type=Path,
        default=None,
        help="an artefact written by `ledgerloop llm-report`. Rendering it makes "
        "no calls: a report regenerated twice must not spend quota twice.",
    )
    evaluation.add_argument(
        "--out",
        type=Path,
        default=Path("EVALUATION.md"),
        help="report destination. Regenerated in full every run and gitignored, "
        "because a committed report is one that can be quietly corrected.",
    )

    demo = subparsers.add_parser(
        "demo",
        help="the whole thing: generate, calibrate, reconcile, open the UI",
    )
    demo.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/generated"),
        help="where the generated corpora go (default data/generated, gitignored)",
    )
    demo.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_ROOT,
        help=f"where the run record is written (default {RUNS_ROOT})",
    )
    demo.add_argument(
        "--bundle",
        type=Path,
        default=Path("reports/calibration.json"),
        help="where the fitted calibration bundle is written and read",
    )
    demo.add_argument(
        "--seed", type=int, default=42, help="seed for the demonstrated corpus"
    )
    demo.add_argument(
        "--difficulty",
        type=Difficulty,
        choices=list(Difficulty),
        default=Difficulty.STANDARD,
        help="anomaly prevalence dial for the demonstrated corpus",
    )
    demo.add_argument(
        "--split",
        type=SplitName,
        choices=list(SplitName),
        default=SplitName.TEST,
        help="which split to reconcile. Defaults to `test` -- 742 records and "
        "294 evaluation links -- because that is the corpus every number in "
        "README.md and EVALUATION.md is measured on, and a demo that opened on a "
        "different one would show a reviewer figures the documents do not "
        "contain. `--split dev` is the 60-order corpus: it still clears the "
        "challenge's 50+ bar and is faster, but its exception recall rests on "
        "five records and means very little. The UI's Run tab opens either.",
    )
    demo.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate corpora that already exist. Off by default: generation "
        "is a pure function of (seed, split, difficulty), so an existing "
        "directory holds byte-identical data and rewriting it buys nothing.",
    )
    demo.add_argument(
        "--refit",
        action="store_true",
        help="refit the calibration bundle even if one is already on disk",
    )
    demo.add_argument(
        "--no-ui",
        action="store_true",
        help="stop after the reconciliation and print the UI command instead of "
        "launching it. What CI and the smoke test use.",
    )
    _add_llm_flags(demo)

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

    scale = subparsers.add_parser(
        "scale",
        help="benchmark the pipeline over a series of corpus sizes, up to the "
        "scale split's 5,000 orders",
    )
    scale.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALE_SIZES),
        help="order counts to run, smallest first (default: "
        + " ".join(str(n) for n in DEFAULT_SCALE_SIZES)
        + "). The small end is the size of `test`, so the curve is anchored to "
        "the corpus every published number comes from.",
    )
    scale.add_argument("--seed", type=int, default=42, help="RNG seed for every size")
    scale.add_argument(
        "--difficulty",
        type=Difficulty,
        choices=list(Difficulty),
        default=Difficulty.STANDARD,
        help="anomaly prevalence dial",
    )
    scale.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/generated"),
        help="where the scale corpora are written and looked for",
    )
    scale.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`",
    )
    scale.add_argument(
        "--regenerate",
        action="store_true",
        help="rebuild the corpora even where they already exist",
    )
    scale.add_argument(
        "--out",
        type=Path,
        default=Path("reports/scale.json"),
        help="where the artefact is written (default reports/scale.json)",
    )

    comparison = subparsers.add_parser(
        "comparison",
        help="run the headline configuration with and without one change, and "
        "score both arms over the same corpora",
    )
    comparison.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset directories. Every one is run twice -- once per arm.",
    )
    comparison.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`, applied to both arms",
    )
    comparison.add_argument(
        "--switch",
        default="split-completion",
        choices=sorted(COMPARABLE),
        help="which change to compare (default split-completion, the most "
        "recent). Exactly one RunConfig field differs between the arms.",
    )
    comparison.add_argument(
        "--out",
        type=Path,
        default=Path("reports/comparison.json"),
        help="artefact destination (default reports/comparison.json)",
    )

    llm_report = subparsers.add_parser(
        "llm-report",
        help="run the production LLM path once, measured, with a no-LLM control",
    )
    llm_report.add_argument(
        "--data",
        type=Path,
        required=True,
        help="a generated dataset directory (as written by `ledgerloop generate`)",
    )
    llm_report.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="a bundle written by `ledgerloop calibrate`",
    )
    _add_llm_flags(llm_report)
    llm_report.add_argument(
        "--max-calls",
        type=int,
        default=30,
        help="hard budget for this run (default 30). Exceeding it aborts the LLM "
        "path rather than quietly burning free-tier quota, and the artefact "
        "counts the refusals.",
    )
    llm_report.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="response cache for this run (default the configured production "
        "cache). Point it somewhere empty to measure a cold run.",
    )
    llm_report.add_argument(
        "--offline-provider",
        action="store_true",
        help="drive the run with the prompt-reading stand-in in "
        "`llm/offline_analyst.py` instead of a live model. Every call, token, "
        "cache, latency, failure and gate figure is then still measured on the "
        "real code path, but NOTHING here is a claim about a language model's "
        "answer quality -- the artefact records `live: false` and the printed "
        "summary says so. Drop the flag on a machine with a provider key.",
    )
    llm_report.add_argument(
        "--out",
        type=Path,
        default=Path("reports/llm_report.json"),
        help="artefact destination (default reports/llm_report.json)",
    )
    return parser


def _add_llm_flags(parser: argparse.ArgumentParser) -> None:
    """``--no-llm``, ``--llm-key-env`` and ``--llm-providers`` on every command.

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
        help="environment variable holding a key shared by every rung of the "
        "provider ladder (default LEDGERLOOP_LLM_API_KEY). Per-provider "
        "variables -- GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY -- take "
        "precedence. None of them set means the deterministic path.",
    )
    parser.add_argument(
        "--llm-providers",
        default=None,
        help="comma-separated failover ladder, overriding "
        "LEDGERLOOP_LLM_PROVIDERS and the default "
        "groq,gemini,openrouter,ollama. A rung with no credential is skipped, "
        "except ollama, which needs none and is therefore only used when it is "
        "named here or given a base URL.",
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

    From Phase 2.2 the provider is a **ladder** rather than a single endpoint.
    Nothing above this line changed: the ladder satisfies the same one-method
    protocol, so the cache, the budget, the validation, the gates and the cost
    ledger sit exactly where they were.
    """
    wanted = not args.no_llm
    config = LLMConfig(enabled=wanted)
    if max_calls is not None:
        config = config.model_copy(update={"max_calls_per_run": max_calls})
    if not wanted:
        return LLMClient(config=config, provider=None)
    ladder = build_ladder(config, environ=_llm_env(args), order=_llm_order(args))
    return LLMClient(config=config, provider=ladder)


def _llm_env(args: argparse.Namespace) -> dict[str, str]:
    """The environment the ladder reads, with ``--llm-key-env`` folded in.

    A custom key variable is copied into the slot the ladder treats as the
    shared credential, so ``--llm-key-env MY_KEY`` keeps working exactly as it
    did at Step 9 rather than becoming a flag the ladder quietly ignores.
    """
    env = dict(os.environ)
    key = env.get(args.llm_key_env)
    if key:
        env["LEDGERLOOP_LLM_API_KEY"] = key
    return env


def _llm_order(args: argparse.Namespace) -> tuple[str, ...] | None:
    raw = getattr(args, "llm_providers", None)
    if not raw:
        return None
    return tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())


def _llm_disabled_reason(args: argparse.Namespace) -> str:
    if args.no_llm:
        return "--no-llm"
    return (
        f"no provider key in ${args.llm_key_env} or any of "
        + ", ".join(f"${name}" for name in sorted(set(PROVIDER_KEY_ENVS.values())))
    )


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


def _distinct_probabilities(run: SystemRun) -> tuple[float, ...]:
    """Every distinct calibrated probability the run's decisions carried.

    Ascending, so the report can print the distribution rather than assert it.
    Two values is the bimodal shape that makes the three-way policy behave
    two-way, and that claim is only checkable if the values are shown.
    """
    return tuple(sorted({decision.calibrated_p for decision in run.matched.decisions}))


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
            "T3 rejected on contention",
            "two settlements of the same merchant both had a claim on the credit; "
            "their nets agree inside the band and the name is the same string, so "
            "no lexical reading says which one it paid",
            matched.lexical.rejected_on_contention,
        ),
        (
            "T3 settlements already referenced",
            "the bank had written the settlement's own UTR on a credit, so the "
            "statement has already said where the payout went; a whole-net match "
            "on a name would contradict it",
            matched.lexical.settlements_already_referenced,
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
            bundle=bundle,
            review_band_probabilities=_distinct_probabilities(run),
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

    write_report(
        args.out,
        render_report(
            scored,
            manifest=manifest,
            truth=truth,
            llm_baseline=artifacts.llm_baseline,
            ablation=artifacts.ablation,
            sweep=artifacts.sweep,
            comparison=artifacts.comparison,
            llm_report=artifacts.llm_report,
        ),
    )

    llm_baseline, ablation, sweep = (
        artifacts.llm_baseline,
        artifacts.ablation,
        artifacts.sweep,
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


class _Artifacts(NamedTuple):
    """The optional artefacts ``eval`` renders sections from."""

    llm_baseline: LLMBaselineArtifact | None = None
    ablation: AblationArtifact | None = None
    sweep: SweepArtifact | None = None
    comparison: ComparisonArtifact | None = None
    llm_report: LLMReportArtifact | None = None


def _load_artifacts(args: argparse.Namespace) -> _Artifacts | None:
    """Read the optional artefacts, or report which one is unreadable.

    A missing path is an absent section. A path that exists and will not parse
    is an error, not an absence: silently omitting a section the caller asked
    for would produce a report quietly missing a table someone requested.
    """
    loaded: list[object] = []
    for path, loader, label in (
        (args.llm_baseline, LLMBaselineArtifact.load, "--llm-baseline"),
        (args.ablation, AblationArtifact.load, "--ablation"),
        (args.sweep, SweepArtifact.load, "--sweep"),
        (getattr(args, "comparison", None), ComparisonArtifact.load, "--comparison"),
        (getattr(args, "llm_report", None), LLMReportArtifact.load, "--llm-report"),
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
    baseline, ablation, sweep, comparison, llm_report = loaded
    assert baseline is None or isinstance(baseline, LLMBaselineArtifact)
    assert ablation is None or isinstance(ablation, AblationArtifact)
    assert sweep is None or isinstance(sweep, SweepArtifact)
    assert comparison is None or isinstance(comparison, ComparisonArtifact)
    assert llm_report is None or isinstance(llm_report, LLMReportArtifact)
    return _Artifacts(baseline, ablation, sweep, comparison, llm_report)


#: The corpora the calibration bundle is fitted from.
#:
#: Five ``train`` seeds and four ``calibration`` seeds, disjoint, and neither is
#: ever ``test`` -- ``CalibrationProvenance`` refuses to construct a bundle that
#: breaks either rule. The same seeds the Makefile uses, declared here as data so
#: the two cannot drift.
DEMO_TRAIN_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEMO_CALIBRATION_SEEDS: tuple[int, ...] = (47, 48, 49, 50)


def _generate_if_absent(
    directory: Path, config: GeneratorConfig, *, regenerate: bool
) -> bool:
    """Generate one corpus unless it is already there. Returns whether it ran.

    Generation is a pure function of ``(seed, split, difficulty, order_count)``,
    so an existing directory holds byte-identical data and rewriting it buys
    nothing but wall clock. ``--regenerate`` forces it for anyone who wants to
    watch that claim hold.
    """
    if directory.is_dir() and (directory / "manifest.json").is_file() and not regenerate:
        return False
    generate_to_disk(config, directory)
    return True


def _run_comparison(args: argparse.Namespace) -> int:
    """Both arms of the Phase 2.3 change, over the same corpora.

    ``--switch`` picks which change, from a fixed list -- not an arbitrary
    configuration the caller supplies. A comparison whose arms the caller could
    define freely would be one whose meaning changed with the invocation, and
    the artefact would then have to be read alongside the command line that
    produced it. The artefact names the change it measured instead.
    """
    directories: list[Path] = list(args.data)
    try:
        bundle = _resolve_bundle(args.calibration, directories[0])
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        artifact = run_comparison(directories, bundle=bundle, switch=args.switch)
    except ValueError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 1
    artifact.save(args.out)

    print(f"comparison over {len(directories)} corpus/corpora: {artifact.change}")
    for row in artifact.rows:
        before, after = row.before, row.after
        print(
            f"  {row.difficulty:9s} recall "
            f"{before.of('recall').rendered()} -> {after.of('recall').rendered()} "
            f"({row.delta('recall'):+.4f})"
        )
        print(
            f"            match rate "
            f"{before.of('match_rate').rendered()} -> "
            f"{after.of('match_rate').rendered()} "
            f"({row.delta('match_rate'):+.4f})"
        )
        print(
            f"            precision "
            f"{before.of('precision').rendered()} -> "
            f"{after.of('precision').rendered()} - false positives "
            f"{int(before.of('false_positives').mean * len(before.runs))} -> "
            f"{int(after.of('false_positives').mean * len(after.runs))}"
        )
        print(f"            tuning hash {before.tuning_hash} -> {after.tuning_hash}")
    print(
        "  precision held at every difficulty and on every seed"
        if artifact.precision_held_everywhere
        else "  WARNING: the after arm produced false positives"
    )
    print(f"  wrote {args.out}")
    return 0


def _run_scale(args: argparse.Namespace) -> int:
    """Walk the size curve and report quality and cost side by side.

    The precision column is printed first and the throughput last, which is the
    order of their importance and the reverse of the order the item was written
    in. A false positive at 5,000 orders is a defect; a slow run is a machine.
    """
    bundle: CalibrationBundle | None = None
    if args.calibration is not None:
        try:
            bundle = CalibrationBundle.load(args.calibration)
        except FileNotFoundError:
            print(f"no calibration bundle at {args.calibration}", file=sys.stderr)
            return 1

    try:
        artifact = run_scale(
            args.data_dir,
            sizes=args.sizes,
            bundle=bundle,
            seed=args.seed,
            difficulty=args.difficulty,
            regenerate=args.regenerate,
        )
    except ValueError as exc:
        print(f"scale run refused: {exc}", file=sys.stderr)
        return 1
    artifact.save(args.out)

    print(f"scale curve · seed {artifact.seed} · {artifact.difficulty}")
    print(f"  machine: {artifact.machine}")
    print("  orders   records  precision   recall  match rate    FP   wall  rec/s")
    for point in artifact.points:
        print(
            f"  {point.orders:6d}  {point.records:8d}     "
            f"{point.precision:.4f}   {point.recall:.4f}      "
            f"{point.match_rate:.4f}  {point.false_positives:4d}  "
            f"{point.wall_clock_ms / 1000:5.1f}s  {point.records_per_second:,.0f}"
        )

    dirty = [p for p in artifact.points if p.false_positives]
    if dirty:
        print(
            "  FALSE POSITIVES at "
            + ", ".join(f"{p.orders} orders ({p.false_positives})" for p in dirty),
            file=sys.stderr,
        )
    else:
        print("  precision held at every size ✓")
    print(f"  wrote {args.out}")
    return 0 if not dirty else 1


def _run_llm_report(args: argparse.Namespace) -> int:
    """One measured run of the production LLM path, with its own control.

    Refuses rather than degrades when a live run was asked for and no provider
    is reachable. Every other command in this project falls back to the
    deterministic path when there is no key, and that is right for them --
    their job is to produce a reconciliation. This command's job is to produce
    a *measurement of the LLM path*, and silently measuring nothing would put a
    row of zeros where a reader expects an observation.
    """
    directory: Path = args.data
    config = LLMConfig(enabled=not args.no_llm, max_calls_per_run=args.max_calls)
    if args.cache_dir is not None:
        config = config.model_copy(update={"cache_dir": args.cache_dir})

    live = False
    provider: object | None = None
    if args.offline_provider:
        provider = OfflineAnalyst()
    elif config.enabled:
        provider = build_ladder(config, environ=_llm_env(args), order=_llm_order(args))
        live = provider is not None

    if provider is None:
        reason = _llm_disabled_reason(args)
        artifact = LLMReportArtifact(ran=False, reason=reason)
        artifact.save(args.out)
        print(f"llm-report did not run: {reason}")
        print(f"      wrote {args.out} recording that it did not run")
        print("      a row of zeros for a path that never executed would be a")
        print("      false measurement, so none is written")
        return 0

    rungs = configured_rungs(config, environ=_llm_env(args), order=_llm_order(args))
    client = LLMClient(config=config, provider=provider)  # type: ignore[arg-type]
    try:
        bundle = _resolve_bundle(args.calibration, directory)
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    artifact = run_llm_report(directory, client=client, bundle=bundle, live=live)
    artifact.save(args.out)

    print(f"llm-report on {directory}")
    if live:
        print(
            f"      LIVE: ladder {' -> '.join(spec.name for spec in rungs)}; "
            f"{artifact.provider_used} answered at fallback depth "
            f"{artifact.fallback_depth}"
        )
    else:
        print(
            f"      OFFLINE: answered by `{OFFLINE_ANALYST_NAME}`, a documented "
            "rule that reads the prompt and nothing else."
        )
        print(
            "      Calls, tokens, latency, cache, failures and every gate figure "
            "below are measured"
        )
        print(
            "      machinery. NO claim is made here about any language model's "
            "answer quality."
        )
    for failure in artifact.provider_failures:
        print(f"      rung declined - {failure}")
    cost = artifact.cost
    print(
        f"      {cost.llm_calls} call(s), {cost.cache_hits} cache hit(s), "
        f"{cost.total_tokens} token(s), {cost.wall_clock_ms} ms of provider time"
    )
    print(
        f"      {artifact.calls_per_100_records:.2f} call(s) per 100 records; "
        f"actual Rs {cost.actual_cost_inr:.2f}, equivalent paid Rs "
        f"{cost.equivalent_paid_cost_inr:.2f}"
    )
    print(
        f"      {artifact.calls_refused} call(s) refused, "
        f"{artifact.validation_failures} schema failure(s) retried"
    )
    print(
        f"      narrations {artifact.narrations_accepted}/"
        f"{artifact.narrations_offered} accepted - proposals "
        f"{artifact.proposals_accepted}/{artifact.proposals_returned} accepted - "
        f"{artifact.rejected_ungrounded} ungrounded refused - "
        f"{artifact.demoted} demoted on arithmetic"
    )
    with_llm, without = artifact.with_llm, artifact.without_llm
    print(
        f"      with the model:    precision {with_llm.precision:.4f} - recall "
        f"{with_llm.recall:.4f} - match rate {with_llm.match_rate:.4f} - "
        f"exception recall {with_llm.exception_recall:.4f}"
    )
    print(
        f"      without the model: precision {without.precision:.4f} - recall "
        f"{without.recall:.4f} - match rate {without.match_rate:.4f} - "
        f"exception recall {without.exception_recall:.4f}"
    )
    if artifact.metrics_unchanged:
        print("      the model changed no published metric")
    else:
        print(
            "      a published metric moved: the model's accepted narration "
            "repairs changed what the"
        )
        print(
            "      deterministic ladder had to read. Every decision above was "
            "still made by the ladder,"
        )
        print("      the policy and verify_arithmetic -- see the two rows.")
    print(f"      wrote {args.out}")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    """Generate, calibrate, reconcile, then open the UI.

    Every stage is the same command a reader can run on its own -- this chains
    them rather than reimplementing any of them, and prints each underlying
    invocation so the demo teaches the CLI instead of hiding it.
    """
    data_dir: Path = args.data_dir
    corpus = data_dir / f"{args.split.value}-{args.difficulty.value}-{args.seed}"

    print("LedgerLoop demo")
    print("  1. generate    three heterogeneous sources plus link-level ground truth")
    print("  2. calibrate   fit the blender and tau_high on train + calibration")
    print("  3. reconcile   run the LangGraph pipeline over the demo corpus")
    print("  4. inspect     open the four screens")
    print()

    # --- 1. generate -------------------------------------------------------
    print("[1/4] generating corpora")
    generated = 0
    for seed in DEMO_TRAIN_SEEDS:
        target = data_dir / f"train-standard-{seed}"
        generated += _generate_if_absent(
            target,
            GeneratorConfig(split=SplitName.TRAIN, seed=seed),
            regenerate=args.regenerate,
        )
    for seed in DEMO_CALIBRATION_SEEDS:
        target = data_dir / f"calibration-standard-{seed}"
        generated += _generate_if_absent(
            target,
            GeneratorConfig(split=SplitName.CALIBRATION, seed=seed),
            regenerate=args.regenerate,
        )
    generated += _generate_if_absent(
        corpus,
        GeneratorConfig(
            split=args.split, difficulty=args.difficulty, seed=args.seed
        ),
        regenerate=args.regenerate,
    )
    total = len(DEMO_TRAIN_SEEDS) + len(DEMO_CALIBRATION_SEEDS) + 1
    print(
        f"      {generated} generated, {total - generated} already present "
        f"(generation is a pure function of the seed, so an existing corpus is "
        f"byte-identical)"
    )
    truth = load_ground_truth(corpus)
    print(
        f"      demo corpus {corpus}: {len(truth.records)} records, "
        f"{len(truth.evaluation_pairs)} evaluation links, "
        f"{len(truth.unmatchable_refs)} unmatchable by construction"
    )

    # --- 2. calibrate ------------------------------------------------------
    bundle_path: Path = args.bundle
    print()
    print("[2/4] fitting the calibration bundle")
    if bundle_path.is_file() and not args.refit:
        print(f"      {bundle_path} already exists (pass --refit to redo it)")
    else:
        config = RunConfig(run_id="demo-calibrate", split=SplitName.TRAIN)
        try:
            bundle = fit_from_corpora(
                harvest_corpora(
                    [data_dir / f"train-standard-{s}" for s in DEMO_TRAIN_SEEDS],
                    config=config,
                ),
                harvest_corpora(
                    [
                        data_dir / f"calibration-standard-{s}"
                        for s in DEMO_CALIBRATION_SEEDS
                    ],
                    config=config,
                ),
                target_precision=config.thresholds.target_auto_match_precision,
            )
        except (FittingError, ValueError) as exc:
            print(f"calibration refused: {exc}", file=sys.stderr)
            return 1
        bundle.save(bundle_path)
        selection = bundle.thresholds
        print(
            f"      wrote {bundle_path}: tau_high = {selection.tau_high:.6f}, "
            f"fitted on {bundle.provenance.calibration_split.value} seeds "
            + ", ".join(str(s) for s in bundle.provenance.calibration_seeds)
        )
        print(
            f"      achieved precision {selection.achieved_precision:.4f} "
            f"[{selection.precision_ci_low:.4f}, {selection.precision_ci_high:.4f}] "
            f"on {selection.auto_matched} calibration links "
            f"({selection.false_positives} wrong)"
        )
    print("      the `test` split is never fitted on; the bundle's provenance says so")

    # --- 3. reconcile ------------------------------------------------------
    print()
    print("[3/4] reconciling")
    if not langgraph_available():
        print(
            "LangGraph is not installed. Install the demo extra with "
            '`uv pip install -e ".[demo]"`, or run `ledgerloop eval` which '
            "needs no extra.",
            file=sys.stderr,
        )
        return 1
    client = _client_for(args)
    try:
        bundle_for_run = _resolve_bundle(bundle_path, corpus)
    except (FileNotFoundError, StaleCalibrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = run_graph(corpus, bundle=bundle_for_run, client=client, store=args.runs_dir)
    if not result.ok:
        print(
            f"the run failed at node `{result.failed_node}`: {result.error}",
            file=sys.stderr,
        )
        return 1

    run = result.require()
    metrics = run.metrics
    links = metrics.link_metrics
    assert links is not None  # evaluate() always populates it
    print(
        f"      {len(result.node_log)} node visit(s), "
        f"{result.residual_iterations} residual pass(es), "
        f"{len(result.audit.events)} audit event(s) in {result.wall_clock_ms} ms"
    )
    print(
        f"      precision {metrics.auto_match_precision:.4f} "
        f"[{links.precision_ci_low:.4f}, {links.precision_ci_high:.4f}] - "
        f"recall {links.recall:.4f} - match rate {metrics.match_rate:.4f}"
    )
    print(
        f"      {links.true_positives} correct - {links.false_positives} false "
        f"positives costing {format_minor(links.false_positive_cost_minor)} - "
        f"{links.false_negatives} missed"
    )
    print(
        f"      {len(run.exceptions)} exception(s) covering "
        f"{format_minor(sum(e.impact_minor for e in run.exceptions))}, "
        f"exception recall {run.coverage.recall:.4f} over "
        f"{len(run.coverage.expected)} - {len(run.coverage.unmatchable)} "
        "unmatchable (the honest floor)"
    )
    if not run.llm_available:
        print(
            f"      llm: disabled ({_llm_disabled_reason(args)}); every number "
            "above is deterministic"
        )
    else:
        cost = run.cost
        print(
            f"      llm: {cost.llm_calls} call(s), {cost.total_tokens} tokens, "
            f"equivalent paid ₹{cost.equivalent_paid_cost_inr:.2f}. It proposed; "
            "it never decided and never did arithmetic."
        )
    print(f"      wrote {args.runs_dir / run.config.run_id}")

    # --- 4. the UI ---------------------------------------------------------
    print()
    print("[4/4] the four screens")
    # Located by path rather than by import: importing the module would make
    # Streamlit a hard dependency of the CLI, and `ledgerloop eval` must keep
    # working without either optional extra.
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    if args.no_ui:
        print("      skipped (--no-ui). Open them with:")
        print(f"          {Path(sys.executable).name} -m streamlit run {app}")
        return 0
    print("      opening Streamlit. Ctrl-C to stop.")
    print(
        "      Run · Results · Exceptions · Audit replay. Every number is read "
        "from the run record; the UI computes nothing."
    )
    print()
    try:
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app)]
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0


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
    if args.command == "comparison":
        return _run_comparison(args)
    if args.command == "scale":
        return _run_scale(args)
    if args.command == "llm-report":
        return _run_llm_report(args)
    if args.command == "demo":
        return _run_demo(args)
    if args.command == "ablation":
        return _run_ablation(args)
    if args.command == "sweep":
        return _run_sweep(args)
    if args.command == "baseline-llm":
        return _run_llm_baseline(args)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
