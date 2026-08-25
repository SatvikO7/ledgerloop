"""Read a generated dataset's ground truth back off disk.

The exact inverse of :mod:`ledgerloop.generator.emitters`, and it shares that
module's filename constants so the two cannot drift apart. There is a
round-trip test asserting that loading a freshly generated dataset reproduces
the in-memory :class:`~ledgerloop.models.truth.GroundTruth` it was written from.

**Why the evaluator reads truth from disk at all.** It could take the generator's
return value directly, and the tests sometimes do. But the reported numbers must
come from the same artefacts a reader can inspect -- if ``EVALUATION.md`` is
computed against an in-process object while the committed CSVs say something
else, the discrepancy is invisible. Reading the files closes that gap.

Amounts go through :func:`~ledgerloop.money.parse_minor_units` rather than
``int()``: the columns are already in paise, and routing them through the money
gate keeps the no-float invariant enforced at the one boundary where text
becomes money.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ledgerloop.generator.emitters import (
    GROUND_TRUTH_LINKS_FILE,
    GROUND_TRUTH_RECORDS_FILE,
    MANIFEST_FILE,
)
from ledgerloop.models.enums import (
    AnomalyClass,
    Difficulty,
    ExpectedStatus,
    LinkType,
    SplitName,
)
from ledgerloop.models.refs import RecordRef
from ledgerloop.models.truth import GroundTruth, GroundTruthLink, GroundTruthRecord
from ledgerloop.money import parse_minor_units

__all__ = ["DatasetManifest", "load_ground_truth", "load_manifest"]


@dataclass(frozen=True)
class DatasetManifest:
    """The dataset's own description of itself.

    Carries the identity a metric needs to be comparable to another metric:
    which split, which difficulty, which seed, and -- critically -- which
    generator version. A number produced against generator ``0.2.0`` is not
    comparable to one produced against ``0.3.0``, and the report prints the
    version so that is checkable rather than assumed.
    """

    split: SplitName
    difficulty: Difficulty
    seed: int
    generator_version: str
    counts: dict[str, int]
    scenario_draws: dict[AnomalyClass, int]
    money: dict[str, int]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload: object = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest(directory: Path) -> DatasetManifest:
    """Read ``manifest.json``."""
    payload = _read_json(directory / MANIFEST_FILE)
    draws_raw = payload.get("scenario_draws", {})
    money_raw = payload.get("money", {})
    counts_raw = payload.get("counts", {})
    return DatasetManifest(
        split=SplitName(payload["split"]),
        difficulty=Difficulty(payload["difficulty"]),
        seed=int(payload["seed"]),
        generator_version=str(payload["generator_version"]),
        counts={str(key): int(value) for key, value in counts_raw.items()},
        scenario_draws={
            AnomalyClass(key): int(value) for key, value in draws_raw.items()
        },
        money={
            str(key): parse_minor_units(str(value), field=f"manifest.money.{key}")
            for key, value in money_raw.items()
        },
    )


def _load_links(directory: Path) -> tuple[GroundTruthLink, ...]:
    return tuple(
        GroundTruthLink(
            link_type=LinkType(row["link_type"]),
            source_ref=RecordRef.parse(row["source_ref"]),
            target_ref=RecordRef.parse(row["target_ref"]),
            amount_minor=parse_minor_units(row["amount_paise"], field="link.amount_paise"),
            anomaly_class=AnomalyClass(row["anomaly_class"]),
        )
        for row in _read_csv(directory / GROUND_TRUTH_LINKS_FILE)
    )


def _load_records(directory: Path) -> tuple[GroundTruthRecord, ...]:
    return tuple(
        GroundTruthRecord(
            record_ref=RecordRef.parse(row["record_ref"]),
            expected_status=ExpectedStatus(row["expected_status"]),
            anomaly_class=AnomalyClass(row["anomaly_class"]),
            impact_minor=parse_minor_units(row["impact_paise"], field="record.impact_paise"),
            # The emitter writes `note or ""`, so an empty cell means None.
            # Reading it back as "" would make the round-trip inexact.
            note=row["note"] or None,
        )
        for row in _read_csv(directory / GROUND_TRUTH_RECORDS_FILE)
    )


def load_ground_truth(directory: Path) -> GroundTruth:
    """Reassemble :class:`GroundTruth` from the three files the generator wrote."""
    manifest = load_manifest(directory)
    return GroundTruth(
        split=manifest.split,
        difficulty=manifest.difficulty,
        seed=manifest.seed,
        generator_version=manifest.generator_version,
        links=_load_links(directory),
        records=_load_records(directory),
        scenario_draws=dict(manifest.scenario_draws),
    )
