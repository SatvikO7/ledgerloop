"""Top-level generation: seed in, dataset out.

Ground truth is produced *from the generator's own record of what it did*, never
inferred from the emitted files. That ordering is the whole reason this step
comes before any matcher exists: a bug here would silently invalidate every
metric downstream, and a truth set reverse-engineered from the data can only
ever agree with the data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ledgerloop.config import GeneratorConfig
from ledgerloop.generator.baseline import build_clean_world
from ledgerloop.generator.emitters import write_dataset
from ledgerloop.generator.ground_truth import build_ground_truth
from ledgerloop.generator.scenarios import apply_scenario
from ledgerloop.generator.world import DraftWorld
from ledgerloop.models.enums import AnomalyClass
from ledgerloop.models.truth import GroundTruth

__all__ = ["GeneratedDataset", "generate", "generate_to_disk"]


@dataclass(frozen=True)
class GeneratedDataset:
    """A generated world and its truth, before anything touches the disk."""

    config: GeneratorConfig
    world: DraftWorld
    truth: GroundTruth

    @property
    def conservation_residual_minor(self) -> int:
        """Money unaccounted for after declared anomalies. **Must be zero.**

        ``sum(settlement-linked credits) - sum(declared nets) - sum(declared deltas)``

        This is the property test's subject and the reason every scenario has to
        declare its ``bank_delta_minor``. A non-zero residual means a scenario
        moved money without saying so, and every metric computed on this dataset
        would be measuring the wrong world.
        """
        return (
            self.world.settled_credit_total_minor()
            - self.world.declared_net_total_minor()
            - self.world.declared_bank_delta_minor()
        )


def _draw_order(rng: random.Random, config: GeneratorConfig) -> AnomalyClass:
    """Sample one scenario from the configured prevalence.

    Population is sorted by enum value so the draw sequence depends only on the
    seed, never on dict iteration order.
    """
    population = sorted(config.prevalence, key=lambda anomaly: anomaly.value)
    weights = [config.prevalence[anomaly] for anomaly in population]
    return rng.choices(population, weights=weights, k=1)[0]


def _seed_one_of_each(world: DraftWorld, rng: random.Random) -> None:
    """Place one effect per anomaly class before the prevalence draw runs.

    Only under ``ensure_class_coverage``. A 60-order dev set produces roughly
    eight settlements, so its ``amount`` aspects saturate quickly and a fixture
    can end up containing no chargeback at all -- silently retiring a whole
    branch of the matcher from the regression suite.

    **Seeding runs first, not last.** A pass that ran after the draws would be
    competing for aspects that are already taken, which is exactly the situation
    it exists to escape.

    Seeded effects are **not** counted as draws: ``scenario_draws`` keeps
    describing the sampled distribution, so the prevalence check stays truthful
    about what the dial produced.
    """
    for anomaly in sorted(AnomalyClass, key=lambda item: item.value):
        if anomaly is AnomalyClass.CLEAN:
            continue
        for order in world.orders:
            if apply_scenario(world, rng, anomaly, order.order_id):
                break


def generate(config: GeneratorConfig) -> GeneratedDataset:
    """Build a dataset in memory.

    Two RNG streams, seeded separately. The baseline world and the scenario
    draws advance independently, so changing the prevalence dial does not
    reshuffle the underlying orders -- which is what makes the three difficulty
    columns comparable rather than three unrelated datasets.
    """
    # The split name is mixed into the seed. Without it, `train` and `test` at
    # the same seed would share their first 300 orders -- the smaller split
    # being a literal prefix of the larger. Every "held-out" number would then
    # be reported on data the blender had already been fitted on, which is the
    # exact leak the three-way split discipline exists to prevent.
    baseline_rng = random.Random(f"{config.seed}:{config.split.value}:baseline")
    scenario_rng = random.Random(f"{config.seed}:{config.split.value}:scenario")

    world = build_clean_world(baseline_rng, config.effective_order_count)

    if config.ensure_class_coverage:
        _seed_one_of_each(world, scenario_rng)

    for order in list(world.orders):
        anomaly = _draw_order(scenario_rng, config)
        world.record_draw(anomaly)
        apply_scenario(world, scenario_rng, anomaly, order.order_id)

    truth = build_ground_truth(
        world,
        split=config.split,
        difficulty=config.difficulty,
        seed=config.seed,
        generator_version=config.generator_version,
    )
    return GeneratedDataset(config=config, world=world, truth=truth)


def generate_to_disk(config: GeneratorConfig, directory: Path) -> GeneratedDataset:
    """Generate and write all five files plus the manifest."""
    dataset = generate(config)
    write_dataset(directory, dataset.world, dataset.truth)
    return dataset
