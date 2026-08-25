"""Synthetic data generation -- ground truth first, data derived from it.

Two phases (see :mod:`ledgerloop.generator.world`):

1. :func:`~ledgerloop.generator.baseline.build_clean_world` builds a world where
   every rupee reconciles.
2. :mod:`~ledgerloop.generator.scenarios` breaks it in eleven labelled ways,
   each recording what it did.

:func:`~ledgerloop.generator.ground_truth.build_ground_truth` then reads truth
off those records, so the answer key is never reverse-engineered from the data.
"""

from ledgerloop.generator.generate import GeneratedDataset, generate, generate_to_disk
from ledgerloop.generator.ground_truth import build_ground_truth

__all__ = ["GeneratedDataset", "build_ground_truth", "generate", "generate_to_disk"]
