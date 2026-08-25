"""The similarity-search interface.

DECISION: ChromaDB and sentence-transformer embeddings are cut from the MVP.

PLAN.md §6.3 proposed embedding merchant-name variants into ChromaDB so kNN
could resolve ``RZRPAY SFTWR P L`` -> ``Razorpay Software Private Limited``.
Two reasons that is the wrong tool here:

1. **Sentence embeddings are weak at exactly this task.** MiniLM is trained on
   natural-language semantics and has no reason to place a vowel-dropped
   consonant skeleton near its expansion. What actually solves it is
   deterministic: uppercase, strip legal suffixes (``PVT``, ``LTD``, ``P L``),
   expand a small abbreviation table, then fuzzy-match the skeleton.
2. **The vocabulary is tiny and self-generated.** Merchant names come from our
   own generator -- on the order of tens of distinct names. A vector database
   for that is infrastructure without a job.

So T3 ships lexical-only, and ``semantic_score`` stays 0.0 in the feature
vector. The semantic path is scheduled as an *ablation row* rather than a
dependency: if it is added later, the honest outcome is a measured comparison,
and "we tried embeddings and lexical won" is a stronger result than a silent
dependency that never earned its place.

This module contains **no implementation** -- only the contract a
``ChromaVectorRepo`` would satisfy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

__all__ = ["SimilarityHit", "VectorRepo"]


class SimilarityHit(NamedTuple):
    """One retrieval result."""

    key: str
    score: float
    """Similarity in ``[0, 1]``, higher is closer.

    Normalised on purpose: raw cosine distance and raw fuzzy ratios are on
    different scales, and ``FeatureVector.semantic_score`` is bounded to
    ``[0, 1]`` so the blender sees a consistent range whatever backs it.
    """


@runtime_checkable
class VectorRepo(Protocol):
    """Nearest-neighbour lookup over short strings (merchant-name variants)."""

    def index(self, key: str, text: str, /, **metadata: object) -> None:
        ...

    def query(self, text: str, *, top_k: int = 5) -> Sequence[SimilarityHit]:
        """Nearest matches, ordered by descending score.

        Implementations must break score ties deterministically (by ``key``),
        so seeded runs stay reproducible.
        """
        ...

    def clear(self) -> None:
        ...
