"""The graph repository interface.

DECISION: Neo4j is cut from the MVP; NetworkX is the real implementation.

PLAN.md §4 Phase 4 required the NetworkX fallback to produce *identical
decisions* to Neo4j on the fixture set. That acceptance criterion is an
admission that the graph database contributes zero decision quality -- it buys
a container, a driver, a query dialect and a health check, in exchange for
output that must match the in-memory version exactly.

The four T4 rules (sibling completion, path closure, exclusivity pruning, ring
detection) are constraint propagation over an adjacency structure, which is a
few dozen lines in memory. So the in-memory implementation ships, and this
Protocol exists so a ``Neo4jGraphRepo`` can be added later without touching a
single call site. That is a better story than requiring a graph database:
"swappable backend, zero-infra default".

This module deliberately contains **no implementation**. Step 0 fixes the
contract; the NetworkX repository lands with T4.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ledgerloop.models.enums import LinkType
from ledgerloop.models.refs import RecordRef

__all__ = ["GraphEdge", "GraphRepo"]


class GraphEdge(Protocol):
    """Minimal shape of an edge as the rules consume it."""

    @property
    def source_ref(self) -> RecordRef: ...

    @property
    def target_ref(self) -> RecordRef: ...

    @property
    def link_type(self) -> LinkType: ...


@runtime_checkable
class GraphRepo(Protocol):
    """Entity graph over ``Order -> Payment -> Settlement -> BankTxn``.

    Implementations must be deterministic: given the same nodes and edges
    inserted in the same order, every query returns results in the same order.
    Non-deterministic traversal order would make seeded runs irreproducible and
    break the golden regression test.
    """

    def add_node(self, ref: RecordRef, /, **attributes: object) -> None:
        """Insert or update a node."""
        ...

    def add_edge(
        self,
        source: RecordRef,
        target: RecordRef,
        link_type: LinkType,
        /,
        **attributes: object,
    ) -> None:
        """Insert a typed, directed edge."""
        ...

    def neighbours(
        self, ref: RecordRef, link_type: LinkType | None = None
    ) -> Sequence[RecordRef]:
        """Direct successors of ``ref``, optionally filtered by edge type."""
        ...

    def has_edge(self, source: RecordRef, target: RecordRef, link_type: LinkType) -> bool:
        ...

    def edges_of_type(self, link_type: LinkType) -> Sequence[GraphEdge]:
        ...

    def path_exists(self, source: RecordRef, target: RecordRef) -> bool:
        """Whether any directed path connects the two records.

        Backs the *path closure* rule: ``Order -> Payment -> Settlement`` known
        and ``Settlement -> BankTxn`` known implies ``Order -> BankTxn``.
        """
        ...

    def siblings_in_settlement(self, settlement: RecordRef) -> Sequence[RecordRef]:
        """Payments belonging to one settlement.

        Backs *sibling completion*: when most of a settlement's payments are
        matched to bank credit B, the remainder are constrained to B.
        """
        ...

    def consumed_credits(self) -> Iterable[RecordRef]:
        """Bank credits already fully accounted for.

        Backs *exclusivity*, which prunes the T2 search space -- a credit that
        is fully consumed cannot absorb further payments.
        """
        ...

    def clear(self) -> None:
        """Drop all nodes and edges. Required so tests share no state."""
        ...
