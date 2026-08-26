"""The in-memory entity graph. The real T4 backend.

``graph/interface.py`` settled the argument at Step 0: PLAN.md §4 required the
NetworkX fallback to produce *identical decisions* to Neo4j on the fixture set,
which is an admission that the database contributes no decision quality. The
four T4 rules are adjacency lookups over a few hundred nodes.

This module is the consequence. It satisfies :class:`~ledgerloop.graph.
interface.GraphRepo` with plain dictionaries, and NetworkX is not a dependency
either -- an adjacency list is what NetworkX would give us, and writing it
costs about as much as importing it. The Protocol is what keeps a
``NetworkXGraphRepo`` or ``Neo4jGraphRepo`` a drop-in rather than a rewrite.

DETERMINISM IS THE HARD REQUIREMENT
------------------------------------
The Protocol says every query must return results in insertion order, and that
is not a nicety: a rule that iterates neighbours in ``set`` order would make
seeded runs irreproducible and break the golden regression test. So adjacency
is kept in insertion-ordered lists with a companion set for membership, rather
than in sets alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ledgerloop.models.enums import LinkType, RecordType
from ledgerloop.models.refs import RecordRef

__all__ = ["Edge", "MemoryGraphRepo"]


@dataclass(frozen=True)
class Edge:
    """One typed, directed edge. Satisfies the ``GraphEdge`` Protocol."""

    source_ref: RecordRef
    target_ref: RecordRef
    link_type: LinkType
    attributes: dict[str, object] = field(default_factory=dict)


class MemoryGraphRepo:
    """Adjacency over ``Order -> Payment -> Settlement -> BankTxn``."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, object]] = {}
        self._order: list[RecordRef] = []
        self._out: dict[str, list[Edge]] = {}
        self._edges: dict[LinkType, list[Edge]] = {}
        self._seen: set[tuple[str, str, LinkType]] = set()

    # -- construction -----------------------------------------------------

    def add_node(self, ref: RecordRef, /, **attributes: object) -> None:
        if ref.key not in self._nodes:
            self._nodes[ref.key] = {}
            self._order.append(ref)
        self._nodes[ref.key].update(attributes)

    def add_edge(
        self,
        source: RecordRef,
        target: RecordRef,
        link_type: LinkType,
        /,
        **attributes: object,
    ) -> None:
        """Insert a typed edge. Repeating one is a no-op, not a duplicate.

        Both endpoints are created on demand: a rule that had to remember to
        add nodes first would eventually forget, and a missing node is a silent
        traversal dead end rather than an error.
        """
        self.add_node(source)
        self.add_node(target)
        signature = (source.key, target.key, link_type)
        if signature in self._seen:
            return
        self._seen.add(signature)
        edge = Edge(
            source_ref=source, target_ref=target, link_type=link_type, attributes=attributes
        )
        self._out.setdefault(source.key, []).append(edge)
        self._edges.setdefault(link_type, []).append(edge)

    def clear(self) -> None:
        self._nodes.clear()
        self._order.clear()
        self._out.clear()
        self._edges.clear()
        self._seen.clear()

    # -- queries ----------------------------------------------------------

    def nodes(self, record_type: RecordType | None = None) -> Sequence[RecordRef]:
        """Every node, in insertion order, optionally filtered by type."""
        if record_type is None:
            return tuple(self._order)
        return tuple(ref for ref in self._order if ref.record_type is record_type)

    def attributes(self, ref: RecordRef) -> dict[str, object]:
        return dict(self._nodes.get(ref.key, {}))

    def neighbours(
        self, ref: RecordRef, link_type: LinkType | None = None
    ) -> Sequence[RecordRef]:
        return tuple(
            edge.target_ref
            for edge in self._out.get(ref.key, ())
            if link_type is None or edge.link_type is link_type
        )

    def has_edge(self, source: RecordRef, target: RecordRef, link_type: LinkType) -> bool:
        return (source.key, target.key, link_type) in self._seen

    def edges_of_type(self, link_type: LinkType) -> Sequence[Edge]:
        return tuple(self._edges.get(link_type, ()))

    def path_exists(self, source: RecordRef, target: RecordRef) -> bool:
        """Whether any directed path connects the two records.

        Breadth-first over the insertion-ordered adjacency, so the traversal
        order -- and therefore any rule that short-circuits on it -- is stable.
        Backs *path closure*.
        """
        if source.key == target.key:
            return True
        seen = {source.key}
        queue = [source]
        while queue:
            current = queue.pop(0)
            for edge in self._out.get(current.key, ()):
                key = edge.target_ref.key
                if key == target.key:
                    return True
                if key not in seen:
                    seen.add(key)
                    queue.append(edge.target_ref)
        return False

    def siblings_in_settlement(self, settlement: RecordRef) -> Sequence[RecordRef]:
        """Payments belonging to one settlement.

        Read off the reverse of ``PAYMENT_SETTLED_IN``, which the sources
        assert by nesting -- so this is lookup, not inference. Backs *sibling
        completion*.
        """
        return tuple(
            edge.source_ref
            for edge in self._edges.get(LinkType.PAYMENT_SETTLED_IN, ())
            if edge.target_ref.key == settlement.key
        )

    def consumed_credits(self) -> Iterable[RecordRef]:
        """Bank credits marked as fully accounted for.

        The flag is set by T4's exclusivity rule, which is the only thing that
        knows how much of a credit has been absorbed. Backs *exclusivity
        pruning*: a consumed credit cannot take on more payments.
        """
        return tuple(
            ref
            for ref in self._order
            if ref.record_type is RecordType.BANK_TXN
            and bool(self._nodes[ref.key].get("consumed", False))
        )

    def mark_consumed(self, ref: RecordRef, *, consumed: bool = True) -> None:
        self.add_node(ref, consumed=consumed)

    def __len__(self) -> int:
        return len(self._order)
