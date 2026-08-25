"""Record references -- the addressing scheme for everything else.

Candidates, decisions, evidence items, exceptions and ground-truth links all
point at source records. They do it through :class:`RecordRef` rather than bare
ID strings so that ``"PAY-88301"`` can never be silently compared against an
order ID, and so every reference carries its type for display and traversal.
"""

from __future__ import annotations

from pydantic import field_validator

from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.enums import RecordType

__all__ = ["RecordRef"]

_SEPARATOR = ":"


class RecordRef(FrozenLedgerModel):
    """A typed pointer to one canonical record.

    Frozen and hashable, so refs work as dict keys and set members throughout
    the matcher -- the subset-sum solver and the graph rules both need
    membership tests over sets of payments.
    """

    record_type: RecordType
    record_id: str

    @field_validator("record_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("record_id must not be empty")
        if _SEPARATOR in value:
            raise ValueError(
                f"record_id must not contain {_SEPARATOR!r}; it would make the "
                "canonical key ambiguous"
            )
        return value

    @property
    def key(self) -> str:
        """Stable flat key, e.g. ``"payment:PAY-88301"``.

        Used for CSV columns, audit-log payloads and graph node IDs, where a
        nested object would be noise.
        """
        return f"{self.record_type.value}{_SEPARATOR}{self.record_id}"

    @classmethod
    def parse(cls, key: str) -> RecordRef:
        """Inverse of :attr:`key`."""
        type_part, separator, id_part = key.partition(_SEPARATOR)
        if not separator:
            raise ValueError(f"{key!r} is not a record key (expected 'type:id')")
        return cls(record_type=RecordType(type_part), record_id=id_part)

    def __str__(self) -> str:
        return self.key


def order_ref(record_id: str) -> RecordRef:
    return RecordRef(record_type=RecordType.ORDER, record_id=record_id)


def payment_ref(record_id: str) -> RecordRef:
    return RecordRef(record_type=RecordType.PAYMENT, record_id=record_id)


def settlement_ref(record_id: str) -> RecordRef:
    return RecordRef(record_type=RecordType.SETTLEMENT, record_id=record_id)


def bank_ref(record_id: str) -> RecordRef:
    return RecordRef(record_type=RecordType.BANK_TXN, record_id=record_id)


__all__ += ["bank_ref", "order_ref", "payment_ref", "settlement_ref"]
