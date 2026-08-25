"""Shared Pydantic configuration and the money-field annotation.

Two things live here because every other model file needs them:

1. :class:`LedgerModel` -- the strict base config used by every contract.
2. :data:`MinorUnits` -- the annotated ``int`` type that enforces the
   no-float-in-the-money-path invariant *at the schema boundary*, so a float
   cannot enter a model even via ``model_validate`` on parsed JSON.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import core_schema

from ledgerloop.money import assert_minor

__all__ = ["FrozenLedgerModel", "LedgerModel", "MinorUnits"]


class LedgerModel(BaseModel):
    """Base for mutable contracts (state that accumulates during a run).

    ``extra="forbid"`` matters more than it looks: LLM output is validated
    against these schemas, and silently accepting an unexpected key is exactly
    how a hallucinated field ends up ignored instead of caught.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )


class FrozenLedgerModel(LedgerModel):
    """Base for immutable contracts.

    Used for anything that must be hashable or must not drift after creation:
    record references, ground truth, audit events, and :class:`RunConfig`.
    Append-only auditing (PLAN.md §3.2) depends on decisions never being
    mutated in place -- a revision writes a new record instead.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )


class _MinorUnitsAnnotation:
    """Validator that routes an ``int`` field through :func:`assert_minor`.

    Pydantic would happily coerce ``499900.0`` to ``499900`` for a plain ``int``
    field in non-strict mode, and ``True`` to ``1``. Both are breaches of the
    money invariant that would pass schema validation silently. This annotation
    makes the invariant part of the type, so every money field on every model
    is protected without each model having to remember a validator.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        del source_type, handler
        return core_schema.no_info_plain_validator_function(
            lambda value: assert_minor(value, field="minor_units"),
            serialization=core_schema.simple_ser_schema("int"),
        )


#: An ``int`` counting minor units, guarded against ``float`` and ``bool``.
#:
#: Serialises as a plain JSON integer, so ``model_validate_json(model_dump_json())``
#: round-trips exactly. Text-to-int conversion happens once, explicitly, at the
#: ingest boundary via :func:`~ledgerloop.money.parse_minor_units` or
#: :func:`~ledgerloop.money.parse_major_to_minor`; models themselves accept only
#: ``int``, so there is no second, implicit coercion path to reason about.
MinorUnits = Annotated[int, _MinorUnitsAnnotation()]
