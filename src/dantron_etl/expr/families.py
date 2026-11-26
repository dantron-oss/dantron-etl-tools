"""What a value holds, as far as the language can tell before a row is read.

A family is coarse: text, a number, a date, a truth value. Every type check in the language is
made on families. Where it matters, the exact type is worked out too, from the schema, from a
literal, or from the parts of a value built out of other values.
"""

import ast
import enum
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Final, NamedTuple

import polars as pl


class Family(enum.Enum):
    """The kinds of value an operation works with: text, number, date, true/false.

    Coarse on purpose: whole numbers and decimals are one family, so `[qty] > 2.5` is allowed. NULL fits
    every family. UNKNOWN (no schema) means unchecked, never refused.
    """

    TEXT = "text"
    NUMBER = "a number"
    DATE = "a date"
    BOOLEAN = "a truth value"
    NULL = "null"
    UNKNOWN = "an unknown type"


# The two families that meet anything: a null value belongs to every family, and an unchecked one
# cannot be argued with. Both are skipped by every requirement below.
WILDCARD_FAMILIES: Final[frozenset[Family]] = frozenset({Family.NULL, Family.UNKNOWN})


def family_of_dtype(dtype: pl.DataType) -> Family:
    """The family a column belongs to; UNKNOWN for a dtype this module does not recognise, rather than a guess."""
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        return Family.TEXT
    if dtype == pl.Boolean:
        return Family.BOOLEAN
    if dtype.is_numeric():
        return Family.NUMBER
    if dtype in (pl.Date, pl.Datetime):
        return Family.DATE
    if dtype == pl.Null:
        return Family.NULL
    return Family.UNKNOWN


def family_of_value(value: Any) -> Family:
    """The family a literal belongs to, known from the text alone, so `'100' + '20'` is refused with no frame."""
    if value is None:
        return Family.NULL
    if isinstance(value, bool):
        return Family.BOOLEAN
    if isinstance(value, (int, float, Decimal)):
        return Family.NUMBER
    if isinstance(value, str):
        return Family.TEXT
    return Family.DATE


def dtype_of_value(value: Any) -> pl.DataType:
    """The exact type a literal holds, for the one check that must tell 3 from 1.5 (add_months)."""
    if value is None:
        return pl.Null()
    if isinstance(value, bool):
        return pl.Boolean()
    if isinstance(value, int):
        return pl.Int64()
    if isinstance(value, float):
        return pl.Float64()
    if isinstance(value, Decimal):
        return pl.Decimal(38, places_of(value))
    if isinstance(value, str):
        return pl.String()
    return pl.Date()


def places_of(value: Decimal) -> int:
    """The places a typed number was written with: 1.10 has 2, 7 has 0, and 1e-9 has 9."""
    exponent = value.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def combined_dtype(dtypes: Sequence[pl.DataType | None]) -> pl.DataType | None:
    """What a value built out of other values holds, worked out from its parts.

    Parts that all hold the same type give that type. Whole numbers mixed together give a whole
    number, and one decimal among them makes the answer a decimal. A null part is skipped, because
    there is no value in it to widen anything.

    None means the answer is not known. One unknown part is enough to make it unknown, which is the
    honest answer rather than a guess.
    """
    if any(dtype is None for dtype in dtypes):
        return None
    stated = [dtype for dtype in dtypes if dtype != pl.Null]
    if not stated:
        return pl.Null()
    if all(dtype == stated[0] for dtype in stated):
        return stated[0]
    if all(dtype.is_integer() for dtype in stated):
        return pl.Int64()
    if all(dtype.is_numeric() for dtype in stated):
        return pl.Float64()
    return None


def arithmetic_dtype(
    operation: type[ast.AST], left: pl.DataType | None, right: pl.DataType | None
) -> pl.DataType | None:
    """What an arithmetic result holds, worked out from its two operands.

    Division is the one to watch. It gives a decimal even when both sides are whole, because it is
    true division here as everywhere else in this language: 3 / 2 is 1.5. The other three operators
    keep two whole numbers whole.
    """
    if operation is ast.Div:
        return pl.Float64()
    return combined_dtype([left, right])


def holding_phrase(value: Any) -> str:
    """How a refusal describes what a literal is, in the words an author used to write it."""
    if value is None:
        return "is null"
    if isinstance(value, bool):
        return "is a truth value"
    if isinstance(value, int):
        return "is a whole number"
    if isinstance(value, (float, Decimal)):
        return "is a decimal"
    if isinstance(value, str):
        return "is text"
    return "is a date"


def compatible(one: Family, other: Family) -> bool:
    """Whether two operands may meet: same family, or one of them meets anything."""
    return one in WILDCARD_FAMILIES or other in WILDCARD_FAMILIES or one is other


class Typed(NamedTuple):
    """One compiled part of an expression, with what a refusal about it would need to say.

    `node` is kept, not a rendered string, because rendering is only needed for an error. `holds` is a
    phrase, so a refusal can say what a column actually holds. `expr` is the polars expression so far.
    `dtype` is the exact type where it can be worked out (from the schema, the literal, or the parts), or
    None; add_months() refuses a count of months whose type is None or a decimal. `constant` is the exact
    value of a typed number, as written; it is used in place of `expr` beside an exact column.
    """

    expr: Any
    family: Family
    node: ast.expr
    holds: str
    dtype: pl.DataType | None = None
    constant: Decimal | None = None
