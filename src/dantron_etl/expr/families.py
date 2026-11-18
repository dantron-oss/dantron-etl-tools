"""What a value holds, as far as the language can tell before a row is read.

A family is coarse: text, a number, a date, a truth value. Every type check in the language is
made on families. Where it matters, the exact type is worked out too, from the schema, from a
literal, or from the parts of a value built out of other values.
"""

import ast
import enum
from collections.abc import Sequence
from typing import Any, Final, NamedTuple

import polars as pl


class Family(enum.Enum):
    """The type families an operation may work within, named as an author would name them.

    Coarse on purpose, and the coarseness is the design. A whole number and a decimal are one
    family because separating them would refuse ``[qty] > 2.5``, which the engine answers perfectly
    well -- and a check stricter than the engine costs an author capability they cannot argue with.
    ``NULL`` belongs to every family, because a null is the one value that may meet anything.
    ``UNKNOWN`` is what a column has when no schema was supplied: unknown means unchecked, never
    refused, so a caller with no frame in hand still gets every other check.
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
    """The family a column belongs to, from what it holds.

    Anything this module does not recognise is ``UNKNOWN`` rather than a guess: an unrecognised
    dtype means the check has nothing true to say, and refusing on a guess would take capability
    away from a caller whose frame is merely unusual.
    """
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
    """The family a literal belongs to, which is knowable from the text with no schema at all.

    This is what lets ``'100' + '20'`` and ``concat([code], 1)`` be refused at the boundary a
    definition crosses first, where there is no frame to consult.
    """
    if value is None:
        return Family.NULL
    if isinstance(value, bool):
        return Family.BOOLEAN
    if isinstance(value, (int, float)):
        return Family.NUMBER
    if isinstance(value, str):
        return Family.TEXT
    return Family.DATE


def dtype_of_value(value: Any) -> pl.DataType:
    """The exact type a literal holds, beside the coarser family it belongs to.

    One check needs this much detail. A count of months may be 3 and may not be 1.5, and the
    family calls both of those a number.
    """
    if value is None:
        return pl.Null()
    if isinstance(value, bool):
        return pl.Boolean()
    if isinstance(value, int):
        return pl.Int64()
    if isinstance(value, float):
        return pl.Float64()
    if isinstance(value, str):
        return pl.String()
    return pl.Date()


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
    if isinstance(value, float):
        return "is a decimal"
    if isinstance(value, str):
        return "is text"
    return "is a date"


def compatible(one: Family, other: Family) -> bool:
    """Whether two operands may meet: same family, or one of them meets anything."""
    return one in WILDCARD_FAMILIES or other in WILDCARD_FAMILIES or one is other


class Typed(NamedTuple):
    """One compiled sub-expression, carrying what a refusal about it would need to say.

    ``node`` rather than a rendered string because rendering is only ever wanted on the way to an
    error, and rendering every subtree eagerly would do quadratic work on a healthy expression.
    ``holds`` is a phrase and not a family so that a column can report the dtype it actually holds:
    an author told "expected text" without being told what the column holds goes to the schema to
    find out why, which is a round trip the message should have saved them.

    ``expr`` is the column expression built so far.

    ``dtype`` is what this value holds, where the language can work that out. The family is
    coarse on purpose: a whole number and a decimal are both numbers, and almost every check here
    is happier that way. One check is not. add_months() moves a date by whole months, so it has to
    tell 3 from 1.5, in a column as well as in a line of the file.

    A leaf fills it in from what it is: a column from the schema, a literal from the value written.
    A value built out of other values fills it in from its parts. Arithmetic and coalesce() work
    theirs out, abs() and round() hand their subject's type on, and year() and len() always give a
    whole number. None means the language could not work the type out, which is what a count of
    months is refused for: a count nobody can show is whole is not one to move a date by.
    """

    expr: Any
    family: Family
    node: ast.expr
    holds: str
    dtype: pl.DataType | None = None
