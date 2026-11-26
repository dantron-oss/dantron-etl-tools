"""Arithmetic on exact decimals, built so that nothing rounds and nothing too large goes unseen.

Every exact answer here is a Decimal with 38 digits in all and a stated number of places. Each
builder gives the answer and a true/false column that is true on a row whose answer would need
more than 38 digits. On such a row the answer is null, and the engine is never asked for it, so it
cannot fail or wrap round. The compiler turns the true/false column into a check the step runs
before it writes anything, which stops the step and names the rows.

Why: docs/design/expressions.md#exact-decimals
"""

from decimal import Decimal
from typing import Final, NamedTuple

import polars as pl

from ..column_types import EXACT_DIGITS, exact_dtype

#: What every refusal about an answer too large to hold tells the author to do.
TOO_LARGE_REMEDY: Final[str] = (
    "An exact decimal holds 38 digits in all, counting its places. Round an operand first with round(), so "
    "the answer needs fewer places, or convert the columns to Float64 with the cast_columns tool for "
    "figures that large"
)


class Exact(NamedTuple):
    """One exact answer: its column, its places, and where it is too large to hold."""

    expr: pl.Expr
    places: int
    too_large: pl.Expr


def places_of(dtype: pl.DataType) -> int:
    """The places of an exact dtype; a whole number counts as 0."""
    return dtype.scale if isinstance(dtype, pl.Decimal) else 0  # type: ignore[attr-defined]


def literal(value: Decimal) -> pl.Expr:
    """A typed number as an exact literal with the places it was written with."""
    exponent = value.as_tuple().exponent
    places = max(0, -exponent) if isinstance(exponent, int) else 0
    return pl.lit(value, dtype=exact_dtype(places))


def holds(value: Decimal) -> bool:
    """Whether an exact decimal can hold a typed number as written: 38 digits in all, at most."""
    _, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        return False
    places = max(0, -exponent)
    whole = max(len(digits) + exponent, 0)
    return places <= EXACT_DIGITS and whole + places <= EXACT_DIGITS


def largest(places: int) -> pl.Expr:
    """The largest value an exact decimal with these places holds: 38 nines."""
    # Built from its digits: arithmetic on a Decimal would round it to the default context's 28 digits.
    return pl.lit(Decimal((0, (9,) * EXACT_DIGITS, -places)), dtype=exact_dtype(places))


def at_places(value: pl.Expr, dtype: pl.DataType, places: int) -> tuple[pl.Expr, pl.Expr]:
    """``value`` written with ``places`` places, and where it has too many whole digits for that.

    A whole number becomes exact with 0 places first. More places never round. A value too large to
    hold at those places is null, and the second column is true there.
    """
    if dtype == pl.Decimal(EXACT_DIGITS, places):
        return value, pl.lit(False)
    written = value.cast(exact_dtype(places), strict=False)
    return written, value.is_not_null() & written.is_null()


def add(left: pl.Expr, left_dtype: pl.DataType, right: pl.Expr, right_dtype: pl.DataType, *, subtract: bool) -> Exact:
    """``left + right`` or ``left - right`` with the larger of the two places, exactly.

    Each side is written at those places first. The answer is too large when either side does not
    fit there, or when both lean the same way and together pass 38 digits.
    """
    places = max(places_of(left_dtype), places_of(right_dtype))
    first, first_misfit = at_places(left, left_dtype, places)
    second, second_misfit = at_places(right, right_dtype, places)
    if subtract:
        second = -second
    same_way = ((first > 0) & (second > 0)) | ((first < 0) & (second < 0))
    # Worked out on sizes that each fit, so the check itself cannot pass 38 digits.
    past = same_way & (first.abs() > largest(places) - second.abs())
    too_large = first_misfit | second_misfit | past.fill_null(False)
    return Exact(_unless(too_large, first) + _unless(too_large, second), places, too_large)


def _unless(too_large: pl.Expr, value: pl.Expr) -> pl.Expr:
    """``value``, or null on a row whose answer is too large, so the engine never works it out."""
    return pl.when(too_large).then(None).otherwise(value)
