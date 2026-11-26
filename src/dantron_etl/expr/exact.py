"""Arithmetic on exact decimals, built so that nothing rounds and nothing too large goes unseen.

Every exact answer here is a Decimal with 38 digits in all and a stated number of places. Each
builder gives the answer and a true/false column that is true on a row whose answer would need
more than 38 digits. On such a row the answer is null, and the engine is never asked for it, so it
cannot fail or wrap round. The compiler turns the true/false column into a check the step runs
before it writes anything, which stops the step and names the rows.

Why: docs/design/expressions.md#exact-decimals
"""

from decimal import Decimal
from typing import Any, Final, NamedTuple

import polars as pl

from ..column_types import EXACT_DIGITS, exact_dtype

#: What every refusal about an answer too large to hold tells the author to do.
TOO_LARGE_REMEDY: Final[str] = (
    "An exact decimal holds 38 digits in all, counting its places. Round an operand first with round(), so "
    "the answer needs fewer places, or convert the columns to Float64 with the cast_columns tool for "
    "figures that large"
)


class Exact(NamedTuple):
    """One exact answer: its column, its places, and where it is too large to hold.

    `dtype` is the answer's type where it is not an exact decimal with those places, as for nullif(),
    which gives its first value back in that value's own type.
    """

    expr: pl.Expr
    places: int
    too_large: pl.Expr
    dtype: pl.DataType | None = None


#: A value of an exact calculation: its column and its type, a whole number or an exact decimal.
Part = tuple[pl.Expr, pl.DataType]


def places_of(dtype: pl.DataType) -> int:
    """The places of an exact dtype; a whole number counts as 0."""
    return dtype.scale if isinstance(dtype, pl.Decimal) else 0  # type: ignore[attr-defined]


def literal(value: Decimal) -> pl.Expr:
    """A typed number as an exact literal with the places it was written with."""
    return pl.lit(value, dtype=exact_dtype(_places(value)))


def holds(value: Decimal) -> bool:
    """Whether an exact decimal can hold a typed number as written: 38 digits in all, at most."""
    places = _places(value)
    return places <= EXACT_DIGITS and _fits(value, places)


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


def multiply(left: pl.Expr, left_dtype: pl.DataType, right: pl.Expr, right_dtype: pl.DataType) -> Exact:
    """``left * right`` with the two places added, exactly: 812.40 * 0.0725 is 58.899000.

    The caller refuses a product whose places pass 38 before this is built. The digits each side
    holds, places included, are multiplied as whole numbers, and the point is put back. The engine's
    own product is not used: it first writes both sides at the larger places, which rounds, and
    gives a blank where a large value meets a small one. The answer is too large when the two
    whole numbers multiply to 10^38 or more (see `_product_reaches`).
    """
    places = places_of(left_dtype) + places_of(right_dtype)
    first, second = _digits_held(left, left_dtype), _digits_held(right, right_dtype)
    too_large = _product_reaches(first.abs(), second.abs())
    zero = pl.lit(0, dtype=pl.Int128)
    product = pl.when(too_large).then(None).otherwise(
        pl.when(too_large).then(zero).otherwise(first) * pl.when(too_large).then(zero).otherwise(second)
    )
    return Exact(_with_places(product, places), places, too_large)


#: 10^19, where a whole number of up to 38 digits is split in two, so each half times the other fits.
_HALF: Final[int] = 10**19


def _digits_held(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """The digits a value holds, places included, as a whole number with its sign: -812.40 gives -81240.

    Read from the value's own text, which is exact, into the engine's 128-bit whole numbers, which
    hold 38 digits.
    """
    if isinstance(dtype, pl.Decimal):
        value = value.cast(pl.String).str.replace(".", "", literal=True)
    return value.cast(pl.Int128)


def _with_places(number: pl.Expr, places: int) -> pl.Expr:
    """A 128-bit whole number read as digits with ``places`` of them after the point: 58899000 at 6 is 58.899000.

    Done through text, which is exact. The number must be below 10^38.
    """
    if places == 0:
        return number.cast(pl.String).cast(exact_dtype(0))
    digits = number.abs().cast(pl.String).str.zfill(places + 1)
    sign = pl.when(number < 0).then(pl.lit("-")).otherwise(pl.lit(""))
    text = pl.concat_str([sign, digits.str.head(-places), pl.lit("."), digits.str.tail(places)])
    return text.cast(exact_dtype(places))


def _product_reaches(first: pl.Expr, second: pl.Expr) -> pl.Expr:
    """True where ``first * second`` is 10^38 or more. Both are below 10^38 and not negative.

    Each is split into a high and a low half at 10^19. When both high halves are above zero, the
    product is at least 10^38. Otherwise at most one cross term is left, below 10^38, and the product
    is ``cross * 10^19 + low``, which reaches 10^38 exactly when ``cross + low // 10^19`` reaches 10^19.
    No step passes what a 128-bit whole number holds.
    """
    half = pl.lit(_HALF, dtype=pl.Int128)
    first_high, first_low = first // half, first % half
    second_high, second_low = second // half, second % half
    both_high = (first_high > 0) & (second_high > 0)
    zero = pl.lit(0, dtype=pl.Int128)
    first_high = pl.when(both_high).then(zero).otherwise(first_high)
    second_high = pl.when(both_high).then(zero).otherwise(second_high)
    cross = first_high * second_low + first_low * second_high
    reaches = both_high | (cross + (first_low * second_low) // half >= half)
    return reaches.fill_null(False)


def coalesce(*parts: Part) -> Exact:
    """The first value that is not null, with the largest places of them all.

    Too large where the value chosen has too many whole digits to be written at those places.
    """
    places = max(places_of(dtype) for _, dtype in parts)
    written = [at_places(value, dtype, places) for value, dtype in parts]
    chosen_misfit = pl.coalesce([pl.when(value.is_not_null()).then(misfit) for (value, _), (_, misfit) in zip(parts, written)])
    return Exact(pl.coalesce([value for value, _ in written]), places, chosen_misfit.fill_null(False))


def least(*parts: Part) -> Exact:
    """The smallest value, nulls skipped, with the largest places of them all."""
    return _ordered(pl.min_horizontal, parts)


def greatest(*parts: Part) -> Exact:
    """The largest value, nulls skipped, with the largest places of them all."""
    return _ordered(pl.max_horizontal, parts)


def _ordered(pick: Any, parts: tuple[Part, ...]) -> Exact:
    places = max(places_of(dtype) for _, dtype in parts)
    written = [at_places(value, dtype, places) for value, dtype in parts]
    too_large = pl.any_horizontal([misfit for _, misfit in written])
    return Exact(pick([value for value, _ in written]), places, too_large)


def nullif(subject: Part, other: Part) -> Exact:
    """Null where the first value equals the second, otherwise the first, in its own type."""
    (value, dtype), (against, _) = subject, other
    return Exact(pl.when(value == against).then(None).otherwise(value), places_of(dtype), pl.lit(False), dtype)


def remainder(dividend: Part, divisor: Part) -> Exact:
    """The remainder of the first value divided by the second, with the first one's sign, as SQL's MOD.

    Both are written at the larger places, and the remainder is worked out on the digits they hold,
    which is exact. A zero divisor gives null here; the step has already stopped on it.
    """
    places = max(places_of(dividend[1]), places_of(divisor[1]))
    first, first_misfit = at_places(*dividend, places)
    second, second_misfit = at_places(*divisor, places)
    held, by = _digits_held(first, exact_dtype(places)), _digits_held(second, exact_dtype(places)).abs()
    by = pl.when(by == 0).then(None).otherwise(by)
    left = held.abs() % by
    # The engine has no minus sign for a 128-bit whole number, so it is taken from zero.
    signed = pl.when(held < 0).then(pl.lit(0, dtype=pl.Int128) - left).otherwise(left)
    return Exact(_with_places(signed, places), places, first_misfit | second_misfit)


def _places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _fits(value: Decimal, places: int) -> bool:
    """Whether a typed number can be written with ``places`` places in 38 digits."""
    _, digits, exponent = value.as_tuple()
    return isinstance(exponent, int) and max(len(digits) + exponent, 0) + places <= EXACT_DIGITS


def _unless(too_large: pl.Expr, value: pl.Expr) -> pl.Expr:
    """``value``, or null on a row whose answer is too large, so the engine never works it out."""
    return pl.when(too_large).then(None).otherwise(value)
