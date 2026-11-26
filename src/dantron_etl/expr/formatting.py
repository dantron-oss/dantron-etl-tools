"""Writing one number as text, the way format_number() does.

This is the per-row writer behind format_number(): the digits grouped in threes, a fixed number of
places, and the separators the author stated.
"""

from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal, localcontext
from typing import Any, Final

# The one rounding format_number uses, whatever the process-wide decimal context says. Every field is
# stated, because a bare Context() copies the process-wide defaults, which can be changed too.
_ROUNDING: Final[Context] = Context(prec=MAX_PREC, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)


def rounded_half_away(value: Decimal, decimals: int) -> Decimal:
    """An exact value rounded half away from zero to ``decimals`` places, as round() rounds it."""
    with localcontext(_ROUNDING):
        return value.quantize(Decimal((0, (1,), -decimals)), rounding=ROUND_HALF_UP)


def _grouped(digits: str, separator: str) -> str:
    """The integer digits grouped in threes with `separator`; the sign is kept out, or -1832 reads -,832."""
    if not separator:
        return digits
    lead = len(digits) % 3 or 3
    return separator.join([digits[:lead], *(digits[at : at + 3] for at in range(lead, len(digits), 3))])


def written_as_text(value: Any, decimals: int, thousands: str, point: str, *, negative: bool | None = None) -> str | None:
    """One number written out with exactly the digits the column holds.

    A whole number is exact; a decimal is written as the double it really is. The value goes through an
    exact Decimal, which serves both. Infinity and NaN are written as null, like a null, so a cell never
    shows a plausible wrong figure. `negative` says whether to write a minus sign, for a caller that has
    already rounded a negative value to zero; left out, it is read from `value`.

    copy_abs() is used, not abs(), which rounds to the process-wide Decimal context and could move a figure.
    """
    number = Decimal(value)
    # Before the sign test, which raises on Decimal('NaN'); and Decimal(inf) would group as 'In,fin,ity'.
    if not number.is_finite():
        return None
    # The rounding is stated, so another thread's Decimal context cannot move a figure.
    with localcontext(_ROUNDING):
        rendered = format(number.copy_abs(), f".{decimals}f")
    whole, _, fraction = rendered.partition(".")
    text = _grouped(whole, thousands)
    if fraction:
        text = f"{text}{point}{fraction}"
    below_zero = number < 0 if negative is None else negative
    return f"-{text}" if below_zero else text
