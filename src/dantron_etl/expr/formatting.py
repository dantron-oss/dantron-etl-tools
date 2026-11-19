"""Writing one number as text, the way format_number() does.

This is the per-row writer behind format_number(): the digits grouped in threes, a fixed number of
places, and the separators the author stated.
"""

from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Any, Final

# The one rounding format_number uses, whatever the process-wide decimal context says. Every field is
# stated, because a bare Context() copies the process-wide defaults, which can be changed too.
_ROUNDING: Final[Context] = Context(prec=MAX_PREC, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)


def _grouped(digits: str, separator: str) -> str:
    """The integer digits split into threes from the right and rejoined with ``separator``.

    The sign is not passed in, because a minus is not one of the digits being grouped: grouping a
    rendered '-1832' from the right gives '-,832' -- a figure that still reads like a number and
    is spotted by nobody.
    """
    if not separator:
        return digits
    lead = len(digits) % 3 or 3
    return separator.join([digits[:lead], *(digits[at : at + 3] for at in range(lead, len(digits), 3))])


def written_as_text(value: Any, decimals: int, thousands: str, point: str, *, negative: bool | None = None) -> str | None:
    """One number written out, faithful to the value the column holds.

    "Faithful to the value" is the single rule, and it has two readings because there are two ways
    a column holds a figure. A whole number is exact, so 9007199254740993 is written with those
    digits. A decimal is held as a double, so the digits written are the double's: the same
    9007199254740993 written as a decimal is held as ...992 and reads that way. Neither is a
    rounding choice made here -- each is the value itself, asked to describe itself to as many
    places as the line asked for.

    Every value goes through an exact decimal on the way, which is what makes both readings come
    out right from one piece of code: it converts a whole number without going near a double, and
    it converts a double to the digits that double really is, rather than to the shorter decimal
    somebody typed to produce it.

    **A figure that is not a number has no written form, so it is written as null.** inf, -inf and
    nan arrive here in the ordinary way -- '/' is true division on doubles, so [total] / [count] is
    inf on the row where the count is zero and 0.0 / 0.0 is nan -- and the answer is the one a null
    already gets, for the same reason: whoever reads the cell has to be able to tell "no figure"
    from a figure, and never be handed a plausible-looking wrong one.

    ``negative`` says whether to write a minus sign, for a caller that rounded the value first and
    so may hand over a zero that was below zero before it was rounded. Left out, the sign is read
    from ``value``.

    copy_abs() and not abs(), because abs() on a Decimal is a *context* operation: it rounds its
    result to the process-wide context's prec using its rounding, so a prec set by any library
    anywhere in the process would move a published figure -- 651,623.80 silently becomes
    651,620.00 under prec = 5. copy_abs() only drops the sign and consults nothing.
    """
    number = Decimal(value)
    # Before the sign test at the end, because Decimal('NaN') < 0 *signals* InvalidOperation rather
    # than answering False the way the float comparison would, and a raw decimal exception out of a
    # select is the one thing this module promises cannot happen. Grouping is unsafe here too:
    # Decimal(inf) renders 'Infinity', which _grouped() would put in a report cell as 'In,fin,ity'.
    if not number.is_finite():
        return None
    # format() rounds a Decimal with the current thread's context, which any code in the process
    # may have changed. The rounding is stated here instead, so a figure cannot move with someone
    # else's setting. format_number() hands over a value round() has already rounded, so this
    # meets a half only on a value round() returned as it was, past its fifteenth digit.
    with localcontext(_ROUNDING):
        rendered = format(number.copy_abs(), f".{decimals}f")
    whole, _, fraction = rendered.partition(".")
    text = _grouped(whole, thousands)
    if fraction:
        text = f"{text}{point}{fraction}"
    below_zero = number < 0 if negative is None else negative
    return f"-{text}" if below_zero else text
