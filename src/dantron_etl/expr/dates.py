"""Counting between dates and building dates, written once for the tools and the functions.

``date_diff`` and ``months_between`` / ``days_between`` count with :func:`count_between`.
``assemble_date``, ``make_date`` and ``month_end`` build a date with :func:`date_or_null` and
:func:`days_in_month`. Both the tool and the function call the same code, so they cannot drift.
Each takes column expressions and gives one back; nothing here reads a row.
"""

from typing import Final, Literal

import polars as pl

__all__ = ["MAX_YEAR", "MIN_YEAR", "count_between", "date_or_null", "days_in_month"]

#: The years a calendar date can have.
MIN_YEAR: Final[int] = 1
MAX_YEAR: Final[int] = 9999

_LONG_MONTHS: Final[list[int]] = [1, 3, 5, 7, 8, 10, 12]
_SHORT_MONTHS: Final[list[int]] = [4, 6, 9, 11]


def count_between(
    start: pl.Expr,
    end: pl.Expr,
    unit: Literal["month", "day"],
    counting: Literal["inclusive", "elapsed"],
) -> pl.Expr:
    """The whole months or days from ``start`` to ``end``, two date columns, as a signed Int64.

    A month counts once the end's day of the month has reached the start's: 31 January to 28
    February is one month, and 1 January to 31 December is twelve. ``inclusive`` counts the unit
    each end falls in, so the same day at both ends is one; ``elapsed`` counts completed units only,
    one less in size. A backwards range counts negative and keeps its size. A null on either side
    gives a null, never a zero.
    """
    # Measured from the earlier date to the later one; the sign is decided separately, afterwards.
    earlier = pl.min_horizontal(start, end)
    later = pl.max_horizontal(start, end)

    if unit == "month":
        magnitude = (
            (later.dt.year() - earlier.dt.year()) * 12
            + (later.dt.month() - earlier.dt.month())
            + 1
            - pl.when(later.dt.day() < earlier.dt.day()).then(1).otherwise(0)
        )
    else:
        magnitude = (later - earlier).dt.total_days() + 1

    if counting == "elapsed":
        magnitude = magnitude - 1

    # The null case first: a comparison against a null date is itself null, and would otherwise fall
    # to the backwards arm and be counted as though it were a real range.
    missing = start.is_null() | end.is_null()
    forward = end >= start
    return pl.when(missing).then(None).when(forward).then(magnitude).otherwise(-magnitude).cast(pl.Int64)


def days_in_month(year: pl.Expr, month: pl.Expr) -> pl.Expr:
    """How many days ``month`` has in ``year``: 28 to 31, or null for a month outside 1 to 12."""
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    return (
        pl.when(month.is_in(_LONG_MONTHS))
        .then(31)
        .when(month.is_in(_SHORT_MONTHS))
        .then(30)
        .when(month == 2)
        .then(pl.when(leap).then(29).otherwise(28))
        .otherwise(None)
    )


def date_or_null(year: pl.Expr, month: pl.Expr, day: pl.Expr) -> pl.Expr:
    """The date the three whole-number parts name, or null when they name no date.

    Null when any part is null, the year is outside 1 to 9999, the month outside 1 to 12, or the day
    outside the month (30 February, 31 April). Never an error and never a nearby date.
    """
    valid = (
        year.is_between(MIN_YEAR, MAX_YEAR)
        & month.is_between(1, 12)
        & (day >= 1)
        & (day <= days_in_month(year, month))
    ).fill_null(False)
    # The engine raises on an impossible date even in a row a 'when' discards, so those rows are
    # given a harmless date to build and then dropped.
    safe_year = pl.when(valid).then(year).otherwise(pl.lit(2000))
    safe_month = pl.when(valid).then(month).otherwise(pl.lit(1))
    safe_day = pl.when(valid).then(day).otherwise(pl.lit(1))
    return pl.when(valid).then(pl.date(safe_year, safe_month, safe_day)).otherwise(None)
