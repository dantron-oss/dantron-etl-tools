"""Every function the language has, each defined once.

One entry per function says which arguments it takes, what it gives back, how its column is built,
and what it is for. The whitelist, the arity check and the result types are all read from this
table. To add a function, add one entry and the function that builds its column.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import polars as pl

from .arguments import (
    LARGEST_COUNT,
    REWRITE_REASON,
    SEPARATOR_REASON,
    Characters,
    Choice,
    DateFormat,
    Namespace,
    Param,
    Pattern,
    QuotedText,
    Value,
    WholeNumber,
    probe_pattern,
)
from .dates import MAX_YEAR, MIN_YEAR, count_between, date_or_null, days_in_month
from .errors import Refusal
from .families import Family
from .formatting import written_as_text


@dataclass(frozen=True)
class NoAnswer:
    """The rows a function has no answer for. The step stops on them instead of writing a null.

    `when` is given what the builder is given and is true on such a row. `problem` says what is wrong:
    {0}, {1} ... stand for the arguments as the author wrote them, and {found} for the values found
    in argument number `shows`. `remedy` says what to do.
    """

    when: Callable[..., pl.Expr]
    problem: str
    shows: int
    remedy: str


@dataclass(frozen=True)
class Function:
    """Everything about one function: its arguments, its result, its builder, its summary and example.

    `params` lists the arguments in order. The last `optional` of them may be left out, and the builder
    is then given fewer. A `variadic` function has one entry that every argument must meet, and needs at
    least two. `result` is the family returned ("common": the
    family the arguments agree on). `result_dtype` is the exact type where known ("subject": the first
    argument's; "common": worked out from all). `to_polars` builds the column from a polars expression
    per value argument and the Python value per literal argument. `typed` passes the first argument's
    dtype as `dtype=` (None without a schema), for round(), format_number(), floor(), ceil() and
    sign(). `no_answer` names the rows the function has no answer for; a tool stops the step on them.
    """

    name: str
    params: tuple[Param, ...]
    result: Family | Literal["common"]
    to_polars: Callable[..., pl.Expr]
    summary: str
    example: str
    variadic: bool = False
    optional: int = 0
    result_dtype: pl.DataType | Literal["subject", "common"] | None = None
    typed: bool = False
    no_answer: NoAnswer | None = None
    #: How the value arguments meet each other, beyond sharing a family. "compared": tested for
    #: equality, so a timestamp against a date is refused as '=' refuses it. "ordered": put in order,
    #: so timestamps on different clocks are refused. "arithmetic": worked out as the operators work,
    #: so an unsigned whole number is widened to Int64 and an exact Decimal is refused.
    meets: Literal["compared", "ordered", "arithmetic"] | None = None
    #: The argument that divides: a zero there stops the step, as a '/' by zero does.
    divides_by: int | None = None


def _coalesce(*parts: pl.Expr) -> pl.Expr:
    return pl.coalesce(list(parts))


def _concat(*parts: pl.Expr) -> pl.Expr:
    """Join the pieces; null if any piece is null. Why: docs/design/expressions.md#the-function-table"""
    return pl.concat_str(list(parts), separator="", ignore_nulls=False)


# Significant digits a decimal is read to before round() decides whether it is a half.
_READ_DIGITS: Final[dict[pl.DataType, int]] = {pl.Float64(): 15, pl.Float32(): 6}


def _round(subject: pl.Expr, digits: int, *, dtype: pl.DataType | None = None) -> pl.Expr:
    """A number rounded to `digits` places, halves away from zero, like a spreadsheet's ROUND.

    A Float64 is read to 15 significant digits (Float32: 6), so 1.005 rounds to 1.01 as written, though
    its binary value is just below the half. Whole numbers come back as they are; an exact Decimal is
    rounded on its own digits; null, NaN and infinity pass through. Without a schema, the choice is made
    per batch. Why: docs/design/expressions.md#rounding-like-a-spreadsheet-15-significant-digits
    """
    if dtype is None:
        return subject.map_batches(lambda series: _rounded_series(series, digits), is_elementwise=True)
    return _rounded(subject, digits, dtype)


def _rounded(subject: pl.Expr, digits: int, dtype: pl.DataType) -> pl.Expr:
    if isinstance(dtype, pl.Decimal):
        return subject.round(digits, mode="half_away_from_zero")
    read = _READ_DIGITS.get(dtype)
    if read is None:
        if dtype.is_integer() or dtype == pl.Null:
            return subject  # a whole number, or a null: nothing to round
        return subject.round(digits)  # not a number: the engine refuses it, as it always did
    value = subject.cast(pl.Float64)
    magnitude = value.abs().log10().floor()
    half_a_unit = pl.lit(0.5) * pl.lit(10.0).pow(magnitude - (read - 1))
    nudged = (value + value.sign() * half_a_unit).round(digits, mode="half_away_from_zero")
    within_reading = (magnitude + digits) <= (read - 2)
    return pl.when(within_reading).then(nudged).otherwise(value).cast(dtype)


def _rounded_series(series: pl.Series, digits: int) -> pl.Series:
    """round() on a column whose type was not known when the expression was built."""
    return series.to_frame("value").select(_rounded(pl.col("value"), digits, series.dtype)).to_series().alias(series.name)


def _abs(subject: pl.Expr) -> pl.Expr:
    return subject.abs()


def _without_negative_zero(value: pl.Expr) -> pl.Expr:
    """-0.0 as 0.0, every other value as it is: a minus sign on a zero reads as a debit in a written file."""
    return pl.when(value == 0).then(value.abs()).otherwise(value)


def _floor(subject: pl.Expr, *, dtype: pl.DataType | None = None) -> pl.Expr:
    # A whole number comes back as it is; a Decimal keeps its places; a null of no type stays null.
    return subject if dtype == pl.Null else _without_negative_zero(subject.floor())


def _ceil(subject: pl.Expr, *, dtype: pl.DataType | None = None) -> pl.Expr:
    return subject if dtype == pl.Null else _without_negative_zero(subject.ceil())


def _sign(subject: pl.Expr, *, dtype: pl.DataType | None = None) -> pl.Expr:
    return subject if dtype == pl.Null else _without_negative_zero(subject.sign())


def _mod(dividend: pl.Expr, divisor: pl.Expr) -> pl.Expr:
    """The remainder with the sign of the dividend, as SQL's MOD: mod(-7, 3) is -1.

    The engine's own % takes the divisor's sign (-7 % 3 is 2), so the remainder is worked out on the
    two sizes and given the dividend's sign, which is exact for numbers with a fraction too.
    """
    return _without_negative_zero(dividend.sign() * (dividend.abs() % divisor.abs()))


def _least(*parts: pl.Expr) -> pl.Expr:
    # Nulls are skipped, as PostgreSQL's LEAST skips them; only all nulls give null.
    return pl.min_horizontal(list(parts))


def _greatest(*parts: pl.Expr) -> pl.Expr:
    return pl.max_horizontal(list(parts))


def _nullif(subject: pl.Expr, other: pl.Expr) -> pl.Expr:
    return pl.when(subject == other).then(None).otherwise(subject)


def _year(subject: pl.Expr) -> pl.Expr:
    return subject.dt.year()


# month(), day() and the rest give Int32, as year() does, so month([d]) * 100 cannot wrap round.
def _month(subject: pl.Expr) -> pl.Expr:
    return subject.dt.month().cast(pl.Int32)


def _day(subject: pl.Expr) -> pl.Expr:
    return subject.dt.day().cast(pl.Int32)


def _quarter(subject: pl.Expr) -> pl.Expr:
    return subject.dt.quarter().cast(pl.Int32)


def _day_of_week(subject: pl.Expr) -> pl.Expr:
    # The engine counts as ISO does: Monday is 1, Sunday is 7.
    return subject.dt.weekday().cast(pl.Int32)


# The engine counts a move of days in nanoseconds, and a move longer than this wraps round to a wrong date.
_MOST_DAYS: Final[int] = 106_751


def _add_days(subject: pl.Expr, count: pl.Expr) -> pl.Expr:
    """Move a date by a count of days, built as add_months() builds its offset ('3d').

    A day is a calendar day, so a timestamp keeps its time of day when the clocks change.
    """
    offset = pl.concat_str([count.cast(pl.String), pl.lit("d")], ignore_nulls=False)
    return subject.dt.offset_by(offset)


def _too_many_days(subject: pl.Expr, count: pl.Expr) -> pl.Expr:
    return ~count.is_between(-_MOST_DAYS, _MOST_DAYS)


def _month_start(subject: pl.Expr) -> pl.Expr:
    return subject.cast(pl.Date).dt.month_start()


def _parse_date(text: pl.Expr, format_: str) -> pl.Expr:
    """Text read as a date in the format; null where it does not match or names no date in the years 1 to 9999."""
    parsed = text.str.strptime(pl.Date, format_, strict=False, exact=True)
    return pl.when(parsed.dt.year().is_between(MIN_YEAR, MAX_YEAR)).then(parsed)


def _unreadable(text: pl.Expr, format_: str) -> pl.Expr:
    return text.is_not_null() & _parse_date(text, format_).is_null()


def _add_months(subject: pl.Expr, count: pl.Expr) -> pl.Expr:
    """Move a date by a count of months, clamped to the end of a short month (31 Jan + 1 = 28 Feb).

    The count may come from a column, so it is built into polars' offset text ('3mo'). A null on either
    side gives null.
    """
    offset = pl.concat_str([count.cast(pl.String), pl.lit("mo")], ignore_nulls=False)
    return subject.dt.offset_by(offset)


def _months_between(start: pl.Expr, end: pl.Expr, counting: str) -> pl.Expr:
    """Whole months from start to end: the date_diff tool's count, from the same shared function."""
    return count_between(start.cast(pl.Date), end.cast(pl.Date), "month", counting)  # type: ignore[arg-type]


def _days_between(start: pl.Expr, end: pl.Expr, counting: str) -> pl.Expr:
    """Days from start to end: the date_diff tool's count, from the same shared function."""
    return count_between(start.cast(pl.Date), end.cast(pl.Date), "day", counting)  # type: ignore[arg-type]


def _make_date(year: pl.Expr, month: pl.Expr, day: pl.Expr) -> pl.Expr:
    """The date three whole numbers name, or null: the assemble_date tool's build, shared."""
    return date_or_null(year.cast(pl.Int64), month.cast(pl.Int64), day.cast(pl.Int64))


def _month_end(subject: pl.Expr) -> pl.Expr:
    """The last day of the month a date falls in, built the way assemble_date's clamp builds it."""
    day = subject.cast(pl.Date)
    year = day.dt.year().cast(pl.Int64)
    month = day.dt.month().cast(pl.Int64)
    return date_or_null(year, month, days_in_month(year, month))


def _format_number(
    subject: pl.Expr, decimals: int, thousands: str, point: str, *, dtype: pl.DataType | None = None
) -> pl.Expr:
    """A number written as text, digits grouped in threes, to exactly `decimals` places.

    The value is rounded by round()'s rule first, so the two never disagree; a negative value that rounds
    to zero keeps its minus sign (-0.004 reads -0.00). A null stays null. Written row by row, since polars
    has no kernel for grouping digits. The width and separators come from the file, never from a row.
    """
    # Digits from round()'s answer; the sign from the value itself, so -0.004 reads -0.00.
    return pl.struct(value=subject, rounded=_round(subject, decimals, dtype=dtype)).map_elements(
        lambda row: None
        if row["rounded"] is None
        else written_as_text(row["rounded"], decimals, thousands, point, negative=_below_zero(row["value"])),
        return_dtype=pl.String,
    )


def _below_zero(value: object) -> bool:
    """Whether a value is a number below zero; false for null and NaN."""
    try:
        return bool(value is not None and value < 0)  # type: ignore[operator]
    except TypeError:
        return False


def _uuid5(namespace: uuid.UUID, subject: pl.Expr) -> pl.Expr:
    """A stable id from a namespace and a value: Python's own uuid.uuid5, row by row.

    polars has no SHA-1, so there is no column kernel; about 5 microseconds a row. A null value gives a
    null id. Why: docs/design/expressions.md#the-function-table
    """
    return subject.map_elements(
        lambda value: str(uuid.uuid5(namespace, value)),
        return_dtype=pl.String,
    )


def _lower(subject: pl.Expr) -> pl.Expr:
    return subject.str.to_lowercase()


def _upper(subject: pl.Expr) -> pl.Expr:
    return subject.str.to_uppercase()


def _trim(subject: pl.Expr) -> pl.Expr:
    return subject.str.strip_chars()


def _len(subject: pl.Expr) -> pl.Expr:
    # Characters in the text, not rows.
    return subject.str.len_chars()


def _starts_with(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    return subject.str.starts_with(needle)


def _ends_with(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    return subject.str.ends_with(needle)


def _contains(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    # literal=True: a dot in a code must not mean "any character". Patterns go through regex_match().
    return subject.str.contains(needle, literal=True)


def _left(subject: pl.Expr, count: int) -> pl.Expr:
    # A negative count is PostgreSQL's: all but the last that many. The engine counts the same way.
    return subject.str.head(count)


def _right(subject: pl.Expr, count: int) -> pl.Expr:
    return subject.str.tail(count)


def _substr(subject: pl.Expr, start: int, length: int | None = None) -> pl.Expr:
    # `start` counts from 1, as SQL does; the engine counts from 0. No length runs to the end.
    return subject.str.slice(start - 1, length)


def _split_part(subject: pl.Expr, delimiter: str, part: int) -> pl.Expr:
    """The `part`th piece of text split at `delimiter`, from 1; negative counts from the end.

    A part past the last is empty text, as PostgreSQL gives it; a null is null.
    """
    pieces = subject.str.split(delimiter).list.get(part - 1 if part > 0 else part, null_on_oob=True)
    return pl.when(subject.is_not_null()).then(pieces.fill_null(""))


def _pad_left(subject: pl.Expr, length: int, fill: str) -> pl.Expr:
    # Never cuts: text already longer than `length` is left as it is.
    return subject.str.pad_start(length, fill)


def _pad_right(subject: pl.Expr, length: int, fill: str) -> pl.Expr:
    return subject.str.pad_end(length, fill)


def _regex_match(subject: pl.Expr, pattern: str) -> pl.Expr:
    # Unanchored: does the pattern occur anywhere.
    return subject.str.contains(pattern)


def _regex_extract(subject: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """The text one group matched. A group the pattern does not have is refused, not left null.

    Group 0 is the whole match. polars counts the groups with no row read.
    """
    groups = len(pl.select(pl.lit("", dtype=pl.String).str.extract_groups(pattern)).to_series().struct.fields)
    if group > groups:
        held = f"has {groups} group{'s' if groups != 1 else ''}" if groups else "has no groups (no parentheses)"
        raise Refusal(
            f"regex_extract() was asked for group {group}, but the pattern '{pattern}' {held}, so every row "
            f"would be null; put parentheses round the part to keep, or ask for group 0 for the whole match"
        )
    return subject.str.extract(pattern, group)


def _regex_replace(subject: pl.Expr, pattern: str, replacement: str) -> pl.Expr:
    """Replace every match with the replacement text exactly as written.

    polars reads '$name' in a replacement as a group, so '$' is doubled; the pattern is wrapped in (?:...)
    so polars always takes the path where doubling means one thing.
    """
    engine_pattern = f"(?:{pattern})"
    engine_replacement = replacement.replace("$", "$$")
    probe_pattern("regex_replace", pattern, lambda probe: probe.str.replace_all(engine_pattern, engine_replacement))
    return subject.str.replace_all(engine_pattern, engine_replacement)


_ON_TEXT: Final[Value] = Value(Family.TEXT, "works on text")

_COUNTING: Final[Choice] = Choice(("inclusive", "elapsed"))

_WHOLE_PART: Final[Value] = Value(Family.NUMBER, "builds a date from numbers", whole="builds a date from whole numbers")

_ON_NUMBERS: Final[Value] = Value(Family.NUMBER, "works on numbers")

_AGAINST_TEXT: Final[Value] = Value(Family.TEXT, "compares text against text")

# The table. To add a function, add one entry here and its builder above; never add grammar.
# A test checks the manual's list of functions (section 4.2) against this table.
DEFINITIONS: Final[tuple[Function, ...]] = (
    Function(
        "coalesce",
        params=(Value(None, "needs every value it chooses between in the same type family"),),
        variadic=True,
        result="common",
        result_dtype="common",
        to_polars=_coalesce,
        summary="The first value that is not null: coalesce([discount], 0.0) gives 0.0 where discount is blank.",
        example="coalesce([discount], 0.0)",
    ),
    Function(
        "round",
        params=(_ON_NUMBERS, WholeNumber()),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_round,
        typed=True,
        summary="A number rounded to some decimal places, halves away from zero like a spreadsheet: 2.5 gives 3.",
        example="round([amount], 2)",
    ),
    Function(
        "year",
        params=(Value(Family.DATE, "reads the year of a date"),),
        result=Family.NUMBER,
        result_dtype=pl.Int32(),
        to_polars=_year,
        summary="The year of a date, as a whole number.",
        example="year([starts_on])",
    ),
    Function(
        "month",
        params=(Value(Family.DATE, "reads the month of a date"),),
        result=Family.NUMBER,
        result_dtype=pl.Int32(),
        to_polars=_month,
        summary="The month of a date, 1 to 12.",
        example="month([starts_on])",
    ),
    Function(
        "day",
        params=(Value(Family.DATE, "reads the day of a date"),),
        result=Family.NUMBER,
        result_dtype=pl.Int32(),
        to_polars=_day,
        summary="The day of the month of a date, 1 to 31.",
        example="day([starts_on])",
    ),
    Function(
        "quarter",
        params=(Value(Family.DATE, "reads the quarter of a date"),),
        result=Family.NUMBER,
        result_dtype=pl.Int32(),
        to_polars=_quarter,
        summary="The quarter of the year a date falls in, 1 to 4; January to March is 1.",
        example="quarter([starts_on])",
    ),
    Function(
        "day_of_week",
        params=(Value(Family.DATE, "reads the day of the week of a date"),),
        result=Family.NUMBER,
        result_dtype=pl.Int32(),
        to_polars=_day_of_week,
        summary="The day of the week of a date, counted the ISO way: Monday is 1 and Sunday is 7.",
        example="day_of_week([ordered_on])",
    ),
    Function(
        "add_months",
        params=(
            Value(Family.DATE, "moves a date"),
            Value(Family.NUMBER, "moves a date by a count of months", whole="moves a date by whole months"),
        ),
        result=Family.DATE,
        to_polars=_add_months,
        summary="A date moved by whole months; 31 January plus one month is 28 February.",
        example="add_months([starts_on], [months] - 1)",
    ),
    Function(
        "add_days",
        params=(
            Value(Family.DATE, "moves a date"),
            Value(Family.NUMBER, "moves a date by a count of days", whole="moves a date by whole days"),
        ),
        result=Family.DATE,
        result_dtype="subject",
        to_polars=_add_days,
        summary="A date moved by whole days; a negative count goes back.",
        example="add_days([ordered_on], 30)",
        no_answer=NoAnswer(
            when=_too_many_days,
            problem=(
                f"moves a date with add_days() by more than {_MOST_DAYS:,} days (about 292 years), further than "
                "the engine can count exactly: {1} holds {found}"
            ),
            shows=1,
            remedy="The step stops rather than write a wrong date. Check that {1} is the count of days you meant",
        ),
    ),
    Function(
        "months_between",
        params=(
            Value(Family.DATE, "counts between dates", no_time_zone=True),
            Value(Family.DATE, "counts between dates", no_time_zone=True),
            _COUNTING,
        ),
        result=Family.NUMBER,
        result_dtype=pl.Int64(),
        to_polars=_months_between,
        summary="Whole months from the first date to the second, counted 'inclusive' or 'elapsed' as date_diff counts.",
        example="months_between([starts_on], [ends_on], 'inclusive')",
    ),
    Function(
        "days_between",
        params=(
            Value(Family.DATE, "counts between dates", no_time_zone=True),
            Value(Family.DATE, "counts between dates", no_time_zone=True),
            _COUNTING,
        ),
        result=Family.NUMBER,
        result_dtype=pl.Int64(),
        to_polars=_days_between,
        summary="Days from the first date to the second, counted 'inclusive' or 'elapsed' as date_diff counts.",
        example="days_between([ordered_on], [paid_on], 'elapsed')",
    ),
    Function(
        "make_date",
        params=(_WHOLE_PART, _WHOLE_PART, _WHOLE_PART),
        result=Family.DATE,
        result_dtype=pl.Date(),
        to_polars=_make_date,
        summary="A date built from a year, a month and a day; null when they make no date, such as 30 February.",
        example="make_date([year], [month], 1)",
    ),
    Function(
        "month_end",
        params=(Value(Family.DATE, "finds the end of a date's month", no_time_zone=True),),
        result=Family.DATE,
        result_dtype=pl.Date(),
        to_polars=_month_end,
        summary="The last day of the month a date falls in.",
        example="month_end([ordered_on])",
    ),
    Function(
        "month_start",
        params=(Value(Family.DATE, "finds the start of a date's month", no_time_zone=True),),
        result=Family.DATE,
        result_dtype=pl.Date(),
        to_polars=_month_start,
        summary="The first day of the month a date falls in.",
        example="month_start([ordered_on])",
    ),
    Function(
        "parse_date",
        params=(Value(Family.TEXT, "reads text as a date"), DateFormat()),
        result=Family.DATE,
        result_dtype=pl.Date(),
        to_polars=_parse_date,
        summary="Text read as a date in a stated format; text that does not match stops the step.",
        example="parse_date([shipped], '%d/%m/%Y')",
        no_answer=NoAnswer(
            when=_unreadable,
            problem="reads text as a date with parse_date(), and {0} holds text that is not a date in the format {1}: {found}",
            shows=0,
            remedy=(
                "The step stops rather than leave those dates blank. Correct the text, or give those rows a "
                "case_when rung of their own before the rung that reads them"
            ),
        ),
    ),
    Function(
        "abs",
        params=(_ON_NUMBERS,),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_abs,
        summary="A number without its sign.",
        example="abs([amount])",
    ),
    Function(
        "floor",
        params=(_ON_NUMBERS,),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_floor,
        typed=True,
        summary="A number taken down to the whole number at or below it, in the number's own type: -2.5 gives -3.",
        example="floor([hours])",
    ),
    Function(
        "ceil",
        params=(_ON_NUMBERS,),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_ceil,
        typed=True,
        summary="A number taken up to the whole number at or above it, in the number's own type: 2.1 gives 3.",
        example="ceil([hours])",
    ),
    Function(
        "sign",
        params=(_ON_NUMBERS,),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_sign,
        typed=True,
        summary="-1 for a number below zero, 0 for zero, 1 above zero.",
        example="sign([balance])",
    ),
    Function(
        "mod",
        params=(_ON_NUMBERS, _ON_NUMBERS),
        result=Family.NUMBER,
        result_dtype="common",
        to_polars=_mod,
        meets="arithmetic",
        divides_by=1,
        summary="The remainder of the first number divided by the second, with the first one's sign: mod(-7, 3) is -1.",
        example="mod([month_number], 3)",
    ),
    Function(
        "least",
        params=(Value(None, "needs every value it compares in the same type family"),),
        variadic=True,
        result="common",
        result_dtype="common",
        to_polars=_least,
        meets="ordered",
        summary="The smallest of the values; nulls are skipped, and only all nulls give null.",
        example="least([due_on], [paid_on])",
    ),
    Function(
        "greatest",
        params=(Value(None, "needs every value it compares in the same type family"),),
        variadic=True,
        result="common",
        result_dtype="common",
        to_polars=_greatest,
        meets="ordered",
        summary="The largest of the values; nulls are skipped, and only all nulls give null.",
        example="greatest([amount], 0)",
    ),
    Function(
        "nullif",
        params=(
            Value(None, "needs both values in the same type family"),
            Value(None, "needs both values in the same type family"),
        ),
        result="common",
        result_dtype="subject",
        to_polars=_nullif,
        meets="compared",
        summary="Null where the first value equals the second, otherwise the first: nullif([code], '') blanks empty text.",
        example="nullif([discount], 0)",
    ),
    Function(
        "lower",
        params=(_ON_TEXT,),
        result=Family.TEXT,
        to_polars=_lower,
        summary="Text in lower case.",
        example="lower([name])",
    ),
    Function(
        "upper",
        params=(_ON_TEXT,),
        result=Family.TEXT,
        to_polars=_upper,
        summary="Text in upper case.",
        example="upper([code])",
    ),
    Function(
        "trim",
        params=(_ON_TEXT,),
        result=Family.TEXT,
        to_polars=_trim,
        summary="Text without the spaces at its start and end.",
        example="trim([name])",
    ),
    Function(
        "len",
        params=(_ON_TEXT,),
        result=Family.NUMBER,
        result_dtype=pl.UInt32(),
        to_polars=_len,
        summary="The number of characters in a piece of text.",
        example="len([code])",
    ),
    Function(
        "left",
        params=(_ON_TEXT, WholeNumber(least=-LARGEST_COUNT)),
        result=Family.TEXT,
        to_polars=_left,
        summary="The first characters of text; a negative count gives all but the last that many, as in SQL.",
        example="left([code], 3)",
    ),
    Function(
        "right",
        params=(_ON_TEXT, WholeNumber(least=-LARGEST_COUNT)),
        result=Family.TEXT,
        to_polars=_right,
        summary="The last characters of text; a negative count gives all but the first that many, as in SQL.",
        example="right([account], 4)",
    ),
    Function(
        "substr",
        params=(_ON_TEXT, WholeNumber(least=1), WholeNumber()),
        optional=1,
        result=Family.TEXT,
        to_polars=_substr,
        summary="Part of text: from a start counted from 1, for a length of characters, or to the end if none.",
        example="substr([reference], 4, 2)",
    ),
    Function(
        "split_part",
        params=(_ON_TEXT, Characters(), WholeNumber(least=-LARGEST_COUNT, nonzero=True)),
        result=Family.TEXT,
        to_polars=_split_part,
        summary="One part of text split at a delimiter, counted from 1; past the last part gives empty text.",
        example="split_part([reference], '-', 2)",
    ),
    Function(
        "pad_left",
        params=(_ON_TEXT, WholeNumber(), Characters(exactly_one=True)),
        result=Family.TEXT,
        to_polars=_pad_left,
        summary="Text padded at the start to a length with one character; longer text is never cut.",
        example="pad_left([account], 6, '0')",
    ),
    Function(
        "pad_right",
        params=(_ON_TEXT, WholeNumber(), Characters(exactly_one=True)),
        result=Family.TEXT,
        to_polars=_pad_right,
        summary="Text padded at the end to a length with one character; longer text is never cut.",
        example="pad_right([name], 20, ' ')",
    ),
    Function(
        "starts_with",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_starts_with,
        summary="Whether text starts with other text, matched exactly, case included.",
        example="starts_with([code], 'AB-')",
    ),
    Function(
        "ends_with",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_ends_with,
        summary="Whether text ends with other text, matched exactly, case included.",
        example="ends_with([file_name], '.csv')",
    ),
    Function(
        "contains",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_contains,
        summary="Whether text contains other text, matched exactly; a dot is a dot, not a pattern.",
        example="contains([note], 'refund')",
    ),
    Function(
        "regex_extract",
        params=(_ON_TEXT, Pattern(), WholeNumber()),
        result=Family.TEXT,
        to_polars=_regex_extract,
        summary="The part of text matched by one bracketed group of a regular expression, or null if no match.",
        example="regex_extract([reference], '([0-9]+)', 1)",
    ),
    Function(
        "regex_replace",
        params=(_ON_TEXT, Pattern(), QuotedText(REWRITE_REASON)),
        result=Family.TEXT,
        to_polars=_regex_replace,
        summary="Text with every match of a regular expression replaced by fixed text.",
        example="regex_replace(lower([name]), '[^a-z0-9]+', ' ')",
    ),
    Function(
        "regex_match",
        params=(_ON_TEXT, Pattern()),
        result=Family.BOOLEAN,
        to_polars=_regex_match,
        summary="Whether a regular expression matches anywhere in the text.",
        example="regex_match([code], '^[A-Z]{3}$')",
    ),
    Function(
        "concat",
        params=(Value(Family.TEXT, "joins text"),),
        variadic=True,
        result=Family.TEXT,
        to_polars=_concat,
        summary="Pieces of text joined end to end; null if any piece is null.",
        example="concat('order ', [number])",
    ),
    Function(
        "format_number",
        params=(
            Value(Family.NUMBER, "writes a number as text"),
            WholeNumber(),
            QuotedText(SEPARATOR_REASON),
            QuotedText(SEPARATOR_REASON),
        ),
        result=Family.TEXT,
        result_dtype=pl.String(),
        to_polars=_format_number,
        typed=True,
        summary="A number written as text: decimal places, then the thousands and decimal separators.",
        example="format_number([total], 2, ',', '.')",
    ),
    Function(
        "uuid5",
        params=(Namespace(), Value(Family.TEXT, "derives an id from text")),
        result=Family.TEXT,
        result_dtype=pl.String(),
        to_polars=_uuid5,
        summary="A stable id (RFC 4122 version 5) from a namespace and some text; the same text, the same id.",
        example="uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', concat([code], '|', [period]))",
    ),
)

FUNCTIONS: Final[dict[str, Function]] = {function.name: function for function in DEFINITIONS}

ALLOWED_FUNCTIONS: Final[frozenset[str]] = frozenset(FUNCTIONS)
