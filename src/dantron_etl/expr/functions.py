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
    REWRITE_REASON,
    SEPARATOR_REASON,
    Choice,
    Namespace,
    Param,
    Pattern,
    QuotedText,
    Value,
    WholeNumber,
    probe_pattern,
)
from .dates import count_between, date_or_null, days_in_month
from .errors import Refusal
from .families import Family
from .formatting import written_as_text


@dataclass(frozen=True)
class Function:
    """Everything about one function, in one place.

    ``params`` lists the arguments in order. Their number is fixed, with no defaulted arguments:
    one spelling per call keeps a configuration file reviewable, and a reviewer should not have to
    know a default to know what a line does. A ``variadic`` function has a single entry, which
    every argument it is given must meet, and it needs at least two.

    ``result`` is the family the function hands back, so that nesting is checked rather than
    assumed. Without it, ``upper(len([label]))`` compiles and fails in the engine. ``"common"``
    means the family its arguments agree on, which is what coalesce() gives.

    ``result_dtype`` is what the result holds exactly, beside the family, where that is known.
    year() and len() always give a whole number, so a count of months may be built out of either,
    and format_number() and uuid5() always give a plain string. ``"subject"`` means whatever the
    first argument holds, which is what abs() and round() give. ``"common"`` means worked out from
    all the arguments, which is what coalesce() gives.

    ``to_polars`` builds the column expression. It is given a column expression for each value
    argument and the Python value for each argument written in the file, in order.

    ``summary`` says in one line what the function gives. ``example`` is one call, as an author
    would write it.
    """

    name: str
    params: tuple[Param, ...]
    result: Family | Literal["common"]
    to_polars: Callable[..., pl.Expr]
    summary: str
    example: str
    variadic: bool = False
    result_dtype: pl.DataType | Literal["subject", "common"] | None = None


def _coalesce(*parts: pl.Expr) -> pl.Expr:
    return pl.coalesce(list(parts))


def _concat(*parts: pl.Expr) -> pl.Expr:
    """Null in, null out, like everything else here -- and the safer of the two behaviours.

    Where a joined value is a key for matching, dropping a missing part builds a *shorter* key
    that matches the wrong rows, whereas a null key matches nothing and shows up as a null in
    a count. An author who wants the other behaviour writes it: concat(coalesce([code], ''), '-x').
    """
    return pl.concat_str(list(parts), separator="", ignore_nulls=False)


def _round(subject: pl.Expr, digits: int) -> pl.Expr:
    return subject.round(digits)


def _abs(subject: pl.Expr) -> pl.Expr:
    return subject.abs()


def _year(subject: pl.Expr) -> pl.Expr:
    return subject.dt.year()


def _add_months(subject: pl.Expr, count: pl.Expr) -> pl.Expr:
    """Move a date by a count of months, landing on the last day of a month that is too short.

    The count is built into the engine's own offset text -- '3mo', '-1mo' -- rather than
    passed as a number, because the count can come out of a column and so is not known here.
    The engine clamps in both directions, which is the behaviour wanted: 31 January plus one
    month is 28 February, not 3 March.

    A null on either side gives a null. The offset text is null when the count is null, and an
    offset of null leaves a null date behind.
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


def _format_number(subject: pl.Expr, decimals: int, thousands: str, point: str) -> pl.Expr:
    """A number written as text: grouped in threes, to exactly ``decimals`` places.

    **Rounding follows the value the column holds, not the decimal somebody typed.** -15.215 at
    two places is -15.21, for three reasons worth stating together:

    * it is what the figure this library is measured against says, because that figure is
      formatted by the same rule;
    * it is the more honest reading -- an amount summed out of decimals *is* the decimal, and
      -15.215 as one is -15.21499999999999985789..., which is below the half;
    * and it is a disagreement with round() in this same language, which takes the other view
      and gives -15.22. The two are written side by side here, and in the manual, so that a
      reader meets the disagreement in the catalog rather than in a finished report.

    **This works a row at a time, which is a cost nothing here pays without an argument.**
    The alternatives were measured rather than assumed: round(), a cast to a 2-place decimal
    type and the obvious scale-then-round all answer -15.22, and a very wide decimal type lands
    on the right side of this tie by luck and drifts the other way at wider scales still. There
    is no route through the engine's own kernels that keeps the rule above. What makes the cost
    payable is where this belongs: it writes a figure for somebody to read, so it runs at the
    end of a flow on the rows that reach a report, not in the middle of one on a million rows.
    Put it last, after the filtering and the joining, and the cost is the rows a person sees.

    A null is left null, the way it is everywhere else here: a formatted 0.00 where nothing was
    measured cannot be told from a real zero by whoever reads the column.

    The width and both separators are read out of the file and never out of a row. How many
    places a figure is written to is a decision a reviewer reads in the line, the way round()'s
    digits and regex_extract()'s group are -- a width that varied row by row would not be a width
    at all.
    """
    return subject.map_elements(
        lambda value: written_as_text(value, decimals, thousands, point),
        return_dtype=pl.String,
    )


def _uuid5(namespace: uuid.UUID, subject: pl.Expr) -> pl.Expr:
    """A stable id derived from a namespace and a value: RFC 4122 version 5, verbatim.

    The whole worth of this function is that the derivation is not this library's, so it is the
    standard library's own that runs, rather than a SHA-1 and a bit-twiddle written out here
    that would have to be kept equal to it by review. A hand-rolled digest measured 2.4x
    quicker and was refused at that price: an id only this library can reproduce is not a key,
    and the contract here is equality with uuid.uuid5 rather than resemblance to it.

    **This works a row at a time, like format_number, and here the engine leaves no choice.**
    There is no SHA-1 in it -- its hash() is a fast non-cryptographic one, which is a different
    number -- so there is no kernel to vectorise against. A batch-at-a-time map measured 6%
    quicker and would have to skip the nulls by hand, which is not a trade worth making. The
    cost is about 5 microseconds a row, so a build over tens of thousands of rows spends a
    fraction of a second here. Derive the id once, where the key is built, rather than
    recomputing it in each step that joins on it.

    A null value gives a null id, left to the engine's own skipping rather than written here. A
    missing value hashed as some stand-in would put every row that is missing a part under one
    real-looking id, which reads downstream as a legitimate key with several rows behind it
    instead of as data that was never there. An author who wants the other behaviour writes
    uuid5('...', coalesce([code], 'missing')), where the choice is visible.

    The namespace is read first, and out of the file rather than out of a row, so that a
    malformed one is refused in the line the author wrote instead of inside a select that has
    already written ids for every row before it.
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
    # Characters in a piece of text, not rows in a frame: an expression here is one column's
    # worth of values, so a row count would have no meaning an author could use.
    return subject.str.len_chars()


def _starts_with(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    return subject.str.starts_with(needle)


def _ends_with(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    return subject.str.ends_with(needle)


def _contains(subject: pl.Expr, needle: pl.Expr) -> pl.Expr:
    # literal=True, deliberately. A substring test that quietly accepted patterns would make a
    # dot in an account code mean "any character", and the wrong rows would match without
    # anyone noticing. The regex functions are the only door to regular expressions.
    return subject.str.contains(needle, literal=True)


def _regex_match(subject: pl.Expr, pattern: str) -> pl.Expr:
    # Unanchored, like the extract beside it: does the pattern occur anywhere. contains()
    # cannot answer this, because contains() is literal by design, which leaves no way at all
    # to ask a pattern question and get a truth value back -- and a filter needs one.
    return subject.str.contains(pattern)


def _regex_extract(subject: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """The text one group matched. A group the pattern does not have is refused, not left null.

    Group 0 is the whole match. The pattern's groups are counted by the engine itself, with no row
    read, so a pattern written without parentheses and asked for group 1 is caught in the file.
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
    """A rewrite whose replacement is text, verbatim, whatever the engine reads into it.

    The engine underneath reads '$name' in a replacement as a capture group and deletes it
    when no such group exists, so a replacement of '$5.00' lands as '.00' -- a hidden
    mini-language inside a string literal, in a language for amounts. Doubling the sign is how
    that engine spells a literal one. The pattern is wrapped so the engine takes its expansion
    path for every pattern rather than only for some, which keeps the doubling meaning one
    thing: a single-character pattern is otherwise handled by a fast path that would leave the
    doubled sign in the result.
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

# The table. Adding a function to the language means adding one entry here, and the function that
# builds it above. It never means adding grammar: an author-facing operator has to be spelled,
# documented, reviewed in a file and read by a non-developer, whereas one more named function is
# free of all of that. The manual's list of functions (MANUAL.md section 4.2) is checked against
# this table by a test.
DEFINITIONS: Final[tuple[Function, ...]] = (
    Function(
        "coalesce",
        params=(Value(None, "needs every value it chooses between in the same type family"),),
        variadic=True,
        result="common",
        result_dtype="common",
        to_polars=_coalesce,
        summary="The first of its arguments that is not null.",
        example="coalesce([discount], 0.0)",
    ),
    Function(
        "round",
        params=(_ON_NUMBERS, WholeNumber()),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_round,
        summary="A number rounded to a given number of decimal places.",
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
        "add_months",
        params=(
            Value(Family.DATE, "moves a date"),
            Value(Family.NUMBER, "moves a date by a count of months", whole="moves a date by whole months"),
        ),
        result=Family.DATE,
        to_polars=_add_months,
        summary="A date moved by a whole number of months, kept inside a month that is too short.",
        example="add_months([starts_on], [months] - 1)",
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
        summary="Whole months from the first date to the second, signed; 'inclusive' or 'elapsed', as date_diff counts.",
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
        summary="Days from the first date to the second, signed; 'inclusive' or 'elapsed', as date_diff counts.",
        example="days_between([ordered_on], [paid_on], 'elapsed')",
    ),
    Function(
        "make_date",
        params=(_WHOLE_PART, _WHOLE_PART, _WHOLE_PART),
        result=Family.DATE,
        result_dtype=pl.Date(),
        to_polars=_make_date,
        summary="The date a year, a month and a day name, or null when they name no date (30 February).",
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
        "abs",
        params=(_ON_NUMBERS,),
        result=Family.NUMBER,
        result_dtype="subject",
        to_polars=_abs,
        summary="A number without its sign.",
        example="abs([amount])",
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
        summary="Text without the whitespace at its start and end.",
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
        "starts_with",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_starts_with,
        summary="Whether text starts with other text, matched exactly.",
        example="starts_with([code], 'AB-')",
    ),
    Function(
        "ends_with",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_ends_with,
        summary="Whether text ends with other text, matched exactly.",
        example="ends_with([file_name], '.csv')",
    ),
    Function(
        "contains",
        params=(_ON_TEXT, _AGAINST_TEXT),
        result=Family.BOOLEAN,
        to_polars=_contains,
        summary="Whether text contains other text, matched exactly rather than as a pattern.",
        example="contains([note], 'refund')",
    ),
    Function(
        "regex_extract",
        params=(_ON_TEXT, Pattern(), WholeNumber()),
        result=Family.TEXT,
        to_polars=_regex_extract,
        summary="The part of text that one group of a regular expression matched, or null.",
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
        summary="Whether a regular expression matches anywhere in text.",
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
        summary="A number written as text, to a fixed number of places, with the separators stated.",
        example="format_number([total], 2, ',', '.')",
    ),
    Function(
        "uuid5",
        params=(Namespace(), Value(Family.TEXT, "derives an id from text")),
        result=Family.TEXT,
        result_dtype=pl.String(),
        to_polars=_uuid5,
        summary="A stable id (RFC 4122 version 5) derived from a namespace and a piece of text.",
        example="uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', concat([code], '|', [period]))",
    ),
)

FUNCTIONS: Final[dict[str, Function]] = {function.name: function for function in DEFINITIONS}

ALLOWED_FUNCTIONS: Final[frozenset[str]] = frozenset(FUNCTIONS)
