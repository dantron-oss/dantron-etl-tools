"""Helpers a tool calls to check its settings against the frames it was given.

A tool's ``validate_inputs`` and ``execute`` see the settings and the frames together. These
helpers answer the questions most tools ask there: is this column on this port, does this
expression compile against these columns, what type does it give, may the step write a column of
this name. Each refuses with a :class:`ConfigError` (or :class:`DataError`) that names the field.
"""

from collections.abc import Sequence
from typing import Final

import polars as pl

from .config import ConfigField
from .errors import ConfigError, DataError
from .column_types import EXACT_DIGITS
from .expr import (
    Division,
    ExpressionError,
    RowCheck,
    RowReference,
    compile_exact,
    compile_expr,
    divisions,
    exact_places,
    row_checks,
    row_references,
)
from .expr.exact import TOO_LARGE_REMEDY

__all__ = [
    "REPLACE_FIELD",
    "check_output_column",
    "compile_for",
    "compile_predicate_for",
    "ZONE_FIELD",
    "as_calendar_date",
    "check_zone_name",
    "compile_exact_for",
    "compiled_dtype",
    "exact_places_for",
    "refuse_exact_too_large",
    "refuse_rows_without_an_answer",
    "rows_without_an_answer",
    "row_references_for",
    "require_stated_zone",
    "require_columns",
]


def compile_for(frame: pl.DataFrame, text: str, field: str) -> pl.Expr:
    """Compile the expression ``text`` against ``frame``'s columns and types, and return it.

    ``field`` is the setting the text came from. A misspelled column, a function given the wrong
    type, or bad syntax is refused as a :class:`ConfigError` that names the field and quotes the
    text.

    Why: docs/design/core.md#expression-errors-become-config-errors
    """
    try:
        return compile_expr(text, frame.schema)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def refuse_rows_without_an_answer(
    frame: pl.DataFrame,
    text: str,
    field: str,
    *,
    among: pl.Expr | None = None,
    shown: Sequence[str] | None = None,
) -> None:
    """Stop the step on a row of ``frame`` that ``text`` has no answer for.

    That is a ``/`` whose divisor is zero, and a function call with no answer on that row, such as
    text ``parse_date()`` cannot read. Raises :class:`DataError` naming the field, the expression,
    what was wrong and up to five rows. Zero means 0, 0.0 or -0.0. A null divisor gives null and is
    allowed; so is not-a-number.

    ``among`` limits the check to the rows the expression is used on (``case_when`` passes the
    rows that reach a case). Other tools pass nothing.

    ``shown`` names the columns the refusal's rows show, when ``frame`` carries columns of the
    tool's own beside the input's (multi_row_formula's ``Row-1:amount``). Left out, all of them.

    Why: docs/design/core.md#division-by-zero-stops-the-step
    """
    found, checks = _answer_checks(frame, text, field)
    if not (found or checks) or frame.height == 0:
        return
    used = pl.lit(True) if among is None else among.fill_null(False)
    for division in found:
        zero = (division.expr == 0).fill_null(False) & used
        if not frame.select(zero.any()).item():
            continue
        raise DataError(
            f"config field {field!r} divides by zero: the divisor `{division.divisor}` is zero, in "
            f"expression: `{text}`. A division by zero has no answer, so the step stops rather than "
            f"write infinity or not-a-number; give the rows whose divisor is zero a case_when rung "
            f"of their own, before the one that divides",
            offending=_showing(frame.filter(zero), shown),
            total=frame.height,
        )
    for check in checks:
        failing = check.failing.fill_null(False) & used
        if not frame.select(failing.any()).item():
            continue
        offending = frame.filter(failing)
        values = offending.select(check.shown.alias("value")).unique(maintain_order=True).head(_VALUES_SHOWN)
        problem, remedy = check.refusal(values["value"].to_list())
        raise DataError(
            f"config field {field!r} {problem}, in expression: `{text}`. {remedy}",
            offending=_showing(offending, shown),
            total=frame.height,
        )


def _showing(rows: pl.DataFrame, shown: Sequence[str] | None) -> pl.DataFrame:
    """``rows`` with only the ``shown`` columns, or all of them when ``shown`` is None."""
    return rows if shown is None else rows.select(shown)


def _answer_checks(frame: pl.DataFrame, text: str, field: str) -> tuple[tuple[Division, ...], tuple[RowCheck, ...]]:
    """The divisions and per-row checks of ``text`` against ``frame``'s columns. Refuses as :func:`compile_for`."""
    try:
        return divisions(text, frame.schema), row_checks(text, frame.schema)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def row_references_for(text: str, field: str) -> tuple[RowReference, ...]:
    """The row references ``[Row-N:column]`` and ``[Row+N:column]`` in ``text``, for multi_row_formula.

    A reach of 0 rows, or past ``ROW_REACH``, is refused as a :class:`ConfigError` naming ``field``.
    """
    try:
        return row_references(text)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def rows_without_an_answer(frame: pl.DataFrame, text: str, field: str) -> pl.Expr | None:
    """A true/false expression, true on each row of ``frame`` that ``text`` has no answer for.

    These are the rows :func:`refuse_rows_without_an_answer` stops on, or None when no row can be
    one. It is for a tool that works its rows out a few at a time: it finds the failing row with
    this, then stops on it with :func:`refuse_rows_without_an_answer`, so the refusal reads the same.
    """
    found, checks = _answer_checks(frame, text, field)
    parts = [(division.expr == 0).fill_null(False) for division in found]
    parts += [check.failing.fill_null(False) for check in checks]
    return pl.any_horizontal(parts) if parts else None


def exact_places_for(frame: pl.DataFrame, text: str, field: str) -> int | None:
    """The places ``text`` gives as an exact value against ``frame``, or None when it gives none.

    An exact decimal gives its places, a whole number 0, a typed number the places it was written
    with (``0.0`` has 1). A float, text, a date or a null gives None. Refuses as :func:`compile_for`.
    """
    try:
        return exact_places(text, frame.schema)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def compile_exact_for(frame: pl.DataFrame, text: str, field: str, places: int) -> tuple[pl.Expr, pl.Expr]:
    """Compile ``text`` as an exact decimal with ``places`` places, to sit beside other exact values.

    Returns the expression and a true/false expression that is true on a row whose value has too
    many whole digits for those places; pass it to :func:`refuse_exact_too_large`. A float is refused
    as a :class:`ConfigError` naming the field.
    """
    try:
        return compile_exact(text, frame.schema, places)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def refuse_exact_too_large(
    frame: pl.DataFrame, too_large: pl.Expr, field: str, what: str, *, among: pl.Expr | None = None
) -> None:
    """Stop the step on a row of ``frame`` where an exact answer needs more than 38 digits.

    ``too_large`` is true on such a row. ``what`` says what the step works out, in words that follow
    the field's name, such as "sums [amount] as an exact decimal with 2 places". Raises
    :class:`DataError` naming the field and up to five rows, and saying to round first. ``among``
    limits the check to the rows the answer is used on, as in :func:`refuse_rows_without_an_answer`.
    """
    if frame.height == 0:
        return
    used = pl.lit(True) if among is None else among.fill_null(False)
    failing = too_large.fill_null(False) & used
    if not frame.select(failing.any()).item():
        return
    raise DataError(
        f"config field {field!r} {what}, and on some rows the answer needs more than {EXACT_DIGITS} "
        f"digits. {TOO_LARGE_REMEDY}",
        offending=frame.filter(failing),
        total=frame.height,
    )


# How many different values a refusal quotes back, beside the rows it shows.
_VALUES_SHOWN: Final[int] = 5


def compiled_dtype(frame: pl.DataFrame, text: str, field: str) -> tuple[str, pl.DataType]:
    """Return ``(column name, dtype)`` of what ``text`` produces against ``frame``'s schema.

    Worked out from the schema alone, so an empty frame gives the same answer as a full one.
    Refuses, as :func:`compile_for` does, an expression that does not compile or cannot apply to
    these columns.
    """
    return _produced(frame, compile_for(frame, text, field), text, field)


def compile_predicate_for(frame: pl.DataFrame, text: str, field: str) -> pl.Expr:
    """Compile ``text`` like :func:`compile_for` and refuse it unless it gives a true/false value.

    ``[amount]`` written for ``[amount] > 0`` is refused here, naming the field, the column and
    the type it gave. A column that already holds true/false is accepted as written.
    """
    expression = compile_for(frame, text, field)
    column, dtype = _produced(frame, expression, text, field)
    if dtype != pl.Boolean:
        raise ConfigError(
            f"config field {field!r} does not give a truth value: column {column!r} holds "
            f"{dtype}, not a boolean, in expression: `{text}`"
        )
    return expression


def _produced(frame: pl.DataFrame, expression: pl.Expr, text: str, field: str) -> tuple[str, pl.DataType]:
    """Return the name and dtype one compiled expression gives, as a :class:`ConfigError` on failure."""
    try:
        produced = frame.lazy().select(expression).collect_schema()
    except pl.exceptions.PolarsError as error:
        raise ConfigError(
            f"config field {field!r} cannot be applied to these columns: {type(error).__name__}: "
            f"{error}, in expression: `{text}`"
        ) from error
    name, dtype = next(iter(produced.items()))
    return name, dtype


def require_columns(frame: pl.DataFrame, names: Sequence[str], field: str, port: str) -> None:
    """Refuse when any of ``names`` is not a column of ``frame``.

    ``field`` is the setting that named them and ``port`` the input port ``frame`` came from. The
    :class:`ConfigError` names both and lists the columns that are there, so a misspelling or a
    wrong case is easy to see.
    """
    present = frame.columns
    absent = [name for name in names if name not in present]
    if absent:
        raise ConfigError(
            f"config field {field!r} names column(s) {', '.join(absent)} that are not on input port "
            f"{port!r}; the columns on {port!r} are: {', '.join(present)}"
        )


# The shared ``replace`` setting that :func:`check_output_column` reads. It has no default, so a
# missing value means "not asked".
REPLACE_FIELD: Final[ConfigField] = ConfigField(
    type="boolean",
    description="Whether writing this column may overwrite a column of that name that is already "
    "there; leave it out and a name already in use is refused.",
)


def check_output_column(frame: pl.DataFrame, name: str, field: str, *, replace: bool | None) -> None:
    """Check that a step may write a column called ``name`` into ``frame``.

    ``field`` is the setting that holds the name; ``replace`` is the step's ``replace`` setting.
    Raises :class:`ConfigError` when:

    - ``name`` is empty or only whitespace;
    - ``name`` is already a column and ``replace`` is not true;
    - ``replace`` is true and ``name`` is not a column (likely a misspelling).

    Why: docs/design/core.md#writing-a-column-name
    """
    if not name.strip():
        # Not quoted back: a blank name prints as nothing.
        raise ConfigError(
            f"config field {field!r} must not be blank: a column name has to carry at least one "
            f"character that is not whitespace, or nothing can point at the column it writes"
        )

    present = name in frame.columns
    if replace and not present:
        raise ConfigError(
            f"config field {field!r} asks to replace column {name!r}, which the input does not "
            f"have; replacing a name nothing carries would write a new column and leave the one "
            f"meant for repair untouched. The columns present are: {', '.join(frame.columns)}"
        )
    if present and not replace:
        raise ConfigError(
            f"config field {field!r} names output column {name!r}, which the input already carries; "
            f"say replace: true to overwrite it deliberately, or write to a name nothing uses"
        )


# The shared setting that names the time zone a timestamp is read in as a date. No default:
# which day a timestamp falls on depends on the zone, so the author states it.
ZONE_FIELD: Final[ConfigField] = ConfigField(
    type="string",
    description="The time zone in which a timestamp that carries a zone is read as a date, such as "
    "UTC or America/Toronto. Required when one does, refused when none does; no default.",
)


def check_zone_name(zone: str, field: str) -> None:
    """Refuse a time zone name the engine does not know, naming the field and the name."""
    try:
        pl.Series([None], dtype=pl.Datetime("us")).dt.replace_time_zone(zone)
    except pl.exceptions.PolarsError:
        raise ConfigError(
            f"config field {field!r} names the time zone {zone!r}, which is not one the engine knows; "
            f"write an IANA name such as UTC or America/Toronto"
        ) from None


def require_stated_zone(dtype: pl.DataType, zone: str | None, field: str, *, reading: str) -> None:
    """Refuse reading a zone-aware timestamp as a date when no zone is stated.

    ``dtype`` is the column's type, ``zone`` the stated zone (or ``None``) and ``field`` the
    setting to name. ``reading`` says what is being read, for the message: a column, or a side of
    a range. A zone stated for a column without one is not refused here.
    """
    carried = dtype.time_zone if isinstance(dtype, pl.Datetime) else None
    if carried is not None and zone is None:
        raise ConfigError(
            f"{reading} is a timestamp in time zone {carried!r}, read here as a date, and which day it "
            f"falls on depends on the zone; write {field}: UTC to read it as the UTC date, as before, "
            f"or the zone whose calendar the dates belong to"
        )


def as_calendar_date(expression: pl.Expr, dtype: pl.DataType, zone: str | None) -> pl.Expr:
    """Return ``expression`` as a date. A zone-aware timestamp is read in ``zone`` when one is given."""
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None and zone is not None:
        return expression.dt.convert_time_zone(zone).dt.date()
    return expression.cast(pl.Date)
