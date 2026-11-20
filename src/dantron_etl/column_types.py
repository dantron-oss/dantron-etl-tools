"""The type names a flow may write for a column: ``String``, ``Int64``, ``Float64``, ``Boolean``,
``Date``, ``Datetime`` and ``Time``.

Every tool that lets a flow state a column's type (reading a CSV, casting, unpivoting) uses these
names, spelled as polars spells them. Also here: the check for decimals that would lose a fraction
when read as a whole number.

Why: docs/design/core.md#one-set-of-type-names
"""

from typing import Final

import polars as pl

#: The types a column may be declared as, in the spelling the query engine uses.
DECLARED_TYPES: Final[tuple[str, ...]] = ("String", "Int64", "Float64", "Boolean", "Date", "Datetime", "Time")

#: Each declared name, as the dtype it means.
DECLARED_DTYPE: Final[dict[str, pl.DataType]] = {
    "String": pl.String,
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime("us"),
    "Time": pl.Time,
}

#: The types that take a format when read from text.
TEMPORAL_TYPES: Final[frozenset[str]] = frozenset({"Date", "Datetime", "Time"})

DATE_DEFAULT_FORMAT: Final[str] = "%Y-%m-%d"
DATETIME_DEFAULT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%f"
TIME_DEFAULT_FORMAT: Final[str] = "%H:%M:%S.%f"


def cut_fractions(column: pl.Series, target: pl.DataType) -> pl.Series:
    """Return a boolean series, true on each row where reading ``column`` as ``target`` would cut
    a fraction off.

    polars reads 1.99 as 1 and -2.7 as -2 without a warning, so tools ask this before a cast. Only
    a decimal read as a whole number can be true. Null, not-a-number and infinity are false here;
    polars refuses the last two itself.
    """
    decimal = column.dtype.is_float() or isinstance(column.dtype, pl.Decimal)
    if not (target.is_integer() and decimal):
        return pl.Series(column.name, [False] * column.len(), dtype=pl.Boolean)
    value = column.cast(pl.Float64) if column.dtype.is_float() else column
    finite = value.is_finite() if column.dtype.is_float() else value.is_not_null()
    return (finite & (value != value.round(0))).fill_null(False)


def fraction_refusal(column: str, values: pl.Series, declared: str, *, melted: bool = False) -> str:
    """Return the refusal for a decimal with a fraction, naming the column and up to three values."""
    shown = ", ".join(str(value) for value in values.unique(maintain_order=True).head(3).to_list())
    kind = "melted column" if melted else "column"
    return (
        f"{kind} {column!r} holds decimals with a fraction, such as {shown}, so it does not read as "
        f"{declared} without cutting the fraction off; round it first, with round([{column}], 0) in "
        f"a formula step, or say on_invalid: set_null to empty those values instead"
    )
