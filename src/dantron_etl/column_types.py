"""The vocabulary for naming a column's type, shared by every tool that lets a flow state one.

Two tools need these words and a third will. A CSV states no types at all, so its reader infers them
and has to be correctable; a cast step states them outright, because two frames about to be stacked
have to agree and something has to say which of them is right. Each offers the same names, in the
same spelling, and tools meant to answer a shared question *identically* rather than similarly do
not stay that way when the answer is written down in two files.

Spelled the way the query engine spells them, so a reader who knows one knows the other, and every
one of them survives a definition file unchanged -- none is a word YAML would quietly turn into a
boolean, a null or a number.
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

#: Only these three carry an order that text cannot settle on its own, so only these take a format.
#: A source already holding real temporal values needs none, which is why not every tool offers one.
TEMPORAL_TYPES: Final[frozenset[str]] = frozenset({"Date", "Datetime", "Time"})

DATE_DEFAULT_FORMAT: Final[str] = "%Y-%m-%d"
DATETIME_DEFAULT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%f"
TIME_DEFAULT_FORMAT: Final[str] = "%H:%M:%S.%f"


def cut_fractions(column: pl.Series, target: pl.DataType) -> pl.Series:
    """True on each row where reading ``column`` as ``target`` would cut a fraction off.

    The engine reads 1.99 as the whole number 1 and -2.7 as -2, without a word. So a decimal with a
    fraction is a value that does not read as a whole number, and both tools that read a column as
    a stated type ask this first. Only a decimal read as a whole number can lose a fraction; for any
    other pair every row is false. Null, not-a-number and infinity are not counted here: the engine
    refuses the last two on its own.
    """
    decimal = column.dtype.is_float() or isinstance(column.dtype, pl.Decimal)
    if not (target.is_integer() and decimal):
        return pl.Series(column.name, [False] * column.len(), dtype=pl.Boolean)
    value = column.cast(pl.Float64) if column.dtype.is_float() else column
    finite = value.is_finite() if column.dtype.is_float() else value.is_not_null()
    return (finite & (value != value.round(0))).fill_null(False)


def fraction_refusal(column: str, values: pl.Series, declared: str, *, melted: bool = False) -> str:
    """The sentence that refuses a decimal with a fraction, naming the column and a few of the values."""
    shown = ", ".join(str(value) for value in values.unique(maintain_order=True).head(3).to_list())
    kind = "melted column" if melted else "column"
    return (
        f"{kind} {column!r} holds decimals with a fraction, such as {shown}, so it does not read as "
        f"{declared} without cutting the fraction off; round it first, with round([{column}], 0) in "
        f"a formula step, or say on_invalid: set_null to empty those values instead"
    )
