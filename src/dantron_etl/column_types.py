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
