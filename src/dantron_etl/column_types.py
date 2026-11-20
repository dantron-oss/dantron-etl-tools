"""The type names a flow may write for a column: ``String``, ``Int64``, ``Float64``, ``Decimal``,
``Boolean``, ``Date``, ``Datetime`` and ``Time``.

Every tool that lets a flow state a column's type (reading a CSV, casting, unpivoting) uses these
names, spelled as polars spells them. ``Decimal`` is an exact decimal: the flow states its
``places`` beside it, and its total digits are always 38. Also here: the check for decimals that
would lose a fraction when read as a whole number.

Why: docs/design/core.md#one-set-of-type-names
"""

from typing import Final

import polars as pl

from .errors import ConfigError

#: The name of the exact decimal type. A flow states its places beside it.
EXACT_TYPE: Final[str] = "Decimal"

#: The total digits of every exact column, the most the engine holds. Only the places vary.
EXACT_DIGITS: Final[int] = 38

#: The types a column may be declared as, in the spelling the query engine uses.
DECLARED_TYPES: Final[tuple[str, ...]] = (
    "String",
    "Int64",
    "Float64",
    EXACT_TYPE,
    "Boolean",
    "Date",
    "Datetime",
    "Time",
)

#: Each declared name with a fixed dtype, as the dtype it means. ``Decimal`` is not here, because its
#: dtype depends on its places: use ``declared_dtype`` or ``exact_dtype``.
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


def exact_dtype(places: int) -> pl.Decimal:
    """Return the dtype of an exact column with ``places`` digits after the point: 38 digits in all.

    Raises ``ValueError`` for places that are not a whole number from 0 to 38. A flow's own setting
    is checked first, by ``check_places``, so this is a guard against the library's own mistakes.
    """
    if not _is_whole(places) or not 0 <= places <= EXACT_DIGITS:
        raise ValueError(f"an exact column's places must be a whole number from 0 to {EXACT_DIGITS}, not {places!r}")
    return pl.Decimal(EXACT_DIGITS, places)


def declared_dtype(declared: str, places: int | None = None) -> pl.DataType:
    """Return the dtype a declared type name means. ``Decimal`` needs its places; no other type takes them.

    Raises ``ValueError`` when the places do not suit the type, and ``KeyError`` for a name that is not
    declared.
    """
    if declared == EXACT_TYPE:
        if places is None:
            raise ValueError("an exact column needs its places")
        return exact_dtype(places)
    if places is not None:
        raise ValueError(f"only Decimal takes places, not {declared}")
    return DECLARED_DTYPE[declared]


def check_places(declared: str, places: object, field: str) -> int | None:
    """Refuse a ``places`` setting that does not suit the declared type; return it for ``Decimal``.

    ``places`` is required with ``Decimal`` and must be a whole number from 0 to 38. Any other type
    takes no places, so this returns None for it. ``field`` is where the places are written, such as
    ``columns[0].places``, and every refusal names it.
    """
    if declared != EXACT_TYPE:
        if places is not None:
            raise ConfigError(
                f"config field {field!r} is {places!r} beside type {declared}; only Decimal takes places, so remove it"
            )
        return None
    if places is None:
        raise ConfigError(
            f"config field {field!r} is missing; type Decimal needs it: the number of digits after the "
            f"point, from 0 to {EXACT_DIGITS}, such as places: 2"
        )
    if not _is_whole(places):
        raise ConfigError(
            f"config field {field!r} is {places!r}; places is a whole number from 0 to {EXACT_DIGITS}, "
            f"written without a point or quotes, such as places: 2"
        )
    if not 0 <= places <= EXACT_DIGITS:
        raise ConfigError(
            f"config field {field!r} is {places}; places runs from 0 to {EXACT_DIGITS}, because an exact "
            f"column holds {EXACT_DIGITS} digits in all"
        )
    return places


def not_yet_exact(tool: str, field: str, instead: str) -> str:
    """Return the refusal for a tool that does not take ``Decimal`` yet, naming the field and what to do."""
    return f"config field {field!r} names type Decimal, and {tool} does not take exact decimals yet; {instead}"


def _is_whole(value: object) -> bool:
    """Whether ``value`` is a whole number. ``True`` is not, though Python counts it as 1."""
    return isinstance(value, int) and not isinstance(value, bool)


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
