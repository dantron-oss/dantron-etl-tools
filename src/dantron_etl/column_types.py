"""The type names a flow may write for a column: ``String``, ``Int64``, ``Float64``, ``Decimal``,
``Boolean``, ``Date``, ``Datetime`` and ``Time``.

Every tool that lets a flow state a column's type (reading a CSV, casting, unpivoting) uses these
names, spelled as polars spells them. ``Decimal`` is an exact decimal: the flow states its
``places`` beside it, and its total digits are always 38. Also here: the check for decimals that
would lose a fraction when read as a whole number.

Why: docs/design/core.md#one-set-of-type-names
"""

from typing import Final, NamedTuple

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


# ---------------------------------------------------------------------------
# Converting to and from an exact column
# Why: docs/design/core.md#exact-conversions
# ---------------------------------------------------------------------------


class ExactConversion(NamedTuple):
    """A column converted to an exact type, and the rows whose values did not fit.

    ``values`` is ``Decimal(38, places)``, with the source's name and length. It is null where the
    source was null and where the value did not fit. ``misfits`` is a Boolean series of the same
    length, true exactly where the source held a value that did not fit. It is never true on a null.
    """

    values: pl.Series
    misfits: pl.Series


#: Text an exact column reads: digits on both sides of an optional point, and an optional minus.
_PLAIN_DIGITS: Final[str] = r"^(?P<sign>-?)(?P<whole>[0-9]+)(?:\.(?P<fraction>[0-9]+))?$"

#: Text the engine writes for a float: the shortest that reads back the same, sometimes with an exponent.
_FLOAT_TEXT: Final[str] = (
    r"^(?P<sign>-?)(?P<whole>[0-9]+)(?:\.(?P<fraction>[0-9]*))?(?:[eE](?P<exponent>[+-]?[0-9]+))?$"
)

#: Significant digits a float is read to: the reading round() uses.
_READ_DIGITS: Final[dict[pl.DataType, int]] = {pl.Float64(): 15, pl.Float32(): 6}


def exact_from_text(column: pl.Series, places: int) -> ExactConversion:
    """Convert a text column to ``Decimal(38, places)``, exactly or not at all.

    A value converts when it is written as plain digits, with an optional point and an optional
    leading minus (``-12.50``), and its value fits: at most ``places`` digits after the point once
    trailing zeros are dropped, and at most 38 - ``places`` before it once leading zeros are. So
    ``1.2300`` fits 2 places and ``1.005`` does not. Exponents, spaces, commas, a plus sign, and a
    point with no digit on one side never convert. Raises ``TypeError`` for a column that is not text.
    """
    _require(column, column.dtype == pl.String, "text")
    return _conversion(column, column, places)


def exact_from_whole(column: pl.Series, places: int) -> ExactConversion:
    """Convert a whole-number column to ``Decimal(38, places)``.

    Every value fits unless it has more than 38 - ``places`` digits. Raises ``TypeError`` for a
    column that is not a whole number.
    """
    _require(column, column.dtype.is_integer(), "whole numbers")
    return _conversion(column, column.cast(pl.String), places)


def exact_from_exact(column: pl.Series, places: int) -> ExactConversion:
    """Convert an exact column to ``Decimal(38, places)``, more or fewer places than it holds.

    More places is always exact, unless a value has more than 38 - ``places`` digits before the
    point. Fewer places converts only the values that already fit: 1.2300 becomes 1.23, and 1.2350
    does not fit 2 places. Raises ``TypeError`` for a column that is not exact.
    """
    _require(column, isinstance(column.dtype, pl.Decimal), "exact decimals")
    return _conversion(column, column.cast(pl.String), places)


def exact_from_float(column: pl.Series, places: int) -> ExactConversion:
    """Convert a float column to ``Decimal(38, places)``, reading each value to 15 significant digits.

    That is the reading ``round`` uses (6 digits for a Float32). The shortest text that reads back as
    the same float is rounded half away from zero to 15 significant digits, and the result must then
    fit: 0.1 + 0.2 reads as 0.300000000000000 and fits 2 places as 0.30; 1.005 reads as 1.005 and
    does not fit 2 places. Not-a-number and infinity never fit. -0.0 becomes 0. Raises
    ``TypeError`` for a column that is not a float.
    """
    _require(column, column.dtype in _READ_DIGITS, "floats")
    reading = column.to_frame("value").select(_float_reading(_READ_DIGITS[column.dtype])).to_series()
    return _conversion(column, reading, places)


def to_exact(column: pl.Series, places: int) -> ExactConversion:
    """Convert any column that can hold a number to ``Decimal(38, places)``, choosing by its type.

    Text, whole numbers, exact decimals and floats go to the helpers above. A column of only nulls
    with no type gives nulls. Raises ``TypeError`` for any other type, such as a date.
    """
    dtype = column.dtype
    if dtype == pl.String:
        return exact_from_text(column, places)
    if dtype.is_integer():
        return exact_from_whole(column, places)
    if isinstance(dtype, pl.Decimal):
        return exact_from_exact(column, places)
    if dtype in _READ_DIGITS:
        return exact_from_float(column, places)
    if dtype == pl.Null:
        return _conversion(column, column.cast(pl.String), places)
    raise TypeError(f"column {column.name!r} holds {dtype}, which does not convert to an exact decimal")


def exact_to_float(column: pl.Series) -> pl.Series:
    """Convert an exact column to Float64, landing each value on the nearest double.

    It goes through the digits as text. The engine's own conversion can miss by one step once the
    digits held, counting every place, pass 2**53, or the column holds more than 22 places.
    Raises ``TypeError`` for a column that is not exact.
    """
    _require(column, isinstance(column.dtype, pl.Decimal), "exact decimals")
    return column.cast(pl.String).cast(pl.Float64)


def exact_misfit_refusal(column: str, values: pl.Series, places: int, *, melted: bool = False) -> str:
    """Return the refusal for values that do not fit ``Decimal`` with ``places``, naming up to three."""
    shown = ", ".join(
        repr(value) if isinstance(value, str) else str(value)
        for value in values.unique(maintain_order=True).head(3).to_list()
    )
    kind = "melted column" if melted else "column"
    if values.dtype == pl.String:
        remedy = (
            "Text converts only when written as plain digits, with an optional point and minus sign, "
            "such as -12.50; correct the source"
        )
    else:
        remedy = f"Round it first, with round([{column}], {places}) in a formula step"
    return (
        f"{kind} {column!r} holds values that do not fit Decimal with {places} places without rounding, "
        f"such as {shown}; a value fits when it has at most {places} places and at most "
        f"{EXACT_DIGITS - places} digits before the point. {remedy}, or say on_invalid: set_null to "
        f"empty those values instead"
    )


def _require(column: pl.Series, holds: bool, kind: str) -> None:
    if not holds:
        raise TypeError(f"column {column.name!r} holds {column.dtype}, not {kind}")


def _conversion(source: pl.Series, text: pl.Series, places: int) -> ExactConversion:
    """Convert ``text`` exactly, and mark each row of ``source`` that held a value and gave none."""
    values = text.to_frame("text").select(_exact_from_plain(pl.col("text"), places)).to_series().alias(source.name)
    misfits = (source.is_not_null() & values.is_null()).alias(source.name)
    return ExactConversion(values, misfits)


def _exact_from_plain(text: pl.Expr, places: int) -> pl.Expr:
    """The exact value of text written as plain digits, or null where it is not or does not fit."""
    dtype = exact_dtype(places)
    parts = text.str.extract_groups(_PLAIN_DIGITS)
    sign = parts.struct.field("sign")
    whole = parts.struct.field("whole").str.strip_chars_start("0")
    fraction = parts.struct.field("fraction").fill_null("").str.strip_chars_end("0")
    # The engine would also empty a value too wide for the type; the library counts the digits itself.
    fits = (whole.str.len_chars() <= EXACT_DIGITS - places) & (fraction.str.len_chars() <= places)
    canonical = pl.concat_str(
        sign,
        pl.when(whole == "").then(pl.lit("0")).otherwise(whole),
        pl.when(fraction == "").then(pl.lit("")).otherwise(pl.lit(".") + fraction),
    )
    return pl.when(fits).then(canonical.cast(dtype, strict=False)).otherwise(pl.lit(None, dtype=dtype))


def _float_reading(significant: int) -> pl.Expr:
    """A float column as plain digits, rounded half away from zero to ``significant`` digits.

    Null where the float is null, not a number or infinite. The engine writes the shortest text that
    reads back as the same float, such as 0.30000000000000004 or 1e-7; its digits are cut to
    ``significant``, the last kept digit goes up when the first one cut is 5 or more, and the point
    is put back where it belongs.
    """
    parts = pl.col("value").cast(pl.String).str.extract_groups(_FLOAT_TEXT)
    sign = parts.struct.field("sign")
    fraction = parts.struct.field("fraction").fill_null("")
    exponent = parts.struct.field("exponent").fill_null("0").cast(pl.Int64, strict=False)
    # The value is sign × digits × 10^point, with every digit written, leading zeros dropped.
    digits = pl.concat_str(parts.struct.field("whole"), fraction).str.strip_chars_start("0")
    count = digits.str.len_chars().cast(pl.Int64)
    point = exponent - fraction.str.len_chars().cast(pl.Int64)
    cut = pl.max_horizontal(count - significant, pl.lit(0, dtype=pl.Int64))
    kept = digits.str.slice(0, count - cut)
    kept = pl.when(kept == "").then(pl.lit("0")).otherwise(kept).cast(pl.Int64)
    up = (digits.str.slice(count - cut, 1) >= "5").fill_null(False).cast(pl.Int64)
    rounded = (kept + up).cast(pl.String)
    point = point + cut
    # A point at or after the digits: zeros after them. More than 39 zeros cannot fit any exact
    # column, so writing only 39 leaves the answer the same.
    zeros = pl.lit("0" * (EXACT_DIGITS + 1)).str.slice(
        0, pl.max_horizontal(pl.min_horizontal(point, EXACT_DIGITS + 1), 0)
    )
    after = pl.concat_str(rounded, zeros)
    # A point inside or before the digits: pad with leading zeros, then put the point in.
    shift = -point
    padded = rounded.str.zfill(pl.max_horizontal(shift + 1, 0))
    split = padded.str.len_chars().cast(pl.Int64) - shift
    inside = pl.concat_str(padded.str.slice(0, split), pl.lit("."), padded.str.slice(split))
    plain = pl.when(point >= 0).then(after).otherwise(inside)
    return pl.concat_str(sign, plain)
