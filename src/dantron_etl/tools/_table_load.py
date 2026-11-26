"""The checks a frame passes before it is loaded into a table, shared by write_table and publish_tables.

Columns that do not match the table's, an empty frame not allowed, values that will not fit the
declared column type or a numeric column's places and digits, and, for a merge, a key that does not
pick out one row.

Why: docs/design/database.md#numbers-into-a-numeric-column
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..authoring import DataError
from ..column_types import EXACT_DIGITS, to_exact
from ..db import probe_columns, require_matching_columns

#: Where a refusal says how a float was read, by its type.
_FLOAT_READING: Final[Mapping[Any, str]] = {
    pl.Float64: "a Float64 is read to 15 significant digits, as round reads it",
    pl.Float32: "a Float32 is read to 6 significant digits, as round reads it",
}


def require_matching_table(connection: Any, frame: pl.DataFrame, qualified: str, stamp: str | None = None) -> tuple[str, ...]:
    """Prove the table is there and that the frame's columns are exactly its columns. Returns them.

    Probing the columns is the same statement that proves the table exists. ``stamp`` is a column
    the writer adds itself, so it counts on the frame's side.
    """
    table_columns = probe_columns(connection, qualified)
    require_matching_columns(frame.columns, table_columns, qualified, stamp=stamp)
    return table_columns


def require_rows(tool: str, frame: pl.DataFrame, qualified: str, allow_empty: bool, *, consequence: str = "") -> None:
    """Refuse an empty frame unless the step allows one: a replace of nothing empties the table."""
    if frame.height == 0 and not allow_empty:
        raise DataError(
            f"{tool} refuses an empty frame for table {qualified!r}{consequence}; set allow_empty "
            f"to accept it",
            offending=frame,
            total=0,
        )


def require_landable_columns(
    frame: pl.DataFrame,
    declared: Mapping[str, str | None],
    qualified: str,
    numerics: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> None:
    """Refuse a column whose values will not go into the column the table declares, naming it.

    SQLite stores 'heavy' in an integer column and reports success; Postgres fails later in its own words.
    A type this library does not know (a domain, an enum, JSON) is passed over, not refused. An exact
    column never goes into a float column. ``numerics`` gives each numeric column's declared digits and
    places: a number goes into ``numeric(p,s)`` only when every value fits, since the database would
    round the rest without a word.
    """
    for column, dtype in frame.schema.items():
        wanted = declared.get(column)
        if wanted is None or _fits(dtype, wanted):
            continue
        remedy = ""
        if isinstance(dtype, pl.Decimal) and wanted == "Float64":
            remedy = (
                ". An exact column is never written to a float column, which would keep only the nearest "
                "binary value; convert it with cast_columns to Float64 first, where the change shows, or "
                "declare the table's column numeric"
            )
        raise DataError(
            f"column {column!r} carries {dtype} where table {qualified!r} declares its column as "
            f"{wanted}{remedy}; nothing is written",
            offending=frame,
            total=frame.height,
        )
    for column, (digits, places) in (numerics or {}).items():
        if column not in frame.columns or places is None or digits is None or not _is_number(frame.schema[column]):
            continue
        misfits = numeric_misfits(frame[column], digits, places)
        if misfits.any():
            raise DataError(
                _numeric_refusal(frame[column].filter(misfits), digits, places, qualified),
                offending=frame.filter(misfits),
                total=frame.height,
            )


def numeric_misfits(column: pl.Series, digits: int, places: int) -> pl.Series:
    """True on each row whose value a ``numeric(digits, places)`` column would not hold as it is.

    An exact or whole number fits when it has at most ``places`` places and at most ``digits - places``
    digits before the point; a negative ``places`` also asks for that many zeros at the end. A float
    is read to 15 significant digits (6 for a Float32) and must then fit; not-a-number and infinity
    never do. A null always fits.
    """
    conversion = to_exact(column, _checked_places(column, digits, places))
    whole = (
        conversion.values.cast(pl.String)
        .str.strip_chars_start("-")
        .str.split(".")
        .list.first()
        .str.strip_chars_start("0")
    )
    misfits = conversion.misfits | (whole.str.len_chars() > digits - places).fill_null(False)
    if places < 0:
        misfits = misfits | ~((whole == "") | whole.str.ends_with("0" * -places)).fill_null(True)
    return misfits.alias(column.name)


def landing_frame(frame: pl.DataFrame, numerics: Mapping[str, tuple[int | None, int | None]]) -> pl.DataFrame:
    """``frame`` as it is written: each float column bound for ``numeric(p,s)`` as its exact reading.

    The checks have already found that every value fits, so the database stores what was checked,
    the float read to 15 significant digits, and rounds nothing. Every other column is unchanged.
    """
    readings = [
        to_exact(frame[column], _checked_places(frame[column], digits, places)).values
        for column, (digits, places) in numerics.items()
        if column in frame.columns and digits is not None and places is not None and frame.schema[column].is_float()
    ]
    return frame.with_columns(readings) if readings else frame


def _checked_places(column: pl.Series, digits: int, places: int) -> int:
    """The places a value is read at to check it against ``numeric(digits, places)``.

    Normally the column's own places. A negative ``places`` means whole numbers. More places than an
    exact column can hold (38) leave no room before the point, so an exact or whole value is read at
    the places it already has, which always fit; a float is read at as many places as leave room for
    the column's digits before the point.
    """
    if places <= EXACT_DIGITS:
        return max(places, 0)
    if isinstance(column.dtype, pl.Decimal):
        return column.dtype.scale or 0
    if not column.dtype.is_float():
        return 0
    return max(0, EXACT_DIGITS - max(digits - places, 0))


def _is_number(dtype: pl.DataType) -> bool:
    return dtype.is_integer() or dtype.is_float() or isinstance(dtype, pl.Decimal)


def _numeric_refusal(values: pl.Series, digits: int, places: int, qualified: str) -> str:
    column = values.name
    shown = ", ".join(str(value) for value in values.unique(maintain_order=True).head(3).to_list())
    reading = _FLOAT_READING.get(values.dtype)
    read = f"; {reading}" if reading else ""
    return (
        f"column {column!r} holds values that table {qualified!r} would round or cannot hold in its column "
        f"declared numeric({digits},{places}), such as {shown}: that column keeps {places} places and at most "
        f"{digits - places} digits before the point, and the database rounds the rest without a word{read}. "
        f"Round it first, with round([{column}], {max(places, 0)}) in a formula step, or widen the table's "
        f"column; nothing is written"
    )


def require_one_row_per_key(frame: pl.DataFrame, key: Sequence[str], qualified: str) -> None:
    """Refuse a merge whose frame has a null in a key column, or the same key twice.

    A merge matches each incoming row to one published row. Two rows with one key would be blended
    into one stored row on one engine and refused on the other. A null key never matches anything,
    so the row would be added again on every run.
    """
    carried = [column for column in key if column in frame.columns]
    if not carried or frame.height == 0:
        return
    with_null = frame.filter(pl.any_horizontal(pl.col(carried).is_null()))
    if with_null.height:
        raise DataError(
            f"the merge into table {qualified!r} matches rows on {', '.join(carried)}, and a row "
            f"with a null there matches nothing, so it would be added again on every run. Remove "
            f"those rows or fill the key upstream; nothing is written",
            offending=with_null,
            total=frame.height,
        )
    repeated = frame.filter(pl.struct(carried).is_duplicated())
    if repeated.height:
        first = repeated.select(carried).row(0)
        shown = first[0] if len(first) == 1 else first
        raise DataError(
            f"the merge into table {qualified!r} matches rows on {', '.join(carried)}, and the key "
            f"{shown!r} arrives more than once, so the stored row would be a blend of them. Keep one "
            f"row per key upstream (remove_duplicates or group_by); nothing is written",
            offending=repeated,
            total=frame.height,
        )


#: The base dtype each remaining declared type is, for the types that admit no widening at all.
_DECLARED_BASE: Final[Mapping[str, Any]] = {
    "String": pl.String,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Time": pl.Time,
}


def _fits(dtype: pl.DataType, declared: str) -> bool:
    """Whether a frame column of ``dtype`` can land in a column declared ``declared``.

    A column of nothing but nulls fits anywhere: it carries no value to be wrong about. A whole
    number lands in a decimal column, because widening loses nothing; a decimal does not land in a
    whole-number column, because narrowing does, and quietly. Everything else has to be the type it
    says it is.
    """
    if dtype == pl.Null:
        return True
    if declared == "Int64":
        return dtype.is_integer()
    if declared == "Float64":
        return dtype.is_numeric() and not isinstance(dtype, pl.Decimal)
    if declared == "Decimal":
        return dtype.is_numeric()
    base = _DECLARED_BASE.get(declared)
    # A declared type this module has no entry for is passed over, exactly as an unrecognised SQL
    # type is. The two vocabularies live in different modules and the reading side is a safety net:
    # one growing a name the other has not learned should cost the check, never the load.
    return base is None or dtype.base_type() == base
