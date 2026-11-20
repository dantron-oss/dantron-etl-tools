"""The checks a frame passes before it is loaded into a table, shared by write_table and publish_tables.

Columns that do not match the table's, an empty frame not allowed, values that will not fit the
declared column type, and, for a merge, a key that does not pick out one row.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..authoring import DataError
from ..db import probe_columns, require_matching_columns


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


def require_landable_columns(frame: pl.DataFrame, declared: Mapping[str, str | None], qualified: str) -> None:
    """Refuse a column whose values will not go into the column the table declares, naming it.

    SQLite stores 'heavy' in an integer column and reports success; Postgres fails later in its own words.
    A type this library does not know (a domain, an enum, JSON) is passed over, not refused.
    """
    for column, dtype in frame.schema.items():
        wanted = declared.get(column)
        if wanted is None or _fits(dtype, wanted):
            continue
        raise DataError(
            f"column {column!r} carries {dtype} where table {qualified!r} declares its column as "
            f"{wanted}; nothing is written",
            offending=frame,
            total=frame.height,
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
        return dtype.is_numeric()
    base = _DECLARED_BASE.get(declared)
    # A declared type this module has no entry for is passed over, exactly as an unrecognised SQL
    # type is. The two vocabularies live in different modules and the reading side is a safety net:
    # one growing a name the other has not learned should cost the check, never the load.
    return base is None or dtype.base_type() == base
