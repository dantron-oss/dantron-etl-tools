"""Per-row status: which rows of a port a step changed, and why.

A tool that empties or changes some values can return a status frame beside the port: one row per
data row, in the same order, each with a level and a list of reasons. This module fixes the shape of
that frame and holds the helpers that build one. The boundary checks every status a tool returns.

Why: docs/design/core.md#notes-and-per-row-status
"""

from collections.abc import Sequence
from typing import Final

import polars as pl

from .errors import PortError

__all__ = [
    "STATUS_LEVELS",
    "STATUS_REASON",
    "STATUS_SCHEMA",
    "STATUS_VALUE_CHARS",
    "row_status",
    "unreadable_report",
    "value_caption",
]


# The levels a status row may have. The boundary refuses any other word.
STATUS_LEVELS: Final[tuple[str, ...]] = ("ok", "warning", "error")

# The one shape of every status frame. ``reason`` is a list of ``{column, detail}`` structs.
# Why: docs/design/core.md#the-shape-of-a-status-frame
STATUS_REASON: Final[pl.DataType] = pl.List(pl.Struct({"column": pl.String, "detail": pl.String}))
STATUS_SCHEMA: Final[pl.Schema] = pl.Schema({"level": pl.String, "reason": STATUS_REASON})

# The most characters of a value a reason quotes.
STATUS_VALUE_CHARS: Final[int] = 60


def value_caption(column: str, *, limit: int = STATUS_VALUE_CHARS) -> pl.Expr:
    """Return an expression that shows ``column``'s value as short quoted text, for a reason.

    It never returns null: a null value reads "a value that cannot be shown as text". A value
    longer than ``limit`` characters is cut and says so. A list or struct column cannot be cast to
    text and is refused by polars.
    """
    text = pl.col(column).cast(pl.String, strict=False)
    shortened = (
        pl.when(text.str.len_chars() > limit)
        .then(pl.format("{}... (shortened from {} characters)", text.str.slice(0, limit), text.str.len_chars()))
        .otherwise(text)
    )
    return (
        pl.when(text.is_null())
        .then(pl.lit("a value that cannot be shown as text"))
        .otherwise(pl.format("'{}'", shortened))
    )


def unreadable_report(
    frame: pl.DataFrame,
    column: str,
    failed: pl.Series,
    declared: str,
) -> tuple[pl.Series, str]:
    """Describe the values of ``column`` that did not read as the type ``declared``.

    ``failed`` is the step's boolean mask of rows that failed, taken before the values were
    emptied. Returns ``(details, note)``: a series with a reason for each failed row (null
    elsewhere), and a note counting them and quoting the first value. Call it only when ``failed``
    has at least one true value; a status with nothing marked is refused at the boundary.

    Why: docs/design/core.md#one-helper-for-unreadable-values
    """
    captioned = frame.select(value_caption(column).alias("caption")).with_columns(failed.alias("failed"))
    details = captioned.select(
        pl.when(pl.col("failed")).then(
            pl.format("held {}, which did not read as {}", pl.col("caption"), pl.lit(declared))
        )
    ).to_series()
    example = captioned.filter(pl.col("failed")).get_column("caption").head(1).to_list()

    lost, total = int(failed.sum()), failed.len()
    # Two decimal places, so 100% does not print as 1e+02.
    share = lost / total * 100 if total else 0.0
    note = (
        f"{lost:,} of {total:,} values in {column!r} ({share:.2f}%) did not "
        f"read as {declared} and are now empty (for example {example[0] if example else None})"
    )
    return details, note


def row_status(marks: Sequence[tuple[str, pl.Series]], height: int) -> pl.DataFrame:
    """Build a status frame of ``height`` rows from a list of marks.

    A mark is ``(column, details)``: a series of ``height`` reasons, a sentence on a changed row and
    null elsewhere. A row with any reason is ``warning``; the rest are ``ok``. Reasons on one row
    keep the order of the marks. With no marks, every row is ``ok``.

    Refuses (:class:`PortError`) a mark whose length is not ``height``.

    Why: docs/design/core.md#row-status-checks-each-mark
    """
    misaligned = [column for column, detail in marks if detail.len() != height]
    if misaligned:
        raise PortError(
            f"a per-row status of {height:,} row(s) cannot be built from mark(s) on "
            f"{', '.join(misaligned)} of another length; a mark carries one reason per row of the "
            f"port it describes, and one of another height marks the wrong rows rather than fewer"
        )

    if not marks:
        return pl.DataFrame({"level": ["ok"] * height, "reason": [[]] * height}, schema=STATUS_SCHEMA)

    # Keyed by position, because two marks may name the same column.
    details = pl.DataFrame({str(index): detail for index, (_, detail) in enumerate(marks)})
    reason = pl.concat_list(
        *(
            pl.when(pl.col(str(index)).is_not_null())
            .then(pl.concat_list(pl.struct(column=pl.lit(column, dtype=pl.String), detail=pl.col(str(index)))))
            .otherwise(pl.lit([], dtype=STATUS_REASON))
            for index, (column, _) in enumerate(marks)
        )
    )
    return details.select(reason=reason).select(
        level=pl.when(pl.col("reason").list.len() > 0).then(pl.lit("warning")).otherwise(pl.lit("ok")),
        reason=pl.col("reason"),
    )
