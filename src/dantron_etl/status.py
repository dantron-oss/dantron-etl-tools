"""Per-row status: what a step did to each row of a port it degraded.

A tool that empties or degrades some values can say which rows, and why, beside the port. This
module fixes the shape of that status frame and gives tool authors the helpers that build one. The
boundary checks every status a tool hands back against this shape.

A step may also send back a **per-row status** beside a port: one row per row of that port, in the
same order, saying which rows it degraded and why. That is the half a note cannot carry. A note says
595 values would not read as a number, which is a quantity and nothing else -- it cannot say whether
those were figures the run destroyed or text that was never a figure, and the output column cannot
say either, because after the step every one of them is null. The status can, so the sentence and
the rows behind it travel together. It rides in the envelope rather than in the frame for the same
reason a note does: what a step did is not a fact about the data, and a column would flow into the
next join and out into the published workbook. Almost every step reports none, and the boundary
refuses one that says only that every row is fine.
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


# What a status row may say about the row it describes. Published rather than left to each tool,
# because the reader of a status is a renderer: a level it has no drawing for shows as neither a
# tick nor a warning nor a cross, so the row shows nothing at all and reads as fine. Three words
# only -- "this row came through", "something about it was degraded", "something about it failed" --
# and the boundary refuses a fourth.
STATUS_LEVELS: Final[tuple[str, ...]] = ("ok", "warning", "error")

# One published shape for every status frame in the library, so nothing reading one has to work out
# per port what it is looking at.
#
# ``reason`` is a *list* because one row can be degraded twice over -- two columns cast in the same
# step, both unreadable on the same row -- and a single reason per row would drop the second in
# silence, sending a reader to the wrong column. It holds structs rather than sentences because the
# question asked of a status is "which rows does this warning cover", answered by matching
# ``reason.column`` exactly; against prose that match is a substring search, which is a different
# question wearing the same answer.
STATUS_REASON: Final[pl.DataType] = pl.List(pl.Struct({"column": pl.String, "detail": pl.String}))
STATUS_SCHEMA: Final[pl.Schema] = pl.Schema({"level": pl.String, "reason": STATUS_REASON})

# How much of an offending value a reason quotes. A reason is read in a table cell beside the row it
# marks, so it is a caption rather than a copy: this is about one line of one, which was enough to
# recognise every value the run that prompted this had emptied. The rest is not lost -- it is on the
# input port, one panel away -- and an uncapped quote would put a whole cell of free text into every
# reason of every marked row, which on a column that failed six hundred times is the frame again.
STATUS_VALUE_CHARS: Final[int] = 60


def value_caption(column: str, *, limit: int = STATUS_VALUE_CHARS) -> pl.Expr:
    """``column``'s value as short quoted text, for a reason that says what was there.

    The whole of what a marked row is asked: *what did this hold before the step emptied it?* An
    output column cannot answer -- every value that failed to read is null afterwards, which is the
    defect this channel closes -- and sending the reader to the input port to find out costs them
    the hunt, and for a step that reshapes rows (an unpivot melts one row into several) there is no
    row to find. So the value is quoted here, where the two sides can be put together honestly.

    Two properties, both load-bearing. It never answers null: a reason built with
    :func:`polars.format` around a null propagates the null through the whole sentence, and a marked
    row whose reason vanished is a row the status calls degraded and cannot say a thing about. And
    it is capped at ``limit`` characters, saying so when it cuts, so the length of a reason does not
    follow the length of whatever text was in the cell.

    Both of those are about a *value*. A column whose type has no text form at all -- a list, a
    struct -- refuses the cast outright rather than answering null, and this refuses with it, which
    is the honest answer: what the reader would be shown is not one value. Nothing reaches here that
    way in practice, because a column is captioned only where a cast to a scalar type has just read
    it, and that cast refuses first.
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
    """One reason per row for the values of ``column`` that would not read as ``declared``, and the note counting them.

    Written once for the two tools that read a column as a type the flow states. They already answer
    that question in the same words down to the wording of their notes, deliberately, and a second
    spelling of the same sentence is how a reader ends up trusting the rule they learned second.

    The reasons and the note come back from one call because they are two accounts of one fact, and
    the failure worth designing out is them drifting apart: an edit that changed how the rows are
    marked while leaving the count to be worked out a few lines away would leave a reader looking at
    595 marks under a sentence that says something else. Here the count, the share, the example and
    the reasons are all read off ``failed``, so there is nothing left for them to disagree about.

    ``failed`` is the step's own mask, taken before the values were replaced, so this never has to
    re-derive which rows lost something -- afterwards it could not: a null that failed to read and a
    null that was always there are the same null, which is the whole defect this channel closes. A
    blank therefore gets no reason here, because it never failed.

    The note quotes the first offending value, captioned exactly as the reasons caption it, for the
    one sentence a reader sees without opening anything: "595 values failed" sends them looking,
    "595 values like 'Canadian Business Growth Fund' failed" usually settles it where they stand.

    Called with something marked. A mask with nothing set describes no rows, so the note it would
    write counts nothing and quotes nothing, and the all-``ok`` status built beside it is refused at
    the boundary as inert -- which is why both callers ask the mask before they ask for this.
    """
    captioned = frame.select(value_caption(column).alias("caption")).with_columns(failed.alias("failed"))
    details = captioned.select(
        pl.when(pl.col("failed")).then(
            pl.format("held {}, which did not read as {}", pl.col("caption"), pl.lit(declared))
        )
    ).to_series()
    example = captioned.filter(pl.col("failed")).get_column("caption").head(1).to_list()

    lost, total = int(failed.sum()), failed.len()
    # Two decimal places rather than two significant figures. `.2g` is a general format, and the
    # share this sentence reaches most often is exactly the one it renders unreadably: a column that
    # is text from top to bottom fails on every row, and 100 comes out as `1e+02`. A reader who has
    # to decode the number in a warning is back to not trusting the warning.
    share = lost / total * 100 if total else 0.0
    note = (
        f"{lost:,} of {total:,} values in {column!r} ({share:.2f}%) did not "
        f"read as {declared} and are now empty (for example {example[0] if example else None})"
    )
    return details, note


def row_status(marks: Sequence[tuple[str, pl.Series]], height: int) -> pl.DataFrame:
    """A status frame of ``height`` rows, built from one detail column per thing that could degrade.

    A mark is a column name and a series of reasons the same height as the port, holding the
    sentence for a row that was degraded and null for a row that was not. One series rather than a
    mask beside a sentence, because the two would be free to disagree: a row the mask called
    degraded and the sentence had nothing to say about would be marked and unexplained, which is the
    old count again with extra steps.

    Marks are applied in the order they are given, which is the order the step did the work in, so a
    row degraded by two columns reads in the order they were processed rather than in whatever order
    a set iterated.

    With no marks at all this hands back a frame of ``ok``. That is not a status worth publishing --
    the boundary refuses an inert one -- and it is exactly what a caller stacking a status per block
    needs for the blocks that came through clean.

    Every mark is measured against ``height`` rather than taken on trust, and the check belongs here
    rather than at the boundary: the boundary compares a finished status against the port it
    describes, which catches a wrong total and nothing finer. A caller stacking one block per melted
    column hands over several blocks, and a block one row short beside a block one row long adds up
    to a total the boundary is happy with, having slid every mark between them onto the wrong row --
    the exact failure this channel exists to make impossible, arriving inside the tool that builds
    it.
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

    # Keyed by position rather than by column name: two marks may name the same column -- a column
    # read twice, a column melted alongside itself -- and a frame keyed by name would silently keep
    # one of them.
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
