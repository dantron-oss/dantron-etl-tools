"""Bringing exact columns of different places to one type, for the tools that put two of them together.

Two exact columns of different places hold the same kind of value, so a tool that stacks or links
them widens both to the larger places. Widening adds zeros and loses nothing, except for a value
with so many digits before the point that the larger places leave no room for them. That value is
refused by name, never rounded.
"""

import polars as pl

from ..authoring import DataError
from ..column_types import EXACT_DIGITS, exact_dtype, exact_from_exact


def larger_exact_type(first: pl.DataType, second: pl.DataType) -> pl.DataType | None:
    """The exact type two different exact types are both widened to; None unless both are exact and differ."""
    if first == second or not (isinstance(first, pl.Decimal) and isinstance(second, pl.Decimal)):
        return None
    return exact_dtype(max(first.scale or 0, second.scale or 0))


def widen_exact(
    frame: pl.DataFrame, column: str, places: int, *, where: str, wider: str, wider_where: str = ""
) -> pl.Series:
    """``frame[column]`` at ``places`` places and 38 digits, refusing a value the widening cannot hold.

    ``where`` says where the column is, such as "on input port 'top'". ``wider`` is the name of the
    column with more places, which the refusal says to round, and ``wider_where`` says where that one
    is when it is somewhere else. Raises ``DataError`` listing the rows.
    """
    conversion = exact_from_exact(frame[column], places)
    if conversion.misfits.any():
        held = frame.schema[column].scale or 0
        shown = ", ".join(str(value) for value in frame[column].filter(conversion.misfits).head(3).to_list())
        raise DataError(
            f"column {column!r} {where} holds values with more than {EXACT_DIGITS - places} digits before "
            f"the point, such as {shown}, so they cannot be widened to the {places} places of column "
            f"{wider!r}{wider_where}: an exact column at {places} places holds at most {EXACT_DIGITS - places} "
            f"digits before the point. Round {wider!r}{wider_where} to {held} places first, with "
            f"round([{wider}], {held}) in a formula step",
            offending=frame.filter(conversion.misfits),
            total=frame.height,
        )
    return conversion.values
